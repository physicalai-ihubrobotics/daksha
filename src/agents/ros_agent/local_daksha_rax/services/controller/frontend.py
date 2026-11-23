"""The session's robot service: the agent's only path to the (simulated) robot.

Adapted from RAX real/rainbow services/controller/frontend.py (Apache-2.0), with the episode
lifecycle of the RAX simulation branches (start_episode, evaluate, finish).

One process per session owns the backend. It speaks newline-delimited JSON-RPC (MCP tools)
on a private Unix socket, multiplexing Claude Code's stdio bridge (harness/mcp_bridge.py)
and the agent's own Python scripts (auto_research_agent/api/runtime.py); every call runs on
one dispatcher thread, so two clients never drive the robot at once.

Modes:
- exploration: lifecycle tools + robot tools; episodes play seeds 1000+i from a fixed budget.
- formal: robot tools + status only, for a frozen controller; the evaluator resets episodes.
"""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import sys
import threading
import time
import uuid

from services.backends import make_backend
from services.json_io import jsonable
from .config import Configuration
from .errors import public_failure
from .session import Session

PROTOCOL_VERSION = '2025-06-18'
MAX_REQUEST_BYTES = 32*1024*1024


def tool(name, description, properties=None, required=()):
    return {'name': name, 'description': description, 'inputSchema': {'type': 'object',
            'properties': properties or {}, 'required': list(required), 'additionalProperties': False}}


STATUS = tool('status', 'Session status: task, exploration episodes used and left, the current episode, and the '
                        'live robot summary.')
START = tool('start_episode', 'Close the current episode (if any) and start the next one: the robot returns to home, '
                              'grippers open, and the task objects and target bowls are placed in a new layout. Consumes one '
                              'exploration episode. Takes no seed or layout.')
EVALUATE = tool('evaluate', 'The trusted binary result of the current episode: task_complete true when every '
                            'TASK.md condition holds now. Does not end the episode.')
FINISH = tool('finish', 'End the current episode (records its final evaluate result). The robot stays where it is '
                        'until the next start_episode.')


class Frontend:
    def __init__(self, session, backend, *, mode='exploration', started_at=None):
        self.s, self.backend, self.mode = session, backend, mode
        self.started_at = started_at
        self._dispatch = threading.Lock()
        self._backend_tools = {t['name']: t for t in backend.tools()}
        reserved = {'status', 'start_episode', 'evaluate', 'finish'} & set(self._backend_tools)
        if reserved:
            raise ValueError(f'Backend tools may not reuse harness tool names: {sorted(reserved)}')

    def tools(self):
        lifecycle = [START, EVALUATE, FINISH] if self.mode == 'exploration' else []
        return [STATUS, *lifecycle, *self._backend_tools.values()]

    def status(self):
        config = self.s.config
        value = {'task': config.task, 'robot': config.robot, 'observation_profile': config.observation_profile,
                 'mode': self.mode}
        if self.mode == 'exploration':
            current = self.s.state['current']
            value['episodes'] = {'budget': config.episodes, 'started': self.s.state['episodes_started'],
                                 'left': self.s.episodes_left(),
                                 'current': None if current is None else {
                                     'number': current['index'] + 1,
                                     'evaluations': len(current['evaluations'])},
                                 'finished_results': [h.get('final') for h in self.s.state['history']]}
        if self.started_at is not None:
            value['session_seconds'] = config.session_seconds
            value['seconds_since_launch'] = round(time.time() - self.started_at, 1)
        try:
            value['robot_status'] = self.backend.status()
        except Exception as exc:
            value['robot_status_error'] = public_failure(exc, operation='status')
        return value

    def _lifecycle(self, name, arguments):
        if arguments:
            raise ValueError(f'{name} takes no arguments')
        if name == 'start_episode':
            previous = None
            if self.s.state['current'] is not None:
                previous = self.s.end_episode(self.backend.task_state()['task_complete'])
                self.backend.end_episode()
            current = self.s.begin_episode()
            self.backend.reset_episode(current['seed'])
            return {'episode': current['index'] + 1, 'episodes_left': self.s.episodes_left(),
                    'previous_episode_complete': None if previous is None else previous['final'],
                    'note': 'New layout. Observe before moving.'}
        if self.s.state['current'] is None:
            raise PermissionError('No episode is running: call start_episode first')
        complete = bool(self.backend.task_state()['task_complete'])
        if name == 'evaluate':
            self.s.record_evaluation(complete)
            return {'task_complete': complete}
        ended = self.s.end_episode(complete)
        self.backend.end_episode()
        return {'episode': ended['index'] + 1, 'task_complete': complete, 'episodes_left': self.s.episodes_left()}

    def call(self, name, arguments):
        """One tools/call; returns an MCP CallToolResult."""
        call_id = uuid.uuid4().hex
        failure = {'stage': 'request'}
        with self._dispatch:
            try:
                if not isinstance(arguments, dict):
                    raise ValueError('Arguments must be an object')
                self.s.audit({'event': 'mcp_request', 'call_id': call_id, 'tool': name, 'arguments': arguments})
                if name == 'status':
                    if arguments:
                        raise ValueError('status takes no arguments')
                    value, images = self.status(), []
                elif self.mode == 'exploration' and name in ('start_episode', 'evaluate', 'finish'):
                    failure['stage'] = 'episode'
                    value, images = self._lifecycle(name, arguments), []
                elif name in self._backend_tools:
                    failure['stage'] = 'validation'
                    self.backend.validate(name, arguments)
                    failure['stage'] = 'robot_execution'
                    value, images = self.backend.call(name, arguments)
                else:
                    raise PermissionError(f'Unknown tool {name!r}')
                value = jsonable(value)
                result = {'content': [{'type': 'text', 'text': json.dumps(value, allow_nan=False)}, *images]}
                self.s.audit({'event': 'mcp_result', 'call_id': call_id, 'tool': name,
                              'result': result['content'][0]['text'][:65536], 'images': len(images)})
                return result
            except Exception as exc:
                handled = isinstance(exc, (ValueError, PermissionError)) or getattr(exc, 'handled_stop', False)
                error = jsonable(public_failure(exc, operation=name, stage=failure['stage'], traceback_tail=not handled))
                try:
                    self.s.audit({'event': 'mcp_error', 'call_id': call_id, **error})
                except Exception:
                    error['audit_recording_failed'] = True
                return {'isError': True, 'content': [{'type': 'text', 'text': json.dumps({'error': error})}]}

    def handle(self, request):
        if not isinstance(request, dict):
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'Invalid request'}}
        ident = request.get('id')
        try:
            if request.get('jsonrpc') != '2.0':
                raise ValueError('Invalid JSON-RPC')
            method = request.get('method')
            if method == 'initialize':
                result = {'protocolVersion': PROTOCOL_VERSION, 'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'robot', 'version': '1.0'}}
            elif isinstance(method, str) and method.startswith('notifications/'):
                return None
            elif method == 'ping':
                result = {}
            elif method == 'tools/list':
                result = {'tools': self.tools()}
            elif method == 'tools/call':
                params = request['params']
                result = self.call(params['name'], params.get('arguments', {}))
            else:
                raise PermissionError('Unsupported method')
            return {'jsonrpc': '2.0', 'id': ident, 'result': result}
        except Exception as exc:
            failure = public_failure(exc)
            return {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32000, 'message': failure['reason'], 'data': failure}}

    def _respond(self, line):
        try:
            request = json.loads(line)
        except (ValueError, UnicodeError):
            return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Invalid request'}}
        return self.handle(request)

    def serve_socket(self, listener, stop=None):
        """Multiplex up to 16 clients; requests run one at a time on this thread."""
        listener.setblocking(False)
        buffers = {}
        with selectors.DefaultSelector() as poll:
            poll.register(listener, selectors.EVENT_READ)
            try:
                while not (stop and stop.is_set()):
                    for key, _ in poll.select(timeout=0.5):
                        connection = key.fileobj
                        if connection is listener:
                            try:
                                client, _ = listener.accept()
                            except BlockingIOError:
                                continue
                            if len(buffers) >= 16:
                                client.close()
                                continue
                            client.setblocking(False)
                            buffers[client] = bytearray()
                            poll.register(client, selectors.EVENT_READ)
                            continue
                        try:
                            block = connection.recv(65536)
                            if not block:
                                raise ConnectionError('Client closed')
                            pending = buffers[connection]
                            pending.extend(block)
                            if len(pending) > MAX_REQUEST_BYTES:
                                raise ConnectionError('Request exceeds wire limit')
                            while b'\n' in pending:
                                line, _, tail = pending.partition(b'\n')
                                pending[:] = tail
                                response = self._respond(line)
                                if response is not None:
                                    connection.settimeout(60)
                                    connection.sendall(json.dumps(response, allow_nan=False).encode() + b'\n')
                                    connection.setblocking(False)
                        except (ConnectionError, OSError):
                            poll.unregister(connection)
                            buffers.pop(connection, None)
                            connection.close()
            finally:
                for client in buffers:
                    client.close()

    def close(self):
        try:
            self.backend.close()
        finally:
            self.s.close()


def serve(config, socket_path, *, mode='exploration', started_at=None, ready=None, stop=None,
          state_name='state.json', audit_name='mcp.jsonl'):
    session = Session(config, state_name=state_name, audit_name=audit_name)
    try:
        backend = make_backend(config)
    except BaseException:
        session.close()
        raise
    frontend = Frontend(session, backend, mode=mode, started_at=started_at)
    socket_path = Path(socket_path)
    try:
        socket_path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            listener.listen(16)
            print(f'MCP listening on {socket_path}', flush=True)
            if ready is not None:
                ready(frontend)
            frontend.serve_socket(listener, stop)
    finally:
        frontend.close()
        socket_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--started-at', type=float)
    args = parser.parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    config = Configuration.load(args.config)

    def stop(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    serve(config, args.socket, started_at=args.started_at)


if __name__ == '__main__':
    main()
