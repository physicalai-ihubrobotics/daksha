"""Stdlib MCP client for robot calls from Python (the RAX runtime API).

    from api.runtime import Context
    with Context() as ctx:
        ctx.value('daksha_status')                 # decoded JSON reply
        ctx.call('daksha_observe', cameras=['world'])  # full MCP result (JSON text + images)

A controller is a module that defines ``main(ctx)``. Run it against the session with

    python -m api.runtime run code/<project>/controller.py

The formal evaluation runs the same ``main(ctx)`` unattended on held-out layouts.
"""
import base64
import importlib.util
import itertools
import json
import os
import socket
import sys
import threading


class ToolError(RuntimeError):
    """A tool returned an error; ``.error`` is the decoded error object."""

    def __init__(self, tool, error):
        self.tool, self.error = tool, error
        reason = error.get('reason') if isinstance(error, dict) else str(error)
        super().__init__(f'{tool} failed: {reason}')


def _find_socket():
    """The robot socket: $RAX_MCP_SOCKET, else the session's robot.sock found from the working
    directory (the workspace sits inside the session folder), else the UI agent's socket."""
    if os.environ.get('RAX_MCP_SOCKET'):
        return os.environ['RAX_MCP_SOCKET']
    here = os.path.abspath(os.getcwd())
    while True:
        cand = os.path.join(here, 'robot.sock')
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    cand = f'/tmp/daksha_rax_{os.getuid()}.sock'
    if os.path.exists(cand):
        return cand
    raise RuntimeError('no robot socket: set RAX_MCP_SOCKET to the session\'s robot.sock '
                       '(or run from inside the session workspace)')


class Context:
    def __init__(self, path=None):
        self._socket = socket.socket(socket.AF_UNIX)
        self._socket.connect(path or _find_socket())
        self._reader = self._socket.makefile('r', encoding='utf-8')
        self._writer = self._socket.makefile('w', encoding='utf-8')
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self.request('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                                    'clientInfo': {'name': 'agent-python', 'version': '1'}})

    def request(self, method, params=None):
        with self._lock:
            ident = next(self._ids)
            self._writer.write(json.dumps({'jsonrpc': '2.0', 'id': ident, 'method': method, 'params': params or {}}) + '\n')
            self._writer.flush()
            line = self._reader.readline()
            if not line:
                raise ConnectionError('Robot service closed the connection')
            response = json.loads(line)
            if response.get('id') != ident:
                raise RuntimeError('MCP request failed: identity mismatch')
            if 'error' in response:
                raise RuntimeError('MCP request failed: ' + json.dumps(response['error']))
            return response['result']

    def tools(self):
        return self.request('tools/list')['tools']

    def call(self, tool, **arguments):
        """Call one tool; returns {'content': [...]}. Raises ToolError on a tool error."""
        result = self.request('tools/call', {'name': tool, 'arguments': arguments})
        if result.get('isError'):
            text = next((b.get('text', '') for b in result.get('content', []) if b.get('type') == 'text'), '{}')
            try:
                error = json.loads(text).get('error', text)
            except ValueError:
                error = text
            raise ToolError(tool, error)
        return result

    def value(self, tool, **arguments):
        """Call a tool and decode its JSON text block."""
        result = self.call(tool, **arguments)
        return json.loads(next(b['text'] for b in result['content'] if b.get('type') == 'text'))

    @staticmethod
    def images(result):
        """Decoded image bytes of a call result, in order."""
        return [base64.b64decode(b['data']) for b in result.get('content', []) if b.get('type') == 'image']

    def close(self):
        for f in (self._reader, self._writer, self._socket):
            try:
                f.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_controller(path):
    spec = importlib.util.spec_from_file_location('controller', path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'main', None)):
        raise SystemExit(f'{path} defines no main(ctx)')
    return module


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != 'run':
        raise SystemExit('usage: python -m api.runtime run code/<project>/controller.py')
    module = load_controller(argv[1])
    with Context() as ctx:
        result = module.main(ctx)
    if result is not None:
        print(json.dumps(result, default=str, indent=1))


if __name__ == '__main__':
    main()
