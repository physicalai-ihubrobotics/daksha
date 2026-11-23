"""Local vision-language model client for Ollama (stdlib only, no API keys).

Models (all vision-capable on this machine; same 0..1000 box convention):
    35b -> qwen3.5:35b  (default: MoE, ~5 s detection, best plans)
    9b  -> qwen3.5:9b
    8b  -> qwen3-vl:8b

    vlm = OllamaVLM()                              # http://localhost:11434, model qwen3.5:35b
    vlm = OllamaVLM(model='9b')                    # aliases or any full Ollama model name
    vlm.detect(rgb, ['apple', 'bowl'])             # -> [{'label', 'bbox': [x1,y1,x2,y2] px, 'point': [u,v] px}]
    vlm.ask(rgb, 'Is the gripper holding something?')

Qwen3-VL answers boxes and points in a 0..1000 normalised frame of the image it was sent;
this client converts them back to pixels of the original image. Timeouts, refusals and
unparseable replies raise VLMError: a caller must treat them as "no answer", never as a negative.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

DEFAULT_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
MODEL_ALIASES = {'35b': 'qwen3.5:35b', '9b': 'qwen3.5:9b', '8b': 'qwen3-vl:8b'}


def resolve_model(name):
    return MODEL_ALIASES.get(str(name).lower().strip(), name)


DEFAULT_MODEL = resolve_model(os.environ.get('LOCAL_VLM_MODEL', '35b'))


class VLMError(RuntimeError):
    pass


def _jpeg_b64(image, max_side=640, quality=85):
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert('RGB')
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, 'JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _size(image):
    if isinstance(image, np.ndarray):
        return image.shape[1], image.shape[0]
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).size
    return image.size


def strip_think(text):
    """Drop a <think>...</think> block (or an unterminated one) from a reply."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    return re.sub(r'<think>.*', '', text, flags=re.S).strip()


def detection_schema(labels=None):
    box = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 4, 'maxItems': 4}
    point = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 2, 'maxItems': 2}
    label = {'type': 'string'}
    if labels is not None:
        label['enum'] = list(labels)
    item = {'type': 'object', 'required': ['label', 'bbox_2d', 'point_2d'],
            'properties': {'label': label, 'bbox_2d': box, 'point_2d': point}}
    return {'type': 'object', 'required': ['objects'],
            'properties': {'objects': {'type': 'array', 'items': item,
                                       **({'maxItems': 2*len(labels)} if labels is not None else {})}}}


def parse_json(text):
    """The first JSON value in a reply (tolerates ```json fences and surrounding prose)."""
    text = strip_think(text)
    text = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    for opener, closer in (('[', ']'), ('{', '}')):
        i, j = text.find(opener), text.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except ValueError:
                continue
    raise VLMError(f'unparseable reply: {text[:200]!r}')


class OllamaVLM:
    def __init__(self, host=DEFAULT_HOST, model=DEFAULT_MODEL, timeout=180, num_ctx=8192, keep_alive='30m'):
        self.host, self.model, self.timeout = host.rstrip('/'), resolve_model(model), timeout
        self.num_ctx, self.keep_alive = num_ctx, keep_alive
        self.last = {}  # latency and token counts of the last call, for the UI

    # ------------------------------------------------------------------ transport
    def chat(self, prompt, images=(), *, json_mode=False, schema=None, prefill=None, system=None, max_tokens=1024,
             temperature=0.0):
        """One chat turn.

        Qwen3-VL / Qwen3.5 may write a <think> block even with think=false, which burns the token budget.
        `prefill` starts the assistant reply (e.g. '{"objects": [') so the model continues the JSON directly:
        fastest, ~3x fewer tokens than constrained decoding. `schema`/`json_mode` constrain decoding instead
        (slower, used as the retry). The returned text includes the prefill."""
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt, 'images': [_jpeg_b64(i) for i in images]})
        if prefill:
            messages.append({'role': 'assistant', 'content': prefill})
        body = {'model': self.model, 'messages': messages, 'stream': False, 'think': False,
                'keep_alive': self.keep_alive,
                'options': {'temperature': temperature, 'num_predict': max_tokens, 'num_ctx': self.num_ctx}}
        if schema is not None:
            body['format'] = schema
        elif json_mode:
            body['format'] = 'json'
        req = urllib.request.Request(f'{self.host}/api/chat', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                reply = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise VLMError(f'Ollama at {self.host} unreachable or timed out: {exc}') from None
        if 'error' in reply:
            raise VLMError(f'Ollama error: {reply["error"]}')
        self.last = {'seconds': round(time.time() - t, 2), 'model': self.model,
                     'prompt_tokens': reply.get('prompt_eval_count'), 'output_tokens': reply.get('eval_count'),
                     'done_reason': reply.get('done_reason')}
        if reply.get('done_reason') == 'length':
            raise VLMError('reply truncated (token budget): no answer')
        return strip_think((prefill or '') + reply['message']['content'])

    def available(self):
        try:
            with urllib.request.urlopen(f'{self.host}/api/tags', timeout=5) as r:
                names = [m['name'] for m in json.loads(r.read())['models']]
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return False, []
        return self.model in names, names

    def warm_up(self):
        """Load the model into GPU memory now, so the first real question is not a cold start."""
        try:
            self.chat('Reply with the single word OK.', prefill='O', max_tokens=16)
        except VLMError as exc:
            if 'truncated' not in str(exc):
                raise

    # ------------------------------------------------------------------ helpers
    def ask(self, image, question, **kw):
        return self.chat(question, [image] if image is not None else [], **kw)

    def ask_json(self, image, question, prefill='{', **kw):
        """A JSON object: prefilled reply first, constrained JSON decoding if that fails."""
        images = [image] if image is not None else []
        try:
            value = parse_json(self.chat(question, images, prefill=prefill, **kw))
            if isinstance(value, dict):
                return value
        except VLMError:
            pass
        kw['max_tokens'] = 2*kw.get('max_tokens', 1024)
        return parse_json(self.chat(question, images, json_mode=True, **kw))

    def detect(self, image, labels=None):
        """Find visible objects, optionally restricting detection to the given labels."""
        w, h = _size(image)
        if labels is None:
            prompt = ('Identify every clearly visible movable object and container on the table. Give each a short, '
                      'specific common name that a person could use to refer to it. Exclude the robot, people, '
                      'table, and background. For each visible item give its bounding box bbox_2d '
                      '[x1, y1, x2, y2] and centre point_2d [x, y]. Coordinates use a 0 to 1000 scale. '
                      'Output JSON only: {"objects": [{"label": "<short name>", "bbox_2d": '
                      '[x1, y1, x2, y2], "point_2d": [x, y]}]}')
        else:
            prompt = ('Locate each of these objects in the image: ' + ', '.join(labels) + '. '
                      'For each one that is clearly visible give its bounding box bbox_2d [x1, y1, x2, y2] and the '
                      'centre of the object itself, point_2d [x, y]. Leave out objects that are not visible. '
                      'Coordinates use a 0 to 1000 scale. Output JSON only: {"objects": [{"label": "<name>", '
                      '"bbox_2d": [x1, y1, x2, y2], "point_2d": [x, y]}]}')
        def usable(d):
            items = d.get('objects') if isinstance(d, dict) else d
            return isinstance(items, list) and any(isinstance(x, dict) and 'bbox_2d' in x and 'label' in x
                                                   for x in items)
        try:
            data = parse_json(self.chat(prompt, [image], prefill='{"objects": [',
                                        max_tokens=120*len(labels) + 100 if labels else 1400))
            if not usable(data):
                # qwen3.5 sometimes answers the prefill with '{"objects": [{}]}' (an empty
                # entry, not "nothing visible"): ask again with constrained decoding.
                raise VLMError('no usable detection in the prefilled reply')
        except VLMError:  # truncated, malformed or empty: retry with schema-constrained decoding
            data = parse_json(self.chat(prompt, [image], schema=detection_schema(labels), max_tokens=1600))
        if isinstance(data, dict):
            data = data.get('objects') or data.get('detections') or [data]
        out, wanted = [], {l.lower().replace(' ', '_'): l for l in labels} if labels is not None else None
        for d in data if isinstance(data, list) else []:
            try:
                label = str(d['label']).lower().strip().replace(' ', '_')
                x1, y1, x2, y2 = (float(v) for v in d['bbox_2d'])
            except (KeyError, TypeError, ValueError):
                continue
            match = (wanted.get(label) or next((v for k, v in wanted.items()
                                                 if k in label or label in k), None)) if wanted is not None else label
            if not match:
                continue
            sx, sy = w/1000.0, h/1000.0
            box = [x1*sx, y1*sy, x2*sx, y2*sy]
            pt = d.get('point_2d')
            try:
                point = [float(pt[0])*sx, float(pt[1])*sy]
            except (TypeError, ValueError, IndexError):
                point = [(box[0] + box[2])/2, (box[1] + box[3])/2]
            out.append({'label': match, 'bbox': [round(v, 1) for v in box], 'point': [round(v, 1) for v in point]})
        return out
