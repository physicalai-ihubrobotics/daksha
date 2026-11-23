"""Daksha sim backend + a local VLM tool (RAX `vlm-perception` skill's `<vlm>` tool = `vlm_query`).

Same robot tools as DakshaSimBackend, plus `vlm_query`, served by a local Ollama model
(default qwen3.5:35b; LOCAL_VLM_MODEL=35b|9b|8b or any Ollama name; OLLAMA_HOST). No hosted API is used.
Select it with `python -m harness.launcher prepare --name s1 --robot daksha_sim_ollama`.
"""
import json
import pathlib

from local_pipeline.ollama_vlm import OllamaVLM, VLMError
from services.daksha_sim.backend import DakshaSimBackend, CAMERAS

VLM_TOOL = {
    'name': 'vlm_query',
    'description': (
        'Ask the local vision-language model (Ollama, LOCAL_VLM_MODEL) about a saved camera frame. Give the frame_id '
        'from daksha_observe and a camera. With `detect` (object names) it returns boxes and centre points in '
        'PIXELS of that image: [{"label", "bbox": [x1,y1,x2,y2], "point": [u,v]}], ready for '
        'daksha_pixel_to_position. Otherwise it answers `prompt` (set json=true for a JSON reply). A timeout, '
        'refusal or truncated reply comes back as an error: treat it as no answer, never as a negative. '
        'Takes ~5 s per call on this machine; ask between motions, not during them.'),
    'inputSchema': {'type': 'object', 'additionalProperties': False, 'required': ['frame_id', 'camera'],
                    'properties': {
                        'frame_id': {'type': 'string'},
                        'camera': {'type': 'string', 'enum': list(CAMERAS)},
                        'prompt': {'type': 'string', 'maxLength': 4000},
                        'detect': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 12},
                        'json': {'type': 'boolean'},
                        'max_tokens': {'type': 'integer', 'minimum': 16, 'maximum': 2048}}},
}


class DakshaSimOllamaBackend(DakshaSimBackend):
    def __init__(self, config):
        super().__init__(config)
        self.vlm = OllamaVLM()

    def tools(self):
        return [*super().tools(), VLM_TOOL]

    def validate(self, name, args):
        if name != 'vlm_query':
            return super().validate(name, args)
        if not args.get('prompt') and not args.get('detect'):
            raise ValueError('vlm_query needs prompt or detect')

    def call(self, name, args):
        if name != 'vlm_query':
            return super().call(name, args)
        frame = pathlib.Path(self.frames_dir)/str(args['frame_id'])/f'{args["camera"]}.png'
        if '/' in str(args['frame_id']) or not frame.exists():
            raise ValueError(f'No saved image for frame {args["frame_id"]!r} camera {args["camera"]!r}; observe first')
        png = frame.read_bytes()
        try:
            if args.get('detect'):
                value = {'detections': self.vlm.detect(png, args['detect'])}
            else:
                text = self.vlm.chat(args['prompt'], [png], json_mode=bool(args.get('json')),
                                     max_tokens=int(args.get('max_tokens', 512)))
                value = {'answer': json.loads(text) if args.get('json') else text}
        except (VLMError, ValueError) as exc:
            raise RuntimeError(f'vlm_query gave no answer: {exc}') from None
        value['model'] = self.vlm.last
        return value, []
