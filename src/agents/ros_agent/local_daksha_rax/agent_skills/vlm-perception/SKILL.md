---
name: vlm-perception
description: Use the harness's vision-language model tool for image questions (identify, point at, segment, check) from direct tool calls or inside Python controllers, with your own prompts and response handling. Use when simple perception rules are insufficient or break on new views.
---

# VLM perception

The harness exposes one model tool (`vlm_query`) through MCP, callable directly or from code
with `ctx.call("vlm_query", ...)`. It adds no perception instructions, task logic or action
selection: you write the prompts and the code that reads the answers. Read the tool's
schema and the branch's API reference for request fields, structured output and limits.

## What to ask

- **2-D questions to the VLM, 3-D questions to geometry.** VLMs are reliable for
  identity, visibility, gross state, named points and masks. They are not reliable for
  small 3-D changes ("lifted?", "touching?"): measure those with depth, kinematics or
  the gripper.
- **Ask for more geometry, not less.** Several named points or a mask survive a hard
  view better than one point.
- **Point, then verify.** Ask for the same named, distinctive feature in each view, and
  accept it only when [object-localization](../object-localization/SKILL.md) puts the
  answers at one physical point. Keep a non-VLM fallback.
- **Draw before trusting.** Learn the model's coordinate convention (axis order,
  normalization, which image size it refers to) and draw a new prompt's output on the
  image once before using it in code.
- Prefer a simple, hard-coded rule only where it is verified across layouts; colour,
  size and dark-pixel rules tend to break on new views. Test non-perception uses, such
  as structured decisions, before relying on them.

## How to call

- The tool reads a SAVED frame, not image bytes: call `daksha_observe` first, then pass its
  `frame_id` and a `camera` name. Do not send image data or print base64 into your context.
- Request structured output for anything code reads (`json: true`, or `detect`), and give
  `max_tokens` room: a truncated reply is no answer.
- Treat a timeout, an error or an unparsable reply as no answer, never as a negative.
- The model runs locally (Ollama): no spending cap, no reasoning-effort setting, no tool
  calls of its own. A call takes seconds and can delay robot calls where the host
  serializes them: ask between motions, not during them.

Model answers are controller input, not trusted evidence of task success.

## This harness (local_daksha_rax, robot `daksha_sim_ollama`)

- The tool is `vlm_query`, served by a local Ollama `qwen3-vl:8b` (~5 s per call, no spending cap).
  It reads a saved frame: pass `frame_id` from `daksha_observe` and a `camera`.
- Arguments: `frame_id` (string, required), `camera` (string, required), and either
  `detect` (list of up to 12 names) or `prompt` (string; with `json: true` for a JSON answer),
  optional `max_tokens` (16-2048). No other arguments are accepted.
- `detect` returns `{"detections": [{"label": str, "bbox": [x1, y1, x2, y2], "point": [u, v]}, ...]}`
  in pixels of that image (already converted from Qwen's 0..1000 frame); feed sampled box pixels
  to `daksha_pixel_to_position` and keep the points above the table.
- `prompt` returns `{"answer": <text, or the parsed JSON with json: true>}`.
- If the session runs plain `daksha_sim`, this tool is absent: use colour/depth perception instead.
- On the real robot (RAX UI agent) the vision model is reached through `daksha_find` /
  `daksha_track` (boxes + 3-D positions) instead of `vlm_query`.
