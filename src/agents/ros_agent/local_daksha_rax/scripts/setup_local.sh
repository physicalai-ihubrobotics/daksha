#!/usr/bin/env bash
# Set up the project in /home/ihub/daksha_venv and check its local Ollama model.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${DAKSHA_PYTHON:-/home/ihub/daksha_venv/bin/python}"
MODEL="${LOCAL_VLM_MODEL:-qwen3.5:35b}"
case "$MODEL" in
  35b) MODEL=qwen3.5:35b ;;
  9b) MODEL=qwen3.5:9b ;;
  8b) MODEL=qwen3-vl:8b ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found: $PYTHON (set DAKSHA_PYTHON to override)" >&2
  exit 1
fi

if ! "$PYTHON" -c 'import mujoco' >/dev/null 2>&1; then
  command -v uv >/dev/null || { echo "uv is required to install MuJoCo into $PYTHON" >&2; exit 1; }
  uv pip install --python "$PYTHON" 'mujoco==4.0.0'
fi

PYTHONPATH="$ROOT" "$PYTHON" - <<'PY'
import importlib

for module in ('mujoco', 'numpy', 'scipy', 'PIL', 'yaml', 'placo'):
    importlib.import_module(module)
print('Python dependencies available in the Daksha environment.')
PY

if ! curl --fail --silent "${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags" >/dev/null; then
  echo "Ollama is not responding at ${OLLAMA_HOST:-http://127.0.0.1:11434}; start it with 'ollama serve'." >&2
  exit 1
fi

if ! ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "$MODEL"; then
  ollama pull "$MODEL"
fi
echo "Setup ready: $ROOT using $PYTHON and Ollama model $MODEL."
