#!/usr/bin/env bash
# Source this file from any shell to use the shared Daksha environment here.
DAKSHA_RAX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DAKSHA_PYTHON="${DAKSHA_PYTHON:-/home/ihub/daksha_venv/bin/python}"
export PATH="$(dirname "$DAKSHA_PYTHON"):$PATH"
export PYTHONPATH="$DAKSHA_RAX_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export LOCAL_VLM_MODEL="${LOCAL_VLM_MODEL:-35b}"
unset DAKSHA_RAX_ROOT
echo "Daksha RAX active: $DAKSHA_PYTHON; Ollama $OLLAMA_HOST; model $LOCAL_VLM_MODEL"
