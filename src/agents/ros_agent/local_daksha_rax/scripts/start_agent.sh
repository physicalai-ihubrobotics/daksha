#!/usr/bin/env bash
# Prepare (if needed) and launch a Daksha RAX session with Claude Code.
#   scripts/start_agent.sh <name> [extra launch args, e.g. --non-interactive]
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?usage: scripts/start_agent.sh <session-name> [launch args]}"; shift || true
source scripts/activate_local.sh
PY="$DAKSHA_PYTHON"
if [ ! -d "runtime/sessions/$NAME" ]; then
  $PY -m harness.launcher prepare --name "$NAME" --episodes "${EPISODES:-24}"
fi
exec $PY -m harness.launcher launch --name "$NAME" "$@"
