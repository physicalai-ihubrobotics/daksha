#!/bin/bash
# start_robot.sh
# ----------------------------------------------------------------------------
# Starts the automation launch-control web panel (http://localhost:8100).
# The panel itself auto-launches the robot bringup stack on its first
# startup (skipped if one is already running -- see
# _bringup_already_running() in bring_up.py); Kill/relaunch after
# that is done from the web UI.
#
# Paths are resolved from this script's own location, not hardcoded, so the
# same file works no matter which machine/user the workspace is cloned under
# (as long as the folder layout .../src/automation/bash/start_robot.sh is
# preserved).
# ----------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "[start_robot] repo root: $REPO_ROOT"

cd "$REPO_ROOT/src"
echo "[start_robot] working directory: $(pwd)"

source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

echo "[start_robot] starting automation launch-control panel..."
exec ros2 run automation bring_up.py
