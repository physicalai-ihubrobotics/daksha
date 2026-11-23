#!/usr/bin/env bash
# Install the Daksha Python environment with uv.
#
#   1. install uv (Astral's installer, into ~/.local/bin) if it's missing
#   2. create the venv (default ~/daksha_venv, Python 3.12, with
#      --system-site-packages so rclpy and the other ROS packages stay visible)
#   3. uv pip install robot_setup/requirements.txt into it
#   4. write ~/activate_daksha.sh
#   5. run verify_env.sh
#
# Safe to re-run: an existing venv is reused and brought in line with
# requirements.txt. Undo everything with uninstall.sh.
#
#   src/robot_setup/scripts/install.sh            # from a fresh checkout
#   ros2 run robot_setup install.sh               # after colcon build
#
# Options:
#   --venv DIR     venv location        (default: $DAKSHA_VENV or ~/daksha_venv)
#   --python VER   Python version       (default: 3.12, ROS Jazzy's python)
#   --recreate     delete the venv first and build it from scratch
#   --bashrc       add `source ~/activate_daksha.sh` to ~/.bashrc
#   -h, --help     show this help

set -euo pipefail

VENV="${DAKSHA_VENV:-$HOME/daksha_venv}"
PYVER="3.12"
RECREATE=0
BASHRC=0

usage() { sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --venv)     VENV="$2"; shift 2 ;;
        --python)   PYVER="$2"; shift 2 ;;
        --recreate) RECREATE=1; shift ;;
        --bashrc)   BASHRC=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- Locate the workspace from this script's own location -------------------
# Works both from src/robot_setup/scripts and install/robot_setup/lib/robot_setup:
# walk up until a directory contains src/robot_setup/package.xml.
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
WS="$SCRIPT_DIR"
while [ "$WS" != "/" ] && [ ! -f "$WS/src/robot_setup/package.xml" ]; do
    WS="$(dirname "$WS")"
done
[ "$WS" != "/" ] || die "could not find the workspace (no src/robot_setup above $SCRIPT_DIR)"
PKG="$WS/src/robot_setup"

# --- ROS distro --------------------------------------------------------------
if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
else
    mapfile -t distros < <(ls -d /opt/ros/*/ 2>/dev/null)
    [ ${#distros[@]} -eq 1 ] || die "found ${#distros[@]} ROS installs in /opt/ros; source the one you want first"
    ROS_SETUP="${distros[0]%/}/setup.bash"
fi

echo "workspace: $WS"
echo "venv:      $VENV (Python $PYVER)"
echo "ROS:       $ROS_SETUP"

# --- 1. uv -------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null; then
    step "uv $(uv --version | cut -d' ' -f2) already installed"
else
    step "Installing uv"
    command -v curl >/dev/null || die "curl is needed to install uv: sudo apt install curl"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    command -v uv >/dev/null || die "uv install failed"
fi

# --- 2. venv -----------------------------------------------------------------
if [ "$RECREATE" -eq 1 ] && [ -e "$VENV" ]; then
    step "Removing $VENV (--recreate)"
    "$SCRIPT_DIR/uninstall.sh" --venv "$VENV" --venv-only --yes
fi
if [ -x "$VENV/bin/python" ]; then
    step "Reusing existing venv $VENV"
    grep -q '^include-system-site-packages = true' "$VENV/pyvenv.cfg" \
        || die "$VENV was created without --system-site-packages, so rclpy won't import. Re-run with --recreate."
else
    step "Creating venv $VENV"
    # /usr/bin/python3 is the interpreter ROS's packages are built for.
    # -q: uv would print "Activate with: source <venv>/bin/activate", which is
    # NOT the right way to activate - use ~/activate_daksha.sh instead.
    uv venv -q --system-site-packages --python "$PYVER" "$VENV"
fi
PY="$VENV/bin/python"

# --- 3. packages -------------------------------------------------------------
step "Installing Python packages (uv)"
uv pip install --python "$PY" -r "$PKG/requirements.txt"

# --- 4. activation script ----------------------------------------------------
ACTIVATE="$HOME/activate_daksha.sh"
step "Writing $ACTIVATE"
sed -e "s|@ROS_SETUP@|$ROS_SETUP|" -e "s|@VENV@|$VENV|" -e "s|@WS@|$WS|" \
    "$PKG/templates/activate_daksha.sh.in" > "$ACTIVATE"
chmod +x "$ACTIVATE"

if [ "$BASHRC" -eq 1 ]; then
    line="source \"$ACTIVATE\""
    if grep -qxF "$line" "$HOME/.bashrc" 2>/dev/null; then
        echo "~/.bashrc already sources it"
    else
        printf '\n# Daksha robot environment (robot_setup)\n%s\n' "$line" >> "$HOME/.bashrc"
        echo "added to ~/.bashrc"
    fi
fi

# --- 5. verify ---------------------------------------------------------------
step "Verifying"
set +u; source "$ACTIVATE" >/dev/null; set -u
"$SCRIPT_DIR/verify_env.sh"

printf '\n\033[1;32mDone.\033[0m Activate with:\n    source %s\n' "$ACTIVATE"
