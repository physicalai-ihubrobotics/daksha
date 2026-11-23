#!/usr/bin/env bash
# Check the Daksha environment. Run after `source ~/activate_daksha.sh`.
#
#   src/robot_setup/scripts/verify_env.sh
#   ros2 run robot_setup verify_env.sh
#
# Checks EVERY package in robot_setup/requirements.txt - installed, version
# matches the pin, modules import - under both the venv's python and
# /usr/bin/python3 (what `ros2 run` nodes usually use). Also checks ROS,
# dependency consistency, the damiao CLI, uv and the workspace.
# Exits non-zero if anything fails.
#
# Options:
#   --venv-only    skip the /usr/bin/python3 pass (faster)
#   -h, --help     show this help

set -uo pipefail

SYSTEM_PASS=1
case "${1:-}" in
    --venv-only) SYSTEM_PASS=0 ;;
    -h|--help)   sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
    "") ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
esac

fail=0
ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=1; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
section() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# --- Locate requirements.txt from this script's own location ------------------
# Works from src/robot_setup/scripts and install/robot_setup/lib/robot_setup.
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
WS="$SCRIPT_DIR"
while [ "$WS" != "/" ] && [ ! -f "$WS/src/robot_setup/package.xml" ]; do
    WS="$(dirname "$WS")"
done
REQS="$WS/src/robot_setup/requirements.txt"
CHECKER="$WS/src/robot_setup/scripts/check_requirements.py"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

section "Environment"
if [ -n "${VIRTUAL_ENV:-}" ]; then ok "venv active: $VIRTUAL_ENV"
else bad "no venv active - run: source ~/activate_daksha.sh"; fi
if [ -n "${ROS_DISTRO:-}" ]; then ok "ROS $ROS_DISTRO sourced"
else bad "ROS not sourced"; fi
[ -f "$REQS" ] || { bad "requirements file not found: $REQS"; exit 1; }

# A plain `source ~/daksha_venv/bin/activate` (or ROS from ~/.bashrc) is NOT
# enough: placo then loads ROS's eigenpy and /usr/bin/python3 can't see the
# venv. Check for what activate_daksha.sh sets up, and stop early with one
# clear message instead of a page of import failures.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    site="$("$VIRTUAL_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    missing=()
    case ":${PYTHONPATH:-}:" in *":$site:"*) ;; *) missing+=("venv site-packages not on PYTHONPATH") ;; esac
    if [ -d "$site/cmeel.prefix/lib" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in *":$site/cmeel.prefix/lib:"*) ;;
            *) missing+=("placo libs (cmeel.prefix/lib) not on LD_LIBRARY_PATH") ;; esac
    fi
    if [ ${#missing[@]} -gt 0 ]; then
        bad "~/activate_daksha.sh is not sourced in this shell (the venv alone is not enough):"
        printf '          - %s\n' "${missing[@]}"
        printf '\n  Fix:  source ~/activate_daksha.sh   then run this again.\n'
        printf '        (re-run install.sh with --bashrc to do that in every new terminal)\n'
        exit 1
    fi
    ok "activate_daksha.sh environment present"
fi

# --- ROS Python packages (from apt, not requirements.txt) --------------------
section "ROS Python packages"
pythons=()
[ -n "${VIRTUAL_ENV:-}" ] && pythons+=("$VIRTUAL_ENV/bin/python")
[ "$SYSTEM_PASS" -eq 1 ] && pythons+=(/usr/bin/python3)
for py in "${pythons[@]}"; do
    for mod in rclpy; do
        if out="$("$py" -c "import $mod" 2>&1)"; then ok "$py: import $mod"
        else bad "$py: import $mod -> ${out##*$'\n'}"; fi
    done
done

# --- Every package in requirements.txt ---------------------------------------
for py in "${pythons[@]}"; do
    section "requirements.txt with $py"
    "$py" "$CHECKER" "$REQS" ${VIRTUAL_ENV:+--venv "$VIRTUAL_ENV"} || fail=1
done

# --- Dependency consistency --------------------------------------------------
section "Dependency consistency (uv pip check)"
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$UV" ]; then
    if out="$("$UV" pip check --python "$VIRTUAL_ENV/bin/python" 2>&1)"; then
        ok "${out##*$'\n'}"
    else
        bad "conflicting dependencies:"
        printf '%s\n' "$out" | sed 's/^/          /'
    fi
fi

# --- CLIs --------------------------------------------------------------------
section "Tools"
if command -v damiao >/dev/null && damiao --help >/dev/null 2>&1; then
    ok "damiao CLI: $(command -v damiao)"
else
    bad "damiao CLI not on PATH"
fi
if [ -x "$UV" ]; then ok "$("$UV" --version)"
else bad "uv not installed - run install.sh"; fi

# --- Workspace ---------------------------------------------------------------
section "Workspace"
if ros2 pkg prefix robot_setup >/dev/null 2>&1; then
    ok "workspace sourced ($(ros2 pkg prefix robot_setup | xargs dirname | xargs dirname))"
else
    # Only a warning: on a fresh machine install.sh runs before the first build.
    warn "workspace not built/sourced yet - run colcon build, then source ~/activate_daksha.sh"
fi

echo
if [ "$fail" -eq 0 ]; then printf '\033[1;32mAll checks passed.\033[0m\n'
else printf '\033[1;31mSome checks failed.\033[0m\n'; fi
exit $fail
