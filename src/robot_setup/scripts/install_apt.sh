#!/usr/bin/env bash
# Install APT packages listed in robot_setup/apt_requirements.txt.
#
# Reads one package name per line (ignores blank lines and # comments).
# Safe to re-run: already-installed packages are skipped by apt.
#
#   src/robot_setup/scripts/install_apt.sh
#   ros2 run robot_setup install_apt.sh       (after colcon build)
#
# Options:
#   --dry-run   show packages that would be installed, don't install
#   --no-update skip `apt update` (use cached package lists)
#   -h, --help  show this help

set -euo pipefail

DRY_RUN=0
SKIP_UPDATE=0

usage() { sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --no-update) SKIP_UPDATE=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# --- Locate apt_requirements.txt from this script's own location -------------
# Works from src/robot_setup/scripts and install/robot_setup/lib/robot_setup.
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
WS="$SCRIPT_DIR"
while [ "$WS" != "/" ] && [ ! -f "$WS/src/robot_setup/package.xml" ]; do
    WS="$(dirname "$WS")"
done
[ "$WS" != "/" ] || die "could not find the workspace (no src/robot_setup above $SCRIPT_DIR)"

REQS="$WS/src/robot_setup/apt_requirements.txt"
[ -f "$REQS" ] || die "apt_requirements.txt not found at $REQS"

# --- Parse the file ----------------------------------------------------------
mapfile -t packages < <(grep -v '^\s*#' "$REQS" | grep -v '^\s*$')
[ ${#packages[@]} -gt 0 ] || { echo "No packages in $REQS."; exit 0; }

echo "APT packages from $REQS:"
printf '  %s\n' "${packages[@]}"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "(dry-run: not installing)"
    exit 0
fi

command -v apt-get >/dev/null || die "apt-get not found - this script is for Debian/Ubuntu only"

# --- apt update --------------------------------------------------------------
if [ "$SKIP_UPDATE" -eq 0 ]; then
    step "apt update"
    sudo apt-get update -qq
fi

# --- Install -----------------------------------------------------------------
step "Installing ${#packages[@]} packages"
sudo apt-get install -y "${packages[@]}"

echo
printf '\033[1;32mDone.\033[0m\n'

# --- Quick verification ------------------------------------------------------
step "Verifying"
fail=0
for pkg in "${packages[@]}"; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then
        printf '  \033[32mOK\033[0m    %s\n' "$pkg"
    else
        printf '  \033[31mFAIL\033[0m  %s (not found after install)\n' "$pkg"
        fail=1
    fi
done
exit $fail
