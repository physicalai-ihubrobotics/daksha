#!/usr/bin/env bash
# Build and install the PEAK PCAN-USB (Pro FD) socketcan driver (peak_usb).
#
# The Jetson Tegra kernel ships without CONFIG_CAN_PEAK_USB, so the adapter
# shows up in lsusb but never gets a CAN interface. This script:
#   1. fetches only drivers/net/can/usb/peak_usb from the matching stable kernel
#   2. builds peak_usb.ko against the running kernel's headers
#   3. installs it, loads it now and on every boot
#   4. adds a udev rule that brings each PEAK channel up automatically
#      (bitrate, restart-ms, txqueuelen from /etc/default/pcan)
#
# The onboard Jetson ports (mttcan, can0-can3) are not touched. PEAK channels
# get the next free names (usually can4/can5); use --status to see which.
#
#   src/robot_setup/scripts/install_pcan.sh
#   ros2 run robot_setup install_pcan.sh       (after colcon build)
#
# Re-run after a kernel update: the module is built for one kernel only.
#
# Options:
#   --bitrate N   CAN bitrate for PEAK channels (default 1000000)
#   --rebuild     rebuild even if peak_usb is already installed
#   --status      show which CAN interfaces are PEAK / onboard, then exit
#   --uninstall   remove the module, udev rule and config
#   -h, --help    show this help

set -euo pipefail

BITRATE=1000000
REBUILD=0
ACTION=install

usage() { sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bitrate)   [ $# -ge 2 ] || { echo "--bitrate needs a value" >&2; exit 2; }
                     BITRATE="$2"; shift 2 ;;
        --rebuild)   REBUILD=1; shift ;;
        --status)    ACTION=status; shift ;;
        --uninstall) ACTION=uninstall; shift ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

[[ "$BITRATE" =~ ^[0-9]+$ ]] || die "bitrate must be a number, got '$BITRATE'"

KERNEL="$(uname -r)"
KBASE="${KERNEL%%-*}"                       # 6.8.12-1021-tegra -> 6.8.12
KDIR="/lib/modules/$KERNEL/build"
SRC="${XDG_CACHE_HOME:-$HOME/.cache}/pcan-build/linux-$KBASE"
DRIVER_DIR="$SRC/drivers/net/can/usb/peak_usb"
MOD_DEST="/lib/modules/$KERNEL/updates/peak_usb.ko"
MODLOAD_CONF=/etc/modules-load.d/peak_usb.conf
UDEV_RULE=/etc/udev/rules.d/80-pcan-up.rules
HELPER=/usr/local/sbin/pcan-up
DEFAULTS=/etc/default/pcan
PEAK_USB_ID="0c72:"                         # PEAK-System vendor ID

# Print "iface driver state" for every CAN interface.
can_status() {
    local found=0 d ifc drv state
    for d in /sys/class/net/*; do
        [ "$(cat "$d/type" 2>/dev/null)" = 280 ] || continue   # ARPHRD_CAN
        ifc="$(basename "$d")"
        drv="$(basename "$(readlink -f "$d/device/driver" 2>/dev/null)" 2>/dev/null || true)"
        state="$(ip -details link show "$ifc" | grep -o 'can state [A-Z-]*' | cut -d' ' -f3 || true)"
        printf '  %-8s %-10s %-6s %s\n' "$ifc" "${drv:-virtual}" \
            "$(cat "$d/operstate")" "${state:-}"
        found=1
    done
    [ $found -eq 1 ] || echo "  (no CAN interfaces)"
}

# Avoid `cmd | grep -q` here: with pipefail, grep -q exiting early SIGPIPEs cmd
# and the pipeline reports failure even on a match.
peak_present() { local out; out="$(lsusb 2>/dev/null || true)"; grep -qi "$PEAK_USB_ID" <<<"$out"; }
peak_loaded()  { [ -d /sys/module/peak_usb ]; }

peak_ifaces() {
    local d
    for d in /sys/class/net/*; do
        [ "$(basename "$(readlink -f "$d/device/driver" 2>/dev/null)" 2>/dev/null)" = peak_usb ] \
            && basename "$d"
    done
    return 0
}

if [ "$ACTION" = status ]; then
    echo "CAN interfaces (name / driver / link / bus state):"
    can_status
    peak_present && echo "PEAK adapter: plugged in" || echo "PEAK adapter: not plugged in"
    peak_loaded  && echo "peak_usb module: loaded"  || echo "peak_usb module: not loaded"
    exit 0
fi

[ "$EUID" -ne 0 ] || die "run as your normal user; the script calls sudo itself"

# --- Uninstall ---------------------------------------------------------------
if [ "$ACTION" = uninstall ]; then
    step "Removing PEAK driver and config"
    sudo modprobe -r peak_usb 2>/dev/null || warn "peak_usb not unloaded (in use or not loaded)"
    sudo rm -f "$MOD_DEST" "$MODLOAD_CONF" "$UDEV_RULE" "$HELPER" "$DEFAULTS"
    sudo depmod -a "$KERNEL"
    sudo udevadm control --reload-rules
    echo "Build cache left at $SRC (delete it by hand if you want)."
    printf '\033[1;32mDone.\033[0m\n'
    exit 0
fi

# --- Pre-flight --------------------------------------------------------------
step "Checking prerequisites (kernel $KERNEL)"
[ -f "$KDIR/Makefile" ] || die "kernel headers missing at $KDIR"
[ -f "$KDIR/include/linux/can/dev/peak_canfd.h" ] \
    || die "headers lack include/linux/can/dev/peak_canfd.h - cannot build peak_usb"
if grep -q '^CONFIG_MODULE_SIG_FORCE=y' "$KDIR/.config" 2>/dev/null; then
    die "kernel enforces module signatures; an unsigned peak_usb.ko will not load"
fi
if peak_present; then
    echo "  PEAK adapter: $(lsusb | grep -i "$PEAK_USB_ID" | cut -d' ' -f6-)"
else
    warn "no PEAK adapter in lsusb - installing anyway, plug it in afterwards"
fi

# --- Dependencies ------------------------------------------------------------
step "Installing build dependencies"
sudo apt-get update -qq
sudo apt-get install -y git build-essential can-utils

# --- Build -------------------------------------------------------------------
if [ $REBUILD -eq 0 ] && [ -f "$MOD_DEST" ] && modinfo -k "$KERNEL" peak_usb >/dev/null 2>&1; then
    step "peak_usb already installed for $KERNEL (use --rebuild to rebuild)"
else
    step "Fetching peak_usb source from linux v$KBASE"
    if [ ! -d "$SRC/.git" ]; then
        mkdir -p "$(dirname "$SRC")"
        git clone --depth 1 --branch "v$KBASE" --filter=blob:none --sparse \
            https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git "$SRC" \
            || die "git clone failed (no network, or tag v$KBASE does not exist)"
    fi
    git -C "$SRC" sparse-checkout set drivers/net/can/usb/peak_usb
    [ -f "$DRIVER_DIR/pcan_usb_fd.c" ] || die "driver source not found in $DRIVER_DIR"

    step "Building peak_usb.ko"
    make -C "$KDIR" M="$DRIVER_DIR" clean >/dev/null
    make -C "$KDIR" M="$DRIVER_DIR" CONFIG_CAN_PEAK_USB=m -j"$(nproc)" modules
    [ -f "$DRIVER_DIR/peak_usb.ko" ] || die "build finished but peak_usb.ko is missing"

    step "Installing module to $MOD_DEST"
    sudo modprobe -r peak_usb 2>/dev/null || true
    sudo install -D -m 644 "$DRIVER_DIR/peak_usb.ko" "$MOD_DEST"
    sudo depmod -a "$KERNEL"
fi

# --- Auto-load and auto bring-up ---------------------------------------------
step "Configuring load at boot and automatic bring-up (bitrate $BITRATE)"
echo peak_usb | sudo tee "$MODLOAD_CONF" >/dev/null

sudo tee "$DEFAULTS" >/dev/null <<EOF
# Settings for PEAK CAN channels, applied by $HELPER on plug-in/boot.
# Written by install_pcan.sh. Edit and replug (or run: sudo $HELPER <iface>).
BITRATE=$BITRATE
RESTART_MS=100
TXQUEUELEN=1000
EOF

sudo tee "$HELPER" >/dev/null <<'EOF'
#!/bin/sh
# Bring up one PEAK CAN channel. Called by udev; installed by install_pcan.sh.
IFACE="$1"
[ -n "$IFACE" ] || { echo "usage: $0 <iface>" >&2; exit 2; }
BITRATE=1000000; RESTART_MS=100; TXQUEUELEN=1000
[ -r /etc/default/pcan ] && . /etc/default/pcan
ip link set "$IFACE" down 2>/dev/null
ip link set "$IFACE" type can bitrate "$BITRATE" restart-ms "$RESTART_MS" || exit 1
ip link set "$IFACE" txqueuelen "$TXQUEUELEN"
ip link set "$IFACE" up || exit 1
logger -t pcan-up "$IFACE up at $BITRATE bit/s"
EOF
sudo chmod 755 "$HELPER"

sudo tee "$UDEV_RULE" >/dev/null <<EOF
# Bring up PEAK USB CAN channels when they appear. Written by install_pcan.sh.
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", DRIVERS=="peak_usb", RUN+="$HELPER %k"
EOF
sudo udevadm control --reload-rules

# --- Load now ----------------------------------------------------------------
step "Loading peak_usb"
sudo modprobe can_dev
sudo modprobe peak_usb || die "modprobe peak_usb failed - see: sudo dmesg | tail"
sleep 1
# Interfaces that existed before the rule was installed: bring them up directly.
for ifc in $(peak_ifaces); do
    sudo "$HELPER" "$ifc" || warn "could not bring up $ifc"
done

# --- Report ------------------------------------------------------------------
step "Result"
echo "CAN interfaces (name / driver / link / bus state):"
can_status
mapfile -t peaks < <(peak_ifaces)
echo
if [ ${#peaks[@]} -gt 0 ]; then
    printf '\033[1;32mDone.\033[0m PEAK channels: %s\n' "${peaks[*]}"
    echo "Scan motors with:"
    for ifc in "${peaks[@]}"; do echo "  damiao scan --channel $ifc"; done
    echo "Names can change if adapters are added; check with: $0 --status"
else
    printf '\033[1;32mDriver installed.\033[0m '
    echo "No PEAK channels yet - plug in the adapter, then run: $0 --status"
fi
