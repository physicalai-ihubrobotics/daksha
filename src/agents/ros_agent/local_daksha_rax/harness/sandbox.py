"""Detect whether bubblewrap can isolate processes on this host.

Ubuntu 24.04 restricts unprivileged user namespaces (kernel.apparmor_restrict_unprivileged_userns=1),
which stops bwrap, and therefore Claude Code's Bash sandbox and the formal evaluator's sandbox,
unless an AppArmor profile allows it. See README.md "Full isolation" for the one-time fix.
"""
import functools
import shutil
import subprocess

FIX = """\
Full isolation needs bubblewrap, which this host's AppArmor policy blocks. One-time fix (needs sudo):

  sudo tee /etc/apparmor.d/bwrap >/dev/null <<'EOF'
  abi <abi/4.0>,
  include <tunables/global>
  profile bwrap /usr/bin/bwrap flags=(unconfined) {
    userns,
    include if exists <local/bwrap>
  }
  EOF
  sudo apparmor_parser -r /etc/apparmor.d/bwrap
"""


@functools.lru_cache(maxsize=None)
def bwrap_level():
    """'full' (mount + network isolation), 'mount' (no network namespace) or 'none'."""
    if not shutil.which('bwrap'):
        return 'none'

    def works(*extra):
        try:
            return subprocess.run(['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', *extra,
                                   '/bin/true'], capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False
    if works('--unshare-all'):
        return 'full'
    if works('--unshare-user', '--unshare-pid', '--unshare-ipc'):
        return 'mount'
    return 'none'
