# robot_setup

Installs and removes the Python environment for the Daksha workspace:

- a [uv](https://docs.astral.sh/uv/)-managed venv at `~/daksha_venv`
  (Python 3.12, `--system-site-packages` so `rclpy` and the other ROS packages
  stay visible) with everything in [requirements.txt](requirements.txt),
  including **placo** and **damiao-motor**
- `~/activate_daksha.sh`, which sources ROS, the venv and this workspace in
  one step

## Install

On a fresh machine (before the first build), run it from the source tree:

```bash
~/robot/src/robot_setup/scripts/install.sh
cd ~/robot && colcon build
source ~/activate_daksha.sh
```

uv is installed automatically into `~/.local/bin` if it's missing (Astral's
official installer, needs `curl`). No sudo is needed.

From then on, open a terminal and run `source ~/activate_daksha.sh`. To do
that automatically in every new terminal, run the install with `--bashrc`.

`install.sh` is safe to re-run: it reuses the venv and brings it in line with
`requirements.txt`. After a build it also runs as
`ros2 run robot_setup install.sh`.

| Option          | Effect |
|-----------------|--------|
| `--venv DIR`    | Put the venv somewhere else (default: `$DAKSHA_VENV` or `~/daksha_venv`) |
| `--python VER`  | Python version (default `3.12`, ROS Jazzy's python) |
| `--recreate`    | Delete the venv and build it from scratch |
| `--bashrc`      | Add `source ~/activate_daksha.sh` to `~/.bashrc` |

## Deactivating the environment

After `source ~/activate_daksha.sh`, two shell functions are defined:

```bash
deactivate_daksha     # leave the env, restore PATH / PYTHONPATH / LD_LIBRARY_PATH
daksha_uninstall      # uninstall and deactivate in one step (see below)
```

`deactivate_daksha` is the full inverse of `activate_daksha.sh`: it calls
`deactivate`, then restores every variable the activate script changed back to
the values they had before. Use it when you want to leave the environment
without removing it.

## Uninstall

```bash
# While activate_daksha.sh is sourced:
daksha_uninstall         # remove and deactivate in one step (ask first)
daksha_uninstall --yes   # same without the prompt

# Or from a plain terminal (won't deactivate the shell it runs in):
~/robot/src/robot_setup/scripts/uninstall.sh
```

Both list what they will remove and ask for confirmation before deleting
anything. `daksha_uninstall` is a shell function defined by `activate_daksha.sh`
so it can call `deactivate_daksha` after the script finishes — a child process
cannot change the environment of its parent shell, so the bare script alone
can't deactivate.

What is removed:

- the venv (refuses to delete a directory that has no `pyvenv.cfg`)
- `~/activate_daksha.sh`, but only if `install.sh` generated it
- the `~/.bashrc` line that `--bashrc` added, with a backup at
  `~/.bashrc.daksha.bak`

The workspace, ROS and `~/.local` are not touched.

| Option         | Effect |
|----------------|--------|
| `--venv DIR`   | Venv to remove (default: `$DAKSHA_VENV` or `~/daksha_venv`) |
| `--venv-only`  | Remove only the venv; keep the activation script and `~/.bashrc` line |
| `--uv`         | Also remove uv (`~/.local/bin/uv`, `uvx`) and its cache |
| `-y`, `--yes`  | Don't ask for confirmation |

## Checking the environment

```bash
source ~/activate_daksha.sh
~/robot/src/robot_setup/scripts/verify_env.sh     # or: ros2 run robot_setup verify_env.sh
```

For **every** package in `requirements.txt`, it checks that the package is
installed, that its version matches the pin, and that its modules import
(`opencv-contrib-python` → `cv2`, `PyYAML` → `yaml`, `pin` → `pinocchio`, …).
It runs this check under both the venv's python and `/usr/bin/python3`, which
`ros2 run` nodes usually use. Each import runs in its own process, so a module
that segfaults is reported instead of stopping the check. A package that
resolves from outside the venv, such as `~/.local`, is flagged.

It also checks that `rclpy` imports, runs `uv pip check` for conflicting
dependencies, and checks the `damiao` CLI, uv and that the workspace is
sourced. `install.sh` runs it at the end; use `--venv-only` to skip the
`/usr/bin/python3` pass.

To check one requirements file on its own:

```bash
python ~/robot/src/robot_setup/scripts/check_requirements.py requirements.txt --venv "$VIRTUAL_ENV"
```

## Requirements

[requirements.txt](requirements.txt) is the single list for the venv. To add a
dependency, add it there and re-run `install.sh`. Packages in the commented
part at the bottom (ML stack, vendor SDKs, simulators) need per-machine
installs, so uncomment them only where they apply.

A few entries differ from the versions the list was generated with, because
the originals can't be installed together on Python 3.12:

| Entry          | Why |
|----------------|-----|
| `placo>=0.11`  | `collusion` uses the placo 0.11 API (coal 3). The list had `0.9.23`. |
| `pin` unpinned | `pin==3.8.0` needs cmeel-boost 1.89 and placo 0.11 needs 1.90; the resolver picks a compatible `pin` (currently 4.1.0). |
| `psutil>=5.9.0`, `matplotlib>=3.5.1` | Those old versions have no Python 3.12 wheels and would need compiling. |

## What `activate_daksha.sh` does, and why

1. Sources `/opt/ros/<distro>/setup.bash`, the venv, and
   `<workspace>/install/setup.bash`.
2. Puts the venv's site-packages **first** on `PYTHONPATH`. This makes the
   venv's packages visible to nodes started with `ros2 run` / `ros2 launch`,
   which usually run under `/usr/bin/python3` rather than the venv's python.
3. Puts placo's native libraries **first** on `LD_LIBRARY_PATH`.

Steps 2 and 3 are needed because ROS Jazzy ships its own `eigenpy`, built for
numpy 1.x, in `/opt/ros/jazzy`. placo needs numpy 2 and its own eigenpy. If
ROS's copy is found first, either as the Python module or as `libeigenpy.so`,
`import placo` fails with
`undefined symbol: EIGENPY_ARRAY_APIPyArray_RUNTIME_VERSION` or segfaults.
placo is a cmeel wheel, so its modules live under `cmeel.prefix/`, and that
directory is added explicitly. Python ignores `.pth` files for `PYTHONPATH`
entries, so nothing else would add it.

The cmeel libraries don't replace any ROS or system library at runtime: their
sonames differ (boost 1.90, assimp 6, tinyxml2 11), and the one shared library,
qhull, is the same 8.0.2 version on both sides.

Nothing in the package hardcodes a path. The scripts find the workspace from
their own location, and the ROS distro from `$ROS_DISTRO` or `/opt/ros/*`.
Those paths are written into `~/activate_daksha.sh` when it is generated.

## Files

```
robot_setup/
├── scripts/
│   ├── install.sh                # uv + venv + packages + ~/activate_daksha.sh
│   ├── uninstall.sh              # removes what install.sh created
│   ├── verify_env.sh             # environment checks
│   └── check_requirements.py     # per-package check used by verify_env.sh
├── templates/
│   └── activate_daksha.sh.in     # template for ~/activate_daksha.sh
└── requirements.txt              # everything that goes into the venv
```

## Known issues

- **Packages are also installed in `~/.local`.** numpy 2.5.4, an older placo
  and eigenpy were installed with `pip install --user`. Every Python process
  on this machine sees them, even outside the venv. ROS Jazzy's compiled
  Python modules are built against numpy 1.x, so anything that imports them
  outside `activate_daksha.sh` can crash. These scripts don't touch
  `~/.local`.
- `pytest==6.2.5` and `requests==2.25.1` are older than the apt versions
  (7.4.4 and 2.31.0). Because the venv comes first on `PYTHONPATH`,
  `colcon test` picks up the venv's pytest 6.2.5. If tests misbehave, relax
  those two pins.