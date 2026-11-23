"""PYTHONPATH for the isolated placo install, for use in launch files.

placo 0.11 (the only version with Python 3.12 wheels) requires numpy 2,
while ROS Jazzy and the system scipy are built against numpy 1.26. So placo,
coal, numpy 2 and a matching scipy live in their own --target directory and
are put on the path of the collision node only - every other process keeps
the system numpy.

    python3 -m pip install --target ~/.placo_py312 --break-system-packages \
        placo coal scipy

Imports nothing heavy, so launch files can use it without loading placo.
"""

import os
import sys

DEFAULT_DIR = os.path.expanduser(
    f"~/.placo_py{sys.version_info.major}{sys.version_info.minor}")


def placo_env(directory=DEFAULT_DIR):
    """additional_env for a launch_ros Node, or {} if not installed there
    (placo then has to be importable some other way)."""
    if not os.path.isdir(directory):
        return {}
    # cmeel wheels install modules under cmeel.prefix and register them with
    # a .pth file, which is ignored for PYTHONPATH entries - add it directly.
    cmeel = os.path.join(
        directory, "cmeel.prefix", "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    paths = [cmeel, directory, os.environ.get("PYTHONPATH", "")]
    return {"PYTHONPATH": os.pathsep.join(p for p in paths if p)}
