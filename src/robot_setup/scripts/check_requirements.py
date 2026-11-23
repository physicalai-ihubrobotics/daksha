#!/usr/bin/env python3
"""Check every package in requirements.txt against the current interpreter.

    check_requirements.py REQUIREMENTS_FILE [--venv DIR]

For each active (uncommented) requirement:
  1. the distribution is installed
  2. its version satisfies the specifier (==, >=, <, ...)
  3. each of its top-level modules imports - in a separate process, so a
     module that segfaults (e.g. a numpy-ABI mismatch) is reported, not fatal

With --venv, a package found outside the venv (system or ~/.local) is flagged,
since install.sh is meant to provide it.

Exit status: 0 if everything passed, 1 otherwise.
"""

import argparse
import os
import subprocess
import sys
from importlib import metadata

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

# Distributions whose top-level names aren't importable modules, or that the
# metadata maps wrongly. Value: modules to import instead (empty = skip).
IMPORT_OVERRIDES = {
    "pin": ["pinocchio"],
    "pytest": ["pytest"],
    "typing-extensions": ["typing_extensions"],
}
# Top-level names that are shipped data/helpers, not modules to import.
SKIP_MODULES = {"tests", "test", "testing", "docs", "examples", "benchmarks", "cmeel"}


def parse_requirements(path):
    reqs = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            try:
                reqs.append(Requirement(line))
            except InvalidRequirement as e:
                print(f"  {RED}FAIL{RESET}  {path}:{lineno}: cannot parse '{line}': {e}")
                reqs.append(None)
    return reqs


def find_distribution(name):
    """The installed distribution for name, matched case/underscore-insensitively."""
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        pass
    want = canonicalize_name(name)
    for dist in metadata.distributions():
        if canonicalize_name(dist.metadata.get("Name") or "") == want:
            return dist
    return None


def modules_of(dist, name):
    """Top-level importable modules a distribution provides."""
    key = canonicalize_name(name)
    if key in IMPORT_OVERRIDES:
        return IMPORT_OVERRIDES[key]
    mods = set()
    top_level = dist.read_text("top_level.txt")
    if top_level:
        mods.update(m.strip() for m in top_level.split() if m.strip())
    else:
        for f in dist.files or []:
            parts = f.parts
            if not parts or parts[0].endswith((".dist-info", ".egg-info", ".data")):
                continue
            if parts[0] in ("..", "cmeel.prefix", "bin"):
                continue
            if len(parts) == 1 and parts[0].endswith(".py"):
                mods.add(parts[0][:-3])
            elif len(parts) > 1 and parts[-1] == "__init__.py" and len(parts) == 2:
                mods.add(parts[0])
            elif len(parts) == 1 and parts[0].endswith(".so"):
                mods.add(parts[0].split(".", 1)[0])
    mods = {m for m in mods if m and not m.startswith("_") and m not in SKIP_MODULES
            and "/" not in m and "-" not in m}
    # Fall back to the distribution name itself.
    return sorted(mods) or [name.replace("-", "_").lower()]


def try_import(module):
    """(ok, message). Runs in a child process so a crash can't take us down."""
    try:
        r = subprocess.run([sys.executable, "-c", f"import {module}"],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "import timed out"
    if r.returncode == 0:
        return True, ""
    if r.returncode < 0:
        return False, f"crashed (signal {-r.returncode})"
    lines = [l for l in (r.stderr or r.stdout).strip().splitlines() if l.strip()]
    return False, lines[-1] if lines else f"exit {r.returncode}"


def in_dir(path, directory):
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(directory)]) \
            == os.path.realpath(directory)
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requirements")
    ap.add_argument("--venv", help="flag packages found outside this venv")
    args = ap.parse_args()

    failed = 0
    reqs = parse_requirements(args.requirements)
    failed += reqs.count(None)

    for req in filter(None, reqs):
        if req.marker is not None and not req.marker.evaluate():
            continue
        label = f"{req.name}{req.specifier}"
        dist = find_distribution(req.name)
        if dist is None:
            print(f"  {RED}FAIL{RESET}  {label:32} not installed")
            failed += 1
            continue

        version = dist.version
        problems = []
        if req.specifier and not req.specifier.contains(version, prereleases=True):
            problems.append(f"version {version} does not match {req.specifier}")
        for mod in modules_of(dist, req.name):
            ok, msg = try_import(mod)
            if not ok:
                problems.append(f"import {mod}: {msg}")

        where = ""
        location = str(dist.locate_file(""))
        if args.venv and not in_dir(location, args.venv):
            where = f"  {YELLOW}(from {location}, not the venv){RESET}"

        if problems:
            print(f"  {RED}FAIL{RESET}  {label:32} {version}{where}")
            for p in problems:
                print(f"          - {p}")
            failed += 1
        else:
            print(f"  {GREEN}OK{RESET}    {label:32} {version}{where}")

    total = sum(1 for r in reqs if r is not None)
    print(f"  {total - failed}/{total} requirements OK" if failed else f"  all {total} requirements OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
