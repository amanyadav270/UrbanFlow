"""Check and install startup dependencies using the interpreter that runs UrbanFlow."""
from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ("requirements.txt", "requirements-rl.txt")
IMPORT_NAMES = {"scikit-learn": "sklearn"}


def required_packages():
    requirements = {}
    indexes = []
    for filename in REQUIREMENTS:
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
            entry = line.split("#", 1)[0].strip()
            if not entry:
                continue
            if entry.startswith("--extra-index-url "):
                indexes.extend(entry.split())
            else:
                name = re.match(r"[a-zA-Z0-9_-]+", entry).group()
                requirements[name] = entry
    return requirements, indexes


def check_dependencies(requirements):
    missing, broken = [], {}
    importlib.invalidate_caches()
    for name in requirements:
        try:
            version(name)
        except PackageNotFoundError:
            missing.append(name)
    if not missing:
        for name in requirements:
            try:
                importlib.import_module(IMPORT_NAMES.get(name, name))
            except Exception as error:
                broken[name] = str(error)
    return missing, broken


def install_packages(indexes, package_args):
    # Prefer the project drive for temporary wheel files, since the
    # system drive may have too little space for a CPU PyTorch wheel.
    try:
        scratch = TemporaryDirectory(prefix="urbanflow-pip-", dir=ROOT)
    except OSError:
        scratch = TemporaryDirectory(prefix="urbanflow-pip-")
    with scratch as directory:
        env = {**os.environ, "TEMP": directory, "TMP": directory, "TMPDIR": directory}
        return subprocess.call(
            [sys.executable, "-X", "utf8", "-m", "pip", "install", "--no-cache-dir",
             "--disable-pip-version-check", *indexes, *package_args],
            cwd=ROOT, env=env,
        )


def main():
    args = sys.argv[1:]
    check_only = "--check-only" in args
    install = "--install-dependencies" in args
    forwarded = [arg for arg in args if arg not in ("--check-only", "--install-dependencies")]
    print(f"Python interpreter: {sys.executable}", flush=True)
    print(f"Python version: {sys.version.split()[0]}", flush=True)
    if sys.version_info < (3, 10):
        print("UrbanFlow requires Python 3.10 or newer.")
        return 1

    requirements, indexes = required_packages()
    missing, broken = check_dependencies(requirements)
    if missing:
        print(f"Missing packages: {', '.join(missing)}", flush=True)
        if not install and not check_only and sys.stdin.isatty():
            try:
                install = input("Install the missing packages in this Python environment? Type Y to install: ").strip().lower() == "y"
            except EOFError:
                install = False
        if not install:
            print("To install and verify, run INSTALL_RL_DEPENDENCIES.bat from the same terminal.")
            return 1

        print("Installing missing packages in the interpreter shown above...", flush=True)
        package_args = [requirements[name] for name in missing]
        try:
            status = install_packages(indexes, package_args)
        except OSError as error:
            print(f"Could not start installation: {error}")
            return 1
        if status:
            print("Installation failed. See the pip error above.")
            return status
        missing, broken = check_dependencies(requirements)
        if missing:
            print(f"Installation did not make these packages available: {', '.join(missing)}")
            return 1

    if broken:
        for name, error in broken.items():
            print(f"Could not import {name}: {error}")
        print("Repair the packages above in the Python environment shown above, then rerun START_URBANFLOW.bat.")
        return 1

    try:
        from urbanflow.core.kinematic_sim import KinematicSimulator
        from urbanflow.rl.live import checkpoint_directory, load

        simulator = KinematicSimulator("classroom_grid")
        load(simulator, "PPO_NEIGHBOR")
        path = checkpoint_directory(simulator, "PPO_NEIGHBOR") / "weights.pt"
        print(f"Verified checkpoint: {path.relative_to(ROOT)}", flush=True)
    except Exception as error:
        print(f"Directional v2 checkpoint verification failed: {error}", file=sys.stderr)
        return 1

    print("Multi-Agent RL Model ready — 4 agents loaded.", flush=True)
    if check_only:
        return 0
    return subprocess.call([sys.executable, "-X", "utf8", str(ROOT / "run_demo.py"), *forwarded], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
