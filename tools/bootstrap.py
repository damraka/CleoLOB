"""One-command source checkout bootstrap: python tools/bootstrap.py [--rl]."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl", action="store_true", help="also install PPO/DQN research dependencies")
    parser.add_argument("--environment", type=Path, default=Path(".venv"))
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or later is required")
    environment = args.environment.resolve()
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(environment)
    root = Path(__file__).resolve().parents[1]
    subprocess.run([str(python), "-m", "pip", "install", "-e", ".[dev,rl]" if args.rl else ".[dev]"],
                   cwd=root, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], cwd=root, check=True)
    subprocess.run([str(python), "-m", "lob.cli", "--help"], cwd=root, check=True)
    print("Environment ready. Run its python -m lob.cli smoke --out results/smoke/<new-name>.")


if __name__ == "__main__":
    main()
