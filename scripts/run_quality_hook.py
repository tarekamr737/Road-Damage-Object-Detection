from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

COMMANDS = {
    "ruff-check": ["-m", "ruff", "check", "app", "src", "scripts", "tests"],
    "ruff-format-check": [
        "-m",
        "ruff",
        "format",
        "--check",
        "app",
        "src",
        "scripts",
        "tests",
    ],
    "synthetic-inference-smoke": [
        "-m",
        "pytest",
        "tests/smoke/test_inference_service_synthetic.py",
    ],
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_python(root: Path) -> Path:
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run project quality hooks through the local virtual environment."
    )
    parser.add_argument("hook", choices=sorted(COMMANDS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    python = project_python(root)
    command = [str(python), *COMMANDS[args.hook]]
    completed = subprocess.run(command, cwd=root, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
