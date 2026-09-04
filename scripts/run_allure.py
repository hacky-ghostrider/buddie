#!/usr/bin/env python3
"""Generate, view, or archive Buddie Allure reports (Windows-friendly).

Examples:
    uv run python scripts/run_allure.py
    uv run python scripts/run_allure.py --serve
    uv run python scripts/run_allure.py --archive
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
ALLURE_LATEST = _ROOT / "data" / "reports" / "allure" / "latest"
ALLURE_HISTORY = _ROOT / "data" / "reports" / "allure" / "history"
ALLURE_TEST = _ROOT / "tests" / "test_buddie_eval_allure.py"


def _allure_cli() -> str | None:
    """Resolve Allure CLI on Windows (.cmd shims need a full path for subprocess)."""
    for name in ("allure", "allure.cmd"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _allure_install_hint() -> str:
    return (
        "Allure CLI not found on PATH.\n"
        "Install (pick one):\n"
        "  npm install -g allure-commandline\n"
        "  scoop install allure\n"
        "  choco install allure\n"
        "Docs: https://allurereport.org/docs/install/"
    )


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=_ROOT)


def generate() -> int:
    ALLURE_LATEST.mkdir(parents=True, exist_ok=True)
    return _run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ALLURE_TEST),
            f"--alluredir={ALLURE_LATEST.as_posix()}",
        ]
    )


def serve() -> int:
    if not ALLURE_LATEST.is_dir() or not any(ALLURE_LATEST.iterdir()):
        print(f"No Allure results in {ALLURE_LATEST}. Run generate first.")
        return 1
    allure = _allure_cli()
    if not allure:
        print(_allure_install_hint())
        return 1
    return _run([allure, "serve", str(ALLURE_LATEST)])


def generate_html() -> int:
    if not ALLURE_LATEST.is_dir() or not any(ALLURE_LATEST.iterdir()):
        print(f"No Allure results in {ALLURE_LATEST}. Run generate first.")
        return 1
    allure = _allure_cli()
    if not allure:
        print(_allure_install_hint())
        return 1
    out_dir = ALLURE_LATEST / "html"
    code = _run([allure, "generate", str(ALLURE_LATEST), "-o", str(out_dir), "--clean"])
    if code == 0:
        index = out_dir / "index.html"
        print(f"Static report: {index}")
        print("Open index.html in your browser if `allure serve` is unavailable.")
    return code


def archive() -> int:
    if not ALLURE_LATEST.is_dir() or not any(ALLURE_LATEST.iterdir()):
        print(f"Nothing to archive: {ALLURE_LATEST} is empty.")
        return 1
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    dest = ALLURE_HISTORY / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for item in ALLURE_LATEST.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    print(f"Archived to {dest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Buddie Allure report helper")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Open Allure UI for data/reports/allure/latest (requires Allure CLI)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Build static HTML under data/reports/allure/latest/html/",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Copy latest results to data/reports/allure/history/<timestamp>/",
    )
    args = parser.parse_args()

    if args.serve:
        return serve()
    if args.html:
        return generate_html()
    if args.archive:
        return archive()
    return generate()


if __name__ == "__main__":
    raise SystemExit(main())
