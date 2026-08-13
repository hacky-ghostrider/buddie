#!/usr/bin/env python3
"""Seed the deterministic fictional employee dataset (JSON store).

Usage:
    uv run python scripts/seed_employee_data.py
    uv run python scripts/seed_employee_data.py --force

Re-running without ``--force`` is a no-op when the file already matches the
default seed and employee count (no duplicate records).
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.employees.generator import DEFAULT_EMPLOYEE_COUNT, DEFAULT_SEED
from app.employees.store import EmployeeStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite even when seed/count already match",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_EMPLOYEE_COUNT,
        help=f"Employee count (10–50, default {DEFAULT_EMPLOYEE_COUNT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Deterministic generator seed (default {DEFAULT_SEED})",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level)
    store = EmployeeStore(settings.employee_data_path)
    dataset = store.seed(
        employee_count=args.count,
        seed=args.seed,
        force=args.force,
    )
    print(
        f"Employee store ready: path={store.path} "
        f"employees={dataset.employee_count} seed={dataset.seed} "
        f"holidays={len(dataset.holidays)} as_of={dataset.as_of_date}"
    )
    demo = next(e for e in dataset.employees if e.employee_id == "E-1101")
    print(
        f"Primary demo: {demo.employee_id} {demo.full_name} "
        f"({demo.department}) vacation={demo.leave_balance.vacation}"
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
