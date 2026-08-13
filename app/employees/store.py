"""JSON-file employee store — local deterministic HR data (not RAG)."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.employees.exceptions import EmployeeNotFoundError, EmployeeStoreError
from app.employees.generator import (
    DEFAULT_AS_OF,
    DEFAULT_EMPLOYEE_COUNT,
    DEFAULT_SEED,
    generate_employee_dataset,
)
from app.employees.models import CompanyHoliday, EmployeeDataset, EmployeeRecord

logger = logging.getLogger(__name__)

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "employees" / "employees.json"
)


class EmployeeStore:
    """Read/write the deterministic employee JSON dataset.

    Re-seeding overwrites with identical content (same seed) — no duplicates.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH
        self._lock = threading.RLock()
        self._cache: EmployeeDataset | None = None

    @property
    def path(self) -> Path:
        """Filesystem path of the dataset document."""
        return self._path

    def seed(
        self,
        *,
        employee_count: int = DEFAULT_EMPLOYEE_COUNT,
        seed: int = DEFAULT_SEED,
        force: bool = False,
    ) -> EmployeeDataset:
        """Generate and persist the dataset.

        Args:
            employee_count: Number of employees (10–50; default 30).
            seed: Deterministic generator seed.
            force: When False, skip write if file already matches seed/count.

        Returns:
            Loaded / written ``EmployeeDataset``.
        """
        dataset = generate_employee_dataset(
            employee_count=employee_count,
            seed=seed,
            as_of=DEFAULT_AS_OF,
        )
        with self._lock:
            if self._path.is_file() and not force:
                try:
                    existing = self._read_unlocked()
                    if (
                        existing.seed == dataset.seed
                        and existing.employee_count == dataset.employee_count
                        and existing.as_of_date == dataset.as_of_date
                    ):
                        logger.info(
                            "Employee dataset already seeded: path=%s count=%s",
                            self._path,
                            existing.employee_count,
                        )
                        self._cache = existing
                        return existing
                except EmployeeStoreError:
                    pass

            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = dataset.model_dump(mode="json")
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tmp.replace(self._path)
            self._cache = dataset
            logger.info(
                "Employee dataset written: path=%s count=%s seed=%s",
                self._path,
                dataset.employee_count,
                dataset.seed,
            )
            return dataset

    def ensure_seeded(self) -> EmployeeDataset:
        """Load existing dataset or seed the default 30-employee corpus."""
        with self._lock:
            if self._path.is_file():
                return self._read_unlocked()
        return self.seed()

    def load(self) -> EmployeeDataset:
        """Load dataset from disk (seeds defaults when missing)."""
        return self.ensure_seeded()

    def get_employee(self, employee_id: str) -> EmployeeRecord:
        """Fetch one employee or raise ``EmployeeNotFoundError``."""
        eid = employee_id.strip().upper()
        dataset = self.load()
        record = dataset.by_id().get(eid)
        if record is None:
            raise EmployeeNotFoundError(f"Unknown employee_id: {eid}")
        return record

    def list_employee_ids(self) -> list[str]:
        """Return sorted employee ids."""
        return sorted(emp.employee_id for emp in self.load().employees)

    def holidays(self) -> list[CompanyHoliday]:
        """Return shared company holidays."""
        return list(self.load().holidays)

    def update_employee(self, record: EmployeeRecord) -> EmployeeRecord:
        """Replace one employee record and persist the dataset."""
        with self._lock:
            dataset = self._read_unlocked() if self._path.is_file() else self.load()
            employees = list(dataset.employees)
            replaced = False
            for index, existing in enumerate(employees):
                if existing.employee_id == record.employee_id:
                    employees[index] = record
                    replaced = True
                    break
            if not replaced:
                raise EmployeeNotFoundError(
                    f"Unknown employee_id: {record.employee_id}"
                )
            updated = dataset.model_copy(update={"employees": employees})
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = updated.model_dump(mode="json")
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tmp.replace(self._path)
            self._cache = updated
            logger.info(
                "Employee record updated: employee_id=%s path=%s",
                record.employee_id,
                self._path,
            )
            return record

    def clear_cache(self) -> None:
        """Drop in-memory cache (tests)."""
        with self._lock:
            self._cache = None

    def _read_unlocked(self) -> EmployeeDataset:
        if self._cache is not None:
            return self._cache
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            dataset = EmployeeDataset.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            raise EmployeeStoreError(
                f"Failed to read employee dataset at {self._path}: {exc}"
            ) from exc
        self._cache = dataset
        return dataset


__all__ = ["EmployeeStore"]
