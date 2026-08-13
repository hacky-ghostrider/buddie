"""Employee package — structured HR data (not RAG)."""

from app.employees.generator import (
    DEFAULT_AS_OF,
    DEFAULT_EMPLOYEE_COUNT,
    DEFAULT_SEED,
    generate_employee_dataset,
)
from app.employees.service import EmployeeService, normalize_employee_id
from app.employees.store import EmployeeStore

__all__ = [
    "DEFAULT_AS_OF",
    "DEFAULT_EMPLOYEE_COUNT",
    "DEFAULT_SEED",
    "EmployeeService",
    "EmployeeStore",
    "generate_employee_dataset",
    "normalize_employee_id",
]
