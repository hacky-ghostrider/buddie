"""Employee domain exceptions — structured HR data layer failures."""

from __future__ import annotations


class EmployeeError(Exception):
    """Base class for employee-layer failures."""


class EmployeeNotFoundError(EmployeeError):
    """Requested employee id does not exist in the store."""


class EmployeeNotVerifiedError(EmployeeError):
    """Protected employee tool called without a verified session context."""


class EmployeeVerificationError(EmployeeError):
    """Employee id format or identity check failed."""


class EmployeeStoreError(EmployeeError):
    """Employee JSON store could not be read or written."""


class EmployeeValidationError(EmployeeError):
    """Invalid tool arguments or business-rule violation for employee ops."""


class LeaveRequestNotConfirmedError(EmployeeError):
    """Write tool refused because explicit user confirmation was missing."""


__all__ = [
    "EmployeeError",
    "EmployeeNotFoundError",
    "EmployeeNotVerifiedError",
    "EmployeeVerificationError",
    "EmployeeStoreError",
    "EmployeeValidationError",
    "LeaveRequestNotConfirmedError",
]
