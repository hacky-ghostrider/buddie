"""Employee verification + structured HR HTTP endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_employee_service
from app.employees.exceptions import EmployeeVerificationError
from app.employees.models import VerifyEmployeeRequest, VerifyEmployeeResponse
from app.employees.service import EmployeeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/employees", tags=["employees"])


@router.post(
    "/verify",
    response_model=VerifyEmployeeResponse,
    summary="Verify an employee id against the structured employee store",
)
def verify_employee(
    request: VerifyEmployeeRequest,
    service: EmployeeService = Depends(get_employee_service),
) -> VerifyEmployeeResponse:
    """Identity check before protected employee tools may run."""
    try:
        return service.verify_employee(request.employee_id)
    except EmployeeVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected employee verify failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Employee verification failed unexpectedly",
        ) from exc
