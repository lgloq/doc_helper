from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps.auth import get_current_user, require_admin
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.department import DepartmentCreate, DepartmentRead, DepartmentUpdate
from app.services.departments.service import DepartmentService

router = APIRouter(prefix="/departments", tags=["departments"])


@router.get("", response_model=list[DepartmentRead])
def list_departments(
    _current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> list[DepartmentRead]:
    service = DepartmentService(session)
    return service.list_departments()


@router.post("", response_model=DepartmentRead)
def create_department(
    payload: DepartmentCreate,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> DepartmentRead:
    service = DepartmentService(session)
    return service.create_department(payload)


@router.put("/{department_id}", response_model=DepartmentRead)
def update_department(
    department_id: UUID,
    payload: DepartmentUpdate,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> DepartmentRead:
    service = DepartmentService(session)
    return service.update_department(department_id, payload)


@router.delete("/{department_id}", status_code=204)
def delete_department(
    department_id: UUID,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> None:
    service = DepartmentService(session)
    service.delete_department(department_id)
