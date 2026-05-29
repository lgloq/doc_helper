from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps.auth import require_admin
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.user import UserCreate, UserDepartmentUpdate, UserRead, UserUpdate
from app.services.users.service import UserService

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserRead])
def list_users(
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
    q: str | None = Query(default=None),
    is_active: bool | None = Query(default=None),
) -> list[UserRead]:
    service = UserService(session)
    return service.list_users(query=q, is_active=is_active)


@router.post("", response_model=UserRead)
def create_user(
    payload: UserCreate,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> UserRead:
    service = UserService(session)
    return service.create_user(payload)


@router.put("/{user_id}", response_model=UserRead)
def update_user(
    user_id: UUID,
    payload: UserUpdate,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> UserRead:
    service = UserService(session)
    return service.update_user(user_id, payload, actor=admin)


@router.delete("/{user_id}", status_code=204)
def delete_user(
    user_id: UUID,
    admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> None:
    service = UserService(session)
    service.deactivate_user(user_id, actor=admin)


@router.patch("/{user_id}/department", response_model=UserRead)
def update_user_department(
    user_id: UUID,
    payload: UserDepartmentUpdate,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_db_session),
) -> UserRead:
    service = UserService(session)
    return service.update_user_department(user_id, payload)
