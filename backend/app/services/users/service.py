from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.department import Department
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, UserDepartmentUpdate, UserRead, UserUpdate


class UserService:
    def __init__(self, session: Session):
        self.session = session
        self.user_repository = UserRepository(session)

    def list_users(
        self, query: str | None = None, is_active: bool | None = None, limit: int | None = None
    ) -> list[UserRead]:
        users = self.user_repository.list_all(query=query, is_active=is_active, limit=limit)
        return [UserRead.model_validate(u) for u in users]

    def create_user(self, data: UserCreate) -> UserRead:
        role = self._get_role(data.role_name)
        department, team_name = self._resolve_department(data.department_id)
        user = User(
            email=data.email,
            full_name=data.full_name,
            password_hash=hash_password(data.password),
            role_id=role.id,
            department_id=department.id if department else None,
            team_name=team_name,
            is_active=data.is_active,
        )
        self.user_repository.add(user)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该邮箱已存在。") from None

        self.session.expire_all()
        created = self.user_repository.get_by_id(user.id)
        return UserRead.model_validate(created)

    def update_user(self, user_id: UUID, data: UserUpdate, actor: User) -> UserRead:
        user = self.user_repository.get_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在。")

        next_role = user.role
        next_is_active = user.is_active
        if data.role_name is not None:
            next_role = self._get_role(data.role_name)
        if data.is_active is not None:
            next_is_active = data.is_active

        if user.id == actor.id and (next_role is None or next_role.name != RoleName.ADMIN or not next_is_active):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不能停用或移除自己的管理员权限。")
        self._ensure_not_removing_last_admin(user, next_role.name if next_role else None, next_is_active)

        if data.email is not None:
            user.email = data.email
        if data.full_name is not None:
            user.full_name = data.full_name
        if data.password is not None:
            user.password_hash = hash_password(data.password)
        if data.role_name is not None and next_role is not None:
            user.role_id = next_role.id
        if "department_id" in data.model_fields_set:
            department, team_name = self._resolve_department(data.department_id)
            user.department_id = department.id if department else None
            user.team_name = team_name
        if data.is_active is not None:
            user.is_active = data.is_active

        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该邮箱已存在。") from None

        self.session.expire_all()
        updated = self.user_repository.get_by_id(user_id)
        return UserRead.model_validate(updated)

    def deactivate_user(self, user_id: UUID, actor: User) -> None:
        user = self.user_repository.get_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在。")
        if user.id == actor.id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不能停用当前登录用户。")
        self._ensure_not_removing_last_admin(user, user.role.name if user.role else None, False)
        user.is_active = False
        self.session.commit()

    def update_user_department(self, user_id: UUID, data: UserDepartmentUpdate) -> UserRead:
        user = self.user_repository.get_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在。")

        department, team_name = self._resolve_department(data.department_id)
        user.department_id = department.id if department else None
        user.team_name = team_name

        self.session.commit()
        # re-query with eager load to ensure relationships are populated
        self.session.expire_all()
        updated = self.user_repository.get_by_id(user_id)
        return UserRead.model_validate(updated)

    def _get_role(self, role_name: RoleName) -> Role:
        role = self.session.scalar(select(Role).where(Role.name == role_name))
        if role is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="角色不存在。")
        return role

    def _resolve_department(self, department_id: UUID | None) -> tuple[Department | None, str | None]:
        if department_id is None:
            return None, None
        department = self.session.get(Department, department_id)
        if department is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="部门不存在。")
        return department, department.name

    def _ensure_not_removing_last_admin(
        self, user: User, next_role_name: RoleName | None, next_is_active: bool
    ) -> None:
        if user.role is None or user.role.name != RoleName.ADMIN:
            return
        if next_role_name == RoleName.ADMIN and next_is_active:
            return

        active_admin_count = self.session.scalar(
            select(func.count())
            .select_from(User)
            .join(Role, User.role_id == Role.id)
            .where(Role.name == RoleName.ADMIN, User.is_active.is_(True))
        )
        if active_admin_count == 1:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="至少需要保留一个启用的管理员。")
