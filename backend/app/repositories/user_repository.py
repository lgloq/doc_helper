from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from app.models.department import Department
from app.models.user import User

_user_loader_options = [selectinload(User.role), selectinload(User.department)]


class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_email(self, email: str) -> User | None:
        statement = select(User).options(*_user_loader_options).where(User.email == email)
        return self.session.scalar(statement)

    def get_by_id(self, user_id: UUID) -> User | None:
        statement = select(User).options(*_user_loader_options).where(User.id == user_id)
        return self.session.scalar(statement)

    def list_all(
        self,
        query: str | None = None,
        is_active: bool | None = None,
        limit: int | None = None,
    ) -> Sequence[User]:
        statement = select(User).options(*_user_loader_options).order_by(User.email.asc())
        if query:
            pattern = f"%{query.strip()}%"
            statement = statement.outerjoin(Department, User.department_id == Department.id)
            statement = statement.where(
                or_(
                    User.email.ilike(pattern),
                    User.full_name.ilike(pattern),
                    User.team_name.ilike(pattern),
                    Department.path.ilike(pattern),
                    Department.org_code_path.ilike(pattern),
                )
            )
        if is_active is not None:
            statement = statement.where(User.is_active.is_(is_active))
        if limit is not None:
            statement = statement.limit(limit)
        return list(self.session.scalars(statement).all())

    def add(self, user: User) -> User:
        self.session.add(user)
        return user
