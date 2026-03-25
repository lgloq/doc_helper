from __future__ import annotations

from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.role import Role
from app.models.enums import RoleName


class RoleRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_name(self, name: RoleName) -> Role | None:
        statement = select(Role).where(Role.name == name)
        return self.session.scalar(statement)

    def list_all(self) -> Sequence[Role]:
        statement = select(Role).order_by(Role.name.asc())
        return list(self.session.scalars(statement).all())

    def get_by_id(self, role_id: UUID) -> Role | None:
        statement = select(Role).where(Role.id == role_id)
        return self.session.scalar(statement)
