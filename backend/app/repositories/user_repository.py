from __future__ import annotations

from typing import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.user import User


class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_email(self, email: str) -> User | None:
        statement = select(User).options(selectinload(User.role)).where(User.email == email)
        return self.session.scalar(statement)

    def get_by_id(self, user_id: UUID) -> User | None:
        statement = select(User).options(selectinload(User.role)).where(User.id == user_id)
        return self.session.scalar(statement)

    def list_all(self) -> Sequence[User]:
        statement = select(User).options(selectinload(User.role)).order_by(User.email.asc())
        return list(self.session.scalars(statement).all())

    def add(self, user: User) -> User:
        self.session.add(user)
        return user
