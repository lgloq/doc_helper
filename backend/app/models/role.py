from __future__ import annotations

from sqlalchemy import Enum, String
from sqlalchemy.orm import Mapped, relationship, mapped_column

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import RoleName


class Role(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "roles"

    name: Mapped[RoleName] = mapped_column(
        Enum(RoleName, native_enum=False),
        unique=True,
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)

    users = relationship("User", back_populates="role")
    acl_entries = relationship("DocumentACL", back_populates="role")
