from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Department(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "departments"
    __table_args__ = (UniqueConstraint("parent_id", "org_code", name="uq_departments_parent_org_code"),)

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("departments.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    id_path: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    stable_code: Mapped[str] = mapped_column(String(5), nullable=False, unique=True)
    org_code: Mapped[str] = mapped_column(String(64), nullable=False)
    org_code_path: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    parent = relationship("Department", remote_side="Department.id", back_populates="children")
    children = relationship("Department", back_populates="parent", cascade="save-update, merge")
    users = relationship("User", back_populates="department")
