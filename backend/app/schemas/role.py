from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.models.enums import RoleName
from app.schemas.base import ORMModel


class RoleRead(ORMModel):
    id: UUID
    name: RoleName
    description: str | None = None
    created_at: datetime
    updated_at: datetime
