from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.schemas.base import ORMModel
from app.schemas.role import RoleRead


class UserRead(ORMModel):
    id: UUID
    email: str
    full_name: str
    team_name: str | None = None
    is_active: bool
    role: RoleRead | None = None
    created_at: datetime
    updated_at: datetime
