from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, model_validator

from app.models.enums import RoleName
from app.schemas.base import ORMModel
from app.schemas.department import DepartmentRead
from app.schemas.role import RoleRead


class UserDepartmentUpdate(BaseModel):
    department_id: UUID | None


class UserCreate(BaseModel):
    email: str
    full_name: str
    password: str
    role_name: RoleName
    department_id: UUID | None = None
    is_active: bool = True

    @model_validator(mode="after")
    def validate_fields(self):
        self.email = self.email.strip().lower()
        self.full_name = self.full_name.strip()
        if not self.email or "@" not in self.email:
            raise ValueError("邮箱格式不正确")
        if not self.full_name:
            raise ValueError("姓名不能为空")
        if len(self.password) < 6:
            raise ValueError("密码至少 6 位")
        return self


class UserUpdate(BaseModel):
    email: str | None = None
    full_name: str | None = None
    password: str | None = None
    role_name: RoleName | None = None
    department_id: UUID | None = None
    is_active: bool | None = None

    @model_validator(mode="after")
    def validate_fields(self):
        if self.email is not None:
            self.email = self.email.strip().lower()
            if not self.email or "@" not in self.email:
                raise ValueError("邮箱格式不正确")
        if self.full_name is not None:
            self.full_name = self.full_name.strip()
            if not self.full_name:
                raise ValueError("姓名不能为空")
        if self.password is not None and len(self.password) < 6:
            raise ValueError("密码至少 6 位")
        return self


class UserRead(ORMModel):
    id: UUID
    email: str
    full_name: str
    team_name: str | None = None
    department_id: UUID | None = None
    department: DepartmentRead | None = None
    is_active: bool
    role: RoleRead | None = None
    created_at: datetime
    updated_at: datetime
