from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, model_validator

from app.schemas.base import ORMModel


class DepartmentRead(ORMModel):
    id: UUID
    name: str
    parent_id: UUID | None = None
    path: str
    id_path: str
    stable_code: str
    org_code: str
    org_code_path: str
    depth: int


class DepartmentCreate(BaseModel):
    name: str
    parent_id: UUID | None = None

    @model_validator(mode="after")
    def validate_name(self):
        name = self.name.strip() if self.name else ""
        if not name:
            raise ValueError("部门名称不能为空")
        if "/" in name:
            raise ValueError("部门名称不能包含 /")
        self.name = name
        return self


class DepartmentUpdate(BaseModel):
    name: str | None = None
    parent_id: UUID | None = None  # None 表示不改；显式传 null = 移到顶层

    @model_validator(mode="after")
    def validate_name(self):
        if self.name is not None:
            name = self.name.strip()
            if not name:
                raise ValueError("部门名称不能为空")
            if "/" in name:
                raise ValueError("部门名称不能包含 /")
            self.name = name
        return self
