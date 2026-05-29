from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.department import Department
from app.repositories.department_repository import DepartmentRepository
from app.schemas.department import DepartmentCreate, DepartmentRead, DepartmentUpdate

MAX_PATH_LENGTH = 512


class DepartmentService:
    def __init__(self, session: Session):
        self.session = session
        self.repository = DepartmentRepository(session)

    def list_departments(self) -> list[DepartmentRead]:
        departments = self.repository.get_all()
        return [DepartmentRead.model_validate(d) for d in departments]

    def create_department(self, data: DepartmentCreate) -> DepartmentRead:
        name = data.name.strip()
        if not name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="部门名称不能为空")
        if "/" in name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="部门名称不能包含 /")

        parent = None
        depth = 0
        if data.parent_id is not None:
            parent = self.repository.get_by_id(data.parent_id)
            if parent is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="父部门不存在")
            depth = parent.depth + 1

        path = f"{parent.path}/{name}" if parent else f"/{name}"
        if len(path) > MAX_PATH_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"部门路径长度超过上限 ({MAX_PATH_LENGTH})，请缩短部门名称或减少层级深度",
            )

        department = Department(
            id=uuid4(),
            name=name,
            parent_id=data.parent_id,
            path=path,
            depth=depth,
        )
        self.repository.add(department)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="同级部门下已存在同名部门",
            ) from None
        self.session.refresh(department)
        return DepartmentRead.model_validate(department)

    def update_department(self, department_id: UUID, data: DepartmentUpdate) -> DepartmentRead:
        department = self.repository.get_by_id(department_id)
        if department is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="部门不存在")

        old_path = department.path
        old_depth = department.depth

        # 1. 确定最终名称
        new_name = department.name
        if data.name is not None:
            name = data.name.strip()
            if not name:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="部门名称不能为空")
            if "/" in name:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="部门名称不能包含 /")
            new_name = name

        # 2. 确定最终父级（在任何修改之前完成校验）
        new_parent_id = department.parent_id
        if "parent_id" in data.model_fields_set:
            new_parent_id = data.parent_id

        new_parent = None
        if new_parent_id is not None:
            if new_parent_id == department_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不能将部门移到自身")
            new_parent = self.repository.get_by_id(new_parent_id)
            if new_parent is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="目标父部门不存在")
            # 防环：基于 old_path 校验，不受改名影响
            if new_parent.path.startswith(old_path + "/"):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不能将部门移到其子部门下")

        # 3. 计算最终 path / depth（基于 old_path，不修改 department.path）
        if new_parent is not None:
            new_path = f"{new_parent.path}/{new_name}"
            new_depth = new_parent.depth + 1
        else:
            new_path = f"/{new_name}"
            new_depth = 0

        if len(new_path) > MAX_PATH_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"部门路径长度超过上限 ({MAX_PATH_LENGTH})",
            )

        path_changed = old_path != new_path or old_depth != new_depth

        # 4. 应用变更到 department
        department.name = new_name
        department.parent_id = new_parent_id
        department.path = new_path
        department.depth = new_depth

        if path_changed and old_path != new_path:
            depth_delta = new_depth - old_depth
            self.repository.update_descendant_paths(old_path, new_path, depth_delta)

        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="同级部门下已存在同名部门",
            ) from None
        self.session.refresh(department)
        return DepartmentRead.model_validate(department)

    def delete_department(self, department_id: UUID) -> None:
        department = self.repository.get_by_id(department_id)
        if department is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="部门不存在")

        if self.repository.has_children(department_id):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该部门下存在子部门，请先删除子部门")
        if self.repository.has_users(department_id):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该部门下存在关联用户，无法删除")
        if self.repository.has_acl_references(department_id):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该部门被文档权限引用，无法删除")

        self.session.delete(department)
        self.session.commit()
