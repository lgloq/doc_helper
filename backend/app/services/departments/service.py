from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.department import Department
from app.repositories.department_repository import DepartmentRepository
from app.schemas.department import DepartmentCreate, DepartmentRead, DepartmentUpdate
from app.services.departments.codes import (
    DepartmentCodeSpaceExhausted,
    build_org_code_path,
    generate_child_org_code,
    generate_root_org_code,
    generate_stable_code,
)

MAX_PATH_LENGTH = 512
MAX_ID_PATH_LENGTH = 2048
MAX_ORG_CODE_PATH_LENGTH = 1024


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

        department_id = uuid4()
        path = f"{parent.path}/{name}" if parent else f"/{name}"
        id_path = f"{parent.id_path}/{department_id}" if parent else f"/{department_id}"
        stable_code = self._generate_stable_code()
        org_code = self._generate_org_code(parent)
        org_code_path = build_org_code_path(parent.org_code_path if parent else None, org_code)
        if len(path) > MAX_PATH_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"部门路径长度超过上限 ({MAX_PATH_LENGTH})，请缩短部门名称或减少层级深度",
            )
        if len(id_path) > MAX_ID_PATH_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"部门稳定路径长度超过上限 ({MAX_ID_PATH_LENGTH})，请减少层级深度",
            )
        if len(org_code_path) > MAX_ORG_CODE_PATH_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"部门编号路径长度超过上限 ({MAX_ORG_CODE_PATH_LENGTH})，请减少层级深度",
            )

        department = Department(
            id=department_id,
            name=name,
            parent_id=data.parent_id,
            path=path,
            id_path=id_path,
            stable_code=stable_code,
            org_code=org_code,
            org_code_path=org_code_path,
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
        old_id_path = department.id_path
        old_org_code = department.org_code
        old_org_code_path = department.org_code_path
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
        parent_changed = new_parent_id != department.parent_id

        new_parent = None
        if new_parent_id is not None:
            if new_parent_id == department_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不能将部门移到自身")
            new_parent = self.repository.get_by_id(new_parent_id)
            if new_parent is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="目标父部门不存在")
            # 防环：基于稳定 old_id_path 校验，不受改名影响
            if new_parent.id_path.startswith(old_id_path + "/"):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不能将部门移到其子部门下")

        # 3. 计算最终 path / id_path / org_code / depth（基于旧路径，不提前修改 department）
        new_org_code = department.org_code
        if parent_changed:
            new_org_code = self._generate_org_code(new_parent)

        if new_parent is not None:
            new_path = f"{new_parent.path}/{new_name}"
            new_id_path = f"{new_parent.id_path}/{department.id}"
            new_org_code_path = build_org_code_path(new_parent.org_code_path, new_org_code)
            new_depth = new_parent.depth + 1
        else:
            new_path = f"/{new_name}"
            new_id_path = f"/{department.id}"
            new_org_code_path = build_org_code_path(None, new_org_code)
            new_depth = 0

        if len(new_path) > MAX_PATH_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"部门路径长度超过上限 ({MAX_PATH_LENGTH})",
            )
        if len(new_id_path) > MAX_ID_PATH_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"部门稳定路径长度超过上限 ({MAX_ID_PATH_LENGTH})",
            )
        if len(new_org_code_path) > MAX_ORG_CODE_PATH_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"部门编号路径长度超过上限 ({MAX_ORG_CODE_PATH_LENGTH})",
            )

        subtree_changed = (
            old_path != new_path
            or old_id_path != new_id_path
            or old_org_code != new_org_code
            or old_org_code_path != new_org_code_path
            or old_depth != new_depth
        )

        # 4. 应用变更到 department
        department.name = new_name
        department.parent_id = new_parent_id
        department.path = new_path
        department.id_path = new_id_path
        department.org_code = new_org_code
        department.org_code_path = new_org_code_path
        department.depth = new_depth

        if subtree_changed:
            depth_delta = new_depth - old_depth
            self.repository.update_descendant_paths(
                old_path,
                new_path,
                old_id_path,
                new_id_path,
                old_org_code,
                new_org_code,
                old_org_code_path,
                new_org_code_path,
                depth_delta,
            )

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

    def _generate_stable_code(self) -> str:
        try:
            return generate_stable_code(self.repository.get_stable_codes())
        except DepartmentCodeSpaceExhausted as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from None

    def _generate_org_code(self, parent: Department | None) -> str:
        try:
            sibling_codes = self.repository.get_sibling_org_codes(parent.id if parent else None)
            if parent is None:
                return generate_root_org_code(sibling_codes)
            return generate_child_org_code(parent.org_code, sibling_codes)
        except DepartmentCodeSpaceExhausted as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from None
