from __future__ import annotations

from uuid import UUID

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.models.department import Department


class DepartmentRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, department: Department) -> Department:
        self.session.add(department)
        return department

    def get_by_id(self, department_id: UUID) -> Department | None:
        return self.session.get(Department, department_id)

    def get_all(self) -> list[Department]:
        statement = select(Department).order_by(Department.depth.asc(), Department.name.asc())
        return list(self.session.scalars(statement).all())

    def get_children(self, parent_id: UUID) -> list[Department]:
        statement = select(Department).where(Department.parent_id == parent_id).order_by(Department.name.asc())
        return list(self.session.scalars(statement).all())

    def get_stable_codes(self) -> set[str]:
        statement = select(Department.stable_code)
        return {code for code in self.session.scalars(statement).all() if code}

    def get_sibling_org_codes(self, parent_id: UUID | None) -> set[str]:
        if parent_id is None:
            statement = select(Department.org_code).where(Department.parent_id.is_(None))
        else:
            statement = select(Department.org_code).where(Department.parent_id == parent_id)
        return {code for code in self.session.scalars(statement).all() if code}

    def get_ancestor_ids(self, id_path: str) -> list[UUID]:
        """查询 id_path 对应部门自身 + 所有祖先的 ID。

        拆分稳定 ID 路径为精确路径列表，用 IN 匹配，避免展示名称变化影响权限。
        如 '/id-a/id-b/id-c' → ['/id-a/id-b/id-c', '/id-a/id-b', '/id-a']
        """
        ancestor_paths = [id_path]
        parts = id_path.split("/")
        # parts[0]=""（开头/）, parts[1:]=各段
        for i in range(len(parts) - 1, 1, -1):
            ancestor_paths.append("/".join(parts[:i]))

        statement = select(Department.id).where(Department.id_path.in_(ancestor_paths))
        return list(self.session.scalars(statement).all())

    def has_children(self, department_id: UUID) -> bool:
        statement = select(exists().where(Department.parent_id == department_id))
        return bool(self.session.scalar(statement))

    def has_users(self, department_id: UUID) -> bool:
        from app.models.user import User

        statement = select(exists().where(User.department_id == department_id))
        return bool(self.session.scalar(statement))

    def has_acl_references(self, department_id: UUID) -> bool:
        from app.models.document import DocumentACL

        statement = select(exists().where(DocumentACL.department_id == department_id))
        return bool(self.session.scalar(statement))

    def update_descendant_paths(
        self,
        old_path: str,
        new_path: str,
        old_id_path: str,
        new_id_path: str,
        old_org_code: str,
        new_org_code: str,
        _old_org_code_path: str,
        new_org_code_path: str,
        depth_delta: int,
    ) -> int:
        r"""级联更新所有子孙部门的展示 path、稳定 id_path、组织编号和 depth。

        子树匹配使用稳定 id_path；UUID 路径不含 LIKE 通配符，避免中文展示路径干扰。
        """
        like_pattern = old_id_path + "/%"
        statement = (
            select(Department)
            .where(Department.id_path.like(like_pattern))
            .order_by(Department.depth.asc(), Department.path.asc())
        )
        descendants = list(self.session.scalars(statement).all())
        updated_org_paths: dict[UUID, str] = {}

        for department in descendants:
            department.path = new_path + department.path[len(old_path) :]
            department.id_path = new_id_path + department.id_path[len(old_id_path) :]
            department.org_code = new_org_code + department.org_code[len(old_org_code) :]
            parent_org_code_path = updated_org_paths.get(department.parent_id, new_org_code_path)
            department.org_code_path = f"{parent_org_code_path}/{department.org_code}"
            department.depth = department.depth + depth_delta
            updated_org_paths[department.id] = department.org_code_path

        return len(descendants)

    def find_by_name(self, name: str) -> Department | None:
        statement = select(Department).where(Department.name == name).limit(1)
        return self.session.scalar(statement)
