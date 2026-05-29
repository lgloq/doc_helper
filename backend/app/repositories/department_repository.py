from __future__ import annotations

from uuid import UUID

from sqlalchemy import exists, func, select, update
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

    def get_ancestor_ids(self, path: str) -> list[UUID]:
        """查询 path 对应部门自身 + 所有祖先的 ID。

        拆分 path 为精确路径列表，用 IN 匹配，避免 LIKE 通配符注入。
        如 '/a/b/c' → ['/a/b/c', '/a/b', '/a']
        """
        ancestor_paths = [path]
        parts = path.split("/")
        # parts[0]=""（开头/）, parts[1:]=各段
        for i in range(len(parts) - 1, 1, -1):
            ancestor_paths.append("/".join(parts[:i]))

        statement = select(Department.id).where(Department.path.in_(ancestor_paths))
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

    def update_descendant_paths(self, old_path: str, new_path: str, depth_delta: int) -> int:
        r"""级联更新所有子孙部门的 path 和 depth。

        对 LIKE 模式做转义，避免 _ % \ 被当作通配符。
        """
        escaped = old_path.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like_pattern = escaped + "/%"
        stmt = (
            update(Department)
            .where(Department.path.like(like_pattern, escape="\\"))
            .values(
                path=new_path + func.substring(Department.path, len(old_path) + 1),
                depth=Department.depth + depth_delta,
            )
        )
        result = self.session.execute(stmt)
        return result.rowcount

    def find_by_name(self, name: str) -> Department | None:
        statement = select(Department).where(Department.name == name).limit(1)
        return self.session.scalar(statement)
