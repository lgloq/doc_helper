from __future__ import annotations

from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.department import Department
from app.models.document import Document, DocumentACL
from app.models.enums import DocumentStatus, PrincipalType, RoleName
from app.models.role import Role
from app.models.user import User
from app.services.permissions.service import PermissionFilterBuilder

ORG_CODE_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _login(client: TestClient, email: str, password: str = "pass") -> str:
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def _make_department(session: Session, name: str, parent: Department | None = None) -> Department:
    if parent:
        path = f"{parent.path}/{name}"
        depth = parent.depth + 1
    else:
        path = f"/{name}"
        depth = 0
    dept_id = uuid4()
    id_path = f"{parent.id_path}/{dept_id}" if parent else f"/{dept_id}"
    if parent:
        sibling_count = session.scalar(
            select(func.count()).select_from(Department).where(Department.parent_id == parent.id)
        )
    else:
        sibling_count = session.scalar(
            select(func.count()).select_from(Department).where(Department.parent_id.is_(None))
        )
    suffix = ORG_CODE_CHARS[sibling_count or 0]
    org_code = f"{parent.org_code}{suffix}" if parent else f"A{suffix}"
    dept = Department(
        id=dept_id,
        name=name,
        parent_id=parent.id if parent else None,
        path=path,
        id_path=id_path,
        stable_code=f"S{dept_id.hex[:4].upper()}",
        org_code=org_code,
        org_code_path=f"{parent.org_code_path}/{org_code}" if parent else f"/{org_code}",
        depth=depth,
    )
    session.add(dept)
    session.flush()
    return dept


def _make_user(session: Session, role: Role, email: str, dept: Department | None = None) -> User:
    user = User(
        email=email,
        full_name=email.split("@")[0],
        password_hash=hash_password("pass"),
        is_active=True,
        role_id=role.id,
        department_id=dept.id if dept else None,
        team_name=dept.name if dept else None,
    )
    session.add(user)
    session.flush()
    return user


def test_department_ancestor_inheritance(db_session: Session) -> None:
    """用户在子部门，ACL 授权父部门 → 应可访问。"""
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    # 部门树：/tech/backend
    tech = _make_department(db_session, "tech")
    backend = _make_department(db_session, "backend", parent=tech)

    admin = _make_user(db_session, admin_role, "admin@test.com")
    user = _make_user(db_session, viewer_role, "dev@test.com", dept=backend)

    doc = Document(title="Tech Doc", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add(doc)
    db_session.flush()

    # ACL 授权给 tech 父部门
    db_session.add(
        DocumentACL(
            document_id=doc.id,
            principal_type=PrincipalType.TEAM,
            department_id=tech.id,
            team_name="tech",
            can_view=True,
            can_manage=False,
        )
    )
    db_session.commit()

    builder = PermissionFilterBuilder()
    decision = builder.get_document_decision(db_session, user, doc)
    assert decision.can_view is True
    assert decision.can_manage is False


def test_department_same_level_no_access(db_session: Session) -> None:
    """用户在 /tech/backend，ACL 授权 /tech/frontend → 应不可访问。"""
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    tech = _make_department(db_session, "tech")
    backend = _make_department(db_session, "backend", parent=tech)
    frontend = _make_department(db_session, "frontend", parent=tech)

    admin = _make_user(db_session, admin_role, "admin@test.com")
    user = _make_user(db_session, viewer_role, "dev@test.com", dept=backend)

    doc = Document(title="Frontend Doc", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add(doc)
    db_session.flush()

    db_session.add(
        DocumentACL(
            document_id=doc.id,
            principal_type=PrincipalType.TEAM,
            department_id=frontend.id,
            team_name="frontend",
            can_view=True,
            can_manage=False,
        )
    )
    db_session.commit()

    builder = PermissionFilterBuilder()
    decision = builder.get_document_decision(db_session, user, doc)
    assert decision.can_view is False


def test_department_grandparent_access(db_session: Session) -> None:
    """用户在 /a/b/c，ACL 授权 /a → 应可访问。"""
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    a = _make_department(db_session, "a")
    b = _make_department(db_session, "b", parent=a)
    c = _make_department(db_session, "c", parent=b)

    admin = _make_user(db_session, admin_role, "admin@test.com")
    user = _make_user(db_session, viewer_role, "user@test.com", dept=c)

    doc = Document(title="Root Doc", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add(doc)
    db_session.flush()

    db_session.add(
        DocumentACL(
            document_id=doc.id,
            principal_type=PrincipalType.TEAM,
            department_id=a.id,
            can_view=True,
            can_manage=True,
        )
    )
    db_session.commit()

    builder = PermissionFilterBuilder()
    decision = builder.get_document_decision(db_session, user, doc)
    assert decision.can_view is True
    assert decision.can_manage is True


def test_legacy_team_name_fallback(db_session: Session) -> None:
    """旧 team_name ACL 仍然生效（向后兼容）。"""
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    admin = _make_user(db_session, admin_role, "admin@test.com")
    # 用户只有 team_name，没有 department_id
    user = User(
        email="user@test.com",
        full_name="user",
        password_hash=hash_password("pass"),
        is_active=True,
        role_id=viewer_role.id,
        team_name="platform",
    )
    db_session.add(user)
    db_session.flush()

    doc = Document(title="Legacy Doc", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add(doc)
    db_session.flush()

    db_session.add(
        DocumentACL(
            document_id=doc.id,
            principal_type=PrincipalType.TEAM,
            team_name="platform",
            can_view=True,
            can_manage=False,
        )
    )
    db_session.commit()

    builder = PermissionFilterBuilder()
    decision = builder.get_document_decision(db_session, user, doc)
    assert decision.can_view is True


def test_no_department_user_no_team_access(db_session: Session) -> None:
    """无部门用户不应匹配任何部门 ACL。"""
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    dept = _make_department(db_session, "tech")
    admin = _make_user(db_session, admin_role, "admin@test.com")
    user = _make_user(db_session, viewer_role, "user@test.com")

    doc = Document(title="Tech Doc", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add(doc)
    db_session.flush()

    db_session.add(
        DocumentACL(
            document_id=doc.id,
            principal_type=PrincipalType.TEAM,
            department_id=dept.id,
            can_view=True,
            can_manage=False,
        )
    )
    db_session.commit()

    builder = PermissionFilterBuilder()
    decision = builder.get_document_decision(db_session, user, doc)
    assert decision.can_view is False


def test_accessible_ids_query_with_department(db_session: Session) -> None:
    """build_accessible_document_ids_query 应正确返回部门继承可见文档。"""
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    tech = _make_department(db_session, "tech")
    backend = _make_department(db_session, "backend", parent=tech)

    admin = _make_user(db_session, admin_role, "admin@test.com")
    user = _make_user(db_session, viewer_role, "dev@test.com", dept=backend)

    doc_tech = Document(title="Tech Doc", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    doc_backend = Document(title="Backend Doc", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    doc_other = Document(title="Other Doc", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add_all([doc_tech, doc_backend, doc_other])
    db_session.flush()

    db_session.add(
        DocumentACL(
            document_id=doc_tech.id,
            principal_type=PrincipalType.TEAM,
            department_id=tech.id,
            team_name="tech",
            can_view=True,
            can_manage=False,
        )
    )
    db_session.add(
        DocumentACL(
            document_id=doc_backend.id,
            principal_type=PrincipalType.TEAM,
            department_id=backend.id,
            team_name="backend",
            can_view=True,
            can_manage=True,
        )
    )
    db_session.commit()

    builder = PermissionFilterBuilder()
    visible_ids = set(
        db_session.scalars(builder.build_accessible_document_ids_query(db_session, user, require_manage=False)).all()
    )

    assert doc_tech.id in visible_ids
    assert doc_backend.id in visible_ids
    assert doc_other.id not in visible_ids

    manageable_ids = set(
        db_session.scalars(builder.build_accessible_document_ids_query(db_session, user, require_manage=True)).all()
    )

    assert doc_tech.id not in manageable_ids
    assert doc_backend.id in manageable_ids


def test_wildcard_characters_in_dept_name(db_session: Session) -> None:
    """部门名含 _ % 等 LIKE 通配符字符，不应产生越权匹配。"""
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    # 部门名含 _ 和 %
    dept_a = _make_department(db_session, "a_b")
    dept_pct = _make_department(db_session, "x%y")

    admin = _make_user(db_session, admin_role, "admin@test.com")
    user_a = _make_user(db_session, viewer_role, "user_a@test.com", dept=dept_a)

    doc = Document(title="Wildcard Doc", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add(doc)
    db_session.flush()

    # ACL 授权给 x%y 部门
    db_session.add(
        DocumentACL(
            document_id=doc.id,
            principal_type=PrincipalType.TEAM,
            department_id=dept_pct.id,
            can_view=True,
            can_manage=False,
        )
    )
    db_session.commit()

    builder = PermissionFilterBuilder()
    # a_b 部门的用户不应能访问 x%y 的文档
    decision = builder.get_document_decision(db_session, user_a, doc)
    assert decision.can_view is False


def test_path_cascade_after_rename(db_session: Session) -> None:
    """父部门改名后，子部门的 path 应级联更新。"""
    from app.repositories.department_repository import DepartmentRepository

    parent = _make_department(db_session, "old_name")
    child = _make_department(db_session, "child", parent=parent)
    grandchild = _make_department(db_session, "grandchild", parent=child)
    db_session.commit()

    assert child.path == "/old_name/child"
    assert grandchild.path == "/old_name/child/grandchild"

    # 改名
    repo = DepartmentRepository(db_session)
    old_path = parent.path
    old_id_path = parent.id_path
    old_org_code = parent.org_code
    old_org_code_path = parent.org_code_path
    parent.name = "new_name"
    parent.path = "/new_name"
    repo.update_descendant_paths(
        old_path,
        "/new_name",
        old_id_path,
        old_id_path,
        old_org_code,
        old_org_code,
        old_org_code_path,
        old_org_code_path,
        0,
    )
    db_session.commit()

    db_session.refresh(child)
    db_session.refresh(grandchild)
    assert child.path == "/new_name/child"
    assert grandchild.path == "/new_name/child/grandchild"


def test_department_duplicate_name_returns_409(client: TestClient, db_session: Session) -> None:
    """同级同名部门应返回 409。"""
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _make_user(db_session, admin_role, "admin@test.com")
    db_session.commit()

    token = _login(client, "admin@test.com")

    # 创建第一个
    r1 = client.post(
        "/api/v1/departments",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "重复部"},
    )
    assert r1.status_code == 200

    # 同名创建 → 409
    r2 = client.post(
        "/api/v1/departments",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "重复部"},
    )
    assert r2.status_code == 409


def test_acl_referenced_department_cannot_be_deleted(client: TestClient, db_session: Session) -> None:
    """被 ACL 引用的部门不能删除，应返回 409。"""
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin_user = _make_user(db_session, admin_role, "admin@test.com")
    db_session.commit()

    token = _login(client, "admin@test.com")

    # 创建部门
    dept_resp = client.post(
        "/api/v1/departments",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "被引用部"},
    )
    dept_id = UUID(dept_resp.json()["id"])

    # 创建文档并设置 ACL
    doc = Document(title="Ref Doc", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin_user.id)
    db_session.add(doc)
    db_session.flush()
    db_session.add(
        DocumentACL(
            document_id=doc.id,
            principal_type=PrincipalType.TEAM,
            department_id=dept_id,
            can_view=True,
            can_manage=False,
        )
    )
    db_session.commit()

    # 删除 → 409
    del_resp = client.delete(
        f"/api/v1/departments/{dept_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert del_resp.status_code == 409
    assert "权限引用" in del_resp.json()["detail"]
