from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.department import Department
from app.models.document import Document, DocumentACL
from app.models.enums import DocumentStatus, PrincipalType, RoleName
from app.models.role import Role
from app.models.user import User

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


def _setup_base(db_session: Session):
    """创建基础角色和部门层级。"""
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    db_session.add_all([admin_role, viewer_role, manager_role])
    db_session.flush()

    dept_root = _make_department(db_session, "测试总部")
    dept_tech = _make_department(db_session, "测试技术部", dept_root)
    dept_backend = _make_department(db_session, "测试后端组", dept_tech)
    dept_market = _make_department(db_session, "测试市场部", dept_root)

    return {
        "admin_role": admin_role,
        "viewer_role": viewer_role,
        "manager_role": manager_role,
        "dept_root": dept_root,
        "dept_tech": dept_tech,
        "dept_backend": dept_backend,
        "dept_market": dept_market,
    }


class TestDocumentAccessDebug:
    def test_admin_can_use_debug_endpoint(self, client: TestClient, db_session: Session) -> None:
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["viewer_role"].id,
            department_id=fixtures["dept_backend"].id,
            team_name="测试后端组",
        )
        db_session.add_all([admin, viewer])
        db_session.flush()
        doc = Document(
            title="Test Doc",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(viewer.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == str(doc.id)
        assert data["user_id"] == str(viewer.id)
        assert "can_view" in data
        assert "can_manage" in data
        assert "reason" in data
        assert "checks" in data
        assert "evaluated_user" in data
        assert "evaluated_document" in data
        assert "department_context" in data
        assert data["evaluated_user"]["email"] == "viewer@test.com"
        assert data["evaluated_user"]["department_path"] == "/测试总部/测试技术部/测试后端组"
        assert data["evaluated_document"]["title"] == "Test Doc"

    def test_viewer_cannot_use_debug_endpoint(self, client: TestClient, db_session: Session) -> None:
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["viewer_role"].id,
        )
        db_session.add_all([admin, viewer])
        db_session.flush()
        doc = Document(
            title="Test Doc",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.commit()

        token = _login(client, "viewer@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(admin.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_admin_user_hits_admin_check(self, client: TestClient, db_session: Session) -> None:
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        db_session.add(admin)
        db_session.flush()
        doc = Document(
            title="Test Doc",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(admin.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_view"] is True
        assert data["can_manage"] is True
        assert data["matched_rule"]["source"] == "admin"
        assert data["matched_rule"]["can_view"] is True
        assert data["matched_rule"]["can_manage"] is True
        admin_check = next(c for c in data["checks"] if c["source"] == "admin")
        assert admin_check["matched"] is True

    def test_owner_hits_owner_check(self, client: TestClient, db_session: Session) -> None:
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["viewer_role"].id,
        )
        db_session.add_all([admin, viewer])
        db_session.flush()
        doc = Document(
            title="Viewer Owns",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=viewer.id,
        )
        db_session.add(doc)
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(viewer.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_view"] is True
        assert data["can_manage"] is True
        assert data["matched_rule"]["source"] == "owner"
        assert data["matched_rule"]["can_view"] is True
        assert data["matched_rule"]["can_manage"] is True
        owner_check = next(c for c in data["checks"] if c["source"] == "owner")
        assert owner_check["matched"] is True

    def test_department_acl_ancestor_inheritance(self, client: TestClient, db_session: Session) -> None:
        """子部门用户继承父部门 ACL，match_type = "ancestor"。"""
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["viewer_role"].id,
            department_id=fixtures["dept_backend"].id,
            team_name="测试后端组",
        )
        db_session.add_all([admin, viewer])
        db_session.flush()
        doc = Document(
            title="Tech Doc",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.flush()
        # ACL 授权给父部门 测试技术部
        db_session.add(
            DocumentACL(
                document_id=doc.id,
                principal_type=PrincipalType.TEAM,
                department_id=fixtures["dept_tech"].id,
                can_view=True,
                can_manage=False,
            )
        )
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(viewer.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_view"] is True
        assert data["can_manage"] is False
        assert data["matched_rule"]["match_type"] == "ancestor"
        assert data["matched_rule"]["department_path"] == "/测试总部/测试技术部"
        assert "继承" in data["reason"]

    def test_sibling_department_no_access(self, client: TestClient, db_session: Session) -> None:
        """同级其他部门用户不命中，can_view = false。"""
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        market_viewer = User(
            email="market@test.com",
            full_name="Market",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["viewer_role"].id,
            department_id=fixtures["dept_market"].id,
            team_name="测试市场部",
        )
        db_session.add_all([admin, market_viewer])
        db_session.flush()
        doc = Document(
            title="Tech Only",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.flush()
        db_session.add(
            DocumentACL(
                document_id=doc.id,
                principal_type=PrincipalType.TEAM,
                department_id=fixtures["dept_tech"].id,
                can_view=True,
                can_manage=False,
            )
        )
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(market_viewer.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_view"] is False
        assert data["can_manage"] is False
        assert data["matched_rule"] is None

    def test_user_acl_hit(self, client: TestClient, db_session: Session) -> None:
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["viewer_role"].id,
        )
        db_session.add_all([admin, viewer])
        db_session.flush()
        doc = Document(
            title="User ACL Doc",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.flush()
        db_session.add(
            DocumentACL(
                document_id=doc.id,
                principal_type=PrincipalType.USER,
                user_id=viewer.id,
                can_view=True,
                can_manage=True,
            )
        )
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(viewer.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_view"] is True
        assert data["can_manage"] is True
        assert data["matched_rule"]["source"] == "user"

    def test_role_acl_hit(self, client: TestClient, db_session: Session) -> None:
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["viewer_role"].id,
        )
        db_session.add_all([admin, viewer])
        db_session.flush()
        doc = Document(
            title="Role ACL Doc",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.flush()
        db_session.add(
            DocumentACL(
                document_id=doc.id,
                principal_type=PrincipalType.ROLE,
                role_id=fixtures["viewer_role"].id,
                can_view=True,
                can_manage=False,
            )
        )
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(viewer.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_view"] is True
        assert data["can_manage"] is False
        assert data["matched_rule"]["source"] == "role"

    def test_public_acl_hit(self, client: TestClient, db_session: Session) -> None:
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["viewer_role"].id,
        )
        db_session.add_all([admin, viewer])
        db_session.flush()
        doc = Document(
            title="Public Doc",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.flush()
        db_session.add(
            DocumentACL(
                document_id=doc.id,
                principal_type=PrincipalType.PUBLIC,
                can_view=True,
                can_manage=False,
            )
        )
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(viewer.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_view"] is True
        assert data["matched_rule"]["source"] == "public"

    def test_no_effective_permission_acl_does_not_match_rule(
        self,
        client: TestClient,
        db_session: Session,
    ) -> None:
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["viewer_role"].id,
        )
        db_session.add_all([admin, viewer])
        db_session.flush()
        doc = Document(
            title="No Effective ACL Doc",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.flush()
        db_session.add(
            DocumentACL(
                document_id=doc.id,
                principal_type=PrincipalType.USER,
                user_id=viewer.id,
                can_view=False,
                can_manage=False,
            )
        )
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(viewer.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_view"] is False
        assert data["can_manage"] is False
        assert data["matched_rule"] is None
        user_check = next(c for c in data["checks"] if c["source"] == "user")
        assert user_check["matched"] is False
        assert "未授予查看或管理权限" in user_check["message"]

    def test_nonexistent_user_returns_404(self, client: TestClient, db_session: Session) -> None:
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        db_session.add(admin)
        db_session.flush()
        doc = Document(
            title="Test Doc",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.commit()

        token = _login(client, "admin@test.com")
        fake_user_id = str(uuid4())
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": fake_user_id},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404

    def test_nonexistent_document_returns_404(self, client: TestClient, db_session: Session) -> None:
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        db_session.add(admin)
        db_session.commit()

        token = _login(client, "admin@test.com")
        fake_doc_id = str(uuid4())
        resp = client.get(
            f"/api/v1/documents/{fake_doc_id}/access-debug",
            params={"user_id": str(admin.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404

    def test_department_context_in_response(self, client: TestClient, db_session: Session) -> None:
        """验证 department_context 返回完整的祖先链。"""
        fixtures = _setup_base(db_session)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["admin_role"].id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=fixtures["viewer_role"].id,
            department_id=fixtures["dept_backend"].id,
            team_name="测试后端组",
        )
        db_session.add_all([admin, viewer])
        db_session.flush()
        doc = Document(
            title="Test Doc",
            description=None,
            status=DocumentStatus.ACTIVE,
            owner_user_id=admin.id,
        )
        db_session.add(doc)
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get(
            f"/api/v1/documents/{doc.id}/access-debug",
            params={"user_id": str(viewer.id)},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        ctx = data["department_context"]
        assert ctx["user_department_path"] == "/测试总部/测试技术部/测试后端组"
        assert len(ctx["ancestor_department_ids"]) == 3
        assert "/测试总部/测试技术部/测试后端组" in ctx["ancestor_department_paths"]
        assert "/测试总部/测试技术部" in ctx["ancestor_department_paths"]
        assert "/测试总部" in ctx["ancestor_department_paths"]
