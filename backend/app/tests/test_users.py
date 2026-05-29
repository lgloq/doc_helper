from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.department import Department
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User


def _login(client: TestClient, email: str, password: str = "pass") -> str:
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def _make_department(dept_id, name: str = "tech") -> Department:
    return Department(
        id=dept_id,
        name=name,
        path=f"/{name}",
        id_path=f"/{dept_id}",
        stable_code=f"S{dept_id.hex[:4].upper()}",
        org_code="A1",
        org_code_path="/A1",
        depth=0,
    )


class TestUsersAPI:
    def test_admin_list_users(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()

        dept_id = uuid4()
        dept = _make_department(dept_id)
        db_session.add(dept)
        db_session.flush()

        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
            department_id=dept.id,
            team_name="tech",
        )
        db_session.add_all([admin, viewer])
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.get("/api/v1/users", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 2
        viewer_data = next(u for u in data if u["email"] == "viewer@test.com")
        assert viewer_data["role"]["name"] == "viewer"
        assert viewer_data["department"]["path"] == "/tech"

    def test_viewer_cannot_list_users(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()

        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
        )
        db_session.add_all([admin, viewer])
        db_session.commit()

        token = _login(client, "viewer@test.com")
        resp = client.get("/api/v1/users", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

    def test_admin_set_user_department(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()

        dept_id = uuid4()
        dept = _make_department(dept_id)
        db_session.add(dept)
        db_session.flush()

        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
        )
        db_session.add_all([admin, viewer])
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.patch(
            f"/api/v1/users/{viewer.id}/department",
            headers={"Authorization": f"Bearer {token}"},
            json={"department_id": str(dept.id)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["department_id"] == str(dept.id)
        assert data["team_name"] == "tech"

    def test_admin_clear_user_department(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()

        dept_id = uuid4()
        dept = _make_department(dept_id)
        db_session.add(dept)
        db_session.flush()

        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
            department_id=dept.id,
            team_name="tech",
        )
        db_session.add_all([admin, viewer])
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.patch(
            f"/api/v1/users/{viewer.id}/department",
            headers={"Authorization": f"Bearer {token}"},
            json={"department_id": None},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["department_id"] is None
        assert data["team_name"] is None

    def test_set_nonexistent_department_returns_404(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()

        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
        )
        db_session.add_all([admin, viewer])
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.patch(
            f"/api/v1/users/{viewer.id}/department",
            headers={"Authorization": f"Bearer {token}"},
            json={"department_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404

    def test_auth_me_returns_department(self, client: TestClient, db_session: Session) -> None:
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add(viewer_role)
        db_session.flush()

        dept_id = uuid4()
        dept = _make_department(dept_id)
        db_session.add(dept)
        db_session.flush()

        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
            department_id=dept.id,
            team_name="tech",
        )
        db_session.add(viewer)
        db_session.commit()

        token = _login(client, "viewer@test.com")
        resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["department"] is not None
        assert data["department"]["path"] == "/tech"

    def test_empty_body_returns_422(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()

        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
        )
        db_session.add_all([admin, viewer])
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.patch(
            f"/api/v1/users/{viewer.id}/department",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
        assert resp.status_code == 422

    def test_admin_create_user(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()

        dept_id = uuid4()
        dept = _make_department(dept_id)
        db_session.add(dept)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        db_session.add(admin)
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.post(
            "/api/v1/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "email": "new-user@test.com",
                "full_name": "New User",
                "password": "secret123",
                "role_name": "viewer",
                "department_id": str(dept.id),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "new-user@test.com"
        assert data["role"]["name"] == "viewer"
        assert data["department_id"] == str(dept.id)
        assert data["team_name"] == "tech"
        assert data["is_active"] is True

    def test_create_duplicate_email_returns_409(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        existing = User(
            email="existing@test.com",
            full_name="Existing",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
        )
        db_session.add_all([admin, existing])
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.post(
            "/api/v1/users",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "email": "existing@test.com",
                "full_name": "Duplicate",
                "password": "secret123",
                "role_name": "viewer",
            },
        )
        assert resp.status_code == 409

    def test_admin_update_user(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        manager_role = Role(name=RoleName.MANAGER, description="Manager")
        db_session.add_all([admin_role, viewer_role, manager_role])
        db_session.flush()

        dept_id = uuid4()
        dept = _make_department(dept_id)
        db_session.add(dept)
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
        )
        db_session.add_all([admin, viewer])
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.put(
            f"/api/v1/users/{viewer.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "email": "updated@test.com",
                "full_name": "Updated User",
                "password": "newpass123",
                "role_name": "manager",
                "department_id": str(dept.id),
                "is_active": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "updated@test.com"
        assert data["full_name"] == "Updated User"
        assert data["role"]["name"] == "manager"
        assert data["department_id"] == str(dept.id)

        updated_token = _login(client, "updated@test.com", "newpass123")
        assert updated_token

    def test_admin_list_users_query_and_active_filter(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()

        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        active = User(
            email="alpha@test.com",
            full_name="Alpha User",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
        )
        inactive = User(
            email="beta@test.com",
            full_name="Beta User",
            password_hash=hash_password("pass"),
            is_active=False,
            role_id=viewer_role.id,
        )
        db_session.add_all([admin, active, inactive])
        db_session.commit()

        token = _login(client, "admin@test.com")
        query_resp = client.get("/api/v1/users?q=alpha", headers={"Authorization": f"Bearer {token}"})
        assert query_resp.status_code == 200
        assert [user["email"] for user in query_resp.json()] == ["alpha@test.com"]

        inactive_resp = client.get("/api/v1/users?is_active=false", headers={"Authorization": f"Bearer {token}"})
        assert inactive_resp.status_code == 200
        assert {user["email"] for user in inactive_resp.json()} == {"beta@test.com"}

    def test_admin_delete_user_deactivates(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        viewer = User(
            email="viewer@test.com",
            full_name="Viewer",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=viewer_role.id,
        )
        db_session.add_all([admin, viewer])
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.delete(f"/api/v1/users/{viewer.id}", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 204
        db_session.refresh(viewer)
        assert viewer.is_active is False

        login_resp = client.post("/api/v1/auth/login", json={"email": "viewer@test.com", "password": "pass"})
        assert login_resp.status_code == 401

    def test_admin_cannot_delete_self(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        admin = User(
            email="admin@test.com",
            full_name="Admin",
            password_hash=hash_password("pass"),
            is_active=True,
            role_id=admin_role.id,
        )
        db_session.add(admin)
        db_session.commit()

        token = _login(client, "admin@test.com")
        resp = client.delete(f"/api/v1/users/{admin.id}", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 400
