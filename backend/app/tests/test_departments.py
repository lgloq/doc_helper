from __future__ import annotations

import re
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User


def _create_user(
    db_session: Session, role: Role, email: str, password: str, department_id=None, team_name=None
) -> User:
    user = User(
        email=email,
        full_name=email.split("@")[0],
        password_hash=hash_password(password),
        is_active=True,
        role_id=role.id,
        department_id=department_id,
        team_name=team_name,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _login(client: TestClient, email: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


def _assert_stable_code(value: str) -> None:
    assert re.fullmatch(r"[A-Z][0-9A-Z]{4}", value)


class TestDepartmentCRUD:
    def test_create_department(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")
        response = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "技术部"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "技术部"
        assert data["path"] == "/技术部"
        assert data["id_path"] == f"/{data['id']}"
        _assert_stable_code(data["stable_code"])
        assert re.fullmatch(r"[A-Z][0-9A-Z]", data["org_code"])
        assert data["org_code_path"] == f"/{data['org_code']}"
        assert data["depth"] == 0
        assert data["parent_id"] is None

    def test_create_sub_department(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")

        # 创建父部门
        parent_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "技术部"},
        )
        parent_id = parent_resp.json()["id"]
        parent_id_path = parent_resp.json()["id_path"]
        parent_org_code = parent_resp.json()["org_code"]
        parent_org_code_path = parent_resp.json()["org_code_path"]

        # 创建子部门
        child_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "后端组", "parent_id": parent_id},
        )
        assert child_resp.status_code == 200
        data = child_resp.json()
        assert data["name"] == "后端组"
        assert data["path"] == "/技术部/后端组"
        assert data["id_path"] == f"{parent_id_path}/{data['id']}"
        _assert_stable_code(data["stable_code"])
        assert len(data["org_code"]) == len(parent_org_code) + 1
        assert data["org_code"].startswith(parent_org_code)
        assert data["org_code_path"] == f"{parent_org_code_path}/{data['org_code']}"
        assert data["depth"] == 1
        assert data["parent_id"] == parent_id

    def test_sibling_org_code_space_limit_returns_409(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")
        parent_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "技术部"},
        )
        parent_id = parent_resp.json()["id"]

        for index in range(36):
            response = client.post(
                "/api/v1/departments",
                headers={"Authorization": f"Bearer {token}"},
                json={"name": f"子部门{index}", "parent_id": parent_id},
            )
            assert response.status_code == 200

        overflow_response = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "额外部门", "parent_id": parent_id},
        )
        assert overflow_response.status_code == 409
        assert "同级部门数量" in overflow_response.json()["detail"]

    def test_list_departments(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")

        client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "技术部"},
        )
        client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "市场部"},
        )

        response = client.get(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        names = {d["name"] for d in response.json()}
        assert "技术部" in names
        assert "市场部" in names

    def test_update_department_name(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")

        create_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "技术部"},
        )
        dept_id = create_resp.json()["id"]
        old_id_path = create_resp.json()["id_path"]
        old_stable_code = create_resp.json()["stable_code"]
        old_org_code = create_resp.json()["org_code"]
        old_org_code_path = create_resp.json()["org_code_path"]

        update_resp = client.put(
            f"/api/v1/departments/{dept_id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "研发部"},
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["name"] == "研发部"
        assert update_resp.json()["path"] == "/研发部"
        assert update_resp.json()["id_path"] == old_id_path
        assert update_resp.json()["stable_code"] == old_stable_code
        assert update_resp.json()["org_code"] == old_org_code
        assert update_resp.json()["org_code_path"] == old_org_code_path

    def test_delete_department(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")

        create_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "临时部"},
        )
        dept_id = create_resp.json()["id"]

        delete_resp = client.delete(
            f"/api/v1/departments/{dept_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert delete_resp.status_code == 204

    def test_delete_department_with_children_fails(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")

        parent_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "技术部"},
        )
        parent_id = parent_resp.json()["id"]

        client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "后端组", "parent_id": parent_id},
        )

        delete_resp = client.delete(
            f"/api/v1/departments/{parent_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert delete_resp.status_code == 409
        assert "子部门" in delete_resp.json()["detail"]

    def test_delete_department_with_users_fails(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()

        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")

        dept_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "测试部"},
        )
        dept_id = UUID(dept_resp.json()["id"])

        # 给一个用户关联此部门
        _create_user(db_session, viewer_role, "viewer@example.com", "pass", department_id=dept_id)
        db_session.commit()

        delete_resp = client.delete(
            f"/api/v1/departments/{dept_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert delete_resp.status_code == 409
        assert "关联用户" in delete_resp.json()["detail"]

    def test_move_department_cannot_be_descendant(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")

        parent_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "技术部"},
        )
        parent_id = parent_resp.json()["id"]

        child_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "后端组", "parent_id": parent_id},
        )
        child_id = child_resp.json()["id"]

        # 尝试把父部门移到子部门下 → 应该失败
        update_resp = client.put(
            f"/api/v1/departments/{parent_id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"parent_id": child_id},
        )
        assert update_resp.status_code == 400

    def test_department_name_rejects_slash(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")

        response = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "tech/infra"},
        )
        assert response.status_code == 422

    def test_viewer_cannot_create_department(self, client: TestClient, db_session: Session) -> None:
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
        db_session.add_all([admin_role, viewer_role])
        db_session.flush()
        _create_user(db_session, viewer_role, "viewer@example.com", "viewer-pass")
        db_session.commit()

        token = _login(client, "viewer@example.com", "viewer-pass")

        response = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "测试部"},
        )
        assert response.status_code == 403

    def test_simultaneous_rename_and_move(self, client: TestClient, db_session: Session) -> None:
        """同时改名并移动父级，防环应基于旧 path。"""
        admin_role = Role(name=RoleName.ADMIN, description="Admin")
        db_session.add(admin_role)
        db_session.flush()
        _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
        db_session.commit()

        token = _login(client, "admin@example.com", "admin-pass")

        # 创建 /root → /root/child → /root/child/grandchild
        root_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "root"},
        )
        root_id = root_resp.json()["id"]
        child_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "child", "parent_id": root_id},
        )
        child_id = child_resp.json()["id"]
        old_child_id_path = child_resp.json()["id_path"]
        old_child_stable_code = child_resp.json()["stable_code"]
        old_child_org_code = child_resp.json()["org_code"]
        old_child_org_code_path = child_resp.json()["org_code_path"]
        gc_resp = client.post(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "grandchild", "parent_id": child_id},
        )
        gc_id = gc_resp.json()["id"]

        # 把 child 改名并移到 root 同级（合法操作）
        update_resp = client.put(
            f"/api/v1/departments/{child_id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": "renamed", "parent_id": None},
        )
        assert update_resp.status_code == 200
        data = update_resp.json()
        assert data["name"] == "renamed"
        assert data["path"] == "/renamed"
        assert data["id_path"] == f"/{child_id}"
        assert data["id_path"] != old_child_id_path
        assert data["stable_code"] == old_child_stable_code
        assert len(data["org_code"]) == 2
        assert data["org_code"] != old_child_org_code
        assert data["org_code_path"] == f"/{data['org_code']}"
        assert data["org_code_path"] != old_child_org_code_path
        assert data["parent_id"] is None

        # grandchild 的 path 应级联更新
        gc_check = client.get(
            "/api/v1/departments",
            headers={"Authorization": f"Bearer {token}"},
        )
        gc_data = next(d for d in gc_check.json() if d["id"] == gc_id)
        assert gc_data["path"] == "/renamed/grandchild"
        assert gc_data["id_path"] == f"/{child_id}/{gc_id}"
        assert gc_data["org_code"].startswith(data["org_code"])
        assert gc_data["org_code_path"] == f"{data['org_code_path']}/{gc_data['org_code']}"


def test_acl_invalid_department_returns_404(client: TestClient, db_session: Session) -> None:
    """ACL 写入时无效 department_id 应返回 404。"""
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
    db_session.commit()

    token = _login(client, "admin@example.com", "admin-pass")

    # 上传文档
    from io import BytesIO

    upload_resp = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test.txt", BytesIO(b"hello"), "text/plain")},
        data={"title": "ACL Test Doc", "status": "active"},
    )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["document"]["id"]

    # 用不存在的 department_id 写 ACL → 404
    acl_resp = client.post(
        f"/api/v1/documents/{doc_id}/acl",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "principal_type": "team",
            "department_id": "00000000-0000-0000-0000-000000000000",
            "can_view": True,
            "can_manage": False,
        },
    )
    assert acl_resp.status_code == 404
    assert "department" in acl_resp.json()["detail"].lower()
