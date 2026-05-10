from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User


FORBIDDEN_DETAIL = "当前角色无权执行该操作。"


def _create_user(db_session: Session, role: Role, email: str, team_name: str | None, password: str) -> User:
    user = User(
        email=email,
        full_name=email.split("@")[0],
        password_hash=hash_password(password),
        team_name=team_name,
        is_active=True,
        role_id=role.id,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _login(client: TestClient, email: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


def test_upload_ingest_and_chunk_visibility(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add_all([admin_role, viewer_role])
    db_session.flush()

    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")

    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("policy.txt", BytesIO(b"First paragraph.\n\nSecond paragraph for citation."), "text/plain")},
        data={"title": "Policy Notes", "description": "upload", "status": "active"},
    )
    assert upload_response.status_code == 200
    document_payload = upload_response.json()
    document_id = document_payload["document"]["id"]
    version_id = document_payload["version"]["id"]

    ingest_response = client.post(
        f"/api/v1/documents/{document_id}/ingest",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"version_id": version_id},
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunk_count"] >= 1

    versions_response = client.get(
        f"/api/v1/documents/{document_id}/versions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert versions_response.status_code == 200
    assert versions_response.json()[0]["ingest_status"] == "ready"

    forbidden_chunks = client.get(
        f"/api/v1/documents/{document_id}/chunks",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert forbidden_chunks.status_code == 404

    acl_response = client.post(
        f"/api/v1/documents/{document_id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"principal_type": "user", "user_id": str(viewer.id), "can_view": True, "can_manage": False},
    )
    assert acl_response.status_code == 200

    chunks_response = client.get(
        f"/api/v1/documents/{document_id}/chunks",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert chunks_response.status_code == 200
    chunk_payload = chunks_response.json()[0]
    assert chunk_payload["document_version_id"] == version_id
    assert chunk_payload["paragraph_start"] == 1
    assert chunk_payload["preview"]


def test_csv_upload_ingest_exposes_table_text_in_chunks(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")

    csv_bytes = "Request,Approver,SLA\nData export,Admin,1 day\nRefund,Manager,2 days\n".encode("utf-8")
    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("approvals.csv", BytesIO(csv_bytes), "text/csv")},
        data={"title": "Approval Matrix", "description": "csv upload", "status": "active"},
    )
    assert upload_response.status_code == 200
    document_payload = upload_response.json()
    document_id = document_payload["document"]["id"]
    version_id = document_payload["version"]["id"]
    assert document_payload["version"]["mime_type"] == "text/csv"

    ingest_response = client.post(
        f"/api/v1/documents/{document_id}/ingest",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"version_id": version_id},
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunk_count"] >= 1

    chunks_response = client.get(
        f"/api/v1/documents/{document_id}/chunks",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert chunks_response.status_code == 200
    chunk_text = "\n".join(item["content"] for item in chunks_response.json())
    assert "Table row: approvals. Request=Data export; Approver=Admin; SLA=1 day." in chunk_text
    assert "Request=Refund; Approver=Manager; SLA=2 days." in chunk_text


def test_document_management_write_endpoints_require_admin_role(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add_all([admin_role, manager_role, viewer_role])
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    manager = _create_user(db_session, manager_role, "manager@example.com", "platform", "manager-pass")
    _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    manager_token = _login(client, "manager@example.com", "manager-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")

    viewer_upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {viewer_token}"},
        files={"file": ("viewer-notes.txt", BytesIO(b"Viewer upload should be forbidden."), "text/plain")},
        data={"title": "Viewer Upload", "description": "forbidden", "status": "active"},
    )
    assert viewer_upload_response.status_code == 403
    assert viewer_upload_response.json()["detail"] == FORBIDDEN_DETAIL

    admin_upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("admin-notes.txt", BytesIO(b"Admin upload succeeds."), "text/plain")},
        data={"title": "Admin Managed Doc", "description": "allowed", "status": "active"},
    )
    assert admin_upload_response.status_code == 200
    document_payload = admin_upload_response.json()
    document_id = document_payload["document"]["id"]
    version_id = document_payload["version"]["id"]

    manager_acl_response = client.post(
        f"/api/v1/documents/{document_id}/acl",
        headers={"Authorization": f"Bearer {manager_token}"},
        json={"principal_type": "user", "user_id": str(manager.id), "can_view": True, "can_manage": False},
    )
    assert manager_acl_response.status_code == 403
    assert manager_acl_response.json()["detail"] == FORBIDDEN_DETAIL

    manager_ingest_response = client.post(
        f"/api/v1/documents/{document_id}/ingest",
        headers={"Authorization": f"Bearer {manager_token}"},
        json={"version_id": version_id},
    )
    assert manager_ingest_response.status_code == 403
    assert manager_ingest_response.json()["detail"] == FORBIDDEN_DETAIL

    manager_version_upload_response = client.post(
        f"/api/v1/documents/{document_id}/versions/upload",
        headers={"Authorization": f"Bearer {manager_token}"},
        files={"file": ("manager-version.txt", BytesIO(b"Manager version upload should be forbidden."), "text/plain")},
    )
    assert manager_version_upload_response.status_code == 403
    assert manager_version_upload_response.json()["detail"] == FORBIDDEN_DETAIL

    admin_ingest_response = client.post(
        f"/api/v1/documents/{document_id}/ingest",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"version_id": version_id},
    )
    assert admin_ingest_response.status_code == 200
    assert admin_ingest_response.json()["chunk_count"] >= 1
