from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User
from app.services.ingestion.markitdown_parser import MarkItDownParser


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


def test_xlsx_upload_ingest_exposes_markitdown_table_text_in_chunks(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        MarkItDownParser,
        "_convert_local_to_markdown",
        lambda self, path: "## Approvals\n\n| Request | Approver | SLA |\n| --- | --- | --- |\n| Data export | Admin | 1 day |",
    )
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={
            "file": (
                "approvals.xlsx",
                BytesIO(b"fake xlsx payload"),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"title": "Approval Workbook", "description": "xlsx upload", "status": "active"},
    )
    assert upload_response.status_code == 200
    document_payload = upload_response.json()
    document_id = document_payload["document"]["id"]
    version_id = document_payload["version"]["id"]
    assert document_payload["version"]["mime_type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

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
    chunk_payloads = chunks_response.json()
    chunk_text = "\n".join(item["content"] for item in chunk_payloads)
    assert "Table row: Sheet: Approvals. Request=Data export; Approver=Admin; SLA=1 day." in chunk_text
    assert any(item["citation_metadata"]["parser_name"] == "markitdown:xlsx" for item in chunk_payloads)
    assert any(item["citation_metadata"].get("sheet_name") == "Approvals" for item in chunk_payloads)
    assert any(item["citation_metadata"].get("table_headers") == ["Request", "Approver", "SLA"] for item in chunk_payloads)


def test_xls_upload_ingest_exposes_real_markitdown_sheet_metadata(client: TestClient, db_session: Session) -> None:
    xlwt = pytest.importorskip("xlwt")

    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Legacy Budget")
    sheet.write_merge(0, 0, 0, 2, "Budget Summary")
    sheet.write(1, 0, "Workstream")
    sheet.write(1, 1, "Owner")
    sheet.write(1, 2, "Spend")
    sheet.write(2, 0, "Platform")
    sheet.write(2, 1, "Mei")
    sheet.write(2, 2, 125000)

    payload = BytesIO()
    workbook.save(payload)
    payload.seek(0)

    admin_token = _login(client, "admin@example.com", "admin-pass")
    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={
            "file": (
                "legacy-budget.xls",
                payload,
                "application/vnd.ms-excel",
            )
        },
        data={"title": "Legacy Budget Workbook", "description": "xls upload", "status": "active"},
    )
    assert upload_response.status_code == 200
    document_payload = upload_response.json()
    document_id = document_payload["document"]["id"]
    version_id = document_payload["version"]["id"]
    assert document_payload["version"]["mime_type"] == "application/vnd.ms-excel"

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
    chunk_payloads = chunks_response.json()
    chunk_text = "\n".join(item["content"] for item in chunk_payloads)
    assert "Table row: Sheet: Legacy Budget / Budget Summary. Workstream=Platform; Owner=Mei; Spend=125000." in chunk_text
    assert all("Unnamed:" not in item["content"] for item in chunk_payloads)
    assert any(item["citation_metadata"].get("parser_name") == "markitdown:xls" for item in chunk_payloads)
    assert any(item["citation_metadata"].get("sheet_name") == "Legacy Budget" for item in chunk_payloads)
    assert any(item["citation_metadata"].get("table_headers") == ["Workstream", "Owner", "Spend"] for item in chunk_payloads)


def test_empty_parse_ingest_fails_instead_of_ready_with_zero_chunks(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("empty.txt", BytesIO(b"   \n\n\t"), "text/plain")},
        data={"title": "Empty Extraction", "description": "empty parse", "status": "active"},
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
    assert ingest_response.status_code == 422
    assert "No searchable text chunks" in ingest_response.json()["detail"]

    versions_response = client.get(
        f"/api/v1/documents/{document_id}/versions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert versions_response.status_code == 200
    version_payload = versions_response.json()[0]
    assert version_payload["ingest_status"] == "failed"
    assert "No searchable text chunks" in version_payload["ingest_error"]


def test_exact_duplicate_upload_with_same_title_is_rejected(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    file_bytes = b"Same content for duplicate detection."

    first_upload = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("policy.txt", BytesIO(file_bytes), "text/plain")},
        data={"title": "Policy Notes", "description": "first upload", "status": "active"},
    )
    assert first_upload.status_code == 200

    duplicate_upload = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("policy-copy.txt", BytesIO(file_bytes), "text/plain")},
        data={"title": "Policy Notes", "description": "duplicate upload", "status": "active"},
    )
    assert duplicate_upload.status_code == 409
    assert "内容完全相同" in duplicate_upload.json()["detail"]

    documents_response = client.get(
        "/api/v1/documents",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert documents_response.status_code == 200
    assert sum(1 for item in documents_response.json() if item["title"] == "Policy Notes") == 1


def test_exact_duplicate_version_upload_is_rejected(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    file_bytes = b"Same version payload."

    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("policy.txt", BytesIO(file_bytes), "text/plain")},
        data={"title": "Versioned Policy", "description": "first upload", "status": "active"},
    )
    assert upload_response.status_code == 200
    document_id = upload_response.json()["document"]["id"]

    duplicate_version_upload = client.post(
        f"/api/v1/documents/{document_id}/versions/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("policy-v2.txt", BytesIO(file_bytes), "text/plain")},
    )
    assert duplicate_version_upload.status_code == 409
    assert "无需重复上传相同版本" in duplicate_version_upload.json()["detail"]

    versions_response = client.get(
        f"/api/v1/documents/{document_id}/versions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert versions_response.status_code == 200
    assert len(versions_response.json()) == 1


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
