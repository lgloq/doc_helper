from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import hash_password
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User



def _create_user(db_session: Session, role: Role, email: str, password: str) -> User:
    user = User(
        email=email,
        full_name=email.split("@")[0],
        password_hash=hash_password(password),
        team_name=None,
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



def _upload_document(client: TestClient, token: str, title: str, content: str) -> dict:
    response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (f"{title}.txt", BytesIO(content.encode("utf-8")), "text/plain")},
        data={"title": title, "status": "active"},
    )
    assert response.status_code == 200
    return response.json()



def _upload_document_version(client: TestClient, token: str, document_id: str, filename: str, content: str) -> dict:
    response = client.post(
        f"/api/v1/documents/{document_id}/versions/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, BytesIO(content.encode("utf-8")), "text/plain")},
    )
    assert response.status_code == 200
    return response.json()



def _ingest_version(client: TestClient, token: str, document_id: str, version_id: str) -> dict:
    response = client.post(
        f"/api/v1/documents/{document_id}/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json={"version_id": version_id},
    )
    assert response.status_code == 200
    return response.json()



def test_document_versions_and_diff_flow(client: TestClient, db_session: Session, monkeypatch) -> None:
    monkeypatch.setenv("DIFF_SUMMARY_PROVIDER", "deterministic")
    get_settings.cache_clear()

    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
    db_session.commit()

    token = _login(client, "admin@example.com", "admin-pass")

    cache_store: dict[str, str] = {}

    class FakeRedis:
        def get(self, key: str):
            return cache_store.get(key)

        def setex(self, key: str, ttl: int, value: str):
            cache_store[key] = value

    monkeypatch.setattr("app.services.diff.service.get_redis_client", lambda: FakeRedis())

    v1_content = (
        "Privileged access requests require manager approval.\n\n"
        "Release changes must be communicated one day before deployment.\n\n"
        "Legacy contact: call office hotline."
    )
    v1_upload = _upload_document(client, token, "Access Policy", v1_content)
    document_id = v1_upload["document"]["id"]
    version1_id = v1_upload["version"]["id"]
    _ingest_version(client, token, document_id, version1_id)

    document_before = client.get(
        f"/api/v1/documents/{document_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert document_before.status_code == 200
    assert document_before.json()["current_version_id"] == version1_id

    v2_content = (
        "Privileged access requests require security and manager approval.\n\n"
        "Release changes must be communicated two days before deployment.\n\n"
        "Rollback steps must be reviewed before production deployment."
    )
    v2_upload = _upload_document_version(client, token, document_id, "access_policy_v2.txt", v2_content)
    version2_id = v2_upload["version"]["id"]
    assert v2_upload["version"]["version_number"] == 2

    document_after_upload = client.get(
        f"/api/v1/documents/{document_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert document_after_upload.status_code == 200
    assert document_after_upload.json()["current_version_id"] == version1_id

    _ingest_version(client, token, document_id, version2_id)

    versions_response = client.get(
        f"/api/v1/documents/{document_id}/versions",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert versions_response.status_code == 200
    versions = versions_response.json()
    assert [item["version_number"] for item in versions] == [2, 1]

    version_detail = client.get(
        f"/api/v1/documents/{document_id}/versions/{version2_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert version_detail.status_code == 200
    assert version_detail.json()["version_number"] == 2
    assert version_detail.json()["is_current"] is True

    diff_response = client.get(
        f"/api/v1/documents/{document_id}/diff",
        headers={"Authorization": f"Bearer {token}"},
        params={"from_version": version1_id, "to_version": version2_id},
    )
    assert diff_response.status_code == 200
    diff_payload = diff_response.json()
    assert diff_payload["modified_count"] >= 1
    assert diff_payload["added_count"] + diff_payload["deleted_count"] + diff_payload["modified_count"] >= 1
    assert diff_payload["changes"]
    assert "security and manager approval" in diff_payload["unified_diff"]
    assert diff_payload["impact_hints"]

    summary_response = client.post(
        f"/api/v1/documents/{document_id}/diff/summary",
        headers={"Authorization": f"Bearer {token}"},
        json={"from_version_id": version1_id, "to_version_id": version2_id},
    )
    assert summary_response.status_code == 200
    summary_payload = summary_response.json()
    assert summary_payload["summary"]
    assert summary_payload["summary_provider"] in {"deterministic", "deterministic_fallback", "openai-compatible"}
    assert summary_payload["cache_hit"] is False
    assert summary_payload["modifications"]
    assert summary_payload["impact_hints"]

    summary_response_cached = client.post(
        f"/api/v1/documents/{document_id}/diff/summary",
        headers={"Authorization": f"Bearer {token}"},
        json={"from_version_id": version1_id, "to_version_id": version2_id},
    )
    assert summary_response_cached.status_code == 200
    cached_payload = summary_response_cached.json()
    assert cached_payload["cache_hit"] is True
    assert cached_payload["summary"] == summary_payload["summary"]

    summary_response_refreshed = client.post(
        f"/api/v1/documents/{document_id}/diff/summary",
        headers={"Authorization": f"Bearer {token}"},
        json={"from_version_id": version1_id, "to_version_id": version2_id, "force_refresh": True},
    )
    assert summary_response_refreshed.status_code == 200
    refreshed_payload = summary_response_refreshed.json()
    assert refreshed_payload["cache_hit"] is False
    assert refreshed_payload["summary"]
