from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User
from app.workers.settings import ARQ_QUEUE_INGEST


def _create_user(db_session: Session, role: Role, email: str, password: str) -> User:
    user = User(
        email=email,
        full_name=email.split("@")[0],
        password_hash=hash_password(password),
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


class FakeJob:
    job_id: str = "test-job-" + uuid4().hex[:8]


def test_async_ingest_returns_operation_job_and_reuses_client_request_id(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")

    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("async-doc.txt", BytesIO(b"Async ingestion test content."), "text/plain")},
        data={"title": "Async Doc", "description": "test", "status": "active"},
    )
    assert upload_response.status_code == 200
    document_id = upload_response.json()["document"]["id"]
    version_id = upload_response.json()["version"]["id"]

    fake_job = FakeJob()
    payload = {"version_id": version_id, "client_request_id": "ingest-request-1"}

    with patch("app.services.jobs.service.create_pool", new_callable=AsyncMock) as mock_pool:
        mock_redis = AsyncMock()
        mock_redis.enqueue_job = AsyncMock(return_value=fake_job)
        mock_redis.aclose = AsyncMock()
        mock_pool.return_value = mock_redis

        first = client.post(
            f"/api/v1/documents/{document_id}/ingest/async",
            headers={"Authorization": f"Bearer {admin_token}"},
            json=payload,
        )
        second = client.post(
            f"/api/v1/documents/{document_id}/ingest/async",
            headers={"Authorization": f"Bearer {admin_token}"},
            json=payload,
        )

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["id"] == second_payload["id"]
    assert first_payload["job_type"] == "document_ingest"
    assert first_payload["status"] == "queued"
    assert first_payload["client_request_id"] == "ingest-request-1"
    assert first_payload["resource_type"] == "document_version"
    assert first_payload["resource_id"] == version_id
    assert first_payload["arq_job_id"] == fake_job.job_id
    assert first_payload["request_payload"]["document_id"] == document_id
    assert first_payload["request_payload"]["document_version_id"] == version_id
    assert mock_redis.enqueue_job.await_count == 1
    assert mock_redis.enqueue_job.await_args.kwargs["_queue_name"] == ARQ_QUEUE_INGEST

    versions_response = client.get(
        f"/api/v1/documents/{document_id}/versions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert versions_response.status_code == 200
    assert versions_response.json()[0]["ingest_status"] == "processing"


def test_async_ingest_requires_admin(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add_all([admin_role, viewer_role])
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
    _create_user(db_session, viewer_role, "viewer@example.com", "viewer-pass")
    db_session.commit()

    viewer_token = _login(client, "viewer@example.com", "viewer-pass")
    admin_token = _login(client, "admin@example.com", "admin-pass")
    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("admin-doc.txt", BytesIO(b"Content."), "text/plain")},
        data={"title": "Admin Doc", "status": "active"},
    )
    document_id = upload_response.json()["document"]["id"]

    response = client.post(
        f"/api/v1/documents/{document_id}/ingest/async",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert response.status_code == 403


def test_async_ingest_rejects_duplicate_processing(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")

    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("dup-doc.txt", BytesIO(b"Duplicate check."), "text/plain")},
        data={"title": "Dup Doc", "status": "active"},
    )
    document_id = upload_response.json()["document"]["id"]
    version_id = upload_response.json()["version"]["id"]

    fake_job = FakeJob()

    with patch("app.services.jobs.service.create_pool", new_callable=AsyncMock) as mock_pool:
        mock_redis = AsyncMock()
        mock_redis.enqueue_job = AsyncMock(return_value=fake_job)
        mock_redis.aclose = AsyncMock()
        mock_pool.return_value = mock_redis

        first = client.post(
            f"/api/v1/documents/{document_id}/ingest/async",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"version_id": version_id},
        )
        assert first.status_code == 200

        second = client.post(
            f"/api/v1/documents/{document_id}/ingest/async",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"version_id": version_id},
        )
        assert second.status_code == 409
        assert "重复提交" in second.json()["detail"]


def test_async_ingest_restores_version_status_when_enqueue_fails(
    client: TestClient,
    db_session: Session,
) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")

    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {admin_token}"},
        files={"file": ("failed-enqueue.txt", BytesIO(b"enqueue failure"), "text/plain")},
        data={"title": "Failed Enqueue", "status": "active"},
    )
    assert upload_response.status_code == 200
    document_id = upload_response.json()["document"]["id"]
    version_id = upload_response.json()["version"]["id"]

    with patch("app.services.jobs.service.create_pool", new_callable=AsyncMock) as mock_pool:
        mock_pool.side_effect = RuntimeError("redis unavailable")
        response = client.post(
            f"/api/v1/documents/{document_id}/ingest/async",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"version_id": version_id},
        )

    assert response.status_code == 503

    versions_response = client.get(
        f"/api/v1/documents/{document_id}/versions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert versions_response.status_code == 200
    assert versions_response.json()[0]["ingest_status"] == "pending"
