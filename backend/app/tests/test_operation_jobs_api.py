from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.enums import RoleName
from app.models.operation_job import OperationJob
from app.models.role import Role
from app.models.user import User
from app.services.jobs.service import (
    JOB_TYPE_CHAT_MESSAGE,
    JOB_TYPE_DOCUMENT_DIFF_SUMMARY,
    JOB_TYPE_DOCUMENT_INGEST,
    JOB_TYPE_EVAL_RUN,
    OperationJobService,
)
from app.workers.settings import ARQ_QUEUE_CHAT, ARQ_QUEUE_DIFF, ARQ_QUEUE_EVAL, ARQ_QUEUE_INGEST
from app.workers.tasks import (
    ChatWorkerSettings,
    DiffWorkerSettings,
    EvalWorkerSettings,
    IngestWorkerSettings,
    run_ingest_document,
    run_operation_job,
)


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


def _create_session(client: TestClient, token: str) -> str:
    response = client.post(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "异步会话"},
    )
    assert response.status_code == 200
    return response.json()["id"]


class FakeJob:
    job_id: str = "job-" + uuid4().hex[:8]


def test_async_chat_message_returns_operation_job_and_reuses_client_request_id(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add(viewer_role)
    db_session.flush()
    _create_user(db_session, viewer_role, "viewer@example.com", "viewer-pass")
    db_session.commit()

    token = _login(client, "viewer@example.com", "viewer-pass")
    session_id = _create_session(client, token)
    fake_job = FakeJob()

    with patch("app.services.jobs.service.create_pool", new_callable=AsyncMock) as mock_pool:
        mock_redis = AsyncMock()
        mock_redis.enqueue_job = AsyncMock(return_value=fake_job)
        mock_redis.aclose = AsyncMock()
        mock_pool.return_value = mock_redis

        payload = {
            "content": "请总结一下这次会话。",
            "top_k": 3,
            "client_request_id": "chat-async-request-1",
        }
        first = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages/async",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        second = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages/async",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["id"] == second_payload["id"]
    assert first_payload["job_type"] == "chat_message"
    assert first_payload["status"] == "queued"
    assert first_payload["client_request_id"] == "chat-async-request-1"
    assert first_payload["resource_type"] == "chat_session"
    assert first_payload["resource_id"] == session_id
    assert first_payload["arq_job_id"] == fake_job.job_id
    assert first_payload["request_payload"]["content"] == "请总结一下这次会话。"
    assert first_payload["request_payload"]["top_k"] == 3
    assert mock_redis.enqueue_job.await_count == 1
    assert mock_redis.enqueue_job.await_args.kwargs["_queue_name"] == ARQ_QUEUE_CHAT


def test_async_chat_message_rejects_client_request_id_payload_conflict(
    client: TestClient,
    db_session: Session,
) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add(viewer_role)
    db_session.flush()
    _create_user(db_session, viewer_role, "viewer@example.com", "viewer-pass")
    db_session.commit()

    token = _login(client, "viewer@example.com", "viewer-pass")
    session_id = _create_session(client, token)
    fake_job = FakeJob()

    with patch("app.services.jobs.service.create_pool", new_callable=AsyncMock) as mock_pool:
        mock_redis = AsyncMock()
        mock_redis.enqueue_job = AsyncMock(return_value=fake_job)
        mock_redis.aclose = AsyncMock()
        mock_pool.return_value = mock_redis

        first = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages/async",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "content": "第一条后台请求",
                "top_k": 3,
                "client_request_id": "chat-async-conflict-1",
            },
        )
        second = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages/async",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "content": "不同内容不应复用旧任务",
                "top_k": 3,
                "client_request_id": "chat-async-conflict-1",
            },
        )

    assert first.status_code == 200
    assert second.status_code == 409
    assert "client_request_id" in second.json()["detail"]
    assert mock_redis.enqueue_job.await_count == 1


def test_async_chat_message_raced_existing_job_is_not_enqueued_again(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add(viewer_role)
    db_session.flush()
    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "viewer-pass")
    db_session.commit()

    token = _login(client, "viewer@example.com", "viewer-pass")
    session_id = _create_session(client, token)
    existing_job = OperationJob(
        job_type=JOB_TYPE_CHAT_MESSAGE,
        status="queued",
        user_id=viewer.id,
        client_request_id="chat-race-request-1",
        resource_type="chat_session",
        resource_id=session_id,
        request_payload={
            "session_id": session_id,
            "content": "并发请求",
            "top_k": 3,
            "client_request_id": "chat-race-request-1",
        },
    )
    db_session.add(existing_job)
    db_session.commit()
    db_session.refresh(existing_job)

    monkeypatch.setattr(OperationJobService, "_find_existing_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(OperationJobService, "_create_job", lambda *args, **kwargs: (existing_job, False))

    with patch("app.services.jobs.service.create_pool", new_callable=AsyncMock) as mock_pool:
        response = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages/async",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "content": "并发请求",
                "top_k": 3,
                "client_request_id": "chat-race-request-1",
            },
        )

    assert response.status_code == 200
    assert response.json()["id"] == str(existing_job.id)
    mock_pool.assert_not_awaited()


def test_async_chat_message_raced_existing_job_still_rejects_payload_conflict(
    client: TestClient,
    db_session: Session,
    monkeypatch,
) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add(viewer_role)
    db_session.flush()
    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "viewer-pass")
    db_session.commit()

    token = _login(client, "viewer@example.com", "viewer-pass")
    session_id = _create_session(client, token)
    existing_job = OperationJob(
        job_type=JOB_TYPE_CHAT_MESSAGE,
        status="queued",
        user_id=viewer.id,
        client_request_id="chat-race-conflict-1",
        resource_type="chat_session",
        resource_id=session_id,
        request_payload={
            "session_id": session_id,
            "content": "原始问题",
            "top_k": 3,
            "client_request_id": "chat-race-conflict-1",
        },
    )
    db_session.add(existing_job)
    db_session.commit()
    db_session.refresh(existing_job)

    monkeypatch.setattr(OperationJobService, "_find_existing_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(OperationJobService, "_create_job", lambda *args, **kwargs: (existing_job, False))

    with patch("app.services.jobs.service.create_pool", new_callable=AsyncMock) as mock_pool:
        response = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages/async",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "content": "不同问题",
                "top_k": 3,
                "client_request_id": "chat-race-conflict-1",
            },
        )

    assert response.status_code == 409
    assert "client_request_id" in response.json()["detail"]
    mock_pool.assert_not_awaited()


def test_get_operation_job_enforces_owner_scope(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add_all([admin_role, viewer_role])
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
    _create_user(db_session, viewer_role, "owner@example.com", "owner-pass")
    _create_user(db_session, viewer_role, "other@example.com", "other-pass")
    db_session.commit()

    owner_token = _login(client, "owner@example.com", "owner-pass")
    other_token = _login(client, "other@example.com", "other-pass")
    admin_token = _login(client, "admin@example.com", "admin-pass")
    session_id = _create_session(client, owner_token)
    fake_job = FakeJob()

    with patch("app.services.jobs.service.create_pool", new_callable=AsyncMock) as mock_pool:
        mock_redis = AsyncMock()
        mock_redis.enqueue_job = AsyncMock(return_value=fake_job)
        mock_redis.aclose = AsyncMock()
        mock_pool.return_value = mock_redis
        create_response = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages/async",
            headers={"Authorization": f"Bearer {owner_token}"},
            json={
                "content": "后台任务权限检查",
                "top_k": 2,
                "client_request_id": "chat-async-job-scope-1",
            },
        )

    assert create_response.status_code == 200
    job_id = create_response.json()["id"]

    owner_response = client.get(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {owner_token}"})
    admin_response = client.get(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {admin_token}"})
    other_response = client.get(f"/api/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {other_token}"})

    assert owner_response.status_code == 200
    assert admin_response.status_code == 200
    assert other_response.status_code == 404


def test_operation_job_service_routes_job_types_to_dedicated_queues() -> None:
    assert OperationJobService._queue_name_for_job_type(JOB_TYPE_CHAT_MESSAGE) == ARQ_QUEUE_CHAT
    assert OperationJobService._queue_name_for_job_type(JOB_TYPE_DOCUMENT_DIFF_SUMMARY) == ARQ_QUEUE_DIFF
    assert OperationJobService._queue_name_for_job_type(JOB_TYPE_DOCUMENT_INGEST) == ARQ_QUEUE_INGEST
    assert OperationJobService._queue_name_for_job_type(JOB_TYPE_EVAL_RUN) == ARQ_QUEUE_EVAL


def test_operation_worker_settings_listen_on_dedicated_queues() -> None:
    assert ChatWorkerSettings.queue_name == ARQ_QUEUE_CHAT
    assert ChatWorkerSettings.functions == [run_operation_job]
    assert DiffWorkerSettings.queue_name == ARQ_QUEUE_DIFF
    assert DiffWorkerSettings.functions == [run_operation_job]
    assert EvalWorkerSettings.queue_name == ARQ_QUEUE_EVAL
    assert EvalWorkerSettings.functions == [run_operation_job]
    assert IngestWorkerSettings.queue_name == ARQ_QUEUE_INGEST
    assert IngestWorkerSettings.functions == [run_operation_job, run_ingest_document]
