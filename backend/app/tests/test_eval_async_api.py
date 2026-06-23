from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.eval import EvalCase, EvalResult, EvalRun
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User
from app.schemas.eval import EvalRunRequest
from app.services.eval.service import EvalService


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
    job_id: str = "eval-job-" + uuid4().hex[:8]


def _seed_admin_and_case(db_session: Session) -> User:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    admin = _create_user(db_session, admin_role, "admin@example.com", "admin-pass")
    db_session.add(
        EvalCase(
            dataset_name="async_eval_dataset",
            case_name="single_async_case",
            acting_user_email="admin@example.com",
            question="What does the handbook require?",
            expected_document_titles=["Public Handbook"],
            forbidden_document_titles=[],
            expected_answer_keywords=["holiday"],
        )
    )
    db_session.commit()
    return admin


def test_async_eval_returns_queued_run_and_reuses_client_request_id(client: TestClient, db_session: Session) -> None:
    _seed_admin_and_case(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    fake_job = FakeJob()

    with patch("app.services.eval.async_service.create_pool", new_callable=AsyncMock) as mock_pool:
        mock_redis = AsyncMock()
        mock_redis.enqueue_job = AsyncMock(return_value=fake_job)
        mock_redis.aclose = AsyncMock()
        mock_pool.return_value = mock_redis

        payload = {
            "dataset_name": "async_eval_dataset",
            "top_k": 4,
            "seed_demo_cases": False,
            "client_request_id": "eval-async-request-1",
        }
        first = client.post("/api/v1/eval/run/async", headers={"Authorization": f"Bearer {admin_token}"}, json=payload)
        second = client.post("/api/v1/eval/run/async", headers={"Authorization": f"Bearer {admin_token}"}, json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["id"] == second_payload["id"]
    assert first_payload["status"] == "queued"
    assert first_payload["results"] == []
    assert first_payload["summary_json"]["client_request_id"] == "eval-async-request-1"
    assert first_payload["summary_json"]["client_request_status"] == "queued"
    assert first_payload["summary_json"]["requested_top_k"] == 4
    assert len(first_payload["summary_json"]["selected_case_ids"]) == 1
    assert first_payload["summary_json"]["job_id"] == fake_job.job_id
    assert mock_redis.enqueue_job.await_count == 1

    runs = db_session.query(EvalRun).filter(EvalRun.dataset_name == "async_eval_dataset").all()
    assert len(runs) == 1


def test_execute_queued_eval_run_completes_selected_cases(db_session: Session, monkeypatch) -> None:
    admin = _seed_admin_and_case(db_session)
    service = EvalService(db_session)
    run_detail, should_enqueue = service.create_queued_run(
        admin,
        EvalRunRequest(
            dataset_name="async_eval_dataset",
            top_k=3,
            seed_demo_cases=False,
            client_request_id="eval-async-execute-1",
        ),
    )
    assert should_enqueue is True
    assert run_detail.status == "queued"

    def _fake_evaluate_case(self, run, case, top_k):
        assert top_k == 3
        return EvalResult(
            run_id=run.id,
            case_id=case.id,
            acting_user_email=case.acting_user_email,
            retrieval_hit_rate=1.0,
            citation_accuracy=1.0,
            answer_faithfulness=1.0,
            permission_isolation_correct=True,
            overall_pass=True,
            details_json={
                "case_name": case.case_name,
                "case_annotations": {"expected_outcome": "answer"},
                "metric_breakdown": {
                    "retrieval": {"score": 1.0},
                    "citation": {"score": 1.0},
                    "faithfulness": {"score": 1.0},
                    "permission_isolation": {"score": 1.0, "passed": True},
                    "overall": {"score": 1.0},
                },
            },
        )

    monkeypatch.setattr(EvalService, "_evaluate_case", _fake_evaluate_case)
    completed = service.execute_queued_run(run_detail.id)

    assert completed.status == "completed"
    assert completed.summary_json["client_request_status"] == "completed"
    assert completed.summary_json["client_request_id"] == "eval-async-execute-1"
    assert completed.summary_json["requested_top_k"] == 3
    assert len(completed.results) == 1
    assert completed.results[0].overall_pass is True

def test_async_eval_enqueue_failure_marks_run_failed(client: TestClient, db_session: Session) -> None:
    _seed_admin_and_case(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")

    with patch("app.services.eval.async_service.create_pool", new_callable=AsyncMock) as mock_pool:
        mock_pool.side_effect = RuntimeError("redis unavailable")
        response = client.post(
            "/api/v1/eval/run/async",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={
                "dataset_name": "async_eval_dataset",
                "top_k": 4,
                "seed_demo_cases": False,
                "client_request_id": "eval-async-enqueue-failed",
            },
        )

    assert response.status_code == 503
    runs = db_session.query(EvalRun).filter(EvalRun.dataset_name == "async_eval_dataset").all()
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].finished_at is not None
    assert runs[0].summary_json["client_request_status"] == "failed"
    assert runs[0].error_text == "异步评测任务提交失败，请稍后重试。"


def test_list_eval_runs_marks_stale_queued_run_as_failed(client: TestClient, db_session: Session) -> None:
    _seed_admin_and_case(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    stale_run = EvalRun(
        dataset_name="async_eval_dataset",
        status="queued",
        total_cases=1,
        created_at=datetime.now(UTC) - timedelta(minutes=10),
        summary_json={"client_request_id": "eval-stale-queued", "client_request_status": "queued"},
    )
    db_session.add(stale_run)
    db_session.commit()

    response = client.get("/api/v1/eval/runs", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == 200
    refreshed_run = db_session.get(EvalRun, stale_run.id)
    assert refreshed_run is not None
    assert refreshed_run.status == "failed"
    assert refreshed_run.finished_at is not None
    assert refreshed_run.summary_json["client_request_status"] == "failed"


def test_list_eval_runs_keeps_running_run_with_recent_progress(client: TestClient, db_session: Session) -> None:
    _seed_admin_and_case(db_session)
    admin_token = _login(client, "admin@example.com", "admin-pass")
    running_run = EvalRun(
        dataset_name="async_eval_dataset",
        status="running",
        total_cases=5,
        started_at=datetime.now(UTC) - timedelta(hours=1),
        summary_json={
            "client_request_id": "eval-running-with-progress",
            "client_request_status": "running",
            "completed_cases": 3,
            "total_cases": 5,
            "last_progress_at": datetime.now(UTC).isoformat(),
        },
    )
    db_session.add(running_run)
    db_session.commit()

    response = client.get("/api/v1/eval/runs", headers={"Authorization": f"Bearer {admin_token}"})

    assert response.status_code == 200
    refreshed_run = db_session.get(EvalRun, running_run.id)
    assert refreshed_run is not None
    assert refreshed_run.status == "running"
    assert refreshed_run.finished_at is None
    assert refreshed_run.summary_json["client_request_status"] == "running"