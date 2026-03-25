from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.eval import EvalCase
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User


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



def _upload_and_ingest(client: TestClient, token: str, title: str, content: str) -> str:
    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (f"{title}.txt", BytesIO(content.encode("utf-8")), "text/plain")},
        data={"title": title, "status": "active"},
    )
    assert upload_response.status_code == 200
    payload = upload_response.json()
    document_id = payload["document"]["id"]
    version_id = payload["version"]["id"]

    ingest_response = client.post(
        f"/api/v1/documents/{document_id}/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json={"version_id": version_id},
    )
    assert ingest_response.status_code == 200
    return document_id



def test_eval_run_reports_metrics_and_permission_isolation(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, manager_role, admin_role])
    db_session.flush()

    _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    _create_user(db_session, manager_role, "manager@example.com", "platform", "manager-pass")
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    db_session.add_all(
        [
            EvalCase(
                dataset_name="demo_permission_eval",
                case_name="manager_can_find_platform_runbook",
                acting_user_email="manager@example.com",
                question="What does the platform release checklist require?",
                expected_document_titles=["Platform Runbook"],
                forbidden_document_titles=[],
                expected_answer_keywords=["release", "checklist"],
            ),
            EvalCase(
                dataset_name="demo_permission_eval",
                case_name="viewer_cannot_see_platform_runbook",
                acting_user_email="viewer@example.com",
                question="What does the platform release checklist require?",
                expected_document_titles=[],
                forbidden_document_titles=["Platform Runbook"],
                expected_answer_keywords=[],
            ),
            EvalCase(
                dataset_name="demo_permission_eval",
                case_name="viewer_can_find_public_handbook",
                acting_user_email="viewer@example.com",
                question="What does the company handbook say about holiday schedule?",
                expected_document_titles=["Public Handbook"],
                forbidden_document_titles=["Platform Runbook"],
                expected_answer_keywords=["holiday", "schedule"],
            ),
        ]
    )
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    platform_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "Platform Runbook",
        "Platform release checklist and deployment runbook for the platform team.",
    )
    public_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "Public Handbook",
        "Company handbook and holiday schedule for all employees.",
    )

    acl_team = client.post(
        f"/api/v1/documents/{platform_doc_id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"principal_type": "team", "team_name": "platform", "can_view": True, "can_manage": False},
    )
    assert acl_team.status_code == 200

    acl_public = client.post(
        f"/api/v1/documents/{public_doc_id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"principal_type": "public", "can_view": True, "can_manage": False},
    )
    assert acl_public.status_code == 200

    run_response = client.post(
        "/api/v1/eval/run",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"dataset_name": "demo_permission_eval", "top_k": 4, "seed_demo_cases": False},
    )
    assert run_response.status_code == 200
    payload = run_response.json()

    assert payload["status"] == "completed"
    assert payload["summary_json"]["total_cases"] == 3
    assert len(payload["results"]) == 3
    assert payload["summary_json"]["permission_isolation_pass_rate"] == 1.0

    viewer_forbidden_case = next(
        item for item in payload["results"] if item["details_json"]["case_name"] == "viewer_cannot_see_platform_runbook"
    )
    assert viewer_forbidden_case["permission_isolation_correct"] is True
    assert viewer_forbidden_case["details_json"]["permission_checks"]["forbidden_in_retrieval"] == []
    assert viewer_forbidden_case["details_json"]["permission_checks"]["forbidden_in_citations"] == []
    assert viewer_forbidden_case["details_json"]["permission_checks"]["forbidden_in_answer"] == []
    assert viewer_forbidden_case["details_json"]["trace_id"] is not None

    get_run_response = client.get(
        f"/api/v1/eval/runs/{payload['id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_run_response.status_code == 200
    assert len(get_run_response.json()["results"]) == 3

    list_runs_response = client.get(
        "/api/v1/eval/runs",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_runs_response.status_code == 200
    assert len(list_runs_response.json()) == 1
