from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User
from app.schemas.search import SearchDebugInfo
from app.services.diagnostics import build_eval_pipeline_diagnosis, build_trace_pipeline_diagnosis


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


def _upload_and_ingest(client: TestClient, token: str, title: str, content: str) -> None:
    upload_response = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (f"{title}.txt", BytesIO(content.encode("utf-8")), "text/plain")},
        data={"title": title, "status": "active"},
    )
    assert upload_response.status_code == 200
    payload = upload_response.json()

    ingest_response = client.post(
        f"/api/v1/documents/{payload['document']['id']}/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json={"version_id": payload["version"]["id"]},
    )
    assert ingest_response.status_code == 200


def test_eval_pipeline_diagnosis_marks_inaccessible_expected_documents(
    client: TestClient,
    db_session: Session,
) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add_all([admin_role, viewer_role])
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(client, admin_token, "Platform Runbook", "release checklist and deployment controls")

    diagnosis = build_eval_pipeline_diagnosis(
        session=db_session,
        actor=viewer,
        expected_titles=["Platform Runbook"],
        expected_outcome="answer",
        overall_pass=False,
        retrieval_breakdown={"score": 0.0, "retrieved_fact_recall": 0.0},
        citation_breakdown={"score": 0.0, "evidence_fact_recall": 0.0},
        faithfulness_breakdown={"score": 0.0},
        permission_breakdown={"passed": True, "score": 1.0},
        permission_checks={},
        retrieval_debug=SearchDebugInfo(
            accessible_document_count=0,
            lexical_candidate_count=0,
            vector_candidate_count=0,
            fusion_strategy="none",
        ),
        matched_expected_titles=[],
        missing_expected_titles=["platform runbook"],
        matched_citation_titles=[],
        missing_citation_titles=["platform runbook"],
        unsupported_answer_facts=[],
        unsupported_answer_claims=[],
        insufficient_evidence=True,
    )

    assert diagnosis["stage"] == "permission_filter"
    assert diagnosis["reason_code"] == "expected_documents_not_accessible"
    assert diagnosis["signals"]["inaccessible_expected_titles"] == ["Platform Runbook"]


def test_trace_pipeline_diagnosis_marks_unsupported_claims() -> None:
    diagnosis = build_trace_pipeline_diagnosis(
        retrieval_debug=SearchDebugInfo(
            accessible_document_count=12,
            lexical_candidate_count=8,
            vector_candidate_count=4,
            fusion_strategy="rrf",
            pre_rerank_count=10,
            post_rerank_count=3,
        ),
        selected_citation_count=2,
        evidence_audit={
            "status": "needs_review",
            "unsupported_count": 2,
        },
        error_text=None,
        insufficient_evidence=False,
    )

    assert diagnosis["stage"] == "answer_generation"
    assert diagnosis["reason_code"] == "unsupported_answer_claims"
