from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
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



def _upload_and_ingest(client: TestClient, token: str, title: str, content: str) -> None:
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



def test_observability_traces_capture_chat_roundtrip(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add_all([admin_role, viewer_role])
    db_session.flush()

    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")

    _upload_and_ingest(
        client,
        admin_token,
        "Incident Playbook",
        "Incident response checklist\n\nNotify stakeholders immediately and document every mitigation step.",
    )

    session_response = client.post(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"title": "Trace Test"},
    )
    assert session_response.status_code == 200
    session_id = session_response.json()["id"]

    message_response = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"content": "What does the incident response checklist require?", "top_k": 4},
    )
    assert message_response.status_code == 200

    list_response = client.get(
        "/api/v1/observability/traces",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_response.status_code == 200
    traces = list_response.json()
    assert len(traces) == 1

    trace = traces[0]
    assert trace["trace_type"] == "chat_qa"
    assert trace["query_text"] == "What does the incident response checklist require?"
    assert trace["retrieved_chunks_json"]
    assert trace["selected_citations_json"]
    assert trace["created_at"]

    detail_response = client.get(
        f"/api/v1/observability/traces/{trace['id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert detail_response.status_code == 200
    assert detail_response.json()["trace_metadata"]["confidence"] in {"high", "medium", "low", "insufficient"}

    forbidden_detail = client.get(
        f"/api/v1/observability/traces/{trace['id']}",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert forbidden_detail.status_code == 404
