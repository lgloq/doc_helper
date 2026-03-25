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



def test_chat_roundtrip_persists_history_and_citations(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(
        client,
        admin_token,
        "Incident Playbook",
        "Incident response checklist\n\nNotify stakeholders immediately and document every mitigation step.",
    )

    create_session_response = client.post(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"title": "New Chat"},
    )
    assert create_session_response.status_code == 200
    session_id = create_session_response.json()["id"]

    message_response = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"content": "What does the incident response checklist require?", "top_k": 4},
    )

    assert message_response.status_code == 200
    payload = message_response.json()
    assert payload["assistant_message"]["role"] == "assistant"
    assert payload["assistant_message"]["insufficient_evidence"] is False
    assert payload["citations"]
    first_citation = payload["citations"][0]
    assert first_citation["document_title"] == "Incident Playbook"
    assert first_citation["version_number"] == 1
    assert first_citation["chunk_id"]
    assert first_citation["preview"]
    assert payload["retrieval_debug"]["accessible_document_count"] == 1

    list_sessions_response = client.get(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_sessions_response.status_code == 200
    assert len(list_sessions_response.json()) == 1

    detail_response = client.get(
        f"/api/v1/chat/sessions/{session_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert len(detail_payload["messages"]) == 2
    assert detail_payload["messages"][1]["citations"]



def test_chat_returns_insufficient_evidence_when_retrieval_is_empty(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(
        client,
        admin_token,
        "Holiday Notes",
        "Office holiday calendar and public leave process.",
    )

    session_response = client.post(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"title": "New Chat"},
    )
    session_id = session_response.json()["id"]

    message_response = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"content": "What is the production database failover SLA?", "top_k": 4},
    )
    assert message_response.status_code == 200
    payload = message_response.json()
    assert payload["assistant_message"]["insufficient_evidence"] is True
    assert payload["assistant_message"]["confidence"] == "insufficient"
    assert payload["citations"] == []



def test_chat_inherits_permission_aware_retrieval(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, manager_role, admin_role])
    db_session.flush()

    _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    _create_user(db_session, manager_role, "manager@example.com", "platform", "manager-pass")
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")
    manager_token = _login(client, "manager@example.com", "manager-pass")

    document_id = _upload_and_ingest(
        client,
        admin_token,
        "Platform Runbook",
        "Platform release checklist and deployment runbook for the platform team.",
    )

    acl_response = client.post(
        f"/api/v1/documents/{document_id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"principal_type": "team", "team_name": "platform", "can_view": True, "can_manage": False},
    )
    assert acl_response.status_code == 200

    manager_session = client.post(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {manager_token}"},
        json={"title": "New Chat"},
    ).json()["id"]
    viewer_session = client.post(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {viewer_token}"},
        json={"title": "New Chat"},
    ).json()["id"]

    query_payload = {"content": "What does the platform release checklist say?", "top_k": 4}
    manager_chat = client.post(
        f"/api/v1/chat/sessions/{manager_session}/messages",
        headers={"Authorization": f"Bearer {manager_token}"},
        json=query_payload,
    )
    viewer_chat = client.post(
        f"/api/v1/chat/sessions/{viewer_session}/messages",
        headers={"Authorization": f"Bearer {viewer_token}"},
        json=query_payload,
    )

    assert manager_chat.status_code == 200
    assert viewer_chat.status_code == 200
    assert manager_chat.json()["citations"]
    assert manager_chat.json()["assistant_message"]["insufficient_evidence"] is False
    assert viewer_chat.json()["assistant_message"]["insufficient_evidence"] is True
    assert viewer_chat.json()["citations"] == []
