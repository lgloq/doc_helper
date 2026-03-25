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



def _create_chat_session(client: TestClient, token: str) -> str:
    response = client.post(
        "/api/v1/chat/sessions",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "New Chat"},
    )
    assert response.status_code == 200
    return response.json()["id"]



def _ask_question(client: TestClient, token: str, session_id: str, content: str) -> dict:
    response = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
        json={"content": content, "top_k": 4},
    )
    assert response.status_code == 200
    return response.json()



def test_task_extraction_persists_structured_items(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(
        client,
        token,
        "Incident Checklist",
        "Notify stakeholders immediately. Document every mitigation step. Prepare the incident report before closeout.",
    )
    session_id = _create_chat_session(client, token)
    _ask_question(client, token, session_id, "What does the incident checklist require?")

    extract_response = client.post(
        "/api/v1/tasks/extract",
        headers={"Authorization": f"Bearer {token}"},
        json={"session_id": session_id, "max_items": 5},
    )
    assert extract_response.status_code == 200
    payload = extract_response.json()
    assert payload["items"]
    first_item = payload["items"][0]
    assert first_item["source_message_id"]
    assert first_item["priority"] in {"high", "medium", "low"}
    assert first_item["source_citations"] is not None

    list_response = client.get(
        "/api/v1/tasks",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_response.status_code == 200
    assert len(list_response.json()) >= 1



def test_weekly_report_generation_persists_structured_sections(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(
        client,
        token,
        "Release Guidance",
        "Review the release checklist. Verify rollback readiness. Notify stakeholders about deployment timing.",
    )
    session_id = _create_chat_session(client, token)
    _ask_question(client, token, session_id, "What should we do before the release?")
    _ask_question(client, token, session_id, "What is the production database failover SLA?")

    report_response = client.post(
        "/api/v1/reports/weekly",
        headers={"Authorization": f"Bearer {token}"},
        json={"session_id": session_id},
    )
    assert report_response.status_code == 200
    report = report_response.json()["report"]
    assert report["completed_this_week"]
    assert report["risks_blockers"]
    assert report["next_week_plan"]
    assert report["reference_sources"]
    assert report["source_message_ids"]

    list_response = client.get(
        "/api/v1/reports",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1



def test_faq_generation_persists_grounded_entries(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    token = _login(client, "admin@example.com", "admin-pass")
    _upload_and_ingest(
        client,
        token,
        "Access Policy",
        "Employees should review access policy changes monthly and verify manager approval for privileged access.",
    )
    session_id = _create_chat_session(client, token)
    chat_payload = _ask_question(client, token, session_id, "How should employees handle privileged access requests?")
    assert chat_payload["citations"]

    faq_response = client.post(
        "/api/v1/faqs/generate",
        headers={"Authorization": f"Bearer {token}"},
        json={"session_id": session_id, "max_entries": 3},
    )
    assert faq_response.status_code == 200
    payload = faq_response.json()
    assert payload["entries"]
    first_entry = payload["entries"][0]
    assert first_entry["question"]
    assert first_entry["answer"]
    assert first_entry["source_citations"]

    list_response = client.get(
        "/api/v1/faqs",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_response.status_code == 200
    assert len(list_response.json()) >= 1
