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


def test_search_is_permission_aware_for_same_query(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, manager_role, admin_role])
    db_session.flush()

    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    manager = _create_user(db_session, manager_role, "manager@example.com", "platform", "manager-pass")
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    viewer_token = _login(client, "viewer@example.com", "viewer-pass")
    manager_token = _login(client, "manager@example.com", "manager-pass")

    team_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "Platform Runbook",
        "Platform release checklist and deployment runbook",
    )
    public_doc_id = _upload_and_ingest(
        client,
        admin_token,
        "Public Handbook",
        "Company handbook and holiday schedule",
    )

    acl_team = client.post(
        f"/api/v1/documents/{team_doc_id}/acl",
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

    query = {"query": "Platform release checklist and deployment runbook", "top_k": 3}
    manager_search = client.post("/api/v1/search", headers={"Authorization": f"Bearer {manager_token}"}, json=query)
    viewer_search = client.post("/api/v1/search", headers={"Authorization": f"Bearer {viewer_token}"}, json=query)

    assert manager_search.status_code == 200
    assert viewer_search.status_code == 200

    manager_titles = [item["document_title"] for item in manager_search.json()["matched_chunks"]]
    viewer_titles = [item["document_title"] for item in viewer_search.json()["matched_chunks"]]

    assert "Platform Runbook" in manager_titles
    assert "Platform Runbook" not in viewer_titles
    assert manager_search.json()["debug"]["accessible_document_count"] != viewer_search.json()["debug"]["accessible_document_count"]


def test_search_returns_chat_ready_scores_and_citation_metadata(client: TestClient, db_session: Session) -> None:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add(admin_role)
    db_session.flush()
    _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    document_id = _upload_and_ingest(
        client,
        admin_token,
        "FAQ Notes",
        "Service restart guide\n\nUse the maintenance window and notify stakeholders.",
    )

    search_response = client.post(
        "/api/v1/search",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"query": "Service restart guide", "top_k": 2},
    )

    assert search_response.status_code == 200
    payload = search_response.json()
    assert payload["matched_chunks"]
    first = payload["matched_chunks"][0]
    assert first["document_id"] == document_id
    assert first["score"]["fused"] >= 0
    assert first["score"]["rerank"] >= first["score"]["fused"]
    assert first["score"]["lexical_raw"] > 0
    assert first["citation_preview"]["document_title"] == "FAQ Notes"
    assert first["citation_preview"]["chunk_id"]
    assert "paragraph_start" in first
    assert payload["debug"]["pre_rerank_count"] >= payload["debug"]["post_rerank_count"] >= 1
    assert payload["debug"]["rerank_strategy"] == "heuristic-overlap"
