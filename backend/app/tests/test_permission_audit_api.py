from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.document import Document, DocumentACL
from app.models.enums import DocumentStatus, PrincipalType, RoleName
from app.models.observability import TraceLog
from app.models.role import Role
from app.models.user import User


def _login(client: TestClient, email: str, password: str = "pass") -> str:
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def _setup_users(session: Session) -> dict[str, object]:
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    session.add_all([admin_role, viewer_role])
    session.flush()

    admin = User(
        email="admin@test.com",
        full_name="Admin",
        password_hash=hash_password("pass"),
        is_active=True,
        role_id=admin_role.id,
    )
    viewer = User(
        email="viewer@test.com",
        full_name="Viewer",
        password_hash=hash_password("pass"),
        is_active=True,
        role_id=viewer_role.id,
    )
    session.add_all([admin, viewer])
    session.flush()
    return {
        "admin_role": admin_role,
        "viewer_role": viewer_role,
        "admin": admin,
        "viewer": viewer,
    }


def _document(session: Session, owner: User, title: str) -> Document:
    document = Document(
        title=title,
        description=None,
        status=DocumentStatus.ACTIVE,
        owner_user_id=owner.id,
    )
    session.add(document)
    session.flush()
    return document


def test_admin_can_inspect_user_visible_scope(client: TestClient, db_session: Session) -> None:
    fixtures = _setup_users(db_session)
    admin = fixtures["admin"]
    viewer = fixtures["viewer"]
    public_doc = _document(db_session, admin, "公开制度")
    role_doc = _document(db_session, admin, "普通员工手册")
    owned_doc = _document(db_session, viewer, "个人维护文档")
    db_session.add_all(
        [
            DocumentACL(
                document_id=public_doc.id,
                principal_type=PrincipalType.PUBLIC,
                can_view=True,
                can_manage=False,
            ),
            DocumentACL(
                document_id=role_doc.id,
                principal_type=PrincipalType.ROLE,
                role_id=fixtures["viewer_role"].id,
                can_view=True,
                can_manage=True,
            ),
        ]
    )
    db_session.commit()

    token = _login(client, "admin@test.com")
    resp = client.get(
        f"/api/v1/permissions/users/{viewer.id}/scope",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["evaluated_user"]["email"] == "viewer@test.com"
    assert data["visible_document_count"] == 3
    assert data["manageable_document_count"] == 2
    assert data["permission_summary"]["owner_count"] == 1
    assert data["permission_summary"]["public_acl_count"] == 1
    assert data["permission_summary"]["role_acl_count"] == 1
    assert {item["id"] for item in data["visible_documents"]} == {
        str(public_doc.id),
        str(role_doc.id),
        str(owned_doc.id),
    }


def test_non_admin_cannot_inspect_permission_scope(client: TestClient, db_session: Session) -> None:
    fixtures = _setup_users(db_session)
    token = _login(client, "viewer@test.com")

    resp = client.get(
        f"/api/v1/permissions/users/{fixtures['admin'].id}/scope",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 403


def test_acl_impact_reports_newly_visible_users(client: TestClient, db_session: Session) -> None:
    fixtures = _setup_users(db_session)
    admin = fixtures["admin"]
    viewer = fixtures["viewer"]
    document = _document(db_session, admin, "即将公开的制度")
    db_session.commit()

    token = _login(client, "admin@test.com")
    resp = client.post(
        f"/api/v1/permissions/documents/{document.id}/acl/impact",
        json={
            "principal_type": "public",
            "can_view": True,
            "can_manage": False,
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["operation"] == "create"
    assert data["affected_user_count"] == 1
    assert data["newly_visible_user_count"] == 1
    assert data["users_preview"][0]["id"] == str(viewer.id)
    assert data["users_preview"][0]["impact"] == "newly_visible"


def test_permission_probe_search_records_audit_trace(client: TestClient, db_session: Session) -> None:
    fixtures = _setup_users(db_session)
    admin = fixtures["admin"]
    _document(db_session, admin, "矩阵集高密受限材料")
    db_session.commit()

    viewer_token = _login(client, "viewer@test.com")
    resp = client.post(
        "/api/v1/search",
        json={"query": "普通查看用户能否直接查看矩阵集高密受限文档？", "top_k": 5},
        headers={"Authorization": f"Bearer {viewer_token}"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["matched_chunks"] == []
    assert data["debug"]["permission_probe_early_stop_applied"] is True
    assert data["debug"]["permission_refusal_reason_code"] == "permission_probe_blocked_target"

    traces = db_session.query(TraceLog).filter(TraceLog.trace_type == "permission_denied_retrieval").all()
    assert len(traces) == 1
    assert traces[0].user_id == fixtures["viewer"].id
    assert traces[0].trace_metadata["permission_probe_target_hint"] == "矩阵集高密"

    admin_token = _login(client, "admin@test.com")
    audit_resp = client.get(
        "/api/v1/permissions/audit/traces",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert audit_resp.status_code == 200
    assert audit_resp.json()[0]["trace_type"] == "permission_denied_retrieval"
