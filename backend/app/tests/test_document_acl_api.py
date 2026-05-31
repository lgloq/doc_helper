from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.document import Document, DocumentACL
from app.models.enums import DocumentStatus, PrincipalType, RoleName
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


def test_document_list_is_filtered_by_acl(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    manager_role = Role(name=RoleName.MANAGER, description="Manager")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, manager_role, admin_role])
    db_session.flush()

    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    manager = _create_user(db_session, manager_role, "manager@example.com", "platform", "manager-pass")
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    public_doc = Document(title="Public Handbook", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    team_doc = Document(title="Platform Plan", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    explicit_doc = Document(title="Viewer Brief", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    secret_doc = Document(title="Admin Secret", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add_all([public_doc, team_doc, explicit_doc, secret_doc])
    db_session.flush()

    db_session.add_all(
        [
            DocumentACL(document_id=public_doc.id, principal_type=PrincipalType.PUBLIC, can_view=True, can_manage=False),
            DocumentACL(document_id=team_doc.id, principal_type=PrincipalType.TEAM, team_name="platform", can_view=True, can_manage=False),
            DocumentACL(document_id=explicit_doc.id, principal_type=PrincipalType.USER, user_id=viewer.id, can_view=True, can_manage=False),
        ]
    )
    db_session.commit()

    viewer_token = _login(client, "viewer@example.com", "viewer-pass")
    manager_token = _login(client, "manager@example.com", "manager-pass")
    admin_token = _login(client, "admin@example.com", "admin-pass")

    viewer_docs = client.get("/api/v1/documents", headers={"Authorization": f"Bearer {viewer_token}"})
    manager_docs = client.get("/api/v1/documents", headers={"Authorization": f"Bearer {manager_token}"})
    admin_docs = client.get("/api/v1/documents", headers={"Authorization": f"Bearer {admin_token}"})

    assert viewer_docs.status_code == 200
    assert {item["title"] for item in viewer_docs.json()} == {"Public Handbook", "Viewer Brief"}

    assert manager_docs.status_code == 200
    assert {item["title"] for item in manager_docs.json()} == {"Public Handbook", "Platform Plan"}

    assert admin_docs.status_code == 200
    assert {item["title"] for item in admin_docs.json()} == {
        "Public Handbook",
        "Platform Plan",
        "Viewer Brief",
        "Admin Secret",
    }


def test_acl_endpoints_require_manage_access_and_allow_assignment(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")

    document = Document(title="Restricted Policy", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add(document)
    db_session.commit()

    viewer_token = _login(client, "viewer@example.com", "viewer-pass")
    admin_token = _login(client, "admin@example.com", "admin-pass")

    forbidden_acl_response = client.get(
        f"/api/v1/documents/{document.id}/acl",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert forbidden_acl_response.status_code == 404

    create_acl_response = client.post(
        f"/api/v1/documents/{document.id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "principal_type": "user",
            "user_id": str(viewer.id),
            "can_view": True,
            "can_manage": False,
        },
    )
    assert create_acl_response.status_code == 200
    assert create_acl_response.json()["user_email"] == "viewer@example.com"

    acl_list_response = client.get(
        f"/api/v1/documents/{document.id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert acl_list_response.status_code == 200
    assert len(acl_list_response.json()) == 1

    viewer_document_response = client.get(
        f"/api/v1/documents/{document.id}",
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert viewer_document_response.status_code == 200
    assert viewer_document_response.json()["title"] == "Restricted Policy"


def test_acl_delete_endpoint_removes_assignment(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    document = Document(title="Restricted Policy", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add(document)
    db_session.flush()
    acl_entry = DocumentACL(
        document_id=document.id,
        principal_type=PrincipalType.USER,
        user_id=viewer.id,
        can_view=True,
        can_manage=False,
    )
    db_session.add(acl_entry)
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    delete_response = client.delete(
        f"/api/v1/documents/{document.id}/acl/{acl_entry.id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert delete_response.status_code == 204
    acl_list_response = client.get(
        f"/api/v1/documents/{document.id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert acl_list_response.status_code == 200
    assert acl_list_response.json() == []


def test_acl_upsert_with_no_permissions_removes_existing_entry(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    document = Document(title="Restricted Policy", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add(document)
    db_session.flush()
    db_session.add(
        DocumentACL(
            document_id=document.id,
            principal_type=PrincipalType.USER,
            user_id=viewer.id,
            can_view=True,
            can_manage=False,
        )
    )
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    revoke_response = client.post(
        f"/api/v1/documents/{document.id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "principal_type": "user",
            "user_id": str(viewer.id),
            "can_view": False,
            "can_manage": False,
        },
    )

    assert revoke_response.status_code == 200
    assert revoke_response.json() is None
    acl_list_response = client.get(
        f"/api/v1/documents/{document.id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert acl_list_response.status_code == 200
    assert acl_list_response.json() == []


def test_acl_list_cleans_existing_no_permission_entries(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    admin_role = Role(name=RoleName.ADMIN, description="Admin")
    db_session.add_all([viewer_role, admin_role])
    db_session.flush()

    viewer = _create_user(db_session, viewer_role, "viewer@example.com", "sales", "viewer-pass")
    admin = _create_user(db_session, admin_role, "admin@example.com", None, "admin-pass")
    document = Document(title="Restricted Policy", description=None, status=DocumentStatus.ACTIVE, owner_user_id=admin.id)
    db_session.add(document)
    db_session.flush()
    db_session.add_all(
        [
            DocumentACL(
                document_id=document.id,
                principal_type=PrincipalType.USER,
                user_id=viewer.id,
                can_view=False,
                can_manage=False,
            ),
            DocumentACL(
                document_id=document.id,
                principal_type=PrincipalType.PUBLIC,
                can_view=True,
                can_manage=False,
            ),
        ]
    )
    db_session.commit()

    admin_token = _login(client, "admin@example.com", "admin-pass")
    acl_list_response = client.get(
        f"/api/v1/documents/{document.id}/acl",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert acl_list_response.status_code == 200
    data = acl_list_response.json()
    assert len(data) == 1
    assert data[0]["principal_type"] == "public"
