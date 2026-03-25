from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import hash_password
from app.models.enums import RoleName
from app.models.role import Role
from app.models.user import User


def test_login_and_me_returns_current_user(client: TestClient, db_session: Session) -> None:
    viewer_role = Role(name=RoleName.VIEWER, description="Viewer")
    db_session.add(viewer_role)
    db_session.flush()
    db_session.add(
        User(
            email="viewer@example.com",
            full_name="Viewer User",
            password_hash=hash_password("viewer-pass"),
            team_name="sales",
            is_active=True,
            role_id=viewer_role.id,
        )
    )
    db_session.commit()

    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": "viewer@example.com", "password": "viewer-pass"},
    )

    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    me_response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert me_response.status_code == 200
    payload = me_response.json()
    assert payload["email"] == "viewer@example.com"
    assert payload["role"]["name"] == "viewer"
