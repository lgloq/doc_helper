from fastapi.testclient import TestClient

from app.main import app


def test_liveness_check_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["checks"][0]["name"] == "api"
