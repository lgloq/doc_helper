from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.db.redis import check_redis_connection
from app.db.session import check_database_connection
from app.schemas.health import HealthDependencyStatus, HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
def liveness_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="api",
        checks=[HealthDependencyStatus(name="api", status="ok")],
    )


@router.get("/ready", response_model=HealthResponse)
def readiness_check():
    database_ok = check_database_connection()
    redis_ok = check_redis_connection()

    response = HealthResponse(
        status="ok" if database_ok and redis_ok else "degraded",
        service="api",
        checks=[
            HealthDependencyStatus(name="database", status="ok" if database_ok else "error"),
            HealthDependencyStatus(name="redis", status="ok" if redis_ok else "error"),
        ],
    )

    if database_ok and redis_ok:
        return response

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response.model_dump(),
    )
