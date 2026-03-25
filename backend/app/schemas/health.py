from pydantic import BaseModel


class HealthDependencyStatus(BaseModel):
    name: str
    status: str


class HealthResponse(BaseModel):
    status: str
    service: str
    checks: list[HealthDependencyStatus]
