from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.schemas.user import UserRead


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserRead
