from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=200)


class RegisterRequest(LoginRequest):
    display_name: str | None = Field(default=None, max_length=120)


class HistorySaveRequest(BaseModel):
    runtimeSessionId: str = Field(min_length=1, max_length=128)
    runtimeSessionToken: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=255)
    summaryText: str | None = None
    proofreadText: str | None = None
