from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=200)


class RegisterRequest(LoginRequest):
    display_name: str | None = Field(default=None, max_length=120)


class BootstrapAdminRequest(RegisterRequest):
    pass


class UpdateDisplayNameRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=120)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=8, max_length=200)
    new_password: str = Field(min_length=8, max_length=200)


class HistorySaveRequest(BaseModel):
    runtimeSessionId: str = Field(min_length=1, max_length=128)
    runtimeSessionToken: str = Field(default="", max_length=128)
    title: str | None = Field(default=None, max_length=255)
    summaryText: str | None = None
    proofreadText: str | None = None


class MeetingSourceRequest(BaseModel):
    runtimeSessionId: str = Field(default="", max_length=128)
    historyId: str = Field(default="", max_length=128)

    @model_validator(mode="after")
    def one_source(self):
        if bool(self.runtimeSessionId) == bool(self.historyId):
            raise ValueError("one_meeting_source_required")
        return self


class MeetingRecapRequest(MeetingSourceRequest):
    prompt: str = Field(default="", max_length=4000)


class MeetingQuestionRequest(MeetingSourceRequest):
    question: str = Field(min_length=1, max_length=2000)
