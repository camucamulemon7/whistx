from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ...core.security import read_guest_artifact_grant
from ...db import get_db
from ...services.auth_service import get_optional_user_from_request
from ...services import runtime_artifact_service

router = APIRouter()


@router.get("/api/transcript/{session_id}.txt", response_model=None)
def get_txt(
    session_id: str, request: Request, db: Session = Depends(get_db)
) -> Response:
    user = get_optional_user_from_request(request, db)
    return runtime_artifact_service.get_txt(
        session_id,
        user_id=user.id if user is not None else None,
        guest_grant_id=read_guest_artifact_grant(request),
    )


@router.get("/api/transcript/{session_id}.jsonl", response_model=None)
def get_jsonl(
    session_id: str, request: Request, db: Session = Depends(get_db)
) -> Response:
    user = get_optional_user_from_request(request, db)
    return runtime_artifact_service.get_jsonl(
        session_id,
        user_id=user.id if user is not None else None,
        guest_grant_id=read_guest_artifact_grant(request),
    )


@router.get("/api/transcript/{session_id}.zip", response_model=None)
def get_zip(
    session_id: str, request: Request, db: Session = Depends(get_db)
) -> Response:
    user = get_optional_user_from_request(request, db)
    return runtime_artifact_service.get_zip(
        session_id,
        user_id=user.id if user is not None else None,
        guest_grant_id=read_guest_artifact_grant(request),
    )


@router.get("/api/transcripts/{session_id}/screenshots/{filename}", response_model=None)
def get_screenshot(
    session_id: str,
    filename: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Response:
    user = get_optional_user_from_request(request, db)
    return runtime_artifact_service.get_screenshot(
        session_id,
        filename,
        user_id=user.id if user is not None else None,
        guest_grant_id=read_guest_artifact_grant(request),
    )


@router.get("/api/transcripts/{session_id}/audio/{filename}", response_model=None)
def get_debug_audio(
    session_id: str,
    filename: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Response:
    user = get_optional_user_from_request(request, db)
    return runtime_artifact_service.get_debug_audio(
        session_id,
        filename,
        user_id=user.id if user is not None else None,
        guest_grant_id=read_guest_artifact_grant(request),
    )
