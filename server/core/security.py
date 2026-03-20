from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from typing import Any

from fastapi import Request
from fastapi.responses import Response

from ..config import settings
from ..transcript_store import resolve_transcript_path


def runtime_access_allowed(*, transcripts_dir, session_id: str, token: str | None) -> bool:
    base_path = resolve_transcript_path(transcripts_dir, session_id, "txt")
    if base_path is None:
        return False
    metadata_path = base_path.with_suffix(".meta.json")
    if not metadata_path.exists():
        return True
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = loaded if isinstance(loaded, dict) else {}
    required = str(metadata.get("accessToken") or "").strip()
    if not required:
        return True
    return secrets.compare_digest(required, str(token or ""))


def serialize_user(user) -> dict[str, Any] | None:
    if user is None:
        return None
    return {
        "id": user.id,
        "email": user.email,
        "displayName": user.display_name,
        "isAdmin": bool(user.is_admin),
        "isActive": bool(user.is_active),
        "approvedAt": user.approved_at.isoformat() if user.approved_at else None,
    }


def signed_payload(value: dict[str, Any]) -> str:
    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    body = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    signature = hmac.new(settings.app_session_secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    token = base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("=")
    return f"{body}.{token}"


def unsigned_payload(value: str | None) -> dict[str, Any] | None:
    if not value or "." not in value:
        return None
    body, token = value.split(".", 1)
    expected = hmac.new(settings.app_session_secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    if not secrets.compare_digest(base64.urlsafe_b64encode(expected).decode("utf-8").rstrip("="), token):
        return None
    padded = body + "=" * (-len(body) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")))
    except Exception:
        return None


def set_oidc_state_cookie(*, response: Response, request: Request, cookie_name: str, payload: dict[str, Any]) -> None:
    response.set_cookie(
        key=cookie_name,
        value=signed_payload(payload),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=10 * 60,
        path="/",
    )


def read_oidc_state_cookie(*, request: Request, cookie_name: str) -> dict[str, Any] | None:
    return unsigned_payload(request.cookies.get(cookie_name))


def clear_oidc_state_cookie(*, response: Response, request: Request, cookie_name: str) -> None:
    response.delete_cookie(
        key=cookie_name,
        path="/",
        secure=request.url.scheme == "https",
        httponly=True,
        samesite="lax",
    )


def set_session_cookie(*, response: Response, request: Request, cookie_name: str, session_id: str) -> None:
    response.set_cookie(
        key=cookie_name,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=settings.app_session_days * 24 * 60 * 60,
        path="/",
    )


def clear_session_cookie(*, response: Response, request: Request, cookie_name: str) -> None:
    response.delete_cookie(
        key=cookie_name,
        path="/",
        secure=request.url.scheme == "https",
        httponly=True,
        samesite="lax",
    )
