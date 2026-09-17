from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from sqlalchemy.orm import Session

from ...db import get_db
from ...deps import get_current_admin
from ...models import User
from ...services.admin_service import (
    AdminServiceError,
    approve_pending_user,
    list_pending_users_payload,
    list_users_payload,
    update_user_role,
)

router = APIRouter()


@router.get("/api/admin/pending-users")
def admin_pending_users(
    user: User = Depends(get_current_admin), db: Session = Depends(get_db)
) -> JSONResponse:
    return JSONResponse(list_pending_users_payload(db))


@router.post("/api/admin/pending-users/{user_id}/approve")
def admin_approve_pending_user(
    user_id: int,
    user: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> JSONResponse:
    try:
        return JSONResponse(
            approve_pending_user(db, pending_user_id=user_id, admin=user)
        )
    except AdminServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code})


@router.get("/api/admin/users")
def admin_users(
    q: str = Query(default=""),
    user: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> JSONResponse:
    return JSONResponse(list_users_payload(db, query=q))


@router.post("/api/admin/users/{user_id}/role")
async def admin_update_user_role(
    user_id: int,
    request: Request,
    user: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid_json"})
    try:
        return JSONResponse(
            update_user_role(db, user_id=user_id, role=payload.get("role"))
        )
    except AdminServiceError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code})


@router.get("/admin", response_model=None)
def admin_page(user: User = Depends(get_current_admin)) -> Response:
    del user
    path = Path("web") / "admin.html"
    if not path.exists():
        return HTMLResponse(status_code=404, content="not found")
    return FileResponse(str(path), media_type="text/html")


@router.get("/api/admin/settings")
def admin_settings(user: User = Depends(get_current_admin)) -> JSONResponse:
    from ...core.config import settings
    from ...core.config.overrides import SECRETS, read_overrides
    saved = read_overrides()
    active = {
        'HISTORY_RETENTION_DAYS': str(settings.history_retention_days),
        'ENABLE_SELF_SIGNUP': str(int(settings.enable_self_signup)),
        'ALLOW_GUEST_TRANSCRIPTION': str(int(settings.allow_guest_transcription)),
        'ASR_BACKEND': settings.asr_backend, 'ASR_MODEL': settings.asr_model,
        'ASR_BASE_URL': settings.openai_base_url or '', 'ASR_API_KEY': settings.openai_api_key,
        'SUMMARY_BASE_URL': settings.summary_base_url or '', 'SUMMARY_MODEL': settings.summary_model,
        'SUMMARY_API_KEY': settings.summary_api_key,
    }
    values = {**active, **saved}
    values['HISTORY_RETENTION_DAYS'] = values['HISTORY_RETENTION_DAYS'] or '0'
    values['ASR_BACKEND'] = values['ASR_BACKEND'] or 'whisper'
    for key in ('ENABLE_SELF_SIGNUP', 'ALLOW_GUEST_TRANSCRIPTION'):
        values[key] = values[key] or '0'
    return JSONResponse({'values': {key: value for key, value in values.items() if key not in SECRETS},
                         'configuredSecrets': {key: bool(values[key]) for key in SECRETS}})


@router.put("/api/admin/settings")
async def save_admin_settings(request: Request, user: User = Depends(get_current_admin)) -> JSONResponse:
    from ...core.config.overrides import save_overrides
    try:
        save_overrides(await request.json())
    except (ValueError, TypeError):
        return JSONResponse(status_code=400, content={'error': 'invalid_settings'})
    return JSONResponse({'ok': True, 'restartRequired': True})
