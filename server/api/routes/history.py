from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy.orm import Session

from ...core.logging import emit_container_log
from ...db import get_db
from ...deps import get_current_user
from ...models import User
from ...schemas import HistorySaveRequest
from ...services.history_service import (
    HistoryError,
    build_history_create_payload,
    build_history_detail_payload_for_user,
    build_history_list_payload,
    create_history_from_payload,
    delete_history_for_user,
    get_history_download_response,
    get_history_audio_response,
    get_history_screenshot_response,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post('/api/history')
async def create_history(
    payload: HistorySaveRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    emit_container_log(__name__, "debug", "history save requested: user_id=%s runtime_session_id=%s", user.id, payload.runtimeSessionId)
    logger.debug("history save requested: user_id=%s runtime_session_id=%s", user.id, payload.runtimeSessionId)
    try:
        history = create_history_from_payload(db, user=user, payload=payload)
    except HistoryError as exc:
        return JSONResponse(status_code=exc.status_code, content={'error': exc.code})
    return JSONResponse(build_history_create_payload(history))


@router.get('/api/history')
async def get_history_list(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: str | None = Query(default=None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    emit_container_log(__name__, "debug", "history list requested: user_id=%s limit=%s offset=%s query=%s", user.id, limit, offset, q or "")
    logger.debug("history list requested: user_id=%s limit=%s offset=%s query=%s", user.id, limit, offset, q or "")
    return JSONResponse(build_history_list_payload(db, user=user, limit=limit, offset=offset, query=q))


@router.get('/api/history/{history_id}')
async def get_history_detail(
    history_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    emit_container_log(__name__, "debug", "history detail requested: user_id=%s history_id=%s", user.id, history_id)
    logger.debug("history detail requested: user_id=%s history_id=%s", user.id, history_id)
    try:
        payload = build_history_detail_payload_for_user(db, user=user, history_id=history_id)
    except HistoryError as exc:
        return JSONResponse(status_code=exc.status_code, content={'error': exc.code})
    return JSONResponse(payload)


@router.delete('/api/history/{history_id}')
async def delete_history_entry(
    history_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    deleted = delete_history_for_user(db, user=user, history_id=history_id)
    if not deleted:
        return JSONResponse(status_code=404, content={'error': 'history_not_found'})
    return JSONResponse({'ok': True, 'historyId': history_id})


@router.get('/api/history/{history_id}/download.txt', response_model=None)
async def download_history_txt(history_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Response:
    response = get_history_download_response(db, user=user, history_id=history_id, kind='txt')
    return response or HTMLResponse(status_code=404, content='not found')


@router.get('/api/history/{history_id}/download.jsonl', response_model=None)
async def download_history_jsonl(history_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Response:
    response = get_history_download_response(db, user=user, history_id=history_id, kind='jsonl')
    return response or HTMLResponse(status_code=404, content='not found')


@router.get('/api/history/{history_id}/download.zip', response_model=None)
async def download_history_zip(history_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Response:
    response = get_history_download_response(db, user=user, history_id=history_id, kind='zip')
    return response or HTMLResponse(status_code=404, content='not found')


@router.get('/api/history/{history_id}/screenshots/{filename}', response_model=None)
async def get_history_screenshot(
    history_id: str,
    filename: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    response = get_history_screenshot_response(db, user=user, history_id=history_id, filename=filename)
    return response or HTMLResponse(status_code=404, content='not found')


@router.get('/api/history/{history_id}/audio/{filename}', response_model=None)
async def get_history_audio(
    history_id: str,
    filename: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    response = get_history_audio_response(db, user=user, history_id=history_id, filename=filename)
    return response or HTMLResponse(status_code=404, content='not found')
