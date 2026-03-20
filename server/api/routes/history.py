from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy.orm import Session

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
    get_history_download_response,
    get_history_screenshot_response,
)

router = APIRouter()


@router.post('/api/history')
async def create_history(
    payload: HistorySaveRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
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
    return JSONResponse(build_history_list_payload(db, user=user, limit=limit, offset=offset, query=q))


@router.get('/api/history/{history_id}')
async def get_history_detail(
    history_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    try:
        payload = build_history_detail_payload_for_user(db, user=user, history_id=history_id)
    except HistoryError as exc:
        return JSONResponse(status_code=exc.status_code, content={'error': exc.code})
    return JSONResponse(payload)


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
