from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from ...db import get_db
from ...deps import get_current_user
from ...models import User
from ...schemas import HistorySaveRequest
from ... import legacy_app as legacy

router = APIRouter()


@router.post("/api/history")
async def create_history(
    payload: HistorySaveRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    return await legacy.create_history(payload, user, db)


@router.get("/api/history")
async def get_history_list(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: str | None = Query(default=None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    return await legacy.get_history_list(limit, offset, q, user, db)


@router.get("/api/history/{history_id}")
async def get_history_detail(
    history_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JSONResponse:
    return await legacy.get_history_detail(history_id, user, db)


@router.get("/api/history/{history_id}/download.txt", response_model=None)
async def download_history_txt(history_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Response:
    return await legacy.download_history_txt(history_id, user, db)


@router.get("/api/history/{history_id}/download.jsonl", response_model=None)
async def download_history_jsonl(history_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Response:
    return await legacy.download_history_jsonl(history_id, user, db)


@router.get("/api/history/{history_id}/download.zip", response_model=None)
async def download_history_zip(history_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Response:
    return await legacy.download_history_zip(history_id, user, db)


@router.get("/api/history/{history_id}/screenshots/{filename}", response_model=None)
async def get_history_screenshot(
    history_id: str,
    filename: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    return await legacy.get_history_screenshot(history_id, filename, user, db)
