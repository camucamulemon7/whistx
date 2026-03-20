from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import Response

from ... import legacy_app as legacy

router = APIRouter()


@router.get("/api/transcript/{session_id}.txt", response_model=None)
async def get_txt(session_id: str, token: str | None = Query(default=None)) -> Response:
    return await legacy.get_txt(session_id, token)


@router.get("/api/transcript/{session_id}.jsonl", response_model=None)
async def get_jsonl(session_id: str, token: str | None = Query(default=None)) -> Response:
    return await legacy.get_jsonl(session_id, token)


@router.get("/api/transcript/{session_id}.zip", response_model=None)
async def get_zip(session_id: str, token: str | None = Query(default=None)) -> Response:
    return await legacy.get_zip(session_id, token)


@router.get("/api/transcripts/{session_id}/screenshots/{filename}", response_model=None)
async def get_screenshot(session_id: str, filename: str, token: str | None = Query(default=None)) -> Response:
    return await legacy.get_screenshot(session_id, filename, token)
