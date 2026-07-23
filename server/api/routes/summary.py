from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from ... import runtime
from ...core.logging import emit_container_log
from ...core.config import settings
from ...core.rate_limit import consume
from ...deps import get_current_user
from ...models import User

router = APIRouter()
logger = logging.getLogger(__name__)


def _allow_costly_request(bucket: str, user: User) -> bool:
    allowed = consume(
        bucket=bucket,
        subject=f"user:{user.id}",
        limit=settings.costly_api_rate_limit_requests,
        window_seconds=settings.costly_api_rate_limit_window_seconds,
    )
    if not allowed:
        logger.warning("costly API rate limit exceeded: bucket=%s user_id=%s", bucket, user.id)
    return allowed


@router.post("/api/summarize")
async def summarize(
    payload: runtime.SummarizeRequest,
    user: User = Depends(get_current_user),
) -> JSONResponse:
    if not _allow_costly_request("summary", user):
        return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
    emit_container_log(__name__, "debug", "summary requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    logger.debug("summary requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    return await runtime.summarize(payload)


@router.post("/api/proofread")
async def proofread(
    payload: runtime.ProofreadRequest,
    user: User = Depends(get_current_user),
) -> JSONResponse:
    if not _allow_costly_request("proofread", user):
        return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
    emit_container_log(__name__, "debug", "proofread requested(route): chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    logger.debug("proofread requested(route): chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    return await runtime.proofread(payload)


@router.post("/api/proofread/stream")
async def proofread_stream(
    payload: runtime.ProofreadRequest,
    user: User = Depends(get_current_user),
) -> Response:
    if not _allow_costly_request("proofread", user):
        return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
    emit_container_log(__name__, "debug", "proofread stream requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    logger.debug("proofread stream requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    return await runtime.proofread_stream(payload)
