from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response

from ... import legacy_app as legacy
from ...core.logging import emit_container_log
from ...deps import get_current_user
from ...models import User

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/summarize")
async def summarize(payload: legacy.SummarizeRequest, user: User = Depends(get_current_user)) -> JSONResponse:
    emit_container_log(__name__, "debug", "summary requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    logger.debug("summary requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    return await legacy.summarize(payload)


@router.post("/api/proofread")
async def proofread(payload: legacy.ProofreadRequest, user: User = Depends(get_current_user)) -> JSONResponse:
    emit_container_log(__name__, "debug", "proofread requested(route): chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    logger.debug("proofread requested(route): chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    return await legacy.proofread(payload)


@router.post("/api/proofread/stream")
async def proofread_stream(payload: legacy.ProofreadRequest, user: User = Depends(get_current_user)) -> Response:
    emit_container_log(__name__, "debug", "proofread stream requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    logger.debug("proofread stream requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    return await legacy.proofread_stream(payload)
