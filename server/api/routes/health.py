from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ... import legacy_app as legacy
from ...core.logging import emit_container_log

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/api/health")
async def health() -> JSONResponse:
    emit_container_log(__name__, "debug", "health requested")
    logger.debug("health requested")
    return await legacy.health()
