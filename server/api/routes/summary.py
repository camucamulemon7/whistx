from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ... import runtime
from ...core.logging import emit_container_log
from ...core.config import settings
from ...core.rate_limit import consume
from ...deps import get_current_user
from ...models import User
from ...core.blocking import blocking_work_pool
from ...schemas import MeetingSourceRequest, MeetingRecapRequest, MeetingQuestionRequest
from ...services.meeting_source import MeetingError, load_meeting
from ...services.meeting_intelligence import generate_recap, read_insights, recap_markdown
from ...services.meeting_stream_response import stream_answer, stream_events
from ...services.meeting_refinement import refine_events

router = APIRouter()
logger = logging.getLogger(__name__)


async def _meeting_source(payload: MeetingSourceRequest, user: User):
    return await blocking_work_pool.run(
        "artifact", load_meeting, user_id=user.id,
        runtime_session_id=payload.runtimeSessionId, history_id=payload.historyId,
    )


@router.post("/api/meeting/insights")
async def meeting_insights(payload: MeetingSourceRequest, user: User = Depends(get_current_user)) -> JSONResponse:
    try:
        snapshot = await _meeting_source(payload, user)
        return JSONResponse(await blocking_work_pool.run("artifact", read_insights, snapshot))
    except MeetingError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code})


@router.post("/api/meeting/recap")
async def meeting_recap(payload: MeetingRecapRequest, user: User = Depends(get_current_user)) -> JSONResponse:
    if not await asyncio.to_thread(_allow_costly_request, "summary", user):
        return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
    if runtime.SUMMARIZER is None:
        return JSONResponse(status_code=503, content={"error": "summary_not_configured"})
    try:
        snapshot = await _meeting_source(payload, user)
        recap = await blocking_work_pool.run("llm", generate_recap, snapshot, runtime.SUMMARIZER,
                                             prompt=payload.prompt, max_chars=settings.summary_input_max_chars)
        return JSONResponse({**snapshot.public(), "recap": recap, "summary": recap_markdown(recap)})
    except MeetingError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code})
    except Exception as exc:
        logger.warning("meeting recap failed: %s", type(exc).__name__)
        return JSONResponse(status_code=502, content={"error": "meeting_model_unavailable"})


@router.post("/api/meeting/ask")
async def meeting_ask(payload: MeetingQuestionRequest, user: User = Depends(get_current_user)) -> Response:
    if not await asyncio.to_thread(_allow_costly_request, "meeting_qa", user):
        return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
    if runtime.SUMMARIZER is None:
        return JSONResponse(status_code=503, content={"error": "summary_not_configured"})
    try:
        snapshot = await _meeting_source(payload, user)
    except MeetingError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code})
    return StreamingResponse(stream_answer(snapshot, runtime.SUMMARIZER, payload.question),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@router.post("/api/meeting/refine")
async def meeting_refine(payload: MeetingSourceRequest, user: User = Depends(get_current_user)) -> Response:
    if not await asyncio.to_thread(_allow_costly_request, "asr", user):
        return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
    try:
        snapshot = await _meeting_source(payload, user)
        if payload.historyId or not snapshot.finalized:
            raise MeetingError("refinement_requires_completed_runtime", 409)
    except MeetingError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.code})
    return StreamingResponse(stream_events(lambda cancelled: refine_events(snapshot, cancelled=cancelled,
                             allow_request=lambda: _allow_costly_request("asr", user)), pool="asr"),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


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
    if not await asyncio.to_thread(_allow_costly_request, "summary", user):
        return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
    emit_container_log(__name__, "debug", "summary requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    logger.debug("summary requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    return await runtime.summarize(payload)


@router.post("/api/proofread")
async def proofread(
    payload: runtime.ProofreadRequest,
    user: User = Depends(get_current_user),
) -> JSONResponse:
    if not await asyncio.to_thread(_allow_costly_request, "proofread", user):
        return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
    emit_container_log(__name__, "debug", "proofread requested(route): chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    logger.debug("proofread requested(route): chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    return await runtime.proofread(payload)


@router.post("/api/proofread/stream")
async def proofread_stream(
    payload: runtime.ProofreadRequest,
    user: User = Depends(get_current_user),
) -> Response:
    if not await asyncio.to_thread(_allow_costly_request, "proofread", user):
        return JSONResponse(status_code=429, content={"error": "rate_limit_exceeded"})
    emit_container_log(__name__, "debug", "proofread stream requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    logger.debug("proofread stream requested: chars=%s language=%s", len(payload.text or ""), payload.language or "auto")
    return await runtime.proofread_stream(payload)
