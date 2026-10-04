from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from ..services.diarization_service import run_diarization_for_session
from ..core.config import settings
from ..core.public_errors import public_error
from ..repositories import quota_repository
from ..transcription.factory import create_live_session
from ..transcription.media import (
    _store_screenshot_for_chunk,
    _save_debug_audio_chunk,
    _store_debug_raw_audio_for_final,
    _store_debug_audio_for_final,
    _resolve_existing_debug_audio_url,
    _merge_wav_chunks,
)
from ..transcription.messages import (
    as_str as _as_str,
    chunk_payload_size_error as _chunk_payload_size_error,
    parse_chunk_message as _parse_chunk_message,
    validate_start_message as _validate_start_message,
    validate_telemetry_message as _validate_telemetry_message,
    validate_chunk_order as _validate_chunk_order,
)
from ..transcription.session import ChunkMessage, LiveSession
from ..transcription.text_processing import (
    _build_prompt,
    _append_context,
    _sanitize_transcript_text,
    _light_proofread,
    _should_drop_boundary_fragment,
    _accumulate_asr_usage,
    _retry_weird_transcription_if_needed,
    _trim_overlap_prefix,
    _coerce_monotonic_bounds,
    _is_near_duplicate,
)
from ..transcription.worker import WorkerDependencies, run_session_worker
from ..core.logging import emit_container_log
from ..core.rate_limit import consume as consume_rate_limit

from ..core.runtime_resources import RuntimeResources

logger = logging.getLogger(__name__)


async def ws_transcribe(ws: WebSocket, *, resources: RuntimeResources) -> None:
    await ws.accept()
    resources.active_sockets.add(ws)
    await _broadcast_conn_count(resources=resources)

    session: LiveSession | None = None
    worker_task: asyncio.Task[None] | None = None
    is_guest = bool(getattr(ws.state, "is_guest", False))
    guest_audio_bytes = 0
    guest_asr_requests = 0
    stop_requested = False
    invalid_message_count = 0

    try:
        while True:
            raw = await ws.receive_text()
            if len(raw.encode("utf-8")) > settings.ws_max_message_bytes:
                await _safe_send(
                    ws,
                    {
                        "type": "error",
                        "message": "message_too_large",
                        "maxBytes": settings.ws_max_message_bytes,
                    },
                )
                await ws.close(code=4409, reason="message_too_large")
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                invalid_message_count, should_close = await _reject_invalid_message(
                    ws,
                    {"type": "error", "message": "invalid_json"},
                    invalid_message_count,
                )
                if should_close:
                    break
                continue

            if not isinstance(data, dict):
                invalid_message_count, should_close = await _reject_invalid_message(
                    ws,
                    {"type": "error", "message": "invalid_payload"},
                    invalid_message_count,
                )
                if should_close:
                    break
                continue

            msg_type = str(data.get("type", "")).strip().lower()

            if msg_type == "start":
                if session is not None:
                    await _safe_send(
                        ws, {"type": "error", "message": "already_started"}
                    )
                    continue
                start_error = _validate_start_message(
                    data,
                    prompt_max_chars=settings.ws_prompt_max_chars,
                    vocabulary_max_chars=settings.ws_vocabulary_max_chars,
                )
                if start_error is not None:
                    invalid_message_count, should_close = await _reject_invalid_message(
                        ws,
                        {"type": "error", "message": start_error},
                        invalid_message_count,
                    )
                    if should_close:
                        break
                    continue

                try:
                    session = _create_session(
                        data,
                        resources=resources,
                        owner_user_id=getattr(ws.state, "authenticated_user_id", None),
                        guest_grant_digest=getattr(
                            ws.state, "guest_artifact_grant_digest", None
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Session creation failed")
                    await _safe_send(
                        ws,
                        {
                            "type": "error",
                            "message": "session_create_failed",
                            **public_error("transcription_failed", exc, logger),
                        },
                    )
                    continue
                worker_task = asyncio.create_task(_session_worker(ws, session, resources=resources))
                logger.info(
                    "ws session ready: session=%s source=%s language=%s diarization=%s",
                    session.session_id,
                    session.audio_source,
                    session.language or "auto",
                    session.collect_audio_for_diarization,
                )
                emit_container_log(
                    __name__,
                    "info",
                    "ws session ready: session=%s source=%s language=%s diarization=%s",
                    session.session_id,
                    session.audio_source,
                    session.language or "auto",
                    session.collect_audio_for_diarization,
                )

                await _safe_send(
                    ws,
                    {
                        "type": "info",
                        "message": "ready",
                        "state": "ready",
                        "sessionId": session.session_id,
                        "backend": f"openai:{settings.asr_model}",
                        "diarizationEnabled": session.collect_audio_for_diarization,
                        "diarizationNumSpeakers": session.diarization_num_speakers,
                        "diarizationMinSpeakers": session.diarization_min_speakers,
                        "diarizationMaxSpeakers": session.diarization_max_speakers,
                    },
                )
                await _broadcast_conn_count(resources=resources)
                continue

            if msg_type == "chunk":
                if session is None:
                    await _safe_send(ws, {"type": "error", "message": "not_started"})
                    continue

                size_error = _chunk_payload_size_error(
                    data,
                    max_audio_bytes=settings.max_chunk_bytes,
                    max_screenshot_bytes=settings.ws_screenshot_max_bytes,
                )
                if size_error is not None:
                    invalid_message_count, should_close = await _reject_invalid_message(
                        ws,
                        {"type": "error", "message": size_error},
                        invalid_message_count,
                    )
                    if should_close:
                        break
                    continue

                chunk = _parse_chunk_message(
                    data,
                    max_audio_bytes=settings.max_chunk_bytes,
                    max_screenshot_bytes=settings.ws_screenshot_max_bytes,
                )
                if chunk is None:
                    invalid_message_count, should_close = await _reject_invalid_message(
                        ws,
                        {"type": "error", "message": "invalid_chunk"},
                        invalid_message_count,
                    )
                    if should_close:
                        break
                    continue

                ordering_error = _validate_chunk_order(session, chunk)
                if ordering_error is not None:
                    await _safe_send(ws, ordering_error)
                    continue

                if len(chunk.audio_bytes) > settings.max_chunk_bytes:
                    await _safe_send(
                        ws,
                        {
                            "type": "error",
                            "message": "chunk_too_large",
                            "maxBytes": settings.max_chunk_bytes,
                        },
                    )
                    continue

                if (
                    is_guest
                    and guest_audio_bytes + len(chunk.audio_bytes)
                    > settings.guest_ws_max_audio_bytes
                ):
                    await _safe_send(
                        ws, {"type": "error", "message": "guest_audio_limit"}
                    )
                    await ws.close(code=4408, reason="guest_audio_limit")
                    break
                if (
                    is_guest
                    and guest_asr_requests >= settings.guest_ws_max_asr_requests
                ):
                    await _safe_send(
                        ws, {"type": "error", "message": "guest_asr_request_limit"}
                    )
                    await ws.close(code=4408, reason="guest_asr_request_limit")
                    break
                rate_limit_subject = str(
                    getattr(ws.state, "rate_limit_subject", "unknown")
                )
                if not consume_rate_limit(
                    bucket="asr",
                    subject=rate_limit_subject,
                    limit=settings.costly_api_rate_limit_requests,
                    window_seconds=settings.costly_api_rate_limit_window_seconds,
                ):
                    logger.warning(
                        "ASR rate limit exceeded: subject=%s", rate_limit_subject
                    )
                    await _safe_send(
                        ws, {"type": "error", "message": "rate_limit_exceeded"}
                    )
                    await ws.close(code=4429, reason="asr_rate_limit")
                    break

                try:
                    session.queue.put_nowait(chunk)
                    session.last_chunk_seq = chunk.seq
                    session.last_chunk_offset_ms = chunk.offset_ms
                    if is_guest:
                        guest_audio_bytes += len(chunk.audio_bytes)
                        guest_asr_requests += 1
                    logger.debug(
                        "ws chunk queued: session=%s seq=%s duration_ms=%s queue=%s",
                        session.session_id,
                        chunk.seq,
                        chunk.duration_ms,
                        session.queue.qsize(),
                    )
                    emit_container_log(
                        __name__,
                        "debug",
                        "ws chunk queued: session=%s seq=%s duration_ms=%s queue=%s",
                        session.session_id,
                        chunk.seq,
                        chunk.duration_ms,
                        session.queue.qsize(),
                    )
                except asyncio.QueueFull:
                    logger.warning(
                        "ws server busy: session=%s seq=%s queue=%s",
                        session.session_id,
                        chunk.seq,
                        session.queue.qsize(),
                    )
                    emit_container_log(
                        __name__,
                        "warning",
                        "ws server busy: session=%s seq=%s queue=%s",
                        session.session_id,
                        chunk.seq,
                        session.queue.qsize(),
                    )
                    await _safe_send(
                        ws,
                        {
                            "type": "error",
                            "message": "server_busy",
                            "detail": "queue_full",
                        },
                    )
                continue

            if msg_type == "telemetry":
                if session is None:
                    await _safe_send(ws, {"type": "error", "message": "not_started"})
                    continue
                telemetry_error = _validate_telemetry_message(
                    data, max_chars=settings.ws_telemetry_max_chars
                )
                if telemetry_error is not None:
                    invalid_message_count, should_close = await _reject_invalid_message(
                        ws,
                        {"type": "error", "message": telemetry_error},
                        invalid_message_count,
                    )
                    if should_close:
                        break
                    continue
                event_name = _as_str(data.get("event")) or "unknown"
                detail = data.get("detail")
                if event_name in {
                    "degraded_capture_enabled",
                    "server_busy_acknowledged",
                    "transcription_failed_acknowledged",
                }:
                    logger.warning(
                        "client telemetry: session=%s event=%s detail=%s",
                        session.session_id,
                        event_name,
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                    )
                    emit_container_log(
                        __name__,
                        "warning",
                        "client telemetry: session=%s event=%s detail=%s",
                        session.session_id,
                        event_name,
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                    )
                else:
                    logger.debug(
                        "client telemetry: session=%s event=%s detail=%s",
                        session.session_id,
                        event_name,
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                    )
                    emit_container_log(
                        __name__,
                        "debug",
                        "client telemetry: session=%s event=%s detail=%s",
                        session.session_id,
                        event_name,
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                    )
                continue

            if msg_type == "stop":
                logger.info(
                    "ws stop received: session=%s",
                    session.session_id if session is not None else "unknown",
                )
                emit_container_log(
                    __name__,
                    "info",
                    "ws stop received: session=%s",
                    session.session_id if session is not None else "unknown",
                )
                stop_requested = True
                await _safe_send(
                    ws,
                    {
                        "type": "info",
                        "message": "stopping",
                        "sessionId": session.session_id
                        if session is not None
                        else None,
                    },
                )
                break

            if msg_type == "ping":
                await _safe_send(ws, {"type": "pong", "ts": data.get("ts")})
                continue

            await _safe_send(ws, {"type": "error", "message": "unsupported_message"})

    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            await session.queue.put(None)

        if worker_task is not None:
            try:
                await asyncio.wait_for(worker_task, timeout=60)
            except asyncio.TimeoutError:
                worker_task.cancel()

        if session is not None:
            await run_diarization_for_session(ws, session, diarizer=resources.diarizer, send=_safe_send, config=settings)
            _mark_session_finalized(session)
            if stop_requested:
                await _safe_send(
                    ws,
                    {
                        "type": "info",
                        "message": "finalized",
                        "state": "completed",
                        "sessionId": session.session_id,
                    },
                )

        resources.active_sockets.discard(ws)
        await _broadcast_conn_count(resources=resources)



async def _session_worker(ws: WebSocket, session: LiveSession, *, resources: RuntimeResources) -> None:
    await run_session_worker(
        ws,
        session,
        WorkerDependencies(
            settings=settings,
            observer=resources.observer,
            logger=logger,
            prepare_audio=lambda **kwargs: _prepare_audio_for_asr(resources=resources, **kwargs),
            merge_wav_chunks=_merge_wav_chunks,
            safe_send=_safe_send,
            build_prompt=_build_prompt,
            accumulate_usage=_accumulate_asr_usage,
            retry_weird_transcription=_retry_weird_transcription_if_needed,
            save_debug_chunk=_save_debug_audio_chunk,
            sanitize_text=_sanitize_transcript_text,
            trim_overlap=_trim_overlap_prefix,
            light_proofread=_light_proofread,
            should_drop_boundary=_should_drop_boundary_fragment,
            is_near_duplicate=_is_near_duplicate,
            coerce_bounds=_coerce_monotonic_bounds,
            store_screenshot=_store_screenshot_for_chunk,
            store_raw_audio=_store_debug_raw_audio_for_final,
            store_asr_audio=_store_debug_audio_for_final,
            resolve_audio_url=_resolve_existing_debug_audio_url,
            append_context=_append_context,
            clip_trace_text=_clip_trace_text,
            emit_log=emit_container_log,
        ),
    )



def _create_session(
    payload: dict[str, Any],
    *,
    resources: RuntimeResources,
    owner_user_id: int | None = None,
    guest_grant_digest: str | None = None,
) -> LiveSession:
    if resources.transcriber_factory is None:
        raise RuntimeError("transcriber_not_ready")
    return create_live_session(
        payload,
        settings=settings,
        transcriber_factory=resources.transcriber_factory,
        diarizer_available=resources.diarizer is not None,
        owner_user_id=owner_user_id,
        guest_grant_digest=guest_grant_digest,
    )



def _prepare_audio_for_asr(*, resources: RuntimeResources, session: LiveSession, item: ChunkMessage):
    if resources.audio_preprocessor is None:
        raise RuntimeError("audio_preprocessor_not_ready")
    prepared = resources.audio_preprocessor.prepare(
        audio_bytes=item.audio_bytes,
        mime_type=item.mime_type,
        previous_tail_pcm=session.overlap_tail_pcm,
        chunk_duration_ms=item.duration_ms,
        source_mode=session.audio_source,
        speech_ratio=item.speech_ratio,
        active_ms=item.active_ms,
        silence_ms=item.silence_ms,
    )
    session.overlap_tail_pcm = prepared.tail_pcm
    return prepared



def _clip_trace_text(text: str, limit: int = 8000) -> str:
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "...(truncated)"



async def _broadcast_conn_count(*, resources: RuntimeResources) -> None:
    count = await asyncio.to_thread(quota_repository.count_active_connections)
    payload = {"type": "conn", "count": count}
    dead: list[WebSocket] = []

    for sock in list(resources.active_sockets):
        ok = await _safe_send(sock, payload)
        if not ok:
            dead.append(sock)

    for sock in dead:
        resources.active_sockets.discard(sock)



async def _safe_send(ws: WebSocket, payload: dict[str, Any]) -> bool:
    try:
        await ws.send_json(payload)
        return True
    except Exception:  # noqa: BLE001
        return False



async def _reject_invalid_message(
    ws: WebSocket,
    payload: dict[str, Any],
    current_count: int,
) -> tuple[int, bool]:
    next_count = current_count + 1
    message = dict(payload)
    message["invalidCount"] = next_count
    await _safe_send(ws, message)
    if next_count < settings.ws_max_invalid_messages:
        return next_count, False
    await ws.close(code=4400, reason="too_many_invalid_messages")
    return next_count, True



def _mark_session_finalized(session: LiveSession) -> None:
    metadata = session.store.read_metadata()
    metadata["finalized"] = True
    metadata["finalizedAt"] = datetime.now(timezone.utc).isoformat()
    session.store.write_metadata(metadata)
