from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from typing import Any, Callable

from fastapi import Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .auth import (
    SESSION_COOKIE_NAME,
    create_user_session,
)
from .asr import SessionTranscriber
from .audio_pipeline import AudioPreprocessor
from .core.config import settings
from .core.blocking import blocking_work_pool
from .db import db_session, get_db
from .diarizer import PyannoteSpeakerDiarizer, SpeakerTurn
from .langfuse_observer import make_langfuse_observer
from .models import User
from .openai_whisper import OpenAIWhisperTranscriber
from .services.auth_service import (
    get_optional_user_from_request as auth_service_get_optional_user_from_request,
    map_keycloak_auth_error as auth_service_map_keycloak_auth_error,
    upsert_keycloak_user as auth_service_upsert_keycloak_user,
)
from .services.history_service import (
    cleanup_expired_runtime_data,
)
from .services.glossary_service import (
    apply_shared_glossary_replacements,
    load_shared_glossary,
)
from .repositories import quota_repository, session_repository
from .services.oidc_service import (
    build_authorization_url as oidc_build_authorization_url,
    exchange_code as oidc_exchange_code,
    fetch_discovery as oidc_fetch_discovery,
    fetch_json as oidc_fetch_json,
    fetch_userinfo as oidc_fetch_userinfo,
)
from .summarizer import OpenAISummarizer
from .transcript_store import (
    read_jsonl_records,
)
from .transcription.factory import create_live_session
from .transcription.media import (
    _store_screenshot_for_chunk,
    _save_debug_audio_chunk,
    _store_debug_raw_audio_for_final,
    _store_debug_audio_for_final,
    _resolve_existing_debug_audio_url,
    _merge_wav_chunks,
)
from .transcription.messages import (
    as_int as _as_int,
    as_str as _as_str,
    chunk_payload_size_error as _chunk_payload_size_error,
    parse_chunk_message as _parse_chunk_message,
    validate_start_message as _validate_start_message,
    validate_telemetry_message as _validate_telemetry_message,
    validate_chunk_order as _validate_chunk_order,
)
from .transcription.session import ChunkMessage, LiveSession
from .transcription.text_processing import (
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
from .transcription.worker import WorkerDependencies, run_session_worker
from .core.security import (
    clear_oidc_state_cookie as security_clear_oidc_state_cookie,
    clear_session_cookie as security_clear_session_cookie,
    client_ip as security_client_ip,
    external_url_for as security_external_url_for,
    read_oidc_state_cookie as security_read_oidc_state_cookie,
    serialize_user as security_serialize_user,
    set_oidc_state_cookie as security_set_oidc_state_cookie,
    set_session_cookie as security_set_session_cookie,
    signed_payload as security_signed_payload,
    unsigned_payload as security_unsigned_payload,
)
from .core.logging import emit_container_log
from .core.rate_limit import consume as consume_rate_limit


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_DIARIZATION_SPEAKERS = 12


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str | None = None
    prompt: str | None = None


class ProofreadRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str | None = None
    mode: str | None = None


TRANSCRIBER_FACTORY: Callable[[], SessionTranscriber] | None = None
AUDIO_PREPROCESSOR: AudioPreprocessor | None = None
SUMMARIZER: OpenAISummarizer | None = None
PROOFREADER: OpenAISummarizer | None = None
DIARIZER: PyannoteSpeakerDiarizer | None = None
ACTIVE_SOCKETS: set[WebSocket] = set()
LANGFUSE_OBSERVER = None
CLEANUP_TASK: asyncio.Task | None = None
EVENT_LOOP_MONITOR_TASK: asyncio.Task | None = None
OIDC_STATE_COOKIE_NAME = "whistx_oidc_state"
KEYCLOAK_PROVIDER = "keycloak"
KEYCLOAK_DISCOVERY_CACHE: dict[str, Any] | None = None
CLEANUP_INTERVAL_SECONDS = 6 * 60 * 60


async def on_startup() -> None:
    global \
        TRANSCRIBER_FACTORY, \
        AUDIO_PREPROCESSOR, \
        SUMMARIZER, \
        PROOFREADER, \
        DIARIZER, \
        LANGFUSE_OBSERVER, \
        CLEANUP_TASK, \
        EVENT_LOOP_MONITOR_TASK

    _validate_runtime_configuration()
    _run_cleanup_once("startup")
    CLEANUP_TASK = asyncio.create_task(_periodic_cleanup_loop())
    EVENT_LOOP_MONITOR_TASK = asyncio.create_task(_event_loop_lag_monitor())
    logger.info(
        "startup config: history_dir=%s keycloak=%s self_signup=%s require_verified_email=%s",
        settings.history_dir,
        _keycloak_login_enabled(),
        settings.enable_self_signup,
        settings.keycloak_require_email_verified,
    )
    emit_container_log(
        __name__,
        "info",
        "startup config: history_dir=%s keycloak=%s self_signup=%s require_verified_email=%s",
        settings.history_dir,
        _keycloak_login_enabled(),
        settings.enable_self_signup,
        settings.keycloak_require_email_verified,
    )
    LANGFUSE_OBSERVER = make_langfuse_observer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        environment=settings.langfuse_environment,
        release=settings.langfuse_release,
        enabled=settings.langfuse_enabled,
    )

    TRANSCRIBER_FACTORY = _build_transcriber_factory()
    AUDIO_PREPROCESSOR = AudioPreprocessor(
        ffmpeg_bin=settings.ffmpeg_bin,
        sample_rate=settings.asr_preprocess_sample_rate,
        overlap_ms=settings.asr_overlap_ms,
        enabled=settings.asr_preprocess_enabled,
    )
    try:
        SUMMARIZER = OpenAISummarizer(
            api_key=settings.summary_api_key,
            base_url=settings.summary_base_url,
            model=settings.summary_model,
            temperature=settings.summary_temperature,
            summary_system_prompt=settings.summary_system_prompt,
            summary_prompt_template=settings.summary_prompt_template,
            proofread_system_prompt=settings.proofread_system_prompt,
            proofread_prompt_template=settings.proofread_prompt_template,
            timeout_seconds=settings.summary_api_timeout_seconds,
            observer=LANGFUSE_OBSERVER,
        )
    except Exception as exc:  # noqa: BLE001
        SUMMARIZER = None
        logger.warning("summary disabled: %s", exc)

    try:
        PROOFREADER = OpenAISummarizer(
            api_key=settings.proofread_api_key,
            base_url=settings.proofread_base_url,
            model=settings.proofread_model,
            temperature=settings.proofread_temperature,
            summary_system_prompt=settings.summary_system_prompt,
            summary_prompt_template=settings.summary_prompt_template,
            proofread_system_prompt=settings.proofread_system_prompt,
            proofread_prompt_template=settings.proofread_prompt_template,
            timeout_seconds=settings.proofread_api_timeout_seconds,
            observer=LANGFUSE_OBSERVER,
        )
    except Exception as exc:  # noqa: BLE001
        PROOFREADER = None
        logger.warning("proofread disabled: %s", exc)

    if settings.diarization_enabled:
        try:
            DIARIZER = PyannoteSpeakerDiarizer(
                hf_token=settings.diarization_hf_token,
                model=settings.diarization_model,
                ffmpeg_bin=settings.ffmpeg_bin,
                device=settings.diarization_device,
                sample_rate=settings.diarization_sample_rate,
                num_speakers=settings.diarization_num_speakers,
                min_speakers=settings.diarization_min_speakers,
                max_speakers=settings.diarization_max_speakers,
            )
            DIARIZER.preflight()
        except Exception as exc:  # noqa: BLE001
            DIARIZER = None
            logger.warning("diarization disabled: %s", exc)
    else:
        DIARIZER = None

    logger.info(
        "whistx started (model=%s, ws=%s)", settings.asr_model, settings.ws_path
    )
    emit_container_log(
        __name__,
        "info",
        "whistx started (model=%s, ws=%s)",
        settings.asr_model,
        settings.ws_path,
    )


async def on_shutdown() -> None:
    global CLEANUP_TASK, EVENT_LOOP_MONITOR_TASK
    for task in (CLEANUP_TASK, EVENT_LOOP_MONITOR_TASK):
        if task is None:
            continue
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    CLEANUP_TASK = None
    EVENT_LOOP_MONITOR_TASK = None
    if LANGFUSE_OBSERVER is not None:
        LANGFUSE_OBSERVER.flush()
        LANGFUSE_OBSERVER.shutdown()


def _run_cleanup_once(reason: str) -> None:
    try:
        with db_session() as db:
            cleanup_expired_runtime_data(db)
            deleted_sessions = 0
            oldest_expired_at = None
            for _ in range(10):
                result = session_repository.prune_expired_sessions(
                    db,
                    now=datetime.now(timezone.utc),
                    batch_size=1_000,
                )
                deleted_sessions += result.deleted_count
                oldest_expired_at = oldest_expired_at or result.oldest_expired_at
                if result.deleted_count < 1_000:
                    break
            if deleted_sessions:
                logger.info(
                    "expired session cleanup completed: reason=%s deleted=%d oldest_expired_at=%s",
                    reason,
                    deleted_sessions,
                    oldest_expired_at.isoformat() if oldest_expired_at else None,
                )
    except Exception:
        logger.warning(
            "scheduled cleanup failed during %s", reason, exc_info=True
        )


async def _periodic_cleanup_loop() -> None:
    try:
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            await asyncio.to_thread(_run_cleanup_once, "periodic cleanup")
    except asyncio.CancelledError:
        raise


async def _event_loop_lag_monitor() -> None:
    interval_seconds = 1.0
    loop = asyncio.get_running_loop()
    expected = loop.time() + interval_seconds
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            current = loop.time()
            lag_seconds = max(0.0, current - expected)
            if lag_seconds >= 0.25:
                logger.warning(
                    "event loop lag detected: lag_ms=%d",
                    round(lag_seconds * 1000),
                )
            expected = current + interval_seconds
    except asyncio.CancelledError:
        raise


async def health() -> JSONResponse:
    active_connections = await asyncio.to_thread(
        quota_repository.count_active_connections
    )
    return JSONResponse(
        {
            "status": "ok",
            "model": settings.asr_model,
            "asrReady": TRANSCRIBER_FACTORY is not None,
            "asrBackend": settings.asr_backend,
            "capturePacketMs": 250 if settings.asr_backend == "qwen3_vllm" else 1000,
            "summaryModel": settings.summary_model if SUMMARIZER else None,
            "proofreadModel": settings.proofread_model if PROOFREADER else None,
            "diarizationEnabled": DIARIZER is not None,
            "diarizationModel": settings.diarization_model if DIARIZER else None,
            "diarizationDefaultNumSpeakers": settings.diarization_num_speakers,
            "diarizationDefaultMinSpeakers": settings.diarization_min_speakers,
            "diarizationDefaultMaxSpeakers": settings.diarization_max_speakers,
            "diarizationSpeakerCap": MAX_DIARIZATION_SPEAKERS,
            "banners": list(settings.ui_banners),
            "uiBrandTitle": settings.app_brand_title,
            "uiBrandTagline": settings.app_brand_tagline,
            "uiPromptTemplates": list(settings.ui_prompt_templates),
            "selfSignupEnabled": settings.enable_self_signup,
            "keycloakEnabled": _keycloak_login_enabled(),
            "keycloakButtonLabel": settings.keycloak_button_label,
            "wsPath": settings.ws_path,
            "liveWsPath": settings.ws_path + "/live",
            "meetingInsights": True,
            "activeConnections": active_connections,
        }
    )


def _validate_runtime_configuration() -> None:
    if settings.app_session_secret.strip() == "change-me":
        logger.warning(
            "APP_SESSION_SECRET is using the default placeholder; set a strong secret before running in shared environments"
        )
    if settings.keycloak_enabled and not _keycloak_login_enabled():
        logger.warning(
            "KEYCLOAK_ENABLED is set but issuer/client_id is incomplete; keycloak login will be disabled"
        )


async def auth_keycloak_login(request: Request) -> Response:
    if not _keycloak_login_enabled():
        return JSONResponse(status_code=404, content={"error": "keycloak_disabled"})

    discovery = await asyncio.to_thread(_get_keycloak_discovery)
    state = secrets.token_urlsafe(24)
    code_verifier = secrets.token_urlsafe(48)
    code_challenge = _pkce_code_challenge(code_verifier)
    redirect_uri = security_external_url_for(request, "auth_keycloak_callback")
    authorization_url = _build_keycloak_authorization_url(
        discovery=discovery,
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
    )

    response = Response(status_code=302)
    response.headers["Location"] = authorization_url
    _set_oidc_state_cookie(
        response,
        request,
        {"state": state, "code_verifier": code_verifier, "redirect_uri": redirect_uri},
    )
    return response


async def auth_keycloak_callback(
    request: Request,
    db: Session = Depends(get_db),
) -> Response:
    if not _keycloak_login_enabled():
        return HTMLResponse(status_code=404, content="not found")

    state_payload = _read_oidc_state_cookie(request)
    _response = Response(status_code=302)
    _clear_oidc_state_cookie(_response, request)

    state = request.query_params.get("state") or ""
    code = request.query_params.get("code") or ""
    if not state_payload or not code or state_payload.get("state") != state:
        _response.headers["Location"] = "/?authError=keycloak_state"
        return _response

    try:
        discovery = await asyncio.to_thread(_get_keycloak_discovery)
        token_payload = await asyncio.to_thread(
            _exchange_keycloak_code,
            discovery,
            code,
            str(state_payload.get("redirect_uri") or ""),
            str(state_payload.get("code_verifier") or ""),
        )
        userinfo = await asyncio.to_thread(
            _fetch_keycloak_userinfo,
            discovery,
            str(token_payload.get("access_token") or ""),
        )
        user = _upsert_keycloak_user(db, userinfo)
        user.last_login_at = datetime.now(timezone.utc)
        session_id = create_user_session(
            db,
            user=user,
            user_agent=request.headers.get("user-agent"),
            ip_address=security_client_ip(request),
        )
        db.commit()
    except PermissionError:
        db.rollback()
        _response.headers["Location"] = "/?authError=approval_required"
        return _response
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.warning("keycloak login failed: %s", exc)
        _response.headers["Location"] = (
            f"/?authError={auth_service_map_keycloak_auth_error(exc)}"
        )
        return _response

    _set_session_cookie(_response, request, session_id)
    _response.headers["Location"] = "/"
    return _response


async def summarize(payload: SummarizeRequest) -> JSONResponse:
    if SUMMARIZER is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "summary_not_configured",
                "detail": "SUMMARY_API_KEY (or ASR_API_KEY / OPENAI_API_KEY) is missing",
            },
        )

    raw_text = payload.text.strip()
    if not raw_text:
        return JSONResponse(status_code=400, content={"error": "empty_text"})
    if len(raw_text) > settings.summary_input_max_chars:
        return JSONResponse(
            status_code=413,
            content={
                "error": "summary_input_too_large",
                "maxChars": settings.summary_input_max_chars,
            },
        )
    if len(_as_str(payload.prompt)) > settings.ws_prompt_max_chars:
        return JSONResponse(
            status_code=413,
            content={
                "error": "summary_prompt_too_large",
                "maxChars": settings.ws_prompt_max_chars,
            },
        )

    language = _as_str(payload.language) or settings.default_language
    prompt = _as_str(payload.prompt)
    trace_context = (
        LANGFUSE_OBSERVER.create_trace_context(
            name="api.summarize",
            input={
                "language": language,
                "chars": len(raw_text),
                "customPrompt": bool(prompt),
            },
        )
        if LANGFUSE_OBSERVER is not None
        else None
    )

    try:
        result = await blocking_work_pool.run(
            "llm",
            SUMMARIZER.summarize_long,
            text=raw_text,
            language=language,
            max_chars=settings.summary_input_max_chars,
            custom_template=prompt,
            trace_context=trace_context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Summary failed")
        return JSONResponse(
            status_code=502, content={"error": "summary_failed", "detail": str(exc)}
        )

    return JSONResponse(
        {
            "summary": result.text,
            "model": result.model,
            "inputChars": len(raw_text),
            "chunkCount": result.chunk_count,
            "reduced": result.reduced,
        }
    )


async def proofread(payload: ProofreadRequest) -> JSONResponse:
    if PROOFREADER is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "proofread_not_configured",
                "detail": "PROOFREAD_API_KEY / SUMMARY_API_KEY / ASR_API_KEY (or OPENAI_API_KEY) is missing",
            },
        )

    raw_text = payload.text.strip()
    if not raw_text:
        return JSONResponse(status_code=400, content={"error": "empty_text"})
    if len(raw_text) > settings.proofread_input_max_chars:
        return JSONResponse(
            status_code=413,
            content={
                "error": "proofread_input_too_large",
                "maxChars": settings.proofread_input_max_chars,
            },
        )

    language = _as_str(payload.language) or settings.default_language
    mode = _normalize_proofread_mode(_as_str(payload.mode))
    glossary_payload = await asyncio.to_thread(load_shared_glossary)
    glossary_text = str(glossary_payload.get("text") or "").strip()
    logger.info("Proofread requested: chars=%d language=%s", len(raw_text), language)
    trace_context = (
        LANGFUSE_OBSERVER.create_trace_context(
            name="api.proofread",
            input={"language": language, "chars": len(raw_text), "mode": mode},
        )
        if LANGFUSE_OBSERVER is not None
        else None
    )

    try:
        result = await blocking_work_pool.run(
            "llm",
            PROOFREADER.proofread_long,
            text=raw_text,
            language=language,
            max_chars=settings.proofread_input_max_chars,
            mode=mode,
            glossary_text=glossary_text,
            trace_context=trace_context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Proofread failed")
        return JSONResponse(
            status_code=502, content={"error": "proofread_failed", "detail": str(exc)}
        )

    corrected_text = apply_shared_glossary_replacements(result.text, glossary_text)
    return JSONResponse(
        {
            "corrected": corrected_text,
            "model": result.model,
            "inputChars": len(raw_text),
            "chunkCount": result.chunk_count,
            "reduced": result.reduced,
            "mode": mode,
        }
    )


async def proofread_stream(payload: ProofreadRequest) -> Response:
    if PROOFREADER is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "proofread_not_configured",
                "detail": "PROOFREAD_API_KEY / SUMMARY_API_KEY / ASR_API_KEY (or OPENAI_API_KEY) is missing",
            },
        )

    raw_text = payload.text.strip()
    if not raw_text:
        return JSONResponse(status_code=400, content={"error": "empty_text"})
    if len(raw_text) > settings.proofread_input_max_chars:
        return JSONResponse(
            status_code=413,
            content={
                "error": "proofread_input_too_large",
                "maxChars": settings.proofread_input_max_chars,
            },
        )

    language = _as_str(payload.language) or settings.default_language
    mode = _normalize_proofread_mode(_as_str(payload.mode))
    glossary_payload = await asyncio.to_thread(load_shared_glossary)
    glossary_text = str(glossary_payload.get("text") or "").strip()
    trace_context = (
        LANGFUSE_OBSERVER.create_trace_context(
            name="api.proofread.stream",
            input={"language": language, "chars": len(raw_text), "mode": mode},
        )
        if LANGFUSE_OBSERVER is not None
        else None
    )

    def event_stream():
        assembled_parts: list[str] = []
        try:
            for event in PROOFREADER.proofread_stream_long(
                text=raw_text,
                language=language,
                max_chars=settings.proofread_input_max_chars,
                mode=mode,
                glossary_text=glossary_text,
                trace_context=trace_context,
            ):
                if str(event.get("type") or "") == "delta":
                    assembled_parts.append(str(event.get("delta") or ""))
                yield _format_sse(event)
            original_text = "".join(assembled_parts).strip()
            corrected_text = apply_shared_glossary_replacements(
                original_text, glossary_text
            )
            if corrected_text and corrected_text != original_text:
                yield _format_sse({"type": "final_text", "text": corrected_text})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Proofread stream failed")
            yield _format_sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()
    ACTIVE_SOCKETS.add(ws)
    await _broadcast_conn_count()

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
                            "detail": str(exc),
                        },
                    )
                    continue
                worker_task = asyncio.create_task(_session_worker(ws, session))
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
                await _broadcast_conn_count()
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
            await _run_diarization_for_session(ws, session)
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

        ACTIVE_SOCKETS.discard(ws)
        await _broadcast_conn_count()


async def _session_worker(ws: WebSocket, session: LiveSession) -> None:
    await run_session_worker(
        ws,
        session,
        WorkerDependencies(
            settings=settings,
            observer=LANGFUSE_OBSERVER,
            logger=logger,
            prepare_audio=_prepare_audio_for_asr,
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
    owner_user_id: int | None = None,
    guest_grant_digest: str | None = None,
) -> LiveSession:
    if TRANSCRIBER_FACTORY is None:
        raise RuntimeError("transcriber_not_ready")
    return create_live_session(
        payload,
        settings=settings,
        transcriber_factory=TRANSCRIBER_FACTORY,
        diarizer_available=DIARIZER is not None,
        owner_user_id=owner_user_id,
        guest_grant_digest=guest_grant_digest,
    )


def _build_transcriber_factory() -> Callable[[], SessionTranscriber]:
    if settings.asr_backend == "qwen3_vllm":
        from .qwen_asr import QwenBatchTranscriber
        return lambda: QwenBatchTranscriber(api_key=settings.openai_api_key, base_url=settings.openai_base_url, model=settings.asr_model)
    if _use_realtime_asr(settings.asr_model):
        raise RuntimeError(
            "Realtime ASR models are not supported in the current build. Use a Whisper-compatible ASR_MODEL."
        )

    return lambda: OpenAIWhisperTranscriber(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.asr_model,
        observer=LANGFUSE_OBSERVER,
    )


def _use_realtime_asr(model: str) -> bool:
    lowered = (model or "").strip().lower()
    return "voxtral" in lowered and "realtime" in lowered


def _prepare_audio_for_asr(*, session: LiveSession, item: ChunkMessage):
    if AUDIO_PREPROCESSOR is None:
        raise RuntimeError("audio_preprocessor_not_ready")
    prepared = AUDIO_PREPROCESSOR.prepare(
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


async def _run_diarization_for_session(ws: WebSocket, session: LiveSession) -> None:
    if DIARIZER is None:
        session.store.cleanup_chunks()
        return
    if not session.collect_audio_for_diarization:
        session.store.cleanup_chunks()
        return
    if not session.audio_chunks:
        session.store.cleanup_chunks()
        return

    await _safe_send(
        ws,
        {
            "type": "info",
            "message": "diarization_started",
            "sessionId": session.session_id,
        },
    )

    try:
        patch_map = await blocking_work_pool.run(
            "diarization",
            _apply_diarization_labels,
            session,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Diarization failed: session=%s", session.session_id)
        await _safe_send(
            ws,
            {
                "type": "error",
                "message": "diarization_failed",
                "sessionId": session.session_id,
                "detail": str(exc),
            },
        )
        patch_map = {}

    if patch_map:
        payload = [
            {"seq": seq, "speaker": speaker}
            for seq, speaker in sorted(patch_map.items())
        ]
        await _safe_send(
            ws,
            {
                "type": "speaker_patch",
                "sessionId": session.session_id,
                "segments": payload,
            },
        )

    await _safe_send(
        ws,
        {
            "type": "info",
            "message": "diarization_done",
            "sessionId": session.session_id,
        },
    )
    if not settings.diarization_keep_chunks:
        session.store.cleanup_chunks()


def _apply_diarization_labels(session: LiveSession) -> dict[int, str]:
    if DIARIZER is None:
        return {}

    turns = DIARIZER.diarize(
        session_id=session.session_id,
        chunks=session.audio_chunks,
        work_dir=settings.diarization_work_dir,
        num_speakers=session.diarization_num_speakers,
        min_speakers=session.diarization_min_speakers,
        max_speakers=session.diarization_max_speakers,
    )
    if not turns:
        return {}

    records = read_jsonl_records(session.store.jsonl_path)
    if not records:
        return {}

    patch_map: dict[int, str] = {}
    updated = False

    for rec in records:
        if rec.get("type") != "final":
            continue
        start_ms = _as_int(rec.get("tsStart"), 0)
        end_ms = _as_int(rec.get("tsEnd"), start_ms)
        if end_ms < start_ms:
            end_ms = start_ms

        speaker = _pick_speaker(turns, start_ms, end_ms)
        if not speaker:
            continue

        current = str(rec.get("speaker", "")).strip()
        if current == speaker:
            continue

        rec["speaker"] = speaker
        seq = _as_int(rec.get("seq"), -1)
        if seq >= 0:
            patch_map[seq] = speaker
        updated = True

    if updated:
        session.store.rewrite_records(records)
        logger.info(
            "Diarization applied: session=%s speakers=%d segments=%d requested_num=%d requested_min=%d requested_max=%d",
            session.session_id,
            len({t.speaker for t in turns}),
            len(patch_map),
            session.diarization_num_speakers,
            session.diarization_min_speakers,
            session.diarization_max_speakers,
        )

    return patch_map


def _pick_speaker(turns: list[SpeakerTurn], start_ms: int, end_ms: int) -> str | None:
    if not turns:
        return None

    s = max(0, start_ms)
    e = max(s + 1, end_ms)
    overlap_by_speaker: dict[str, int] = {}

    for turn in turns:
        if turn.end_ms <= s:
            continue
        if turn.start_ms >= e:
            break

        overlap = min(e, turn.end_ms) - max(s, turn.start_ms)
        if overlap <= 0:
            continue
        overlap_by_speaker[turn.speaker] = (
            overlap_by_speaker.get(turn.speaker, 0) + overlap
        )

    if overlap_by_speaker:
        return max(overlap_by_speaker.items(), key=lambda item: item[1])[0]

    center = (s + e) // 2
    nearest: SpeakerTurn | None = None
    nearest_distance: int | None = None

    for turn in turns:
        turn_center = (turn.start_ms + turn.end_ms) // 2
        distance = abs(turn_center - center)
        if nearest is None or nearest_distance is None or distance < nearest_distance:
            nearest = turn
            nearest_distance = distance

    if nearest is None or nearest_distance is None or nearest_distance > 3_000:
        return None
    return nearest.speaker


def _normalize_proofread_mode(value: str) -> str:
    lowered = (value or "").strip().lower()
    if lowered in {"translate_ja", "translate_en"}:
        return lowered
    return "proofread"


def _clip_trace_text(text: str, limit: int = 8000) -> str:
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "...(truncated)"


@contextmanager
def _noop_span():
    yield None


async def _broadcast_conn_count() -> None:
    count = await asyncio.to_thread(quota_repository.count_active_connections)
    payload = {"type": "conn", "count": count}
    dead: list[WebSocket] = []

    for sock in list(ACTIVE_SOCKETS):
        ok = await _safe_send(sock, payload)
        if not ok:
            dead.append(sock)

    for sock in dead:
        ACTIVE_SOCKETS.discard(sock)


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


def _format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _keycloak_login_enabled() -> bool:
    return bool(
        settings.keycloak_enabled
        and settings.keycloak_issuer
        and settings.keycloak_client_id
    )


def _pkce_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _signed_payload(value: dict[str, Any]) -> str:
    return security_signed_payload(value)


def _unsigned_payload(value: str | None) -> dict[str, Any] | None:
    return security_unsigned_payload(value)


def _set_oidc_state_cookie(
    response: Response, request: Request, payload: dict[str, Any]
) -> None:
    security_set_oidc_state_cookie(
        response=response,
        request=request,
        cookie_name=OIDC_STATE_COOKIE_NAME,
        payload=payload,
    )


def _read_oidc_state_cookie(request: Request) -> dict[str, Any] | None:
    return security_read_oidc_state_cookie(
        request=request, cookie_name=OIDC_STATE_COOKIE_NAME
    )


def _clear_oidc_state_cookie(response: Response, request: Request) -> None:
    security_clear_oidc_state_cookie(
        response=response, request=request, cookie_name=OIDC_STATE_COOKIE_NAME
    )


def _fetch_json(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    return oidc_fetch_json(url, method=method, data=data, headers=headers)


def _get_keycloak_discovery() -> dict[str, Any]:
    global KEYCLOAK_DISCOVERY_CACHE
    if KEYCLOAK_DISCOVERY_CACHE is not None:
        return KEYCLOAK_DISCOVERY_CACHE
    KEYCLOAK_DISCOVERY_CACHE = oidc_fetch_discovery(
        str(settings.keycloak_issuer or ""),
        fetcher=_fetch_json,
    )
    return KEYCLOAK_DISCOVERY_CACHE


def _build_keycloak_authorization_url(
    *,
    discovery: dict[str, Any],
    redirect_uri: str,
    state: str,
    code_challenge: str,
) -> str:
    return oidc_build_authorization_url(
        authorization_endpoint=str(discovery["authorization_endpoint"]),
        client_id=settings.keycloak_client_id,
        redirect_uri=redirect_uri,
        scope=settings.keycloak_scope,
        state=state,
        code_challenge=code_challenge,
    )


def _exchange_keycloak_code(
    discovery: dict[str, Any],
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    return oidc_exchange_code(
        token_endpoint=str(discovery["token_endpoint"]),
        client_id=settings.keycloak_client_id,
        client_secret=settings.keycloak_client_secret,
        code=code,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        fetcher=_fetch_json,
    )


def _fetch_keycloak_userinfo(
    discovery: dict[str, Any], access_token: str
) -> dict[str, Any]:
    return oidc_fetch_userinfo(
        str(discovery["userinfo_endpoint"]),
        access_token,
        fetcher=_fetch_json,
    )


def _upsert_keycloak_user(db: Session, userinfo: dict[str, Any]) -> User:
    return auth_service_upsert_keycloak_user(db, userinfo)


def _mark_session_finalized(session: LiveSession) -> None:
    metadata = session.store.read_metadata()
    metadata["finalized"] = True
    metadata["finalizedAt"] = datetime.now(timezone.utc).isoformat()
    session.store.write_metadata(metadata)


def _serialize_user(user: User | None) -> dict[str, Any] | None:
    return security_serialize_user(user)


def _get_optional_user(request: Request, db: Session) -> User | None:
    return auth_service_get_optional_user_from_request(request, db)


def _set_session_cookie(response: Response, request: Request, session_id: str) -> None:
    security_set_session_cookie(
        response=response,
        request=request,
        cookie_name=SESSION_COOKIE_NAME,
        session_id=session_id,
    )


def _clear_session_cookie(response: Response, request: Request) -> None:
    security_clear_session_cookie(
        response=response, request=request, cookie_name=SESSION_COOKIE_NAME
    )
