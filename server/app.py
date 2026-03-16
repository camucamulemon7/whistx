from __future__ import annotations

import asyncio
import base64
import difflib
import json
import logging
import re
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .asr import SessionTranscriber
from .audio_pipeline import AudioPreprocessor
from .config import settings
from .diarizer import AudioChunk, PyannoteSpeakerDiarizer, SpeakerTurn
from .langfuse_observer import make_langfuse_observer
from .openai_whisper import OpenAIWhisperTranscriber
from .summarizer import OpenAISummarizer
from .transcript_store import (
    TranscriptRecord,
    TranscriptStore,
    format_srt,
    read_jsonl_records,
    resolve_transcript_path,
)
from .voxtral_realtime import VoxtralRealtimeTranscriber


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="whistx", version="2.0.0")
MAX_DIARIZATION_SPEAKERS = 12


@dataclass(slots=True)
class ChunkMessage:
    seq: int
    offset_ms: int
    duration_ms: int
    mime_type: str
    audio_bytes: bytes


@dataclass(slots=True)
class LiveSession:
    session_id: str
    language: str | None
    audio_source: str
    base_prompt: str
    temperature: float
    context_prompt_enabled: bool
    context_max_chars: int
    context_recent_lines: int
    context_term_limit: int
    context_history: list[str]
    transcript_history: list[str]
    context_terms: list[str]
    last_emitted_text: str
    collect_audio_for_diarization: bool
    diarization_num_speakers: int
    diarization_min_speakers: int
    diarization_max_speakers: int
    audio_chunks: list[AudioChunk]
    store: TranscriptStore
    queue: asyncio.Queue[ChunkMessage | None]
    transcriber: SessionTranscriber
    overlap_tail_pcm: bytes


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


@app.on_event("startup")
async def on_startup() -> None:
    global TRANSCRIBER_FACTORY, AUDIO_PREPROCESSOR, SUMMARIZER, PROOFREADER, DIARIZER, LANGFUSE_OBSERVER

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

    logger.info("whistx started (model=%s, ws=%s)", settings.asr_model, settings.ws_path)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if LANGFUSE_OBSERVER is not None:
        LANGFUSE_OBSERVER.flush()
        LANGFUSE_OBSERVER.shutdown()


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model": settings.asr_model,
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
            "activeConnections": len(ACTIVE_SOCKETS),
        }
    )


@app.post("/api/summarize")
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

    language = _as_str(payload.language) or settings.default_language
    prompt = _as_str(payload.prompt)
    trace_context = LANGFUSE_OBSERVER.create_trace_context(
        name="api.summarize",
        input={"language": language, "chars": len(raw_text), "customPrompt": bool(prompt)},
    ) if LANGFUSE_OBSERVER is not None else None

    try:
        result = await asyncio.to_thread(
            SUMMARIZER.summarize_long,
            text=raw_text,
            language=language,
            max_chars=settings.summary_input_max_chars,
            custom_template=prompt,
            trace_context=trace_context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Summary failed")
        return JSONResponse(status_code=502, content={"error": "summary_failed", "detail": str(exc)})

    return JSONResponse(
        {
            "summary": result.text,
            "model": result.model,
            "inputChars": len(raw_text),
            "chunkCount": result.chunk_count,
            "reduced": result.reduced,
        }
    )


@app.post("/api/proofread")
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

    language = _as_str(payload.language) or settings.default_language
    mode = _normalize_proofread_mode(_as_str(payload.mode))
    logger.info("Proofread requested: chars=%d language=%s", len(raw_text), language)
    trace_context = LANGFUSE_OBSERVER.create_trace_context(
        name="api.proofread",
        input={"language": language, "chars": len(raw_text), "mode": mode},
    ) if LANGFUSE_OBSERVER is not None else None

    try:
        result = await asyncio.to_thread(
            PROOFREADER.proofread_long,
            text=raw_text,
            language=language,
            max_chars=settings.proofread_input_max_chars,
            mode=mode,
            trace_context=trace_context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Proofread failed")
        return JSONResponse(status_code=502, content={"error": "proofread_failed", "detail": str(exc)})

    return JSONResponse(
        {
            "corrected": result.text,
            "model": result.model,
            "inputChars": len(raw_text),
            "chunkCount": result.chunk_count,
            "reduced": result.reduced,
            "mode": mode,
        }
    )


@app.post("/api/proofread/stream")
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

    language = _as_str(payload.language) or settings.default_language
    mode = _normalize_proofread_mode(_as_str(payload.mode))
    trace_context = LANGFUSE_OBSERVER.create_trace_context(
        name="api.proofread.stream",
        input={"language": language, "chars": len(raw_text), "mode": mode},
    ) if LANGFUSE_OBSERVER is not None else None

    def event_stream():
        try:
            for event in PROOFREADER.proofread_stream_long(
                text=raw_text,
                language=language,
                max_chars=settings.proofread_input_max_chars,
                mode=mode,
                trace_context=trace_context,
            ):
                yield _format_sse(event)
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


@app.websocket(settings.ws_path)
async def ws_transcribe(ws: WebSocket) -> None:
    await ws.accept()
    ACTIVE_SOCKETS.add(ws)
    await _broadcast_conn_count()

    session: LiveSession | None = None
    worker_task: asyncio.Task[None] | None = None

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await _safe_send(ws, {"type": "error", "message": "invalid_json"})
                continue

            if not isinstance(data, dict):
                await _safe_send(ws, {"type": "error", "message": "invalid_payload"})
                continue

            msg_type = str(data.get("type", "")).strip().lower()

            if msg_type == "start":
                if session is not None:
                    await _safe_send(ws, {"type": "error", "message": "already_started"})
                    continue

                try:
                    session = _create_session(data)
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

                chunk = _parse_chunk_message(data)
                if chunk is None:
                    await _safe_send(ws, {"type": "error", "message": "invalid_chunk"})
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

                try:
                    session.queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    await _safe_send(
                        ws,
                        {
                            "type": "error",
                            "message": "server_busy",
                            "detail": "queue_full",
                        },
                    )
                continue

            if msg_type == "stop":
                await _safe_send(ws, {"type": "info", "message": "stopping"})
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

        ACTIVE_SOCKETS.discard(ws)
        await _broadcast_conn_count()


async def _session_worker(ws: WebSocket, session: LiveSession) -> None:
    emitted_segments = 0
    emitted_chars = 0
    trace_context = LANGFUSE_OBSERVER.create_trace_context(
        name="asr.session",
        input={
            "sessionId": session.session_id,
            "language": session.language or "",
            "audioSource": session.audio_source,
            "model": settings.asr_model,
            "diarizationEnabled": session.collect_audio_for_diarization,
        },
    ) if LANGFUSE_OBSERVER is not None else None
    try:
        while True:
            item = await session.queue.get()
            if item is None:
                break

            if session.collect_audio_for_diarization:
                try:
                    chunk_path = session.store.save_audio_chunk(
                        seq=item.seq,
                        mime_type=item.mime_type,
                        audio_bytes=item.audio_bytes,
                    )
                    session.audio_chunks.append(
                        AudioChunk(
                            seq=item.seq,
                            path=chunk_path,
                            offset_ms=item.offset_ms,
                            duration_ms=item.duration_ms,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Chunk save failed for diarization: session=%s seq=%s err=%s",
                        session.session_id,
                        item.seq,
                        exc,
                    )

            try:
                prepared = _prepare_audio_for_asr(session=session, item=item)
                prompt = _build_prompt(session)
                result = await asyncio.to_thread(
                    session.transcriber.transcribe_chunk,
                    prepared.audio_bytes,
                    mime_type=prepared.mime_type,
                    language=session.language,
                    prompt=prompt,
                    temperature=session.temperature,
                    trace_context=trace_context,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Transcription failed: session=%s seq=%s", session.session_id, item.seq)
                await _safe_send(
                    ws,
                    {
                        "type": "error",
                        "message": "transcription_failed",
                        "seq": item.seq,
                        "detail": str(exc),
                    },
                )
                continue

            text = result.text.strip()
            text = _sanitize_transcript_text(text, language=session.language)
            if not text:
                await _safe_send(ws, {"type": "ack", "seq": item.seq, "empty": True})
                continue

            if _is_near_duplicate(text, session.last_emitted_text):
                await _safe_send(
                    ws,
                    {"type": "ack", "seq": item.seq, "duplicate": True},
                )
                continue

            ts_base_offset = max(0, item.offset_ms - prepared.overlap_ms_used)
            ts_start = ts_base_offset + (result.start_ms or 0)
            if result.end_ms is not None:
                ts_end = ts_base_offset + result.end_ms
            else:
                ts_end = item.offset_ms + max(item.duration_ms, 600)

            if ts_end < ts_start:
                ts_end = ts_start

            record = TranscriptRecord(
                type="final",
                segmentId=f"{item.seq:06d}",
                seq=item.seq,
                text=text,
                tsStart=ts_start,
                tsEnd=ts_end,
                chunkOffsetMs=item.offset_ms,
                chunkDurationMs=item.duration_ms,
                language=session.language,
                createdAt=datetime.now(timezone.utc).isoformat(),
            )
            session.store.append_final(record)
            session.last_emitted_text = text
            session.transcript_history.append(text)
            _append_context(session, text)
            emitted_segments += 1
            emitted_chars += len(text)

            await _safe_send(
                ws,
                {
                    "type": "final",
                    "segmentId": record.segmentId,
                    "seq": record.seq,
                    "text": record.text,
                    "tsStart": record.tsStart,
                    "tsEnd": record.tsEnd,
                    "speaker": record.speaker,
                },
            )

        if LANGFUSE_OBSERVER is not None and trace_context is not None:
            with LANGFUSE_OBSERVER.generation(
                name="asr.session.result",
                model=settings.asr_model,
                output={
                    "segmentCount": emitted_segments,
                    "charCount": emitted_chars,
                    "finalTranscript": _clip_trace_text("\n".join(session.transcript_history)),
                },
                trace_context=trace_context,
            ):
                pass
    finally:
        try:
            await asyncio.to_thread(session.transcriber.close)
        except Exception:  # noqa: BLE001
            logger.debug("session transcriber close failed: session=%s", session.session_id, exc_info=True)



def _create_session(payload: dict[str, Any]) -> LiveSession:
    if TRANSCRIBER_FACTORY is None:
        raise RuntimeError("transcriber_not_ready")

    base_session_id = TranscriptStore.sanitize_or_generate(_as_str(payload.get("sessionId")))
    runtime_session_id = TranscriptStore.make_runtime_session_id(base_session_id)

    language = _normalize_asr_language(_as_str(payload.get("language")))
    audio_source = _normalize_audio_source(_as_str(payload.get("audioSource")))
    prompt = _as_str(payload.get("prompt")) or settings.default_prompt
    temperature = _as_float(payload.get("temperature"), settings.default_temperature)
    diarization_requested = _as_bool(payload.get("diarizationEnabled"), True)
    diarization_num_speakers, diarization_min_speakers, diarization_max_speakers = (
        _parse_diarization_speaker_params(payload)
    )

    return LiveSession(
        session_id=runtime_session_id,
        language=language,
        audio_source=audio_source,
        base_prompt=prompt,
        temperature=temperature,
        context_prompt_enabled=settings.context_prompt_enabled,
        context_max_chars=settings.context_max_chars,
        context_recent_lines=settings.context_recent_lines,
        context_term_limit=settings.context_term_limit,
        context_history=[],
        transcript_history=[],
        context_terms=[],
        last_emitted_text="",
        collect_audio_for_diarization=DIARIZER is not None and diarization_requested,
        diarization_num_speakers=diarization_num_speakers,
        diarization_min_speakers=diarization_min_speakers,
        diarization_max_speakers=diarization_max_speakers,
        audio_chunks=[],
        store=TranscriptStore(settings.transcripts_dir, runtime_session_id),
        queue=asyncio.Queue(maxsize=settings.max_queue_size),
        transcriber=TRANSCRIBER_FACTORY(),
        overlap_tail_pcm=b"",
    )


def _build_transcriber_factory() -> Callable[[], SessionTranscriber]:
    if _use_realtime_asr(settings.asr_model):
        return lambda: VoxtralRealtimeTranscriber(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.asr_model,
            ffmpeg_bin=settings.ffmpeg_bin,
            observer=LANGFUSE_OBSERVER,
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
    )
    session.overlap_tail_pcm = prepared.tail_pcm
    return prepared


def _build_prompt(session: LiveSession) -> str | None:
    parts: list[str] = []
    if session.base_prompt:
        parts.append(session.base_prompt.strip())

    if session.context_prompt_enabled and (session.context_history or session.context_terms):
        if (session.language or "").lower().startswith("en"):
            header = "Recent transcript context:"
            terms_header = "Key terms:"
        elif not session.language:
            header = "Recent transcript context. Keep the same spoken language as the audio:"
            terms_header = "Key terms from recent transcript:"
        else:
            header = "直前の文字起こし文脈:"
            terms_header = "直前の重要語:"
        if session.context_history:
            parts.append(f"{header}\n" + "\n".join(session.context_history))
        if session.context_terms:
            parts.append(f"{terms_header}\n" + ", ".join(session.context_terms))

    merged = "\n\n".join(part for part in parts if part).strip()
    if session.context_max_chars > 0 and len(merged) > session.context_max_chars:
        merged = merged[-session.context_max_chars :].lstrip()
    return merged or None


def _append_context(session: LiveSession, text: str) -> None:
    if not session.context_prompt_enabled:
        return
    if session.context_max_chars <= 0:
        return

    cleaned = " ".join(text.split()).strip()
    cleaned = _sanitize_transcript_text(cleaned, language=session.language)
    if not cleaned:
        return

    session.context_history.append(cleaned)
    session.context_history = session.context_history[-session.context_recent_lines :]

    merged_terms = _merge_context_terms(
        existing=session.context_terms,
        new_terms=_extract_context_terms(cleaned),
        limit=session.context_term_limit,
    )
    session.context_terms = _trim_context_terms_to_budget(
        terms=merged_terms,
        max_chars=session.context_max_chars,
        history=session.context_history,
    )


CONTEXT_LATIN_TERM_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9.+/_-]{1,31}\b")
CONTEXT_KATAKANA_TERM_RE = re.compile(r"[ァ-ヶー]{3,}")
CONTEXT_CJK_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,12}")


def _extract_context_terms(text: str) -> list[str]:
    tokens: list[str] = []
    for pattern in (CONTEXT_LATIN_TERM_RE, CONTEXT_KATAKANA_TERM_RE, CONTEXT_CJK_TERM_RE):
        for match in pattern.finditer(text):
            token = match.group(0).strip(".,:;()[]{}<>\"'")
            if len(token) < 2:
                continue
            if token.isdigit():
                continue
            if token.lower() in {"recent", "transcript", "context"}:
                continue
            tokens.append(token)

    ranked = sorted(
        set(tokens),
        key=lambda item: (
            0 if re.search(r"[A-Z0-9]", item) else 1,
            -len(item),
            item.lower(),
        ),
    )
    return ranked


def _merge_context_terms(*, existing: list[str], new_terms: list[str], limit: int) -> list[str]:
    merged = list(existing)
    for term in new_terms:
        merged = [item for item in merged if item != term]
        merged.append(term)
    return merged[-limit:]


def _trim_context_terms_to_budget(
    *,
    terms: list[str],
    max_chars: int,
    history: list[str],
) -> list[str]:
    if max_chars <= 0:
        return terms

    history_text = "\n".join(history)
    budget = max(160, max_chars // 2) - len(history_text)
    if budget <= 0:
        return []

    kept: list[str] = []
    used = 0
    for term in reversed(terms):
        add = len(term) + (2 if kept else 0)
        if used + add > budget:
            continue
        kept.append(term)
        used += add
    kept.reverse()
    return kept


REPEAT_COLLAPSE_RE = re.compile(r"(.{2,16}?)\1{3,}")
REPEAT_DETECT_RE = re.compile(r"(.{2,16}?)\1{5,}")
JP_CHAR_CLASS = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
JP_SPACE_BEFORE_RE = re.compile(rf"(?<=[{JP_CHAR_CLASS}])\s+(?=[{JP_CHAR_CLASS}])")
JP_PUNCT_SPACE_RE = re.compile(rf"\s+([、。，．・：；！？）］】」』])|([（［【「『])\s+")


def _sanitize_transcript_text(text: str, *, language: str | None = None) -> str:
    value = " ".join((text or "").split()).strip()
    if not value:
        return ""

    value = _normalize_transcript_spacing(value, language=language)

    # 連続反復を縮約し、意味の薄い暴走出力を抑える。
    for _ in range(3):
        collapsed = REPEAT_COLLAPSE_RE.sub(lambda m: m.group(1) * 2, value)
        if collapsed == value:
            break
        value = collapsed

    if _is_repetition_noise(value):
        return ""
    return value


def _normalize_transcript_spacing(text: str, *, language: str | None) -> str:
    lowered = (language or "").strip().lower()
    if lowered and not lowered.startswith("ja"):
        return text

    value = JP_SPACE_BEFORE_RE.sub("", text)

    def _punct_repl(match: re.Match[str]) -> str:
        if match.group(1):
            return match.group(1)
        return match.group(2)

    value = JP_PUNCT_SPACE_RE.sub(_punct_repl, value)
    return value


def _normalize_compare_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def _is_repetition_noise(text: str) -> bool:
    normalized = _normalize_compare_text(text)
    if len(normalized) < 32:
        return False

    matched = REPEAT_DETECT_RE.search(normalized)
    if not matched:
        return False

    run_len = len(matched.group(0))
    # 1箇所の反復だけで大半を占める場合はノイズ扱い。
    return run_len >= max(36, int(len(normalized) * 0.45))


def _is_near_duplicate(current: str, previous: str) -> bool:
    a = _normalize_compare_text(current)
    b = _normalize_compare_text(previous)
    if not a or not b:
        return False

    if a == b:
        return True

    shorter = min(len(a), len(b))
    longer = max(len(a), len(b))
    if shorter >= 24 and shorter / longer >= 0.8 and (a in b or b in a):
        return True

    return shorter >= 24 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.93


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

    await _safe_send(ws, {"type": "info", "message": "diarization_started"})

    try:
        patch_map = await asyncio.to_thread(_apply_diarization_labels, session)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Diarization failed: session=%s", session.session_id)
        await _safe_send(
            ws,
            {
                "type": "error",
                "message": "diarization_failed",
                "detail": str(exc),
            },
        )
        patch_map = {}

    if patch_map:
        payload = [{"seq": seq, "speaker": speaker} for seq, speaker in sorted(patch_map.items())]
        await _safe_send(
            ws,
            {
                "type": "speaker_patch",
                "segments": payload,
            },
        )

    await _safe_send(ws, {"type": "info", "message": "diarization_done"})
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
        overlap_by_speaker[turn.speaker] = overlap_by_speaker.get(turn.speaker, 0) + overlap

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



def _parse_chunk_message(payload: dict[str, Any]) -> ChunkMessage | None:
    audio_b64 = _as_str(payload.get("audio"))
    if not audio_b64:
        return None

    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception:  # noqa: BLE001
        return None

    mime_type = _as_str(payload.get("mimeType")) or "audio/webm"
    seq = _as_int(payload.get("seq"), 0)
    offset_ms = max(0, _as_int(payload.get("offsetMs"), 0))
    duration_ms = max(200, _as_int(payload.get("durationMs"), 2000))

    return ChunkMessage(
        seq=seq,
        offset_ms=offset_ms,
        duration_ms=duration_ms,
        mime_type=mime_type,
        audio_bytes=audio_bytes,
    )


def _clamp_diarization_speakers(value: int) -> int:
    return max(0, min(MAX_DIARIZATION_SPEAKERS, value))


def _parse_diarization_speaker_params(payload: dict[str, Any]) -> tuple[int, int, int]:
    num = _clamp_diarization_speakers(
        _as_int(payload.get("diarizationNumSpeakers"), settings.diarization_num_speakers)
    )
    min_speakers = _clamp_diarization_speakers(
        _as_int(payload.get("diarizationMinSpeakers"), settings.diarization_min_speakers)
    )
    max_speakers = _clamp_diarization_speakers(
        _as_int(payload.get("diarizationMaxSpeakers"), settings.diarization_max_speakers)
    )

    if num > 0:
        return (num, 0, 0)

    if min_speakers > 0 and max_speakers > 0 and min_speakers > max_speakers:
        min_speakers, max_speakers = max_speakers, min_speakers

    return (0, min_speakers, max_speakers)


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_asr_language(value: str) -> str | None:
    lowered = (value or "").strip().lower()
    if not lowered or lowered == "auto":
        return None
    return lowered


def _normalize_audio_source(value: str) -> str:
    lowered = (value or "").strip().lower()
    if lowered in {"display", "both"}:
        return lowered
    return "mic"


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
    payload = {"type": "conn", "count": len(ACTIVE_SOCKETS)}
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


def _format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/transcript/{session_id}.txt", response_model=None)
async def get_txt(session_id: str) -> Response:
    path = resolve_transcript_path(settings.transcripts_dir, session_id, "txt")
    if not path or not path.exists():
        return HTMLResponse(status_code=404, content="not found")
    return FileResponse(str(path), media_type="text/plain")


@app.get("/api/transcript/{session_id}.jsonl", response_model=None)
async def get_jsonl(session_id: str) -> Response:
    path = resolve_transcript_path(settings.transcripts_dir, session_id, "jsonl")
    if not path or not path.exists():
        return HTMLResponse(status_code=404, content="not found")
    return FileResponse(str(path), media_type="application/x-ndjson")


@app.get("/api/transcript/{session_id}.srt", response_model=None)
async def get_srt(session_id: str) -> Response:
    path = resolve_transcript_path(settings.transcripts_dir, session_id, "jsonl")
    if not path or not path.exists():
        return HTMLResponse(status_code=404, content="not found")

    records = read_jsonl_records(path)
    srt = format_srt(records)
    return PlainTextResponse(status_code=200, content=srt, media_type="text/plain; charset=utf-8")


WEB_DIR = Path("web")
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
