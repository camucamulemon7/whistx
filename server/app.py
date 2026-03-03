from __future__ import annotations

import asyncio
import base64
import difflib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .openai_whisper import OpenAIWhisperTranscriber
from .summarizer import OpenAISummarizer
from .transcript_store import (
    TranscriptRecord,
    TranscriptStore,
    format_srt,
    read_jsonl_records,
    resolve_transcript_path,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="whistx", version="2.0.0")


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
    language: str
    base_prompt: str
    temperature: float
    context_prompt_enabled: bool
    context_max_chars: int
    context_text: str
    last_emitted_text: str
    store: TranscriptStore
    queue: asyncio.Queue[ChunkMessage | None]


class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str | None = None


TRANSCRIBER: OpenAIWhisperTranscriber | None = None
SUMMARIZER: OpenAISummarizer | None = None
ACTIVE_SOCKETS: set[WebSocket] = set()


@app.on_event("startup")
async def on_startup() -> None:
    global TRANSCRIBER, SUMMARIZER

    TRANSCRIBER = OpenAIWhisperTranscriber(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.whisper_model,
    )
    try:
        SUMMARIZER = OpenAISummarizer(
            api_key=settings.summary_api_key,
            base_url=settings.summary_base_url,
            model=settings.summary_model,
            temperature=settings.summary_temperature,
        )
    except Exception as exc:  # noqa: BLE001
        SUMMARIZER = None
        logger.warning("summary disabled: %s", exc)

    logger.info("whistx started (model=%s, ws=%s)", settings.whisper_model, settings.ws_path)


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model": settings.whisper_model,
            "summaryModel": settings.summary_model if SUMMARIZER else None,
            "activeConnections": len(ACTIVE_SOCKETS),
        }
    )


@app.post("/api/summarize")
async def summarize(payload: SummarizeRequest) -> JSONResponse:
    if SUMMARIZER is None:
        return JSONResponse(
            status_code=503,
            content={"error": "summary_not_configured", "detail": "SUMMARY_API_KEY is missing"},
        )

    raw_text = payload.text.strip()
    if not raw_text:
        return JSONResponse(status_code=400, content={"error": "empty_text"})

    truncated = False
    text = raw_text
    if len(text) > settings.summary_input_max_chars:
        text = text[-settings.summary_input_max_chars :]
        truncated = True

    language = _as_str(payload.language) or settings.default_language

    try:
        result = await asyncio.to_thread(SUMMARIZER.summarize, text=text, language=language)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Summary failed")
        return JSONResponse(status_code=502, content={"error": "summary_failed", "detail": str(exc)})

    return JSONResponse(
        {
            "summary": result.text,
            "model": result.model,
            "inputChars": len(text),
            "truncated": truncated,
        }
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

                session = _create_session(data)
                worker_task = asyncio.create_task(_session_worker(ws, session))

                await _safe_send(
                    ws,
                    {
                        "type": "info",
                        "message": "ready",
                        "state": "ready",
                        "sessionId": session.session_id,
                        "backend": f"openai:{settings.whisper_model}",
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

        ACTIVE_SOCKETS.discard(ws)
        await _broadcast_conn_count()


async def _session_worker(ws: WebSocket, session: LiveSession) -> None:
    if TRANSCRIBER is None:
        await _safe_send(ws, {"type": "error", "message": "transcriber_not_ready"})
        return

    while True:
        item = await session.queue.get()
        if item is None:
            break

        try:
            prompt = _build_prompt(session)
            result = await asyncio.to_thread(
                TRANSCRIBER.transcribe_chunk,
                item.audio_bytes,
                mime_type=item.mime_type,
                language=session.language,
                prompt=prompt,
                temperature=session.temperature,
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
        text = _sanitize_transcript_text(text)
        if not text:
            await _safe_send(ws, {"type": "ack", "seq": item.seq, "empty": True})
            continue

        if _is_near_duplicate(text, session.last_emitted_text):
            await _safe_send(
                ws,
                {"type": "ack", "seq": item.seq, "duplicate": True},
            )
            continue

        ts_start = item.offset_ms + (result.start_ms or 0)
        if result.end_ms is not None:
            ts_end = item.offset_ms + result.end_ms
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
        _append_context(session, text)

        await _safe_send(
            ws,
            {
                "type": "final",
                "segmentId": record.segmentId,
                "seq": record.seq,
                "text": record.text,
                "tsStart": record.tsStart,
                "tsEnd": record.tsEnd,
            },
        )



def _create_session(payload: dict[str, Any]) -> LiveSession:
    base_session_id = TranscriptStore.sanitize_or_generate(_as_str(payload.get("sessionId")))
    runtime_session_id = TranscriptStore.make_runtime_session_id(base_session_id)

    language = _as_str(payload.get("language")) or settings.default_language
    prompt = _as_str(payload.get("prompt")) or settings.default_prompt
    temperature = _as_float(payload.get("temperature"), settings.default_temperature)

    return LiveSession(
        session_id=runtime_session_id,
        language=language,
        base_prompt=prompt,
        temperature=temperature,
        context_prompt_enabled=settings.context_prompt_enabled,
        context_max_chars=settings.context_max_chars,
        context_text="",
        last_emitted_text="",
        store=TranscriptStore(settings.transcripts_dir, runtime_session_id),
        queue=asyncio.Queue(maxsize=settings.max_queue_size),
    )


def _build_prompt(session: LiveSession) -> str | None:
    parts: list[str] = []
    if session.base_prompt:
        parts.append(session.base_prompt.strip())

    if session.context_prompt_enabled and session.context_text:
        if session.language.lower().startswith("en"):
            header = "Recent transcript context:"
        else:
            header = "直前の文字起こし文脈:"
        parts.append(f"{header}\n{session.context_text}")

    merged = "\n\n".join(part for part in parts if part).strip()
    return merged or None


def _append_context(session: LiveSession, text: str) -> None:
    if not session.context_prompt_enabled:
        return
    if session.context_max_chars <= 0:
        return

    cleaned = " ".join(text.split()).strip()
    cleaned = _sanitize_transcript_text(cleaned)
    if not cleaned:
        return

    if session.context_text:
        merged = f"{session.context_text}\n{cleaned}"
    else:
        merged = cleaned

    if len(merged) > session.context_max_chars:
        merged = merged[-session.context_max_chars :].lstrip()
    session.context_text = merged


REPEAT_COLLAPSE_RE = re.compile(r"(.{2,16}?)\1{3,}")
REPEAT_DETECT_RE = re.compile(r"(.{2,16}?)\1{5,}")


def _sanitize_transcript_text(text: str) -> str:
    value = " ".join((text or "").split()).strip()
    if not value:
        return ""

    # 連続反復を縮約し、意味の薄い暴走出力を抑える。
    for _ in range(3):
        collapsed = REPEAT_COLLAPSE_RE.sub(lambda m: m.group(1) * 2, value)
        if collapsed == value:
            break
        value = collapsed

    if _is_repetition_noise(value):
        return ""
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
