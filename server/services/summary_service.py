from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..core.blocking import blocking_work_pool
from ..core.config import settings
from ..core.public_errors import public_error
from ..langfuse_observer import LangfuseObserver
from ..schemas import ProofreadRequest, SummarizeRequest
from ..summarizer import OpenAISummarizer
from ..transcription.messages import as_str as _as_str
from .glossary_service import apply_shared_glossary_replacements, load_shared_glossary

logger = logging.getLogger(__name__)


async def summarize(payload: SummarizeRequest, *, summarizer: OpenAISummarizer | None, observer: LangfuseObserver | None = None) -> JSONResponse:
    if summarizer is None:
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
        observer.create_trace_context(
            name="api.summarize",
            input={
                "language": language,
                "chars": len(raw_text),
                "customPrompt": bool(prompt),
            },
        )
        if observer is not None
        else None
    )

    try:
        result = await blocking_work_pool.run(
            "llm",
            summarizer.summarize_long,
            text=raw_text,
            language=language,
            max_chars=settings.summary_input_max_chars,
            custom_template=prompt,
            trace_context=trace_context,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Summary failed")
        return JSONResponse(
            status_code=502, content=public_error("summary_failed", exc, logger)
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



async def proofread(payload: ProofreadRequest, *, proofreader: OpenAISummarizer | None, observer: LangfuseObserver | None = None) -> JSONResponse:
    if proofreader is None:
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
        observer.create_trace_context(
            name="api.proofread",
            input={"language": language, "chars": len(raw_text), "mode": mode},
        )
        if observer is not None
        else None
    )

    try:
        result = await blocking_work_pool.run(
            "llm",
            proofreader.proofread_long,
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
            status_code=502, content=public_error("proofread_failed", exc, logger)
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



async def proofread_stream(payload: ProofreadRequest, *, proofreader: OpenAISummarizer | None, observer: LangfuseObserver | None = None) -> Response:
    if proofreader is None:
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
        observer.create_trace_context(
            name="api.proofread.stream",
            input={"language": language, "chars": len(raw_text), "mode": mode},
        )
        if observer is not None
        else None
    )

    def event_stream():
        assembled_parts: list[str] = []
        try:
            for event in proofreader.proofread_stream_long(
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
            yield _format_sse({"type": "error", **public_error("proofread_failed", exc, logger)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )



def _normalize_proofread_mode(value: str) -> str:
    lowered = (value or "").strip().lower()
    if lowered in {"translate_ja", "translate_en"}:
        return lowered
    return "proofread"



def _format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
