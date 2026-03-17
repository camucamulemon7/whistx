from __future__ import annotations

import io
import logging
from contextlib import contextmanager
from typing import Any

from openai import OpenAI

from .asr import ASRChunkResult
from .langfuse_observer import LangfuseObserver


logger = logging.getLogger(__name__)


class OpenAIWhisperTranscriber:
    def __init__(self, *, api_key: str, base_url: str | None, model: str, observer: LangfuseObserver | None = None):
        if not api_key:
            raise RuntimeError("ASR_API_KEY (or OPENAI_API_KEY) is not set")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.model = model
        self.observer = observer

    def transcribe_chunk(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
        trace_context: dict[str, str] | None = None,
    ) -> ASRChunkResult:
        suffix = _ext_from_mime(mime_type)
        file_obj = io.BytesIO(audio_bytes)
        file_obj.name = f"chunk{suffix}"

        with (self.observer.generation(
            name="asr.transcription",
            model=self.model,
            input={
                "mimeType": mime_type,
                "language": language or "",
                "prompt": bool(prompt),
                "audioBytes": len(audio_bytes),
            },
            metadata={"endpoint": "audio.transcriptions", "responseFormat": "verbose_json"},
            model_parameters={"temperature": temperature},
            trace_context=trace_context,
        ) if self.observer else _noop_generation()) as generation:
            logger.info(
                "ASR POST /v1/audio/transcriptions: model=%s mime=%s bytes=%d language=%s prompt=%s",
                self.model,
                mime_type,
                len(audio_bytes),
                language or "",
                bool(prompt),
            )
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=file_obj,
                language=language or None,
                prompt=prompt or None,
                temperature=temperature,
                response_format="verbose_json",
            )

            text = _extract_text(response).strip()
            if _should_drop_as_silence(response, text):
                text = ""
            logger.info(
                "ASR POST /v1/audio/transcriptions done: model=%s chars=%d",
                self.model,
                len(text),
            )
            if generation is not None:
                generation.update(output={"text": text, "chars": len(text)})

        start_ms, end_ms = _extract_bounds_ms(response)
        return ASRChunkResult(text=text, start_ms=start_ms, end_ms=end_ms)

    def close(self) -> None:
        return None



def _extract_text(response: Any) -> str:
    if isinstance(response, dict):
        value = response.get("text", "")
        return value if isinstance(value, str) else str(value)

    value = getattr(response, "text", "")
    return value if isinstance(value, str) else str(value)



def _extract_bounds_ms(response: Any) -> tuple[int | None, int | None]:
    segments = _read_field(response, "segments")
    if not isinstance(segments, list) or not segments:
        return None, None

    first = segments[0]
    last = segments[-1]

    start_sec = _as_float(_read_field(first, "start"))
    end_sec = _as_float(_read_field(last, "end"))

    if start_sec is None:
        start_sec = 0.0

    if end_sec is None:
        duration_sec = _as_float(_read_field(last, "duration"))
        if duration_sec is not None:
            base_sec = _as_float(_read_field(last, "start")) or start_sec
            end_sec = base_sec + duration_sec

    start_ms = max(0, int(round(start_sec * 1000)))
    end_ms = None if end_sec is None else max(start_ms, int(round(end_sec * 1000)))
    return start_ms, end_ms



def _read_field(obj: Any, field: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)



def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def _ext_from_mime(mime_type: str) -> str:
    lowered = (mime_type or "").lower()
    if "wav" in lowered:
        return ".wav"
    if "webm" in lowered:
        return ".webm"
    if "ogg" in lowered or "opus" in lowered:
        return ".ogg"
    if "mp4" in lowered or "m4a" in lowered:
        return ".m4a"
    if "mpeg" in lowered or "mp3" in lowered:
        return ".mp3"
    return ".webm"


KNOWN_SILENCE_HALLUCINATIONS = {
    "ご清聴ありがとうございました",
    "ご視聴ありがとうございました",
    "ありがとうございました",
}


def _normalize_for_match(text: str) -> str:
    normalized = (
        text.replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
        .replace("。", "")
        .replace("、", "")
        .replace(".", "")
        .replace(",", "")
        .replace("！", "")
        .replace("!", "")
        .replace("？", "")
        .replace("?", "")
    )
    return normalized.strip()


def _should_drop_as_silence(response: Any, text: str) -> bool:
    clean = text.strip()
    if not clean:
        return True

    normalized = _normalize_for_match(clean)
    if normalized in KNOWN_SILENCE_HALLUCINATIONS:
        return True

    segments = _read_field(response, "segments")
    if not isinstance(segments, list) or not segments:
        return False

    no_speech_probs: list[float] = []
    for seg in segments:
        value = _as_float(_read_field(seg, "no_speech_prob"))
        if value is not None:
            no_speech_probs.append(value)

    if not no_speech_probs:
        return False

    # 無音寄り判定が高く、かつ短文なら無音ハルシネーションの可能性が高い。
    return max(no_speech_probs) >= 0.85 and len(normalized) <= 32

@contextmanager
def _noop_generation():
    yield None
