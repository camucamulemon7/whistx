from __future__ import annotations

import io
import logging
import time
from contextlib import contextmanager
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from .asr import ASRChunkResult
from .config import settings
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
        self.retry_max_attempts = max(1, int(getattr(settings, "asr_retry_max_attempts", 1)))
        self.retry_base_delay_ms = max(0, int(getattr(settings, "asr_retry_base_delay_ms", 0)))
        self.multi_pass_enabled = bool(getattr(settings, "asr_multi_pass_enabled", True))

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
        response: Any | None = None
        attempt = 0

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
            def _perform_request(request_temperature: float) -> Any:
                local_attempt = 0
                while True:
                    local_attempt += 1
                    file_obj = io.BytesIO(audio_bytes)
                    file_obj.name = f"chunk{suffix}"
                    logger.info(
                        "ASR POST /v1/audio/transcriptions: model=%s mime=%s bytes=%d language=%s prompt=%s attempt=%d/%d temp=%.2f",
                        self.model,
                        mime_type,
                        len(audio_bytes),
                        language or "",
                        bool(prompt),
                        local_attempt,
                        self.retry_max_attempts,
                        request_temperature,
                    )
                    try:
                        return self.client.audio.transcriptions.create(
                            model=self.model,
                            file=file_obj,
                            language=language or None,
                            prompt=prompt or None,
                            temperature=request_temperature,
                            response_format="verbose_json",
                        ), local_attempt
                    except Exception as exc:  # noqa: BLE001
                        if not _should_retry_openai_error(exc) or local_attempt >= self.retry_max_attempts:
                            raise

                        delay_seconds = _retry_delay_seconds(self.retry_base_delay_ms, local_attempt)
                        logger.warning(
                            "ASR retryable error: model=%s attempt=%d/%d delay_ms=%d err=%s",
                            self.model,
                            local_attempt,
                            self.retry_max_attempts,
                            int(round(delay_seconds * 1000)),
                            exc,
                        )
                        if delay_seconds > 0:
                            time.sleep(delay_seconds)

            while True:
                try:
                    response, used_attempts = _perform_request(temperature)
                    attempt += used_attempts
                    break
                except Exception as exc:  # noqa: BLE001
                    raise

            assert response is not None
            text = _extract_text(response).strip()
            metrics = _extract_confidence_metrics(response)
            if self.multi_pass_enabled and _should_run_multi_pass(text, metrics):
                retry_temperature = max(0.2, float(temperature))
                retry_response, retry_attempts = _perform_request(retry_temperature)
                attempt += retry_attempts
                retry_text = _extract_text(retry_response).strip()
                retry_metrics = _extract_confidence_metrics(retry_response)
                if _prefer_multi_pass_result(
                    original_text=text,
                    original_metrics=metrics,
                    retry_text=retry_text,
                    retry_metrics=retry_metrics,
                ):
                    response = retry_response
                    text = retry_text
                    metrics = retry_metrics
            usage_details = _extract_usage_details(response)
            estimated_tokens = _estimate_token_count(text)
            effective_usage_details = usage_details or (
                {"input": estimated_tokens, "output": 0, "total": estimated_tokens}
                if estimated_tokens > 0
                else None
            )
            if _should_drop_as_silence(response, text, metrics):
                text = ""
            logger.info(
                "ASR POST /v1/audio/transcriptions done: model=%s chars=%d attempts=%d suspicious=%s no_speech=%.3f avg_logprob=%s compression_ratio=%s",
                self.model,
                len(text),
                attempt,
                metrics["suspicious"],
                metrics["max_no_speech_prob"] or 0.0,
                metrics["avg_logprob"],
                metrics["compression_ratio"],
            )
            if generation is not None:
                generation.update(
                    output={
                        "text": text,
                        "chars": len(text),
                        "estimatedTokens": estimated_tokens,
                        "retryCount": max(0, attempt - 1),
                        "suspicious": metrics["suspicious"],
                        "maxNoSpeechProb": metrics["max_no_speech_prob"],
                        "avgLogprob": metrics["avg_logprob"],
                        "compressionRatio": metrics["compression_ratio"],
                    },
                    usage_details=effective_usage_details,
                )

        start_ms, end_ms = _extract_bounds_ms(response)
        return ASRChunkResult(
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
            usage_details=effective_usage_details,
            estimated_tokens=estimated_tokens,
            max_no_speech_prob=metrics["max_no_speech_prob"],
            avg_logprob=metrics["avg_logprob"],
            compression_ratio=metrics["compression_ratio"],
            suspicious=bool(metrics["suspicious"]),
        )

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


def _extract_usage_details(response: Any) -> dict[str, int] | None:
    usage = _read_field(response, "usage")
    if usage is None:
        return None
    prompt_tokens = _read_field(usage, "prompt_tokens")
    completion_tokens = _read_field(usage, "completion_tokens")
    total_tokens = _read_field(usage, "total_tokens")
    payload: dict[str, int] = {}
    for key, value in (
        ("input", prompt_tokens),
        ("output", completion_tokens),
        ("total", total_tokens),
    ):
        try:
            if value is not None:
                payload[key] = int(value)
        except (TypeError, ValueError):
            continue
    return payload or None


def _estimate_token_count(text: str) -> int:
    clean = (text or "").strip()
    if not clean:
        return 0

    ascii_chars = sum(1 for char in clean if ord(char) < 128)
    non_ascii_chars = len(clean) - ascii_chars
    ascii_tokens = ascii_chars / 4.0
    non_ascii_tokens = non_ascii_chars / 1.5
    return max(1, int(round(ascii_tokens + non_ascii_tokens)))


def _extract_confidence_metrics(response: Any) -> dict[str, Any]:
    segments = _read_field(response, "segments")
    no_speech_probs: list[float] = []
    avg_logprobs: list[float] = []
    compression_ratios: list[float] = []

    if isinstance(segments, list):
        for seg in segments:
            no_speech = _as_float(_read_field(seg, "no_speech_prob"))
            avg_logprob = _as_float(_read_field(seg, "avg_logprob"))
            compression_ratio = _as_float(_read_field(seg, "compression_ratio"))
            if no_speech is not None:
                no_speech_probs.append(no_speech)
            if avg_logprob is not None:
                avg_logprobs.append(avg_logprob)
            if compression_ratio is not None:
                compression_ratios.append(compression_ratio)

    max_no_speech_prob = max(no_speech_probs) if no_speech_probs else None
    avg_logprob_value = min(avg_logprobs) if avg_logprobs else None
    compression_ratio_value = max(compression_ratios) if compression_ratios else None

    suspicion_score = 0
    if max_no_speech_prob is not None and max_no_speech_prob >= 0.8:
        suspicion_score += 2
    elif max_no_speech_prob is not None and max_no_speech_prob >= 0.6:
        suspicion_score += 1
    if avg_logprob_value is not None and avg_logprob_value <= -1.0:
        suspicion_score += 1
    if compression_ratio_value is not None and compression_ratio_value >= 2.4:
        suspicion_score += 1

    return {
        "max_no_speech_prob": max_no_speech_prob,
        "avg_logprob": avg_logprob_value,
        "compression_ratio": compression_ratio_value,
        "suspicious": suspicion_score >= 2,
    }


def _should_run_multi_pass(text: str, metrics: dict[str, Any]) -> bool:
    clean = (text or "").strip()
    if not clean:
        return False
    max_no_speech = float(metrics.get("max_no_speech_prob") or 0.0)
    avg_logprob = metrics.get("avg_logprob")
    suspicious = bool(metrics.get("suspicious"))
    return suspicious and len(_normalize_for_match(clean)) <= 48 and (
        max_no_speech >= 0.4 or (avg_logprob is not None and float(avg_logprob) <= -0.8)
    )


def _prefer_multi_pass_result(
    *,
    original_text: str,
    original_metrics: dict[str, Any],
    retry_text: str,
    retry_metrics: dict[str, Any],
) -> bool:
    candidate = (retry_text or "").strip()
    if not candidate:
        return False
    if len(_normalize_for_match(candidate)) <= len(_normalize_for_match(original_text)):
        return False
    retry_no_speech = float(retry_metrics.get("max_no_speech_prob") or 0.0)
    original_no_speech = float(original_metrics.get("max_no_speech_prob") or 0.0)
    return retry_no_speech <= max(0.55, original_no_speech + 0.05)


RETRYABLE_OPENAI_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


def _should_retry_openai_error(exc: Exception) -> bool:
    if isinstance(exc, RETRYABLE_OPENAI_ERRORS):
        return True
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        return isinstance(status_code, int) and status_code >= 500
    return False


def _retry_delay_seconds(base_delay_ms: int, attempt: int) -> float:
    if base_delay_ms <= 0:
        return 0.0
    exponent = max(0, attempt - 1)
    return (base_delay_ms * (2**exponent)) / 1000.0



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
    "チャンネル登録よろしくお願いします",
    "チャンネル登録お願いします",
    "高評価とチャンネル登録お願いします",
    "高評価よろしくお願いします",
    "ご覧いただきありがとうございました",
}

SILENCE_HALLUCINATION_PATTERNS = (
    "チャンネル登録",
    "高評価",
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
    "ご覧いただきありがとうございました",
)


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


def _should_drop_as_silence(response: Any, text: str, metrics: dict[str, Any] | None = None) -> bool:
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

    if any(pattern in clean for pattern in SILENCE_HALLUCINATION_PATTERNS):
        return max(no_speech_probs) >= 0.55 and len(normalized) <= 64

    if metrics and metrics.get("suspicious") and (metrics.get("max_no_speech_prob") or 0.0) >= 0.7 and len(normalized) <= 24:
        return True

    # 無音寄り判定が高く、かつ短文なら無音ハルシネーションの可能性が高い。
    return max(no_speech_probs) >= 0.85 and len(normalized) <= 32

@contextmanager
def _noop_generation():
    yield None
