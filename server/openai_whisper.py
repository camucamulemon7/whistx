from __future__ import annotations

import io
import logging
import time
from contextlib import contextmanager
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from .asr import ASRChunkResult
from .config import settings
from .core.logging import emit_container_log
from .langfuse_observer import LangfuseObserver


logger = logging.getLogger(__name__)


class OpenAIWhisperTranscriber:
    def __init__(self, *, api_key: str, base_url: str | None, model: str, observer: LangfuseObserver | None = None):
        if not api_key:
            raise RuntimeError("ASR_API_KEY (or OPENAI_API_KEY) is not set")

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": settings.asr_api_timeout_seconds}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.model = model
        self.observer = observer
        self.backend_profile = str(getattr(settings, "asr_backend_profile", "whisper") or "whisper")
        self.response_format = _normalize_response_format(getattr(settings, "asr_transcription_response_format", None))
        self.send_language = bool(getattr(settings, "asr_send_language", True))
        self.send_prompt = bool(getattr(settings, "asr_send_prompt", True))
        self.send_temperature = bool(getattr(settings, "asr_send_temperature", True))
        self.expect_segments = bool(getattr(settings, "asr_expect_segments", True))
        self.expect_no_speech_prob = bool(getattr(settings, "asr_expect_no_speech_prob", True))
        self.enable_silence_drop = bool(getattr(settings, "asr_enable_silence_drop", True))
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
        chunk_hints: dict[str, float] | None = None,
        trace_context: dict[str, str] | None = None,
    ) -> ASRChunkResult:
        suffix = _ext_from_mime(mime_type)
        response: Any | None = None
        attempt = 0
        effective_chunk_hints = dict(chunk_hints or {})
        request_preview = self._build_request_kwargs(
            file=io.BytesIO(audio_bytes),
            language=language,
            prompt=prompt,
            temperature=temperature,
        )
        request_response_format = request_preview.get("response_format")
        request_keys = [key for key in request_preview.keys() if key != "file"]
        file_name = f"chunk{suffix}"

        with (self.observer.generation(
            name="asr.transcription",
            model=self.model,
            input={
                "mimeType": mime_type,
                "language": language or "",
                "prompt": bool(prompt),
                "audioBytes": len(audio_bytes),
            },
            metadata={
                "endpoint": "audio.transcriptions",
                "profile": self.backend_profile,
                "responseFormat": request_response_format,
            },
            model_parameters={"temperature": temperature if self.send_temperature else None},
            trace_context=trace_context,
        ) if self.observer else _noop_generation()) as generation:
            def _perform_request(request_temperature: float) -> Any:
                local_attempt = 0
                while True:
                    local_attempt += 1
                    file_obj = io.BytesIO(audio_bytes)
                    file_obj.name = file_name
                    request_kwargs = self._build_request_kwargs(
                        file=file_obj,
                        language=language,
                        prompt=prompt,
                        temperature=request_temperature,
                    )
                    request_keys = [key for key in request_kwargs.keys() if key != "file"]
                    response_format = request_kwargs.get("response_format")
                    should_retry = False
                    emit_container_log(
                        __name__,
                        "info",
                        "ASR POST /v1/audio/transcriptions: profile=%s model=%s mime=%s bytes=%d language=%s prompt=%s attempt=%d/%d temp=%s response_format=%s",
                        self.backend_profile,
                        self.model,
                        mime_type,
                        len(audio_bytes),
                        language or "",
                        bool(prompt),
                        local_attempt,
                        self.retry_max_attempts,
                        f"{request_temperature:.2f}" if self.send_temperature else "disabled",
                        response_format or "",
                    )
                    logger.info(
                        "ASR POST /v1/audio/transcriptions: profile=%s model=%s mime=%s bytes=%d language=%s prompt=%s attempt=%d/%d temp=%s response_format=%s",
                        self.backend_profile,
                        self.model,
                        mime_type,
                        len(audio_bytes),
                        language or "",
                        bool(prompt),
                        local_attempt,
                        self.retry_max_attempts,
                        f"{request_temperature:.2f}" if self.send_temperature else "disabled",
                        response_format or "",
                    )
                    try:
                        return self.client.audio.transcriptions.create(**request_kwargs), local_attempt
                    except Exception as exc:  # noqa: BLE001
                        should_retry = _should_retry_openai_error(exc) and local_attempt < self.retry_max_attempts
                        _log_transcription_error(
                            exc,
                            backend_profile=self.backend_profile,
                            model=self.model,
                            mime_type=mime_type,
                            audio_bytes_len=len(audio_bytes),
                            request_keys=request_keys,
                            response_format=response_format,
                            should_retry=should_retry,
                        )
                        if not should_retry:
                            raise

                        delay_seconds = _retry_delay_seconds(self.retry_base_delay_ms, local_attempt)
                        logger.warning(
                            "ASR retryable error: profile=%s model=%s attempt=%d/%d delay_ms=%d err=%s",
                            self.backend_profile,
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
            parsed = _parse_transcription_response(response)
            text = parsed["text"].strip()
            metrics = _extract_confidence_metrics(
                response,
                text=text,
                expect_no_speech_prob=self.expect_no_speech_prob,
            )
            if self.multi_pass_enabled and _should_run_multi_pass(
                text,
                metrics,
                backend_profile=self.backend_profile,
            ):
                retry_temperature = max(0.2, float(temperature))
                retry_response, retry_attempts = _perform_request(retry_temperature)
                attempt += retry_attempts
                retry_parsed = _parse_transcription_response(retry_response)
                retry_text = retry_parsed["text"].strip()
                retry_metrics = _extract_confidence_metrics(
                    retry_response,
                    text=retry_text,
                    expect_no_speech_prob=self.expect_no_speech_prob,
                )
                if _prefer_multi_pass_result(
                    original_text=text,
                    original_metrics=metrics,
                    retry_text=retry_text,
                    retry_metrics=retry_metrics,
                ):
                    response = retry_response
                    parsed = retry_parsed
                    text = retry_text
                    metrics = retry_metrics
            usage_details = parsed["usage_details"] or _extract_usage_details(response)
            estimated_tokens = _estimate_token_count(text)
            effective_usage_details = usage_details or (
                {"input": estimated_tokens, "output": 0, "total": estimated_tokens}
                if estimated_tokens > 0
                else None
            )
            if self.enable_silence_drop and _should_drop_as_silence(
                response,
                text,
                metrics,
                chunk_hints=effective_chunk_hints,
                expect_no_speech_prob=self.expect_no_speech_prob,
            ):
                text = ""
            emit_container_log(
                __name__,
                "info",
                "ASR POST /v1/audio/transcriptions done: profile=%s model=%s chars=%d attempts=%d suspicious=%s no_speech=%s avg_logprob=%s compression_ratio=%s",
                self.backend_profile,
                self.model,
                len(text),
                attempt,
                metrics["suspicious"],
                metrics["max_no_speech_prob"],
                metrics["avg_logprob"],
                metrics["compression_ratio"],
            )
            logger.info(
                "ASR POST /v1/audio/transcriptions done: profile=%s model=%s chars=%d attempts=%d suspicious=%s no_speech=%s avg_logprob=%s compression_ratio=%s",
                self.backend_profile,
                self.model,
                len(text),
                attempt,
                metrics["suspicious"],
                metrics["max_no_speech_prob"],
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

        start_ms, end_ms = _extract_bounds_ms(parsed if self.expect_segments else response)
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

    def _build_request_kwargs(
        self,
        *,
        file: io.BytesIO,
        language: str | None,
        prompt: str | None,
        temperature: float,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "file": file,
            "model": self.model,
        }
        if self.send_language and language:
            kwargs["language"] = language
        if self.send_prompt and prompt:
            kwargs["prompt"] = prompt
        if self.send_temperature:
            kwargs["temperature"] = temperature
        if self.response_format:
            kwargs["response_format"] = self.response_format
        return kwargs



def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        value = response.get("text", "")
        return value if isinstance(value, str) else str(value)

    value = getattr(response, "text", "")
    return value if isinstance(value, str) else str(value)


def _parse_transcription_response(response: Any) -> dict[str, Any]:
    try:
        if isinstance(response, str):
            text = response
            return {
                "text": text,
                "segments": None,
                "usage_details": _usage_from_estimate(text),
            }

        text = _extract_text(response)
        if not text:
            alt_text = _read_field(response, "transcript")
            if isinstance(alt_text, str):
                text = alt_text
            elif alt_text is not None:
                text = str(alt_text)

        usage_details = _extract_usage_details(response)
        segments = _extract_segments(response)
        return {
            "text": text or "",
            "segments": segments,
            "usage_details": usage_details or _usage_from_estimate(text),
        }
    except Exception:  # noqa: BLE001
        fallback_text = ""
        try:
            fallback_text = _extract_text(response)
        except Exception:  # noqa: BLE001
            fallback_text = ""
        return {
            "text": fallback_text,
            "segments": None,
            "usage_details": _usage_from_estimate(fallback_text),
        }



def _extract_bounds_ms(response: Any) -> tuple[int | None, int | None]:
    segments = _extract_segments(response)
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


def _usage_from_estimate(text: str) -> dict[str, int] | None:
    estimated = _estimate_token_count(text)
    if estimated <= 0:
        return None
    return {"input": estimated, "output": 0, "total": estimated}


def _estimate_token_count(text: str) -> int:
    clean = (text or "").strip()
    if not clean:
        return 0

    ascii_chars = sum(1 for char in clean if ord(char) < 128)
    non_ascii_chars = len(clean) - ascii_chars
    ascii_tokens = ascii_chars / 4.0
    non_ascii_tokens = non_ascii_chars / 1.5
    return max(1, int(round(ascii_tokens + non_ascii_tokens)))


def _extract_confidence_metrics(
    response: Any,
    *,
    text: str = "",
    expect_no_speech_prob: bool = True,
) -> dict[str, Any]:
    segments = _extract_segments(response)
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
    if _has_replacement_chars(text):
        suspicion_score += 2
    if _is_extremely_short_text(text):
        suspicion_score += 1
    if _has_repetition_noise(text):
        suspicion_score += 2
    if _looks_garbled_text(text):
        suspicion_score += 1
    if max_no_speech_prob is None and expect_no_speech_prob:
        suspicion_score = max(0, suspicion_score - 1)

    return {
        "max_no_speech_prob": max_no_speech_prob,
        "avg_logprob": avg_logprob_value,
        "compression_ratio": compression_ratio_value,
        "suspicious": suspicion_score >= 2,
    }


def _should_run_multi_pass(text: str, metrics: dict[str, Any], *, backend_profile: str = "whisper") -> bool:
    clean = (text or "").strip()
    if not clean:
        return False
    max_no_speech = metrics.get("max_no_speech_prob")
    avg_logprob = metrics.get("avg_logprob")
    suspicious = bool(metrics.get("suspicious"))
    normalized_len = len(_normalize_for_match(clean))
    if not suspicious or normalized_len > 48:
        return False
    if max_no_speech is not None:
        return float(max_no_speech) >= 0.4 or (avg_logprob is not None and float(avg_logprob) <= -0.8)
    if backend_profile == "vllm_qwen3":
        return _has_replacement_chars(clean) or _has_repetition_noise(clean) or normalized_len <= 18
    return avg_logprob is not None and float(avg_logprob) <= -0.8


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
    candidate_len = len(_normalize_for_match(candidate))
    original_len = len(_normalize_for_match(original_text))
    if candidate_len <= original_len and bool(retry_metrics.get("suspicious")) >= bool(original_metrics.get("suspicious")):
        return False
    retry_no_speech = retry_metrics.get("max_no_speech_prob")
    original_no_speech = original_metrics.get("max_no_speech_prob")
    if retry_no_speech is None or original_no_speech is None:
        return bool(original_metrics.get("suspicious")) and not bool(retry_metrics.get("suspicious")) or candidate_len > original_len
    return float(retry_no_speech) <= max(0.55, float(original_no_speech) + 0.05)


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


def _should_drop_as_silence(
    response: Any,
    text: str,
    metrics: dict[str, Any] | None = None,
    *,
    chunk_hints: dict[str, Any] | None = None,
    expect_no_speech_prob: bool = True,
) -> bool:
    clean = text.strip()
    if not clean:
        return True

    normalized = _normalize_for_match(clean)
    if normalized in KNOWN_SILENCE_HALLUCINATIONS:
        return True
    if any(pattern in clean for pattern in SILENCE_HALLUCINATION_PATTERNS) and len(normalized) <= 64:
        return True
    if len(normalized) <= 4 and _looks_like_short_noise(clean):
        return True

    speech_ratio = _as_float(_read_field(chunk_hints or {}, "speech_ratio"))
    rms = _as_float(_read_field(chunk_hints or {}, "rms"))
    peak = _as_float(_read_field(chunk_hints or {}, "peak"))
    if speech_ratio is not None and speech_ratio < 0.02 and len(normalized) <= 24:
        return True
    if speech_ratio is not None and speech_ratio < 0.05 and rms is not None and rms < 0.015 and len(normalized) <= 12:
        return True
    if peak is not None and peak < 0.03 and speech_ratio is not None and speech_ratio < 0.03 and len(normalized) <= 8:
        return True

    segments = _extract_segments(response)
    if not isinstance(segments, list) or not segments or not expect_no_speech_prob:
        return False
    no_speech_probs: list[float] = []
    for seg in segments:
        value = _as_float(_read_field(seg, "no_speech_prob"))
        if value is not None:
            no_speech_probs.append(value)

    if not no_speech_probs:
        return False

    if metrics and metrics.get("suspicious") and (metrics.get("max_no_speech_prob") or 0.0) >= 0.7 and len(normalized) <= 24:
        return True

    # 無音寄り判定が高く、かつ短文なら無音ハルシネーションの可能性が高い。
    return max(no_speech_probs) >= 0.85 and len(normalized) <= 32


def _extract_segments(response: Any) -> list[Any] | None:
    segments = _read_field(response, "segments")
    return segments if isinstance(segments, list) else None


def _normalize_response_format(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _has_replacement_chars(text: str) -> bool:
    return "\ufffd" in (text or "")


def _is_extremely_short_text(text: str) -> bool:
    normalized = _normalize_for_match(text or "")
    return bool(normalized) and len(normalized) <= 2


def _has_repetition_noise(text: str) -> bool:
    normalized = _normalize_for_match(text or "")
    if len(normalized) < 12:
        return False
    for unit in range(1, min(6, len(normalized) // 3 + 1)):
        fragment = normalized[:unit]
        if fragment and fragment * max(3, len(normalized) // max(1, unit)) in normalized:
            return True
    return any(char * 6 in normalized for char in set(normalized))


def _looks_garbled_text(text: str) -> bool:
    clean = (text or "").strip()
    if not clean:
        return False
    punctuation = sum(1 for char in clean if not char.isalnum() and not char.isspace())
    return punctuation >= max(4, len(clean) // 2)


def _looks_like_short_noise(text: str) -> bool:
    clean = (text or "").strip()
    return bool(clean) and (clean in {"えー", "あー", "ん", "uh", "um"} or _has_repetition_noise(clean))


def _extract_error_body(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        body = _read_field(exc, "body")
        return "" if body is None else str(body)
    try:
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text
    except Exception:  # noqa: BLE001
        pass
    try:
        content = getattr(response, "content", None)
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        if content is not None:
            return str(content)
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _log_transcription_error(
    exc: Exception,
    *,
    backend_profile: str,
    model: str,
    mime_type: str,
    audio_bytes_len: int,
    request_keys: list[str],
    response_format: str | None,
    should_retry: bool,
) -> None:
    status_code = getattr(exc, "status_code", None)
    body = _extract_error_body(exc)
    level = logging.WARNING if should_retry else logging.ERROR
    logger.log(
        level,
        "ASR request failed: profile=%s model=%s mime=%s bytes=%d request_keys=%s response_format=%s status=%s retry=%s body=%s err=%s",
        backend_profile,
        model,
        mime_type,
        audio_bytes_len,
        request_keys,
        response_format or "",
        status_code,
        should_retry,
        body,
        exc,
    )
    emit_container_log(
        __name__,
        "warning" if should_retry else "error",
        "ASR request failed: profile=%s model=%s mime=%s bytes=%d request_keys=%s response_format=%s status=%s retry=%s body=%s err=%s",
        backend_profile,
        model,
        mime_type,
        audio_bytes_len,
        request_keys,
        response_format or "",
        status_code,
        should_retry,
        body,
        exc,
    )

@contextmanager
def _noop_generation():
    yield None
