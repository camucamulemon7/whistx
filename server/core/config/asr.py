from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

from .base import env_first_non_empty, to_bool_alias, to_float_alias, to_int_alias


@dataclass(frozen=True)
class AsrConfig:
    asr_backend: str
    asr_realtime_window_seconds: int
    asr_high_accuracy_enabled: bool
    asr_high_accuracy_window_seconds: int
    asr_high_accuracy_priority: int
    asr_high_accuracy_max_rt_lag_seconds: float
    asr_high_accuracy_timeout_seconds: float
    openai_api_key: str
    openai_base_url: str | None
    asr_model: str
    asr_api_timeout_seconds: float
    summary_api_timeout_seconds: float
    proofread_api_timeout_seconds: float
    ffmpeg_timeout_seconds: float
    asr_preprocess_enabled: bool
    asr_preprocess_sample_rate: int
    asr_overlap_ms: int
    asr_vad_drop_enabled: bool
    asr_vad_speech_ratio_min: float
    asr_retry_max_attempts: int
    asr_retry_base_delay_ms: int
    asr_multi_pass_enabled: bool
    asr_rescue_retry_enabled: bool
    asr_rescue_retry_temperature: float
    asr_light_proofread_enabled: bool
    default_language: str
    default_prompt: str
    default_temperature: float
    context_prompt_enabled: bool
    context_max_chars: int
    context_recent_lines: int
    context_term_limit: int
    max_queue_size: int
    max_chunk_bytes: int
    asr_worker_concurrency: int
    media_worker_concurrency: int
    llm_worker_concurrency: int
    diarization_worker_concurrency: int
    artifact_worker_concurrency: int
    blocking_worker_queue_timeout_seconds: float


def load_asr_config() -> AsrConfig:
    backend = (env_first_non_empty("ASR_BACKEND") or "whisper").lower()
    if backend not in {"whisper", "qwen3_vllm"}:
        raise ValueError("ASR_BACKEND must be whisper or qwen3_vllm")
    base_url = env_first_non_empty("ASR_BASE_URL", "OPENAI_BASE_URL")
    if base_url is not None or backend == "qwen3_vllm":
        try:
            parsed = urlsplit(base_url or "")
            valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname) and not any(
                (parsed.username, parsed.password, parsed.query, parsed.fragment)
            ) and not any(char.isspace() for char in base_url or "")
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                valid = False
        except ValueError:
            valid = False
        if not valid:
            # Never interpolate the invalid value: it may be a misplaced API key.
            raise ValueError("ASR_BASE_URL must be an absolute http:// or https:// URL without credentials, query, or fragment; check that ASR_BASE_URL and ASR_API_KEY are not swapped")
    return AsrConfig(
        asr_backend=backend,
        asr_realtime_window_seconds=max(1, min(10, to_int_alias(5, "ASR_REALTIME_WINDOW_SECONDS"))),
        asr_high_accuracy_enabled=to_bool_alias(True, "ASR_HIGH_ACCURACY_ENABLED"),
        asr_high_accuracy_window_seconds=max(30, min(120, to_int_alias(60, "ASR_HIGH_ACCURACY_WINDOW_SECONDS"))),
        asr_high_accuracy_priority=max(1, to_int_alias(10, "ASR_HIGH_ACCURACY_PRIORITY")),
        asr_high_accuracy_max_rt_lag_seconds=max(0.25, to_float_alias(2.0, "ASR_HIGH_ACCURACY_MAX_RT_LAG_SECONDS")),
        asr_high_accuracy_timeout_seconds=max(1.0, to_float_alias(180.0, "ASR_HIGH_ACCURACY_TIMEOUT_SECONDS")),
        openai_api_key=env_first_non_empty("ASR_API_KEY", "OPENAI_API_KEY") or "",
        openai_base_url=base_url.rstrip('/') if base_url else None,
        asr_model=env_first_non_empty("ASR_MODEL") or env_first_non_empty("WHISPER_MODEL") or "whisper-1",
        asr_api_timeout_seconds=max(1.0, to_float_alias(60.0, "ASR_API_TIMEOUT_SECONDS")),
        summary_api_timeout_seconds=max(1.0, to_float_alias(60.0, "SUMMARY_API_TIMEOUT_SECONDS")),
        proofread_api_timeout_seconds=max(1.0, to_float_alias(60.0, "PROOFREAD_API_TIMEOUT_SECONDS")),
        ffmpeg_timeout_seconds=max(1.0, to_float_alias(30.0, "FFMPEG_TIMEOUT_SECONDS")),
        asr_preprocess_enabled=to_bool_alias(True, "ASR_PREPROCESS_ENABLED"),
        asr_preprocess_sample_rate=max(8_000, to_int_alias(16_000, "ASR_PREPROCESS_SAMPLE_RATE")),
        asr_overlap_ms=max(0, to_int_alias(3_500, "ASR_OVERLAP_MS")),
        asr_vad_drop_enabled=to_bool_alias(True, "ASR_VAD_DROP_ENABLED"),
        asr_vad_speech_ratio_min=max(0.0, min(1.0, to_float_alias(0.02, "ASR_VAD_SPEECH_RATIO_MIN"))),
        asr_retry_max_attempts=max(1, to_int_alias(3, "ASR_RETRY_MAX_ATTEMPTS")),
        asr_retry_base_delay_ms=max(0, to_int_alias(1_000, "ASR_RETRY_BASE_DELAY_MS")),
        asr_multi_pass_enabled=to_bool_alias(True, "ASR_MULTI_PASS_ENABLED"),
        asr_rescue_retry_enabled=to_bool_alias(True, "ASR_RESCUE_RETRY_ENABLED"),
        asr_rescue_retry_temperature=max(0.0, min(1.0, to_float_alias(0.25, "ASR_RESCUE_RETRY_TEMPERATURE"))),
        asr_light_proofread_enabled=to_bool_alias(True, "ASR_LIGHT_PROOFREAD_ENABLED"),
        default_language=env_first_non_empty("ASR_DEFAULT_LANGUAGE", "DEFAULT_LANGUAGE") or "ja",
        default_prompt=env_first_non_empty("ASR_DEFAULT_PROMPT", "DEFAULT_PROMPT") or "",
        default_temperature=to_float_alias(0.0, "ASR_DEFAULT_TEMPERATURE", "DEFAULT_TEMPERATURE"),
        context_prompt_enabled=to_bool_alias(True, "ASR_CONTEXT_PROMPT_ENABLED", "CONTEXT_PROMPT_ENABLED"),
        context_max_chars=max(0, to_int_alias(2200, "ASR_CONTEXT_MAX_CHARS", "CONTEXT_MAX_CHARS")),
        context_recent_lines=max(1, to_int_alias(4, "ASR_CONTEXT_RECENT_LINES")),
        context_term_limit=max(8, to_int_alias(80, "ASR_CONTEXT_TERM_LIMIT")),
        max_queue_size=max(1, to_int_alias(8, "ASR_MAX_QUEUE_SIZE", "MAX_QUEUE_SIZE")),
        max_chunk_bytes=max(1024, to_int_alias(12 * 1024 * 1024, "ASR_MAX_CHUNK_BYTES", "MAX_CHUNK_BYTES")),
        asr_worker_concurrency=max(1, to_int_alias(4, "ASR_WORKER_CONCURRENCY")),
        media_worker_concurrency=max(1, to_int_alias(2, "MEDIA_WORKER_CONCURRENCY")),
        llm_worker_concurrency=max(1, to_int_alias(4, "LLM_WORKER_CONCURRENCY")),
        diarization_worker_concurrency=max(1, to_int_alias(1, "DIARIZATION_WORKER_CONCURRENCY")),
        artifact_worker_concurrency=max(1, to_int_alias(2, "ARTIFACT_WORKER_CONCURRENCY")),
        blocking_worker_queue_timeout_seconds=max(
            1.0,
            to_float_alias(30.0, "BLOCKING_WORKER_QUEUE_TIMEOUT_SECONDS"),
        ),
    )
