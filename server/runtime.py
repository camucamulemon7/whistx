from __future__ import annotations

import asyncio
import logging
from typing import Callable

from fastapi import WebSocket

from .core.runtime_resources import RuntimeResources
from .transcription import coordinator
from .services import oidc_flow, runtime_cleanup
from .asr import SessionTranscriber
from .audio_pipeline import AudioPreprocessor
from .core.config import settings
from .diarizer import PyannoteSpeakerDiarizer
from .langfuse_observer import make_langfuse_observer
from .openai_whisper import OpenAIWhisperTranscriber
from .summarizer import OpenAISummarizer
from .core.logging import emit_container_log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

resources = RuntimeResources()


CLEANUP_INTERVAL_SECONDS = 6 * 60 * 60


async def on_startup() -> None:

    _validate_runtime_configuration()
    runtime_cleanup.run_cleanup_once("startup")
    resources.cleanup_task = asyncio.create_task(_periodic_cleanup_loop())
    resources.event_loop_monitor_task = asyncio.create_task(_event_loop_lag_monitor())
    logger.info(
        "startup config: history_dir=%s keycloak=%s self_signup=%s require_verified_email=%s",
        settings.history_dir,
        oidc_flow._keycloak_login_enabled(),
        settings.enable_self_signup,
        settings.keycloak_require_email_verified,
    )
    emit_container_log(
        __name__,
        "info",
        "startup config: history_dir=%s keycloak=%s self_signup=%s require_verified_email=%s",
        settings.history_dir,
        oidc_flow._keycloak_login_enabled(),
        settings.enable_self_signup,
        settings.keycloak_require_email_verified,
    )
    resources.observer = make_langfuse_observer(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        environment=settings.langfuse_environment,
        release=settings.langfuse_release,
        enabled=settings.langfuse_enabled,
        capture_content=settings.langfuse_capture_content,
    )
    if settings.langfuse_enabled and settings.langfuse_capture_content:
        logger.warning("Langfuse content capture is enabled: transcription, prompts, glossary, and model output may be sent to the configured telemetry provider; see docs/privacy.md")

    resources.transcriber_factory = _build_transcriber_factory() if settings.openai_api_key else None
    resources.audio_preprocessor = AudioPreprocessor(
        ffmpeg_bin=settings.ffmpeg_bin,
        sample_rate=settings.asr_preprocess_sample_rate,
        overlap_ms=settings.asr_overlap_ms,
        enabled=settings.asr_preprocess_enabled,
    )
    try:
        resources.summarizer = OpenAISummarizer(
            api_key=settings.summary_api_key,
            base_url=settings.summary_base_url,
            model=settings.summary_model,
            translation_model=settings.meeting_translation_model,
            temperature=settings.summary_temperature,
            summary_system_prompt=settings.summary_system_prompt,
            summary_prompt_template=settings.summary_prompt_template,
            proofread_system_prompt=settings.proofread_system_prompt,
            proofread_prompt_template=settings.proofread_prompt_template,
            timeout_seconds=settings.summary_api_timeout_seconds,
            observer=resources.observer,
        )
    except Exception as exc:  # noqa: BLE001
        resources.summarizer = None
        logger.warning("summary disabled: %s", exc)

    try:
        resources.proofreader = OpenAISummarizer(
            api_key=settings.proofread_api_key,
            base_url=settings.proofread_base_url,
            model=settings.proofread_model,
            temperature=settings.proofread_temperature,
            summary_system_prompt=settings.summary_system_prompt,
            summary_prompt_template=settings.summary_prompt_template,
            proofread_system_prompt=settings.proofread_system_prompt,
            proofread_prompt_template=settings.proofread_prompt_template,
            timeout_seconds=settings.proofread_api_timeout_seconds,
            observer=resources.observer,
        )
    except Exception as exc:  # noqa: BLE001
        resources.proofreader = None
        logger.warning("proofread disabled: %s", exc)

    if settings.diarization_enabled:
        try:
            resources.diarizer = PyannoteSpeakerDiarizer(
                hf_token=settings.diarization_hf_token,
                model=settings.diarization_model,
                ffmpeg_bin=settings.ffmpeg_bin,
                device=settings.diarization_device,
                sample_rate=settings.diarization_sample_rate,
                num_speakers=settings.diarization_num_speakers,
                min_speakers=settings.diarization_min_speakers,
                max_speakers=settings.diarization_max_speakers,
            )
            resources.diarizer.preflight()
        except Exception as exc:  # noqa: BLE001
            resources.diarizer = None
            logger.warning("diarization disabled: %s", exc)
    else:
        resources.diarizer = None

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
    for task in (resources.cleanup_task, resources.event_loop_monitor_task):
        if task is None:
            continue
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    resources.cleanup_task = None
    resources.event_loop_monitor_task = None
    if resources.observer is not None:
        resources.observer.flush()
        resources.observer.shutdown()


async def _periodic_cleanup_loop() -> None:
    try:
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
            await asyncio.to_thread(runtime_cleanup.run_cleanup_once, "periodic cleanup")
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


def _validate_runtime_configuration() -> None:
    if settings.app_session_secret.strip() == "change-me":
        logger.warning(
            "APP_SESSION_SECRET is using the default placeholder; set a strong secret before running in shared environments"
        )
    if settings.keycloak_enabled and not oidc_flow._keycloak_login_enabled():
        logger.warning(
            "KEYCLOAK_ENABLED is set but issuer/client_id is incomplete; keycloak login will be disabled"
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
        observer=resources.observer,
    )


def _use_realtime_asr(model: str) -> bool:
    lowered = (model or "").strip().lower()
    return "voxtral" in lowered and "realtime" in lowered


async def ws_transcribe(ws: WebSocket) -> None:
    await coordinator.ws_transcribe(ws, resources=resources)
