"""Runtime capability snapshot built from explicit resources."""
import asyncio
from fastapi.responses import JSONResponse
from ..core.config import settings
from ..core.runtime_resources import RuntimeResources
from ..repositories import quota_repository
from . import oidc_flow

MAX_DIARIZATION_SPEAKERS = 12


async def health(*, resources: RuntimeResources) -> JSONResponse:
    active_connections = await asyncio.to_thread(
        quota_repository.count_active_connections
    )
    return JSONResponse(
        {
            "status": "ok",
            "model": settings.asr_model,
            "asrReady": resources.transcriber_factory is not None,
            "asrBackend": settings.asr_backend,
            "capturePacketMs": 250 if settings.asr_backend == "qwen3_vllm" else 1000,
            "highAccuracyWindowSeconds": settings.asr_high_accuracy_window_seconds if settings.asr_high_accuracy_enabled else None,
            "summaryModel": settings.summary_model if resources.summarizer else None,
            "meetingTranslationModel": (settings.meeting_translation_model or settings.summary_model) if resources.summarizer else None,
            "proofreadModel": settings.proofread_model if resources.proofreader else None,
            "diarizationEnabled": resources.diarizer is not None,
            "diarizationModel": settings.diarization_model if resources.diarizer else None,
            "diarizationDefaultNumSpeakers": settings.diarization_num_speakers,
            "diarizationDefaultMinSpeakers": settings.diarization_min_speakers,
            "diarizationDefaultMaxSpeakers": settings.diarization_max_speakers,
            "diarizationSpeakerCap": MAX_DIARIZATION_SPEAKERS,
            "banners": list(settings.ui_banners),
            "uiBrandTitle": settings.app_brand_title,
            "uiBrandTagline": settings.app_brand_tagline,
            "uiPromptTemplates": list(settings.ui_prompt_templates),
            "selfSignupEnabled": settings.enable_self_signup,
            "keycloakEnabled": oidc_flow._keycloak_login_enabled(),
            "keycloakButtonLabel": settings.keycloak_button_label,
            "wsPath": settings.ws_path,
            "liveWsPath": settings.ws_path + "/live",
            "meetingInsights": True,
            "activeConnections": active_connections,
        }
    )
