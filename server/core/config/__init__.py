from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .app import AppConfig, load_app_config
from .asr import AsrConfig, load_asr_config
from .auth import AuthConfig, load_auth_config
from .base import decode_env_text, to_float, to_int
from .diarization import DiarizationConfig, load_diarization_config
from .observability import ObservabilityConfig, load_observability_config
from .ui import UiConfig, load_ui_config


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    asr: AsrConfig
    auth: AuthConfig
    diarization: DiarizationConfig
    ui: UiConfig
    observability: ObservabilityConfig
    summary_api_key: str
    summary_base_url: str | None
    summary_model: str
    summary_temperature: float
    summary_input_max_chars: int
    summary_system_prompt: str
    summary_prompt_template: str
    proofread_api_key: str
    proofread_base_url: str | None
    proofread_model: str
    proofread_temperature: float
    proofread_input_max_chars: int
    proofread_system_prompt: str
    proofread_prompt_template: str

    def __getattr__(self, name: str) -> Any:
        for section in (self.app, self.asr, self.auth, self.diarization, self.ui, self.observability):
            if hasattr(section, name):
                return getattr(section, name)
        raise AttributeError(name)


def load_settings() -> Settings:
    app = load_app_config()
    asr = load_asr_config()
    auth = load_auth_config()
    diarization = load_diarization_config(app.app_data_dir)
    ui = load_ui_config()
    observability = load_observability_config()
    summary_api_key = os.getenv("SUMMARY_API_KEY", "").strip() or asr.openai_api_key
    summary_base_url = os.getenv("SUMMARY_BASE_URL", "").strip() or asr.openai_base_url
    summary_system_prompt = decode_env_text(os.getenv("SUMMARY_SYSTEM_PROMPT", ""))
    summary_prompt_template = decode_env_text(os.getenv("SUMMARY_PROMPT_TEMPLATE", ""))
    proofread_api_key = os.getenv("PROOFREAD_API_KEY", "").strip() or summary_api_key or asr.openai_api_key
    proofread_base_url = os.getenv("PROOFREAD_BASE_URL", "").strip() or summary_base_url or asr.openai_base_url
    proofread_system_prompt = decode_env_text(os.getenv("PROOFREAD_SYSTEM_PROMPT", ""))
    proofread_prompt_template = decode_env_text(os.getenv("PROOFREAD_PROMPT_TEMPLATE", ""))
    return Settings(
        app=app,
        asr=asr,
        auth=auth,
        diarization=diarization,
        ui=ui,
        observability=observability,
        summary_api_key=summary_api_key,
        summary_base_url=summary_base_url,
        summary_model=os.getenv("SUMMARY_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        summary_temperature=to_float("SUMMARY_TEMPERATURE", 0.2),
        summary_input_max_chars=max(2_000, to_int("SUMMARY_INPUT_MAX_CHARS", 16_000)),
        summary_system_prompt=summary_system_prompt,
        summary_prompt_template=summary_prompt_template,
        proofread_api_key=proofread_api_key,
        proofread_base_url=proofread_base_url,
        proofread_model=os.getenv("PROOFREAD_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        proofread_temperature=to_float("PROOFREAD_TEMPERATURE", 0.0),
        proofread_input_max_chars=max(2_000, to_int("PROOFREAD_INPUT_MAX_CHARS", 24_000)),
        proofread_system_prompt=proofread_system_prompt,
        proofread_prompt_template=proofread_prompt_template,
    )


settings = load_settings()
