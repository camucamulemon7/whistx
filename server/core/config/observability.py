from __future__ import annotations

from dataclasses import dataclass

from .base import env_first_non_empty, to_bool


@dataclass(frozen=True)
class ObservabilityConfig:
    langfuse_enabled: bool
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str | None
    langfuse_environment: str | None
    langfuse_release: str | None


def load_observability_config() -> ObservabilityConfig:
    return ObservabilityConfig(
        langfuse_enabled=to_bool("LANGFUSE_ENABLED", True),
        langfuse_public_key=env_first_non_empty("LANGFUSE_PUBLIC_KEY") or "",
        langfuse_secret_key=env_first_non_empty("LANGFUSE_SECRET_KEY") or "",
        langfuse_host=env_first_non_empty("LANGFUSE_HOST"),
        langfuse_environment=env_first_non_empty("LANGFUSE_ENVIRONMENT", "LANGFUSE_ENV"),
        langfuse_release=env_first_non_empty("LANGFUSE_RELEASE"),
    )
