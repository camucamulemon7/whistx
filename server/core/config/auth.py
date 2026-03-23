from __future__ import annotations

from dataclasses import dataclass

from .base import decode_env_text, env_first_non_empty, to_bool


@dataclass(frozen=True)
class AuthConfig:
    keycloak_enabled: bool
    keycloak_issuer: str | None
    keycloak_client_id: str
    keycloak_client_secret: str
    keycloak_scope: str
    keycloak_button_label: str
    keycloak_require_email_verified: bool


def load_auth_config() -> AuthConfig:
    return AuthConfig(
        keycloak_enabled=to_bool("KEYCLOAK_ENABLED", False),
        keycloak_issuer=env_first_non_empty("KEYCLOAK_ISSUER"),
        keycloak_client_id=env_first_non_empty("KEYCLOAK_CLIENT_ID") or "",
        keycloak_client_secret=env_first_non_empty("KEYCLOAK_CLIENT_SECRET") or "",
        keycloak_scope=env_first_non_empty("KEYCLOAK_SCOPE") or "openid profile email",
        keycloak_button_label=decode_env_text(env_first_non_empty("KEYCLOAK_BUTTON_LABEL") or "Keycloakでログイン"),
        keycloak_require_email_verified=to_bool("KEYCLOAK_REQUIRE_EMAIL_VERIFIED", True),
    )
