from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
from typing import Any
from urllib.parse import urljoin, urlsplit

from fastapi import Request
from fastapi.responses import Response

from .config import settings
from ..transcript_store import resolve_transcript_path


def runtime_access_allowed(*, transcripts_dir, session_id: str, token: str | None) -> bool:
    base_path = resolve_transcript_path(transcripts_dir, session_id, "txt")
    if base_path is None:
        return False
    metadata_path = base_path.with_suffix(".meta.json")
    if not metadata_path.exists():
        return True
    try:
        loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata = loaded if isinstance(loaded, dict) else {}
    required = str(metadata.get("accessToken") or "").strip()
    if not required:
        return True
    return secrets.compare_digest(required, str(token or ""))


def serialize_user(user) -> dict[str, Any] | None:
    if user is None:
        return None
    return {
        "id": user.id,
        "email": user.email,
        "displayName": user.display_name,
        "isAdmin": bool(user.is_admin),
        "isActive": bool(user.is_active),
        "approvedAt": user.approved_at.isoformat() if user.approved_at else None,
    }


def signed_payload(value: dict[str, Any]) -> str:
    raw = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    body = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    signature = hmac.new(settings.app_session_secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    token = base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("=")
    return f"{body}.{token}"


def unsigned_payload(value: str | None) -> dict[str, Any] | None:
    if not value or "." not in value:
        return None
    body, token = value.split(".", 1)
    expected = hmac.new(settings.app_session_secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    if not secrets.compare_digest(base64.urlsafe_b64encode(expected).decode("utf-8").rstrip("="), token):
        return None
    padded = body + "=" * (-len(body) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")))
    except Exception:
        return None


def request_is_secure(request: Request) -> bool:
    scheme = _request_external_scheme(request)
    return scheme in {"https", "wss"}


def external_url_for(request: Request, route_name: str, /, **path_params: Any) -> str:
    url = request.url_for(route_name, **path_params)
    if settings.app_public_url:
        return urljoin(settings.app_public_url.rstrip("/") + "/", str(url.path).lstrip("/"))
    return str(url.replace(scheme=_request_external_scheme(request), netloc=_request_external_host(request)))


def client_ip(request: Request) -> str:
    if _proxy_is_trusted(request):
        forwarded = _first_proxy_value(request.headers.get("x-forwarded-for"))
        if forwarded:
            return forwarded
        real_ip = (request.headers.get("x-real-ip") or "").strip()
        if real_ip:
            return real_ip
    return request.client.host if request.client and request.client.host else "unknown"


def origin_is_allowed(request: Request) -> bool:
    origin = (request.headers.get("origin") or "").strip()
    if not origin:
        return True
    configured = settings.app_public_url or f"{_request_external_scheme(request)}://{_request_external_host(request)}"
    parsed_origin = urlsplit(origin)
    parsed_configured = urlsplit(configured)
    return (
        parsed_origin.scheme.lower(),
        parsed_origin.netloc.lower(),
    ) == (
        parsed_configured.scheme.lower(),
        parsed_configured.netloc.lower(),
    )


def set_oidc_state_cookie(*, response: Response, request: Request, cookie_name: str, payload: dict[str, Any]) -> None:
    response.set_cookie(
        key=cookie_name,
        value=signed_payload(payload),
        httponly=True,
        samesite="lax",
        secure=request_is_secure(request),
        max_age=10 * 60,
        path="/",
    )


def read_oidc_state_cookie(*, request: Request, cookie_name: str) -> dict[str, Any] | None:
    return unsigned_payload(request.cookies.get(cookie_name))


def clear_oidc_state_cookie(*, response: Response, request: Request, cookie_name: str) -> None:
    response.delete_cookie(
        key=cookie_name,
        path="/",
        secure=request_is_secure(request),
        httponly=True,
        samesite="lax",
    )


def set_session_cookie(*, response: Response, request: Request, cookie_name: str, session_id: str) -> None:
    response.set_cookie(
        key=cookie_name,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=request_is_secure(request),
        max_age=settings.app_session_days * 24 * 60 * 60,
        path="/",
    )


def clear_session_cookie(*, response: Response, request: Request, cookie_name: str) -> None:
    response.delete_cookie(
        key=cookie_name,
        path="/",
        secure=request_is_secure(request),
        httponly=True,
        samesite="lax",
    )


def _request_external_scheme(request: Request) -> str:
    if _proxy_is_trusted(request):
        x_forwarded_proto = _first_proxy_value(request.headers.get("x-forwarded-proto"))
        if x_forwarded_proto:
            return x_forwarded_proto.lower()
        forwarded = _parse_forwarded_header(request.headers.get("forwarded"))
        forwarded_proto = forwarded.get("proto")
        if forwarded_proto:
            return forwarded_proto.lower()
    return request.url.scheme


def _request_external_host(request: Request) -> str:
    if _proxy_is_trusted(request):
        x_forwarded_host = _first_proxy_value(request.headers.get("x-forwarded-host"))
        if x_forwarded_host:
            return x_forwarded_host
        forwarded = _parse_forwarded_header(request.headers.get("forwarded"))
        forwarded_host = forwarded.get("host")
        if forwarded_host:
            return forwarded_host
    host = (request.headers.get("host") or "").strip()
    if host and _host_is_allowed(host):
        return host
    return next((item for item in settings.app_allowed_hosts if item != "*"), request.url.netloc)


def _proxy_is_trusted(request: Request) -> bool:
    if not settings.app_trust_proxy_headers or not settings.app_trusted_proxy_ips:
        return False
    peer = request.client.host if request.client and request.client.host else ""
    if "*" in settings.app_trusted_proxy_ips:
        return True
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return False
    for value in settings.app_trusted_proxy_ips:
        try:
            if address in ipaddress.ip_network(value, strict=False):
                return True
        except ValueError:
            continue
    return False


def _host_is_allowed(host: str) -> bool:
    hostname = host[1:].split("]", 1)[0] if host.startswith("[") else host.split(":", 1)[0]
    hostname = hostname.lower()
    for allowed in settings.app_allowed_hosts:
        candidate = allowed.strip().lower()
        if candidate == "*":
            return True
        if candidate.startswith("*.") and hostname.endswith(candidate[1:]):
            return True
        if hostname == candidate.strip("[]"):
            return True
    return False


def _first_proxy_value(value: str | None) -> str:
    if not value:
        return ""
    return value.split(",", 1)[0].strip()


def _parse_forwarded_header(value: str | None) -> dict[str, str]:
    raw = _first_proxy_value(value)
    if not raw:
        return {}
    out: dict[str, str] = {}
    for chunk in raw.split(";"):
        key, sep, parsed = chunk.partition("=")
        if not sep:
            continue
        clean_key = key.strip().lower()
        clean_value = parsed.strip().strip('"')
        if clean_key and clean_value:
            out[clean_key] = clean_value
    return out
