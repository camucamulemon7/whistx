from __future__ import annotations

import json
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

JsonFetcher = Callable[..., dict[str, Any]]


def fetch_json(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = UrlRequest(url, method=method, data=data, headers=headers or {})
    with urlopen(request, timeout=10) as response:  # noqa: S310
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("oidc_response_must_be_object")
    return parsed


def fetch_discovery(issuer: str, *, fetcher: JsonFetcher = fetch_json) -> dict[str, Any]:
    discovery_url = f"{issuer.rstrip('/')}/.well-known/openid-configuration"
    return fetcher(discovery_url)


def build_authorization_url(
    *,
    authorization_endpoint: str,
    client_id: str,
    redirect_uri: str,
    scope: str,
    state: str,
    code_challenge: str,
) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{authorization_endpoint}?{urlencode(params)}"


def exchange_code(
    *,
    token_endpoint: str,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    fetcher: JsonFetcher = fetch_json,
) -> dict[str, Any]:
    form = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    if client_secret:
        form["client_secret"] = client_secret
    return fetcher(
        token_endpoint,
        method="POST",
        data=urlencode(form).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


def fetch_userinfo(
    userinfo_endpoint: str,
    access_token: str,
    *,
    fetcher: JsonFetcher = fetch_json,
) -> dict[str, Any]:
    return fetcher(userinfo_endpoint, headers={"Authorization": f"Bearer {access_token}"})
