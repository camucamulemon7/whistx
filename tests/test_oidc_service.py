from __future__ import annotations

import unittest
from urllib.parse import parse_qs, urlparse

from server.services.oidc_service import (
    build_authorization_url,
    exchange_code,
    fetch_discovery,
    fetch_userinfo,
)


class OidcServiceTests(unittest.TestCase):
    def test_discovery_normalizes_issuer_path(self) -> None:
        requested: list[str] = []

        def fetcher(url: str, **_: object) -> dict[str, str]:
            requested.append(url)
            return {"issuer": "https://idp.example"}

        payload = fetch_discovery("https://idp.example/", fetcher=fetcher)
        self.assertEqual(payload["issuer"], "https://idp.example")
        self.assertEqual(requested, ["https://idp.example/.well-known/openid-configuration"])

    def test_authorization_url_contains_pkce_and_state(self) -> None:
        url = build_authorization_url(
            authorization_endpoint="https://idp.example/auth",
            client_id="whistx",
            redirect_uri="https://app.example/callback",
            scope="openid email",
            state="state-value",
            code_challenge="challenge",
        )
        query = parse_qs(urlparse(url).query)
        self.assertEqual(query["response_type"], ["code"])
        self.assertEqual(query["state"], ["state-value"])
        self.assertEqual(query["code_challenge_method"], ["S256"])

    def test_code_exchange_and_userinfo_apply_expected_headers(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def fetcher(url: str, **kwargs: object) -> dict[str, str]:
            calls.append((url, kwargs))
            return {"ok": "yes"}

        exchange_code(
            token_endpoint="https://idp.example/token",
            client_id="whistx",
            client_secret="secret",
            code="code",
            redirect_uri="https://app.example/callback",
            code_verifier="verifier",
            fetcher=fetcher,
        )
        fetch_userinfo("https://idp.example/userinfo", "access-token", fetcher=fetcher)

        token_call = calls[0][1]
        self.assertEqual(token_call["method"], "POST")
        self.assertIn(b"client_secret=secret", token_call["data"])
        self.assertEqual(
            calls[1][1]["headers"],
            {"Authorization": "Bearer access-token"},
        )


if __name__ == "__main__":
    unittest.main()
