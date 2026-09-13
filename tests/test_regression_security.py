from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'test-session-secret-abcdefghijklmnopqrstuvwxyz12')

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import auth as auth_module
from server import runtime
from server.schemas import LoginRequest
from server.api.routes import auth as auth_routes
from server.api.routes import summary as summary_routes
from server.api.ws import transcribe as ws_routes
from server.core import application as application_core
from server.core import security
from server.core import rate_limit
from server.repositories import quota_repository, session_repository
from server.services import (
    auth_service,
)


class DummyDB:
    def rollback(self) -> None:
        pass


def make_request(client_host: str = '127.0.0.1') -> Request:
    scope = {
        'type': 'http',
        'method': 'POST',
        'path': '/api/auth/login',
        'headers': [(b'host', b'testserver'), (b'user-agent', b'pytest')],
        'client': (client_host, 12345),
        'query_string': b'',
    }
    return Request(scope)


def make_security_settings(**overrides):
    values = {
        'app_public_url': None,
        'app_trust_proxy_headers': False,
        'app_trusted_proxy_ips': (),
        'app_allowed_hosts': ('localhost', '127.0.0.1', 'testserver', 'internal', 'app.example.com'),
        'app_session_secret': 'test-session-secret-abcdefghijklmnopqrstuvwxyz12',
        'app_session_days': 7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        rate_limit.clear()

    def test_login_route_rate_limits_after_repeated_failures(self) -> None:
        payload = LoginRequest(email='user@example.com', password='wrongpass')
        request = make_request()
        db = DummyDB()

        with patch.object(auth_module, 'get_user_by_email', return_value=None):
            for _ in range(5):
                response = auth_routes.auth_login(payload, request, db)
                self.assertEqual(response.status_code, 401)

            response = auth_routes.auth_login(payload, request, db)
            self.assertEqual(response.status_code, 429)
            self.assertIn('too_many_login_attempts', response.body.decode('utf-8'))


    def test_session_lookup_filters_expiry_without_deleting(self) -> None:
        now = datetime.now(timezone.utc)

        class DummySessionDB:
            def execute(self, *_args, **_kwargs):
                raise AssertionError('session lookup must not issue DELETE')

            def scalar(self, statement, *_args, **_kwargs):
                sql = str(statement)
                self.assert_in_sql = 'user_sessions.expires_at >=' in sql
                return 'user'

        db = DummySessionDB()
        user = session_repository.get_user_by_session_id(db, 'sess-1', now=now)
        self.assertEqual(user, 'user')
        self.assertTrue(db.assert_in_sql)


    def test_security_headers_cover_api_static_and_rejected_requests(self) -> None:
        client = TestClient(application_core.create_app())
        responses = (
            client.get('/api/health/live'),
            client.get('/'),
            client.post(
                '/api/auth/logout',
                headers={'origin': 'https://attacker.invalid'},
            ),
        )
        for response in responses:
            self.assertEqual(response.headers.get('x-content-type-options'), 'nosniff')
            self.assertEqual(response.headers.get('x-frame-options'), 'DENY')
            self.assertEqual(response.headers.get('referrer-policy'), 'no-referrer')
            self.assertIn("frame-ancestors 'none'", response.headers.get('content-security-policy', ''))
            self.assertIn('microphone=(self)', response.headers.get('permissions-policy', ''))


    def test_ws_transcribe_rejects_unauthenticated_connection(self) -> None:
        app = FastAPI()
        app.include_router(ws_routes.router)
        client = TestClient(app)

        with patch.object(ws_routes, 'get_optional_user_from_request', return_value=None):
            with self.assertRaises(Exception) as ctx:
                with client.websocket_connect('/ws/transcribe'):
                    pass

        self.assertEqual(getattr(ctx.exception, 'code', None), 4401)


    def test_ws_transcribe_allows_authenticated_active_user(self) -> None:
        app = FastAPI()
        app.include_router(ws_routes.router)
        client = TestClient(app)
        user = SimpleNamespace(id=42, is_active=True)

        async def accept_and_close(ws):
            await ws.accept()
            await ws.close()

        with (
            patch.object(ws_routes, 'get_optional_user_from_request', return_value=user),
            patch.object(ws_routes.runtime, 'ws_transcribe', side_effect=accept_and_close) as handler,
        ):
            with client.websocket_connect('/ws/transcribe'):
                pass

        handler.assert_called_once()


    def test_ws_transcribe_rejects_cross_origin_before_authentication(self) -> None:
        app = FastAPI()
        app.include_router(ws_routes.router)
        client = TestClient(app)

        with patch.object(ws_routes, 'get_optional_user_from_request') as auth_lookup:
            with self.assertRaises(Exception) as ctx:
                with client.websocket_connect(
                    '/ws/transcribe',
                    headers={'origin': 'https://attacker.example'},
                ):
                    pass

        self.assertEqual(getattr(ctx.exception, 'code', None), 4403)
        auth_lookup.assert_not_called()


    def test_ws_transcribe_allows_bounded_guest_when_enabled(self) -> None:
        app = FastAPI()
        app.include_router(ws_routes.router)
        client = TestClient(app)
        guest_settings = SimpleNamespace(
            allow_guest_transcription=True,
            guest_ws_max_connections=10,
            guest_ws_max_per_ip=2,
            guest_ws_max_duration_seconds=30,
        )

        async def accept_and_close(ws):
            self.assertTrue(ws.state.is_guest)
            await ws.accept()
            await ws.close()

        with (
            patch.object(ws_routes, 'settings', guest_settings),
            patch.object(ws_routes, 'get_optional_user_from_request', return_value=None),
            patch.object(ws_routes, 'read_guest_artifact_grant', return_value='guest-grant'),
            patch.object(ws_routes.runtime, 'ws_transcribe', side_effect=accept_and_close) as handler,
        ):
            with client.websocket_connect('/ws/transcribe'):
                pass

        handler.assert_called_once()


    def test_ws_transcribe_rejects_inactive_user(self) -> None:
        app = FastAPI()
        app.include_router(ws_routes.router)
        client = TestClient(app)
        user = SimpleNamespace(id=42, is_active=False)

        with patch.object(ws_routes, 'get_optional_user_from_request', return_value=user):
            with self.assertRaises(Exception) as ctx:
                with client.websocket_connect('/ws/transcribe'):
                    pass

        self.assertEqual(getattr(ctx.exception, 'code', None), 4403)


    def test_auth_me_marks_and_clears_an_invalid_session_cookie(self) -> None:
        request = Request(
            {
                'type': 'http',
                'method': 'GET',
                'path': '/api/auth/me',
                'headers': [(b'cookie', b'whistx_session=stale-session')],
                'client': ('127.0.0.1', 12345),
                'query_string': b'',
            }
        )
        with patch.object(auth_service, 'get_optional_user_from_request', return_value=None):
            with patch.object(auth_service.auth, 'has_admin_account', return_value=True):
                with patch.object(
                    auth_service,
                    'settings',
                    SimpleNamespace(
                        enable_self_signup=True,
                        allow_guest_transcription=True,
                        history_retention_days=7,
                        keycloak_enabled=False,
                        keycloak_issuer='',
                        keycloak_client_id='',
                        keycloak_button_label='Keycloakでログイン',
                    ),
                ):
                    payload = auth_service.build_auth_me_payload(request, SimpleNamespace())
        self.assertFalse(payload['authenticated'])
        self.assertTrue(payload['sessionInvalid'])

        app = FastAPI()
        app.include_router(auth_routes.router)
        app.dependency_overrides[auth_routes.get_db] = lambda: DummyDB()
        with patch.object(auth_routes, 'build_auth_me_payload', return_value=payload):
            response = TestClient(app).get('/api/auth/me', cookies={'whistx_session': 'stale-session'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get('cache-control'), 'no-store, private')
        self.assertEqual(response.headers.get('pragma'), 'no-cache')
        self.assertEqual(response.headers.get('vary'), 'Cookie')
        self.assertIn('whistx_session=', response.headers.get('set-cookie', ''))
        self.assertIn('Max-Age=0', response.headers.get('set-cookie', ''))


    def test_auth_me_guest_grant_cookie_is_available_to_websocket(self) -> None:
        app = FastAPI()
        app.include_router(auth_routes.router)
        app.dependency_overrides[auth_routes.get_db] = lambda: DummyDB()
        payload = {
            'authenticated': False,
            'sessionInvalid': False,
            'user': None,
            'guestTranscriptionAllowed': True,
        }

        with patch.object(auth_routes, 'build_auth_me_payload', return_value=payload):
            response = TestClient(app).get('/api/auth/me')

        self.assertEqual(response.status_code, 200)
        cookie = response.headers.get('set-cookie', '')
        self.assertIn('whistx_guest_artifact=', cookie)
        self.assertIn('Path=/', cookie)
        self.assertNotIn('Path=/api/', cookie)


    def test_keycloak_login_respects_forwarded_https_headers(self) -> None:
        app = FastAPI()
        app.include_router(auth_routes.router)

        with (
            patch.object(
                runtime,
                'settings',
                SimpleNamespace(
                    keycloak_enabled=True,
                    keycloak_issuer='https://idp.example.com/realms/test',
                    keycloak_client_id='client-id',
                ),
            ),
            patch.object(
                security,
                'settings',
                make_security_settings(
                    app_trust_proxy_headers=True,
                    app_trusted_proxy_ips=('*',),
                ),
            ),
        ):
            with patch.object(runtime, '_get_keycloak_discovery', return_value={'authorization_endpoint': 'https://idp.example.com/auth'}):
                with patch.object(
                    runtime,
                    '_build_keycloak_authorization_url',
                    side_effect=lambda **kwargs: f"https://idp.example.com/auth?redirect_uri={kwargs['redirect_uri']}",
                ):
                    with TestClient(app) as client:
                        response = client.get(
                            '/api/auth/keycloak/login',
                            follow_redirects=False,
                            headers={
                                'host': 'internal:8005',
                                'x-forwarded-proto': 'https',
                                'x-forwarded-host': 'app.example.com',
                            },
                        )

        self.assertEqual(response.status_code, 302)
        self.assertIn('https://app.example.com/api/auth/keycloak/callback', response.headers['location'])
        self.assertIn('Secure', response.headers.get('set-cookie', ''))


    def test_session_cookie_helper_marks_forwarded_https_as_secure(self) -> None:
        request = Request(
            {
                'type': 'http',
                'method': 'GET',
                'path': '/api/auth/login',
                'headers': [
                    (b'host', b'internal:8005'),
                    (b'x-forwarded-proto', b'https'),
                    (b'x-forwarded-host', b'app.example.com'),
                ],
                'client': ('127.0.0.1', 12345),
                'query_string': b'',
                'scheme': 'http',
            }
        )
        response = summary_routes.JSONResponse({'ok': True})

        with patch.object(
            security,
            'settings',
            make_security_settings(
                app_trust_proxy_headers=True,
                app_trusted_proxy_ips=('127.0.0.1',),
            ),
        ):
            security.set_session_cookie(
                response=response,
                request=request,
                cookie_name='whistx_session',
                session_id='session-1',
            )

        self.assertIn('Secure', response.headers.get('set-cookie', ''))


    def test_untrusted_forwarded_headers_are_ignored(self) -> None:
        request = Request(
            {
                'type': 'http',
                'method': 'GET',
                'path': '/',
                'headers': [
                    (b'host', b'internal:8005'),
                    (b'x-forwarded-for', b'198.51.100.7'),
                    (b'x-forwarded-proto', b'https'),
                    (b'x-forwarded-host', b'evil.example'),
                ],
                'client': ('203.0.113.10', 12345),
                'query_string': b'',
                'scheme': 'http',
            }
        )
        with patch.object(
            security,
            'settings',
            make_security_settings(
                app_trust_proxy_headers=True,
                app_trusted_proxy_ips=('127.0.0.1',),
            ),
        ):
            self.assertEqual(security.client_ip(request), '203.0.113.10')
            self.assertFalse(security.request_is_secure(request))


    def test_session_tokens_are_hashed_before_database_storage(self) -> None:
        class CaptureDB:
            stored = None

            def execute(self, _statement):
                return None

            def add(self, value):
                self.stored = value

            def flush(self):
                return None

        db = CaptureDB()
        raw_token = auth_module.create_user_session(
            db,
            user=SimpleNamespace(id=7),
            user_agent='test',
            ip_address='127.0.0.1',
        )

        self.assertNotEqual(raw_token, db.stored.id)
        self.assertEqual(db.stored.id, auth_module.hash_session_id(raw_token))
        self.assertEqual(len(db.stored.id), 64)


    def test_session_lookup_hashes_cookie_token(self) -> None:
        with patch.object(session_repository, 'get_user_by_session_id', return_value=None) as lookup:
            auth_module.get_user_by_session_id(SimpleNamespace(), 'raw-cookie-token')

        self.assertEqual(lookup.call_args.args[1], auth_module.hash_session_id('raw-cookie-token'))


    def test_origin_validation_rejects_cross_origin_mutation(self) -> None:
        request = Request(
            {
                'type': 'http',
                'method': 'POST',
                'path': '/api/summarize',
                'headers': [(b'host', b'app.example.com'), (b'origin', b'https://evil.example')],
                'client': ('127.0.0.1', 12345),
                'query_string': b'',
                'scheme': 'https',
            }
        )
        with patch.object(
            security,
            'settings',
            make_security_settings(app_public_url='https://app.example.com'),
        ):
            self.assertFalse(security.origin_is_allowed(request))


    def test_costly_api_rate_limit_is_per_user_and_bucket(self) -> None:
        self.assertTrue(rate_limit.consume(bucket='summary', subject='user:1', limit=1, window_seconds=60))
        self.assertFalse(rate_limit.consume(bucket='summary', subject='user:1', limit=1, window_seconds=60))
        self.assertTrue(rate_limit.consume(bucket='proofread', subject='user:1', limit=1, window_seconds=60))
        self.assertTrue(rate_limit.consume(bucket='summary', subject='user:2', limit=1, window_seconds=60))


    def test_guest_connection_quota_is_shared_and_released(self) -> None:
        first = quota_repository.acquire_connection_lease(
            subject='ip:192.0.2.10',
            is_guest=True,
            ttl_seconds=60,
            guest_total_limit=2,
            guest_subject_limit=1,
        )
        self.assertIsNotNone(first)
        self.assertIsNone(
            quota_repository.acquire_connection_lease(
                subject='ip:192.0.2.10',
                is_guest=True,
                ttl_seconds=60,
                guest_total_limit=2,
                guest_subject_limit=1,
            )
        )
        second = quota_repository.acquire_connection_lease(
            subject='ip:192.0.2.11',
            is_guest=True,
            ttl_seconds=60,
            guest_total_limit=2,
            guest_subject_limit=1,
        )
        self.assertIsNotNone(second)
        self.assertEqual(quota_repository.count_active_connections(), 2)
        self.assertIsNone(
            quota_repository.acquire_connection_lease(
                subject='ip:192.0.2.12',
                is_guest=True,
                ttl_seconds=60,
                guest_total_limit=2,
                guest_subject_limit=1,
            )
        )
        quota_repository.release_connection_lease(first)
        quota_repository.release_connection_lease(second)
        self.assertEqual(quota_repository.count_active_connections(), 0)


    def test_password_change_revokes_sessions_and_rotates_cookie_token(self) -> None:
        user = SimpleNamespace(id=7, password_hash='old-hash')
        request = make_request()
        db = SimpleNamespace(commit=lambda: None)
        with (
            patch.object(auth_module, 'verify_password', return_value=True),
            patch.object(auth_module, 'hash_password', return_value='new-hash'),
            patch.object(auth_module, 'delete_all_user_sessions') as revoke,
            patch.object(auth_module, 'create_user_session', return_value='new-cookie-token'),
        ):
            token = auth_service.change_password(
                user=user,
                current_password='old-password',
                new_password='new-password',
                request=request,
                db=db,
            )

        self.assertEqual(user.password_hash, 'new-hash')
        self.assertEqual(token, 'new-cookie-token')
        revoke.assert_called_once_with(db, 7)


    def test_public_url_is_canonical_for_oidc_urls(self) -> None:
        app = FastAPI()

        @app.get('/callback', name='callback')
        def callback():
            return {}

        request = Request(
            {
                'type': 'http',
                'method': 'GET',
                'path': '/',
                'headers': [(b'host', b'evil.example')],
                'client': ('203.0.113.10', 12345),
                'query_string': b'',
                'scheme': 'http',
                'router': app.router,
            }
        )
        with patch.object(
            security,
            'settings',
            make_security_settings(app_public_url='https://app.example.com'),
        ):
            self.assertEqual(
                security.external_url_for(request, 'callback'),
                'https://app.example.com/callback',
            )

