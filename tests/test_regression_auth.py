from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'test-session-secret-abcdefghijklmnopqrstuvwxyz12')

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.schemas import BootstrapAdminRequest, RegisterRequest
from server.api.routes import admin as admin_routes
from server.api.routes import auth as auth_routes
from server.api.routes import glossary as glossary_routes
from server.api.routes import summary as summary_routes
from server.core import rate_limit
from server.models import User
from server.repositories import session_repository, user_repository
from server.services import (
    admin_service,
    auth_service,
)
from server.deps import get_current_admin, get_current_user


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




class RegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        rate_limit.clear()

    def test_production_admin_bootstrap_requires_trusted_cli(self) -> None:
        payload = BootstrapAdminRequest(
            email='admin@example.com',
            password='secure-password',
            display_name='Administrator',
        )
        with patch.object(auth_service, 'settings', SimpleNamespace(app_env='production')):
            with self.assertRaises(auth_service.AuthServiceError) as ctx:
                auth_service.bootstrap_admin(payload, make_request(), DummyDB())
        self.assertEqual(ctx.exception.code, 'bootstrap_admin_cli_required')
        self.assertEqual(ctx.exception.status_code, 403)


    def test_concurrent_admin_bootstrap_claim_is_rejected(self) -> None:
        payload = BootstrapAdminRequest(
            email='admin@example.com',
            password='secure-password',
            display_name='Administrator',
        )
        conflict = auth_service.IntegrityError('insert bootstrap claim', {}, RuntimeError('unique'))
        with (
            patch.object(auth_service, 'settings', SimpleNamespace(app_env='development')),
            patch.object(auth_service.auth, 'has_admin_account', return_value=False),
            patch.object(auth_service.auth, 'get_user_by_email', return_value=None),
            patch.object(auth_service.user_repository, 'claim_initial_admin_bootstrap', side_effect=conflict),
        ):
            with self.assertRaises(auth_service.AuthServiceError) as ctx:
                auth_service.bootstrap_admin(payload, make_request(), DummyDB())
        self.assertEqual(ctx.exception.code, 'admin_already_exists')
        self.assertEqual(ctx.exception.status_code, 409)


    def test_keycloak_upsert_rejects_unverified_email_when_required(self) -> None:
        db = DummyDB()
        userinfo = {'sub': 'sub-1', 'email': 'user@example.com', 'email_verified': None}

        with patch.object(auth_service, 'settings', SimpleNamespace(keycloak_require_email_verified=True)):
            with self.assertRaises(RuntimeError) as ctx:
                auth_service.upsert_keycloak_user(db, userinfo)

        self.assertEqual(str(ctx.exception), 'keycloak_email_not_verified')


    def test_session_cleanup_reports_batched_delete(self) -> None:
        now = datetime.now(timezone.utc)
        oldest = now - timedelta(days=3)

        class DummyResult:
            rowcount = 25

        class DummySessionDB:
            def scalar(self, *_args, **_kwargs):
                return oldest

            def execute(self, statement, *_args, **_kwargs):
                self.sql = str(statement)
                return DummyResult()

        db = DummySessionDB()
        result = session_repository.prune_expired_sessions(
            db,
            now=now,
            batch_size=25,
        )
        self.assertEqual(result.deleted_count, 25)
        self.assertEqual(result.oldest_expired_at, oldest)
        self.assertIn('LIMIT', db.sql)


    def test_shared_glossary_requires_login_and_admin_for_updates(self) -> None:
        app = FastAPI()
        app.include_router(glossary_routes.router)
        client = TestClient(app)

        self.assertEqual(client.get('/api/glossary/shared').status_code, 401)
        self.assertEqual(client.put('/api/glossary/shared', json={'text': 'secret'}).status_code, 401)

        member = User(
            id=1,
            email='member@example.com',
            password_hash='hash',
            is_active=True,
            is_admin=False,
        )
        app.dependency_overrides[get_current_user] = lambda: member
        with patch.object(
            glossary_routes,
            'load_shared_glossary',
            return_value={'text': 'internal', 'updatedAt': None, 'updatedBy': None},
        ):
            response = client.get('/api/glossary/shared')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['items'], 'internal')

        def reject_member_update():
            raise HTTPException(status_code=403, detail='admin_required')

        app.dependency_overrides[get_current_admin] = reject_member_update
        self.assertEqual(client.put('/api/glossary/shared', json={'text': 'changed'}).status_code, 403)

        admin = User(
            id=2,
            email='admin@example.com',
            password_hash='hash',
            is_active=True,
            is_admin=True,
        )
        app.dependency_overrides[get_current_admin] = lambda: admin
        with patch.object(
            glossary_routes,
            'save_shared_glossary',
            return_value={'text': 'changed', 'updatedAt': 'now', 'updatedBy': admin.email},
        ):
            response = client.put('/api/glossary/shared', json={'text': 'changed'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['items'], 'changed')


    def test_summary_route_allows_authenticated_user(self) -> None:
        app = FastAPI()
        app.include_router(summary_routes.router)
        app.dependency_overrides[get_current_user] = lambda: User(
            id=1,
            email='user@example.com',
            password_hash='hash',
            is_active=True,
            is_admin=False,
        )
        client = TestClient(app)

        async def fake_summarize(payload):
            return summary_routes.JSONResponse({'summary': payload.text, 'model': 'test'})

        with patch.object(summary_routes.runtime, 'summarize', side_effect=fake_summarize):
            response = client.post('/api/summarize', json={'text': 'hello', 'language': 'ja'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['summary'], 'hello')


    def test_auth_profile_update_requires_login(self) -> None:
        app = FastAPI()
        app.include_router(auth_routes.router)
        client = TestClient(app)

        response = client.patch('/api/auth/profile', json={'display_name': 'Updated'})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {'detail': 'login_required'})


    def test_auth_profile_update_allows_authenticated_user(self) -> None:
        app = FastAPI()
        app.include_router(auth_routes.router)
        user = User(
            id=1,
            email='user@example.com',
            password_hash='hash',
            display_name='Before',
            is_active=True,
            is_admin=False,
        )
        app.dependency_overrides[get_current_user] = lambda: user
        client = TestClient(app)

        class DummyDB:
            pass

        app.dependency_overrides[auth_routes.get_db] = lambda: DummyDB()

        with patch.object(auth_routes, 'update_display_name', side_effect=lambda **kwargs: kwargs['user']) as update_mock:
            response = client.patch('/api/auth/profile', json={'display_name': 'Updated'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(update_mock.call_args.kwargs['display_name'], 'Updated')
        self.assertEqual(response.json()['user']['displayName'], 'Before')


    def test_keycloak_error_mapping_is_specific(self) -> None:
        self.assertEqual(
            auth_service.map_keycloak_auth_error(RuntimeError('keycloak_email_not_verified')),
            'keycloak_email_not_verified',
        )
        self.assertEqual(
            auth_service.map_keycloak_auth_error(RuntimeError('keycloak_account_link_required')),
            'keycloak_account_link_required',
        )
        self.assertEqual(auth_service.map_keycloak_auth_error(RuntimeError('something_else')), 'keycloak_failed')


    def test_auth_me_payload_counts_pending_for_admin(self) -> None:
        request = Request(
            {
                'type': 'http',
                'method': 'GET',
                'path': '/api/auth/me',
                'headers': [],
                'client': ('127.0.0.1', 12345),
                'query_string': b'',
            }
        )
        admin_user = User(id=1, email='admin@example.com', password_hash='hash', is_active=True, is_admin=True)
        with patch.object(auth_service, 'get_optional_user_from_request', return_value=admin_user):
            with patch.object(auth_service.auth, 'has_admin_account', return_value=True):
                with patch.object(auth_service.user_repository, 'count_pending_users', return_value=3):
                    with patch.object(
                        auth_service,
                        'settings',
                        SimpleNamespace(
                            enable_self_signup=True,
                            history_retention_days=7,
                            keycloak_enabled=False,
                            keycloak_issuer='',
                            keycloak_client_id='',
                            keycloak_button_label='Keycloakでログイン',
                        ),
                    ):
                        payload = auth_service.build_auth_me_payload(request, SimpleNamespace())
        self.assertTrue(payload['authenticated'])
        self.assertEqual(payload['pendingApprovalCount'], 3)
        self.assertEqual(payload['historyRetentionDays'], 7)
        self.assertFalse(payload['sessionInvalid'])


    def test_admin_users_route_forwards_search_query(self) -> None:
        app = FastAPI()
        app.include_router(admin_routes.router)
        admin_user = User(id=1, email='admin@example.com', password_hash='hash', is_active=True, is_admin=True)
        app.dependency_overrides[admin_routes.get_current_admin] = lambda: admin_user

        class DummyDB:
            pass

        app.dependency_overrides[admin_routes.get_db] = lambda: DummyDB()
        client = TestClient(app)

        with patch.object(admin_routes, 'list_users_payload', return_value={'items': [], 'query': 'alice'}) as payload_mock:
            response = client.get('/api/admin/users?q=alice')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['query'], 'alice')
        self.assertEqual(payload_mock.call_args.kwargs['query'], 'alice')


    def test_admin_service_filters_users_by_display_name_or_email(self) -> None:
        users = [
            User(
                id=1,
                email='alice@example.com',
                password_hash='hash',
                display_name='Alice Admin',
                is_active=True,
                is_admin=True,
                created_at=datetime.now(timezone.utc),
            ),
            User(
                id=2,
                email='bob@example.com',
                password_hash='hash',
                display_name='Bob Member',
                is_active=True,
                is_admin=False,
                created_at=datetime.now(timezone.utc),
            ),
        ]

        with patch.object(admin_service.user_repository, 'search_users', return_value=[users[0]]) as search_mock:
            payload = admin_service.list_users_payload(SimpleNamespace(), query='alice')

        self.assertEqual(payload['query'], 'alice')
        self.assertEqual(len(payload['items']), 1)
        self.assertEqual(payload['items'][0]['email'], 'alice@example.com')
        self.assertEqual(search_mock.call_args.kwargs['query'], 'alice')


    def test_register_user_rejects_when_self_signup_disabled(self) -> None:
        payload = RegisterRequest(email='user@example.com', password='password123', display_name='User')
        db = SimpleNamespace()
        with patch.object(auth_service.auth, 'has_admin_account', return_value=True):
            with patch.object(auth_service, 'settings', SimpleNamespace(enable_self_signup=False)):
                with self.assertRaises(auth_service.AuthServiceError) as ctx:
                    auth_service.register_user(payload, db)
        self.assertEqual(ctx.exception.code, 'self_signup_disabled')
        self.assertEqual(ctx.exception.status_code, 403)


    def test_last_admin_check_locks_admin_rows(self) -> None:
        admin = SimpleNamespace(id=1, is_admin=True)
        with (
            patch.object(user_repository, 'lock_admin_users', return_value=[admin]) as lock_admins,
            patch.object(user_repository, 'get_user_by_id', return_value=admin),
        ):
            with self.assertRaises(admin_service.AdminServiceError) as ctx:
                admin_service.update_user_role(SimpleNamespace(), user_id=1, role='member')

        self.assertEqual(ctx.exception.code, 'last_admin_forbidden')
        lock_admins.assert_called_once()

