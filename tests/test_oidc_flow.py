import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('APP_SESSION_SECRET', 'oidc-flow-tests-secret-abcdefghijklmnopqrstuvwxyz')
from server.services import oidc_flow as service


class OIDCFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = SimpleNamespace(keycloak_enabled=True, keycloak_issuer='https://synthetic.test/realm', keycloak_client_id='synthetic')
        self.flow = service.OIDCFlow()

    def test_discovery_cache_belongs_to_each_flow(self):
        with patch.object(service, 'settings', self.config), patch.object(service, 'oidc_fetch_discovery', return_value={'synthetic': True}) as fetch:
            service._get_keycloak_discovery(self.flow)
            service._get_keycloak_discovery(self.flow)
            self.assertEqual(fetch.call_count, 1)
            service._get_keycloak_discovery(service.OIDCFlow())
            self.assertEqual(fetch.call_count, 2)

    async def test_invalid_callback_state_does_not_contact_provider_or_commit(self):
        request = SimpleNamespace(query_params={'state': 'wrong', 'code': 'synthetic'})
        db = Mock()
        with patch.object(service, 'settings', self.config), \
             patch.object(service, '_read_oidc_state_cookie', return_value={'state': 'expected'}), \
             patch.object(service, '_clear_oidc_state_cookie'), \
             patch.object(service, '_get_keycloak_discovery') as discovery:
            response = await service.auth_keycloak_callback(request, db, flow=self.flow)
        self.assertEqual(response.headers['location'], '/?authError=keycloak_state')
        discovery.assert_not_called()
        db.commit.assert_not_called()

    async def test_successful_callback_commits_before_setting_session_cookie(self):
        request = SimpleNamespace(query_params={'state': 'expected', 'code': 'synthetic'}, headers={})
        db = Mock()
        user = SimpleNamespace(last_login_at=None)
        def cookie(response, request, identifier):
            db.commit.assert_called_once()
            self.assertEqual(identifier, 'synthetic-session')
        with patch.object(service, 'settings', self.config), \
             patch.object(service, '_read_oidc_state_cookie', return_value={'state': 'expected', 'redirect_uri': 'https://synthetic.test/cb', 'code_verifier': 'synthetic'}), \
             patch.object(service, '_clear_oidc_state_cookie'), \
             patch.object(service, '_get_keycloak_discovery', return_value={}), \
             patch.object(service, '_exchange_keycloak_code', return_value={'access_token': 'synthetic'}), \
             patch.object(service, '_fetch_keycloak_userinfo', return_value={}), \
             patch.object(service, '_upsert_keycloak_user', return_value=user), \
             patch.object(service, 'security_client_ip', return_value='127.0.0.1'), \
             patch.object(service, 'create_user_session', return_value='synthetic-session'), \
             patch.object(service, '_set_session_cookie', side_effect=cookie):
            response = await service.auth_keycloak_callback(request, db, flow=self.flow)
        self.assertEqual(response.headers['location'], '/')
        self.assertIsNotNone(user.last_login_at)
        db.rollback.assert_not_called()
