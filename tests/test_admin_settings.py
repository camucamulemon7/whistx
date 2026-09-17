from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'test-session-secret-abcdefghijklmnopqrstuvwxyz12')

from fastapi import FastAPI
from fastapi.testclient import TestClient
from server import deps
from server.db import get_db
from server.api.routes.admin import router
from server.core.config.overrides import read_overrides, save_overrides, load_overrides
from server.services import history_service


class AdminSettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = patch.dict(os.environ, {'APP_DATA_DIR': self.directory.name, 'ASR_API_KEY': 'test-secret'})
        self.env.start()
        self.addCleanup(self.env.stop)
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: None
        self.client = TestClient(app)

    def test_permissions_and_secrets(self):
        with patch.object(deps, 'get_optional_user', return_value=None):
            self.assertEqual(self.client.get('/api/admin/settings').status_code, 401)
        with patch.object(deps, 'get_optional_user', return_value=SimpleNamespace(is_admin=False)):
            self.assertEqual(self.client.put('/api/admin/settings', json={}).status_code, 403)
        save_overrides({'ASR_API_KEY': 'test-secret'})
        with patch.object(deps, 'get_optional_user', return_value=SimpleNamespace(is_admin=True)):
            response = self.client.get('/api/admin/settings')
            self.assertNotIn('test-secret', response.text)
            self.assertTrue(response.json()['configuredSecrets']['ASR_API_KEY'])
            self.assertEqual(self.client.put('/api/admin/settings', json={'APP_SESSION_SECRET': 'bad'}).status_code, 400)
            self.assertEqual(self.client.put('/api/admin/settings', json={'HISTORY_RETENTION_DAYS': '0'}).status_code, 200)

    def test_persistence_and_blank_credential(self):
        save_overrides({'ASR_API_KEY': 'saved-secret', 'HISTORY_RETENTION_DAYS': '0'})
        save_overrides({'ASR_API_KEY': '', 'ASR_MODEL': 'Qwen3-ASR-1.7B'})
        self.assertEqual(read_overrides()['ASR_API_KEY'], 'saved-secret')
        self.assertEqual((Path(self.directory.name) / 'admin-settings.json').stat().st_mode & 0o777, 0o600)
        load_overrides()
        self.assertEqual(os.environ['ASR_API_KEY'], 'saved-secret')
        self.assertEqual(os.environ['ASR_MODEL'], 'Qwen3-ASR-1.7B')

    def test_unlimited_history_never_queries_expired_rows(self):
        with patch.object(history_service, 'settings', SimpleNamespace(history_retention_days=0)):
            self.assertEqual(history_service.cleanup_expired_histories(None), 0)


    def test_container_gateway_maps_saved_urls_without_changing_file(self):
        save_overrides({'ASR_BASE_URL': 'http://localhost:4000/v1', 'SUMMARY_BASE_URL': 'https://[::1]:4001/v1'})
        with patch.dict(os.environ, {'APP_CONTAINER_HOST_GATEWAY': 'host.containers.internal'}):
            load_overrides()
            self.assertEqual(os.environ['ASR_BASE_URL'], 'http://host.containers.internal:4000/v1')
            self.assertEqual(os.environ['SUMMARY_BASE_URL'], 'https://host.containers.internal:4001/v1')
        self.assertEqual(read_overrides()['ASR_BASE_URL'], 'http://localhost:4000/v1')
        with patch.dict(os.environ, {'APP_CONTAINER_HOST_GATEWAY': ''}):
            load_overrides()
            self.assertEqual(os.environ['ASR_BASE_URL'], 'http://localhost:4000/v1')
