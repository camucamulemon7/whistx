from __future__ import annotations

import os
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'test-session-secret-abcdefghijklmnopqrstuvwxyz12')

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.api.routes import health as health_routes
from server.api.routes import summary as summary_routes
from server.core import application as application_core
from server.core.config.app import load_app_config
from server.core import security
from server.core import rate_limit
from server.services import (
    glossary_service,
)
from server.transcription import text_processing






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

    def test_runtime_artifact_access_fails_closed_and_checks_owner_or_guest_grant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts_dir = Path(tmpdir)
            session_id = 'sess-access'
            txt_path = transcripts_dir / f'{session_id}.txt'
            metadata_path = transcripts_dir / f'{session_id}.meta.json'
            txt_path.write_text('secret transcript', encoding='utf-8')

            with patch.object(security, 'settings', make_security_settings()):
                self.assertFalse(
                    security.runtime_access_allowed(
                        transcripts_dir=transcripts_dir,
                        session_id=session_id,
                        user_id=None,
                        guest_grant_id=None,
                    )
                )

                metadata_path.write_text('{invalid', encoding='utf-8')
                self.assertFalse(
                    security.runtime_access_allowed(
                        transcripts_dir=transcripts_dir,
                        session_id=session_id,
                        user_id=1,
                        guest_grant_id=None,
                    )
                )

                metadata_path.write_text('{"finalized":true,"ownerUserId":7}', encoding='utf-8')
                self.assertTrue(
                    security.runtime_access_allowed(
                        transcripts_dir=transcripts_dir,
                        session_id=session_id,
                        user_id=7,
                        guest_grant_id=None,
                    )
                )
                self.assertFalse(
                    security.runtime_access_allowed(
                        transcripts_dir=transcripts_dir,
                        session_id=session_id,
                        user_id=8,
                        guest_grant_id=None,
                    )
                )

                guest_grant = 'guest-browser-grant'
                guest_digest = security.digest_guest_artifact_grant(guest_grant)
                metadata_path.write_text(
                    f'{{"finalized":true,"guestGrantDigest":"{guest_digest}"}}',
                    encoding='utf-8',
                )
                self.assertTrue(
                    security.runtime_access_allowed(
                        transcripts_dir=transcripts_dir,
                        session_id=session_id,
                        user_id=None,
                        guest_grant_id=guest_grant,
                    )
                )
                self.assertFalse(
                    security.runtime_access_allowed(
                        transcripts_dir=transcripts_dir,
                        session_id=session_id,
                        user_id=None,
                        guest_grant_id='another-grant',
                    )
                )


    def test_shared_glossary_service_persists_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(glossary_service, 'settings', SimpleNamespace(app_data_dir=Path(tmpdir))):
                payload = glossary_service.save_shared_glossary(text='PCIe, UCIe', updated_by='user@example.com')
                loaded = glossary_service.load_shared_glossary()

        self.assertEqual(payload['text'], 'PCIe, UCIe')
        self.assertEqual(loaded['text'], 'PCIe, UCIe')
        self.assertEqual(loaded['updatedBy'], 'user@example.com')
        self.assertTrue(loaded['updatedAt'])


    def test_shared_glossary_replacements_apply_alias_mapping(self) -> None:
        text = 'なんど と らすこ を確認しました'
        glossary_text = 'なんど=NAND\nらすこ=Lascaux'

        replaced = glossary_service.apply_shared_glossary_replacements(text, glossary_text)

        self.assertEqual(replaced, 'NAND と Lascaux を確認しました')


    def test_summary_route_requires_login(self) -> None:
        app = FastAPI()
        app.include_router(summary_routes.router)
        client = TestClient(app)

        response = client.post('/api/summarize', json={'text': 'hello', 'language': 'ja'})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {'detail': 'login_required'})


    def test_liveness_does_not_depend_on_database_revision(self) -> None:
        with patch.object(
            health_routes,
            'schema_revision_status',
            side_effect=RuntimeError('database unavailable'),
        ):
            response = health_routes.liveness()
        self.assertEqual(response.status_code, 200)


    def test_readiness_reports_schema_revision_mismatch(self) -> None:
        schema = SimpleNamespace(
            ready=False,
            current_revisions=('20260727_0006',),
            expected_revisions=('20260727_0007',),
            error=None,
        )
        with patch.object(health_routes, 'schema_revision_status', return_value=schema):
            response = asyncio.run(health_routes.readiness())
        self.assertEqual(response.status_code, 503)
        self.assertIn('not_ready', response.body.decode('utf-8'))


    def test_load_app_config_rejects_default_or_short_session_secret(self) -> None:
        with patch.dict(os.environ, {'APP_SESSION_SECRET': 'change-me'}, clear=False):
            with self.assertRaises(RuntimeError):
                load_app_config()
        with patch.dict(os.environ, {'APP_SESSION_SECRET': 'too-short'}, clear=False):
            with self.assertRaises(RuntimeError):
                load_app_config()


    def test_load_app_config_rejects_sqlite_in_production(self) -> None:
        with patch.dict(
            os.environ,
            {
                'APP_ENV': 'production',
                'APP_DB_URL': 'sqlite:///data/app.db',
                'APP_SESSION_SECRET': 'test-session-secret-abcdefghijklmnopqrstuvwxyz12',
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError):
                load_app_config()


    def test_create_app_uses_lifespan_startup_and_shutdown(self) -> None:
        events: list[str] = []

        async def fake_startup() -> None:
            events.append('startup')

        async def fake_shutdown() -> None:
            events.append('shutdown')

        with patch.object(application_core.runtime, 'on_startup', side_effect=fake_startup):
            with patch.object(application_core.runtime, 'on_shutdown', side_effect=fake_shutdown):
                with TestClient(application_core.create_app()):
                    self.assertEqual(events, ['startup'])

        self.assertEqual(events, ['startup', 'shutdown'])


    def test_near_duplicate_detection_does_not_drop_extended_text(self) -> None:
        previous = '本日の会議では新製品の価格改定について説明します'
        current = '本日の会議では新製品の価格改定について詳細を説明します'
        self.assertFalse(text_processing._is_near_duplicate(current, previous))


    def test_near_duplicate_detection_uses_timestamp_gap(self) -> None:
        previous = '価格改定について説明します'
        current = '価格改定について説明します'
        self.assertTrue(
            text_processing._is_near_duplicate(
                current,
                previous,
                current_start_ms=1000,
                previous_end_ms=900,
            )
        )
        self.assertFalse(
            text_processing._is_near_duplicate(
                current,
                previous,
                current_start_ms=5000,
                previous_end_ms=900,
            )
        )

