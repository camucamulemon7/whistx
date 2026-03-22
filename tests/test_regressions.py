from __future__ import annotations

import os
import asyncio
import re
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if 'argon2' not in sys.modules:
    argon2_mod = types.ModuleType('argon2')
    argon2_exc = types.ModuleType('argon2.exceptions')

    class VerifyMismatchError(Exception):
        pass

    class PasswordHasher:
        def hash(self, password: str) -> str:
            return f'hashed:{password}'

        def verify(self, password_hash: str, password: str) -> bool:
            return password_hash == f'hashed:{password}'

    argon2_mod.PasswordHasher = PasswordHasher
    argon2_exc.VerifyMismatchError = VerifyMismatchError
    sys.modules['argon2'] = argon2_mod
    sys.modules['argon2.exceptions'] = argon2_exc

from fastapi import Request
import httpx
from openai import APIConnectionError, BadRequestError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import app as app_module
from server import legacy_app
from server import openai_whisper
from server.asr import ASRChunkResult
from server.app import LoginRequest
from server.core.config.asr import load_asr_config
from server.models import TranscriptHistory, TranscriptSegment, User
from server.services import auth_service, glossary_service, history_service
from server.transcript_store import read_jsonl_records


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
        app_module.LOGIN_ATTEMPTS.clear()

    def test_login_route_rate_limits_after_repeated_failures(self) -> None:
        payload = LoginRequest(email='user@example.com', password='wrongpass')
        request = make_request()
        db = DummyDB()

        with patch.object(app_module, 'get_user_by_email', return_value=None):
            for _ in range(5):
                response = asyncio.run(app_module.auth_login(payload, request, db))
                self.assertEqual(response.status_code, 401)

            response = asyncio.run(app_module.auth_login(payload, request, db))
            self.assertEqual(response.status_code, 429)
            self.assertIn('too_many_login_attempts', response.body.decode('utf-8'))

    def test_asr_retry_config_is_loaded_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                'ASR_RETRY_MAX_ATTEMPTS': '5',
                'ASR_RETRY_BASE_DELAY_MS': '250',
                'ASR_RESCUE_RETRY_ENABLED': '1',
                'ASR_RESCUE_RETRY_TEMPERATURE': '0.35',
            },
            clear=False,
        ):
            config = load_asr_config()

        self.assertEqual(config.asr_retry_max_attempts, 5)
        self.assertEqual(config.asr_retry_base_delay_ms, 250)
        self.assertTrue(config.asr_rescue_retry_enabled)
        self.assertEqual(config.asr_rescue_retry_temperature, 0.35)

    def test_asr_context_defaults_are_expanded(self) -> None:
        config = load_asr_config()
        self.assertEqual(config.context_recent_lines, 4)
        self.assertEqual(config.context_max_chars, 2200)
        self.assertEqual(config.context_term_limit, 80)

    def test_shared_glossary_service_persists_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(glossary_service, 'settings', SimpleNamespace(app_data_dir=Path(tmpdir))):
                payload = glossary_service.save_shared_glossary(text='PCIe, UCIe', updated_by='user@example.com')
                loaded = glossary_service.load_shared_glossary()

        self.assertEqual(payload['text'], 'PCIe, UCIe')
        self.assertEqual(loaded['text'], 'PCIe, UCIe')
        self.assertEqual(loaded['updatedBy'], 'user@example.com')
        self.assertTrue(loaded['updatedAt'])

    def test_keycloak_upsert_rejects_unverified_email_when_required(self) -> None:
        db = DummyDB()
        userinfo = {'sub': 'sub-1', 'email': 'user@example.com', 'email_verified': None}

        with patch.object(app_module, 'settings', SimpleNamespace(keycloak_require_email_verified=True)):
            with self.assertRaises(RuntimeError) as ctx:
                app_module._upsert_keycloak_user(db, userinfo)

        self.assertEqual(str(ctx.exception), 'keycloak_email_not_verified')

    def test_history_save_rejects_unfinalized_runtime_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcripts_dir = root / 'transcripts'
            history_dir = root / 'history'
            transcripts_dir.mkdir()
            history_dir.mkdir()

            session_id = 'sess-1'
            (transcripts_dir / f'{session_id}.txt').write_text('hello\n', encoding='utf-8')
            (transcripts_dir / f'{session_id}.jsonl').write_text(
                '{"type":"final","seq":0,"text":"hello","tsStart":0,"tsEnd":100}\n',
                encoding='utf-8',
            )
            (transcripts_dir / f'{session_id}.meta.json').write_text(
                '{"finalized":false,"accessToken":"token"}',
                encoding='utf-8',
            )

            user = User(
                id=1,
                email='user@example.com',
                password_hash='hash',
                is_active=True,
                is_admin=False,
            )

            with patch.object(
                history_service,
                'settings',
                SimpleNamespace(transcripts_dir=transcripts_dir, history_dir=history_dir),
            ):
                with patch.object(history_service, 'is_runtime_transcript_finalized', return_value=False):
                    with self.assertRaises(history_service.HistoryError) as ctx:
                        history_service.save_history(
                            db=SimpleNamespace(scalar=lambda *_: None, add=lambda *_: None, flush=lambda: None),
                            user=user,
                            runtime_session_id=session_id,
                            runtime_session_token='token',
                            title=None,
                            summary_text=None,
                            proofread_text=None,
                        )

            self.assertEqual(ctx.exception.code, 'runtime_session_not_finalized')
            self.assertEqual(ctx.exception.status_code, 409)

    def test_runtime_snapshot_parse_error_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcripts_dir = root / 'transcripts'
            transcripts_dir.mkdir()

            session_id = 'sess-parse'
            (transcripts_dir / f'{session_id}.txt').write_text('hello\n', encoding='utf-8')
            (transcripts_dir / f'{session_id}.jsonl').write_text('{"type":"final"}\n{broken}\n', encoding='utf-8')

            with patch.object(
                history_service,
                'settings',
                SimpleNamespace(transcripts_dir=transcripts_dir),
            ):
                with self.assertRaises(history_service.HistoryError) as ctx:
                    history_service.load_runtime_snapshot(session_id)

            self.assertEqual(ctx.exception.code, 'transcript_parse_failed')
            self.assertEqual(ctx.exception.status_code, 500)

    def test_history_detail_payload_falls_back_to_runtime_screenshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_dir = root / 'history'
            screenshots_dir = history_dir / '1' / 'hist-1' / 'screenshots'
            screenshots_dir.mkdir(parents=True)
            (screenshots_dir / '000001.png').write_bytes(b'png')
            (screenshots_dir / '000001.webp').write_bytes(b'webp')

            saved_at = datetime.now(timezone.utc)
            history = TranscriptHistory(
                id='hist-1',
                user_id=1,
                runtime_session_id='sess-1',
                title='sample',
                language='ja',
                audio_source='mic',
                segment_count=1,
                plain_text='hello',
                summary_text=None,
                proofread_text=None,
                has_diarization=False,
                artifact_dir='1/hist-1',
                txt_path='1/hist-1/transcript.txt',
                jsonl_path='1/hist-1/transcript.jsonl',
                zip_path='1/hist-1/transcript.zip',
                created_at=saved_at,
                updated_at=saved_at,
                saved_at=saved_at,
            )
            history.segments = [
                TranscriptSegment(
                    seq=1,
                    segment_id=None,
                    text='hello',
                    ts_start=0,
                    ts_end=100,
                    chunk_offset_ms=None,
                    chunk_duration_ms=None,
                    language='ja',
                    speaker=None,
                    screenshot_path=None,
                    created_at=saved_at,
                )
            ]

            with patch.object(
                history_service,
                'settings',
                SimpleNamespace(history_dir=history_dir),
            ):
                payload = history_service.build_history_detail_payload(history)

            screenshot_url = payload['segments'][0]['screenshotUrl']
            self.assertIsNotNone(screenshot_url)
            self.assertTrue(screenshot_url.endswith('/000001.webp'))

    def test_jsonl_strict_reader_raises_on_invalid_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'broken.jsonl'
            path.write_text('{"type":"final"}\n{broken}\n', encoding='utf-8')

            with self.assertRaises(ValueError):
                read_jsonl_records(path, strict=True)

    def test_keycloak_error_mapping_is_specific(self) -> None:
        self.assertEqual(
            app_module._map_keycloak_auth_error(RuntimeError('keycloak_email_not_verified')),
            'keycloak_email_not_verified',
        )
        self.assertEqual(
            app_module._map_keycloak_auth_error(RuntimeError('keycloak_account_link_required')),
            'keycloak_account_link_required',
        )
        self.assertEqual(app_module._map_keycloak_auth_error(RuntimeError('something_else')), 'keycloak_failed')

    def test_history_file_path_rejects_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir) / 'history'
            history_dir.mkdir()
            history = TranscriptHistory(
                id='hist-escape',
                user_id=1,
                runtime_session_id='sess-escape',
                title='escape',
                language='ja',
                audio_source='mic',
                segment_count=0,
                plain_text='x',
                summary_text=None,
                proofread_text=None,
                has_diarization=False,
                artifact_dir='1/hist-escape',
                txt_path='../outside.txt',
                jsonl_path='1/hist-escape/transcript.jsonl',
                zip_path='1/hist-escape/transcript.zip',
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                saved_at=datetime.now(timezone.utc),
            )
            with patch.object(history_service, 'settings', SimpleNamespace(history_dir=history_dir)):
                self.assertIsNone(history_service.get_history_file_path(history, history.txt_path))

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
                            keycloak_enabled=False,
                            keycloak_issuer='',
                            keycloak_client_id='',
                            keycloak_button_label='Keycloakでログイン',
                        ),
                    ):
                        payload = auth_service.build_auth_me_payload(request, SimpleNamespace())
        self.assertTrue(payload['authenticated'])
        self.assertEqual(payload['pendingApprovalCount'], 3)

    def test_trim_overlap_prefix_requires_substantial_match(self) -> None:
        previous = '本日の会議では新製品の価格改定について説明します'
        current = '価格改定について説明します。次に販売計画を確認します'
        self.assertEqual(legacy_app._trim_overlap_prefix(current, previous), '次に販売計画を確認します')

    def test_trim_overlap_prefix_fuzzy_match_handles_small_variation(self) -> None:
        previous = '新製品の価格改定について説明いたします'
        current = '価格改定について説明します。次に販売計画です'
        self.assertEqual(legacy_app._trim_overlap_prefix(current, previous), '次に販売計画です')

    def test_build_prompt_includes_shared_vocabulary(self) -> None:
        session = SimpleNamespace(
            base_prompt='会議用語を優先してください',
            shared_vocabulary='PCIe, UCIe, Blackwell',
            context_prompt_enabled=True,
            context_history=['次回は PCIe 帯域を確認します'],
            context_terms=['帯域', 'Gen6'],
            context_max_chars=400,
            language='ja',
        )
        prompt = legacy_app._build_prompt(session)
        self.assertIsNotNone(prompt)
        self.assertIn('共有用語辞典', prompt)
        self.assertIn('PCIe, UCIe, Blackwell', prompt)
        self.assertIn('利用者プロンプト', prompt)

    def test_openai_whisper_retries_retryable_errors(self) -> None:
        request = httpx.Request('POST', 'https://example.com/v1/audio/transcriptions')
        response = SimpleNamespace(
            text='hello world',
            segments=[{'start': 0.0, 'end': 1.2, 'no_speech_prob': 0.01}],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )
        create_calls = []

        def create(**kwargs):
            create_calls.append(kwargs)
            if len(create_calls) <= 2:
                raise APIConnectionError(message='temporary', request=request)
            return response

        transcriber = openai_whisper.OpenAIWhisperTranscriber(
            api_key='test-key',
            base_url=None,
            model='whisper-1',
            observer=None,
        )
        transcriber.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))

        transcriber.retry_max_attempts = 3
        transcriber.retry_base_delay_ms = 50
        with patch.object(openai_whisper.time, 'sleep', autospec=True) as sleep_mock:
            result = transcriber.transcribe_chunk(
                b'abc',
                mime_type='audio/webm',
                language='ja',
                prompt=None,
                temperature=0.0,
            )

        self.assertEqual(result.text, 'hello world')
        self.assertEqual(len(create_calls), 3)
        self.assertEqual(sleep_mock.call_count, 2)
        self.assertEqual(sleep_mock.call_args_list[0].args[0], 0.05)
        self.assertEqual(sleep_mock.call_args_list[1].args[0], 0.1)

    def test_openai_whisper_extracts_confidence_metrics(self) -> None:
        response = SimpleNamespace(
            text='短い定型文です',
            segments=[
                {
                    'start': 0.0,
                    'end': 1.0,
                    'no_speech_prob': 0.91,
                    'avg_logprob': -1.2,
                    'compression_ratio': 2.6,
                }
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )

        transcriber = openai_whisper.OpenAIWhisperTranscriber(
            api_key='test-key',
            base_url=None,
            model='whisper-1',
            observer=None,
        )
        transcriber.client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=lambda **_kwargs: response))
        )

        result = transcriber.transcribe_chunk(
            b'abc',
            mime_type='audio/webm',
            language='ja',
            prompt=None,
            temperature=0.0,
        )

        self.assertTrue(result.suspicious)
        self.assertAlmostEqual(result.max_no_speech_prob or 0.0, 0.91)
        self.assertAlmostEqual(result.avg_logprob or 0.0, -1.2)
        self.assertAlmostEqual(result.compression_ratio or 0.0, 2.6)

    def test_openai_whisper_multi_pass_prefers_longer_retry_result(self) -> None:
        first_response = SimpleNamespace(
            text='短い候補',
            segments=[{'start': 0.0, 'end': 0.6, 'no_speech_prob': 0.65, 'avg_logprob': -1.1, 'compression_ratio': 2.5}],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )
        second_response = SimpleNamespace(
            text='短い候補ではなく十分に長い改善結果です',
            segments=[{'start': 0.0, 'end': 1.2, 'no_speech_prob': 0.2, 'avg_logprob': -0.2, 'compression_ratio': 1.2}],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return first_response if len(calls) == 1 else second_response

        transcriber = openai_whisper.OpenAIWhisperTranscriber(
            api_key='test-key',
            base_url=None,
            model='whisper-1',
            observer=None,
        )
        transcriber.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
        transcriber.multi_pass_enabled = True

        result = transcriber.transcribe_chunk(
            b'abc',
            mime_type='audio/webm',
            language='ja',
            prompt=None,
            temperature=0.0,
        )

        self.assertEqual(len(calls), 2)
        self.assertIn('改善結果', result.text)

    def test_openai_whisper_does_not_retry_non_retryable_errors(self) -> None:
        request = httpx.Request('POST', 'https://example.com/v1/audio/transcriptions')
        response = httpx.Response(400, request=request, content=b'{}')

        def create(**kwargs):
            raise BadRequestError(message='bad request', response=response, body=None)

        transcriber = openai_whisper.OpenAIWhisperTranscriber(
            api_key='test-key',
            base_url=None,
            model='whisper-1',
            observer=None,
        )
        transcriber.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))

        transcriber.retry_max_attempts = 3
        transcriber.retry_base_delay_ms = 50
        with patch.object(openai_whisper.time, 'sleep', autospec=True) as sleep_mock:
            with self.assertRaises(BadRequestError):
                transcriber.transcribe_chunk(
                    b'abc',
                    mime_type='audio/webm',
                    language='ja',
                    prompt=None,
                    temperature=0.0,
                )

        self.assertEqual(sleep_mock.call_count, 0)

    def test_near_duplicate_detection_does_not_drop_extended_text(self) -> None:
        previous = '本日の会議では新製品の価格改定について説明します'
        current = '本日の会議では新製品の価格改定について詳細を説明します'
        self.assertFalse(legacy_app._is_near_duplicate(current, previous))

    def test_near_duplicate_detection_uses_timestamp_gap(self) -> None:
        previous = '価格改定について説明します'
        current = '価格改定について説明します'
        self.assertTrue(legacy_app._is_near_duplicate(current, previous, current_start_ms=1000, previous_end_ms=900))
        self.assertFalse(legacy_app._is_near_duplicate(current, previous, current_start_ms=5000, previous_end_ms=900))

    def test_light_proofread_collapses_fillers_and_normalizes_digits(self) -> None:
        value = legacy_app._light_proofread('えーと、えーと ２０ ２５ 年の計画です', language='ja')
        self.assertIn('えーと', value)
        self.assertNotIn('えーと、えーと', value)
        self.assertIn('2025', value)

    def test_boundary_fragment_detection_drops_broken_display_chunk(self) -> None:
        self.assertTrue(
            legacy_app._should_drop_boundary_fragment(
                'おすすめとかえええ\ufffd',
                '有識者のみなさんぜひ教えてくださいよということでお願いしますよお願いしますほなじゃあなんかありますかおすすめとか',
                source_mode='display',
                suspicious=False,
            )
        )

    def test_frontend_vad_uses_soft_target_and_hard_max(self) -> None:
        source = (ROOT / 'web' / 'src' / 'app.js').read_text(encoding='utf-8')
        self.assertIn('const VAD_SOFT_CUT_GRACE_MS = 6_000;', source)
        self.assertIn('function chunkHardMaxMs()', source)
        self.assertIn('if (elapsedMs >= chunkHardMaxMs()) {', source)
        self.assertIn('shouldCutChunkOnSilence({ relaxed: elapsedMs >= state.chunkMs })', source)

    def test_weird_transcription_retry_detection_handles_broken_chunk(self) -> None:
        self.assertTrue(
            legacy_app._should_retry_weird_transcription(
                'おすすめとかえええ\ufffd',
                '有識者のみなさんぜひ教えてくださいよということでお願いしますよお願いしますほなじゃあなんかありますかおすすめとか',
                source_mode='display',
                suspicious=False,
            )
        )

    def test_rescue_transcription_result_prefers_cleaner_retry(self) -> None:
        original = ASRChunkResult(text='おすすめとかえええ\ufffd', start_ms=0, end_ms=1000, suspicious=True)
        retry = ASRChunkResult(text='おすすめとか', start_ms=0, end_ms=1000, suspicious=False)
        self.assertTrue(
            legacy_app._prefer_rescue_transcription_result(
                original=original,
                retry=retry,
                previous_text='有識者のみなさんぜひ教えてくださいよということでお願いしますよお願いしますほなじゃあなんかありますか',
                source_mode='display',
            )
        )

    def test_monotonic_bounds_prevent_timestamp_overlap(self) -> None:
        self.assertEqual(legacy_app._coerce_monotonic_bounds(ts_start=8100, ts_end=8900, previous_end_ms=9000), (9000, 9000))
        self.assertEqual(legacy_app._coerce_monotonic_bounds(ts_start=9100, ts_end=9500, previous_end_ms=9000), (9100, 9500))

    def test_register_user_rejects_when_self_signup_disabled(self) -> None:
        payload = app_module.RegisterRequest(email='user@example.com', password='password123', display_name='User')
        db = SimpleNamespace()
        with patch.object(auth_service.auth, 'has_admin_account', return_value=True):
            with patch.object(auth_service, 'settings', SimpleNamespace(enable_self_signup=False)):
                with self.assertRaises(auth_service.AuthServiceError) as ctx:
                    auth_service.register_user(payload, db)
        self.assertEqual(ctx.exception.code, 'self_signup_disabled')
        self.assertEqual(ctx.exception.status_code, 403)

    def test_frontend_workspace_guard_uses_live_auth_state(self) -> None:
        source = (ROOT / 'web' / 'src' / 'app.js').read_text(encoding='utf-8')
        self.assertIn('import { canUseWorkspace as canUseWorkspaceForAuth, persistGuestMode, readGuestMode, serializeUserLabel } from "./auth/session.js";', source)
        self.assertIn('return canUseWorkspaceForAuth(state.auth);', source)
        self.assertIsNone(re.search(r'canUseWorkspaceForAuth\(\s*\)', source))


if __name__ == '__main__':
    unittest.main()
