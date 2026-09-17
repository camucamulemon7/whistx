from __future__ import annotations

import os
import asyncio
import json
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'test-session-secret-abcdefghijklmnopqrstuvwxyz12')


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.core.config.app import load_app_config
from server.core import rate_limit
from server.models import TranscriptHistory, TranscriptSegment, User
from server.services import (
    glossary_service,
    history_service,
    runtime_artifact_service,
)
from server.transcript_store import read_jsonl_records, _render_txt_line


class TranscriptSpeakerRenderingTests(unittest.TestCase):
    def test_unassigned_speaker_has_no_label(self):
        for speaker in (None, "", "  "):
            self.assertEqual(_render_txt_line({"text": "日本語", "speaker": speaker}), "日本語")
        self.assertEqual(_render_txt_line({"text": "日本語", "speaker": "話者1"}), "[話者1] 日本語")


class DummyDB:
    def rollback(self) -> None:
        pass






class RegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        rate_limit.clear()

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
                SimpleNamespace(
                    transcripts_dir=transcripts_dir,
                    history_dir=history_dir,
                    debug_chunks_dir=root / 'debug_chunks',
                ),
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
                SimpleNamespace(history_dir=history_dir, debug_chunks_dir=root / 'debug_chunks'),
            ):
                payload = history_service.build_history_detail_payload(history)

            screenshot_url = payload['segments'][0]['screenshotUrl']
            self.assertIsNotNone(screenshot_url)
            self.assertTrue(screenshot_url.endswith('/000001.webp'))


    def test_history_saved_at_is_serialized_in_app_timezone_when_db_value_is_naive_utc(self) -> None:
        naive_saved_at = datetime(2026, 3, 22, 12, 34, 56)
        history = TranscriptHistory(
            id='hist-timezone',
            user_id=1,
            runtime_session_id='sess-timezone',
            title='sample',
            language='ja',
            audio_source='mic',
            segment_count=0,
            plain_text='hello',
            summary_text=None,
            proofread_text=None,
            has_diarization=False,
            artifact_dir='1/hist-timezone',
            txt_path='1/hist-timezone/transcript.txt',
            jsonl_path='1/hist-timezone/transcript.jsonl',
            zip_path='1/hist-timezone/transcript.zip',
            created_at=naive_saved_at,
            updated_at=naive_saved_at,
            saved_at=naive_saved_at,
        )
        history.segments = []

        expected = '2026-03-22T21:34:56+09:00'

        detail_payload = history_service.build_history_detail_payload(history)
        list_payload = history_service.build_history_list_item(history)
        create_payload = history_service.build_history_create_payload(history)

        self.assertEqual(detail_payload['savedAt'], expected)
        self.assertEqual(list_payload['savedAt'], expected)
        self.assertEqual(create_payload['history']['savedAt'], expected)


    def test_runtime_screenshot_copy_ignores_query_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcripts_dir = root / 'transcripts'
            screenshots_dir = transcripts_dir / '_screenshots' / 'sess-1'
            target_dir = root / 'target'
            screenshots_dir.mkdir(parents=True)
            target_dir.mkdir()
            (screenshots_dir / '000001.webp').write_bytes(b'webp')

            with patch.object(history_service, 'settings', SimpleNamespace(transcripts_dir=transcripts_dir)):
                filename = history_service.copy_runtime_screenshot(
                    'sess-1',
                    {'screenshotPath': '/api/transcripts/sess-1/screenshots/000001.webp?token=abc'},
                    target_dir,
                )

            self.assertEqual(filename, '000001.webp')
            self.assertTrue((target_dir / '000001.webp').exists())


    def test_create_history_rolls_back_and_cleans_temp_artifacts_on_commit_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcripts_dir = root / 'transcripts'
            history_dir = root / 'history'
            transcripts_dir.mkdir()
            history_dir.mkdir()

            session_id = 'sess-commit'
            (transcripts_dir / f'{session_id}.txt').write_text('hello\n', encoding='utf-8')
            (transcripts_dir / f'{session_id}.jsonl').write_text(
                '{"type":"final","seq":0,"text":"hello","tsStart":0,"tsEnd":100}\n',
                encoding='utf-8',
            )
            (transcripts_dir / f'{session_id}.meta.json').write_text(
                '{"finalized": true, "ownerUserId": 1}',
                encoding='utf-8',
            )

            user = User(id=1, email='user@example.com', password_hash='hash', is_active=True, is_admin=False)

            class CommitFailDB:
                def scalar(self, *_args, **_kwargs):
                    return None

                def add(self, *_args, **_kwargs):
                    return None

                def flush(self):
                    return None

                def commit(self):
                    raise RuntimeError('commit_failed')

                def rollback(self):
                    return None

            payload = SimpleNamespace(
                runtimeSessionId=session_id,
                runtimeSessionToken='token',
                title=None,
                summaryText=None,
                proofreadText=None,
            )

            with patch.object(
                history_service,
                'settings',
                SimpleNamespace(
                    transcripts_dir=transcripts_dir,
                    history_dir=history_dir,
                    debug_chunks_dir=root / 'debug_chunks',
                ),
            ):
                with self.assertRaises(RuntimeError):
                    history_service.create_history_from_payload(CommitFailDB(), user=user, payload=payload)

            user_history_dir = history_dir / '1'
            self.assertTrue(user_history_dir.exists())
            self.assertFalse(any(path.name.startswith('.hist_') for path in user_history_dir.glob('**/*')))


    def test_shared_glossary_writes_append_only_revision_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                glossary_service,
                'settings',
                SimpleNamespace(app_data_dir=Path(tmpdir)),
            ):
                first = glossary_service.save_shared_glossary(text='alpha', updated_by='admin@example.com')
                second = glossary_service.save_shared_glossary(text='beta', updated_by='admin@example.com')
                history_lines = glossary_service.glossary_history_path().read_text(encoding='utf-8').splitlines()

        self.assertNotEqual(first['revisionId'], second['revisionId'])
        self.assertEqual([json.loads(line)['text'] for line in history_lines], ['alpha', 'beta'])


    def test_load_app_config_reads_history_retention_days(self) -> None:
        with patch.dict(
            os.environ,
            {
                'APP_SESSION_SECRET': 'test-session-secret-abcdefghijklmnopqrstuvwxyz12',
                'HISTORY_RETENTION_DAYS': '14',
            },
            clear=False,
        ):
            config = load_app_config()
        self.assertEqual(config.history_retention_days, 14)


    def test_history_retention_default_is_unlimited(self) -> None:
        with patch.dict(
            os.environ,
            {
                'APP_SESSION_SECRET': 'test-session-secret-abcdefghijklmnopqrstuvwxyz12',
                'HISTORY_RETENTION_DAYS': '',
            },
            clear=False,
        ):
            config = load_app_config()
        self.assertEqual(config.history_retention_days, 0)


    def test_history_save_uses_sharded_storage_and_zip_is_generated_on_demand(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcripts_dir = root / 'transcripts'
            history_dir = root / 'history'
            debug_chunks_dir = root / 'debug_chunks'
            runtime_dir = transcripts_dir / '2026' / '03' / '22'
            runtime_dir.mkdir(parents=True)
            history_dir.mkdir()
            debug_chunks_dir.mkdir()

            session_id = 'sess-sharded'
            (runtime_dir / f'{session_id}.txt').write_text('hello\n', encoding='utf-8')
            (runtime_dir / f'{session_id}.jsonl').write_text(
                '{"type":"final","seq":0,"text":"hello","tsStart":0,"tsEnd":100}\n',
                encoding='utf-8',
            )
            (runtime_dir / f'{session_id}.meta.json').write_text(
                '{"finalized": true, "ownerUserId": 1}',
                encoding='utf-8',
            )

            user = User(id=1, email='user@example.com', password_hash='hash', is_active=True, is_admin=False)

            class DummyDB:
                def scalar(self, *_args, **_kwargs):
                    return None

                def add(self, *_args, **_kwargs):
                    return None

                def flush(self):
                    return None

                def commit(self):
                    return None

                def rollback(self):
                    return None

            payload = SimpleNamespace(
                runtimeSessionId=session_id,
                runtimeSessionToken='token',
                title='sample',
                summaryText=None,
                proofreadText=None,
            )

            with patch.object(
                history_service,
                'settings',
                SimpleNamespace(
                    transcripts_dir=transcripts_dir,
                    history_dir=history_dir,
                    debug_chunks_dir=debug_chunks_dir,
                ),
            ):
                history = history_service.create_history_from_payload(DummyDB(), user=user, payload=payload)
                with patch.object(history_service, 'get_history_for_user', return_value=history):
                    zip_response = history_service.get_history_download_response(
                        DummyDB(),
                        user=user,
                        history_id=history.id,
                        kind='zip',
                    )

            self.assertRegex(history.artifact_dir or '', r'^1/\d{4}/\d{2}/hist_')
            self.assertIsNone(history.zip_path)
            self.assertTrue((history_dir / (history.txt_path or '')).exists())
            self.assertIsNotNone(zip_response)


    def test_create_history_does_not_commit_before_artifacts_are_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            transcripts_dir = root / 'transcripts'
            history_dir = root / 'history'
            debug_chunks_dir = root / 'debug_chunks'
            runtime_dir = transcripts_dir / '2026' / '03' / '22'
            runtime_dir.mkdir(parents=True)
            history_dir.mkdir()
            debug_chunks_dir.mkdir()

            session_id = 'sess-finalize-order'
            (runtime_dir / f'{session_id}.txt').write_text('hello\n', encoding='utf-8')
            (runtime_dir / f'{session_id}.jsonl').write_text(
                '{"type":"final","seq":0,"text":"hello","tsStart":0,"tsEnd":100}\n',
                encoding='utf-8',
            )
            (runtime_dir / f'{session_id}.meta.json').write_text(
                '{"finalized": true, "ownerUserId": 1}',
                encoding='utf-8',
            )

            user = User(id=1, email='user@example.com', password_hash='hash', is_active=True, is_admin=False)

            class DummyDB:
                def __init__(self) -> None:
                    self.commit_calls = 0
                    self.rollback_calls = 0

                def scalar(self, *_args, **_kwargs):
                    return None

                def add(self, *_args, **_kwargs):
                    return None

                def flush(self):
                    return None

                def commit(self):
                    self.commit_calls += 1
                    return None

                def rollback(self):
                    self.rollback_calls += 1
                    return None

            db = DummyDB()
            payload = SimpleNamespace(
                runtimeSessionId=session_id,
                runtimeSessionToken='token',
                title='sample',
                summaryText=None,
                proofreadText=None,
            )

            with patch.object(
                history_service,
                'settings',
                SimpleNamespace(
                    transcripts_dir=transcripts_dir,
                    history_dir=history_dir,
                    debug_chunks_dir=debug_chunks_dir,
                ),
            ):
                with patch.object(
                    history_service.artifact_storage,
                    'finalize_history_artifacts',
                    side_effect=RuntimeError('finalize_failed'),
                ):
                    with self.assertRaises(RuntimeError):
                        history_service.create_history_from_payload(db, user=user, payload=payload)

            self.assertEqual(db.commit_calls, 0)
            self.assertEqual(db.rollback_calls, 1)
            self.assertFalse(any(path.name.startswith('hist_') for path in history_dir.glob('**/*')))
            self.assertFalse(any(path.name.startswith('.hist_') for path in history_dir.glob('**/*')))


    def test_runtime_zip_includes_sharded_screenshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            transcripts_dir = Path(tmpdir) / 'transcripts'
            runtime_dir = transcripts_dir / '2026' / '03' / '22'
            screenshots_dir = transcripts_dir / '_screenshots' / '2026' / '03' / '22' / 'sess-zip'
            runtime_dir.mkdir(parents=True)
            screenshots_dir.mkdir(parents=True)

            (runtime_dir / 'sess-zip.txt').write_text('hello\n', encoding='utf-8')
            (runtime_dir / 'sess-zip.jsonl').write_text(
                '{"type":"final","seq":0,"text":"hello","tsStart":0,"tsEnd":100}\n',
                encoding='utf-8',
            )
            (runtime_dir / 'sess-zip.meta.json').write_text(
                '{"finalized": true, "ownerUserId": 1}',
                encoding='utf-8',
            )
            (screenshots_dir / '000001.webp').write_bytes(b'webp')

            with patch.object(
                runtime_artifact_service,
                'settings',
                SimpleNamespace(
                    transcripts_dir=transcripts_dir,
                    app_data_dir=Path(tmpdir),
                    artifact_worker_concurrency=2,
                    blocking_worker_queue_timeout_seconds=30,
                ),
            ):
                response = runtime_artifact_service.get_zip(
                    'sess-zip',
                    user_id=1,
                    guest_grant_id=None,
                )

            with zipfile.ZipFile(response.path) as archive:
                names = sorted(archive.namelist())
            asyncio.run(response.background())
            self.assertFalse(Path(response.path).exists())

            self.assertIn('sess-zip.txt', names)
            self.assertIn('sess-zip.jsonl', names)
            self.assertIn('sess-zip/screenshots/000001.webp', names)


    def test_history_audio_response_supports_mp3(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            audio_dir = history_dir / '1' / 'hist-1' / 'audio'
            audio_dir.mkdir(parents=True)
            (audio_dir / 'sample.mp3').write_bytes(b'mp3')
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

            with patch.object(history_service, 'settings', SimpleNamespace(history_dir=history_dir)):
                with patch.object(history_service, 'get_history_for_user', return_value=history):
                    response = history_service.get_history_audio_response(
                        SimpleNamespace(),
                        user=User(id=1, email='user@example.com', password_hash='hash', is_active=True, is_admin=False),
                        history_id='hist-1',
                        filename='sample.mp3',
                    )

            assert response is not None
            self.assertEqual(response.media_type, 'audio/mpeg')


    def test_cleanup_expired_histories_deletes_old_history_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_dir = root / 'history'
            history_dir.mkdir()
            old_dir = history_dir / '1' / '2026' / '02' / 'hist-old'
            old_dir.mkdir(parents=True)
            (old_dir / 'transcript.txt').write_text('old\n', encoding='utf-8')

            old_saved_at = datetime.now(timezone.utc) - timedelta(days=10)
            history = TranscriptHistory(
                id='hist-old',
                user_id=1,
                runtime_session_id='sess-old',
                title='old',
                language='ja',
                audio_source='mic',
                segment_count=1,
                plain_text='old',
                summary_text=None,
                proofread_text=None,
                has_diarization=False,
                artifact_dir='1/2026/02/hist-old',
                txt_path='1/2026/02/hist-old/transcript.txt',
                jsonl_path='1/2026/02/hist-old/transcript.jsonl',
                zip_path=None,
                created_at=old_saved_at,
                updated_at=old_saved_at,
                saved_at=old_saved_at,
            )

            class CleanupDB:
                def __init__(self):
                    self.deleted: list[TranscriptHistory] = []

                def flush(self):
                    return None

            db = CleanupDB()

            with patch.object(history_service, 'settings', SimpleNamespace(history_dir=history_dir, history_retention_days=7)):
                with patch.object(history_service.history_repository, 'list_histories_saved_before', return_value=[history]):
                    with patch.object(history_service.history_repository, 'delete_history', side_effect=lambda _db, item: db.deleted.append(item)):
                        deleted_count = history_service.cleanup_expired_histories(db)

            self.assertEqual(deleted_count, 1)
            self.assertEqual(db.deleted, [history])
            self.assertFalse(old_dir.exists())


    def test_jsonl_strict_reader_raises_on_invalid_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'broken.jsonl'
            path.write_text('{"type":"final"}\n{broken}\n', encoding='utf-8')

            with self.assertRaises(ValueError):
                read_jsonl_records(path, strict=True)


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

