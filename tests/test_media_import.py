"""Synthetic media only: decoding, ownership, cancellation and runtime reuse."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import wave
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'media-test-only-long-session-secret-value')

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from server.api.routes import media
from server.core.config import settings
from server.core.security import runtime_access_allowed
from server.db import Base
from server.models import User
from server.schemas import HistorySaveRequest
from server.services import artifact_storage, history_service
from server.services import media_import, meeting_refinement, meeting_source
from server.services.media_import import decode_media, import_events
from server.services.meeting_source import MeetingError, load_meeting
from server.transcript_store import resolve_transcript_path
from server.transcription.local_agreement import pcm_wav
from tests.whisper_audio_fixtures import pcm, transcriber


class MediaImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cancelled = threading.Event()
        base_settings = settings
        root = self.root

        class Config:
            transcripts_dir = root / 'transcripts'
            debug_chunks_dir = root / 'audio'
            history_dir = root / 'history'
            def __getattr__(self, key):
                return getattr(base_settings, key)

        self.config = Config()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (media_import, meeting_source, meeting_refinement):
            self.stack.enter_context(patch.object(module, 'settings', self.config))
        self.calls = []
        self.closed = False

    def run_import(self, data, *, model=None, allow_request=lambda: True, decoder=None, language='ja'):
        source = self.root / 'upload.media'
        source.write_bytes(pcm_wav(data))
        def copy(source, destination, **kwargs):
            shutil.copyfile(source, destination)
        def recognize(audio, **kwargs):
            self.calls.append((audio, kwargs))
            return SimpleNamespace(text='はい。確認しました。')
        def close():
            self.closed = True
        self.events = []
        for event in import_events(source, filename='合成テスト.wav', language=language, prompt='合成テスト',
                user_id=71, cancelled=self.cancelled,
                transcriber_factory=lambda: model or SimpleNamespace(transcribe_chunk=recognize, close=close),
                allow_request=allow_request, decoder=decoder or copy):
            self.events.append(event)
        return self.events[-1]

    def assert_no_artifacts(self):
        self.assertFalse((self.root / 'upload.media').exists())
        self.assertEqual(list(self.root.rglob('*.jsonl')), [])
        self.assertEqual(list(self.root.rglob('*.meta.json')), [])
        self.assertEqual(list((self.root / 'audio').rglob('*.wav')), [])

    def test_silence_recovery_and_short_quiet_tail_keep_media_clock_and_ownership(self):
        result = self.run_import(pcm(30, amplitude=0) + pcm(.25, amplitude=180))
        self.assertEqual(result['type'], 'done')
        self.assertEqual(len(self.calls), 1)
        row = result['records'][0]
        self.assertEqual((row['tsStart'], row['tsEnd']), (30_000, 30_250))
        self.assertEqual((row['startSample'], row['endSample']), (480_000, 484_000))
        self.assertTrue(self.closed)
        self.assertFalse((self.root / 'upload.media').exists())
        self.assertTrue(runtime_access_allowed(transcripts_dir=self.config.transcripts_dir,
            session_id=result['sessionId'], user_id=71, guest_grant_id=None))
        self.assertFalse(runtime_access_allowed(transcripts_dir=self.config.transcripts_dir,
            session_id=result['sessionId'], user_id=72, guest_grant_id=None))
        snapshot = load_meeting(user_id=71, runtime_session_id=result['sessionId'])
        self.assertTrue(snapshot.finalized)
        self.assertEqual(snapshot.through_ms, 30_250)
        self.assertIn('/audio/file-', snapshot.segments[0]['audioUrl'])
        with self.assertRaises(MeetingError):
            load_meeting(user_id=72, runtime_session_id=result['sessionId'])
        path = resolve_transcript_path(self.config.transcripts_dir, result['sessionId'], 'jsonl')
        metadata = json.loads(path.with_suffix('.meta.json').read_text())
        self.assertEqual(metadata['audioSource'], 'file')
        self.assertEqual(metadata['ownerUserId'], 71)

    def test_completed_import_can_be_manually_refined(self):
        result = self.run_import(pcm(.3))
        snapshot = load_meeting(user_id=71, runtime_session_id=result['sessionId'])
        model = SimpleNamespace(transcribe_chunk=lambda *a, **kw: SimpleNamespace(text='再認識しました。'), client=SimpleNamespace(close=lambda: None))
        events = list(meeting_refinement.refine_events(snapshot, cancelled=self.cancelled,
            transcriber_factory=lambda **kw: model))
        self.assertEqual(events[-1]['type'], 'done')
        self.assertEqual(load_meeting(user_id=71, runtime_session_id=result['sessionId']).segments[0]['text'], '再認識しました。')

    def test_import_history_save_copies_audio_and_rejects_wrong_owner(self):
        result = self.run_import(pcm(.3))
        engine = create_engine('sqlite://')
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        with Session(engine) as db, patch.object(history_service, 'settings', self.config), patch.object(artifact_storage, 'settings', self.config):
            owner = User(id=71, email='synthetic-owner@example.invalid', password_hash='synthetic')
            other = User(id=72, email='synthetic-other@example.invalid', password_hash='synthetic')
            db.add_all([owner, other])
            db.commit()
            payload = HistorySaveRequest(runtimeSessionId=result['sessionId'], title='合成ファイル')
            with self.assertRaises(history_service.HistoryError) as error:
                history_service.create_history_from_payload(db, user=other, payload=payload)
            self.assertEqual(error.exception.code, 'runtime_session_not_found')
            history = history_service.create_history_from_payload(db, user=owner, payload=payload)
            self.assertEqual(history.audio_source, 'file')
            detail = history_service.build_history_detail_payload(history)
            self.assertEqual(detail['audioSource'], 'file')
            self.assertIn('/api/history/', detail['segments'][0]['rawAudioUrl'])
            self.assertTrue(history_service.resolve_history_audio_path(history, 'file-000000.wav').is_file())
            download = history_service.get_history_download_response(db, user=owner, history_id=history.id, kind='txt')
            self.assertIn('はい。確認しました。', Path(download.path).read_text())
            self.assertIsNone(history_service.get_history_download_response(db, user=other, history_id=history.id, kind='txt'))

    def test_silence_and_stationary_noise_do_not_call_model_or_leave_partial_results(self):
        for data in (pcm(1, amplitude=0), pcm(1, amplitude=180, noise=True)):
            with self.subTest(kind=data[:4]), self.assertRaises(MeetingError) as error:
                self.run_import(data)
            self.assertEqual(error.exception.code, 'media_no_speech')
            self.assertEqual(self.calls, [])
            self.assert_no_artifacts()

    def test_whisper_and_qwen_adapters_receive_synthetic_wave_and_language(self):
        from server.qwen_asr import QwenBatchTranscriber
        whisper, requests = transcriber(text='はい')
        result = self.run_import(pcm(.2), model=whisper)
        self.assertEqual(result['records'][0]['text'], 'はい')
        self.assertEqual(len(requests), 1)
        qwen_calls = []
        qwen = QwenBatchTranscriber(api_key='synthetic', base_url='https://synthetic.invalid/v1', model='synthetic')
        qwen.client.close()
        qwen.client = SimpleNamespace(post=lambda url, **options: (
            qwen_calls.append((url, options)) or SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'text': 'はい'})
        ), close=lambda: None)
        result = self.run_import(pcm(.2), model=qwen, language='en')
        self.assertEqual(result['records'][0]['text'], 'はい')
        self.assertEqual(qwen_calls[0][1]['data']['language'], 'en')

    def test_cancellation_after_provider_call_removes_partial_results(self):
        def recognize(*args, **kwargs):
            self.cancelled.set()
            return SimpleNamespace(text='キャンセル後の結果')
        self.run_import(pcm(.3), model=SimpleNamespace(transcribe_chunk=recognize, close=lambda: None))
        self.assertFalse(any(event['type'] == 'done' for event in self.events))
        self.assert_no_artifacts()

    def test_cancel_while_waiting_for_quota_does_not_start_provider(self):
        def deny():
            self.cancelled.set()
            return False
        self.run_import(pcm(.3), allow_request=deny)
        self.assertEqual(self.events[-1]['phase'], 'waiting')
        self.assertEqual(self.calls, [])
        self.assert_no_artifacts()

    def test_provider_failure_and_close_failure_clean_up(self):
        calls = []
        def recognize(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('synthetic_failure')
            return SimpleNamespace(text='前半の結果')
        def close():
            raise RuntimeError('synthetic_close_failure')
        with self.assertRaises(RuntimeError):
            self.run_import(pcm(30.2), model=SimpleNamespace(transcribe_chunk=recognize, close=close))
        self.assertEqual(len(calls), 2)
        self.assert_no_artifacts()

    def test_close_failure_cannot_publish_a_done_event(self):
        def close():
            raise RuntimeError('synthetic_close_failure')
        with self.assertRaises(RuntimeError):
            self.run_import(pcm(.2), model=SimpleNamespace(
                transcribe_chunk=lambda *a, **kw: SimpleNamespace(text='認識結果'), close=close))
        self.assertFalse(any(event['type'] == 'done' for event in self.events))
        self.assert_no_artifacts()

    def test_cancel_during_provider_close_cannot_commit_a_result(self):
        self.run_import(pcm(.2), model=SimpleNamespace(
            transcribe_chunk=lambda *a, **kw: SimpleNamespace(text='中断対象'),
            close=self.cancelled.set))
        self.assertFalse(any(event['type'] == 'done' for event in self.events))
        self.assert_no_artifacts()

    def test_cancel_during_final_metadata_write_cleans_up(self):
        original = media_import.TranscriptStore.write_metadata
        def write(store, metadata):
            original(store, metadata)
            if metadata.get('finalized'):
                self.cancelled.set()
        with patch.object(media_import.TranscriptStore, 'write_metadata', write):
            self.run_import(pcm(.2))
        self.assertFalse(any(event['type'] == 'done' for event in self.events))
        self.assert_no_artifacts()


@unittest.skipUnless(os.environ.get('FFMPEG_BIN') or shutil.which('ffmpeg'), 'FFmpeg required for actual decoder tests')
class MediaDecoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = os.environ.get('FFMPEG_BIN') or shutil.which('ffmpeg')
        self.source = self.root / 'synthetic.wav'
        self.source.write_bytes(pcm_wav(pcm(.4, amplitude=500)))

    def decode(self, source, filename=None, cancelled=None):
        destination = self.root / 'decoded.wav'
        decode_media(source, destination, filename=filename or source.name,
            cancelled=cancelled or threading.Event(), ffmpeg_bin=self.binary)
        return destination

    def test_audio_and_video_formats_extract_only_mono_pcm_audio(self):
        for extension, codec in [('wav', 'pcm_s16le'), ('mp3', 'libmp3lame'), ('m4a', 'aac'),
                                 ('mp4', 'aac'), ('mov', 'aac'), ('webm', 'libopus')]:
            with self.subTest(extension=extension):
                destination = self.root / ('synthetic.' + extension)
                if extension == 'wav':
                    shutil.copyfile(self.source, self.root / 'source.wav')
                    destination = self.root / 'source.wav'
                else:
                    command = [self.binary, '-hide_banner', '-loglevel', 'error', '-y']
                    if extension in ('mp4', 'mov', 'webm'):
                        command += ['-f', 'lavfi', '-i', 'color=c=black:s=32x32:d=0.4:r=10']
                    command += ['-i', str(self.source), '-c:a', codec]
                    if extension in ('mp4', 'mov', 'webm'):
                        command += ['-c:v', 'libvpx-vp9' if extension == 'webm' else 'libx264', '-shortest']
                    subprocess.run([*command, str(destination)], check=True, capture_output=True)
                with wave.open(str(self.decode(destination))) as audio:
                    self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
                    self.assertGreater(audio.getnframes(), 5000)

    def test_corrupt_media_and_video_without_audio_fail(self):
        broken = self.root / 'broken.mp4'
        broken.write_bytes(b'not a video')
        video = self.root / 'silent-video.mp4'
        subprocess.run([self.binary, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi',
            '-i', 'color=c=black:s=32x32:d=0.1:r=10', str(video)], check=True, capture_output=True)
        for source in (broken, video):
            with self.subTest(source=source.name), self.assertRaises(MeetingError) as error:
                self.decode(source)
            self.assertEqual(error.exception.code, 'media_decode_failed')

    def test_duration_limit_and_pre_cancel_are_enforced(self):
        self.source.write_bytes(pcm_wav(pcm(2)))
        with patch.object(media_import, 'MAX_DURATION_SECONDS', 1), self.assertRaises(MeetingError) as error:
            self.decode(self.source)
        self.assertEqual(error.exception.code, 'media_too_long')
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(MeetingError):
            self.decode(self.source, cancelled=cancelled)

    def test_missing_decoder_and_remote_manifest_are_rejected(self):
        with patch('shutil.which', return_value=None), self.assertRaises(MeetingError) as error:
            self.decode(self.source)
        self.assertEqual(error.exception.code, 'media_decoder_unavailable')
        with self.assertRaises(MeetingError) as error:
            self.decode(self.source, filename='remote.m3u8')
        self.assertEqual(error.exception.code, 'media_unsupported_format')

    def test_truncated_audio_is_not_silently_accepted(self):
        self.source.write_bytes(self.source.read_bytes()[:-200])
        with self.assertRaises(MeetingError) as error:
            self.decode(self.source)
        self.assertEqual(error.exception.code, 'media_decode_failed')


class MediaRouteTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(media.router)
        self.app.state.runtime_resources = SimpleNamespace(transcriber_factory=lambda: None, observer=None)
        self.app.dependency_overrides[media.get_current_user] = lambda: SimpleNamespace(id=71)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        allowed = patch.object(media, '_allow', return_value=True)
        allowed.start()
        self.addCleanup(allowed.stop)

    def post(self, filename='synthetic.wav', data=b'wave', **kwargs):
        return self.client.post('/api/media/transcribe', params={'filename': filename}, content=data, **kwargs)

    def test_login_size_empty_format_and_readiness_checks_precede_decoding(self):
        def unauthorized():
            raise HTTPException(401, detail='login_required')
        self.app.dependency_overrides[media.get_current_user] = unauthorized
        self.assertEqual(self.post().status_code, 401)
        self.app.dependency_overrides[media.get_current_user] = lambda: SimpleNamespace(id=71)
        self.assertEqual(self.post(filename='secret.m3u8').status_code, 415)
        self.assertEqual(self.post(data=b'').status_code, 400)
        with patch.object(media, 'MAX_UPLOAD_BYTES', 3):
            self.assertEqual(self.post().status_code, 413)
            self.assertEqual(self.post(headers={'Content-Length': '1'}).status_code, 413)
        self.app.state.runtime_resources.transcriber_factory = None
        self.assertEqual(self.post().status_code, 503)

    def test_upload_owner_and_cleanup_on_success_and_stream_error(self):
        paths = []
        def events(source, **kwargs):
            paths.append(source)
            self.assertEqual(source.read_bytes(), b'wave')
            self.assertEqual(kwargs['user_id'], 71)
            yield {'type': 'status', 'progress': 20}
            yield {'type': 'done', 'sessionId': 'synthetic'}
        with patch.object(media, 'import_events', events):
            response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "done"', response.text)
        self.assertFalse(paths[0].exists())
        def failure(source, **kwargs):
            paths.append(source)
            raise MeetingError('media_decode_failed', 422)
        with patch.object(media, 'import_events', failure):
            response = self.post()
        self.assertIn('media_decode_failed', response.text)
        self.assertFalse(paths[-1].exists())

    def test_request_rate_limit_and_invalid_query_are_rejected(self):
        with patch.object(media, '_allow', return_value=False):
            self.assertEqual(self.post().status_code, 429)
        self.assertEqual(self.post(filename='a' * 241).status_code, 422)
        self.assertEqual(self.client.post('/api/media/transcribe?filename=a.wav&language=unknown', content=b'w').status_code, 422)


if __name__ == '__main__':
    unittest.main()
