"""Actual cookie authorization and FFmpeg with synthetic media and an isolated DB."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'media-security-tests-only-long-session-secret')

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server import auth
from server.api.routes import media, summary
from server.core.application import create_app
from server.core.config import settings
from server.db import Base, get_db
from server.models import User, UserSession
from server.services import artifact_storage, history_service, media_import, meeting_source, runtime_artifact_service
from server.transcript_store import resolve_transcript_path
from server.transcription.local_agreement import pcm_wav
from tests.whisper_audio_fixtures import pcm


@unittest.skipUnless(os.environ.get('FFMPEG_BIN') or shutil.which('ffmpeg'), 'FFmpeg required')
class MediaSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='whistx-security-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        original = settings
        root = self.root
        class Config:
            transcripts_dir = root / 'transcripts'
            debug_chunks_dir = root / 'audio'
            history_dir = root / 'history'
            ffmpeg_bin = os.environ.get('FFMPEG_BIN') or shutil.which('ffmpeg')
            def __getattr__(self, name):
                return getattr(original, name)
        self.config = Config()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (media_import, meeting_source, runtime_artifact_service, history_service, artifact_storage):
            self.stack.enter_context(patch.object(module, 'settings', self.config))
        self.stack.enter_context(patch.object(media, '_allow', return_value=True))
        self.stack.enter_context(patch.object(summary, '_allow_costly_request', return_value=True))
        self.engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
        self.addCleanup(self.engine.dispose)
        Base.metadata.create_all(self.engine)
        self.tokens = {}
        with Session(self.engine) as db:
            for user_id in (71, 72):
                user = User(id=user_id, email=f'synthetic-{user_id}@example.invalid', password_hash='synthetic', is_active=True)
                db.add(user)
                db.flush()
                self.tokens[user_id] = auth.create_user_session(db, user=user, user_agent='synthetic', ip_address='127.0.0.1')
            db.commit()
        def database():
            with Session(self.engine) as db:
                yield db
        self.app = create_app()
        self.app.dependency_overrides[get_db] = database
        self.app.state.runtime_resources = SimpleNamespace(observer=None, transcriber_factory=lambda: SimpleNamespace(
            transcribe_chunk=lambda *a, **kw: SimpleNamespace(text='合成データの文字起こし'), close=lambda: None))
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)
        self.audio = pcm_wav(pcm(.3, amplitude=500))
        self.login(71)

    def login(self, user_id):
        self.client.cookies.clear()
        if user_id:
            self.client.cookies.set('whistx_session', self.tokens[user_id])

    def upload(self, filename='synthetic.wav', data=None, **params):
        return self.client.post('/api/media/transcribe', params={'filename': filename, **params},
                                content=self.audio if data is None else data)

    def done(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        rows = [json.loads(row[5:]) for row in response.text.splitlines() if row.startswith('data:')]
        self.assertEqual(rows[-1]['type'], 'done', response.text)
        return rows[-1]

    def test_cookie_login_invalid_expired_revoked_sessions_and_origin(self):
        with patch.object(media.tempfile, 'NamedTemporaryFile', side_effect=AssertionError('unauthorized upload created a file')):
            self.login(None)
            self.assertEqual(self.upload().status_code, 401)
            self.client.cookies.set('whistx_session', 'invalid-synthetic-session')
            self.assertEqual(self.upload().status_code, 401)
            self.login(71)
            with Session(self.engine) as db:
                session = db.get(UserSession, auth.hash_session_id(self.tokens[71]))
                session.expires_at = auth.utcnow() - timedelta(seconds=1)
                db.commit()
            self.assertEqual(self.upload().status_code, 401)
            self.login(72)
            with Session(self.engine) as db:
                auth.delete_user_session(db, self.tokens[72])
                db.commit()
            self.assertEqual(self.upload().status_code, 401)
        self.assertEqual(self.client.post('/api/media/transcribe?filename=a.wav', content=self.audio,
            headers={'Origin': 'https://unrelated.invalid'}).status_code, 403)
        self.assertEqual(list(self.root.rglob('*.jsonl')), [])

    def test_other_users_cannot_read_save_refine_or_reuse_the_meeting_identity(self):
        result = self.done(self.upload())
        identifier = result['sessionId']
        audio_url = result['records'][0]['rawAudioPath']
        self.assertEqual(self.client.get(audio_url).status_code, 200)
        self.login(72)
        for suffix in ('txt', 'jsonl', 'zip'):
            self.assertEqual(self.client.get(f'/api/transcript/{identifier}.{suffix}').status_code, 404)
        self.assertEqual(self.client.get(audio_url).status_code, 404)
        for endpoint in ('/api/meeting/insights', '/api/meeting/refine', '/api/history'):
            self.assertEqual(self.client.post(endpoint, json={'runtimeSessionId': identifier}).status_code, 404)
        attacker = self.done(self.upload(ownerUserId=71, runtimeSessionId=identifier, user_id=71))
        self.assertNotEqual(attacker['sessionId'], identifier)
        path = resolve_transcript_path(self.config.transcripts_dir, attacker['sessionId'], 'jsonl')
        self.assertEqual(json.loads(path.with_suffix('.meta.json').read_text())['ownerUserId'], 72)
        self.login(71)
        self.assertEqual(self.client.get(attacker['records'][0]['rawAudioPath']).status_code, 404)
        response = self.client.post('/api/history', json={'runtimeSessionId': identifier, 'title': '合成会議'})
        self.assertEqual(response.status_code, 200, response.text)
        history_id = response.json()['history']['id']
        detail = self.client.get('/api/history/' + history_id).json()
        history_audio = detail['segments'][0]['rawAudioUrl']
        self.assertEqual(self.client.get(history_audio).status_code, 200)
        self.login(72)
        self.assertEqual(self.client.get('/api/history/' + history_id).status_code, 404)
        self.assertEqual(self.client.get(history_audio).status_code, 404)
        self.assertEqual(self.client.get('/api/history/' + history_id + '/download.txt').status_code, 404)
        self.assertEqual(self.client.post('/api/meeting/insights', json={'historyId': history_id}).status_code, 404)
        self.login(None)
        self.assertEqual(self.client.get(audio_url).status_code, 404)

    def test_mismatched_extension_corrupt_manifest_and_truncated_input_leave_no_artifacts(self):
        cases = [('wrong.mp4', self.audio), ('wrong.wav', b'not-media'),
                 ('remote.mp4', b'#EXTM3U\nhttps://unrelated.invalid/audio.wav\n'),
                 ('truncated.wav', self.audio[:-200])]
        for filename, data in cases:
            with self.subTest(filename=filename):
                response = self.upload(filename, data)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn('media_decode_failed', response.text)
                self.assertNotIn('"type": "done"', response.text)
        self.assertEqual(list(self.root.rglob('*.jsonl')), [])
        self.assertEqual(list(self.root.rglob('file-*.wav')), [])

    def test_untrusted_names_never_become_paths_or_shell_arguments(self):
        calls = []
        original = subprocess.Popen
        def start(args, **kwargs):
            calls.append((args, kwargs))
            return original(args, **kwargs)
        names = ['../../outside.wav', 'http://unrelated.invalid/audio.wav', '$(touch injected).wav', '-i hostile.wav']
        with patch.object(media_import.subprocess, 'Popen', start):
            for filename in names:
                result = self.done(self.upload(filename))
                self.assertRegex(result['sessionId'], r'^file-[a-f0-9]{32}$')
        for args, options in calls:
            self.assertFalse(options.get('shell', False))
            self.assertEqual(args[args.index('-protocol_whitelist') + 1], 'file,pipe')
            self.assertEqual(args[args.index('-f') + 1], 'wav')
            self.assertNotIn('http://unrelated.invalid/audio.wav', args)
            self.assertIn('-xerror', args)
            self.assertEqual(args[args.index('-threads') + 1], '1')
            self.assertTrue(Path(args[args.index('-i') + 1]).name.startswith('whistx-upload-'))
        self.assertFalse((self.root / 'outside.wav').exists())
        self.assertFalse((Path.cwd() / 'injected').exists())


if __name__ == '__main__':
    unittest.main()
