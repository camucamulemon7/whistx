from __future__ import annotations

import asyncio
import base64
import io
import json
import tempfile
import threading
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from server.services.meeting_intelligence import NO_ANSWER, answer_events, generate_recap, read_insights, validate_chapters
from server.services.meeting_source import MeetingError, MeetingSnapshot, load_meeting
from server.transcription.live import LiveMeeting
from server.transcription.local_agreement import agreed_prefix, pcm_wav, speech_bounds


class MeetingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / 'meeting.jsonl'
        self.rows = [dict(id=str(i), seq=i, text=text, startMs=i*1000, endMs=(i+1)*1000, speaker='', audioUrl=None, imageId=None)
                     for i, text in enumerate(['公開日は未定です。', '田中さんが金曜までに確認します。'])]
        self.snapshot = MeetingSnapshot('runtime:test', path, path.with_suffix('.meeting.json'), self.rows,
                                        [dict(id='screen.png', timeMs=0, url='/api/image')], 'rev1', 2000, False)
        self.chapter = dict(title='公開準備', start_id='0', end_id='1', summary=[dict(text='公開日は未定', source_ids=['0'])],
                            actions=[dict(text='確認する', source_ids=['1'], owner='田中', due='金曜')])

    def test_recap_persists_citations_actual_image_and_staleness(self):
        model = SimpleNamespace(model='fixture', complete_meeting=lambda *a, **kw: json.dumps(dict(chapters=[self.chapter]), ensure_ascii=False))
        recap = generate_recap(self.snapshot, model)
        self.assertEqual(recap['chapters'][0]['imageIds'], ['screen.png'])
        self.assertEqual(recap['chapters'][0]['actions'][0]['sourceIds'], ['1'])
        self.assertTrue(recap['provisional'])
        self.assertFalse(read_insights(self.snapshot)['recap']['stale'])

    def test_chapters_reject_missing_coverage_and_invented_citations(self):
        for changed in [dict(start_id='1'), dict(end_id='0'), dict(summary=[dict(text='捏造', source_ids=['other'])])]:
            with self.subTest(changed=changed), self.assertRaises(MeetingError):
                validate_chapters(dict(chapters=[{**self.chapter, **changed}]), self.rows)

    def test_answer_invalid_citation_replaced_and_valid_answer_saved(self):
        for answer, expected in [('公開は明日です。[S99]', NO_ANSWER), ('公開日は未定です。[S1]', '公開日は未定です。[S1]')]:
            model = SimpleNamespace(model='fixture', stream_meeting=lambda messages: iter([answer]))
            events = list(answer_events(self.snapshot, model, question='公開日は？', cancelled=threading.Event()))
            self.assertEqual(events[-1]['answer'], expected)
        self.assertEqual(len(read_insights(self.snapshot)['turns']), 2)
        self.assertEqual(read_insights(self.snapshot)['turns'][-1]['citations'][0]['id'], '0')

    def test_cancelled_answer_is_not_saved(self):
        stop = threading.Event()
        def stream(messages):
            yield '公開日は'
            stop.set()
            yield '未定です。[S1]'
        events = list(answer_events(self.snapshot, SimpleNamespace(model='fixture', stream_meeting=stream), question='公開日は？', cancelled=stop))
        self.assertNotIn('done', [e['type'] for e in events])
        self.assertFalse(self.snapshot.insight_path.exists())

    def test_recording_without_speech_returns_retryable_error(self):
        snapshot = replace(self.snapshot, segments=[], through_ms=0)
        with self.assertRaises(MeetingError) as caught:
            list(answer_events(snapshot, SimpleNamespace(model='fixture'), question='要点は？', cancelled=threading.Event()))
        self.assertEqual(caught.exception.code, 'empty_transcript')
        self.assertFalse(snapshot.insight_path.exists())

    def test_runtime_ownership_checked_before_read(self):
        with patch('server.services.meeting_source.runtime_access_allowed', return_value=False):
            with self.assertRaises(MeetingError) as caught:
                load_meeting(user_id=42, runtime_session_id='someone-elses-session')
            self.assertEqual(caught.exception.status_code, 404)

    def test_wav_roundtrip_and_silence_gate(self):
        pcm = b'\0\0' * 16000
        self.assertIsNone(speech_bounds(pcm))
        with wave.open(io.BytesIO(pcm_wav(pcm)), 'rb') as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.readframes(16000), pcm)
        self.assertIsNotNone(speech_bounds(b'\xff\x0f' * 16000))

    def test_agreement_japanese_and_incomplete_latin_word(self):
        self.assertEqual(agreed_prefix('本日は予算を', '本日は予算について'), '本日は')
        self.assertEqual(agreed_prefix('hello world', 'hello worlds'), 'hello')
        self.assertEqual(agreed_prefix('公開は未定です。', '公開は未定です。'), '公開は未定です。')


class LiveJournalTests(unittest.IsolatedAsyncioTestCase):
    async def test_ack_is_durable_duplicates_idempotent_and_gaps_request_resend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meeting = LiveMeeting.__new__(LiveMeeting)
            meeting.data = dict(tracks={'mic': dict(seq=-1, received=0, windowStart=0)}, audioBytes=0)
            meeting.store = SimpleNamespace(chunks_dir=root)
            meeting.state_path = root / 'checkpoint.json'
            meeting.state_lock = asyncio.Lock()
            meeting.wake = asyncio.Event()
            meeting.is_guest = False
            events = []
            async def send(event):
                if event['type'] == 'capture_ack':
                    self.assertEqual(json.loads(meeting.state_path.read_text())['tracks']['mic']['seq'], 0)
                    self.assertEqual((root / 'mic.pcm').read_bytes(), b'\x01\0' * 16000)
                events.append(event)
            meeting.send = send
            payload = dict(track='mic', seq=0, sampleStart=0, pcm=base64.b64encode(b'\x01\0' * 16000).decode())
            await meeting.accept_audio(payload)
            await meeting.accept_audio(payload)
            self.assertEqual(meeting.data['audioBytes'], 32000)
            await meeting.accept_audio({**payload, 'seq': 2, 'sampleStart': 32000})
            self.assertEqual(events[-1]['type'], 'resend')
            self.assertEqual(events[-1]['samples'], 16000)


class RefinementTests(unittest.TestCase):
    def test_replay_is_atomic_on_failure_and_preserves_original_on_success(self):
        from server.services.meeting_refinement import refine_events
        from server.services.meeting_source import write_json_atomic
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'transcript.jsonl'
            rows = [dict(type='final', seq=i, segmentId=str(i), text='元の文', tsStart=i*1000, tsEnd=(i+1)*1000,
                         language='ja', rawAudioPath=f'/api/transcripts/test/audio/raw-{i}.wav') for i in range(2)]
            original = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows)
            path.write_text(original)
            write_json_atomic(path.with_suffix('.meta.json'), dict(finalized=True))
            audio = root/'audio.wav'
            audio.write_bytes(pcm_wav(b'\x01\0'*16000))
            snapshot = MeetingSnapshot('runtime:test', path, path.with_suffix('.meeting.json'), [], [], 'r1',2000,True)
            class Transcriber:
                calls = 0
                client = SimpleNamespace(close=lambda: None)
                def __init__(self, **kwargs):
                    pass
                def transcribe_chunk(self, *args, **kwargs):
                    self.calls += 1
                    if self.calls == 2:
                        raise RuntimeError('provider failed')
                    return SimpleNamespace(text='音声から再認識した文')
            with patch('server.services.meeting_refinement.resolve_debug_audio_path', return_value=audio):
                with self.assertRaises(RuntimeError):
                    list(refine_events(snapshot, cancelled=threading.Event(), transcriber_factory=Transcriber))
                self.assertEqual(path.read_text(), original)
                self.assertFalse(path.with_suffix('.original.jsonl').exists())
                Transcriber.transcribe_chunk = lambda *a, **kw: SimpleNamespace(text='音声から再認識した文')
                events = list(refine_events(snapshot, cancelled=threading.Event(), transcriber_factory=Transcriber))
                self.assertEqual(events[-1]['type'], 'done')
                self.assertEqual(path.with_suffix('.original.jsonl').read_text(), original)
                updated = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(updated[0]['originalText'], '元の文')
                self.assertEqual(updated[0]['text'], '音声から再認識した文')


class LiveResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_recovers_agreed_text_audio_position_and_finalizes_once(self):
        from server.core.config import settings
        from server.services.meeting_source import read_json
        class Client:
            def with_options(self, **kwargs):
                return self
            def close(self):
                pass
        class Transcriber:
            client = Client()
            def __init__(self, **kwargs):
                pass
            def transcribe_chunk(self, *args, **kwargs):
                return SimpleNamespace(text='公開日は未定です。', start_ms=None, end_ms=None)
        class Socket:
            state = SimpleNamespace(authenticated_user_id=123, is_guest=False, rate_limit_subject='fixture')
            cookies = {}
            def __init__(self):
                self.events = []
            async def send_json(self, event):
                self.events.append(event)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class Config:
                transcripts_dir = root/'transcripts'
                debug_chunks_dir = root/'audio'
                def __getattr__(self, key):
                    return getattr(settings, key)
            with patch('server.transcription.live.settings', Config()), patch('server.transcription.live.OpenAIWhisperTranscriber', Transcriber), patch('server.transcription.live.consume', return_value=True):
                ws = Socket()
                live = LiveMeeting(ws, dict(tracks=['mic'], language='ja'))
                def packet(seq):
                    return dict(track='mic', seq=seq, sampleStart=seq*32000, pcm=base64.b64encode(b'\xff\x0f'*32000).decode())
                await live.accept_audio(packet(0))
                await live._infer('mic', flush=False)
                await live.accept_audio(packet(1))
                await live._infer('mic', flush=False)
                self.assertEqual(read_json(live.state_path)['stableSegments'][0]['text'], '公開日は未定です。')
                session = live.session_id
                live.close()
                resumed = LiveMeeting(ws, dict(resumeSessionId=session))
                try:
                    self.assertEqual(resumed.data['tracks']['mic']['received'], 64000)
                    await resumed.accept_audio(packet(1))
                    self.assertEqual(resumed.data['audioBytes'], 128000)
                    resumed.stopping = True
                    resumed.wake.set()
                    await resumed.work()
                    records = [row for row in resumed.records if row.get('type') == 'final']
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]['text'], '公開日は未定です。')
                    self.assertTrue(read_json(resumed.store.metadata_path)['finalized'])
                    self.assertEqual(read_json(resumed.state_path)['stableSegments'], [])
                finally:
                    resumed.close()


class MeetingApiTests(unittest.TestCase):
    setUp = MeetingTests.setUp

    def test_routes_require_login_and_stream_selected_snapshot(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from unittest.mock import AsyncMock
        from server.api.routes.summary import router
        from server.deps import get_current_user
        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as client:
            for path in ('insights', 'recap', 'ask', 'refine'):
                payload = dict(runtimeSessionId='test')
                if path == 'ask':
                    payload['question'] = '公開日は？'
                self.assertEqual(client.post('/api/meeting/'+path, json=payload).status_code, 401)
            app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=123)
            model = SimpleNamespace(model='fixture', stream_meeting=lambda messages: iter(['公開日は未定です。[S1]']))
            with patch('server.api.routes.summary._meeting_source', AsyncMock(return_value=self.snapshot)), patch('server.api.routes.summary._allow_costly_request', return_value=True), patch('server.api.routes.summary.runtime.SUMMARIZER', model):
                response = client.post('/api/meeting/ask', json=dict(runtimeSessionId='test', question='公開日は？'))
                self.assertEqual(response.status_code, 200)
                self.assertIn('text/event-stream', response.headers['content-type'])
                events = [json.loads(line.removeprefix('data: ')) for line in response.text.splitlines() if line.startswith('data: ')]
                self.assertEqual(events[-1]['type'], 'done')
                self.assertEqual(events[-1]['citations'][0]['id'], '0')
                self.assertEqual(client.post('/api/meeting/refine', json=dict(runtimeSessionId='test')).status_code, 409)
