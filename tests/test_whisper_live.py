from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
import tempfile
import threading
import unittest
import wave
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault('APP_SESSION_SECRET', 'whisper-test-only-long-session-secret-value')

from server.core.config import settings
from server.services.meeting_source import MeetingError, load_meeting, read_json, write_json_atomic
from server.services.meeting_translation import translation_events
from server.services.runtime_artifact_service import _revision_jsonl_snapshot
from server.transcript_store import read_jsonl_records
from server.transcription.live import LiveMeeting, live_transcribe
from server.transcription.whisper_live import WhisperLiveMeeting

RATE = 16000
PCM = b'\xff\x0f' * RATE


class Socket:
    state = SimpleNamespace(authenticated_user_id=123, is_guest=False, rate_limit_subject='synthetic-whisper')
    cookies = {}

    def __init__(self):
        self.events = []

    async def send_json(self, event):
        self.events.append(copy.deepcopy(event))


class Model:
    def __init__(self, *, text='会議の内容を確認します。', started=None, release=None):
        self.text, self.started, self.release = text, started, release
        self.requests = []
        self.options = {}
        self.closed = False
        self.client = self

    def with_options(self, **kwargs):
        self.options.update(kwargs)
        return self

    def close(self):
        self.closed = True

    def transcribe_chunk(self, audio, **kwargs):
        with wave.open(io.BytesIO(audio)) as wav:
            samples = wav.getnframes()
        self.requests.append(dict(samples=samples, **kwargs))
        if self.started:
            self.started.set()
        if self.release and not self.release.wait(10):
            raise TimeoutError('synthetic HQ release timeout')
        return SimpleNamespace(text=self.text, start_ms=None, end_ms=None)


class WhisperDualLaneTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, stack, directory, *, enabled=True, rt_seconds=5, models=None):
        root = Path(directory)

        class Config:
            transcripts_dir = root / 'transcripts'
            debug_chunks_dir = root / 'audio'
            asr_backend = 'whisper'
            asr_model = 'synthetic-whisper'
            asr_high_accuracy_enabled = enabled
            asr_high_accuracy_window_seconds = 30
            asr_realtime_window_seconds = rt_seconds

            def __getattr__(self, key):
                return getattr(settings, key)

        config = Config()
        for module in ['server.transcription.live', 'server.transcription.whisper_live']:
            stack.enter_context(patch(module + '.settings', config))
        models = models or (Model(), Model(text='高精度で会議の内容を確認しました。'))
        factories = [stack.enter_context(patch('server.transcription.live.OpenAIWhisperTranscriber', return_value=models[0])),
                     stack.enter_context(patch('server.transcription.whisper_live.OpenAIWhisperTranscriber', return_value=models[1]))]
        stack.enter_context(patch('server.transcription.live.consume', return_value=True))
        stack.enter_context(patch('server.transcription.live.runtime_access_allowed', return_value=True))
        return config, models, factories

    async def until(self, predicate):
        async with asyncio.timeout(10):
            while not predicate():
                await asyncio.sleep(.01)

    async def feed(self, live, first, last, *, track='mic', quiet_tail=False):
        for second in range(first, last):
            pcm = PCM
            if quiet_tail and second == last - 1:
                pcm = PCM[:RATE * 2 // 5] + b'\0' * (RATE * 2 * 4 // 5)
            await live.accept_audio(dict(track=track, seq=second, sampleStart=second * RATE,
                                         pcm=base64.b64encode(pcm).decode()))

    async def test_hq_during_capture_rt_continues_tail_finishes_and_resume_exports_snapshot(self):
        started, release = threading.Event(), threading.Event()
        rt, hq = Model(), Model(text='高精度で会議の内容を確認しました。', started=started, release=release)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config, _, factories = self.fixture(stack, directory, models=(rt, hq))
            ws = Socket()
            live = WhisperLiveMeeting(ws, dict(tracks=['mic'], language='ja', sharedVocabulary='WhistX', prompt='議事録'))
            worker = asyncio.create_task(live.work())
            try:
                await self.feed(live, 0, 30)
                await self.until(started.is_set)
                self.assertFalse(live.stopping)
                before = sum(e['type'] == 'final' for e in ws.events)
                await self.feed(live, 30, 40)
                await self.until(lambda: sum(e['type'] == 'final' for e in ws.events) > before)
                self.assertFalse(any(e['type'] == 'transcript_revision' for e in ws.events))
                release.set()
                await self.until(lambda: any(e['type'] == 'transcript_revision' for e in ws.events))
                event = next(e for e in ws.events if e['type'] == 'transcript_revision')
                self.assertEqual((event['startSample'], event['endSample']), (0, 30 * RATE))
                self.assertEqual(len(event['replacesSegmentIds']), 6)
                self.assertEqual(event['record']['originalText'].count('会議'), 6)
                self.assertEqual(hq.requests[0]['samples'], 30 * RATE)
                self.assertEqual(hq.requests[0]['prompt'], 'WhistX 議事録')
                self.assertTrue(all(r['language'] == 'ja' and r['temperature'] == 0 for r in rt.requests + hq.requests))
                self.assertEqual(factories[0].call_args.kwargs['model'], factories[1].call_args.kwargs['model'])
                self.assertEqual(hq.options['timeout'], config.asr_high_accuracy_timeout_seconds)
                self.assertEqual(live.data['asrRequests'], len(rt.requests) + len(hq.requests))
                with patch('server.services.meeting_source.settings', config), patch('server.services.meeting_source.runtime_access_allowed', return_value=True):
                    snapshot = load_meeting(user_id=123, runtime_session_id=live.session_id)
                self.assertFalse(snapshot.finalized)
                translator = SimpleNamespace(complete_meeting=lambda messages, **kwargs: json.dumps({'translations': [
                    {'id': item['id'], 'text': 'Synthetic HQ translation'} for item in json.loads(messages[1]['content'])]}))
                translations = list(translation_events(snapshot, translator, 'en', cancelled=threading.Event()))
                translated = [e for e in translations if e['type'] == 'translation']
                self.assertEqual(len(translated), 1)
                self.assertEqual(translated[0]['sourceText'], event['record']['text'])
                live.stopping = True
                await asyncio.wait_for(worker, 10)
                self.assertTrue(live.data['finalized'])
                rows = read_jsonl_records(live.store.jsonl_path)
                self.assertEqual([r['endSample'] for r in rows], [30 * RATE, 40 * RATE])
                self.assertTrue(all(r['quality'] == 'high_accuracy' for r in rows))
                self.assertEqual(sum(r['type'] == 'revision' for r in read_jsonl_records(live.store.jsonl_path, materialize=False)), 2)
                self.assertEqual([json.loads(line) for line in _revision_jsonl_snapshot(live.store.jsonl_path).splitlines()], rows)
                self.assertTrue(live.store.jsonl_path.with_suffix('.revisions.jsonl').exists())
                self.assertTrue(read_json(live.store.metadata_path)['finalized'])
                session = live.session_id
                live.close()
                resumed = WhisperLiveMeeting(Socket(), dict(resumeSessionId=session, language='en'))
                try:
                    self.assertEqual(resumed.records, rows)
                    self.assertEqual(resumed.data['language'], 'ja')
                    self.assertEqual(resumed.data['tracks']['mic']['hqCommitted'], 40 * RATE)
                finally:
                    resumed.close()
            finally:
                release.set()
                live.disconnected = True
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                live.close()

    async def test_pause_runs_early_without_splitting_realtime_windows(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.fixture(stack, directory)
            live = WhisperLiveMeeting(Socket(), dict(tracks=['mic']))
            try:
                await self.feed(live, 0, 10, quiet_tail=True)
                while await live._infer('mic', flush=False):
                    pass
                self.assertEqual(live._next_hq(), ('mic', 0, 10 * RATE))
                await live._hq_interval(None, *live._next_hq())
                self.assertEqual(live.records[0]['endSample'], 10 * RATE)
                self.assertEqual(live._window_end(25 * RATE), 30 * RATE)
            finally:
                live.close()

    async def test_websocket_route_selects_whisper_dual_lane_and_isolates_tracks(self):
        class RoutedSocket(Socket):
            def __init__(self):
                super().__init__()
                self.frames = iter([json.dumps(dict(type='audio', track=track, seq=second,
                    sampleStart=second * RATE, pcm=base64.b64encode(PCM).decode()))
                    for second in range(5) for track in ['mic', 'display']] + [json.dumps(dict(type='stop'))])

            async def accept(self):
                pass

            async def receive_json(self):
                return dict(type='start', tracks=['mic', 'display'], language='ja')

            async def receive_text(self):
                return next(self.frames)

            async def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.fixture(stack, directory)
            ws = RoutedSocket()
            await asyncio.wait_for(live_transcribe(ws), 10)
            ready = next(e for e in ws.events if e.get('message') == 'ready')
            self.assertEqual(ready['asrBackend'], 'whisper')
            self.assertTrue(ready['highAccuracyEnabled'])
            revisions = [e for e in ws.events if e['type'] == 'transcript_revision']
            self.assertEqual({e['track'] for e in revisions}, {'mic', 'display'})
            for event in revisions:
                self.assertTrue(all(segment.startswith(event['track'] + '-') for segment in event['replacesSegmentIds']))
                self.assertEqual(event['record']['speaker'], '自分（マイク）' if event['track'] == 'mic' else '共有音声')
            self.assertEqual(ws.events[-1]['message'], 'finalized')

    async def test_disabled_hq_and_one_second_windows_still_finalize_realtime(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            _, models, factories = self.fixture(stack, directory, enabled=False, rt_seconds=1)
            live = WhisperLiveMeeting(Socket(), dict(tracks=['mic']))
            try:
                await self.feed(live, 0, 1)
                self.assertTrue(await live._infer('mic', flush=False))
                self.assertEqual(len(live.records), 1)
                live.stopping = True
                await asyncio.wait_for(live.work(), 10)
                self.assertTrue(live.data['finalized'])
                self.assertEqual(live.records[0]['quality'], 'realtime')
                factories[1].assert_not_called()
                self.assertEqual(len(models[0].requests), 1)
            finally:
                live.close()

    async def test_empty_or_failed_hq_keeps_realtime_audio_and_finishes(self):
        for result in ['', RuntimeError('synthetic recognition failure')]:
            with self.subTest(result=type(result).__name__), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                self.fixture(stack, directory)
                live = WhisperLiveMeeting(Socket(), dict(tracks=['mic']))
                original_sleep = asyncio.sleep

                async def fast_retry(delay):
                    await original_sleep(min(delay, .01))

                try:
                    await self.feed(live, 0, 5)
                    await live._infer('mic', flush=False)
                    rt_rows = copy.deepcopy(live.records)
                    decoder = AsyncMock(return_value=result) if isinstance(result, str) else AsyncMock(side_effect=result)
                    live.stopping = True
                    with patch.object(live, '_batch_text', decoder), patch('server.transcription.high_accuracy.asyncio.sleep', fast_retry):
                        await asyncio.wait_for(live.work(), 10)
                    self.assertTrue(live.data['finalized'])
                    self.assertEqual(live.records, rt_rows)
                    self.assertEqual((live.store.chunks_dir / 'mic.pcm').read_bytes(), PCM * 5)
                    reason = 'empty_result' if isinstance(result, str) else 'recognition_failed'
                    self.assertEqual(live.data['hqRetainedIntervals'][0]['reason'], reason)
                    self.assertEqual(decoder.await_count, 1 if isinstance(result, str) else 3)
                finally:
                    live.close()

    async def test_legacy_resume_preserves_old_rows_and_model_identity(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            _, models, _ = self.fixture(stack, directory)
            old = LiveMeeting(Socket(), dict(tracks=['mic']))
            await self.feed(old, 0, 24)
            await old._infer('mic', flush=False)
            original = copy.deepcopy(old.records)
            session = old.session_id
            old.close()
            resumed = WhisperLiveMeeting(Socket(), dict(resumeSessionId=session, language='en'))
            try:
                self.assertEqual([r['text'] for r in resumed.records], [r['text'] for r in original])
                self.assertEqual(resumed.data['tracks']['mic']['hqCommitted'], 24 * RATE)
                self.assertEqual(resumed.data['language'], 'ja')
                await self.feed(resumed, 24, 30)
                while await resumed._infer('mic', flush=False):
                    pass
                self.assertEqual(resumed._next_hq(), ('mic', 24 * RATE, 30 * RATE))
                await resumed._hq_interval(None, *resumed._next_hq())
                self.assertEqual(resumed.records[0]['text'], original[0]['text'])
                self.assertEqual(resumed.records[-1]['quality'], 'high_accuracy')
                data = read_json(resumed.state_path)
                data['asrModel'] = 'another-model'
                write_json_atomic(resumed.state_path, data)
            finally:
                resumed.close()
            with self.assertRaisesRegex(MeetingError, 'meeting_model_mismatch'):
                WhisperLiveMeeting(Socket(), dict(resumeSessionId=session))
            self.assertTrue(models[0].closed)
