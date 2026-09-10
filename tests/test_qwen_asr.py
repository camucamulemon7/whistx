from __future__ import annotations

import asyncio
import base64
import copy
from json import loads
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault('APP_SESSION_SECRET', 'qwen-test-only-long-session-secret-value')

from server.core.config import settings
from server.qwen_asr import batch_payload, batch_request, clean_qwen_text, realtime_url, response_text
from server.transcript_store import read_jsonl_records
from server.transcription.qwen_live import QwenLiveMeeting
from server.transcription.revisions import apply_revision


class ProtocolTests(unittest.TestCase):
    def test_selected_language_uses_the_transcription_endpoint(self):
        for language in ['ja', 'en']:
            endpoint, options = batch_request(b'wav', mime_type='audio/wav', model='qwen', context='', priority=0, language=language)
            self.assertEqual(endpoint, '/audio/transcriptions')
            self.assertEqual(options['data']['language'], language)
            self.assertEqual(options['files']['file'][1], b'wav')
        endpoint, options = batch_request(b'wav', mime_type='audio/wav', model='qwen', context='', priority=0)
        self.assertEqual(endpoint, '/chat/completions')
        self.assertNotIn('language', options['json'])

    def test_same_model_mixed_language_audio_priority_and_markers(self):
        body = batch_payload(b'wav', mime_type='audio/wav', model='Qwen3-ASR-1.7B', context='WhistX API', priority=10)
        self.assertEqual(body['model'], 'Qwen3-ASR-1.7B')
        self.assertEqual(body['priority'], 10)
        self.assertNotIn('language', body)
        self.assertIn('data:audio/wav;base64,', body['messages'][-1]['content'][0]['audio_url']['url'])
        self.assertEqual(clean_qwen_text('language Japanese<asr_text>APIをreviewします。<|im_end|>'), 'APIをreviewします。')
        self.assertEqual(realtime_url('http://localhost:8004/v1/'), 'ws://localhost:8004/v1/realtime')
        self.assertEqual(realtime_url('https://example.test/v1'), 'wss://example.test/v1/realtime')
        for value in ['file:///tmp/a', 'http://user:password@localhost/v1', 'http://localhost/v1?token=a']:
            with self.assertRaises(ValueError):
                realtime_url(value)
        with self.assertRaises(ValueError):
            response_text({'choices': [{'finish_reason': 'length', 'message': {'content': '途中'}}]})

    def test_replacement_scope_idempotence_and_stale_rejection(self):
        rows = [dict(type='final', segmentId=f'{t}-{i}', seq=i, track=t, startSample=i*16000,
                     endSample=(i+1)*16000, tsStart=i*1000, tsEnd=(i+1)*1000, quality='realtime', text='速報')
                for t in ['mic', 'display'] for i in range(2)]
        record = dict(type='final', segmentId='mic-hq', seq=0, track='mic', startSample=0,
                      endSample=32000, tsStart=0, tsEnd=2000, quality='high_accuracy', text='再認識')
        event = dict(type='revision', track='mic', startSample=0, endSample=32000, record=record,
                     replacesSegmentIds=['mic-0', 'mic-1'])
        updated = apply_revision(rows, event)
        self.assertEqual(len(updated), 3)
        self.assertEqual(apply_revision(updated, event), updated)
        self.assertEqual(sum(r['track'] == 'display' for r in updated), 2)
        for change in [dict(replacesSegmentIds=['mic-0']), dict(endSample=16000), dict(track='display')]:
            with self.assertRaises(ValueError):
                apply_revision(rows, {**event, **change})


class DualLaneTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_language_applies_to_short_and_high_accuracy_windows_and_resume(self):
        requests = []
        class Socket:
            state = SimpleNamespace(authenticated_user_id=123, is_guest=False, rate_limit_subject='fixture')
            cookies = {}
            async def send_json(self, event):
                pass
        class Client:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **options):
                requests.append(options['data'])
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(text='指定された日本語の文字起こしです。'))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            class Config:
                transcripts_dir = Path(directory)/'transcripts'
                debug_chunks_dir = Path(directory)/'audio'
                asr_backend = 'qwen3_vllm'
                asr_model = 'Qwen3-ASR-1.7B'
                openai_base_url = 'http://localhost:8004/v1'
                asr_realtime_window_seconds = 5
                def __getattr__(self, key):
                    return getattr(settings, key)
            for module in ['server.transcription.live', 'server.transcription.qwen_live']:
                stack.enter_context(patch(module+'.settings', Config()))
            stack.enter_context(patch('server.transcription.qwen_live.httpx.AsyncClient', Client))
            realtime = stack.enter_context(patch('server.transcription.qwen_live.QwenRealtime'))
            stack.enter_context(patch.object(QwenLiveMeeting, '_request_budget', AsyncMock()))
            live = QwenLiveMeeting(Socket(), dict(tracks=['mic'], language='ja'))
            try:
                pcm = b'\xff\x0f' * 80000
                for seq in range(5):
                    await live.accept_audio(dict(track='mic', seq=seq, sampleStart=seq * 16000,
                                                 pcm=base64.b64encode(pcm[seq * 32000:(seq + 1) * 32000]).decode()))
                await live._rt_window('mic')
                self.assertFalse(live.stopping)
                self.assertEqual(live.records[-1]['language'], 'ja')
                self.assertEqual(live.records[-1]['text'], '指定された日本語の文字起こしです。')
                self.assertEqual(requests[-1]['priority'], '0')
                await live._batch_text(Client(), pcm)
                self.assertEqual(requests[-1]['priority'], str(settings.asr_high_accuracy_priority))
                self.assertTrue(all(request['language'] == 'ja' for request in requests))
                realtime.assert_not_called()
                session_id = live.session_id
            finally:
                live.close()
            resumed = QwenLiveMeeting(Socket(), dict(tracks=['mic'], resumeSessionId=session_id, language='en'))
            try:
                self.assertEqual(resumed.data['language'], 'ja')
                self.assertEqual(resumed.store.read_metadata()['language'], 'ja')
            finally:
                resumed.close()

    async def test_hq_during_capture_rt_continues_and_resume_replays_revision(self):
        class Socket:
            state = SimpleNamespace(authenticated_user_id=123, is_guest=False, rate_limit_subject='fixture')
            cookies = {}
            def __init__(self):
                self.events = []
            async def send_json(self, event):
                self.events.append(copy.deepcopy(event))

        class Stream:
            def __init__(self, **kwargs):
                self.done = False
                self.queue = asyncio.Queue()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def append(self, pcm):
                await self.queue.put(dict(type='transcription.delta', delta='APIをreview。'))
            async def finish(self):
                await self.queue.put(dict(type='transcription.done', text='APIをreviewします。'))
            async def receive(self):
                event = await self.queue.get()
                self.done = event['type'] == 'transcription.done'
                return event

        started, release = asyncio.Event(), asyncio.Event()
        requests = []
        class Client:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, *, json):
                requests.append((url, json))
                started.set()
                await release.wait()
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(choices=[dict(
                    finish_reason='stop', message=dict(content='language Japanese<asr_text>APIのreviewを実施します。'))]))

        async def until(predicate):
            async with asyncio.timeout(10):
                while not predicate():
                    await asyncio.sleep(.01)

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            class Config:
                transcripts_dir = root/'transcripts'
                debug_chunks_dir = root/'audio'
                asr_backend = 'qwen3_vllm'
                asr_model = 'Qwen3-ASR-1.7B'
                openai_base_url = 'http://localhost:8004/v1'
                asr_high_accuracy_window_seconds = 30
                asr_realtime_window_seconds = 5
                asr_high_accuracy_enabled = True
                def __getattr__(self, key):
                    return getattr(settings, key)
            for module in ['server.transcription.live', 'server.transcription.qwen_live']:
                stack.enter_context(patch(module+'.settings', Config()))
            stack.enter_context(patch('server.transcription.qwen_live.QwenRealtime', Stream))
            stack.enter_context(patch('server.transcription.qwen_live.httpx.AsyncClient', Client))
            stack.enter_context(patch.object(QwenLiveMeeting, '_request_budget', AsyncMock()))
            ws = Socket()
            live = QwenLiveMeeting(ws, dict(tracks=['mic'], language='auto', sharedVocabulary='WhistX'))
            worker = asyncio.create_task(live.work())
            async def feed(first, last):
                for i in range(first, last):
                    await live.accept_audio(dict(track='mic', seq=i, sampleStart=i*16000,
                        pcm=base64.b64encode(b'\xff\x0f'*16000).decode()))
                    await asyncio.sleep(.002)
            try:
                await feed(0, 30)
                await asyncio.wait_for(started.wait(), 10)
                self.assertFalse(live.stopping)
                before = len([e for e in ws.events if e['type'] == 'final'])
                await feed(30, 40)
                await until(lambda: len([e for e in ws.events if e['type'] == 'final']) > before)
                self.assertFalse(any(e['type'] == 'transcript_revision' for e in ws.events))
                release.set()
                await until(lambda: any(e['type'] == 'transcript_revision' for e in ws.events))
                self.assertFalse(live.stopping)
                event = next(e for e in ws.events if e['type'] == 'transcript_revision')
                self.assertEqual((event['startSample'], event['endSample']), (0, 30*16000))
                self.assertEqual(len(event['replacesSegmentIds']), 6)
                self.assertEqual(requests[0][0], 'http://localhost:8004/v1/chat/completions')
                self.assertEqual(requests[0][1]['priority'], 10)
                self.assertEqual(requests[0][1]['messages'][0]['content'], 'WhistX')
                live.stopping = True
                await asyncio.wait_for(worker, 10)
                self.assertTrue(live.data['finalized'])
                materialized = read_jsonl_records(live.store.jsonl_path)
                self.assertEqual([r['endSample'] for r in materialized], [30*16000, 40*16000])
                raw = read_jsonl_records(live.store.jsonl_path, materialize=False)
                self.assertEqual(sum(r['type'] == 'revision' for r in raw), 2)
                self.assertEqual(materialized[0]['originalText'].count('API'), 6)
                session_id = live.session_id
                live.close()
                stack.enter_context(patch('server.transcription.live.runtime_access_allowed', return_value=True))
                resumed = QwenLiveMeeting(Socket(), dict(resumeSessionId=session_id))
                try:
                    self.assertEqual(resumed.records, materialized)
                    self.assertEqual(resumed.data['tracks']['mic']['hqCommitted'], 40*16000)
                    self.assertIn('APIのreview', resumed.store.txt_path.read_text())
                finally:
                    resumed.close()
            finally:
                live.disconnected = True
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                live.close()


class MixedCoverageTests(unittest.IsolatedAsyncioTestCase):
    async def test_dropped_japanese_is_redecoded_by_contiguous_source_ranges(self):
        from server.qwen_asr import language_coverage_lost
        japanese = '水をマレーシアから買わなくてはならないのです。'
        english = 'He started writing music for other people.'
        self.assertTrue(language_coverage_lost(japanese + english, english))
        self.assertTrue(language_coverage_lost(japanese + english, japanese))
        self.assertFalse(language_coverage_lost('APIを確認します。', 'APIを確認します。'))
        self.assertFalse(language_coverage_lost(japanese + english, japanese + english))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            class Config:
                transcripts_dir = root/'transcripts'
                debug_chunks_dir = root/'audio'
                asr_backend = 'qwen3_vllm'
                asr_model = 'Qwen3-ASR-1.7B'
                def __getattr__(self, key):
                    return getattr(settings, key)
            for module in ['server.transcription.live', 'server.transcription.qwen_live']:
                stack.enter_context(patch(module+'.settings', Config()))
            ws = SimpleNamespace(state=SimpleNamespace(is_guest=False), send_json=AsyncMock())
            live = QwenLiveMeeting(ws, dict(tracks=['mic']))
            try:
                live._write_audio('mic', 0, b'\xff\x0f' * 480000)
                live.data['tracks']['mic'].update(received=480000, windowStart=480000)
                rows = [live._record('mic', i*160000, (i+1)*160000, text, quality='realtime', seq=i)
                        for i, text in enumerate([japanese, english, japanese])]
                for row in rows:
                    live.store.append_record(row)
                live.records = rows
                decoder = AsyncMock(side_effect=[english, japanese, english, japanese])
                with patch.object(live, '_batch_text', decoder):
                    with patch('server.services.meeting_refinement._atomic_text', side_effect=OSError('simulated TXT failure')):
                        with self.assertRaises(OSError):
                            await live._hq_interval(None, 'mic', 0, 480000)
                self.assertEqual(live.data['tracks']['mic']['hqCommitted'], 0)
                await live._recover_commits()
                await live._recover_commits()
                self.assertEqual(live.data['tracks']['mic']['hqCommitted'], 480000)
                self.assertEqual(sum(r['type'] == 'revision' for r in read_jsonl_records(live.store.jsonl_path, materialize=False)), 1)
                self.assertEqual(decoder.await_count, 4)
                self.assertEqual(len(live.records), 1)
                result = live.records[0]
                self.assertEqual(result['text'], '\n'.join([japanese, english, japanese]))
                self.assertEqual([r['startSample'] for r in result['highAccuracySegments']], [0, 160000, 320000])
                self.assertEqual(result['endSample'], 480000)
                self.assertNotIn('retainedRealtimeSegmentIds', result)
                from server.services.runtime_artifact_service import _qwen_jsonl_snapshot
                exported = [loads(line) for line in _qwen_jsonl_snapshot(live.store.jsonl_path).splitlines()]
                self.assertEqual(exported, live.records)
            finally:
                live.close()

    async def test_unusable_monolingual_revision_keeps_realtime_without_blocking_stop(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            class Config:
                transcripts_dir = root/'transcripts'
                debug_chunks_dir = root/'audio'
                asr_backend = 'qwen3_vllm'
                asr_model = 'Qwen3-ASR-1.7B'
                def __getattr__(self, key):
                    return getattr(settings, key)
            for module in ['server.transcription.live', 'server.transcription.qwen_live']:
                stack.enter_context(patch(module+'.settings', Config()))
            ws = SimpleNamespace(state=SimpleNamespace(is_guest=False), send_json=AsyncMock())
            live = QwenLiveMeeting(ws, dict(tracks=['mic']))
            try:
                live._write_audio('mic', 0, b'\xff\x0f' * 160000)
                live.data['tracks']['mic'].update(received=160000, windowStart=160000)
                row = live._record('mic', 0, 160000, '水をマレーシアから買わなくてはならないのです。', quality='realtime', seq=0)
                live.store.append_record(row)
                live.records = [row]
                with patch.object(live, '_batch_text', AsyncMock(return_value='We need to buy water from Malaysia.')):
                    await live._hq_interval(None, 'mic', 0, 160000)
                self.assertEqual(live.records, [row])
                self.assertEqual(live.data['tracks']['mic']['hqCommitted'], 160000)
                self.assertEqual(ws.send_json.call_args.args[0]['state'], 'retained')
                self.assertEqual(read_jsonl_records(live.store.jsonl_path), [row])
            finally:
                live.close()
