import asyncio
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault('APP_SESSION_SECRET', 'synthetic-hq-observation-test-secret-000000')

from server.langfuse_observer import high_accuracy_observation
from server.transcription.qwen_live import QwenLiveMeeting
from server.transcription.whisper_live import WhisperLiveMeeting
from tests.langfuse_fakes import recording_observer


class ObservationTests(unittest.TestCase):
    def test_refinement_route_passes_the_existing_runtime_observer(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from server.api.routes.summary import router
        from server.deps import get_current_user

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=123)
        observer, _ = recording_observer()
        with TestClient(app) as client, \
                patch('server.api.routes.summary._meeting_source', AsyncMock(return_value=SimpleNamespace(finalized=True))), \
                patch('server.api.routes.summary._allow_costly_request', return_value=True), \
                patch('server.api.routes.summary.runtime.resources.observer', observer), \
                patch('server.api.routes.summary.refine_events', return_value=iter([{'type': 'done'}])) as refine:
            self.assertEqual(client.post('/api/meeting/refine', json={'runtimeSessionId': 'synthetic'}).status_code, 200)
            self.assertIs(refine.call_args.kwargs['observer'], observer)

    def observe(self, observer, **kwargs):
        return high_accuracy_observation(observer, session_id='synthetic-meeting', model='private-model',
            start_ms=1000, end_ms=2000, audio_bytes=32000, language='en', track='mic', **kwargs)

    def test_opt_in_and_failure_cancellation_do_not_export_secrets(self):
        for capture in [False, True]:
            observer, calls = recording_observer(capture_content=capture)
            with self.observe(observer) as observation:
                observation.update(output={'text': 'Synthetic discussion a@example.test https://private.test sk-abcdefghijk', 'succeeded': True})
            with self.assertRaises(RuntimeError):
                with self.observe(observer):
                    raise RuntimeError('provider-secret https://private.test')
            with self.assertRaises(asyncio.CancelledError):
                with self.observe(observer):
                    raise asyncio.CancelledError('private cancellation')
            serialized = json.dumps(calls)
            for private in ['a@example.test', 'private.test', 'sk-abcdefghijk', 'provider-secret', 'private cancellation', 'synthetic-meeting']:
                self.assertNotIn(private, serialized)
            self.assertEqual('Synthetic discussion' in serialized, capture)
            self.assertEqual(calls[1]['updates'][0]['output'], {'succeeded': False, 'failed': True, 'cancelled': False})
            self.assertEqual(calls[2]['updates'][0]['output'], {'succeeded': False, 'failed': False, 'cancelled': True})
            self.assertTrue(all(c['exits'] == [(None, None, None)] for c in calls))
            with self.observe(observer, manual=True):
                pass
            self.assertEqual(len({c['start']['trace_context']['trace_id'] for c in calls}), 1)
            self.assertTrue(calls[-1]['start']['metadata']['manual'])

    def test_disabled_and_telemetry_failures_preserve_work_and_provider_error(self):
        for enabled, fail_at in [(False, None), (True, 'start'), (True, 'enter'), (True, 'update'), (True, 'exit')]:
            with self.subTest(enabled=enabled, fail_at=fail_at):
                observer, calls = recording_observer(enabled=enabled, fail_at=fail_at)
                reached = []
                with self.observe(observer) as observation:
                    reached.append('recognition')
                    observation.update(output={'succeeded': True})
                self.assertEqual(reached, ['recognition'])
                with self.assertRaisesRegex(RuntimeError, 'actual recognition failure'):
                    with self.observe(observer):
                        raise RuntimeError('actual recognition failure')
                if not enabled:
                    self.assertEqual(calls, [])


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_backends_trace_real_retries_but_not_committed_replay(self):
        for meeting_class in [WhisperLiveMeeting, QwenLiveMeeting]:
            with self.subTest(backend=meeting_class.__name__), tempfile.TemporaryDirectory() as directory:
                observer, calls = recording_observer()
                meeting = object.__new__(meeting_class)
                meeting.resources = SimpleNamespace(observer=observer)
                meeting.session_id = 'synthetic-retry-meeting'
                meeting.data = dict(asrModel='synthetic-model', language='ja', tracks={'mic': {'hqCommitted': 0}})
                meeting.records = [dict(type='final', quality='realtime', track='mic', startSample=0,
                    endSample=16000, seq=0, segmentId='mic-rt', text='元の文')]
                meeting.state_lock = asyncio.Lock()
                meeting.state_path = Path(directory) / 'state.json'
                meeting.store = SimpleNamespace(append_revision=lambda event, records: [event['record']])
                meeting._save_clip = AsyncMock()
                meeting.checkpoint = AsyncMock()
                meeting.send = AsyncMock()
                meeting._read_audio = lambda *args: b'\x01\0' * 16000
                meeting._batch_text = AsyncMock(side_effect=[RuntimeError('private provider error'), '合成の高精度結果'])
                with self.assertRaises(RuntimeError):
                    await meeting._hq_interval(None, 'mic', 0, 16000)
                self.assertEqual(meeting.data['tracks']['mic']['hqCommitted'], 0)
                await meeting._hq_interval(None, 'mic', 0, 16000)
                await meeting._hq_interval(None, 'mic', 0, 16000)
                self.assertEqual(meeting._batch_text.await_count, 2)
                self.assertEqual(len(calls), 2)
                self.assertTrue(calls[0]['updates'][0]['output']['failed'])
                self.assertTrue(calls[1]['updates'][0]['output']['succeeded'])
                self.assertEqual(calls[0]['start']['trace_context'], calls[1]['start']['trace_context'])
                self.assertNotIn('private provider', json.dumps(calls))


class RefinementObservationTests(unittest.TestCase):
    def test_default_whisper_and_qwen_factories_trace_refinement_and_skip_cancelled_requests(self):
        from server.services.meeting_refinement import refine_events
        from server.services.meeting_source import MeetingSnapshot

        for backend, factory_path in [('whisper', 'server.services.meeting_refinement.OpenAIWhisperTranscriber'),
                                      ('qwen3_vllm', 'server.qwen_asr.QwenBatchTranscriber')]:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / 'meeting.jsonl'
                row = dict(type='final', seq=0, segmentId='mic-rt', text='元の文', language='ja', track='mic',
                           tsStart=30000, tsEnd=35000, rawAudioPath='/api/transcripts/synthetic/audio/clip.wav')
                path.write_text(json.dumps(row) + '\n')
                path.with_suffix('.meta.json').write_text(json.dumps(dict(finalized=True, prompt='synthetic private prompt')))
                audio = root / 'clip.wav'
                audio.write_bytes(b'synthetic audio fixture')
                snapshot = MeetingSnapshot('runtime:synthetic-meeting', path, path.with_suffix('.meeting.json'), [row], [], 'r1', 35000, True)
                config = SimpleNamespace(asr_backend=backend, openai_api_key='synthetic-key',
                    openai_base_url='http://synthetic.invalid', asr_model='synthetic-model', debug_chunks_dir=root)
                model = SimpleNamespace(transcribe_chunk=lambda *a, **kw: SimpleNamespace(text='合成の結果'),
                                        client=SimpleNamespace(close=lambda: None))
                observer, calls = recording_observer()
                with patch('server.services.meeting_refinement.settings', config), \
                        patch('server.services.meeting_refinement.resolve_debug_audio_path', return_value=audio), \
                        patch(factory_path, return_value=model) as factory:
                    events = list(refine_events(snapshot, cancelled=threading.Event(), observer=observer))
                    self.assertEqual(events[-1]['type'], 'done')
                    factory.assert_called_once()
                    self.assertEqual(len(calls), 1)
                    self.assertEqual((calls[0]['start']['input']['startMs'], calls[0]['start']['input']['endMs']), (30000, 35000))
                    self.assertTrue(calls[0]['start']['metadata']['manual'])
                    self.assertTrue(calls[0]['updates'][0]['output']['succeeded'])
                    self.assertNotIn('synthetic audio', json.dumps(calls))
                    self.assertNotIn('synthetic private prompt', json.dumps(calls))
                    self.assertNotIn('synthetic-key', json.dumps(calls))
                    stop = threading.Event()
                    stop.set()
                    self.assertEqual(list(refine_events(snapshot, cancelled=stop, observer=observer)), [])
                    self.assertEqual(len(calls), 1, 'cancellation before recognition must not emit observations')
