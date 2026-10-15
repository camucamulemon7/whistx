"""Guard repeated-PCM work without wall-clock performance assertions."""
import asyncio
import base64
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault('APP_SESSION_SECRET', 'realtime-performance-test-session-secret')

from server.core.config import settings
from server.pcm_gate import pcm_speech_bounds
from server.transcription.local_agreement import SAMPLE_RATE, pcm_wav
from server.transcription.qwen_live import QwenLiveMeeting
from tests.whisper_audio_fixtures import pcm as synthetic_pcm


class Socket:
    state = SimpleNamespace(authenticated_user_id=123, is_guest=False, rate_limit_subject='synthetic')
    cookies = {}

    def __init__(self):
        self.events = []

    async def send_json(self, event):
        self.events.append(event)


class BurstStream:
    def __init__(self, **kwargs):
        self.done = False
        self.queue = asyncio.Queue()
        self.received = 0
        self.appends = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def append(self, pcm):
        self.appends += 1
        for _ in range(64):
            self.queue.put_nowait(dict(type='transcription.delta', delta='はい'))

    async def finish(self):
        self.queue.put_nowait(dict(type='transcription.done', text='はい'))

    async def receive(self):
        event = await self.queue.get()
        self.received += 1
        self.done = event['type'] == 'transcription.done'
        return event


@contextmanager
def qwen_fixture(*, tracks=('mic',), seconds=1, stream=None, language='auto'):
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        class Config:
            transcripts_dir = Path(directory) / 'transcripts'
            debug_chunks_dir = Path(directory) / 'audio'
            asr_backend = 'qwen3_vllm'
            asr_model = 'synthetic-qwen'
            asr_realtime_window_seconds = seconds
            openai_base_url = 'http://synthetic.invalid/v1'

            def __getattr__(self, key):
                return getattr(settings, key)

        for module in ('live', 'qwen_live'):
            stack.enter_context(patch('server.transcription.' + module + '.settings', Config()))
        stack.enter_context(patch.object(QwenLiveMeeting, '_request_budget', AsyncMock()))
        stack.enter_context(patch('server.transcription.qwen_live.QwenRealtime',
                                  side_effect=lambda **kwargs: stream or BurstStream()))
        live = QwenLiveMeeting(Socket(), dict(tracks=list(tracks), language=language))
        try:
            yield live
        finally:
            live.close()


async def feed(live, track, pcm):
    state = live.data['tracks'][track]
    await live.accept_audio(dict(track=track, seq=state['seq'] + 1,
        sampleStart=state['received'], pcm=base64.b64encode(pcm).decode()))


class RealtimeGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_burst_reuses_gate_and_read_without_dropping_deltas(self):
        with qwen_fixture() as live:
            pcm = synthetic_pcm(1, amplitude=4000)
            await feed(live, 'mic', pcm)
            with patch.object(live, '_read_audio', wraps=live._read_audio) as read, \
                    patch('server.transcription.qwen_live.speech_bounds', wraps=pcm_speech_bounds) as gate:
                await live._rt_window('mic')
            # One forwarding read + one full-prefix read. The final reuses it.
            self.assertEqual(read.call_count, 2)
            self.assertEqual(gate.call_count, 2)  # Prefix + independent quiet tail.
            partials = [e for e in live.ws.events if e['type'] == 'partial' and e.get('text')]
            self.assertEqual(len(partials), 64)
            self.assertEqual(partials[-1]['text'], 'はい' * 64)
            self.assertEqual(live.records[0]['text'], 'はい')
            self.assertEqual(live.records[0]['endSample'], SAMPLE_RATE)
            self.assertEqual((live.store.chunks_dir / 'mic.pcm').read_bytes(), pcm)

    async def test_partial_timestamp_tracks_the_gated_snapshot_during_append(self):
        from server.core.blocking import blocking_work_pool
        started, release = asyncio.Event(), asyncio.Event()
        original_run = blocking_work_pool.run

        async def run(kind, func, *args, **kwargs):
            if func is pcm_speech_bounds and not started.is_set():
                started.set()
                await release.wait()
            return await original_run(kind, func, *args, **kwargs)

        stream = BurstStream()
        with qwen_fixture(seconds=2, stream=stream) as live, patch.object(blocking_work_pool, 'run', run):
            await feed(live, 'mic', synthetic_pcm(1, amplitude=4000))
            worker = asyncio.create_task(live._rt_window('mic'))
            try:
                await asyncio.wait_for(started.wait(), 5)
                await feed(live, 'mic', synthetic_pcm(1, amplitude=4000))
                async with asyncio.timeout(5):
                    while stream.appends < 2:
                        await asyncio.sleep(.001)
                release.set()
                await asyncio.wait_for(worker, 5)
                partials = [e for e in live.ws.events if e['type'] == 'partial' and e.get('text')]
                self.assertEqual(partials[0]['tsEnd'], 1000)
                self.assertEqual(partials[-1]['tsEnd'], 2000)
            finally:
                release.set()
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    async def test_silence_cache_expires_when_audio_grows(self):
        stream = BurstStream()
        with qwen_fixture(seconds=2, stream=stream) as live:
            await feed(live, 'mic', b'\0\0' * SAMPLE_RATE)
            worker = asyncio.create_task(live._rt_window('mic'))
            try:
                async with asyncio.timeout(5):
                    while stream.received < 64:
                        await asyncio.sleep(.001)
                self.assertFalse(any(e.get('text') for e in live.ws.events))
                await feed(live, 'mic', synthetic_pcm(.12) + b'\0\0' * 3200)
                live.stopping = True
                await asyncio.wait_for(worker, 5)
                self.assertTrue(any(e.get('text') for e in live.ws.events if e['type'] == 'partial'))
                self.assertEqual(live.records[0]['text'], 'はい')
                self.assertEqual(live.records[0]['endSample'], SAMPLE_RATE + 5120)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    async def test_cache_is_scoped_to_track_and_window(self):
        with qwen_fixture(tracks=('mic', 'display')) as live:
            await feed(live, 'mic', synthetic_pcm(1, amplitude=4000))
            await feed(live, 'display', b'\0\0' * SAMPLE_RATE)
            await live._rt_window('mic')
            await live._rt_window('display')
            await feed(live, 'mic', b'\0\0' * SAMPLE_RATE)
            await live._rt_window('mic')
            self.assertEqual([(r['track'], r['startSample'], r['endSample']) for r in live.records],
                             [('mic', 0, SAMPLE_RATE)])
            self.assertEqual(live.data['tracks']['mic']['windowStart'], 2 * SAMPLE_RATE)
            self.assertFalse(any(e.get('text') for e in live.ws.events if e.get('track') == 'display'))

    async def test_speech_cpp_live_gates_once_and_keeps_identical_upload(self):
        import httpx
        from server.speech_cpp_asr import SpeechCppBatchTranscriber
        from server.transcription.speech_cpp_live import SpeechCppLiveMeeting
        from tests.test_whisper_live import WhisperDualLaneTests

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config, _, _ = WhisperDualLaneTests().fixture(stack, directory, enabled=False)
            config.asr_backend = 'speech_cpp'
            stack.enter_context(patch('server.transcription.speech_cpp_live.settings', config))
            model = SpeechCppBatchTranscriber(api_key='', base_url='http://synthetic.invalid/v1', model='test')
            model.client.close()
            uploads = []

            def respond(request):
                uploads.append(request.content.split(b'RIFF', 1)[1].rsplit(b'\r\n--', 1)[0])
                return httpx.Response(200, json={'text': 'はい'}, headers={'X-Speech-Stop': 'complete'})

            model.client = httpx.Client(transport=httpx.MockTransport(respond))
            stack.enter_context(patch('server.transcription.speech_cpp_live.SpeechCppBatchTranscriber', return_value=model))
            live = SpeechCppLiveMeeting(Socket(), dict(tracks=['mic'], language='ja'))
            try:
                pcm = b'\0\0' * 8000 + synthetic_pcm(.12) + b'\0\0' * 14080
                # Obtain the unchanged direct-call upload for this exact PCM.
                model.transcribe_chunk(pcm_wav(pcm), mime_type='audio/wav', language='ja', prompt=None)
                await feed(live, 'mic', pcm)
                with patch('server.transcription.whisper_live.whisper_speech_bounds', wraps=pcm_speech_bounds) as gate, \
                        patch('server.pcm_gate.pcm_speech_bounds', side_effect=AssertionError('duplicate gate')):
                    await live._infer('mic', flush=True)
                self.assertEqual(gate.call_count, 1)
                self.assertEqual(uploads[0], uploads[1])
                self.assertEqual(live.records[0]['text'], 'はい')
                self.assertEqual((live.store.chunks_dir / 'mic.pcm').read_bytes(), pcm)
            finally:
                live.close()

    async def test_whisper_bounds_hint_preserves_timestamp_offset(self):
        from tests.whisper_audio_fixtures import transcriber
        pcm = b'\0\0' * SAMPLE_RATE + synthetic_pcm(.12) + b'\0\0' * 3200
        model, uploads = transcriber(segments=[dict(start=0, end=.1)])
        options = dict(mime_type='audio/wav', language='ja', prompt=None, temperature=0)
        before = model.transcribe_chunk(pcm_wav(pcm), **options)
        with patch('server.pcm_gate.pcm_speech_bounds', side_effect=AssertionError('duplicate gate')):
            after = model.transcribe_chunk(pcm_wav(pcm), pcm_bounds=pcm_speech_bounds(pcm), **options)
        self.assertEqual(uploads[0], uploads[1])
        self.assertEqual((before.start_ms, before.end_ms), (after.start_ms, after.end_ms))
        self.assertGreater(after.start_ms, 0)
