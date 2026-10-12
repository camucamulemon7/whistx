import json
import unittest
from unittest.mock import AsyncMock, patch

from server.speech_cpp_asr import speech_batch_request


class SpeechCppBatchTests(unittest.TestCase):
    def make_transcriber(self, response):
        import httpx
        from server.speech_cpp_asr import SpeechCppBatchTranscriber
        model = SpeechCppBatchTranscriber(api_key='', base_url='http://synthetic.test/v1', model='qwen')
        self.assertNotIn('Authorization', model.client.headers)
        model.client.close()
        self.requests = []
        def respond(request):
            self.requests.append(request)
            return response
        model.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(model.close)
        return model

    def test_batch_uses_transcription_endpoint_for_auto_language(self):
        endpoint, request = speech_batch_request(b'RIFF', mime_type='audio/wav',
            model='native-alias', context='meeting', language='auto', priority=10)
        self.assertEqual(endpoint, '/audio/transcriptions')
        for key in ('language','priority','temperature'):
            self.assertNotIn(key, request['data'])

    def test_silence_avoids_request_and_batch_model_limit_is_rejected(self):
        import httpx
        from tests.whisper_audio_fixtures import pcm
        from server.transcription.local_agreement import pcm_wav
        model = self.make_transcriber(httpx.Response(200, json={'text':'partial'}, headers={'X-Speech-Stop':'model_limit'}))
        result = model.transcribe_chunk(pcm_wav(pcm(.3, amplitude=0)), mime_type='audio/wav', language='ja', prompt=None)
        self.assertTrue(result.silence_detected)
        self.assertEqual(self.requests, [])
        with self.assertRaisesRegex(ValueError, 'model_limit'):
            model.transcribe_chunk(pcm_wav(pcm(.3)), mime_type='audio/wav', language='ja', prompt=None)

    def test_legacy_webm_is_decoded_before_native_transcription(self):
        import httpx
        import subprocess
        from tests.whisper_audio_fixtures import pcm
        from server.transcription.local_agreement import pcm_wav
        audio = subprocess.run(['ffmpeg','-v','error','-f','wav','-i','pipe:0','-f','webm','pipe:1'],
                               input=pcm_wav(pcm(.4)), capture_output=True, check=True).stdout
        model = self.make_transcriber(httpx.Response(200, json={'text':'確認しました。'}))
        result = model.transcribe_chunk(audio, mime_type='audio/webm', language='auto', prompt='test')
        self.assertEqual(result.text, '確認しました。')
        request = self.requests[0]
        self.assertEqual(request.url.path, '/v1/audio/transcriptions')
        self.assertIn(b'RIFF', request.content)
        self.assertNotIn(b'name="temperature"', request.content)
        self.assertNotIn(b'name="priority"', request.content)
        self.assertNotIn(b'name="language"', request.content)

    def test_async_high_accuracy_checks_batch_stop_header(self):
        import httpx
        from server.speech_cpp_asr import speech_response_text
        with self.assertRaisesRegex(ValueError, 'model_limit'):
            speech_response_text(httpx.Response(200, json={'text':'partial'}, headers={'llm_provider-x-speech-stop':'model_limit'}))

    def test_gateway_complete_header_and_conflicting_stop_are_checked(self):
        import httpx
        from server.speech_cpp_asr import speech_response_text
        self.assertEqual(speech_response_text(httpx.Response(200, json={'text':'確認'},
            headers={'llm_provider-x-speech-stop':'complete'})), '確認')
        with self.assertRaisesRegex(ValueError, 'model_limit'):
            speech_response_text(httpx.Response(200, json={'text':'partial'},
                headers={'X-Speech-Stop':'complete','llm_provider-x-speech-stop':'model_limit'}))

    def test_http_status_and_timeout_fail_without_a_successful_result(self):
        import httpx
        from tests.whisper_audio_fixtures import pcm
        from server.transcription.local_agreement import pcm_wav
        model = self.make_transcriber(httpx.Response(502, json={'error':'synthetic-upstream-secret'}))
        with self.assertRaises(httpx.HTTPStatusError):
            model.transcribe_chunk(pcm_wav(pcm(.3)), mime_type='audio/wav', language='ja', prompt=None)
        model.client.close()
        def timeout(request):
            raise httpx.ReadTimeout('synthetic timeout', request=request)
        model.client = httpx.Client(transport=httpx.MockTransport(timeout))
        with self.assertRaises(httpx.ReadTimeout):
            model.transcribe_chunk(pcm_wav(pcm(.3)), mime_type='audio/wav', language='ja', prompt=None)

    def test_format_conversion_has_no_audio_filters(self):
        import subprocess
        from server.audio_pipeline import AudioPreprocessor
        with patch('server.audio_pipeline.shutil.which', return_value='/synthetic/ffmpeg'), \
                patch('server.audio_pipeline.subprocess.run', return_value=subprocess.CompletedProcess([],0,b'\0\0'*1600,b'')) as run:
            audio = AudioPreprocessor(ffmpeg_bin='ffmpeg', sample_rate=16000,
                overlap_ms=0, enabled=False).convert_to_wav(audio_bytes=b'synthetic', mime_type='audio/webm')
        self.assertTrue(audio.startswith(b'RIFF'))
        self.assertNotIn('-af', run.call_args.args[0])


class SpeechCppHttpLiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_finer_updates_preserve_windows_and_share_the_track_budget(self):
        import base64
        import tempfile
        from contextlib import ExitStack
        from server.transcription.speech_cpp_live import SpeechCppLiveMeeting
        from tests.test_whisper_live import WhisperDualLaneTests, Model, Socket, PCM, RATE
        helper = WhisperDualLaneTests()
        for tracks, durations in [(['display'], [1.25, 2.5, 3.75, 5]),
                                  (['mic', 'display'], [2.5, 5])]:
            with self.subTest(tracks=tracks), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                config, _, _ = helper.fixture(stack, directory, enabled=False)
                config.asr_backend = 'speech_cpp'
                stack.enter_context(patch('server.transcription.speech_cpp_live.settings', config))
                model = Model()
                stack.enter_context(patch('server.transcription.speech_cpp_live.SpeechCppBatchTranscriber', return_value=model))
                ws = Socket()
                live = SpeechCppLiveMeeting(ws, dict(tracks=tracks, language='ja'))
                try:
                    for seq in range(20):
                        for track in tracks:
                            await live.accept_audio(dict(track=track, seq=seq, sampleStart=seq*RATE//4,
                                pcm=base64.b64encode(PCM[:RATE//2]).decode()))
                            await live._infer(track, flush=False)
                    self.assertEqual([r['samples'] for r in model.requests],
                                     [int(seconds*RATE) for seconds in durations for _ in tracks])
                    for track in tracks:
                        rows = [r for r in live.records if r['track'] == track]
                        self.assertEqual([(r['startSample'], r['endSample']) for r in rows], [(0, 5*RATE)])
                        self.assertEqual((live.store.chunks_dir/(track+'.pcm')).read_bytes(), PCM*5)
                    self.assertEqual(sum(e['type']=='partial' and bool(e.get('text')) for e in ws.events),
                                     (len(durations)-1)*len(tracks))
                    # At most 48 RT calls/min with default 5 s windows, plus
                    # 2 HQ calls per track: headroom under the 60/min budget.
                    self.assertLessEqual(len(model.requests)*12 + 2*len(tracks), 52)
                finally:
                    live.close()

    async def test_finer_silence_checks_do_not_call_asr_and_stop_keeps_the_tail(self):
        import base64
        import tempfile
        from contextlib import ExitStack
        from server.transcription.speech_cpp_live import SpeechCppLiveMeeting
        from tests.test_whisper_live import WhisperDualLaneTests, Model, Socket, RATE
        helper = WhisperDualLaneTests()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config, _, _ = helper.fixture(stack, directory, enabled=False)
            config.asr_backend = 'speech_cpp'
            stack.enter_context(patch('server.transcription.speech_cpp_live.settings', config))
            model = Model()
            stack.enter_context(patch('server.transcription.speech_cpp_live.SpeechCppBatchTranscriber', return_value=model))
            ws = Socket()
            live = SpeechCppLiveMeeting(ws, dict(tracks=['display']))
            try:
                quiet = b'\0\0'*RATE*3
                await live.accept_audio(dict(track='display',seq=0,sampleStart=0,pcm=base64.b64encode(quiet).decode()))
                await live._infer('display', flush=False)
                live.stopping = True
                await live.work()
                self.assertEqual(model.requests, [])
                self.assertEqual(live.records, [])
                self.assertEqual((live.store.chunks_dir/'display.pcm').read_bytes(), quiet)
                self.assertTrue(live.data['finalized'])
            finally:
                live.close()

    async def test_slow_dual_track_inference_coalesces_and_never_queues_parallel_rt(self):
        import asyncio
        import base64
        import tempfile
        import threading
        from contextlib import ExitStack
        from server.transcription.speech_cpp_live import SpeechCppLiveMeeting
        from tests.test_whisper_live import WhisperDualLaneTests, Model, Socket, PCM, RATE
        helper = WhisperDualLaneTests()
        started, release = threading.Event(), threading.Event()

        class SlowModel(Model):
            active = 0
            maximum = 0

            def transcribe_chunk(self, audio, **kwargs):
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                try:
                    return super().transcribe_chunk(audio, **kwargs)
                finally:
                    self.active -= 1

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config, _, _ = helper.fixture(stack, directory, enabled=False)
            config.asr_backend = 'speech_cpp'
            stack.enter_context(patch('server.transcription.speech_cpp_live.settings', config))
            model = SlowModel(started=started, release=release)
            stack.enter_context(patch('server.transcription.speech_cpp_live.SpeechCppBatchTranscriber', return_value=model))
            live = SpeechCppLiveMeeting(Socket(), dict(tracks=['mic','display']))
            worker = asyncio.create_task(live.work())
            try:
                packet = PCM*2 + PCM[:RATE]
                for seq in range(4):
                    for track in ['mic','display']:
                        await live.accept_audio(dict(track=track,seq=seq,sampleStart=seq*RATE*5//2,
                            pcm=base64.b64encode(packet).decode()))
                    if seq == 0:
                        await helper.until(started.is_set)
                live.stopping = True
                release.set()
                await asyncio.wait_for(worker, 10)
                self.assertEqual(model.maximum, 1)
                self.assertLessEqual(len(model.requests), 5, 'pending packets coalesce into bounded windows')
                self.assertEqual(len(live.records), 4)
                self.assertEqual(len({r['segmentId'] for r in live.records}), 4)
                for track in ['mic','display']:
                    self.assertEqual(live.data['tracks'][track]['windowStart'], 10*RATE)
                    self.assertEqual((live.store.chunks_dir/(track+'.pcm')).read_bytes(), PCM*10)
                self.assertTrue(live.data['finalized'])
            finally:
                release.set()
                live.disconnected = True
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                live.close()

    async def test_recording_and_hq_share_gateway_credentials_and_never_open_realtime(self):
        import tempfile
        from contextlib import ExitStack
        from server.transcription.speech_cpp_live import SpeechCppLiveMeeting
        from tests.test_whisper_live import WhisperDualLaneTests, Model, Socket
        helper = WhisperDualLaneTests()
        rt, hq = Model(text='会議の資料を確認します。'), Model(text='会議の資料を確認しました。')
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config, _, _ = helper.fixture(stack, directory, models=(rt,hq))
            config.asr_backend, config.asr_model = 'speech_cpp', 'synthetic-native-alias'
            config.openai_base_url, config.openai_api_key = 'http://gateway.test/v1', 'synthetic-existing-key'
            stack.enter_context(patch('server.transcription.speech_cpp_live.settings', config))
            factory = stack.enter_context(patch('server.transcription.speech_cpp_live.SpeechCppBatchTranscriber', side_effect=[rt,hq]))
            realtime = stack.enter_context(patch('server.qwen_asr.connect', side_effect=AssertionError('unexpected native socket')))
            ws = Socket()
            live = SpeechCppLiveMeeting(ws, dict(tracks=['mic'], language='auto'))
            try:
                await helper.feed(live, 0, 2)
                self.assertTrue(await live._infer('mic', flush=False))
                self.assertTrue(any(e['type']=='partial' and e.get('text') for e in ws.events))
                await helper.feed(live, 2, 5)
                live.stopping = True
                await live.work()
                self.assertTrue(live.data['finalized'])
                self.assertEqual(live.data['asrBackend'], 'speech_cpp')
                self.assertTrue(any(e['type']=='transcript_revision' for e in ws.events))
                self.assertEqual(len(factory.call_args_list),2)
                for call in factory.call_args_list:
                    self.assertEqual(call.kwargs['base_url'],'http://gateway.test/v1')
                    self.assertEqual(call.kwargs['model'],'synthetic-native-alias')
                    self.assertEqual(call.kwargs['api_key'],'synthetic-existing-key')
                self.assertEqual(factory.call_args_list[0].kwargs['timeout'],config.asr_api_timeout_seconds)
                self.assertEqual(factory.call_args_list[1].kwargs['timeout'],config.asr_high_accuracy_timeout_seconds)
                realtime.assert_not_called()
            finally:
                live.close()

    async def test_cancelled_http_recognition_keeps_source_and_does_not_commit_late_result(self):
        import asyncio
        import tempfile
        from contextlib import ExitStack
        from server.transcription.speech_cpp_live import SpeechCppLiveMeeting
        from tests.test_whisper_live import WhisperDualLaneTests, Model, Socket
        helper = WhisperDualLaneTests()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config, _, _ = helper.fixture(stack, directory)
            config.asr_backend = 'speech_cpp'
            stack.enter_context(patch('server.transcription.speech_cpp_live.settings', config))
            stack.enter_context(patch('server.transcription.speech_cpp_live.SpeechCppBatchTranscriber', return_value=Model()))
            live = SpeechCppLiveMeeting(Socket(), dict(tracks=['mic'], language='ja'))
            try:
                await helper.feed(live, 0, 5)
                with patch.object(live.transcriber, 'transcribe_chunk', side_effect=asyncio.CancelledError):
                    with self.assertRaises(asyncio.CancelledError):
                        await live._infer('mic', flush=False)
                self.assertEqual(live.records, [])
                self.assertEqual((live.store.chunks_dir/'mic.pcm').stat().st_size,5*16000*2)
                self.assertEqual(live.data['tracks']['mic']['windowStart'],0)
            finally:
                live.close()

class SpeechCppReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_keyless_native_asr_is_required_and_reports_factory_readiness(self):
        from types import SimpleNamespace
        from fastapi.responses import JSONResponse
        from server.api.routes import health
        for configured in (True, False):
            with patch.object(health, 'settings', SimpleNamespace(openai_api_key='', asr_backend='speech_cpp')), \
                    patch.object(health.runtime, 'resources', SimpleNamespace(transcriber_factory=object() if configured else None)), \
                    patch.object(health, 'schema_revision_status', return_value=SimpleNamespace(
                        ready=True, current_revisions=('synthetic',), expected_revisions=('synthetic',))), \
                    patch.object(health.health_service, 'health', new=AsyncMock(return_value=JSONResponse({}))):
                result = await health._readiness_response()
                payload = json.loads(result.body)
                self.assertTrue(payload['asr']['required'])
                self.assertEqual(payload['asr']['ready'], configured)
                self.assertEqual(result.status_code, 200 if configured else 503)
