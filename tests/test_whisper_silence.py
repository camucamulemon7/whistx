import io
import json
import os
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'synthetic-silence-test-secret-000000000000')

from server.openai_whisper import _should_drop_as_silence
from server.services.meeting_refinement import refine_events
from server.services.meeting_source import MeetingSnapshot
from server.transcription.local_agreement import pcm_wav, speech_bounds
from server.whisper_audio import prepare_whisper_wav, whisper_speech_bounds
from tests.whisper_audio_fixtures import pcm, transcriber


class WhisperSilenceTests(unittest.TestCase):
    def test_silence_noise_and_dc_do_not_call_the_provider_even_without_confidence_metadata(self):
        for signal in [b'\0\0'*16000, pcm(2, amplitude=80, noise=True),
                       b'\xc8\0'*32000, pcm(2, amplitude=240, noise=True), b'\0\0'*480000]:
            with self.subTest(bytes=len(signal)):
                adapter, calls = transcriber('はい')
                result = adapter.transcribe_chunk(pcm_wav(signal), mime_type='audio/wav', language='ja', prompt=None, temperature=0)
                self.assertEqual(result.text, '')
                self.assertTrue(result.silence_detected)
                self.assertEqual(calls, [])
        # The legacy primitive remains available; both providers use the new gate.
        self.assertIsNotNone(speech_bounds(b'\xc8\0'*32000))

    def test_quiet_short_ack_and_closing_words_are_not_a_text_blacklist(self):
        for text in ['はい', 'ご清聴ありがとうございました', 'お疲れ様です', 'ありがとうございました']:
            adapter, calls = transcriber(text)
            result = adapter.transcribe_chunk(pcm_wav(pcm(.12)), mime_type='audio/wav', language='ja', prompt=None, temperature=0)
            self.assertEqual(result.text, text)
            self.assertFalse(result.silence_detected)
            self.assertEqual(len(calls), 1)

    def test_speech_onset_tail_and_original_timestamp_offset_survive_trimming(self):
        source = b'\0\0'*32000 + pcm(.12) + b'\0\0'*64000
        adapter, calls = transcriber('はい', segments=[dict(start=.3, end=.42, no_speech_prob=.01, avg_logprob=-.1)])
        result = adapter.transcribe_chunk(pcm_wav(source), mime_type='audio/wav', language='ja', prompt=None, temperature=0)
        self.assertEqual(result.text, 'はい')
        self.assertEqual((result.start_ms, result.end_ms), (2000, 2120))
        with wave.open(io.BytesIO(calls[0])) as wav:
            retained = wav.readframes(wav.getnframes())
        self.assertEqual(retained, b'\0\0'*4800 + pcm(.12) + b'\0\0'*4800)
        self.assertEqual(source, b'\0\0'*32000 + pcm(.12) + b'\0\0'*64000)

    def test_speech_in_quiet_noise_and_uncertain_nonstationary_audio_are_preserved(self):
        noise = pcm(1, amplitude=240, noise=True)
        voice = pcm(.12)
        self.assertIsNotNone(whisper_speech_bounds(noise+voice+noise))
        # A quiet changing noise/whisper envelope is uncertain; keep it rather
        # than applying the stationary-noise heuristic to all noisy speech.
        varying = b''.join(pcm(.1, amplitude=level, noise=True) for level in [180, 360]*10)
        self.assertIsNotNone(whisper_speech_bounds(varying))
        self.assertIsNotNone(whisper_speech_bounds(pcm(.12, amplitude=240, noise=True)))

    def test_quiet_trailing_phonemes_keep_300ms_context(self):
        strong, quiet_tail = pcm(.12), pcm(.2, amplitude=60)
        audio, offset, silent = prepare_whisper_wav(pcm_wav(strong+quiet_tail+b'\0\0'*32000), 'audio/wav')
        self.assertFalse(silent)
        self.assertEqual(offset, 0)
        with wave.open(io.BytesIO(audio)) as wav:
            retained = wav.readframes(wav.getnframes())
        self.assertTrue(retained.startswith(strong+quiet_tail))

    def test_unsupported_audio_is_left_to_provider_instead_of_guessed_silent(self):
        for data, mime in [(b'unknown', 'audio/webm'), (b'broken wav', 'audio/wav')]:
            self.assertEqual(prepare_whisper_wav(data, mime), (data, 0, False))
        out = io.BytesIO()
        with wave.open(out, 'wb') as wav:
            wav.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\0'*48000)
        data = out.getvalue()
        self.assertEqual(prepare_whisper_wav(data, 'audio/wav'), (data, 0, False))

    def test_response_confidence_does_not_erase_confident_or_mixed_speech(self):
        silent = dict(no_speech_prob=.95, avg_logprob=-1.2)
        confident = dict(no_speech_prob=.95, avg_logprob=-.1)
        for text in ['はい', 'ご清聴ありがとうございました', 'お疲れ様です']:
            for segments in [None, [confident], [silent, confident], [dict(no_speech_prob=.99)],
                             [dict(no_speech_prob=float('nan'), avg_logprob=-1.2)],
                             [dict(no_speech_prob=.99, avg_logprob=float('nan'))],
                             [dict(no_speech_prob=1.2, avg_logprob=-1.2)]]:
                self.assertFalse(_should_drop_as_silence(dict(segments=segments), text))
            self.assertTrue(_should_drop_as_silence(dict(segments=[silent]), text))

    def test_low_confidence_no_speech_response_is_not_retried_for_more_words(self):
        adapter, calls = transcriber('はい', segments=[dict(no_speech_prob=.91, avg_logprob=-1.2, compression_ratio=2.6)])
        adapter.multi_pass_enabled = True
        result = adapter.transcribe_chunk(pcm_wav(pcm(1)), mime_type='audio/wav', language='ja', prompt=None, temperature=0)
        self.assertEqual(result.text, '')
        self.assertTrue(result.silence_detected)
        self.assertEqual(len(calls), 1)

    def test_manual_refinement_removes_only_acoustically_silent_text_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'synthetic.jsonl'
            rows = [dict(type='final', seq=i, segmentId=str(i), text='古い誤認識' if i == 0 else 'はい',
                         language='ja', tsStart=i*1000, tsEnd=(i+1)*1000, rawAudioPath=f'/audio/{i}.wav') for i in range(2)]
            original = ''.join(json.dumps(r)+'\n' for r in rows)
            path.write_text(original)
            path.with_suffix('.meta.json').write_text('{"finalized":true}')
            audio = [root/'0.wav', root/'1.wav']
            audio[0].write_bytes(pcm_wav(pcm(1, amplitude=240, noise=True)))
            audio[1].write_bytes(pcm_wav(pcm(.12)))
            snapshot = MeetingSnapshot('runtime:synthetic', path, root/'meeting.json', [], [], 'r1', 2000, True)
            adapter, calls = transcriber('はい')
            with patch('server.services.meeting_refinement.resolve_debug_audio_path', side_effect=audio):
                events = list(refine_events(snapshot, cancelled=threading.Event(), transcriber_factory=lambda **kw: adapter))
            self.assertEqual(events[-1]['type'], 'done')
            self.assertEqual(len(calls), 1)
            updated = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r['text'] for r in updated], ['', 'はい'])
            self.assertTrue(updated[0]['silenceDetected'])
            self.assertEqual(updated[0]['originalText'], '古い誤認識')
            self.assertEqual(path.with_suffix('.original.jsonl').read_text(), original)


class QwenSilenceTests(unittest.TestCase):
    def test_manual_adapter_gates_silence_noise_and_preserves_quiet_ack(self):
        from server.qwen_asr import QwenBatchTranscriber
        from types import SimpleNamespace
        for signal, silent in [(b'\0\0'*16000, True), (pcm(2, amplitude=240, noise=True), True),
                               (b'\xc8\0'*16000, True), (pcm(.12), False)]:
            adapter = QwenBatchTranscriber(api_key='synthetic', base_url='http://synthetic.invalid/v1', model='qwen')
            adapter.client.close()
            calls = []
            def post(*args, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'text': 'はい'})
            adapter.client = SimpleNamespace(post=post, close=lambda: None)
            result = adapter.transcribe_chunk(pcm_wav(signal), mime_type='audio/wav', language='ja', prompt=None, temperature=0)
            self.assertEqual(result.silence_detected, silent)
            self.assertEqual(result.text, '' if silent else 'はい')
            self.assertEqual(len(calls), 0 if silent else 1)
