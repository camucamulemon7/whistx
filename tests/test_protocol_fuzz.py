"""Reproducible malformed WebSocket payload corpus; no random flaky retries."""
import base64
import os
import random
import unittest

os.environ.setdefault('APP_SESSION_SECRET', 'protocol-fuzz-test-secret-abcdefghijklmnopqrstuvwxyz')
from server.transcription.messages import parse_chunk_message


class ProtocolFuzzTests(unittest.TestCase):
    def test_malformed_fields_are_bounded_and_do_not_crash_parser(self):
        rng = random.Random(77)
        values = [None, {}, [], True, False, '', 'nan', 'inf', float('inf'), float('nan'), -1, 10**100]
        for _ in range(500):
            payload = {key: rng.choice(values) for key in ['seq', 'offsetMs', 'durationMs', 'speechRatio', 'activeMs', 'silenceMs']}
            payload['audio'] = base64.b64encode(b'test audio').decode()
            parsed = parse_chunk_message(payload, max_audio_bytes=32, max_screenshot_bytes=32)
            self.assertIsNotNone(parsed)
            self.assertLessEqual(len(parsed.audio_bytes), 32)
            self.assertGreaterEqual(parsed.speech_ratio, 0)
            self.assertLessEqual(parsed.speech_ratio, 1)
        for payload in [{'audio': 'a'*10000}, {'audio': '!!!'}, {'audio': 'YQ==', 'screenshot': 'a'*10000}]:
            self.assertIsNone(parse_chunk_message(payload, max_audio_bytes=32, max_screenshot_bytes=32))
