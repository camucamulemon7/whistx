from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from server.transcription.factory import create_live_session
from server.transcription.messages import (
    as_bool,
    normalize_asr_language,
    normalize_audio_source,
    parse_chunk_message,
    validate_chunk_order,
)


class _Transcriber:
    def close(self) -> None:
        return None


class TranscriptionMessageTests(unittest.TestCase):
    def test_parse_chunk_message_decodes_and_clamps_client_metrics(self) -> None:
        chunk = parse_chunk_message(
            {
                "audio": base64.b64encode(b"audio").decode(),
                "screenshot": "not-base64",
                "seq": "3",
                "offsetMs": -10,
                "durationMs": 50,
                "speechRatio": 1.8,
                "activeMs": -1,
                "silenceMs": "120",
            }
        )
        self.assertIsNotNone(chunk)
        assert chunk is not None
        self.assertEqual(chunk.audio_bytes, b"audio")
        self.assertEqual(chunk.seq, 3)
        self.assertEqual(chunk.offset_ms, 0)
        self.assertEqual(chunk.duration_ms, 200)
        self.assertEqual(chunk.speech_ratio, 1.0)
        self.assertEqual(chunk.active_ms, 0)
        self.assertEqual(chunk.silence_ms, 120)
        self.assertIsNone(chunk.screenshot_bytes)

    def test_parse_chunk_message_rejects_missing_or_invalid_audio(self) -> None:
        self.assertIsNone(parse_chunk_message({}))
        self.assertIsNone(parse_chunk_message({"audio": "*invalid*"}))

    def test_chunk_order_reports_sequence_and_offset_errors(self) -> None:
        state = SimpleNamespace(last_chunk_seq=4, last_chunk_offset_ms=1000)
        duplicate = parse_chunk_message({"audio": "YQ==", "seq": 4, "offsetMs": 1200})
        regressed = parse_chunk_message({"audio": "YQ==", "seq": 5, "offsetMs": 900})
        accepted = parse_chunk_message({"audio": "YQ==", "seq": 5, "offsetMs": 1000})
        assert duplicate is not None and regressed is not None and accepted is not None
        self.assertEqual(validate_chunk_order(state, duplicate)["detail"], "seq_must_strictly_increase")
        self.assertEqual(validate_chunk_order(state, regressed)["detail"], "offset_ms_must_be_monotonic")
        self.assertIsNone(validate_chunk_order(state, accepted))

    def test_protocol_normalizers_have_safe_defaults(self) -> None:
        self.assertTrue(as_bool("yes", False))
        self.assertFalse(as_bool("off", True))
        self.assertIsNone(normalize_asr_language("auto"))
        self.assertEqual(normalize_asr_language("JA"), "ja")
        self.assertEqual(normalize_audio_source("unknown"), "mic")


class SessionFactoryTests(unittest.TestCase):
    def test_factory_initializes_queue_and_persists_runtime_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(
                default_prompt="default",
                default_temperature=0.1,
                context_prompt_enabled=True,
                context_max_chars=1000,
                context_recent_lines=4,
                context_term_limit=20,
                transcripts_dir=Path(temp_dir),
                max_queue_size=3,
                diarization_num_speakers=0,
                diarization_min_speakers=2,
                diarization_max_speakers=4,
            )
            session = create_live_session(
                {
                    "sessionId": "client",
                    "audioSource": "both",
                    "language": "ja",
                    "diarizationEnabled": True,
                    "diarizationMinSpeakers": 5,
                    "diarizationMaxSpeakers": 2,
                },
                settings=settings,
                transcriber_factory=_Transcriber,
                diarizer_available=True,
            )

            self.assertTrue(session.session_id.startswith("client_"))
            self.assertEqual(session.audio_source, "both")
            self.assertEqual(session.queue.maxsize, 3)
            self.assertEqual((session.diarization_min_speakers, session.diarization_max_speakers), (2, 5))
            metadata = session.store.read_metadata()
            self.assertEqual(metadata["sessionId"], session.session_id)
            self.assertTrue(metadata["diarizationEnabled"])
            self.assertFalse(metadata["finalized"])


if __name__ == "__main__":
    unittest.main()
