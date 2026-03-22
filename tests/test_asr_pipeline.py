from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if "argon2" not in sys.modules:
    argon2_mod = types.ModuleType("argon2")
    argon2_exc = types.ModuleType("argon2.exceptions")

    class VerifyMismatchError(Exception):
        pass

    class PasswordHasher:
        def hash(self, password: str) -> str:
            return f"hashed:{password}"

        def verify(self, password_hash: str, password: str) -> bool:
            return password_hash == f"hashed:{password}"

    argon2_mod.PasswordHasher = PasswordHasher
    argon2_exc.VerifyMismatchError = VerifyMismatchError
    sys.modules["argon2"] = argon2_mod
    sys.modules["argon2.exceptions"] = argon2_exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.audio_pipeline import AudioPreprocessor, PreparedAudio
from server import legacy_app


class AudioPipelineTests(unittest.TestCase):
    def test_compute_audio_metrics_detects_silence(self) -> None:
        preprocessor = object.__new__(AudioPreprocessor)
        metrics = preprocessor._compute_audio_metrics(b"\x00\x00" * 32_000)
        self.assertEqual(metrics["rms"], 0.0)
        self.assertEqual(metrics["peak"], 0.0)
        self.assertEqual(metrics["speech_ratio"], 0.0)

    def test_compute_audio_metrics_detects_speech(self) -> None:
        preprocessor = object.__new__(AudioPreprocessor)
        loud_sample = int(0.15 * 32767).to_bytes(2, byteorder="little", signed=True)
        metrics = preprocessor._compute_audio_metrics(loud_sample * 16_000)
        self.assertGreater(metrics["rms"], 0.1)
        self.assertGreater(metrics["peak"], 0.1)
        self.assertGreater(metrics["speech_ratio"], 0.95)

    def test_session_worker_skips_low_speech_chunk_before_asr(self) -> None:
        prepared = PreparedAudio(
            audio_bytes=b"wav",
            mime_type="audio/wav",
            overlap_ms_used=0,
            tail_pcm=b"",
            rms=0.001,
            peak=0.002,
            speech_ratio=0.0,
            audio_metrics={"rms": 0.001, "peak": 0.002, "speech_ratio": 0.0},
        )
        messages: list[dict[str, object]] = []

        async def fake_safe_send(_ws: object, payload: dict[str, object]) -> bool:
            messages.append(payload)
            return True

        class DummyTranscriber:
            def __init__(self) -> None:
                self.called = False

            def transcribe_chunk(self, *_args: object, **_kwargs: object) -> object:
                self.called = True
                raise AssertionError("transcribe_chunk should not be called")

            def close(self) -> None:
                return None

        session = SimpleNamespace(
            session_id="sess-1",
            language="ja",
            audio_source="mic",
            collect_audio_for_diarization=False,
            queue=asyncio.Queue(),
            transcriber=DummyTranscriber(),
            asr_input_tokens=0,
            asr_output_tokens=0,
            asr_total_tokens=0,
            asr_estimated_tokens=0,
            transcript_history=[],
            last_emitted_text="",
            last_emitted_ts_end=0,
            store=SimpleNamespace(append_final=lambda _record: None),
            base_prompt="",
            context_prompt_enabled=False,
            context_history=[],
            context_terms=[],
            context_max_chars=0,
            context_recent_lines=0,
            context_term_limit=0,
            temperature=0.0,
        )

        async def run_test() -> None:
            await session.queue.put(SimpleNamespace(seq=1, offset_ms=0, duration_ms=1_000, mime_type="audio/webm", audio_bytes=b"a"))
            await session.queue.put(None)
            with patch.object(legacy_app, "_prepare_audio_for_asr", return_value=prepared):
                with patch.object(legacy_app, "_safe_send", side_effect=fake_safe_send):
                    with patch.object(
                        legacy_app,
                        "settings",
                        SimpleNamespace(
                            asr_model="whisper-1",
                            asr_vad_drop_enabled=True,
                            asr_vad_speech_ratio_min=0.02,
                        ),
                    ):
                        await legacy_app._session_worker(object(), session)

        asyncio.run(run_test())
        self.assertFalse(session.transcriber.called)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["type"], "ack")
        self.assertEqual(messages[0]["reason"], "low_speech_ratio")


if __name__ == "__main__":
    unittest.main()
