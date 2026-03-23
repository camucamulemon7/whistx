from __future__ import annotations

import asyncio
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("APP_SESSION_SECRET", "test-session-secret-abcdefghijklmnopqrstuvwxyz12")

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
from server.asr import ASRChunkResult


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

    def test_parse_chunk_message_reads_vad_metrics(self) -> None:
        import base64

        payload = {
            "audio": base64.b64encode(b"audio").decode("ascii"),
            "seq": 3,
            "offsetMs": 1200,
            "durationMs": 2400,
            "speechRatio": 0.35,
            "activeMs": 840,
            "silenceMs": 220,
        }
        chunk = legacy_app._parse_chunk_message(payload)
        assert chunk is not None
        self.assertEqual(chunk.seq, 3)
        self.assertAlmostEqual(chunk.speech_ratio, 0.35)
        self.assertEqual(chunk.active_ms, 840)
        self.assertEqual(chunk.silence_ms, 220)

    def test_overlap_resolution_uses_vad_metadata(self) -> None:
        preprocessor = object.__new__(AudioPreprocessor)
        preprocessor.overlap_ms = 3500
        result_busy = preprocessor._resolve_overlap_ms(
            30000,
            speech_ratio=0.7,
            active_ms=22000,
            silence_ms=100,
            source_mode="both",
        )
        result_silent = preprocessor._resolve_overlap_ms(
            30000,
            speech_ratio=0.05,
            active_ms=400,
            silence_ms=1800,
            source_mode="mic",
        )
        self.assertGreaterEqual(result_busy, result_silent)
        self.assertGreaterEqual(result_busy, 3500)

    def test_session_worker_retries_failed_chunk_with_next_chunk(self) -> None:
        prepared_first = PreparedAudio(
            audio_bytes=legacy_app._merge_wav_chunks([_make_test_wav(16000)]),
            mime_type="audio/wav",
            overlap_ms_used=0,
            tail_pcm=b"",
            rms=0.05,
            peak=0.2,
            speech_ratio=0.5,
            audio_metrics={"rms": 0.05, "peak": 0.2, "speech_ratio": 0.5},
        )
        prepared_second = PreparedAudio(
            audio_bytes=legacy_app._merge_wav_chunks([_make_test_wav(8000)]),
            mime_type="audio/wav",
            overlap_ms_used=0,
            tail_pcm=b"",
            rms=0.05,
            peak=0.2,
            speech_ratio=0.5,
            audio_metrics={"rms": 0.05, "peak": 0.2, "speech_ratio": 0.5},
        )
        messages: list[dict[str, object]] = []
        appended: list[object] = []
        prepare_calls = [prepared_first, prepared_second]

        async def fake_safe_send(_ws: object, payload: dict[str, object]) -> bool:
            messages.append(payload)
            return True

        class FlakyTranscriber:
            def __init__(self) -> None:
                self.calls = 0
                self.byte_sizes: list[int] = []

            def transcribe_chunk(self, audio_bytes: bytes, **_kwargs: object) -> ASRChunkResult:
                self.calls += 1
                self.byte_sizes.append(len(audio_bytes))
                if self.calls == 1:
                    raise RuntimeError("temporary failure")
                return ASRChunkResult(text="recovered text", start_ms=0, end_ms=1200)

            def close(self) -> None:
                return None

        session = SimpleNamespace(
            session_id="sess-2",
            language="ja",
            audio_source="mic",
            collect_audio_for_diarization=False,
            queue=asyncio.Queue(),
            transcriber=FlakyTranscriber(),
            asr_input_tokens=0,
            asr_output_tokens=0,
            asr_total_tokens=0,
            asr_estimated_tokens=0,
            transcript_history=[],
            last_emitted_text="",
            last_emitted_ts_end=0,
            store=SimpleNamespace(append_final=lambda record: appended.append(record)),
            base_prompt="",
            context_prompt_enabled=False,
            context_history=[],
            context_terms=[],
            context_max_chars=0,
            context_recent_lines=0,
            context_term_limit=0,
            temperature=0.0,
            failed_prepared_chunks=[],
        )

        first = SimpleNamespace(
            seq=1,
            offset_ms=0,
            duration_ms=1000,
            mime_type="audio/webm",
            audio_bytes=b"a",
            speech_ratio=0.5,
            active_ms=600,
            silence_ms=100,
            screenshot_bytes=None,
            screenshot_mime_type=None,
        )
        second = SimpleNamespace(
            seq=2,
            offset_ms=1000,
            duration_ms=1000,
            mime_type="audio/webm",
            audio_bytes=b"b",
            speech_ratio=0.5,
            active_ms=650,
            silence_ms=80,
            screenshot_bytes=None,
            screenshot_mime_type=None,
        )

        async def run_test() -> None:
            await session.queue.put(first)
            await session.queue.put(second)
            await session.queue.put(None)
            with patch.object(legacy_app, "_prepare_audio_for_asr", side_effect=prepare_calls):
                with patch.object(legacy_app, "_safe_send", side_effect=fake_safe_send):
                    with patch.object(
                        legacy_app,
                        "settings",
                        SimpleNamespace(
                            asr_model="whisper-1",
                            asr_vad_drop_enabled=False,
                            asr_vad_speech_ratio_min=0.02,
                            asr_preprocess_sample_rate=16000,
                        ),
                    ):
                        await legacy_app._session_worker(object(), session)

        asyncio.run(run_test())
        self.assertEqual(session.transcriber.calls, 2)
        self.assertEqual(len(appended), 1)
        self.assertEqual(appended[0].seq, 1)
        self.assertEqual(appended[0].chunkDurationMs, 2000)
        self.assertGreater(session.transcriber.byte_sizes[1], session.transcriber.byte_sizes[0])
        self.assertEqual(session.failed_prepared_chunks, [])


def _make_test_wav(frame_count: int) -> bytes:
    import wave
    from io import BytesIO

    sample = int(0.1 * 32767).to_bytes(2, byteorder="little", signed=True)
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(16000)
            wav_out.writeframes(sample * frame_count)
        return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
