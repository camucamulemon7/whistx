from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('APP_SESSION_SECRET', 'test-session-secret-abcdefghijklmnopqrstuvwxyz12')

import httpx
from openai import APIConnectionError, BadRequestError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import openai_whisper
from server import summarizer as summarizer_module
from server.asr import ASRChunkResult
from server.core.config.asr import load_asr_config
from server.core import rate_limit
from server.transcription import text_processing








class RegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        rate_limit.clear()

    def test_asr_retry_config_is_loaded_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                'ASR_RETRY_MAX_ATTEMPTS': '5',
                'ASR_RETRY_BASE_DELAY_MS': '250',
                'ASR_RESCUE_RETRY_ENABLED': '1',
                'ASR_RESCUE_RETRY_TEMPERATURE': '0.35',
                'ASR_API_TIMEOUT_SECONDS': '42',
                'SUMMARY_API_TIMEOUT_SECONDS': '43',
                'PROOFREAD_API_TIMEOUT_SECONDS': '44',
                'FFMPEG_TIMEOUT_SECONDS': '45',
            },
            clear=False,
        ):
            config = load_asr_config()

        self.assertEqual(config.asr_retry_max_attempts, 5)
        self.assertEqual(config.asr_retry_base_delay_ms, 250)
        self.assertTrue(config.asr_rescue_retry_enabled)
        self.assertEqual(config.asr_rescue_retry_temperature, 0.35)
        self.assertEqual(config.asr_api_timeout_seconds, 42.0)
        self.assertEqual(config.summary_api_timeout_seconds, 43.0)
        self.assertEqual(config.proofread_api_timeout_seconds, 44.0)
        self.assertEqual(config.ffmpeg_timeout_seconds, 45.0)


    def test_asr_context_defaults_are_expanded(self) -> None:
        config = load_asr_config()
        self.assertEqual(config.context_recent_lines, 4)
        self.assertEqual(config.context_max_chars, 2200)
        self.assertEqual(config.context_term_limit, 80)


    def test_proofread_prompt_includes_shared_glossary(self) -> None:
        prompt = summarizer_module._build_proofread_prompt(
            'なんど の話です',
            'ja',
            mode='proofread',
            glossary_text='なんど=NAND\nらすこ=Lascaux',
        )

        self.assertIn('優先用語辞典', prompt)
        self.assertIn('なんど=NAND', prompt)
        self.assertIn('らすこ=Lascaux', prompt)


    def test_trim_overlap_prefix_requires_substantial_match(self) -> None:
        previous = '本日の会議では新製品の価格改定について説明します'
        current = '価格改定について説明します。次に販売計画を確認します'
        self.assertEqual(text_processing._trim_overlap_prefix(current, previous), '次に販売計画を確認します')


    def test_trim_overlap_prefix_fuzzy_match_handles_small_variation(self) -> None:
        previous = '新製品の価格改定について説明いたします'
        current = '価格改定について説明します。次に販売計画です'
        self.assertEqual(text_processing._trim_overlap_prefix(current, previous), '次に販売計画です')


    def test_build_prompt_includes_shared_vocabulary(self) -> None:
        session = SimpleNamespace(
            base_prompt='会議用語を優先してください',
            shared_vocabulary='PCIe, UCIe, Blackwell',
            context_prompt_enabled=True,
            context_history=['次回は PCIe 帯域を確認します'],
            context_terms=['帯域', 'Gen6'],
            context_max_chars=400,
            language='ja',
        )
        prompt = text_processing._build_prompt(session)
        self.assertIsNotNone(prompt)
        self.assertIn('共有用語辞典', prompt)
        self.assertIn('PCIe, UCIe, Blackwell', prompt)
        self.assertIn('利用者プロンプト', prompt)


    def test_openai_whisper_retries_retryable_errors(self) -> None:
        request = httpx.Request('POST', 'https://example.com/v1/audio/transcriptions')
        response = SimpleNamespace(
            text='hello world',
            segments=[{'start': 0.0, 'end': 1.2, 'no_speech_prob': 0.01}],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )
        create_calls = []

        def create(**kwargs):
            create_calls.append(kwargs)
            if len(create_calls) <= 2:
                raise APIConnectionError(message='temporary', request=request)
            return response

        transcriber = openai_whisper.OpenAIWhisperTranscriber(
            api_key='test-key',
            base_url=None,
            model='whisper-1',
            observer=None,
        )
        transcriber.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))

        transcriber.retry_max_attempts = 3
        transcriber.retry_base_delay_ms = 50
        with patch.object(openai_whisper.time, 'sleep', autospec=True) as sleep_mock:
            result = transcriber.transcribe_chunk(
                b'abc',
                mime_type='audio/webm',
                language='ja',
                prompt=None,
                temperature=0.0,
            )

        self.assertEqual(result.text, 'hello world')
        self.assertEqual(len(create_calls), 3)
        self.assertEqual(sleep_mock.call_count, 2)
        self.assertEqual(sleep_mock.call_args_list[0].args[0], 0.05)
        self.assertEqual(sleep_mock.call_args_list[1].args[0], 0.1)


    def test_openai_whisper_extracts_confidence_metrics(self) -> None:
        response = SimpleNamespace(
            text='短い定型文です',
            segments=[
                {
                    'start': 0.0,
                    'end': 1.0,
                    'no_speech_prob': 0.91,
                    'avg_logprob': -1.2,
                    'compression_ratio': 2.6,
                }
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )

        transcriber = openai_whisper.OpenAIWhisperTranscriber(
            api_key='test-key',
            base_url=None,
            model='whisper-1',
            observer=None,
        )
        transcriber.client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=lambda **_kwargs: response))
        )

        result = transcriber.transcribe_chunk(
            b'abc',
            mime_type='audio/webm',
            language='ja',
            prompt=None,
            temperature=0.0,
        )

        self.assertTrue(result.suspicious)
        self.assertAlmostEqual(result.max_no_speech_prob or 0.0, 0.91)
        self.assertAlmostEqual(result.avg_logprob or 0.0, -1.2)
        self.assertAlmostEqual(result.compression_ratio or 0.0, 2.6)


    def test_openai_whisper_multi_pass_prefers_longer_retry_result(self) -> None:
        first_response = SimpleNamespace(
            text='短い候補',
            segments=[{'start': 0.0, 'end': 0.6, 'no_speech_prob': 0.65, 'avg_logprob': -1.1, 'compression_ratio': 2.5}],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )
        second_response = SimpleNamespace(
            text='短い候補ではなく十分に長い改善結果です',
            segments=[{'start': 0.0, 'end': 1.2, 'no_speech_prob': 0.2, 'avg_logprob': -0.2, 'compression_ratio': 1.2}],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0, total_tokens=1),
        )
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return first_response if len(calls) == 1 else second_response

        transcriber = openai_whisper.OpenAIWhisperTranscriber(
            api_key='test-key',
            base_url=None,
            model='whisper-1',
            observer=None,
        )
        transcriber.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
        transcriber.multi_pass_enabled = True

        result = transcriber.transcribe_chunk(
            b'abc',
            mime_type='audio/webm',
            language='ja',
            prompt=None,
            temperature=0.0,
        )

        self.assertEqual(len(calls), 2)
        self.assertIn('改善結果', result.text)


    def test_openai_whisper_does_not_retry_non_retryable_errors(self) -> None:
        request = httpx.Request('POST', 'https://example.com/v1/audio/transcriptions')
        response = httpx.Response(400, request=request, content=b'{}')

        def create(**kwargs):
            raise BadRequestError(message='bad request', response=response, body=None)

        transcriber = openai_whisper.OpenAIWhisperTranscriber(
            api_key='test-key',
            base_url=None,
            model='whisper-1',
            observer=None,
        )
        transcriber.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))

        transcriber.retry_max_attempts = 3
        transcriber.retry_base_delay_ms = 50
        with patch.object(openai_whisper.time, 'sleep', autospec=True) as sleep_mock:
            with self.assertRaises(BadRequestError):
                transcriber.transcribe_chunk(
                    b'abc',
                    mime_type='audio/webm',
                    language='ja',
                    prompt=None,
                    temperature=0.0,
                )

        self.assertEqual(sleep_mock.call_count, 0)


    def test_light_proofread_collapses_fillers_and_normalizes_digits(self) -> None:
        value = text_processing._light_proofread('えーと、えーと ２０ ２５ 年の計画です', language='ja')
        self.assertIn('えーと', value)
        self.assertNotIn('えーと、えーと', value)
        self.assertIn('2025', value)


    def test_boundary_fragment_detection_drops_broken_display_chunk(self) -> None:
        self.assertTrue(
            text_processing._should_drop_boundary_fragment(
                'おすすめとかえええ\ufffd',
                '有識者のみなさんぜひ教えてくださいよということでお願いしますよお願いしますほなじゃあなんかありますかおすすめとか',
                source_mode='display',
                suspicious=False,
            )
        )


    def test_weird_transcription_retry_detection_handles_broken_chunk(self) -> None:
        self.assertTrue(
            text_processing._should_retry_weird_transcription(
                'おすすめとかえええ\ufffd',
                '有識者のみなさんぜひ教えてくださいよということでお願いしますよお願いしますほなじゃあなんかありますかおすすめとか',
                source_mode='display',
                suspicious=False,
            )
        )


    def test_rescue_transcription_result_prefers_cleaner_retry(self) -> None:
        original = ASRChunkResult(text='おすすめとかえええ\ufffd', start_ms=0, end_ms=1000, suspicious=True)
        retry = ASRChunkResult(text='おすすめとか', start_ms=0, end_ms=1000, suspicious=False)
        self.assertTrue(
            text_processing._prefer_rescue_transcription_result(
                original=original,
                retry=retry,
                previous_text='有識者のみなさんぜひ教えてくださいよということでお願いしますよお願いしますほなじゃあなんかありますか',
                source_mode='display',
            )
        )


    def test_monotonic_bounds_prevent_timestamp_overlap(self) -> None:
        self.assertEqual(
            text_processing._coerce_monotonic_bounds(
                ts_start=8100,
                ts_end=8900,
                previous_end_ms=9000,
            ),
            (9000, 9000),
        )
        self.assertEqual(
            text_processing._coerce_monotonic_bounds(
                ts_start=9100,
                ts_end=9500,
                previous_end_ms=9000,
            ),
            (9100, 9500),
        )

