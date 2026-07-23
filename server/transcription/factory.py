from __future__ import annotations

import asyncio
import secrets
from datetime import datetime, timezone
from typing import Any, Callable

from ..asr import SessionTranscriber
from ..transcript_store import TranscriptStore
from .messages import as_bool, as_float, as_int, as_str, normalize_asr_language, normalize_audio_source
from .session import LiveSession


def create_live_session(
    payload: dict[str, Any],
    *,
    settings: Any,
    transcriber_factory: Callable[[], SessionTranscriber],
    diarizer_available: bool,
) -> LiveSession:
    base_session_id = TranscriptStore.sanitize_or_generate(as_str(payload.get("sessionId")))
    runtime_session_id = TranscriptStore.make_runtime_session_id(base_session_id)
    access_token = secrets.token_urlsafe(18)

    language = normalize_asr_language(as_str(payload.get("language")))
    audio_source = normalize_audio_source(as_str(payload.get("audioSource")))
    prompt = as_str(payload.get("prompt")) or settings.default_prompt
    shared_vocabulary = as_str(payload.get("sharedVocabulary"))
    temperature = as_float(payload.get("temperature"), settings.default_temperature)
    diarization_requested = as_bool(payload.get("diarizationEnabled"), True)
    speaker_counts = _parse_diarization_speaker_params(payload, settings=settings)

    session = LiveSession(
        session_id=runtime_session_id,
        access_token=access_token,
        language=language,
        audio_source=audio_source,
        base_prompt=prompt,
        shared_vocabulary=shared_vocabulary,
        temperature=temperature,
        context_prompt_enabled=settings.context_prompt_enabled,
        context_max_chars=settings.context_max_chars,
        context_recent_lines=settings.context_recent_lines,
        context_term_limit=settings.context_term_limit,
        context_history=[],
        transcript_history=[],
        context_terms=[],
        last_emitted_text="",
        last_emitted_ts_end=0,
        collect_audio_for_diarization=diarizer_available and diarization_requested,
        diarization_num_speakers=speaker_counts[0],
        diarization_min_speakers=speaker_counts[1],
        diarization_max_speakers=speaker_counts[2],
        audio_chunks=[],
        store=TranscriptStore(settings.transcripts_dir, runtime_session_id),
        queue=asyncio.Queue(maxsize=settings.max_queue_size),
        transcriber=transcriber_factory(),
        overlap_tail_pcm=b"",
        last_chunk_seq=-1,
        last_chunk_offset_ms=-1,
        failed_prepared_chunks=[],
        asr_input_tokens=0,
        asr_output_tokens=0,
        asr_total_tokens=0,
        asr_estimated_tokens=0,
    )
    session.store.write_metadata(
        {
            "sessionId": runtime_session_id,
            "accessToken": access_token,
            "language": language,
            "audioSource": audio_source,
            "diarizationEnabled": session.collect_audio_for_diarization,
            "diarizationNumSpeakers": speaker_counts[0],
            "diarizationMinSpeakers": speaker_counts[1],
            "diarizationMaxSpeakers": speaker_counts[2],
            "finalized": False,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
    )
    return session


def _clamp_speaker_count(value: int, maximum: int = 12) -> int:
    return max(0, min(maximum, value))


def _parse_diarization_speaker_params(payload: dict[str, Any], *, settings: Any) -> tuple[int, int, int]:
    num = _clamp_speaker_count(as_int(payload.get("diarizationNumSpeakers"), settings.diarization_num_speakers))
    minimum = _clamp_speaker_count(
        as_int(payload.get("diarizationMinSpeakers"), settings.diarization_min_speakers)
    )
    maximum = _clamp_speaker_count(
        as_int(payload.get("diarizationMaxSpeakers"), settings.diarization_max_speakers)
    )
    if num > 0:
        return (num, 0, 0)
    if minimum > 0 and maximum > 0 and minimum > maximum:
        minimum, maximum = maximum, minimum
    return (0, minimum, maximum)
