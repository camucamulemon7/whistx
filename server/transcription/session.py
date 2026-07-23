from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..asr import SessionTranscriber
from ..diarizer import AudioChunk
from ..transcript_store import TranscriptStore


@dataclass(slots=True)
class ChunkMessage:
    seq: int
    offset_ms: int
    duration_ms: int
    mime_type: str
    audio_bytes: bytes
    speech_ratio: float = 1.0
    active_ms: int = 0
    silence_ms: int = 0
    screenshot_mime_type: str | None = None
    screenshot_bytes: bytes | None = None


@dataclass(slots=True)
class FailedPreparedChunk:
    seq: int
    offset_ms: int
    duration_ms: int
    audio_bytes: bytes
    speech_ratio: float
    active_ms: int
    silence_ms: int


@dataclass(slots=True)
class LiveSession:
    session_id: str
    access_token: str
    language: str | None
    audio_source: str
    base_prompt: str
    shared_vocabulary: str
    temperature: float
    context_prompt_enabled: bool
    context_max_chars: int
    context_recent_lines: int
    context_term_limit: int
    context_history: list[str]
    transcript_history: list[str]
    context_terms: list[str]
    last_emitted_text: str
    last_emitted_ts_end: int
    collect_audio_for_diarization: bool
    diarization_num_speakers: int
    diarization_min_speakers: int
    diarization_max_speakers: int
    audio_chunks: list[AudioChunk]
    store: TranscriptStore
    queue: asyncio.Queue[ChunkMessage | None]
    transcriber: SessionTranscriber
    overlap_tail_pcm: bytes
    last_chunk_seq: int
    last_chunk_offset_ms: int
    failed_prepared_chunks: list[FailedPreparedChunk]
    asr_input_tokens: int
    asr_output_tokens: int
    asr_total_tokens: int
    asr_estimated_tokens: int
