from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ASRChunkResult:
    text: str
    start_ms: int | None
    end_ms: int | None
    usage_details: dict[str, int] | None = None
    estimated_tokens: int = 0
    max_no_speech_prob: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    suspicious: bool = False


class SessionTranscriber(Protocol):
    def transcribe_chunk(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
        trace_context: dict[str, str] | None = None,
    ) -> ASRChunkResult: ...

    def close(self) -> None: ...
