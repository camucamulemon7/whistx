from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ASRChunkResult:
    text: str
    start_ms: int | None
    end_ms: int | None


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
