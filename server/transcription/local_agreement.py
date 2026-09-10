"""Small, provider-independent primitives for HTTP Whisper streaming."""
from __future__ import annotations

import array
import io
import math
import re
import sys
import wave

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320


def pcm_wav(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return out.getvalue()


def speech_bounds(pcm: bytes, *, floor: float = 0.003) -> tuple[int, int] | None:
    """Conservative energy gate: retain low-volume speech and a padded onset.

    This is deliberately not advertised as neural VAD. Silence never enters
    the decoder, while original PCM remains recoverable on disk.
    """
    values = array.array("h")
    values.frombytes(pcm[:len(pcm) // 2 * 2])
    if sys.byteorder != "little":
        values.byteswap()
    first = last = None
    for start in range(0, len(values), FRAME_SAMPLES):
        frame = values[start:start + FRAME_SAMPLES]
        rms = math.sqrt(sum(float(v) * v for v in frame) / max(1, len(frame))) / 32768
        if rms >= floor:
            first = start if first is None else first
            last = start + len(frame)
    return None if first is None else (max(0, first - 4800), min(len(values), last + 1600))


def agreed_prefix(previous: str, current: str) -> str:
    """LocalAgreement-2, Unicode-character based for unspaced Japanese.

    A short tail is withheld. Latin partial words are withheld as a whole.
    Only matching text is stable; acoustic confidence is not a probability.
    """
    length = 0
    for left, right in zip(previous, current):
        if left != right:
            break
        length += 1
    if length == len(current) and current.endswith(("。", "！", "？", ".", "!", "?")):
        return current
    prefix = current[:max(0, length - 2)]
    return re.sub(r"[A-Za-z0-9]+$", "", prefix).rstrip()
