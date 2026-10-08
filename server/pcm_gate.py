"""Conservative PCM gate for ASR; original recordings stay untouched."""
from __future__ import annotations

import array
import io
import math
import sys
import wave

from .transcription.local_agreement import FRAME_SAMPLES, SAMPLE_RATE, pcm_wav, speech_bounds


def pcm_speech_bounds(pcm: bytes) -> tuple[int, int] | None:
    """Reuse the existing floor/padding after removing each frame's DC offset.

    Reject only quiet, sustained, stationary broadband noise. Any active frame
    with lower zero-crossing density or changing energy keeps the whole speech
    envelope; short acknowledgements need no minimum duration or speech ratio.
    This is a signal heuristic, not a speech classifier.
    """
    samples = array.array('h')
    samples.frombytes(pcm[:len(pcm) // 2 * 2])
    if sys.byteorder != 'little':
        samples.byteswap()
    centered = array.array('h')
    energies = []
    crossings = []
    for start in range(0, len(samples), FRAME_SAMPLES):
        frame = samples[start:start + FRAME_SAMPLES]
        mean = sum(frame) / len(frame)
        values = [max(-32768, min(32767, round(v - mean))) for v in frame]
        centered.extend(values)
        energies.append(math.sqrt(sum(v*v for v in values) / len(values)) / 32768)
        crossings.append(sum((a < 0) != (b < 0) for a, b in zip(values, values[1:])) / max(1, len(values)-1))
    if sys.byteorder != 'little':
        centered.byteswap()
    bounds = speech_bounds(centered.tobytes())
    if bounds is None:
        return None
    active = [(rms, zcr) for rms, zcr in zip(energies, crossings) if rms >= 0.003]
    if len(active) >= 25 and all(rms < 0.01 and zcr > 0.35 for rms, zcr in active):
        mean_energy = sum(rms for rms, _ in active) / len(active)
        variation = math.sqrt(sum((rms-mean_energy)**2 for rms, _ in active) / len(active)) / mean_energy
        if variation < 0.1:
            return None
    return bounds


def prepare_pcm_wav(audio: bytes, mime_type: str) -> tuple[bytes, int, bool]:
    """Gate/trim supported WAV; leave unknown formats unchanged, never guess.

    Offset is relative to the original audio, for provider timestamp remapping.
    The silence flag is acoustic evidence, separate from a blank ASR response.
    """
    if 'wav' not in mime_type.lower():
        return audio, 0, False
    try:
        with wave.open(io.BytesIO(audio)) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, SAMPLE_RATE, 'NONE'):
                return audio, 0, False
            pcm = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError, ValueError):
        return audio, 0, False
    bounds = pcm_speech_bounds(pcm)
    if bounds is None:
        return audio, 0, True
    start, end = bounds
    # Keep 300 ms after the energy boundary for quiet trailing phonemes. The
    # shared bounds already include 100 ms, so extend rather than tighten it.
    end = min(len(pcm)//2, end + SAMPLE_RATE//5)
    return pcm_wav(pcm[start*2:end*2]), start*1000//SAMPLE_RATE, False
