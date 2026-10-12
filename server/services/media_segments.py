"""File-only, bounded pause-aware PCM segmentation; no live/HQ settings."""
from __future__ import annotations

import array
from dataclasses import dataclass
import math
import re
import sys
import wave

from ..transcription.local_agreement import SAMPLE_RATE

FRAME = SAMPLE_RATE // 50


@dataclass(frozen=True)
class FilePolicy:
    target_seconds: int = 75
    max_seconds: int = 90
    search_seconds: int = 15
    pause_ms: int = 400
    overlap_ms: int = 800
    strong_pause_ms: int = 800


def file_policy(backend: str, *, language: str = 'auto') -> FilePolicy:
    # speech.cpp v0.8.3 takes much longer inputs, but has a 4096 output-token
    # ceiling and serves one request at a time. Bound memory and head-of-line
    # latency instead of copying recording's 30 s HQ interval.
    if backend == 'speech_cpp':
        return FilePolicy(strong_pause_ms=800 if language in ('auto', '') else 0)
    # Preserve the conventional Whisper input budget and unverified gateways.
    return FilePolicy(target_seconds=30, max_seconds=30, search_seconds=0, overlap_ms=0, strong_pause_ms=0)


@dataclass(frozen=True)
class FileSegment:
    start: int
    end: int
    pcm: bytes
    overlap_samples: int = 0
    boundary: str = 'end'


def frame_energy(pcm: bytes) -> list[float]:
    samples = array.array('h')
    samples.frombytes(pcm)
    if sys.byteorder != 'little':
        samples.byteswap()
    energies = []
    for start in range(0, len(samples), FRAME):
        values = samples[start:start + FRAME]
        mean = sum(values) / len(values)
        energies.append(math.sqrt(sum((v - mean)**2 for v in values) / len(values)) / 32768)
    return energies


def choose_boundary(pcm: bytes, policy: FilePolicy, *, final: bool = False) -> tuple[int, bool]:
    """Prefer a genuine quiet run near the target; otherwise keep context."""
    total = len(pcm) // 2
    target = policy.target_seconds * SAMPLE_RATE
    if policy.search_seconds == 0 or (final and not policy.strong_pause_ms):
        return total, False
    energy = frame_energy(pcm)
    quiet_start = None
    candidates = []
    strong = []
    low = max(SAMPLE_RATE, target - policy.search_seconds * SAMPLE_RATE)
    high = min(total, target + policy.search_seconds * SAMPLE_RATE)
    for i, rms in enumerate(energy + [1.0]):
        if rms < .003 and quiet_start is None:
            quiet_start = i * FRAME
        elif rms >= .003 and quiet_start is not None:
            quiet_end = min(total, i * FRAME)
            # Qwen can silently choose one language for a mixed request.
            # Never join utterance groups across a long pause merely to fill
            # a larger file batch. No language is inferred from signal energy.
            if (policy.strong_pause_ms and quiet_end - quiet_start >= policy.strong_pause_ms * SAMPLE_RATE // 1000
                    and quiet_start >= 3 * SAMPLE_RATE and quiet_end < total):
                strong.append((quiet_start + quiet_end) // 2)
            if quiet_end - quiet_start >= policy.pause_ms * SAMPLE_RATE // 1000:
                padding = policy.pause_ms * SAMPLE_RATE // 2000
                left, right = max(low, quiet_start + padding), min(high, quiet_end - padding)
                if left <= right:
                    candidates.append(max(left, min(target, right)))
            quiet_start = None
    if strong:
        return strong[0], False
    if final:
        return total, False
    if candidates:
        return min(candidates, key=lambda position: abs(position - target)), False
    # No safe pause: use the quietest 20 ms frame near the hard bound. The
    # following interval replays only this boundary's short acoustic context.
    upper = min(total, policy.max_seconds * SAMPLE_RATE)
    lower = max(FRAME, upper - 2 * SAMPLE_RATE)
    indexes = range(lower // FRAME, max(lower // FRAME + 1, upper // FRAME))
    cut = min(indexes, key=lambda i: energy[i]) * FRAME
    return max(FRAME, cut), True


def iter_file_segments(audio: wave.Wave_read, policy: FilePolicy):
    total = audio.getnframes()
    start = 0
    overlap = 0
    while start < total:
        audio.setpos(start)
        pcm = audio.readframes(policy.max_seconds * SAMPLE_RATE)
        remaining = total - start
        cut, forced = choose_boundary(pcm, policy, final=remaining <= policy.max_seconds * SAMPLE_RATE)
        # Avoid a second tiny request when the complete tail fits the budget.
        if forced and remaining <= policy.max_seconds * SAMPLE_RATE and remaining - cut < 10 * SAMPLE_RATE:
            cut, forced = remaining, False
        end = start + cut
        yield FileSegment(start, end, pcm[:cut * 2], overlap, 'limit' if forced else 'pause' if end < total else 'end')
        if end == total:
            break
        overlap = min(policy.overlap_ms * SAMPLE_RATE // 1000, cut // 2) if forced else 0
        start = end - overlap


def trim_overlap_text(previous: str, current: str, *, overlapping: bool) -> str:
    """Only exact, substantial boundary matches; never fuzzy-delete speech."""
    if not overlapping:
        return current
    # Preserve source spelling; ignore punctuation/spacing only for matching.
    def tokens(text):
        return [(match.group().casefold(), match.end(), match.start()) for match in re.finditer(r'\w', text)]
    left, right = tokens(previous), tokens(current)
    # At most a short boundary phrase: a long repeated sentence cannot have
    # been spoken in the 800 ms acoustic overlap. Never remove a whole result.
    for count in range(min(len(left), len(right) - 1, 32), 3, -1):
        if [v for v, _, _ in left[-count:]] != [v for v, _, _ in right[:count]]:
            continue
        end, start = right[count - 1][1], left[-count][2]
        if all(v.isascii() for v, _, _ in right[:count]):
            # English boundaries must contain two complete words; "market"
            # must not consume the prefix of "marketplace".
            def word(char):
                return char.isascii() and char.isalnum()
            if (start and word(previous[start-1])) or (end < len(current) and word(current[end])):
                continue
            if len(re.findall(r'[A-Za-z0-9]+', current[:end])) < 2:
                continue
        return current[end:].lstrip(' 、。，,.!?！？\n')
    return current


def split_limited_segment(segment: FileSegment) -> tuple[FileSegment, FileSegment]:
    """Discard the limited attempt, then retry bounded halves with source times."""
    size = len(segment.pcm) // 2
    middle = size // 2
    energy = frame_energy(segment.pcm)
    candidates = range(max(1, (middle - SAMPLE_RATE * 2) // FRAME),
                       min(len(energy) - 1, (middle + SAMPLE_RATE * 2) // FRAME))
    quiet = [i for i in candidates if max(energy[max(0, i-10):i+10]) < .003]
    cut = min(quiet, key=lambda i: abs(i * FRAME - middle)) * FRAME if quiet else middle
    overlap = 0 if quiet else min(SAMPLE_RATE * 4 // 5, cut // 2)
    left = FileSegment(segment.start, segment.start + cut, segment.pcm[:cut*2],
                       segment.overlap_samples, 'pause' if quiet else 'limit')
    right = FileSegment(segment.start + cut - overlap, segment.end,
                        segment.pcm[(cut-overlap)*2:], overlap, segment.boundary)
    return left, right
