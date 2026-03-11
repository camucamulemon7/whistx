from __future__ import annotations

import io
import logging
import shutil
import subprocess
import wave
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedAudio:
    audio_bytes: bytes
    mime_type: str
    overlap_ms_used: int
    tail_pcm: bytes


class AudioPreprocessor:
    def __init__(
        self,
        *,
        ffmpeg_bin: str,
        sample_rate: int,
        overlap_ms: int,
        enabled: bool,
    ) -> None:
        if shutil.which(ffmpeg_bin) is None:
            raise RuntimeError(f"{ffmpeg_bin} is not installed")
        self.ffmpeg_bin = ffmpeg_bin
        self.sample_rate = sample_rate
        self.overlap_ms = max(0, overlap_ms)
        self.enabled = enabled

    def prepare(
        self,
        *,
        audio_bytes: bytes,
        mime_type: str,
        previous_tail_pcm: bytes,
        chunk_duration_ms: int | None = None,
        source_mode: str = "mic",
    ) -> PreparedAudio:
        pcm = self._decode_to_pcm(audio_bytes=audio_bytes, mime_type=mime_type, source_mode=source_mode)
        overlap_pcm = self._trim_tail(previous_tail_pcm, target_ms=self._resolve_overlap_ms(chunk_duration_ms))
        merged_pcm = overlap_pcm + pcm if overlap_pcm else pcm
        overlap_ms_used = self._pcm_bytes_to_ms(len(overlap_pcm))
        tail_pcm = self._trim_tail(pcm, target_ms=self._resolve_overlap_ms(chunk_duration_ms))
        wav_bytes = self._encode_wav(merged_pcm)
        return PreparedAudio(
            audio_bytes=wav_bytes,
            mime_type="audio/wav",
            overlap_ms_used=overlap_ms_used,
            tail_pcm=tail_pcm,
        )

    def _decode_to_pcm(self, *, audio_bytes: bytes, mime_type: str, source_mode: str) -> bytes:
        suffix = _ext_from_mime(mime_type)
        filters = self._build_filters(source_mode)

        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            suffix,
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-af",
            ",".join(filters),
            "-f",
            "s16le",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, input=audio_bytes, capture_output=True, check=False)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg preprocess failed: {stderr or 'unknown error'}")
        return proc.stdout

    def _build_filters(self, source_mode: str) -> list[str]:
        mode = (source_mode or "mic").strip().lower()
        if mode == "display":
            filters = ["highpass=f=50", "lowpass=f=10000", "volume=1.15"]
            if self.enabled:
                filters.append("acompressor=threshold=0.12:ratio=2.2:attack=20:release=180:makeup=1")
            return filters

        if mode == "both":
            filters = ["highpass=f=80", "lowpass=f=9200"]
            if self.enabled:
                filters.extend(
                    [
                        "afftdn=nf=-20",
                        "dynaudnorm=f=120:g=7",
                        "acompressor=threshold=0.10:ratio=2.0:attack=15:release=160:makeup=1",
                    ]
                )
            return filters

        filters = ["highpass=f=120", "lowpass=f=7600"]
        if self.enabled:
            filters.extend(
                [
                    "afftdn=nf=-25",
                    "dynaudnorm=f=150:g=9",
                ]
            )
        return filters

    def _trim_tail(self, pcm_bytes: bytes, *, target_ms: int) -> bytes:
        if target_ms <= 0 or not pcm_bytes:
            return b""
        tail_bytes = self._ms_to_pcm_bytes(target_ms)
        if tail_bytes <= 0:
            return b""
        return pcm_bytes[-tail_bytes:]

    def _resolve_overlap_ms(self, chunk_duration_ms: int | None) -> int:
        if self.overlap_ms <= 0:
            return 0
        if not chunk_duration_ms or chunk_duration_ms <= 0:
            return self.overlap_ms

        adaptive = int(round(chunk_duration_ms * 0.14))
        lower_bound = min(self.overlap_ms, 2_500)
        upper_bound = max(self.overlap_ms, 4_000)
        return max(lower_bound, min(upper_bound, adaptive))

    def _encode_wav(self, pcm_bytes: bytes) -> bytes:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2)
                wav_out.setframerate(self.sample_rate)
                wav_out.writeframes(pcm_bytes)
            return buffer.getvalue()

    def _ms_to_pcm_bytes(self, duration_ms: int) -> int:
        samples = max(0, int(round(self.sample_rate * duration_ms / 1000)))
        return samples * 2

    def _pcm_bytes_to_ms(self, size_bytes: int) -> int:
        if size_bytes <= 0:
            return 0
        samples = size_bytes // 2
        return max(0, int(round(samples * 1000 / self.sample_rate)))


def _ext_from_mime(mime_type: str) -> str:
    lowered = (mime_type or "").lower()
    if "wav" in lowered:
        return "wav"
    if "webm" in lowered:
        return "webm"
    if "ogg" in lowered or "opus" in lowered:
        return "ogg"
    if "mp4" in lowered or "m4a" in lowered:
        return "mp4"
    if "mpeg" in lowered or "mp3" in lowered:
        return "mp3"
    return "webm"
