from __future__ import annotations

import io
import logging
import shutil
import subprocess
import wave
from array import array
from dataclasses import dataclass

from .config import settings


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedAudio:
    audio_bytes: bytes
    mime_type: str
    overlap_ms_used: int
    tail_pcm: bytes
    rms: float
    peak: float
    speech_ratio: float
    audio_metrics: dict[str, float]


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
        speech_ratio: float | None = None,
        active_ms: int | None = None,
        silence_ms: int | None = None,
    ) -> PreparedAudio:
        pcm = self._decode_to_pcm(audio_bytes=audio_bytes, mime_type=mime_type, source_mode=source_mode)
        target_overlap_ms = self._resolve_overlap_ms(
            chunk_duration_ms,
            speech_ratio=speech_ratio,
            active_ms=active_ms,
            silence_ms=silence_ms,
            source_mode=source_mode,
        )
        overlap_pcm = self._trim_tail(previous_tail_pcm, target_ms=target_overlap_ms)
        merged_pcm = overlap_pcm + pcm if overlap_pcm else pcm
        overlap_ms_used = self._pcm_bytes_to_ms(len(overlap_pcm))
        tail_pcm = self._trim_tail(pcm, target_ms=target_overlap_ms)
        wav_bytes = self._encode_wav(merged_pcm)
        metrics = self._compute_audio_metrics(merged_pcm)
        return PreparedAudio(
            audio_bytes=wav_bytes,
            mime_type="audio/wav",
            overlap_ms_used=overlap_ms_used,
            tail_pcm=tail_pcm,
            rms=metrics["rms"],
            peak=metrics["peak"],
            speech_ratio=metrics["speech_ratio"],
            audio_metrics=metrics,
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
        try:
            proc = subprocess.run(
                cmd,
                input=audio_bytes,
                capture_output=True,
                check=False,
                timeout=settings.ffmpeg_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "ffmpeg preprocess timeout: source_mode=%s timeout=%ss",
                source_mode,
                settings.ffmpeg_timeout_seconds,
            )
            raise RuntimeError("ffmpeg preprocess timeout") from exc
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

    def _resolve_overlap_ms(
        self,
        chunk_duration_ms: int | None,
        *,
        speech_ratio: float | None = None,
        active_ms: int | None = None,
        silence_ms: int | None = None,
        source_mode: str = "mic",
    ) -> int:
        if self.overlap_ms <= 0:
            return 0
        if not chunk_duration_ms or chunk_duration_ms <= 0:
            return self.overlap_ms

        adaptive = int(round(chunk_duration_ms * 0.14))
        lower_bound = min(self.overlap_ms, 2_500)
        upper_bound = max(self.overlap_ms, 4_000)
        resolved = max(lower_bound, min(upper_bound, adaptive))

        ratio = 0.0 if speech_ratio is None else max(0.0, min(1.0, float(speech_ratio)))
        if ratio >= 0.45:
            resolved = int(round(resolved * 1.18))
        elif ratio <= 0.12:
            resolved = int(round(resolved * 0.9))

        if active_ms is not None and chunk_duration_ms > 0:
            activity_ratio = max(0.0, min(1.0, float(active_ms) / float(chunk_duration_ms)))
            if activity_ratio >= 0.65:
                resolved = int(round(resolved * 1.08))

        if silence_ms is not None and silence_ms >= 1200:
            resolved = int(round(resolved * 0.82))

        mode = (source_mode or "mic").strip().lower()
        if mode == "display":
            resolved = int(round(resolved * 1.08))
        elif mode == "both":
            resolved = int(round(resolved * 1.12))

        return max(lower_bound, min(upper_bound, resolved))

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

    def _compute_audio_metrics(self, pcm_bytes: bytes) -> dict[str, float]:
        if not pcm_bytes:
            return {"rms": 0.0, "peak": 0.0, "speech_ratio": 0.0}

        samples = array("h")
        samples.frombytes(pcm_bytes)
        sample_count = len(samples)
        if sample_count <= 0:
            return {"rms": 0.0, "peak": 0.0, "speech_ratio": 0.0}

        sum_squares = 0.0
        peak = 0.0
        speech_samples = 0
        speech_threshold = 900.0 / 32768.0

        for sample in samples:
            normalized = abs(float(sample) / 32768.0)
            sum_squares += normalized * normalized
            if normalized > peak:
                peak = normalized
            if normalized >= speech_threshold:
                speech_samples += 1

        rms = (sum_squares / sample_count) ** 0.5
        speech_ratio = speech_samples / sample_count
        return {
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "speech_ratio": round(speech_ratio, 6),
        }


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
