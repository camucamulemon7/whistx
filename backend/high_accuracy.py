from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import time

import numpy as np

from . import config

_logger = logging.getLogger("REFINE_STAGE")
try:
    _level = getattr(logging, getattr(config, "REFINE_LOG_LEVEL", "INFO").upper(), logging.INFO)
except Exception:
    _level = logging.INFO
_logger.setLevel(_level)


def pcm16le_bytes_to_np(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16)


def np_int16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


@dataclass
class RefinementJob:
    segment_id: str
    pcm16: bytes
    ts: Tuple[int, int]
    coarse_text: str
    language: Optional[str] = None
    queued_at: float = field(default_factory=time.time)


class SegmentAccumulator:
    """時間ベースの長尺チャンクを構築するためのアキュムレータ."""

    def __init__(self, sample_rate: int, chunk_ms: int, overlap_ms: int, min_speech_ms: int):
        self.sample_rate = sample_rate
        self.chunk_ms = max(1000, int(chunk_ms))
        self.overlap_ms = max(0, int(overlap_ms))
        self.min_speech_ms = max(0, int(min_speech_ms))
        self.buffer = bytearray()
        self.head_ts: Optional[int] = None
        self.tail_ts: Optional[int] = None
        self.target_rms = 10 ** (-23.0 / 20.0)  # 約 0.07
        self._noise_profile: Optional[np.ndarray] = None
        self._noise_alpha = 0.1

    def _duration_ms(self) -> int:
        if self.head_ts is None or self.tail_ts is None:
            return 0
        return max(0, self.tail_ts - self.head_ts)

    def remaining_ms(self) -> int:
        dur = self._duration_ms()
        return max(0, self.chunk_ms - dur)

    def add(self, pcm16: bytes, ts: Tuple[int, int], *, force: bool = False) -> Optional[Tuple[bytes, Tuple[int, int]]]:
        seg_ms = max(0, int(ts[1] - ts[0]))
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-9))
        if seg_ms < self.min_speech_ms and not force:
            # ノイズ的な短片断はノイズプロファイルとして学習。ただし十分な音量なら保持
            if rms < self.target_rms * 0.55:
                self._update_noise_profile(samples)
                return None
        if self.head_ts is None:
            self.head_ts = ts[0]
        self.tail_ts = max(ts[1], ts[0])
        self.buffer.extend(pcm16)
        if force or self._duration_ms() >= self.chunk_ms:
            return self.flush(force=force)
        return None

    def flush(self, *, force: bool = False) -> Optional[Tuple[bytes, Tuple[int, int]]]:
        if not self.buffer:
            self.head_ts = None
            self.tail_ts = None
            return None
        if not force and self._duration_ms() < max(self.chunk_ms // 2, 1000):
            return None
        audio = bytes(self.buffer)
        ts = (self.head_ts or 0, self.tail_ts or (self.head_ts or 0))
        # オーバーラップ処理
        keep_bytes = 0
        if self.overlap_ms > 0:
            keep_samples = int(self.sample_rate * (self.overlap_ms / 1000.0))
            keep_bytes = max(0, keep_samples * 2)
        if keep_bytes and len(self.buffer) > keep_bytes:
            tail = self.buffer[-keep_bytes:]
            new_head = max(0, ts[1] - self.overlap_ms)
            self.buffer = bytearray(tail)
            self.head_ts = new_head
            self.tail_ts = ts[1]
        else:
            self.buffer.clear()
            self.head_ts = None
            self.tail_ts = None
        audio = self._post_process(audio)
        return audio, ts

    def _update_noise_profile(self, samples: np.ndarray):
        if samples.size < 32:
            return
        spec = np.abs(np.fft.rfft(samples))
        if self._noise_profile is None or self._noise_profile.shape != spec.shape:
            self._noise_profile = spec
        else:
            self._noise_profile = (1.0 - self._noise_alpha) * self._noise_profile + self._noise_alpha * spec

    def _post_process(self, pcm16: bytes) -> bytes:
        if not pcm16:
            return pcm16
        f32 = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if f32.size == 0:
            return pcm16
        if self._noise_profile is not None:
            spec = np.fft.rfft(f32)
            mag = np.abs(spec)
            phase = np.angle(spec)
            noise = self._noise_profile
            if noise.shape != mag.shape:
                idx = np.linspace(0, noise.shape[0] - 1, mag.shape[0])
                noise = np.interp(idx, np.arange(noise.shape[0]), noise)
            clean_mag = np.maximum(0.0, mag - noise)
            spec_clean = clean_mag * np.exp(1j * phase)
            f32 = np.fft.irfft(spec_clean, n=f32.size)
        rms = float(np.sqrt(np.mean(np.square(f32)) + 1e-9))
        if rms > 0:
            gain = np.clip(self.target_rms / rms, 0.5, 4.0)
            f32 *= gain
        f32 = np.clip(f32, -1.0, 1.0)
        return (f32 * 32768.0).astype(np.int16).tobytes()


class HighAccuracyDecoder:
    """Parakeet CTC モデルを高ビーム幅で再デコードする."""

    def __init__(self, backend, *, language: Optional[str]):
        self.backend = backend
        self.language = language
        self.decoder_type = getattr(config, "HIGH_ACCURACY_DECODER_TYPE", "beamsearch")
        self.beam_width = int(getattr(config, "HIGH_ACCURACY_BEAM_WIDTH", 64))
        self.lm_path = getattr(config, "HIGH_ACCURACY_LM_PATH", "") or None
        self.lm_weight = float(getattr(config, "HIGH_ACCURACY_LM_WEIGHT", 2.0))
        self.word_score = float(getattr(config, "HIGH_ACCURACY_WORD_SCORE", -1.5))
        self.patience = float(getattr(config, "HIGH_ACCURACY_PATIENCE", 1.6))
        self._lock = asyncio.Lock()
        self._supports_decoder_options = hasattr(self.backend, "_apply_decoder_options")
        self.chunk_seconds = 6
        self.overlap_seconds = 1

    async def decode(self, pcm16: bytes, language: Optional[str] = None) -> str:
        lang = language or self.language
        audio = np_int16_to_float32(pcm16le_bytes_to_np(pcm16))
        loop = asyncio.get_running_loop()
        decoder_options: Optional[Dict[str, Any]] = None
        if self._supports_decoder_options:
            decoder_options = {
                "decoder_type": self.decoder_type,
                "beam_size": self.beam_width,
                "alpha": self.lm_weight,
                "beta": self.word_score,
            }
            if self.lm_path:
                decoder_options["lm_path"] = self.lm_path
        async with self._lock:
            try:
                if audio.size > int((self.chunk_seconds + self.overlap_seconds) * 16000):
                    result = await loop.run_in_executor(
                        None,
                        lambda: self._decode_with_chunks(audio, lang, decoder_options),
                    )
                else:
                    result = await loop.run_in_executor(
                        None,
                        lambda: self._run_decode_once(audio, lang, decoder_options),
                    )
            except Exception as exc:
                _logger.exception("high accuracy decode failed: %s", exc)
                result = ""
        return self._cleanup_text(result or "")

    def _decode_with_chunks(self, audio: np.ndarray, language: Optional[str], decoder_options: Optional[Dict[str, Any]]) -> str:
        sr = 16000
        chunk_samples = int(self.chunk_seconds * sr)
        step_samples = int((self.chunk_seconds - self.overlap_seconds) * sr)
        if step_samples <= 0:
            step_samples = chunk_samples
        segments = []
        start = 0
        while start < audio.shape[0]:
            end = min(audio.shape[0], start + chunk_samples)
            chunk = audio[start:end]
            text = self._run_decode_once(chunk, language, decoder_options)
            segments.append(text or "")
            if end >= audio.shape[0]:
                break
            start += step_samples
        return "".join(segments)

    def _run_decode_once(self, audio: np.ndarray, language: Optional[str], decoder_options: Optional[Dict[str, Any]]) -> str:
        kwargs: Dict[str, Any] = {
            "audio_f32_mono_16k": audio,
            "language": language,
            "beam_size": self.beam_width,
            "patience": self.patience,
            "partial": False,
        }
        if self._supports_decoder_options and decoder_options:
            kwargs["decoder_options"] = decoder_options
        return self.backend.transcribe(**kwargs)

    def _cleanup_text(self, text: str) -> str:
        if not text:
            return text
        trimmed = text.strip()
        if not trimmed:
            return trimmed
        # 単音節のノイズ除去（よく誤検知される仮名を対象）
        noise_tokens = {"あ", "え", "い", "う", "お", "ぴ", "えっ", "あっ"}
        if trimmed in noise_tokens and len(trimmed) <= 2:
            return ""
        return trimmed
