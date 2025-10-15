import asyncio
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import os
import difflib

import numpy as np
import torch
import webrtcvad

from . import config
from .asr_backends import create_backend
from .diarizer import GlobalDiarizer
from .hotwords import HotwordStore, boost_hotwords
from .high_accuracy import HighAccuracyDecoder, RefinementJob, SegmentAccumulator
from .metrics import (
    refine_latency_seconds,
    refine_queue_depth,
    longform_chunk_latency_seconds,
    longform_queue_depth,
)


def pcm16le_bytes_to_np(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16)


def np_int16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


logger = logging.getLogger(__name__)

PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "realtime": {
        "partial_interval_ms": config.PARTIAL_INTERVAL_MS,
        "beam_size": 12,
        "patience": 1.1,
        "utterance_silence_ms": config.UTTERANCE_SILENCE_MS,
        "min_final_ms": getattr(config, "MIN_FINAL_MS", 800),
    },
    "balanced": {
        "partial_interval_ms": max(800, config.PARTIAL_INTERVAL_MS),
        "beam_size": 24,
        "patience": 1.3,
        "utterance_silence_ms": max(900, config.UTTERANCE_SILENCE_MS),
        "min_final_ms": max(900, getattr(config, "MIN_FINAL_MS", 800)),
    },
    "high_accuracy": {
        "partial_interval_ms": 1200,
        "beam_size": getattr(config, "HIGH_ACCURACY_BEAM_WIDTH", 48),
        "patience": getattr(config, "HIGH_ACCURACY_PATIENCE", 1.6),
        "utterance_silence_ms": 1200,
        "min_final_ms": 1100,
    },
}

TAIL_PUNCT = getattr(config, "PUNCT_CHARS", "。．.!！?？")

try:
    from huggingface_hub.utils import _tqdm as hf_tqdm_module  # type: ignore
    if not getattr(hf_tqdm_module, "_whistx_patched", False):
        _OriginalTqdm = hf_tqdm_module.tqdm

        class _LoggingTqdm(_OriginalTqdm):  # type: ignore
            def __init__(self, *args, **kwargs):
                self._download_logger = logging.getLogger("MODEL_DOWNLOAD")
                self._last_percent = -1
                super().__init__(*args, **kwargs)
                self._desc_name = getattr(self, 'desc', None) or kwargs.get('desc') or ""

            def _log_progress(self, force: bool = False):
                total = getattr(self, 'total', None)
                current = getattr(self, 'n', None)
                if not total or total <= 0 or current is None:
                    return
                percent = int(min(100, max(0, (current / total) * 100)))
                if not force:
                    if percent == self._last_percent:
                        return
                    if percent < 100 and self._last_percent != -1 and percent - self._last_percent < 10:
                        return
                self._last_percent = percent
                desc_value = getattr(self, 'desc', None) or self._desc_name or "download"
                desc_value = desc_value.replace('"', "'")
                self._download_logger.info(
                    "[MODEL_DOWNLOAD] {\"stage\": \"progress\", \"desc\": \"%s\", \"percent\": %d}",
                    desc_value,
                    percent,
                )

            def update(self, n=1):
                result = super().update(n)
                self._log_progress()
                return result

            def refresh(self, *args, **kwargs):
                result = super().refresh(*args, **kwargs)
                self._log_progress()
                return result

            def close(self):
                self._log_progress(force=True)
                return super().close()

        hf_tqdm_module.tqdm = _LoggingTqdm
        try:
            import huggingface_hub.utils as hf_utils  # type: ignore
            hf_utils.tqdm = _LoggingTqdm
        except Exception:
            pass
        hf_tqdm_module._whistx_patched = True
except Exception:
    pass


class GlobalASR:
    _instances: Dict[str, object] = {}

    @classmethod
    def get(cls, backend_name: Optional[str] = None, language: Optional[str] = None):
        backend = backend_name or getattr(config, 'ASR_BACKEND', 'default')
        lang_key = (language or '').lower()
        if backend == 'parakeet':
            key = f"{backend}:{lang_key or 'default'}"
        else:
            key = backend
        if key not in cls._instances:
            logger.info("[MODEL_DOWNLOAD] {\"backend\": \"%s\", \"language\": \"%s\", \"stage\": \"start\"}", backend, lang_key or 'default')
            cls._instances[key] = create_backend(config, backend_name=backend, language=language)
            descriptor = getattr(cls._instances[key], 'model_id', getattr(cls._instances[key], 'model_name', backend))
            logger.info(
                "[MODEL_DOWNLOAD] {\"backend\": \"%s\", \"language\": \"%s\", \"stage\": \"complete\", \"model\": \"%s\"}",
                backend,
                lang_key or 'default',
                descriptor,
            )
            logger.info("[ASR] backend=%s lang=%s 初期化完了", backend, lang_key or 'default')
        else:
            logger.info("[ASR] backend=%s lang=%s を再利用します", backend, lang_key or 'default')
            descriptor = getattr(cls._instances[key], 'model_id', getattr(cls._instances[key], 'model_name', backend))
            logger.info(
                "[MODEL_DOWNLOAD] {\"backend\": \"%s\", \"language\": \"%s\", \"stage\": \"reuse\", \"model\": \"%s\"}",
                backend,
                lang_key or 'default',
                descriptor,
            )
        return cls._instances[key]


@dataclass
class TranscriptEvent:
    type: str  # "partial" | "final" | "info" | "error"
    text: str
    ts_start: int
    ts_end: int
    seq: Optional[int] = None
    segment_id: Optional[int] = None


class VADSegmenter:
    """Silero を優先し、失敗時は WebRTC VAD にフォールバックするストリーミング VAD。

    feed() は常に 20ms（config.VAD_FRAME_MS）刻みで評価し、
    in_speech の変化と十分な無音で確定区切りを生成します。
    """

    def __init__(self, vad_opts: Optional[Dict[str, Any]] = None):
        vad_opts = vad_opts or {}
        self.backend = str(vad_opts.get("VAD_BACKEND", config.VAD_BACKEND)).lower()
        self.sample_rate = int(vad_opts.get("sampleRate", config.SAMPLE_RATE))
        self.vad_frame_ms = int(vad_opts.get("VAD_FRAME_MS", config.VAD_FRAME_MS))
        self.vad_frame_bytes = int(self.sample_rate * (self.vad_frame_ms / 1000.0)) * 2
        self.utterance_silence_ms = int(vad_opts.get("UTTERANCE_SILENCE_MS", config.UTTERANCE_SILENCE_MS))
        self.force_utterance_ms = int(vad_opts.get("FORCE_UTTERANCE_MS", getattr(config, "FORCE_UTTERANCE_MS", 9000)))
        self.force_overlap_ms = int(vad_opts.get("FORCE_OVERLAP_MS", getattr(config, "FORCE_OVERLAP_MS", 1200)))
        self.silero_threshold = float(vad_opts.get("SILERO_THRESHOLD", config.SILERO_THRESHOLD))
        self.silero_min_silence_ms = int(vad_opts.get("SILERO_MIN_SILENCE_MS", config.SILERO_MIN_SILENCE_MS))
        self.silero_min_speech_ms = int(vad_opts.get("SILERO_MIN_SPEECH_MS", config.SILERO_MIN_SPEECH_MS))
        # Auto-tune
        self.auto_vad_enable = bool(int(vad_opts.get("AUTO_VAD_ENABLE", 1 if getattr(config, 'AUTO_VAD_ENABLE', True) else 0)))
        self.auto_window_ms = int(vad_opts.get("AUTO_VAD_WINDOW_MS", getattr(config, 'AUTO_VAD_WINDOW_MS', 3000)))
        self.auto_step = float(vad_opts.get("AUTO_VAD_STEP", getattr(config, 'AUTO_VAD_STEP', 0.05)))
        self.auto_min_thr = float(vad_opts.get("AUTO_VAD_MIN_THR", getattr(config, 'AUTO_VAD_MIN_THR', 0.35)))
        self.auto_max_thr = float(vad_opts.get("AUTO_VAD_MAX_THR", getattr(config, 'AUTO_VAD_MAX_THR', 0.75)))
        self.auto_target_low = float(vad_opts.get("AUTO_VAD_TARGET_LOW", getattr(config, 'AUTO_VAD_TARGET_LOW', 0.08)))
        self.auto_target_high = float(vad_opts.get("AUTO_VAD_TARGET_HIGH", getattr(config, 'AUTO_VAD_TARGET_HIGH', 0.60)))
        self.auto_rms_low = float(vad_opts.get("AUTO_VAD_RMS_LOW", getattr(config, 'AUTO_VAD_RMS_LOW', 0.02)))
        self.auto_rms_high = float(vad_opts.get("AUTO_VAD_RMS_HIGH", getattr(config, 'AUTO_VAD_RMS_HIGH', 0.05)))
        self.auto_tune_silence = bool(int(vad_opts.get("AUTO_VAD_TUNE_SILENCE", 1 if getattr(config, 'AUTO_VAD_TUNE_SILENCE', True) else 0)))
        self._auto_frames = 0
        self._auto_speech_frames = 0
        self._auto_rms_acc = 0.0
        self._auto_next_eval = self.auto_window_ms
        self.min_final_ms = int(vad_opts.get("MIN_FINAL_MS", getattr(config, "MIN_FINAL_MS", 700)))
        # VAC 設定
        self.vac_enable = bool(int(vad_opts.get("VAC_ENABLE", 1 if getattr(config, 'VAC_ENABLE', False) else 0)))
        self.vac_min_speech_ms = int(vad_opts.get("VAC_MIN_SPEECH_MS", getattr(config, 'VAC_MIN_SPEECH_MS', 120)))
        self.vac_hang_ms = int(vad_opts.get("VAC_HANGOVER_MS", getattr(config, 'VAC_HANGOVER_MS', 260)))
        self.vac_min_final_ms = int(vad_opts.get("VAC_MIN_FINAL_MS", getattr(config, 'VAC_MIN_FINAL_MS', self.min_final_ms)))
        if self.vac_enable:
            self.min_final_ms = min(self.min_final_ms, self.vac_min_final_ms)
        self._vac_frames_required = max(1, int(self.vac_min_speech_ms / self.vad_frame_ms)) if self.vac_enable else 0
        self._vac_release_frames = max(1, int(self.vac_hang_ms / self.vad_frame_ms)) if self.vac_enable else 0
        self._vac_counter = 0
        self._vac_cooldown = 0
        self._vac_active = False
        # Silero-VAD は軽量のため CPU で十分。GPU 依存を避けて安定化。
        self.device = "cpu"
        self._silero = None  # (model, get_speech_timestamps)
        self._silero_window = bytearray()
        # 判定窓（ms）。UTTERANCE_SILENCE_MS の2倍か 1500ms の大きい方
        self._silero_window_ms = max(1500, self.utterance_silence_ms * 2)
        self._tail_ms = max(40, self.vad_frame_ms)
        if self.backend == "silero":
            try:
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    trust_repo=True,
                )
                (get_speech_ts, save_audio, read_audio, VADIterator, collect_chunks) = utils
                model.to(self.device).eval()
                self._silero = (model, get_speech_ts)
            except Exception as e:
                logger.warning("Silero-VAD 初期化失敗 (%s)。WebRTC VAD へフォールバックします。", e)
                self.backend = "webrtc"
        if self.backend != "silero":
            self.vad = webrtcvad.Vad(int(vad_opts.get("VAD_AGGRESSIVENESS", config.VAD_AGGRESSIVENESS)))
        self.reset()

    def reset(self):
        self.buffer = bytearray()
        self.current_chunk = bytearray()
        self.in_speech = False
        self.last_voice_ms = 0
        self.pts_ms = 0

    def _silero_is_speech(self, frame_bytes: bytes) -> bool:
        # 直近ウィンドウにフレームを追加し、末尾近傍に音声区間があれば発話中とみなす
        self._silero_window.extend(frame_bytes)
        max_bytes = int(self.sample_rate * (self._silero_window_ms / 1000.0)) * 2
        if len(self._silero_window) > max_bytes:
            # 古いデータを削る
            del self._silero_window[: len(self._silero_window) - max_bytes]
        if not self._silero:
            return False
        model, get_speech_ts = self._silero
        wav = pcm16le_bytes_to_np(bytes(self._silero_window)).astype(np.float32) / 32768.0
        try:
            wav_t = torch.from_numpy(wav).to(self.device)
            ts_list = get_speech_ts(
                wav_t,
                model,
                sampling_rate=self.sample_rate,
                threshold=self.silero_threshold,
                min_silence_duration_ms=self.silero_min_silence_ms,
                min_speech_duration_ms=self.silero_min_speech_ms,
            )
        except Exception:
            return False
        if not ts_list:
            return False
        tail_samples = int(self.sample_rate * (self._tail_ms / 1000.0))
        end_zone = len(wav) - tail_samples
        for seg in ts_list[::-1]:
            # seg は {'start': s, 'end': e}
            e = int(seg.get('end', 0))
            if e >= end_zone:
                return True
            if e < end_zone:
                break
        return False

    def _vac_filter(self, raw_speech: bool) -> bool:
        if not self.vac_enable:
            return raw_speech
        if raw_speech:
            self._vac_counter = min(self._vac_counter + 1, self._vac_frames_required * 3)
            self._vac_cooldown = self._vac_release_frames
        else:
            if self._vac_cooldown > 0:
                self._vac_cooldown -= 1
            else:
                self._vac_counter = max(0, self._vac_counter - 1)

        if not self._vac_active and self._vac_counter >= self._vac_frames_required:
            self._vac_active = True
        elif self._vac_active and self._vac_counter == 0 and self._vac_cooldown == 0:
            self._vac_active = False
        return self._vac_active

    def feed(self, pcm16_bytes: bytes, pts_ms: int) -> Tuple[Optional[bytes], Optional[Tuple[int, int]], bool, bool]:
        self.buffer.extend(pcm16_bytes)
        emitted = None
        emitted_ts = None
        utterance_closed = False
        while len(self.buffer) >= self.vad_frame_bytes:
            frame = bytes(self.buffer[: self.vad_frame_bytes])
            del self.buffer[: self.vad_frame_bytes]
            if self._silero is not None:
                speech = self._silero_is_speech(frame)
            else:
                speech = self.vad.is_speech(frame, self.sample_rate)
            frame_ms = self.vad_frame_ms
            speech = self._vac_filter(speech)
            # RMS 推定（int16）
            try:
                import numpy as _np
                _i16 = _np.frombuffer(frame, dtype=_np.int16)
                _rms = float(_np.sqrt((_i16.astype(_np.float32) ** 2).mean()) / 32768.0)
            except Exception:
                _rms = 0.0
            if speech:
                self.in_speech = True
                self.last_voice_ms = self.pts_ms + frame_ms
                self.current_chunk.extend(frame)
            else:
                if self.in_speech:
                    self.current_chunk.extend(frame)
            self.pts_ms += frame_ms
            # Auto-tuning window計測
            if self.auto_vad_enable and self.backend == 'silero':
                self._auto_frames += 1
                if speech:
                    self._auto_speech_frames += 1
                self._auto_rms_acc += _rms
                if self.pts_ms >= self._auto_next_eval:
                    # 評価
                    total = max(1, self._auto_frames)
                    ratio = self._auto_speech_frames / total
                    mean_rms = self._auto_rms_acc / total
                    new_thr = self.silero_threshold
                    # ノイズ多めで誤検出: 低RMSだが話中率が高い → しきい値を上げる
                    if mean_rms < self.auto_rms_low and ratio > self.auto_target_high:
                        new_thr = min(self.auto_max_thr, self.silero_threshold + self.auto_step)
                    # 取りこぼし: RMS高いのに話中率が低い → しきい値を下げる
                    elif mean_rms > self.auto_rms_high and ratio < self.auto_target_low:
                        new_thr = max(self.auto_min_thr, self.silero_threshold - self.auto_step)
                    if abs(new_thr - self.silero_threshold) >= 1e-6:
                        self.silero_threshold = new_thr
                        # 次の窓で再評価
                    # silence ms の自動微調整（短片断が多いときは伸ばす）
                    if self.auto_tune_silence:
                        # 直近 current_chunk が短すぎる断片になりがちなら伸ばす
                        # ヒューリスティック: 話中率が高いのに emitted が出にくい → しきい値は適正、silenceを短縮/増加
                        pass
                    # 窓リセット
                    self._auto_frames = 0
                    self._auto_speech_frames = 0
                    self._auto_rms_acc = 0.0
                    self._auto_next_eval = self.pts_ms + self.auto_window_ms
            if self.in_speech and (self.pts_ms - self.last_voice_ms) >= self.utterance_silence_ms:
                # 粗い開始推定（パディング分を含めて移動平均）
                est_len_ms = int(len(self.current_chunk) / 2 / (self.sample_rate / 1000))
                if est_len_ms >= self.min_final_ms:
                    emitted = bytes(self.current_chunk)
                    emitted_ts = (max(0, self.last_voice_ms - est_len_ms), self.last_voice_ms)
                    self.current_chunk.clear()
                    self.in_speech = False
                    utterance_closed = True
                    break
                else:
                    # 短すぎるため破棄
                    self.current_chunk.clear()
                    self.in_speech = False
                    if self.auto_vad_enable and self.auto_tune_silence:
                        # 短片断が続く場合は沈黙しきいを少し増やす
                        self.utterance_silence_ms = min(1500, self.utterance_silence_ms + 100)
            # 長尺発話の強制分割（話し続けている間にも適宜確定出力）
            if self.in_speech:
                chunk_ms = int(len(self.current_chunk) / 2 / (self.sample_rate / 1000))
                if chunk_ms >= self.force_utterance_ms:
                    keep_ms = min(self.force_overlap_ms, chunk_ms // 3)
                    flush_ms = chunk_ms - keep_ms
                    flush_bytes = int(self.sample_rate * (flush_ms / 1000.0)) * 2
                    # 20ms フレーム境界に合わせる
                    flush_bytes -= (flush_bytes % self.vad_frame_bytes)
                    if flush_bytes >= self.vad_frame_bytes:
                        emitted = bytes(self.current_chunk[:flush_bytes])
                        start_ts = self.pts_ms - chunk_ms
                        end_ts = start_ts + flush_ms
                        emitted_ts = (max(0, start_ts), max(end_ts, start_ts))
                        # 末尾オーバーラップを残す
                        self.current_chunk[:] = self.current_chunk[flush_bytes:]
                        # in_speech は継続（区切りは作らない）
                        break
        return emitted, emitted_ts, utterance_closed, self._vac_active


class TranscribeWorker:
    def __init__(self, session_id: str, send_json, opts: Optional[Dict[str, Any]] = None):
        self.session_id = session_id
        self.stop_event = asyncio.Event()
        self.send_json = send_json
        self.opts = opts or {}
        raw_profile = self.opts.get("transcribeProfile", getattr(config, "TRANSCRIBE_PROFILE", "realtime"))
        if isinstance(raw_profile, str):
            profile = raw_profile.lower()
        else:
            profile = str(raw_profile).lower()
        if profile not in PROFILE_DEFAULTS:
            profile = "realtime"
        self.profile = profile
        profile_defaults = PROFILE_DEFAULTS.get(self.profile, PROFILE_DEFAULTS["realtime"])
        # ASR 設定（Parakeet 固定）
        self.model_name = self.opts.get("whisperModel", config.WHISPER_MODEL_NAME)
        self.backend_name = str(self.opts.get("asrBackend", getattr(config, "ASR_BACKEND", "parakeet"))).lower()
        if self.backend_name not in {"parakeet"}:
            logger.info("[ASR] 未対応バックエンド %s を受信したため parakeet に強制変更します", self.backend_name)
            self.backend_name = "parakeet"
        self.lang = self.opts.get("language", config.WHISPER_LANGUAGE)
        if isinstance(self.lang, str):
            self.lang = self.lang.strip()
            if not self.lang:
                self.lang = None
            elif self.lang.lower() == "auto":
                self.lang = None
            else:
                self.lang = self.lang.lower()
        if self.backend_name == 'parakeet_en':
            self.backend_name = 'parakeet'
            if self.lang is None or self.lang == 'ja':
                self.lang = 'en'
        self.model = GlobalASR.get(self.backend_name, language=self.lang)
        # 高精度モード
        opt_high_accuracy = self.opts.get("highAccuracy")
        if opt_high_accuracy is None:
            self.high_accuracy_enabled = (self.profile == "high_accuracy")
        else:
            if isinstance(opt_high_accuracy, str):
                self.high_accuracy_enabled = opt_high_accuracy.strip().lower() in {"1", "true", "yes", "on"}
            else:
                self.high_accuracy_enabled = bool(opt_high_accuracy)
        # チャンクモード
        raw_chunk_mode = self.opts.get("chunkMode", config.CHUNK_MODE)
        chunk_mode = str(raw_chunk_mode).lower() if raw_chunk_mode is not None else "utterance"
        chunk_seconds_override = None
        if chunk_mode in {"1m", "60", "minute", "one-minute"}:
            chunk_mode = "longform"
            chunk_seconds_override = 60
        elif chunk_mode in {"2m", "120", "two-minute"}:
            chunk_mode = "longform"
            chunk_seconds_override = 120
        elif chunk_mode not in {"utterance", "longform"}:
            chunk_mode = "longform" if chunk_mode in {"minutes", "long", "batch"} else "utterance"
        opt_chunk_seconds = self.opts.get("chunkSeconds", chunk_seconds_override if chunk_seconds_override else getattr(config, "LONGFORM_CHUNK_SECONDS", 60))
        try:
            chunk_seconds = int(opt_chunk_seconds)
        except Exception:
            try:
                chunk_seconds = int(float(str(opt_chunk_seconds)))
            except Exception:
                chunk_seconds = getattr(config, "LONGFORM_CHUNK_SECONDS", 60)
        opt_chunk_overlap = self.opts.get("chunkOverlapSeconds", getattr(config, "LONGFORM_OVERLAP_SECONDS", 5))
        try:
            overlap_seconds = float(opt_chunk_overlap)
        except Exception:
            overlap_seconds = float(getattr(config, "LONGFORM_OVERLAP_SECONDS", 5))
        self.chunk_mode = chunk_mode
        self.longform_chunk_ms = max(1000, int(chunk_seconds * 1000))
        self.longform_overlap_ms = max(0, int(overlap_seconds * 1000))
        self.longform_min_speech_ms = int(self.opts.get("longformMinSpeechMs", getattr(config, "LONGFORM_MIN_SPEECH_MS", 1200)))
        self.longform_status_interval_ms = int(self.opts.get("longformStatusIntervalMs", getattr(config, "LONGFORM_STATUS_INTERVAL_MS", 15000)))
        self.longform_accumulator: Optional[SegmentAccumulator] = None
        if self.chunk_mode == "longform":
            self.longform_accumulator = SegmentAccumulator(
                sample_rate=config.SAMPLE_RATE,
                chunk_ms=self.longform_chunk_ms,
                overlap_ms=self.longform_overlap_ms,
                min_speech_ms=self.longform_min_speech_ms,
            )
            longform_queue_depth.set(0)
        queue_maxsize = getattr(config, "AUDIO_QUEUE_MAXSIZE", 256)
        if self.chunk_mode == "longform":
            queue_maxsize = max(queue_maxsize, getattr(config, "LONGFORM_QUEUE_MAXSIZE", queue_maxsize))
            frames_per_chunk = max(1, int(self.longform_chunk_ms / max(1, config.FRAME_MS)))
            queue_maxsize = max(queue_maxsize, frames_per_chunk * 6)
        self.audio_q: asyncio.Queue[Tuple[bytes, int]] = asyncio.Queue(maxsize=queue_maxsize)

        def _int_opt(key: str, default: int) -> int:
            val = self.opts.get(key, default)
            try:
                return int(val)
            except Exception:
                try:
                    return int(float(val))
                except Exception:
                    return default

        utterance_silence_ms = _int_opt("utteranceSilenceMs", profile_defaults.get("utterance_silence_ms", config.UTTERANCE_SILENCE_MS))
        min_final_ms = _int_opt("minFinalMs", profile_defaults.get("min_final_ms", getattr(config, "MIN_FINAL_MS", 800)))
        self.partial_interval_ms = _int_opt("partialIntervalMs", profile_defaults.get("partial_interval_ms", config.PARTIAL_INTERVAL_MS))
        self.beam_size = _int_opt("beamSize", profile_defaults.get("beam_size", 12))
        try:
            self.patience = float(self.opts.get("patience", profile_defaults.get("patience", 1.2)))
        except Exception:
            self.patience = profile_defaults.get("patience", 1.2)
        self.window_seconds = _int_opt("windowSeconds", config.WINDOW_SECONDS)
        self.partial_window_seconds = _int_opt("partialWindowSeconds", getattr(config, "PARTIAL_WINDOW_SECONDS", 3))
        self.window_overlap_seconds = _int_opt("windowOverlapSeconds", config.WINDOW_OVERLAP_SECONDS)

        if self.chunk_mode == "longform":
            base_silence = max(self.longform_min_speech_ms + 800, int(self.longform_chunk_ms * 0.2))
            utterance_silence_ms = max(utterance_silence_ms, base_silence)
            min_final_ms = max(min_final_ms, self.longform_min_speech_ms)

        vad_opts: Dict[str, Any] = {
            "sampleRate": config.SAMPLE_RATE,
            "VAD_BACKEND": self.opts.get("vadBackend", config.VAD_BACKEND),
            "VAD_FRAME_MS": self.opts.get("vadFrameMs", config.VAD_FRAME_MS),
            "VAD_AGGRESSIVENESS": self.opts.get("vadAggressiveness", config.VAD_AGGRESSIVENESS),
            "UTTERANCE_SILENCE_MS": utterance_silence_ms,
            "FORCE_UTTERANCE_MS": self.opts.get("forceUtteranceMs", getattr(config, "FORCE_UTTERANCE_MS", 9000)),
            "FORCE_OVERLAP_MS": self.opts.get("forceOverlapMs", getattr(config, "FORCE_OVERLAP_MS", 1200)),
            "SILERO_THRESHOLD": self.opts.get("sileroThreshold", config.SILERO_THRESHOLD),
            "SILERO_MIN_SILENCE_MS": self.opts.get("sileroMinSilenceMs", config.SILERO_MIN_SILENCE_MS),
            "SILERO_MIN_SPEECH_MS": self.opts.get("sileroMinSpeechMs", config.SILERO_MIN_SPEECH_MS),
            "AUTO_VAD_ENABLE": self.opts.get("autoVadEnable", 1 if getattr(config, 'AUTO_VAD_ENABLE', True) else 0),
            "AUTO_VAD_WINDOW_MS": self.opts.get("autoVadWindowMs", getattr(config, 'AUTO_VAD_WINDOW_MS', 3000)),
            "AUTO_VAD_STEP": self.opts.get("autoVadStep", getattr(config, 'AUTO_VAD_STEP', 0.05)),
            "AUTO_VAD_MIN_THR": self.opts.get("autoVadMinThr", getattr(config, 'AUTO_VAD_MIN_THR', 0.35)),
            "AUTO_VAD_MAX_THR": self.opts.get("autoVadMaxThr", getattr(config, 'AUTO_VAD_MAX_THR', 0.75)),
            "AUTO_VAD_TARGET_LOW": self.opts.get("autoVadTargetLow", getattr(config, 'AUTO_VAD_TARGET_LOW', 0.08)),
            "AUTO_VAD_TARGET_HIGH": self.opts.get("autoVadTargetHigh", getattr(config, 'AUTO_VAD_TARGET_HIGH', 0.60)),
            "AUTO_VAD_RMS_LOW": self.opts.get("autoVadRmsLow", getattr(config, 'AUTO_VAD_RMS_LOW', 0.02)),
            "AUTO_VAD_RMS_HIGH": self.opts.get("autoVadRmsHigh", getattr(config, 'AUTO_VAD_RMS_HIGH', 0.05)),
            "AUTO_VAD_TUNE_SILENCE": self.opts.get("autoVadTuneSilence", 1 if getattr(config, 'AUTO_VAD_TUNE_SILENCE', True) else 0),
            "VAC_ENABLE": self.opts.get("vacEnable", 1 if getattr(config, 'VAC_ENABLE', False) else 0),
            "VAC_MIN_SPEECH_MS": self.opts.get("vacMinSpeechMs", getattr(config, 'VAC_MIN_SPEECH_MS', 220)),
            "VAC_HANGOVER_MS": self.opts.get("vacHangoverMs", getattr(config, 'VAC_HANGOVER_MS', 360)),
            "VAC_MIN_FINAL_MS": self.opts.get("vacMinFinalMs", getattr(config, 'VAC_MIN_FINAL_MS', getattr(config, 'MIN_FINAL_MS', 800))),
        }
        self.punct_split = bool(int(self.opts.get("punctSplit", 0 if not getattr(config, "PUNCT_SPLIT", False) else 1)))
        self.max_history_chars = int(self.opts.get("maxHistoryChars", getattr(config, "MAX_HISTORY_CHARS", 1200)))
        if self.high_accuracy_enabled or self.chunk_mode == "longform":
            self.punct_split = False
        self.segmenter = VADSegmenter({**vad_opts, "MIN_FINAL_MS": min_final_ms})
        self.partial_enabled = self.chunk_mode != "longform"
        self.partial_char_limit = 120 if self.high_accuracy_enabled else None
        self.enable_diar = bool(int(self.opts.get("enableDiarization", 1 if config.ENABLE_DIARIZATION else 0)))
        self.diarizer = GlobalDiarizer.get() if self.enable_diar else None
        self.last_partial_ts = 0
        self.segment_counter = 0
        self.history_text = ""
        self.final_text_by_segment: Dict[str, str] = {}
        self.segment_ts: Dict[str, Tuple[int, int]] = {}
        self.segment_times: Dict[str, float] = {}
        self.prev_in_speech = False
        self._prev_partial_pcm: bytes = b""
        self._partial_acc_text: str = ""
        self.refine_logger = logging.getLogger("REFINE_STAGE")
        try:
            from pathlib import Path
            self.hotwords = HotwordStore((config.DATA_DIR / "_hotwords.json") if hasattr(config, 'DATA_DIR') else Path('data/_hotwords.json'))
        except Exception:
            self.hotwords = None
        self._calib_active = False
        self._calib_end_ms = 0
        self._calib_frames = 0
        self._calib_speech_frames = 0
        self._calib_rms_acc = 0.0
        self.longform_next_status_ts = 0
        self.refiner: Optional[HighAccuracyDecoder] = None
        self.refine_queue: Optional[asyncio.Queue[Optional[RefinementJob]]] = None
        self.refine_task: Optional[asyncio.Task] = None
        self._refine_closed = False
        if self.high_accuracy_enabled:
            self.refiner = HighAccuracyDecoder(self.model, language=self.lang)
            self.refine_queue = asyncio.Queue()
            self.refine_task = asyncio.create_task(self._refine_loop())
            refine_queue_depth.set(0)
        self.streaming_enabled = getattr(self.model, 'supports_streaming', False)
        self._stream_state = None
        if self.streaming_enabled:
            try:
                self._stream_state = self.model.create_stream(language=self.lang, beam_size=self.beam_size, patience=self.patience)
            except Exception as exc:
                logger.warning("streaming backend init failed: %s", exc)
                self.streaming_enabled = False

    async def put_audio(self, pcm16_bytes: bytes, pts_ms: int):
        await self.audio_q.put((pcm16_bytes, pts_ms))

    async def stop(self):
        self.stop_event.set()
        await self.audio_q.put((b"", -1))
        await self._close_refine_queue()
        if self.refine_task and not self.refine_task.done():
            try:
                await self.refine_task
            except Exception:
                pass

    async def start_calibration(self, duration_ms: int = 2000):
        self._calib_active = True
        self._calib_end_ms = self.segmenter.pts_ms + max(500, int(duration_ms))
        self._calib_frames = 0
        self._calib_speech_frames = 0
        self._calib_rms_acc = 0.0
        try:
            await self.send_json({"type": "info", "message": "calibration_started"})
        except Exception:
            pass

    def _next_segment_id(self, ts: Tuple[int, int]) -> str:
        self.segment_counter += 1
        if self.chunk_mode == "longform":
            return f"chunk_{self.segment_counter:04d}"
        return str(self.segment_counter)

    def _refresh_history(self):
        if not self.final_text_by_segment:
            self.history_text = ""
            return
        joined = "".join(self.final_text_by_segment.values())
        if len(joined) > self.max_history_chars:
            self.history_text = joined[-self.max_history_chars :]
        else:
            self.history_text = joined

    async def _send_status(self, stage: str, segment_id: Optional[str], ts: Tuple[int, int], **extra):
        payload = {"type": "status", "stage": stage, "tsStart": ts[0], "tsEnd": ts[1]}
        if segment_id is not None:
            payload["segmentId"] = segment_id
        payload.update(extra)
        try:
            await self.send_json(payload)
        except Exception:
            pass

    def _update_refine_gauge(self):
        if self.refine_queue is not None:
            try:
                refine_queue_depth.set(self.refine_queue.qsize())
            except Exception:
                pass

    def _update_longform_gauge(self):
        if self.longform_accumulator is None:
            return
        depth = 1 if len(self.longform_accumulator.buffer) > 0 else 0
        try:
            longform_queue_depth.set(depth)
        except Exception:
            pass

    async def _send_final(self, segment_id: str, text: str, ts: Tuple[int, int], speaker: Optional[str]):
        message = {
            "type": "final",
            "segmentId": segment_id,
            "text": text,
            "tsStart": ts[0],
            "tsEnd": ts[1],
        }
        if speaker:
            message["speaker"] = speaker
        await self.send_json(message)
        cleaned = text.strip()
        if cleaned:
            self.final_text_by_segment[segment_id] = cleaned
        elif segment_id in self.final_text_by_segment:
            self.final_text_by_segment.pop(segment_id, None)
        self.segment_ts[segment_id] = ts
        self._refresh_history()
        start = self.segment_times.pop(segment_id, None)
        if start is not None:
            duration = max(0.0, time.time() - start)
            try:
                if segment_id.startswith("chunk_"):
                    longform_chunk_latency_seconds.observe(duration)
            except Exception:
                pass

    async def _close_refine_queue(self):
        if self.refine_queue is None or self._refine_closed:
            return
        await self.refine_queue.put(None)
        self._refine_closed = True
        self._update_refine_gauge()

    def _needs_overwrite(self, segment_id: str, original: str, refined: str) -> bool:
        o = (original or "").strip()
        r = (refined or "").strip()
        if o == r:
            return False
        if not o and r:
            return True
        if not r and o:
            # refined が空ならノイズ除去扱いで上書き
            return True
        ratio = difflib.SequenceMatcher(None, o, r).ratio()
        return ratio < 0.9

    async def _emit_final_text(self, text: str, ts: Tuple[int, int], speaker: Optional[str]) -> list[str]:
        pieces: list[str] = []
        if self.punct_split and text:
            puncts = getattr(config, "PUNCT_CHARS", "。．.!！?？")
            buf = ""
            for ch in text:
                buf += ch
                if ch in puncts:
                    pieces.append(buf.strip())
                    buf = ""
            if buf.strip():
                pieces.append(buf.strip())
        else:
            if text.strip():
                pieces.append(text.strip())
        segment_ids: list[str] = []
        for piece in pieces:
            seg_id = self._next_segment_id(ts)
            await self._send_final(seg_id, piece, ts, speaker)
            segment_ids.append(seg_id)
        return segment_ids

    async def _enqueue_refinement(self, segment_id: str, pcm16: bytes, ts: Tuple[int, int], coarse_text: str):
        if not self.high_accuracy_enabled or self.refine_queue is None:
            return
        job = RefinementJob(
            segment_id=segment_id,
            pcm16=pcm16,
            ts=ts,
            coarse_text=coarse_text,
            language=self.lang,
            queued_at=time.time(),
        )
        await self.refine_queue.put(job)
        self.segment_times[segment_id] = job.queued_at
        self.segment_ts[segment_id] = ts
        self._update_refine_gauge()

    async def _maybe_send_longform_status(self):
        if self.chunk_mode != "longform" or self.longform_accumulator is None:
            return
        now = int(time.time() * 1000)
        if now < self.longform_next_status_ts:
            return
        remaining = self.longform_accumulator.remaining_ms()
        if remaining <= 0:
            self.longform_next_status_ts = now + self.longform_status_interval_ms
            return
        await self._send_status("longform_pending", None, (0, 0), remainingMs=remaining)
        self.longform_next_status_ts = now + self.longform_status_interval_ms

    async def _handle_longform_segment(self, pcm16: bytes, ts: Tuple[int, int], utterance_closed: bool):
        if self.longform_accumulator is None:
            return
        chunk = None
        if pcm16:
            chunk = self.longform_accumulator.add(pcm16, ts, force=False)
            await self._maybe_send_longform_status()
            self._update_longform_gauge()
        if chunk:
            audio, chunk_ts = chunk
            await self._run_infer_final(audio, chunk_ts)
            self.longform_next_status_ts = 0
            self._update_longform_gauge()
        if utterance_closed:
            pending = self.longform_accumulator.flush(force=True)
            if pending:
                audio, chunk_ts = pending
                await self._run_infer_final(audio, chunk_ts)
                self.longform_next_status_ts = 0
            self._update_longform_gauge()

    async def _flush_longform_pending(self):
        if self.longform_accumulator is None:
            return
        pending = self.longform_accumulator.flush(force=True)
        if pending:
            await self._run_infer_final(*pending)
            self.longform_next_status_ts = 0
        self._update_longform_gauge()

    async def _refine_loop(self):
        if self.refine_queue is None or self.refiner is None:
            return
        while True:
            job = await self.refine_queue.get()
            if job is None:
                break
            await self._send_status("refining", job.segment_id, job.ts)
            decode_start = time.time()
            refined = await self.refiner.decode(job.pcm16, language=job.language)
            refined = (refined or "").strip()
            if self.hotwords is not None and refined:
                try:
                    refined = boost_hotwords(refined, self.hotwords)
                except Exception:
                    pass
            audio = np_int16_to_float32(pcm16le_bytes_to_np(job.pcm16))
            speaker = None
            if self.diarizer is not None and refined:
                try:
                    speaker = self.diarizer.assign(audio)
                except Exception:
                    speaker = None

            existing_text = self.final_text_by_segment.get(job.segment_id)
            final_text = refined if refined else (job.coarse_text or "")
            if final_text:
                tail = final_text[-1]
                if tail not in TAIL_PUNCT and any(0x3040 <= ord(ch) <= 0x30ff for ch in final_text):
                    final_text = final_text + "。"
            if existing_text is None:
                await self._send_final(job.segment_id, final_text, job.ts, speaker)
            else:
                if self._needs_overwrite(job.segment_id, existing_text, final_text):
                    payload = {
                        "type": "overwrite",
                        "segmentId": job.segment_id,
                        "text": final_text,
                        "tsStart": job.ts[0],
                        "tsEnd": job.ts[1],
                    }
                    if speaker:
                        payload["speaker"] = speaker
                    await self.send_json(payload)
                    if final_text.strip():
                        self.final_text_by_segment[job.segment_id] = final_text.strip()
                    else:
                        self.final_text_by_segment.pop(job.segment_id, None)
                    self.segment_ts[job.segment_id] = job.ts
                    self._refresh_history()
                    self.refine_logger.info(
                        "[REFINE_STAGE] {\"segment\": \"%s\", \"coarse_len\": %d, \"refined_len\": %d}",
                        job.segment_id,
                        len(job.coarse_text or ""),
                        len(final_text),
                    )

            latency_full = max(0.0, time.time() - job.queued_at)
            try:
                refine_latency_seconds.observe(latency_full)
            except Exception:
                pass
            self._update_refine_gauge()
            latency_ms = int((time.time() - decode_start) * 1000)
            await self._send_status("refined", job.segment_id, job.ts, latencyMs=latency_ms)
        try:
            refine_queue_depth.set(0)
        except Exception:
            pass

    async def _run_infer_final(self, pcm16: bytes, ts: Tuple[int, int]):
        audio = np_int16_to_float32(pcm16le_bytes_to_np(pcm16))
        loop = asyncio.get_running_loop()
        if self.streaming_enabled and self._stream_state is not None:
            text = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe_stream(
                    self._stream_state,
                    audio,
                    language=self.lang,
                    beam_size=self.beam_size,
                    patience=self.patience,
                    is_final=True,
                ),
            )
            try:
                self._stream_state = self.model.create_stream(language=self.lang, beam_size=self.beam_size, patience=self.patience)
            except Exception:
                self.streaming_enabled = False
        else:
            text = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    audio,
                    language=self.lang,
                    beam_size=self.beam_size,
                    patience=self.patience,
                    partial=False,
                ),
            )
        if self.hotwords is not None:
            try:
                text = boost_hotwords(text, self.hotwords)
            except Exception:
                pass
        coarse_text = (text or "").strip()
        logger.debug("final len=%.2fs text_chars=%d", len(audio) / 16000.0, len(coarse_text))
        self._prev_partial_pcm = b""
        self._partial_acc_text = ""
        speaker = None
        if self.diarizer is not None and coarse_text:
            try:
                speaker = self.diarizer.assign(audio)
            except Exception:
                speaker = None
        segment_ids = await self._emit_final_text(coarse_text, ts, speaker)
        if self.high_accuracy_enabled and self.refiner and self.refine_queue:
            for seg_id in segment_ids:
                await self._send_status("refining", seg_id, ts)
                piece_text = self.final_text_by_segment.get(seg_id, coarse_text)
                await self._enqueue_refinement(seg_id, pcm16, ts, piece_text)
        if self.chunk_mode == "longform":
            now = time.time()
            for seg_id in segment_ids:
                self.segment_times[seg_id] = now
        return coarse_text

    async def _run_infer_partial(self, pcm16: bytes, ts_end_ms: int):
        if not self.partial_enabled:
            return
        now = int(time.time() * 1000)
        if now - self.last_partial_ts < self.partial_interval_ms:
            return
        self.last_partial_ts = now
        # ストリーミングバックエンドの場合は専用パス
        loop = asyncio.get_running_loop()
        if self.streaming_enabled and self._stream_state is not None:
            audio = np_int16_to_float32(pcm16le_bytes_to_np(pcm16))
            text = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe_stream(
                    self._stream_state,
                    audio,
                    language=self.lang,
                    beam_size=self.beam_size,
                    patience=self.patience,
                    is_final=False,
                ),
            )
            if not isinstance(text, str):
                text = ""
            text = text.strip()
            if not text:
                return
            if self.hotwords is not None:
                try:
                    text = boost_hotwords(text, self.hotwords)
                except Exception:
                    pass
            self._partial_acc_text = text
            self._prev_partial_pcm = pcm16
            await self.send_json(
                {
                    "type": "partial",
                    "seq": 0,
                    "text": text,
                    "tsStart": ts_end_ms - int(len(pcm16) / 2 / (config.SAMPLE_RATE / 1000)),
                    "tsEnd": ts_end_ms,
                }
            )
            return
        # 既存バックエンド: 直前の部分音声がプレフィックスなら差分のみを推論
        if len(pcm16) >= len(self._prev_partial_pcm) and pcm16[: len(self._prev_partial_pcm)] == self._prev_partial_pcm:
            delta = pcm16[len(self._prev_partial_pcm) :]
            if len(delta) == 0:
                # 変化無し
                return
            audio = np_int16_to_float32(pcm16le_bytes_to_np(delta))
            delta_text = await loop.run_in_executor(None, lambda: self.model.transcribe(audio, language=self.lang, beam_size=1, patience=1.0, partial=True))
            if self.hotwords is not None:
                try:
                    delta_text = boost_hotwords(delta_text, self.hotwords)
                except Exception:
                    pass
            # 連結（簡易: スペースや句読点の重複を避ける）
            joiner = "" if (self._partial_acc_text.endswith(tuple(" 　、。.!！?？")) or delta_text.startswith(tuple(" 　、。.!！?？"))) else ""
            self._partial_acc_text = (self._partial_acc_text + joiner + delta_text).strip()
            self._prev_partial_pcm = pcm16
            text = self._partial_acc_text
        else:
            # フル再推論
            audio = np_int16_to_float32(pcm16le_bytes_to_np(pcm16))
            text = await loop.run_in_executor(None, lambda: self.model.transcribe(audio, language=self.lang, beam_size=1, patience=1.0, partial=True))
            if self.hotwords is not None:
                try:
                    text = boost_hotwords(text, self.hotwords)
                except Exception:
                    pass
            self._partial_acc_text = text
            self._prev_partial_pcm = pcm16
        if self.partial_char_limit is not None and text:
            text = text[-self.partial_char_limit :]
        logger.debug("partial len=%.2fs text_chars=%d", len(audio)/16000.0, len(text))
        # CTC系では confidence が取得できないため、簡易フィルタのみ
        if not text.strip():
            return
        await self.send_json(
            {
                "type": "partial",
                "seq": 0,
                "text": text,
                "tsStart": ts_end_ms - int(len(pcm16) / 2 / (config.SAMPLE_RATE / 1000)),
                "tsEnd": ts_end_ms,
            }
        )

    async def run(self):
        acc_bytes = bytearray()
        PROBE_PARTIAL_MS = max(1500, self.partial_interval_ms * 3)
        while True:
            pcm, pts = await self.audio_q.get()
            if pts < 0:
                break
            emitted, ts, closed, _ = self.segmenter.feed(pcm, pts)
            if self._calib_active:
                try:
                    import numpy as _np
                    i16 = _np.frombuffer(pcm, dtype=_np.int16)
                    rms = float(_np.sqrt((i16.astype(_np.float32) ** 2).mean()) / 32768.0)
                except Exception:
                    rms = 0.0
                self._calib_frames += 1
                if self.segmenter.in_speech:
                    self._calib_speech_frames += 1
                self._calib_rms_acc += rms
                if self.segmenter.pts_ms >= self._calib_end_ms:
                    ratio = (self._calib_speech_frames / max(1, self._calib_frames))
                    mean_rms = (self._calib_rms_acc / max(1, self._calib_frames))
                    old_thr = getattr(self.segmenter, 'silero_threshold', 0.5)
                    new_thr = old_thr
                    if mean_rms < getattr(config, 'AUTO_VAD_RMS_LOW', 0.02) and ratio > getattr(config, 'AUTO_VAD_TARGET_HIGH', 0.60):
                        new_thr = min(getattr(config, 'AUTO_VAD_MAX_THR', 0.75), old_thr + getattr(config, 'AUTO_VAD_STEP', 0.05))
                    elif mean_rms > getattr(config, 'AUTO_VAD_RMS_HIGH', 0.05) and ratio < getattr(config, 'AUTO_VAD_TARGET_LOW', 0.08):
                        new_thr = max(getattr(config, 'AUTO_VAD_MIN_THR', 0.35), old_thr - getattr(config, 'AUTO_VAD_STEP', 0.05))
                    try:
                        self.segmenter.silero_threshold = float(new_thr)
                    except Exception:
                        pass
                    self._calib_active = False
                    try:
                        await self.send_json({
                            "type": "calib",
                            "ratio": ratio,
                            "rms": mean_rms,
                            "oldThr": old_thr,
                            "newThr": new_thr,
                            "ts": self.segmenter.pts_ms,
                        })
                    except Exception:
                        pass
            if self.segmenter.in_speech != self.prev_in_speech:
                state = "start" if self.segmenter.in_speech else "end"
                try:
                    await self.send_json({"type": "vad", "state": state, "ts": self.segmenter.pts_ms})
                except Exception:
                    pass
                self.prev_in_speech = self.segmenter.in_speech

            if self.partial_enabled:
                acc_bytes.extend(pcm)
                win_samples = config.SAMPLE_RATE * self.window_seconds
                if len(acc_bytes) // 2 > win_samples:
                    acc_bytes[:] = acc_bytes[-win_samples * 2 :]
                try:
                    if self.segmenter.in_speech:
                        if len(acc_bytes) // 2 > config.SAMPLE_RATE * self.partial_window_seconds:
                            part = bytes(acc_bytes[-config.SAMPLE_RATE * self.partial_window_seconds * 2:])
                        else:
                            part = bytes(acc_bytes)
                        await self._run_infer_partial(part, self.segmenter.pts_ms)
                    else:
                        now = int(time.time() * 1000)
                        if now - self.last_partial_ts >= PROBE_PARTIAL_MS and len(acc_bytes) > 0:
                            await self._run_infer_partial(bytes(acc_bytes), self.segmenter.pts_ms)
                except Exception as e:
                    logger.exception("partial infer failed: %s", e)
            elif self.chunk_mode == "longform":
                await self._maybe_send_longform_status()

            if emitted and ts:
                try:
                    if self.chunk_mode == "longform":
                        await self._handle_longform_segment(emitted, ts, closed)
                    else:
                        await self._run_infer_final(emitted, ts)
                        if self.partial_enabled:
                            acc_bytes.clear()
                except Exception as e:
                    logger.exception("final infer failed: %s", e)
            elif self.chunk_mode == "longform" and closed:
                await self._handle_longform_segment(b"", ts or (self.segmenter.pts_ms, self.segmenter.pts_ms), True)

        if self.chunk_mode == "longform":
            await self._flush_longform_pending()
        elif self.segmenter.in_speech and len(acc_bytes) > 0:
            ts = (self.segmenter.pts_ms - int(len(acc_bytes) / 2 / (config.SAMPLE_RATE / 1000)), self.segmenter.pts_ms)
            try:
                await self._run_infer_final(bytes(acc_bytes), ts)
            except Exception:
                pass
        await self._close_refine_queue()
        if self.refine_task:
            try:
                await self.refine_task
            except Exception:
                pass
