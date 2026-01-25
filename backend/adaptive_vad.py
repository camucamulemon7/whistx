# backend/adaptive_vad.py
"""
適応的VAD (Voice Activity Detection)
環境ノイズに応じて閾値を自動調整
"""

import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from silero_vad_utils import get_speech_timestamps
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False
    logger.warning("Silero VAD not available, using fallback")


class AdaptiveVAD:
    """
    環境ノイズに適応するVAD
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_size_ms: int = 500,
        initial_threshold: float = 0.5,
    ):
        """
        Args:
            sample_rate: サンプリングレート
            window_size_ms: 分析ウィンドウサイズ (ms)
            initial_threshold: 初期VAD閾値
        """
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_size_ms / 1000)
        self.threshold = initial_threshold
        self.noise_floor = 0.0

        # Silero VADモデル
        self.model = None
        if SILERO_AVAILABLE:
            try:
                # Silero VADモデルをロード
                self.model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                logger.info("Silero VAD model loaded")
            except Exception as e:
                logger.error(f"Failed to load Silero VAD: {e}")
                self.model = None

    def auto_calibrate(self, noise_sample: np.ndarray) -> float:
        """
        環境ノイズに応じて閾値を自動調整

        Args:
            noise_sample: ノイズサンプル (float32 numpy array)

        Returns:
            調整後の閾値
        """
        # RMSレベルを計算
        rms = np.sqrt(np.mean(noise_sample ** 2))
        self.noise_floor = rms

        # ノイズレベルに応じて閾値を調整
        # 基本閾値0.3 + ノイズフロアの2倍
        self.threshold = 0.3 + (self.noise_floor * 2)

        # 閾値を0.1〜0.9の範囲に制限
        self.threshold = np.clip(self.threshold, 0.1, 0.9)

        logger.info(f"VAD calibrated: noise_floor={rms:.4f}, threshold={self.threshold:.4f}")

        return self.threshold

    def detect_speech(self, audio_chunk: bytes) -> bool:
        """
        音声チャンクのVAD判定

        Args:
            audio_chunk: 音声データ (PCM16 bytes)

        Returns:
            音声が含まれている場合はTrue
        """
        if self.model is None:
            # Silero VADが利用できない場合、簡易エネルギーベース判定
            return self._energy_based_detection(audio_chunk)

        # PCM16をfloat32に変換
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

        # Silero VADで判定
        try:
            speech_dict = get_speech_timestamps(
                audio,
                self.model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate,
            )
            return len(speech_dict) > 0
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return self._energy_based_detection(audio_chunk)

    def _energy_based_detection(self, audio_chunk: bytes) -> bool:
        """
        エネルギーベースの簡易VAD判定（フォールバック）

        Args:
            audio_chunk: 音声データ (PCM16 bytes)

        Returns:
            音声が含まれている場合はTrue
        """
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio ** 2))

        # ノイズフロアより3dB以上大きい場合は音声と判定
        return rms > (self.noise_floor * 1.4)

    def get_speech_segments(
        self,
        audio: np.ndarray,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> List[Dict]:
        """
        音声区間の検出

        Args:
            audio: 音声データ (float32 numpy array)
            min_speech_duration_ms: 最小音声継続時間
            min_silence_duration_ms: 最小無音継続時間

        Returns:
            音声区間のリスト [{"start": ms, "end": ms}, ...]
        """
        if self.model is None:
            return []

        try:
            speech_segments = get_speech_timestamps(
                audio,
                self.model,
                threshold=self.threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
            )
            return speech_segments
        except Exception as e:
            logger.error(f"Speech segment detection error: {e}")
            return []


# グローバルインスタンス
_vad_instance: Optional[AdaptiveVAD] = None


def get_vad(
    sample_rate: int = 16000,
    window_size_ms: int = 500,
    initial_threshold: float = 0.5,
) -> AdaptiveVAD:
    """
    VADのシングルトンインスタンスを取得

    Args:
        sample_rate: サンプリングレート
        window_size_ms: 分析ウィンドウサイズ
        initial_threshold: 初期閾値

    Returns:
        AdaptiveVADインスタンス
    """
    global _vad_instance

    if _vad_instance is None:
        _vad_instance = AdaptiveVAD(
            sample_rate=sample_rate,
            window_size_ms=window_size_ms,
            initial_threshold=initial_threshold,
        )

    return _vad_instance
