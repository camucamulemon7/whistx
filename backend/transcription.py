# backend/transcription.py
"""
Whisper Large-v3 (Float16) による音声認識モジュール
H200 GPU向け最適化
環境に応じて自動的にモデルサイズとデバイスを調整

VRAM共有: 複数セッションでモデルインスタンスを共有
並列処理: faster-whisperはスレッドセーフ（CTranslate2エンジン）
"""

import os
from faster_whisper import WhisperModel
from typing import List, Dict, Optional, Generator
import numpy as np
import logging

logger = logging.getLogger(__name__)

# 環境変数で設定を上書き可能
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")  # large-v3, medium, small, tiny
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", None)     # cuda, cpu (Noneで自動検出)
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", None)  # float16, int8, float32


class H200Transcriber:
    """
    H200 GPU向け最適化済みWhisperトランスライバー

    VRAM共有: 複数セッションでモデルインスタンスを共有
    並列処理: faster-whisperはスレッドセーフ（CTranslate2エンジン）
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        device_index: int = 0,
    ):
        """
        Whisper Large-v3モデルの初期化

        Args:
            model_size: モデルサイズ (default: large-v3)
            device: デバイス (cuda/cpu)
            compute_type: 計算精度 (float16/int8_float16)
            device_index: GPUデバイスインデックス
        """
        logger.info(f"Loading Whisper model: {model_size}")
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.device_index = device_index

        # Whisperモデルのロード（VRAMに1回だけロードされる）
        # CTranslate2エンジンはスレッドセーフなので、ロック不要
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            device_index=device_index,
            cpu_threads=8,
            num_workers=4,
        )
        logger.info("Whisper model loaded successfully (shared VRAM)")

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "ja",
        initial_prompt: Optional[str] = None,
        beam_size: int = 12,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        temperature: float = 0.0,  # 追加: 確定的出力のため
        best_of: int = 5,            # 追加: 複数サンプリングから最良を選択
    ) -> Generator[Dict, None, None]:
        """
        音声をテキストに変換（並列処理対応 + 高精度化パラメータ）

        CTranslate2エンジンはスレッドセーフなので、
        複数セッションから同時に呼び出し可能（VRAM共有）

        Args:
            audio: 音声データ (float32 numpy array, 16kHz)
            language: 言語コード (ja/en/zhなど)
            initial_prompt: コンテキスト用の初期プロンプト
            beam_size: ビームサーチサイズ（大きいほど精度向上）
            vad_filter: VADフィルタを使用するか
            word_timestamps: 単語レベルのタイムスタンプを返すか
            temperature: サンプリング温度（0=確定的、高いほど多様）
            best_of: サンプリング数から最良を選択

        Yields:
            セグメント情報を含む辞書
        """
        # パラメータ検証
        if temperature < 0 or temperature > 1.0:
            raise ValueError(f"temperature must be between 0 and 1, got {temperature}")

        # ロックなしで並列処理（CTranslate2はスレッドセーフ）
        segments, info = self.model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            temperature=temperature,
            best_of=best_of,
        )

        logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        for segment in segments:
            yield {
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in segment.words
                ] if word_timestamps else []
            }


class TranscriptionContext:
    """
    文脈を維持したトランスクリプション管理
    Initial Promptによる専門用語の認識精度向上
    """

    def __init__(self, history_size: int = 10):
        """
        Args:
            history_size: 保持する履歴の最大数
        """
        self.history: List[str] = []
        self.history_size = history_size

    def add_history(self, text: str) -> None:
        """トランスクリプト履歴を追加"""
        self.history.append(text)
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def get_initial_prompt(self) -> str:
        """Initial Promptを生成"""
        return "\n".join(self.history)

    def clear(self) -> None:
        """履歴をクリア"""
        self.history.clear()


# グローバルインスタンス（シングルトンパターン）
_transcriber_instance: Optional[H200Transcriber] = None


def get_transcriber(
    model_size: str = None,
    device: str = None,
    compute_type: str = None,
) -> H200Transcriber:
    """
    トランスライバーのシングルトンインスタンスを取得

    環境変数または引数で設定を上書き可能

    Args:
        model_size: モデルサイズ (default: 環境変数 WHISPER_MODEL or "large-v3")
        device: デバイス (default: 環境変数 WHISPER_DEVICE or "cuda")
        compute_type: 計算精度 (default: 環境変数 WHISPER_COMPUTE_TYPE or "float16")

    Returns:
        H200Transcriberインスタンス
    """
    global _transcriber_instance

    # 環境変数または引数から設定を取得
    actual_model_size = model_size or WHISPER_MODEL
    actual_device = device or WHISPER_DEVICE
    actual_compute_type = compute_type or WHISPER_COMPUTE_TYPE

    # デバイスの自動検出（環境変数で指定がない場合）
    if actual_device is None:
        try:
            import torch
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {actual_device}")
        except ImportError:
            actual_device = "cpu"
            logger.info("PyTorch not available, using CPU")

    # 計算精度の自動設定（デバイスに応じて）
    if actual_compute_type is None:
        if actual_device == "cuda":
            actual_compute_type = "float16"
        else:
            actual_compute_type = "int8"  # CPUではint8を使う

    if _transcriber_instance is None:
        _transcriber_instance = H200Transcriber(
            model_size=actual_model_size,
            device=actual_device,
            compute_type=actual_compute_type,
        )

    return _transcriber_instance
