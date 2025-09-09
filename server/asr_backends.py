from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import numpy as np


class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_f32_mono_16k: np.ndarray, language: Optional[str] = None,
                   beam_size: int = 1, patience: float = 1.0, partial: bool = False) -> str:
        ...


class ParakeetCTCBackend(ASRBackend):
    def __init__(self, model_id: str, device: str = "cuda"):
        # 遅延ロード（importコストを初回に限定）
        from nemo.collections.asr.models import EncDecCTCModelBPE
        self._EncDecCTCModelBPE = EncDecCTCModelBPE
        self.model_id = model_id
        self.device = device if device in ("cuda", "cpu") else "cpu"
        self.model = self._EncDecCTCModelBPE.from_pretrained(model_name=model_id)
        try:
            self.model = self.model.to(self.device)
        except Exception:
            pass
        self.model.eval()
        # 高速一時ファイル領域（RAMディスク）があれば利用
        ramdir = "/dev/shm"
        self.tmp_dir = ramdir if os.path.isdir(ramdir) else None

    def _to_text(self, obj) -> str:
        # 抽象的に NeMo の戻り値から text を抽出
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        # Hypothesis オブジェクトなど
        txt = getattr(obj, 'text', None)
        if isinstance(txt, str):
            return txt
        if isinstance(obj, dict):
            for k in ('text', 'pred_text', 'prediction'):
                v = obj.get(k)
                if isinstance(v, str):
                    return v
        if isinstance(obj, (list, tuple)):
            # 最初に見つかった非空テキストを返す
            for it in obj:
                s = self._to_text(it)
                if s:
                    return s
            return ""
        return ""

    def transcribe(self, audio_f32_mono_16k: np.ndarray, language: Optional[str] = None,
                   beam_size: int = 1, patience: float = 1.0, partial: bool = False) -> str:
        # NeMo の transcribe はファイル入力が扱いやすいため、一時WAVに出力
        # 16kHz mono float32 を 16-bit PCM で保存
        import soundfile as sf
        with tempfile.NamedTemporaryFile(dir=self.tmp_dir, suffix='.wav', delete=True) as tmp:
            sf.write(tmp.name, audio_f32_mono_16k, 16000, subtype='PCM_16')
            # NeMo の API 差分に対応（RNNT/CTCで引数名が異なることがある）
            try:
                hyps = self.model.transcribe([tmp.name], batch_size=1)
            except TypeError:
                hyps = self.model.transcribe(paths2audio_files=[tmp.name], batch_size=1)
        if not hyps:
            return ""
        return self._to_text(hyps)


class WhisperBackend(ASRBackend):
    def __init__(self, model_name: str, device: str = "cuda", compute_type: str = "float16"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size_or_path=model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_f32_mono_16k: np.ndarray, language: Optional[str] = None,
                   beam_size: int = 1, patience: float = 1.0, partial: bool = False) -> str:
        segments, _ = self.model.transcribe(audio=audio_f32_mono_16k, language=language,
                                            beam_size=beam_size, patience=patience,
                                            temperature=0.0,
                                            condition_on_previous_text=not partial,
                                            no_speech_threshold=0.6)
        return "".join(s.text for s in segments)


def create_backend(cfg) -> ASRBackend:
    backend = getattr(cfg, 'ASR_BACKEND', 'parakeet')
    if backend == 'parakeet':
        return ParakeetCTCBackend(model_id=cfg.PARAKEET_MODEL_ID, device=getattr(cfg, 'ASR_DEVICE', 'cuda'))
    else:
        return WhisperBackend(model_name=cfg.WHISPER_MODEL_NAME, device=cfg.WHISPER_DEVICE, compute_type=cfg.WHISPER_COMPUTE_TYPE)
