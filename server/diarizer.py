from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier

from . import config


@dataclass
class SpeakerCentroid:
    id: str
    emb: np.ndarray
    count: int = 1


class GlobalDiarizer:
    _inst: Optional["OnlineDiarizer"] = None

    @classmethod
    def get(cls) -> "OnlineDiarizer":
        if cls._inst is None:
            cls._inst = OnlineDiarizer()
        return cls._inst


class OnlineDiarizer:
    def __init__(self):
        # cuDNN 不整合を回避するため、デフォルトは CPU を強制
        # （Whisper 本体は CTranslate2 の CUDA を使用し、PyTorch は CPU のみ使用）
        device = "cpu"
        try:
            torch.backends.cudnn.enabled = False
        except Exception:
            pass
        try:
            self.clf = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device},
                savedir=os.path.join(os.getenv("HF_HOME", "./.cache"), "spkrec-ecapa-voxceleb"),
            )
        except Exception:
            # GPU 初期化に失敗した場合は CPU で再初期化
            self.clf = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"},
                savedir=os.path.join(os.getenv("HF_HOME", "./.cache"), "spkrec-ecapa-voxceleb"),
            )
        self.speakers: List[SpeakerCentroid] = []
        self.next_id = 1

    def _embed(self, wav: np.ndarray) -> np.ndarray:
        # wav: float32 [-1,1], mono
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        t = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
        with torch.no_grad():
            emb = self.clf.encode_batch(t).squeeze(0).squeeze(0).cpu().numpy()
        # L2正規化
        norm = np.linalg.norm(emb) + 1e-10
        return emb / norm

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def assign(self, wav: np.ndarray) -> str:
        """埋め込みを計算し、既存話者に割り当て。しきい値未満なら新規話者を作成。
        戻り値: ラベル（"S1" など）
        """
        emb = self._embed(wav)
        if not self.speakers:
            sid = f"S{self.next_id}"; self.next_id += 1
            self.speakers.append(SpeakerCentroid(sid, emb, 1))
            return sid

        # 類似度最大の話者を検索
        scores = [self._cosine(sp.emb, emb) for sp in self.speakers]
        j = int(np.argmax(scores))
        if scores[j] >= config.DIAR_THRESHOLD and self.speakers:
            # セントロイドを更新（移動平均）
            sp = self.speakers[j]
            n = sp.count + 1
            sp.emb = (sp.emb * sp.count + emb) / n
            sp.count = n
            return sp.id
        else:
            if len(self.speakers) >= config.DIAR_MAX_SPEAKERS:
                # 上限時は最も近い話者に割り当て
                sp = self.speakers[j]
                n = sp.count + 1
                sp.emb = (sp.emb * sp.count + emb) / n
                sp.count = n
                return sp.id
            sid = f"S{self.next_id}"; self.next_id += 1
            self.speakers.append(SpeakerCentroid(sid, emb, 1))
            return sid
