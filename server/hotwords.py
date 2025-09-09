from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from rapidfuzz import fuzz, process


class HotwordStore:
    def __init__(self, path: Path):
        self.path = path
        self.words: List[str] = []
        self._mtime = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        if not self.path.exists():
            self.words = []
            self._mtime = 0.0
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.words = [w.strip() for w in data.get("words", []) if isinstance(w, str) and w.strip()]
            self._mtime = self.path.stat().st_mtime
        except Exception:
            self.words = []

    def maybe_reload(self):
        try:
            m = self.path.stat().st_mtime
        except FileNotFoundError:
            m = 0.0
        if m != self._mtime:
            self.load()

    def save(self, words: List[str]):
        words = [w.strip() for w in words if isinstance(w, str) and w.strip()]
        self.words = words
        self.path.write_text(json.dumps({"words": words}, ensure_ascii=False, indent=2), encoding="utf-8")
        self._mtime = self.path.stat().st_mtime


def boost_hotwords(text: str, store: HotwordStore, threshold: int = 90) -> str:
    """軽量な後段補正: 出力テキスト内の曖昧一致を指定ホットワードに置換する。
    threshold: 0-100 の類似度。日本語は 85-92 辺りが妥当。
    """
    if not text or not store.words:
        return text
    store.maybe_reload()
    out = text
    # 単純にテキストをウィンドウ分割して近似一致を置換
    # 語境界の曖昧さがあるため、長いホットワードから順に置換
    for hw in sorted(store.words, key=len, reverse=True):
        # 既に完全一致している場合はスキップ
        if hw in out:
            continue
        # スライド窓で候補抽出（長さ±40%）
        L = max(2, len(hw))
        min_len = max(2, int(L * 0.6))
        max_len = int(L * 1.4) + 1
        i = 0
        changed = False
        while i < len(out):
            for wlen in range(max_len, min_len - 1, -1):
                j = i + wlen
                if j > len(out):
                    continue
                cand = out[i:j]
                score = fuzz.ratio(cand, hw)
                if score >= threshold:
                    out = out[:i] + hw + out[j:]
                    i += len(hw)
                    changed = True
                    break
            i += 1 if not changed else 0
        # 次のホットワードへ
    return out

