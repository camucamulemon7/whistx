from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
import logging
import threading

import numpy as np


logger = logging.getLogger(__name__)


class ASRBackend(ABC):
    supports_streaming: bool = False

    @abstractmethod
    def transcribe(self, audio_f32_mono_16k: np.ndarray, language: Optional[str] = None,
                   beam_size: int = 1, patience: float = 1.0, partial: bool = False) -> str:
        ...

    def create_stream(self, language: Optional[str] = None, **kwargs):
        raise NotImplementedError("Streaming is not supported by this backend")

    def transcribe_stream(self, stream_state, audio_chunk: np.ndarray,
                          language: Optional[str] = None,
                          beam_size: int = 1, patience: float = 1.0,
                          is_final: bool = False) -> str:
        raise NotImplementedError("Streaming is not supported by this backend")


class ParakeetCTCBackend(ASRBackend):
    def __init__(self, model_id: str, device: str = "cuda", hf_token: Optional[str] = None):
        # 遅延ロード（importコストを初回に限定）
        from nemo.collections.asr.models import EncDecCTCModelBPE

        self._EncDecCTCModelBPE = EncDecCTCModelBPE
        self.model_id = model_id
        self.device = device if device in ("cuda", "cpu") else "cpu"
        self._hf_token = hf_token or os.getenv("HF_AUTH_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_HUB_TOKEN")
        logger.info("[ParakeetCTC] モデル初期化開始: %s (device=%s)", model_id, self.device)
        try:
            extra_args: Dict[str, Any] = {}
            if self._hf_token:
                extra_args["token"] = self._hf_token
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", self._hf_token)
                os.environ.setdefault("HF_HUB_TOKEN", self._hf_token)
            self.model = self._EncDecCTCModelBPE.from_pretrained(model_name=model_id, **extra_args)
        except Exception as e:
            logger.exception("[ParakeetCTC] モデル取得に失敗: %s", e)
            raise
        logger.info("[ParakeetCTC] モデル取得完了: %s", model_id)
        try:
            self.model = self.model.to(self.device)
        except Exception:
            pass
        self.model.eval()
        try:
            self.model.freeze()
        except Exception:
            logger.debug("Parakeet model freeze skipped", exc_info=True)
        self._lock = threading.RLock()
        # 高速一時ファイル領域（RAMディスク）があれば利用
        ramdir = "/dev/shm"
        self.tmp_dir = ramdir if os.path.isdir(ramdir) else None
        self._default_decoder_type = "greedy"
        self._default_decoder_params: Optional[Dict[str, Any]] = None
        try:
            decoding_cfg = getattr(self.model.cfg, "decoding", None)
            if decoding_cfg is not None:
                self._default_decoder_type = getattr(decoding_cfg, "strategy", "greedy")
                params = getattr(decoding_cfg, self._default_decoder_type, None)
                if params is not None:
                    if hasattr(params, "to_dict"):
                        params = params.to_dict()  # type: ignore
                    elif hasattr(params, "__dict__"):
                        params = dict(params.__dict__)
                if isinstance(params, dict):
                    self._default_decoder_params = params
        except Exception:
            logger.debug("Parakeet decoder metadata capture failed", exc_info=True)

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

    def _apply_decoder_options(self, options: Optional[Dict[str, Any]]) -> bool:
        if not options:
            return False
        # 現行 NeMo (23.x 系) の change_decoding_strategy は decode_params 等のキーワードを
        # 受け付けず、cfg の書き換えが必要になるケースが多い。安全策として警告のみ出し
        # 既定値のまま利用する。
        logger.info("[ParakeetCTC] decoder options ignored (unsupported by current NeMo API): %s", options)
        return False

    def _restore_decoder(self) -> None:
        try:
            if self._default_decoder_params:
                self.model.change_decoding_strategy(decoder_type=self._default_decoder_type, decode_params=self._default_decoder_params)
            else:
                self.model.change_decoding_strategy(decoder_type=self._default_decoder_type)
        except Exception:
            logger.debug("Failed to restore decoder strategy", exc_info=True)

    def transcribe(self, audio_f32_mono_16k: np.ndarray, language: Optional[str] = None,
                   beam_size: int = 1, patience: float = 1.0, partial: bool = False,
                   decoder_options: Optional[Dict[str, Any]] = None) -> str:
        # NeMo の transcribe はファイル入力が扱いやすいため、一時WAVに出力
        # 16kHz mono float32 を 16-bit PCM で保存
        import soundfile as sf
        with self._lock:
            params_changed = self._apply_decoder_options(decoder_options)
            with tempfile.NamedTemporaryFile(dir=self.tmp_dir, suffix='.wav', delete=True) as tmp:
                sf.write(tmp.name, audio_f32_mono_16k, 16000, subtype='PCM_16')
                try:
                    hyps = self.model.transcribe([tmp.name], batch_size=1)
                except TypeError:
                    hyps = self.model.transcribe(paths2audio_files=[tmp.name], batch_size=1)
            if params_changed:
                self._restore_decoder()
        if not hyps:
            return ""
        return self._to_text(hyps)


class WhisperBackend(ASRBackend):
    def __init__(self, model_name: str, device: str = "cuda", compute_type: str = "float16"):
        from faster_whisper import WhisperModel

        logger.info("[Whisper] モデル初期化開始: %s (device=%s, compute=%s)", model_name, device, compute_type)
        try:
            self.model = WhisperModel(model_size_or_path=model_name, device=device, compute_type=compute_type)
        except Exception as e:
            logger.exception("[Whisper] モデル取得に失敗: %s", e)
            raise
        logger.info("[Whisper] モデル初期化完了: %s", model_name)

    def transcribe(self, audio_f32_mono_16k: np.ndarray, language: Optional[str] = None,
                   beam_size: int = 1, patience: float = 1.0, partial: bool = False) -> str:
        segments, _ = self.model.transcribe(audio=audio_f32_mono_16k, language=language,
                                            beam_size=beam_size, patience=patience,
                                            temperature=0.0,
                                            condition_on_previous_text=not partial,
                                            no_speech_threshold=0.6)
        return "".join(s.text for s in segments)


class SimulWhisperBackend(WhisperBackend):
    supports_streaming = True

    class _StreamState:
        __slots__ = ("buffer", "last_text", "since_last")

        def __init__(self):
            self.buffer = np.zeros(0, dtype=np.float32)
            self.last_text = ""
            self.since_last = 0

    def __init__(self, model_name: str, device: str = "cuda", compute_type: str = "float16",
                 window_seconds: int = 14, step_seconds: int = 4, history_max: int = 1200):
        super().__init__(model_name=model_name, device=device, compute_type=compute_type)
        self.window_seconds = max(4, int(window_seconds))
        self.step_seconds = max(1, int(step_seconds))
        self.history_max = max(200, int(history_max))

    def create_stream(self, language: Optional[str] = None, **kwargs):
        return SimulWhisperBackend._StreamState()

    def _run_decode(self, audio: np.ndarray, language: Optional[str], beam_size: int, patience: float) -> str:
        segments, _ = self.model.transcribe(
            audio=audio,
            language=language,
            beam_size=beam_size,
            patience=patience,
            temperature=0.0,
            condition_on_previous_text=True,
            no_speech_threshold=0.4,
        )
        return "".join(seg.text for seg in segments)

    def transcribe_stream(self, stream_state, audio_chunk: np.ndarray,
                          language: Optional[str] = None,
                          beam_size: int = 1, patience: float = 1.0,
                          is_final: bool = False) -> str:
        if not isinstance(stream_state, SimulWhisperBackend._StreamState):
            raise ValueError("Invalid stream state")
        if audio_chunk.size == 0:
            return ""
        # 追加しつつウィンドウ制約
        stream_state.buffer = np.concatenate((stream_state.buffer, audio_chunk))
        stream_state.since_last += audio_chunk.shape[0]
        max_samples = self.window_seconds * 16000
        if stream_state.buffer.shape[0] > max_samples:
            stream_state.buffer = stream_state.buffer[-max_samples:]

        step_samples = self.step_seconds * 16000
        if not is_final and stream_state.last_text and stream_state.since_last < step_samples:
            return ""

        decoded = self._run_decode(stream_state.buffer, language, beam_size, patience)
        decoded = decoded.strip()
        if not decoded:
            return ""
        if len(decoded) > self.history_max:
            decoded = decoded[-self.history_max:]

        delta = decoded
        prev = stream_state.last_text
        if prev:
            if decoded.startswith(prev):
                delta = decoded[len(prev):]
            else:
                import difflib
                matcher = difflib.SequenceMatcher(a=prev, b=decoded)
                match = matcher.find_longest_match(0, len(prev), 0, len(decoded))
                start = match.b + match.size
                delta = decoded[start:]
        if is_final:
            stream_state.buffer = np.zeros(0, dtype=np.float32)
            stream_state.last_text = ""
            stream_state.since_last = 0
            return decoded
        stream_state.last_text = decoded
        stream_state.since_last = 0
        return delta.strip()


def create_backend(cfg, backend_name: Optional[str] = None, language: Optional[str] = None) -> ASRBackend:
    backend = backend_name or getattr(cfg, 'ASR_BACKEND', 'parakeet')
    lang = (language or '').lower()
    logger.info("[ASR] バックエンド初期化: %s (language=%s)", backend, lang or '<default>')
    use_parakeet = backend == 'parakeet'
    model_id = getattr(cfg, 'PARAKEET_MODEL_ID', 'parakeet')
    if use_parakeet and lang.startswith('en'):
        model_id = getattr(cfg, 'PARAKEET_MODEL_ID_EN', getattr(cfg, 'PARAKEET_MODEL_ID', 'parakeet'))
        if not getattr(cfg, 'ENABLE_PARAKEET_EN', False) and not getattr(cfg, 'HF_AUTH_TOKEN', ''):
            logger.info("[ASR] English Parakeet disabled or token missing. Using Whisper backend instead.")
            use_parakeet = False
    if use_parakeet:
        try:
            return ParakeetCTCBackend(model_id=model_id, device=getattr(cfg, 'ASR_DEVICE', 'cuda'), hf_token=getattr(cfg, 'HF_AUTH_TOKEN', ''))
        except Exception as exc:
            logger.exception("[ASR] Parakeet backend init failed (model=%s). Falling back to Whisper backend.", model_id)
            use_parakeet = False
            backend = 'whisper'
    if backend == 'simulwhisper':
        return SimulWhisperBackend(
            model_name=cfg.WHISPER_MODEL_NAME,
            device=cfg.WHISPER_DEVICE,
            compute_type=cfg.WHISPER_COMPUTE_TYPE,
            window_seconds=getattr(cfg, 'STREAMING_WINDOW_SECONDS', 14),
            step_seconds=getattr(cfg, 'STREAMING_STEP_SECONDS', 4),
            history_max=getattr(cfg, 'STREAMING_HISTORY_MAX', 1200),
        )
    return WhisperBackend(model_name=cfg.WHISPER_MODEL_NAME, device=cfg.WHISPER_DEVICE, compute_type=cfg.WHISPER_COMPUTE_TYPE)
