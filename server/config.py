import os
from pathlib import Path


# 基本設定（環境変数で上書き可）
DATA_DIR = Path(os.getenv("DATA_DIR", "data/transcripts"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# WebSocket
WS_PATH = os.getenv("WS_PATH", "/ws/transcribe")

# オーディオ/ストリーミング設定
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))  # Hz
CHANNELS = int(os.getenv("CHANNELS", 1))
FRAME_MS = int(os.getenv("FRAME_MS", 200))  # クライアントからのフレーム長（目安）
VAD_FRAME_MS = int(os.getenv("VAD_FRAME_MS", 20))  # VAD解析フレーム
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", 2))
UTTERANCE_SILENCE_MS = int(os.getenv("UTTERANCE_SILENCE_MS", 600))  # これ以上の無音で区切り（短め）

# VAD バックエンド
VAD_BACKEND = os.getenv("VAD_BACKEND", "silero")  # "silero" | "webrtc"
# Silero-VAD パラメータ（必要に応じて調整）
SILERO_THRESHOLD = float(os.getenv("SILERO_THRESHOLD", 0.5))  # 0.1〜0.9 目安
SILERO_MIN_SILENCE_MS = int(os.getenv("SILERO_MIN_SILENCE_MS", 500))
SILERO_MIN_SPEECH_MS = int(os.getenv("SILERO_MIN_SPEECH_MS", 150))

# VAD 自動チューニング
AUTO_VAD_ENABLE = os.getenv("AUTO_VAD_ENABLE", "1") == "1"
AUTO_VAD_WINDOW_MS = int(os.getenv("AUTO_VAD_WINDOW_MS", 3000))
AUTO_VAD_STEP = float(os.getenv("AUTO_VAD_STEP", 0.05))
AUTO_VAD_MIN_THR = float(os.getenv("AUTO_VAD_MIN_THR", 0.35))
AUTO_VAD_MAX_THR = float(os.getenv("AUTO_VAD_MAX_THR", 0.75))
AUTO_VAD_TARGET_LOW = float(os.getenv("AUTO_VAD_TARGET_LOW", 0.08))   # 目標話中率の下限
AUTO_VAD_TARGET_HIGH = float(os.getenv("AUTO_VAD_TARGET_HIGH", 0.60))  # 目標話中率の上限
AUTO_VAD_RMS_LOW = float(os.getenv("AUTO_VAD_RMS_LOW", 0.02))   # ~ -34 dBFS 付近
AUTO_VAD_RMS_HIGH = float(os.getenv("AUTO_VAD_RMS_HIGH", 0.05))  # ~ -26 dBFS 付近
AUTO_VAD_TUNE_SILENCE = os.getenv("AUTO_VAD_TUNE_SILENCE", "1") == "1"

# ASR バックエンド
ASR_BACKEND = os.getenv("ASR_BACKEND", "parakeet")  # "parakeet" | "whisper" | "simulwhisper"

# ストリーミング認識（SimulStreaming 相当）
STREAMING_WINDOW_SECONDS = int(os.getenv("STREAMING_WINDOW_SECONDS", 14))
STREAMING_STEP_SECONDS = int(os.getenv("STREAMING_STEP_SECONDS", 4))
STREAMING_HISTORY_MAX = int(os.getenv("STREAMING_HISTORY_MAX", 1200))

# Parakeet-CTC（NeMo）
PARAKEET_MODEL_ID = os.getenv("PARAKEET_MODEL_ID", "nvidia/parakeet-tdt_ctc-0.6b-ja")
PARAKEET_MODEL_ID_EN = os.getenv("PARAKEET_MODEL_ID_EN", "nvidia/parakeet-ctc-1.1b-en")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cuda")  # "cuda" or "cpu"
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "ja")

# Whisper（互換のため残置）
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ja")
WHISPER_TASK = os.getenv("WHISPER_TASK", "transcribe")  # or "translate"

# 部分/確定結果のポリシー
PARTIAL_INTERVAL_MS = int(os.getenv("PARTIAL_INTERVAL_MS", 650))  # 暫定結果の送信間隔
WINDOW_SECONDS = int(os.getenv("WINDOW_SECONDS", 8))
WINDOW_OVERLAP_SECONDS = int(os.getenv("WINDOW_OVERLAP_SECONDS", 2))
PARTIAL_WINDOW_SECONDS = int(os.getenv("PARTIAL_WINDOW_SECONDS", 3))

# 文章処理/信頼度
PUNCT_SPLIT = os.getenv("PUNCT_SPLIT", "0") == "1"  # 句読点での見やすい分割（見た目のみ）
PUNCT_CHARS = os.getenv("PUNCT_CHARS", "。．.!！?？")
PARTIAL_MIN_LOGPROB = float(os.getenv("PARTIAL_MIN_LOGPROB", -1.5))  # 暫定表示の最低信頼度（平均）
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", 1200))
MIN_FINAL_MS = int(os.getenv("MIN_FINAL_MS", 800))

# ウォームアップ
WARMUP_FILE = os.getenv("WARMUP_FILE", "")

# 長尺発話の強制分割（Silero が話し続け判定のままでも定期的に確定出力）
FORCE_UTTERANCE_MS = int(os.getenv("FORCE_UTTERANCE_MS", 9000))  # これ以上続いたら途中で一度確定
FORCE_OVERLAP_MS = int(os.getenv("FORCE_OVERLAP_MS", 1200))      # 次区間へ重ねるオーバーラップ

# 話者分離（簡易）
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "0") == "1"  # 既定OFF
DIAR_THRESHOLD = float(os.getenv("DIAR_THRESHOLD", 0.75))  # 類似度しきい値（cosine）
DIAR_MAX_SPEAKERS = int(os.getenv("DIAR_MAX_SPEAKERS", 8))

# Voice Activity Controller (VAC)
VAC_ENABLE = os.getenv("VAC_ENABLE", "1") == "1"
VAC_MIN_SPEECH_MS = int(os.getenv("VAC_MIN_SPEECH_MS", 220))
VAC_HANGOVER_MS = int(os.getenv("VAC_HANGOVER_MS", 360))
VAC_MIN_FINAL_MS = int(os.getenv("VAC_MIN_FINAL_MS", 700))
