"""Rolling HTTP recognition and HQ through the same configured gateway."""
from ..core.config import settings
from ..speech_cpp_asr import SpeechCppBatchTranscriber
from .whisper_live import WhisperLiveMeeting
from .local_agreement import SAMPLE_RATE


class SpeechCppLiveMeeting(WhisperLiveMeeting):
    backend_name = 'speech_cpp'
    dual_lane_state_key = 'speechCppHttpDualLane'
    realtime_failure_detail = '音声は保存されています。LiteLLMのASRモデルとHTTP接続を確認し、停止処理を再試行してください。'
    hq_lock_name = '.speech-cpp-hq.lock'

    @property
    def update_samples(self):
        # Four decodes per default 5 s window (48/min), leaving room for HQ
        # under the default 60/min request budget. Two RT tracks share it.
        return SAMPLE_RATE * 5 // 4 * len(self.data['tracks'])

    def create_transcriber(self):
        return SpeechCppBatchTranscriber(api_key=settings.openai_api_key,
            base_url=settings.openai_base_url, model=settings.asr_model,
            timeout=settings.asr_api_timeout_seconds)

    def _create_hq_transcriber(self):
        return SpeechCppBatchTranscriber(api_key=settings.openai_api_key,
            base_url=settings.openai_base_url, model=settings.asr_model,
            timeout=settings.asr_high_accuracy_timeout_seconds)
