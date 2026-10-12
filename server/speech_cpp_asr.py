"""speech.cpp HTTP transcription request profile through an OpenAI-compatible gateway."""
from __future__ import annotations
import io
import wave

from .asr import ASRChunkResult
from .qwen_asr import QwenBatchTranscriber, response_text
from .pcm_gate import prepare_pcm_wav
from .core.config import settings


def speech_response_text(response, *, allow_empty=False):
    # LiteLLM preserves provider headers with the llm_provider- prefix.
    # Check both so a native model limit cannot become a successful transcript.
    stops = [response.headers.get(name) for name in ('X-Speech-Stop', 'llm_provider-X-Speech-Stop')]
    if any(stop is not None and stop != 'complete' for stop in stops):
        raise ValueError('speech_cpp_batch_model_limit')
    return response_text(response.json(), allow_empty=allow_empty)


def speech_batch_request(audio: bytes, *, mime_type: str, model: str, context: str,
                         priority: int = 0, language: str = 'auto'):
    data = {'model': model, 'response_format': 'json'}
    if context:
        data['prompt'] = context
    if language and language != 'auto':
        data['language'] = language
    return '/audio/transcriptions', {'data': data, 'files': {'file': ('audio.wav', audio, mime_type)}}


class SpeechCppBatchTranscriber(QwenBatchTranscriber):
    def __init__(self, *, api_key, timeout=None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.client.timeout = timeout if timeout is not None else settings.asr_high_accuracy_timeout_seconds
        if not api_key:
            self.client.headers.pop('Authorization', None)

    def transcribe_chunk(self, audio_bytes: bytes, *, mime_type: str, language: str | None,
                         prompt: str | None, temperature: float = 0, pcm_bounds=None, **kwargs) -> ASRChunkResult:
        # Native speech.cpp accepts WAV only, including legacy WebM recordings
        # submitted for manual refinement or with preprocessing disabled.
        try:
            with wave.open(io.BytesIO(audio_bytes)) as wav:
                is_wav = wav.getcomptype() == 'NONE'
        except (wave.Error, EOFError):
            is_wav = False
        if not is_wav:
            from .audio_pipeline import AudioPreprocessor
            audio_bytes = AudioPreprocessor(ffmpeg_bin=settings.ffmpeg_bin, sample_rate=16000,
                overlap_ms=0, enabled=False).convert_to_wav(audio_bytes=audio_bytes, mime_type=mime_type)
        mime_type = 'audio/wav'
        audio_bytes, _, silent = prepare_pcm_wav(audio_bytes, mime_type, bounds_hint=pcm_bounds)
        if silent:
            return ASRChunkResult(text='', start_ms=None, end_ms=None, silence_detected=True)
        endpoint, options = speech_batch_request(audio_bytes, mime_type=mime_type, model=self.model,
                                                 context=prompt or '', language=language or 'auto')
        response = self.client.post(self.base_url + endpoint, **options)
        response.raise_for_status()
        return ASRChunkResult(text=speech_response_text(response, allow_empty=True), start_ms=None, end_ms=None)
