"""Compatibility names for the shared conservative PCM gate."""
from .pcm_gate import pcm_speech_bounds as whisper_speech_bounds, prepare_pcm_wav as prepare_whisper_wav

__all__ = ["whisper_speech_bounds", "prepare_whisper_wav"]
