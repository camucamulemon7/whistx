"""Long-lived resources passed explicitly to realtime coordinators."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable
from fastapi import WebSocket

from ..asr import SessionTranscriber
from ..audio_pipeline import AudioPreprocessor
from ..diarizer import PyannoteSpeakerDiarizer
from ..langfuse_observer import LangfuseObserver
from ..summarizer import OpenAISummarizer
from ..services.oidc_flow import OIDCFlow


@dataclass
class RuntimeResources:
    oidc_flow: OIDCFlow = field(default_factory=OIDCFlow)
    transcriber_factory: Callable[[], SessionTranscriber] | None = None
    audio_preprocessor: AudioPreprocessor | None = None
    summarizer: OpenAISummarizer | None = None
    proofreader: OpenAISummarizer | None = None
    diarizer: PyannoteSpeakerDiarizer | None = None
    observer: LangfuseObserver | None = None
    active_sockets: set[WebSocket] = field(default_factory=set)
    cleanup_task: asyncio.Task | None = None
    event_loop_monitor_task: asyncio.Task | None = None
