from __future__ import annotations

import base64
import json
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

from websockets.sync.client import connect

from .asr import ASRChunkResult


logger = logging.getLogger(__name__)


class VoxtralRealtimeTranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        ffmpeg_bin: str,
        sample_rate: int = 16_000,
    ):
        if not api_key:
            raise RuntimeError("ASR_API_KEY (or OPENAI_API_KEY) is not set")
        if not base_url:
            raise RuntimeError("ASR_BASE_URL (or OPENAI_BASE_URL) is not set for realtime ASR")
        if shutil.which(ffmpeg_bin) is None:
            raise RuntimeError(f"{ffmpeg_bin} is not installed")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.ffmpeg_bin = ffmpeg_bin
        self.sample_rate = sample_rate
        self._connection = None
        self._lock = threading.Lock()
        self._session_ready = False

    def transcribe_chunk(
        self,
        audio_bytes: bytes,
        *,
        mime_type: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
    ) -> ASRChunkResult:
        del language
        del prompt
        del temperature

        pcm_audio = self._decode_audio_to_pcm(audio_bytes=audio_bytes, mime_type=mime_type)
        if not pcm_audio:
            return ASRChunkResult(text="", start_ms=0, end_ms=None)

        payload = base64.b64encode(pcm_audio).decode("ascii")

        with self._lock:
            self._ensure_connection()
            self._send_event({"type": "input_audio_buffer.append", "audio": payload})
            self._send_event({"type": "input_audio_buffer.commit", "final": True})
            text = self._wait_for_transcript()
            return ASRChunkResult(text=text.strip(), start_ms=0, end_ms=None)

    def close(self) -> None:
        with self._lock:
            if self._connection is None:
                return
            try:
                self._connection.close()
            except Exception:  # noqa: BLE001
                logger.debug("failed to close realtime connection", exc_info=True)
            finally:
                self._connection = None
                self._session_ready = False

    def _ensure_connection(self) -> None:
        if self._connection is not None:
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        self._connection = connect(
            self._realtime_url(),
            additional_headers=headers,
            open_timeout=20,
            close_timeout=5,
            max_size=16 * 1024 * 1024,
        )
        self._send_event({"type": "session.update", "model": self.model})
        self._send_event({"type": "input_audio_buffer.commit"})
        self._session_ready = True
        logger.info("voxtral realtime session connected: model=%s", self.model)

    def _wait_for_transcript(self, *, timeout_sec: float = 45.0) -> str:
        deadline = time.monotonic() + timeout_sec
        deltas: list[str] = []

        while time.monotonic() < deadline:
            event = self._recv_event(timeout_sec=max(0.1, deadline - time.monotonic()))
            event_type = str(event.get("type") or "").strip()

            if event_type in {
                "conversation.item.input_audio_transcription.delta",
                "response.audio_transcript.delta",
                "response.text.delta",
            }:
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    deltas.append(delta)
                continue

            if event_type in {
                "conversation.item.input_audio_transcription.completed",
                "response.audio_transcript.done",
                "response.text.done",
            }:
                transcript = event.get("transcript") or event.get("text")
                if isinstance(transcript, str) and transcript.strip():
                    return transcript
                if deltas:
                    return "".join(deltas)
                continue

            if event_type == "error":
                raise RuntimeError(_extract_error_message(event))

        if deltas:
            return "".join(deltas)
        raise TimeoutError("Timed out waiting for realtime transcription result")

    def _send_event(self, event: dict[str, Any]) -> None:
        self._connection.send(json.dumps(event, ensure_ascii=False))

    def _recv_event(self, *, timeout_sec: float) -> dict[str, Any]:
        self._connection.settimeout(timeout_sec)
        raw = self._connection.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise RuntimeError("invalid realtime event payload")
        return event

    def _decode_audio_to_pcm(self, *, audio_bytes: bytes, mime_type: str) -> bytes:
        suffix = _ext_from_mime(mime_type)
        with tempfile.NamedTemporaryFile(suffix=suffix) as src:
            src.write(audio_bytes)
            src.flush()
            cmd = [
                self.ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                src.name,
                "-ac",
                "1",
                "-ar",
                str(self.sample_rate),
                "-f",
                "s16le",
                "-",
            ]
            proc = subprocess.run(cmd, capture_output=True, check=False)

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg decode failed: {stderr or 'unknown error'}")

        return proc.stdout

    def _realtime_url(self) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            realtime_path = f"{path}/realtime"
        elif path.endswith("/realtime"):
            realtime_path = path
        else:
            realtime_path = "/v1/realtime"
        return urlunparse(parsed._replace(scheme=scheme, path=realtime_path, params="", query="", fragment=""))


def _extract_error_message(event: dict[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    message = event.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return "realtime transcription failed"


def _ext_from_mime(mime_type: str) -> str:
    lowered = (mime_type or "").lower()
    if "wav" in lowered:
        return ".wav"
    if "webm" in lowered:
        return ".webm"
    if "ogg" in lowered or "opus" in lowered:
        return ".ogg"
    if "mp4" in lowered or "m4a" in lowered:
        return ".m4a"
    if "mpeg" in lowered or "mp3" in lowered:
        return ".mp3"
    return ".bin"
