"""Resumable PCM ingestion and rolling LocalAgreement over the configured ASR API.

Audio ACK means fsynced PCM + checkpoint, not completed recognition. Decoder
state is reconstructible from this journal after a socket or worker restart.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import fcntl
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from ..core.blocking import blocking_work_pool
from ..core.config import settings
from ..core.rate_limit import consume
from ..core.security import runtime_access_allowed, read_guest_artifact_grant
from ..openai_whisper import OpenAIWhisperTranscriber
from ..services.meeting_source import MeetingError, read_json, write_json_atomic
from ..transcript_store import TranscriptRecord, TranscriptStore, build_debug_chunks_dir, read_jsonl_records, resolve_transcript_path
from .local_agreement import SAMPLE_RATE, agreed_prefix, pcm_wav, speech_bounds
from .messages import _image_matches_mime_type, validate_start_message
from .factory import _parse_diarization_speaker_params
from .text_processing import _sanitize_transcript_text

logger = logging.getLogger(__name__)
UPDATE_SAMPLES = 2 * SAMPLE_RATE
MAX_WINDOW_SAMPLES = 24 * SAMPLE_RATE
MAX_BACKLOG_SAMPLES = 120 * SAMPLE_RATE
MAX_MEETING_SAMPLES = 4 * 60 * 60 * SAMPLE_RATE


class LiveMeeting:
    def __init__(self, ws: WebSocket, payload: dict):
        self.ws = ws
        self.wake = asyncio.Event()
        self.stopping = False
        self.disconnected = False
        self.send_lock = asyncio.Lock()
        self.state_lock = asyncio.Lock()
        self.lock = None
        self.failures = 0
        self.is_guest = bool(getattr(ws.state, "is_guest", False))
        self.rate_subject = str(getattr(ws.state, "rate_limit_subject", "unknown"))
        resume = str(payload.get("resumeSessionId") or "")
        if resume:
            if not runtime_access_allowed(transcripts_dir=settings.transcripts_dir, session_id=resume,
                                          user_id=getattr(ws.state, "authenticated_user_id", None),
                                          guest_grant_id=read_guest_artifact_grant(ws)):
                raise MeetingError("meeting_not_found", 404)
            path = resolve_transcript_path(settings.transcripts_dir, resume, "jsonl")
            self.store = TranscriptStore.__new__(TranscriptStore)
            self.store.root_dir = settings.transcripts_dir
            self.store.session_id = resume
            self.store.base_path = path.with_suffix("")
            self.store.jsonl_path = path
            self.store.txt_path = path.with_suffix(".txt")
            self.store.metadata_path = path.with_suffix(".meta.json")
            self.store.chunks_dir = settings.transcripts_dir / "_chunks" / resume
            from ..transcript_store import iter_runtime_screenshot_dirs
            self.store.screenshots_dir = next((p for p in iter_runtime_screenshot_dirs(settings.transcripts_dir, resume) if p.exists()), None)
            if self.store.screenshots_dir is None:
                raise MeetingError("meeting_not_found", 404)
            self.data = read_json(path.with_suffix(".live.json"))
            if not self.data or self.data.get("protocolVersion") != 2:
                raise MeetingError("meeting_not_resumable")
        else:
            tracks = payload.get("tracks", ["mixed"])
            if not isinstance(tracks, list) or not 1 <= len(tracks) <= 2 or any(not isinstance(track, str) or track not in {"mic", "display", "mixed"} for track in tracks) or len(set(tracks)) != len(tracks):
                raise MeetingError("invalid_audio_tracks")
            if "mixed" in tracks and len(tracks) > 1:
                raise MeetingError("invalid_audio_tracks")
            session_id = TranscriptStore.make_runtime_session_id(TranscriptStore.sanitize_or_generate(payload.get("sessionId")))
            self.store = TranscriptStore(settings.transcripts_dir, session_id)
            self.data = {
                "protocolVersion": 2, "tracks": {track: {"received": 0, "seq": -1, "windowStart": 0, "lastDecode": 0, "hypothesis": "", "stable": ""} for track in tracks},
                "language": str(payload.get("language") or settings.default_language)[:32],
                "prompt": str(payload.get("prompt") or settings.default_prompt),
                "vocabulary": str(payload.get("sharedVocabulary") or ""),
                "asrRequests": 0, "audioBytes": 0, "stableSegments": [],
                "finalized": False,
                "diarizationRequested": bool(payload.get("diarizationEnabled", False)),
                "speakerCounts": _parse_diarization_speaker_params(payload, settings=settings),
            }
            self.store.write_metadata({
                "sessionId": session_id, "ownerUserId": getattr(ws.state, "authenticated_user_id", None),
                "guestGrantDigest": getattr(ws.state, "guest_artifact_grant_digest", None),
                "language": self.data["language"], "audioSource": payload.get("audioSource", "mic"),
                "mode": "live", "finalized": False, "createdAt": datetime.now(timezone.utc).isoformat(),
                "prompt": self.data["prompt"],
            })
        self.session_id = self.store.session_id
        self.state_path = self.store.jsonl_path.with_suffix(".live.json")
        self.lock = self.store.jsonl_path.with_suffix(".live.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            self.lock = None
            raise MeetingError("meeting_already_connected", 409) from exc
        if resume and self.data.get("asrBackend", "whisper") != settings.asr_backend:
            self.lock.close()
            self.lock = None
            raise MeetingError("meeting_backend_mismatch")
        self.records = read_jsonl_records(self.store.jsonl_path)
        if resume:
            self.data = read_json(self.state_path)
        self.next_seq = max((int(row.get("seq", -1)) for row in self.records if row.get("type") == "final"), default=-1) + 1
        self.audio_dir = build_debug_chunks_dir(settings.debug_chunks_dir, self.session_id)
        # Preserve the original dated audio folder when resuming on another day.
        existing = self.data.get("audioDir")
        if existing:
            candidate = (settings.debug_chunks_dir / existing).resolve()
            if candidate.is_relative_to(settings.debug_chunks_dir.resolve()):
                self.audio_dir = candidate
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.data["audioDir"] = str(self.audio_dir.relative_to(settings.debug_chunks_dir.resolve())) if self.audio_dir.is_absolute() else str(self.audio_dir.relative_to(settings.debug_chunks_dir))
        self.transcriber = self.create_transcriber()
        write_json_atomic(self.state_path, self.data)

    def create_transcriber(self):
        transcriber = OpenAIWhisperTranscriber(api_key=settings.openai_api_key, base_url=settings.openai_base_url, model=settings.asr_model)
        transcriber.multi_pass_enabled = False
        transcriber.retry_max_attempts = 1
        transcriber.client = transcriber.client.with_options(max_retries=0)
        return transcriber

    def close(self):
        if self.transcriber is not None:
            self.transcriber.client.close()
        if self.lock:
            self.lock.close()

    async def send(self, event: dict):
        if self.disconnected:
            return
        async with self.send_lock:
            try:
                await self.ws.send_json({"sessionId": self.session_id, **event})
            except (RuntimeError, WebSocketDisconnect, OSError):
                self.disconnected = True

    def _write_audio(self, track: str, start: int, pcm: bytes):
        path = self.store.chunks_dir / f"{track}.pcm"
        with path.open("r+b" if path.exists() else "w+b") as out:
            # A crash between the audio write and ACK checkpoint can leave a
            # tail. Re-sending the same samples overwrites that unacknowledged tail.
            out.seek(start * 2)
            out.write(pcm)
            out.truncate()
            out.flush()
            os.fsync(out.fileno())

    def _read_audio(self, track: str, start: int, end: int) -> bytes:
        with (self.store.chunks_dir / f"{track}.pcm").open("rb") as source:
            source.seek(start * 2)
            return source.read(max(0, end - start) * 2)

    def _append_record(self, record: TranscriptRecord):
        self.store.append_final(record)
        # A checkpoint must never advance past an undurable final transcript.
        for path in (self.store.jsonl_path, self.store.txt_path):
            with path.open("rb") as source:
                os.fsync(source.fileno())

    async def accept_audio(self, payload: dict):
        if self.data.get("finalized"):
            raise MeetingError("meeting_finalized")
        track = payload.get("track")
        if not isinstance(track, str):
            raise MeetingError("invalid_audio_track")
        state = self.data["tracks"].get(track)
        if state is None:
            raise MeetingError("invalid_audio_track")
        seq, start = payload.get("seq"), payload.get("sampleStart")
        if type(seq) is not int or type(start) is not int or seq < 0 or start < 0:
            raise MeetingError("invalid_audio_sequence")
        if seq <= state["seq"]:
            await self.send({"type": "capture_ack", "track": track, "seq": state["seq"], "samples": state["received"]})
            return
        if seq != state["seq"] + 1 or start != state["received"]:
            await self.send({"type": "resend", "track": track, "seq": state["seq"] + 1, "samples": state["received"]})
            return
        encoded = payload.get("pcm")
        if not isinstance(encoded, str) or len(encoded) > 180_000:
            raise MeetingError("invalid_pcm")
        try:
            pcm = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise MeetingError("invalid_pcm") from exc
        if not pcm or len(pcm) % 2 or len(pcm) > 4 * SAMPLE_RATE * 2:
            raise MeetingError("invalid_pcm")
        if start + len(pcm) // 2 > MAX_MEETING_SAMPLES:
            raise MeetingError("meeting_duration_limit", 413)
        if self.is_guest and self.data["audioBytes"] + len(pcm) > settings.guest_ws_max_audio_bytes:
            raise MeetingError("guest_audio_limit", 429)
        if start - state["windowStart"] > MAX_BACKLOG_SAMPLES:
            await self.send({"type": "backpressure", "message": "認識が遅れています。音声を端末で保持しています。", "retryMs": 2000})
            return
        async with self.state_lock:
            await blocking_work_pool.run("artifact", self._write_audio, track, start, pcm)
            state["received"] += len(pcm) // 2
            state["seq"] = seq
            self.data["audioBytes"] += len(pcm)
            await blocking_work_pool.run("artifact", write_json_atomic, self.state_path, copy.deepcopy(self.data))
        await self.send({"type": "capture_ack", "track": track, "seq": seq, "samples": state["received"]})
        self.wake.set()

    async def accept_screen(self, payload: dict):
        if self.data.get("finalized"):
            raise MeetingError("meeting_finalized")
        encoded, mime = payload.get("image"), payload.get("mimeType")
        stamp = payload.get("timeMs")
        max_ms = max(s["received"] for s in self.data["tracks"].values()) * 1000 // SAMPLE_RATE
        if type(stamp) is not int or stamp < 0 or stamp > max_ms + 3000:
            raise MeetingError("invalid_screen_time")
        last = self.data.get("lastScreenMs", -3000)
        if stamp - last < 1000:
            return
        if not isinstance(encoded, str) or len(encoded) > settings.ws_screenshot_max_bytes * 4 // 3 + 8:
            raise MeetingError("screenshot_too_large", 413)
        try:
            image = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise MeetingError("invalid_screenshot") from exc
        if len(image) > settings.ws_screenshot_max_bytes or not _image_matches_mime_type(image, str(mime or "")):
            raise MeetingError("invalid_screenshot")
        async with self.state_lock:
            # High sequence values keep independent images away from legacy
            # per-segment filenames; screenshots retain their actual capture time.
            filename = await blocking_work_pool.run("artifact", self.store.save_screenshot, seq=10_000_000 + stamp, mime_type=mime, image_bytes=image)
            record = TranscriptRecord(type="screen", segmentId=f"screen-{stamp}", seq=10_000_000 + stamp, text="", tsStart=stamp, tsEnd=stamp,
                                      chunkOffsetMs=stamp, chunkDurationMs=0, language=self.data["language"], createdAt=datetime.now(timezone.utc).isoformat(),
                                      screenshotPath=f"/api/transcripts/{self.session_id}/screenshots/{filename}")
            await blocking_work_pool.run("artifact", self._append_record, record)
            self.records.append(vars_record(record))
            self.data["lastScreenMs"] = stamp
            await blocking_work_pool.run("artifact", write_json_atomic, self.state_path, copy.deepcopy(self.data))
        await self.send({"type": "screen", "timeMs": stamp, "screenshotPath": record.screenshotPath})

    async def _infer(self, track: str, *, flush: bool):
        state = self.data["tracks"][track]
        start = state["windowStart"]
        end = min(state["received"], start + MAX_WINDOW_SAMPLES)
        if end <= start or (not flush and end - state["lastDecode"] < UPDATE_SAMPLES):
            return False
        pcm = await blocking_work_pool.run("artifact", self._read_audio, track, start, end)
        bounds = await blocking_work_pool.run("media", speech_bounds, pcm)
        if bounds is None:
            state.update(windowStart=end, lastDecode=end, hypothesis="", stable="")
            state.pop("stableSegment", None)
            await self.checkpoint()
            return True
        final = flush or end - start >= MAX_WINDOW_SAMPLES or len(pcm) // 2 - bounds[1] >= int(0.65 * SAMPLE_RATE)
        if self.is_guest and self.data["asrRequests"] >= settings.guest_ws_max_asr_requests:
            raise MeetingError("guest_asr_request_limit", 429)
        allowed = await asyncio.to_thread(consume, bucket="asr", subject=self.rate_subject,
                                          limit=settings.costly_api_rate_limit_requests, window_seconds=settings.costly_api_rate_limit_window_seconds)
        if not allowed:
            raise MeetingError("rate_limit_exceeded", 429)
        self.data["asrRequests"] += 1
        started = time.monotonic()
        context = " ".join(str(row.get("text") or "") for row in self.records[-6:] if row.get("type") == "final")[-300:]
        prompt = (self.data["vocabulary"][:250] + " " + self.data["prompt"][:200] + " " + context).strip()
        result = await blocking_work_pool.run("asr", self.transcriber.transcribe_chunk, pcm_wav(pcm), mime_type="audio/wav",
                                             language=self.data["language"] or None, prompt=prompt or None, temperature=0.0)
        text = _sanitize_transcript_text(result.text, language=self.data["language"]).strip()
        previous = state["hypothesis"]
        stable = agreed_prefix(previous, text)
        state.update(lastDecode=end, hypothesis=text, stable=stable)
        segment_id = f"{track}-{start:012d}"
        start_ms = start * 1000 // SAMPLE_RATE + (result.start_ms if result.start_ms is not None else bounds[0] * 1000 // SAMPLE_RATE)
        end_ms = min(end * 1000 // SAMPLE_RATE, start * 1000 // SAMPLE_RATE + (result.end_ms if result.end_ms is not None else bounds[1] * 1000 // SAMPLE_RATE))
        end_ms = max(start_ms, end_ms)
        speaker = "自分（マイク）" if track == "mic" and len(self.data["tracks"]) > 1 else "共有音声" if track == "display" else ""
        event = {"type": "partial", "track": track, "segmentId": segment_id, "seq": self.next_seq,
                 "text": text, "stableText": stable, "tsStart": start_ms, "tsEnd": end_ms, "speaker": speaker,
                 "inferenceMs": round((time.monotonic() - started) * 1000), "backlogMs": (state["received"] - end) * 1000 // SAMPLE_RATE}
        if final:
            existing = any(row.get("type") == "final" and row.get("segmentId") == segment_id for row in self.records)
            if text and not existing:
                audio_name = f"raw-{self.next_seq:06d}.wav"
                await blocking_work_pool.run("artifact", (self.audio_dir / audio_name).write_bytes, pcm_wav(pcm))
                images = [row for row in self.records if row.get("type") == "screen" and row.get("tsStart", 0) <= end_ms]
                image_path = images[-1].get("screenshotPath") if images else None
                record = TranscriptRecord(type="final", segmentId=segment_id, seq=self.next_seq, text=text,
                                          tsStart=start_ms, tsEnd=end_ms, chunkOffsetMs=start * 1000 // SAMPLE_RATE,
                                          chunkDurationMs=(end - start) * 1000 // SAMPLE_RATE, language=self.data["language"],
                                          createdAt=datetime.now(timezone.utc).isoformat(), speaker=speaker or None,
                                          screenshotPath=image_path, rawAudioPath=f"/api/transcripts/{self.session_id}/audio/{audio_name}")
                await blocking_work_pool.run("artifact", self._append_record, record)
                self.records.append(vars_record(record))
                self.next_seq += 1
                await self.send({**vars_record(record), "track": track})
            # Every source sample is retained, even when a hard window ends.
            state.update(windowStart=end, lastDecode=end, hypothesis="", stable="")
            await self.send({"type": "partial", "track": track, "text": "", "stableText": ""})
        else:
            state["stableSegment"] = {**event, "text": stable}
            await self.send(event)
        if final:
            state.pop("stableSegment", None)
        await self.checkpoint()
        return True

    async def checkpoint(self):
        async with self.state_lock:
            self.data["stableSegments"] = [state["stableSegment"] for state in self.data["tracks"].values() if state.get("stableSegment", {}).get("text")]
            await blocking_work_pool.run("artifact", write_json_atomic, self.state_path, copy.deepcopy(self.data))

    async def diarize(self):
        from .. import runtime
        from ..diarizer import AudioChunk
        if not self.data.get("diarizationRequested") or runtime.DIARIZER is None:
            return
        await self.send({"type": "info", "message": "diarization_started"})
        patches = []
        try:
            for track in self.data["tracks"]:
                if track == "mic" and len(self.data["tracks"]) > 1:
                    continue
                rows = [row for row in self.records if row.get("type") == "final" and str(row.get("segmentId", "")).startswith(track + "-")]
                chunks = [AudioChunk(seq=row["seq"], path=self.audio_dir / Path(row.get("rawAudioPath") or f"raw-{row['seq']:06d}.wav").name,
                                     offset_ms=row["chunkOffsetMs"], duration_ms=row["chunkDurationMs"]) for row in rows]
                counts = self.data.get("speakerCounts", [0, 0, 0])
                turns = await blocking_work_pool.run("diarization", runtime.DIARIZER.diarize,
                    session_id=f"{self.session_id}-{track}", chunks=chunks, work_dir=settings.diarization_work_dir,
                    num_speakers=counts[0], min_speakers=counts[1], max_speakers=counts[2])
                for row in rows:
                    speaker = runtime._pick_speaker(turns, row["tsStart"], row["tsEnd"])
                    if speaker:
                        row["speaker"] = ("共有音声 · " if track == "display" else "") + speaker
                        patches.append({"seq": row["seq"], "speaker": row["speaker"]})
            if patches:
                await blocking_work_pool.run("artifact", self.store.rewrite_records, self.records)
                await self.send({"type": "speaker_patch", "segments": patches})
        except Exception as exc:
            logger.warning("live diarization failed: session=%s error=%s", self.session_id, type(exc).__name__)
            await self.send({"type": "error", "message": "diarization_failed", "detail": "話者分離に失敗しました。音声と文字起こしは保存されています。"})
        await self.send({"type": "info", "message": "diarization_done"})

    async def work(self):
        while True:
            await self.wake.wait()
            self.wake.clear()
            try:
                for track in self.data["tracks"]:
                    while await self._infer(track, flush=self.stopping):
                        if not self.stopping:
                            break
                self.failures = 0
                if self.stopping:
                    await self.diarize()
                    self.data["finalized"] = True
                    await self.checkpoint()
                    metadata = self.store.read_metadata()
                    metadata.update(finalized=True, finalizedAt=datetime.now(timezone.utc).isoformat())
                    await blocking_work_pool.run("artifact", write_json_atomic, self.store.metadata_path, metadata)
                    await self.send({"type": "info", "message": "finalized", "state": "completed"})
                    return
                if self.disconnected:
                    return
                # Audio may have arrived while inference was in flight.
                if any(s["received"] - s["lastDecode"] >= UPDATE_SAMPLES for s in self.data["tracks"].values()):
                    self.wake.set()
            except Exception as exc:
                self.failures += 1
                logger.warning("live ASR failed: session=%s error=%s", self.session_id, type(exc).__name__)
                await self.send({"type": "error", "message": "finalize_failed" if self.stopping else "transcription_failed",
                                 "buffered": True, "detail": "音声は保存されています。接続を確認して再試行できます。"})
                if self.stopping or self.disconnected:
                    return
                await asyncio.sleep(min(10, 2 * self.failures))
                self.wake.set()


def vars_record(record):
    from dataclasses import asdict
    return asdict(record)


async def live_transcribe(ws: WebSocket):
    await ws.accept()
    meeting = None
    worker = None
    try:
        initial = await asyncio.wait_for(ws.receive_json(), 15)
        if not isinstance(initial, dict) or initial.get("type") != "start":
            raise MeetingError("start_required")
        error = validate_start_message(initial, prompt_max_chars=settings.ws_prompt_max_chars, vocabulary_max_chars=settings.ws_vocabulary_max_chars)
        if error:
            raise MeetingError(error)
        from .qwen_live import QwenLiveMeeting
        meeting_class = QwenLiveMeeting if settings.asr_backend == "qwen3_vllm" else LiveMeeting
        meeting = await blocking_work_pool.run("artifact", meeting_class, ws, initial)
        await meeting.send({"type": "info", "message": "ready", "protocolVersion": 2, "asrBackend": settings.asr_backend,
                            "tracks": {key: {"seq": value["seq"], "samples": value["received"]} for key, value in meeting.data["tracks"].items()},
                            "records": [row for row in meeting.records if row.get("type") == "final"],
                            "finalized": bool(meeting.data.get("finalized"))})
        if meeting.data.get("finalized"):
            await meeting.send({"type": "info", "message": "finalized", "state": "completed"})
            return
        worker = asyncio.create_task(meeting.work())
        meeting.wake.set()
        invalid = 0
        while True:
            raw = await ws.receive_text()
            if len(raw) > min(settings.ws_max_message_bytes, 3_000_000):
                raise MeetingError("message_too_large", 413)
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise MeetingError("invalid_message")
                kind = data.get("type")
                if kind == "audio":
                    await meeting.accept_audio(data)
                elif kind == "screen":
                    await meeting.accept_screen(data)
                elif kind == "stop":
                    meeting.stopping = True
                    meeting.wake.set()
                    await meeting.send({"type": "info", "message": "stopping"})
                    await worker
                    break
                elif kind == "ping":
                    await meeting.send({"type": "pong"})
                else:
                    raise MeetingError("unsupported_message")
            except (MeetingError, ValueError) as exc:
                invalid += 1
                await meeting.send({"type": "error", "message": getattr(exc, "code", "invalid_json")})
                if invalid >= settings.ws_max_invalid_messages:
                    break
    except (WebSocketDisconnect, TimeoutError):
        pass
    except MeetingError as exc:
        await ws.send_json({"type": "error", "message": exc.code})
    finally:
        if meeting:
            meeting.disconnected = True
            meeting.wake.set()
            # Keep the file lock and worker slot until in-flight inference ends.
            # No forced cancellation of a thread that may still write artifacts.
            if worker:
                await asyncio.shield(worker)
            await asyncio.to_thread(meeting.close)
        try:
            await ws.close()
        except RuntimeError:
            pass
