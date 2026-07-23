from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from ..diarizer import AudioChunk
from ..transcript_store import TranscriptRecord
from .session import FailedPreparedChunk, LiveSession


@dataclass(frozen=True)
class WorkerDependencies:
    settings: Any
    observer: Any
    logger: Any
    prepare_audio: Callable[..., Any]
    merge_wav_chunks: Callable[[list[bytes]], bytes]
    safe_send: Callable[[Any, dict[str, Any]], Awaitable[bool]]
    build_prompt: Callable[[LiveSession], str | None]
    accumulate_usage: Callable[[LiveSession, Any], None]
    retry_weird_transcription: Callable[..., Awaitable[Any]]
    save_debug_chunk: Callable[[LiveSession, Any], None]
    sanitize_text: Callable[..., str]
    trim_overlap: Callable[[str, str], str]
    light_proofread: Callable[..., str]
    should_drop_boundary: Callable[..., bool]
    is_near_duplicate: Callable[..., bool]
    coerce_bounds: Callable[..., tuple[int, int]]
    store_screenshot: Callable[[LiveSession, Any], str | None]
    store_raw_audio: Callable[[LiveSession, Any], str | None]
    store_asr_audio: Callable[[LiveSession, int, bytes], str | None]
    resolve_audio_url: Callable[..., str | None]
    append_context: Callable[[LiveSession, str], None]
    clip_trace_text: Callable[[str], str]
    emit_log: Callable[..., None]


async def run_session_worker(ws: Any, session: LiveSession, deps: WorkerDependencies) -> None:
    emitted_segments = 0
    emitted_chars = 0
    trace_context = (
        deps.observer.create_trace_context(
            name="asr.session",
            input={
                "sessionId": session.session_id,
                "language": session.language or "",
                "audioSource": session.audio_source,
                "model": deps.settings.asr_model,
                "diarizationEnabled": session.collect_audio_for_diarization,
            },
        )
        if deps.observer is not None
        else None
    )
    try:
        while True:
            item = await session.queue.get()
            if item is None:
                break

            if session.collect_audio_for_diarization:
                try:
                    chunk_path = session.store.save_audio_chunk(
                        seq=item.seq,
                        mime_type=item.mime_type,
                        audio_bytes=item.audio_bytes,
                    )
                    session.audio_chunks.append(
                        AudioChunk(
                            seq=item.seq,
                            path=chunk_path,
                            offset_ms=item.offset_ms,
                            duration_ms=item.duration_ms,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    deps.logger.warning(
                        "Chunk save failed for diarization: session=%s seq=%s err=%s",
                        session.session_id,
                        item.seq,
                        exc,
                    )
            deps.save_debug_chunk(session, item)

            prepared = None
            try:
                prepared = deps.prepare_audio(session=session, item=item)
                buffered_count = len(session.failed_prepared_chunks)
                effective_audio_bytes = prepared.audio_bytes
                effective_seq = item.seq
                effective_offset_ms = item.offset_ms
                effective_duration_ms = item.duration_ms
                if session.failed_prepared_chunks:
                    buffered = list(session.failed_prepared_chunks)
                    effective_audio_bytes = deps.merge_wav_chunks(
                        [entry.audio_bytes for entry in buffered] + [prepared.audio_bytes]
                    )
                    effective_seq = buffered[0].seq
                    effective_offset_ms = buffered[0].offset_ms
                    effective_duration_ms = sum(entry.duration_ms for entry in buffered) + item.duration_ms
                    deps.logger.info(
                        "Merging failed chunks before retry: session=%s buffered=%d first_seq=%s current_seq=%s",
                        session.session_id,
                        buffered_count,
                        effective_seq,
                        item.seq,
                    )
                deps.logger.info(
                    "Prepared audio: session=%s seq=%s rms=%.4f peak=%.4f speech_ratio=%.4f overlap_ms=%d",
                    session.session_id,
                    item.seq,
                    prepared.rms,
                    prepared.peak,
                    prepared.speech_ratio,
                    prepared.overlap_ms_used,
                )
                if (
                    deps.settings.asr_vad_drop_enabled
                    and prepared.speech_ratio < deps.settings.asr_vad_speech_ratio_min
                ):
                    deps.logger.info(
                        "Skipping low-speech chunk: session=%s seq=%s speech_ratio=%.4f threshold=%.4f",
                        session.session_id,
                        item.seq,
                        prepared.speech_ratio,
                        deps.settings.asr_vad_speech_ratio_min,
                    )
                    await deps.safe_send(
                        ws,
                        {
                            "type": "ack",
                            "seq": item.seq,
                            "empty": True,
                            "skipped": True,
                            "reason": "low_speech_ratio",
                            "speechRatio": prepared.speech_ratio,
                            "rms": prepared.rms,
                            "peak": prepared.peak,
                        },
                    )
                    continue
                result = await asyncio.to_thread(
                    session.transcriber.transcribe_chunk,
                    effective_audio_bytes,
                    mime_type=prepared.mime_type,
                    language=session.language,
                    prompt=deps.build_prompt(session),
                    temperature=session.temperature,
                    trace_context=trace_context,
                )
                deps.accumulate_usage(session, result)
                result = await deps.retry_weird_transcription(
                    session=session,
                    prepared=prepared,
                    trace_context=trace_context,
                    audio_bytes=effective_audio_bytes,
                    previous_text=session.last_emitted_text,
                    result=result,
                )
                session.failed_prepared_chunks.clear()
            except Exception as exc:  # noqa: BLE001
                deps.logger.exception("Transcription failed: session=%s seq=%s", session.session_id, item.seq)
                if len(session.failed_prepared_chunks) >= 2:
                    session.failed_prepared_chunks = session.failed_prepared_chunks[-1:]
                session.failed_prepared_chunks.append(
                    FailedPreparedChunk(
                        seq=item.seq,
                        offset_ms=item.offset_ms,
                        duration_ms=item.duration_ms,
                        audio_bytes=prepared.audio_bytes if prepared is not None else b"",
                        speech_ratio=item.speech_ratio,
                        active_ms=item.active_ms,
                        silence_ms=item.silence_ms,
                    )
                )
                await deps.safe_send(
                    ws,
                    {
                        "type": "error",
                        "message": "transcription_failed",
                        "seq": item.seq,
                        "buffered": True,
                        "bufferedCount": len(session.failed_prepared_chunks),
                        "detail": str(exc),
                    },
                )
                continue

            text = deps.sanitize_text(result.text.strip(), language=session.language)
            text = deps.trim_overlap(text, session.last_emitted_text)
            text = deps.sanitize_text(text, language=session.language)
            if bool(getattr(deps.settings, "asr_light_proofread_enabled", True)):
                text = deps.light_proofread(text, language=session.language)
            if not text:
                await deps.safe_send(ws, {"type": "ack", "seq": item.seq, "empty": True})
                continue
            if deps.should_drop_boundary(
                text,
                session.last_emitted_text,
                source_mode=session.audio_source,
                suspicious=bool(result.suspicious),
            ):
                await deps.safe_send(
                    ws,
                    {"type": "ack", "seq": item.seq, "empty": True, "skipped": True, "reason": "boundary_fragment"},
                )
                continue

            current_start_hint = max(0, effective_offset_ms - prepared.overlap_ms_used) + (result.start_ms or 0)
            if deps.is_near_duplicate(
                text,
                session.last_emitted_text,
                current_start_ms=current_start_hint,
                previous_end_ms=session.last_emitted_ts_end,
            ):
                await deps.safe_send(ws, {"type": "ack", "seq": item.seq, "duplicate": True})
                continue

            ts_base_offset = max(0, effective_offset_ms - prepared.overlap_ms_used)
            ts_start = ts_base_offset + (result.start_ms or 0)
            ts_end = (
                ts_base_offset + result.end_ms
                if result.end_ms is not None
                else effective_offset_ms + max(effective_duration_ms, 600)
            )
            ts_start, ts_end = deps.coerce_bounds(
                ts_start=ts_start,
                ts_end=ts_end,
                previous_end_ms=session.last_emitted_ts_end,
            )

            record = TranscriptRecord(
                type="final",
                segmentId=f"{effective_seq:06d}",
                seq=effective_seq,
                text=text,
                tsStart=ts_start,
                tsEnd=ts_end,
                chunkOffsetMs=effective_offset_ms,
                chunkDurationMs=effective_duration_ms,
                language=session.language,
                createdAt=datetime.now(timezone.utc).isoformat(),
                screenshotPath=deps.store_screenshot(session, item),
                rawAudioPath=(
                    deps.store_raw_audio(session, item)
                    or deps.resolve_audio_url(session, prefix="raw", seq=item.seq)
                ),
                audioPath=(
                    deps.store_asr_audio(session, effective_seq, effective_audio_bytes)
                    or deps.resolve_audio_url(session, prefix="asr", seq=effective_seq)
                ),
            )
            session.store.append_final(record)
            session.last_emitted_text = text
            session.last_emitted_ts_end = ts_end
            session.transcript_history.append(text)
            deps.append_context(session, text)
            emitted_segments += 1
            emitted_chars += len(text)
            await deps.safe_send(
                ws,
                {
                    "type": "final",
                    "segmentId": record.segmentId,
                    "seq": record.seq,
                    "text": record.text,
                    "tsStart": record.tsStart,
                    "tsEnd": record.tsEnd,
                    "speaker": record.speaker,
                    "screenshotPath": record.screenshotPath,
                    "rawAudioPath": record.rawAudioPath,
                    "audioPath": record.audioPath,
                },
            )
            deps.emit_log(
                __name__,
                "debug",
                "ws final sent: session=%s seq=%s chars=%s ts_start=%s ts_end=%s",
                session.session_id,
                record.seq,
                len(record.text),
                record.tsStart,
                record.tsEnd,
            )

        if deps.observer is not None and trace_context is not None:
            with deps.observer.generation(
                name="asr.session.result",
                model=deps.settings.asr_model,
                output={
                    "segmentCount": emitted_segments,
                    "charCount": emitted_chars,
                    "estimatedTokens": session.asr_estimated_tokens,
                    "finalTranscript": deps.clip_trace_text("\n".join(session.transcript_history)),
                },
                metadata={"usageSource": "api" if session.asr_total_tokens > 0 else "estimated"},
                model_parameters={"estimatedTokens": session.asr_estimated_tokens},
                trace_context=trace_context,
            ):
                pass
    finally:
        try:
            await asyncio.to_thread(session.transcriber.close)
        except Exception:  # noqa: BLE001
            deps.logger.debug("session transcriber close failed: session=%s", session.session_id, exc_info=True)
