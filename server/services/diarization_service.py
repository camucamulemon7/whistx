from __future__ import annotations

import logging
from typing import Any
from fastapi import WebSocket

from ..core.blocking import blocking_work_pool
from ..core.public_errors import public_error
from ..diarizer import PyannoteSpeakerDiarizer, SpeakerTurn
from ..transcript_store import read_jsonl_records
from ..transcription.messages import as_int as _as_int
from ..transcription.session import LiveSession

logger = logging.getLogger(__name__)


async def run_diarization_for_session(ws: WebSocket, session: LiveSession, *, diarizer: PyannoteSpeakerDiarizer | None, send: Any, config: Any) -> None:
    if diarizer is None:
        session.store.cleanup_chunks()
        return
    if not session.collect_audio_for_diarization:
        session.store.cleanup_chunks()
        return
    if not session.audio_chunks:
        session.store.cleanup_chunks()
        return

    await send(
        ws,
        {
            "type": "info",
            "message": "diarization_started",
            "sessionId": session.session_id,
        },
    )

    try:
        patch_map = await blocking_work_pool.run(
            "diarization",
            apply_diarization_labels,
            session,
            diarizer=diarizer, config=config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Diarization failed: session=%s", session.session_id)
        await send(
            ws,
            {
                "type": "error",
                "message": "diarization_failed",
                "sessionId": session.session_id,
                **public_error("diarization_failed", exc, logger),
            },
        )
        patch_map = {}

    if patch_map:
        payload = [
            {"seq": seq, "speaker": speaker}
            for seq, speaker in sorted(patch_map.items())
        ]
        await send(
            ws,
            {
                "type": "speaker_patch",
                "sessionId": session.session_id,
                "segments": payload,
            },
        )

    await send(
        ws,
        {
            "type": "info",
            "message": "diarization_done",
            "sessionId": session.session_id,
        },
    )
    if not config.diarization_keep_chunks:
        session.store.cleanup_chunks()



def apply_diarization_labels(session: LiveSession, *, diarizer: PyannoteSpeakerDiarizer | None, config: Any) -> dict[int, str]:
    if diarizer is None:
        return {}

    turns = diarizer.diarize(
        session_id=session.session_id,
        chunks=session.audio_chunks,
        work_dir=config.diarization_work_dir,
        num_speakers=session.diarization_num_speakers,
        min_speakers=session.diarization_min_speakers,
        max_speakers=session.diarization_max_speakers,
    )
    if not turns:
        return {}

    records = read_jsonl_records(session.store.jsonl_path)
    if not records:
        return {}

    patch_map: dict[int, str] = {}
    updated = False

    for rec in records:
        if rec.get("type") != "final":
            continue
        start_ms = _as_int(rec.get("tsStart"), 0)
        end_ms = _as_int(rec.get("tsEnd"), start_ms)
        if end_ms < start_ms:
            end_ms = start_ms

        speaker = pick_speaker(turns, start_ms, end_ms)
        if not speaker:
            continue

        current = str(rec.get("speaker", "")).strip()
        if current == speaker:
            continue

        rec["speaker"] = speaker
        seq = _as_int(rec.get("seq"), -1)
        if seq >= 0:
            patch_map[seq] = speaker
        updated = True

    if updated:
        session.store.rewrite_records(records)
        logger.info(
            "Diarization applied: session=%s speakers=%d segments=%d requested_num=%d requested_min=%d requested_max=%d",
            session.session_id,
            len({t.speaker for t in turns}),
            len(patch_map),
            session.diarization_num_speakers,
            session.diarization_min_speakers,
            session.diarization_max_speakers,
        )

    return patch_map



def pick_speaker(turns: list[SpeakerTurn], start_ms: int, end_ms: int) -> str | None:
    if not turns:
        return None

    s = max(0, start_ms)
    e = max(s + 1, end_ms)
    overlap_by_speaker: dict[str, int] = {}

    for turn in turns:
        if turn.end_ms <= s:
            continue
        if turn.start_ms >= e:
            break

        overlap = min(e, turn.end_ms) - max(s, turn.start_ms)
        if overlap <= 0:
            continue
        overlap_by_speaker[turn.speaker] = (
            overlap_by_speaker.get(turn.speaker, 0) + overlap
        )

    if overlap_by_speaker:
        return max(overlap_by_speaker.items(), key=lambda item: item[1])[0]

    center = (s + e) // 2
    nearest: SpeakerTurn | None = None
    nearest_distance: int | None = None

    for turn in turns:
        turn_center = (turn.start_ms + turn.end_ms) // 2
        distance = abs(turn_center - center)
        if nearest is None or nearest_distance is None or distance < nearest_distance:
            nearest = turn
            nearest_distance = distance

    if nearest is None or nearest_distance is None or nearest_distance > 3_000:
        return None
    return nearest.speaker
