import asyncio
import json
import re
import struct
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging
import sys

import warnings
from collections import OrderedDict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import config
from .transcribe_worker import TranscribeWorker
from .hotwords import HotwordStore
from .metrics import profile_switch_total


# 3rdパーティ内部の一時的な RuntimeWarning を抑制（faster-whisper の mean 計算等）
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"Mean of empty slice\.")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"invalid value encountered in scalar divide")

root_logger = logging.getLogger()
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(handler)

try:
    _log_level = getattr(config, "LOG_LEVEL", "INFO")
    if isinstance(_log_level, str):
        _log_level_value = getattr(logging, _log_level.upper(), logging.INFO)
    else:
        _log_level_value = int(_log_level)
except Exception:
    _log_level_value = logging.INFO

root_logger.setLevel(_log_level_value)

app = FastAPI()


SESSION_ID_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,95})$")


def safe_session_id(raw: Optional[str], *, allow_generate: bool = False) -> Optional[str]:
    if not isinstance(raw, str):
        raw = ""
    raw = raw.strip()
    if SESSION_ID_RE.fullmatch(raw):
        return raw
    if allow_generate:
        return f"sess-{uuid.uuid4().hex}"
    return None


def resolve_transcript_path(session_id: str, suffix: str) -> Optional[Path]:
    sid = safe_session_id(session_id, allow_generate=False)
    if not sid:
        return None
    return (config.DATA_DIR / sid).with_suffix(suffix)


class SessionFiles:
    def __init__(self, session_id: str):
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        token = uuid.uuid4().hex[:4]
        self.session_id = f"{session_id}_{stamp}_{token}"
        self.base = config.DATA_DIR / self.session_id
        self.jsonl_path = self.base.with_suffix(".jsonl")
        self.txt_path = self.base.with_suffix(".txt")
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
        self._sync_files()

    def _sync_files(self):
        with self.jsonl_path.open("w", encoding="utf-8") as jf:
            for rec in self.records.values():
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with self.txt_path.open("w", encoding="utf-8") as tf:
            for rec in self.records.values():
                text = rec.get("text", "")
                speaker = rec.get("speaker")
                line = f"{speaker}: {text}" if speaker else text
                if not isinstance(line, str):
                    line = str(line)
                if not line.endswith("\n"):
                    line += "\n"
                tf.write(line)

    def write_final(self, text: str, ts_start: int, ts_end: int, segment_id, speaker: str | None = None):
        key = str(segment_id)
        self.records[key] = {
            "type": "final",
            "segmentId": key,
            "text": text,
            "tsStart": ts_start,
            "tsEnd": ts_end,
            **({"speaker": speaker} if speaker else {}),
        }
        self._sync_files()

    def write_overwrite(self, text: str, ts_start: int, ts_end: int, segment_id, speaker: str | None = None):
        key = str(segment_id)
        if not text.strip():
            self.records.pop(key, None)
        else:
            self.records[key] = {
                "type": "final",
                "segmentId": key,
                "text": text,
                "tsStart": ts_start,
                "tsEnd": ts_end,
                **({"speaker": speaker} if speaker else {}),
            }
        self._sync_files()

    def close(self):
        # ファイルは都度同期しているため追加処理は不要
        pass


sessions: Dict[str, Dict] = {}
client_views: Dict[str, WebSocket] = {}
client_counter = 0

async def broadcast_view_count():
    payload = {"type": "conn", "count": len(sessions), "viewers": len(client_views)}
    for ws in list(client_views.values()):
        try:
            await ws.send_json(payload)
        except Exception:
            pass
    for ent in list(sessions.values()):
        w = ent.get("ws")
        if w is None:
            continue
        try:
            await w.send_json(payload)
        except Exception:
            pass

async def broadcast_conn_count():
    try:
        payload = {"type": "conn", "count": len(sessions), "viewers": len(client_views)}
        for sid, ent in list(sessions.items()):
            w = ent.get("ws")
            if w is None:
                continue
            try:
                await w.send_json(payload)
            except Exception:
                pass
    except Exception:
        pass


@app.websocket(config.WS_PATH)
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    global client_counter
    client_counter += 1
    client_id = f"view-{client_counter}"
    client_views[client_id] = ws
    await broadcast_view_count()
    worker: Optional[TranscribeWorker] = None
    files: Optional[SessionFiles] = None
    session_id: Optional[str] = None
    ka_task: Optional[asyncio.Task] = None

    async def send_json(payload):
        # 送信失敗（切断）を握り潰してログノイズを防ぎつつ、ファイル書き込みは継続
        try:
            await ws.send_json(payload)
        except Exception:
            pass
        finally:
            if not files:
                return
            ptype = payload.get("type")
            if ptype == "final":
                try:
                    files.write_final(
                        text=payload.get("text", ""),
                        ts_start=int(payload.get("tsStart", 0)),
                        ts_end=int(payload.get("tsEnd", 0)),
                        segment_id=payload.get("segmentId", ""),
                        speaker=payload.get("speaker"),
                    )
                except Exception:
                    pass
            elif ptype == "overwrite":
                try:
                    files.write_overwrite(
                        text=payload.get("text", ""),
                        ts_start=int(payload.get("tsStart", 0)),
                        ts_end=int(payload.get("tsEnd", 0)),
                        segment_id=payload.get("segmentId", ""),
                        speaker=payload.get("speaker"),
                    )
                except Exception:
                    pass

    async def keepalive():
        try:
            while True:
                await asyncio.sleep(20)
                try:
                    await ws.send_json({"type": "ping", "ts": int(asyncio.get_event_loop().time() * 1000)})
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"]:
                data = json.loads(msg["text"])  # start/stop/info
                mtype = data.get("type")
                if mtype == "start":
                    raw_session = data.get("sessionId")
                    base_session = safe_session_id(raw_session, allow_generate=True)
                    while base_session in sessions:
                        base_session = safe_session_id(None, allow_generate=True)
                    opts = data.get("opts") or {}
                    files = SessionFiles(base_session)
                    session_id = files.session_id
                    while session_id in sessions:
                        files = SessionFiles(base_session)
                        session_id = files.session_id
                    await ws.send_json({"type": "info", "message": "backend_loading", "state": "loading"})
                    worker = TranscribeWorker(session_id, send_json, opts)
                    try:
                        profile_switch_total.labels(profile=worker.profile).inc()
                    except Exception:
                        pass
                    sessions[session_id] = {"worker": worker, "files": files, "ws": ws}
                    asyncio.create_task(worker.run())
                    if ka_task is None:
                        ka_task = asyncio.create_task(keepalive())
                    payload = {"type": "info", "message": "ready", "state": "ready", "sessionId": session_id}
                    backend_name = getattr(worker, 'backend_name', None)
                    if backend_name:
                        payload["backend"] = backend_name
                    if raw_session and raw_session != session_id:
                        payload["normalized"] = True
                    notice = getattr(worker, 'backend_notice', None)
                    if notice:
                        try:
                            await ws.send_json({"type": "info", "message": notice})
                        except Exception:
                            pass
                    await ws.send_json(payload)
                    await broadcast_conn_count()
                elif mtype == "calibrate":
                    dur = int((data.get("durationMs") or 2000))
                    if worker:
                        await worker.start_calibration(dur)
                        try:
                            await ws.send_json({"type": "info", "message": "calibrating"})
                        except Exception:
                            pass
                elif mtype == "stop":
                    if worker:
                        await worker.stop()
                        try:
                            await ws.send_json({"type": "info", "message": "stopping"})
                        except Exception:
                            pass
                        break
            elif "bytes" in msg and msg["bytes"]:
                b = msg["bytes"]
                if len(b) < 8:
                    continue
                seq, pts_ms = struct.unpack("<II", b[:8])
                payload = b[8:]
                if worker:
                    await worker.put_audio(payload, int(pts_ms))
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if client_id in client_views:
            client_views.pop(client_id, None)
        if worker:
            await worker.stop()
        if files:
            files.close()
        if session_id and session_id in sessions:
            sessions.pop(session_id, None)
            try:
                await ws.send_json({"type": "info", "message": "closed"})
            except Exception:
                pass
            await broadcast_conn_count()
        else:
            await broadcast_view_count()
        if ka_task:
            try:
                ka_task.cancel()
            except Exception:
                pass


@app.get("/api/transcript/{session_id}.txt")
async def get_txt(session_id: str):
    p = resolve_transcript_path(session_id, ".txt")
    if not p or not p.exists():
        return HTMLResponse(status_code=404, content="not found")
    return FileResponse(str(p), media_type="text/plain")


@app.get("/api/transcript/{session_id}.jsonl")
async def get_jsonl(session_id: str):
    p = resolve_transcript_path(session_id, ".jsonl")
    if not p or not p.exists():
        return HTMLResponse(status_code=404, content="not found")
    return FileResponse(str(p), media_type="application/jsonl")


# 追加API: Hotwords 管理 / SRT 変換（静的マウントより先に定義すること）
HOTWORDS_PATH = (config.DATA_DIR / "_hotwords.json")
_store = HotwordStore(HOTWORDS_PATH)

def _fmt_srt_time(ms: int) -> str:
    if ms < 0: ms = 0
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


@app.get("/api/hotwords")
async def get_hotwords():
    _store.maybe_reload()
    return {"words": _store.words}


@app.post("/api/hotwords")
async def set_hotwords(payload: dict):
    words = payload.get("words")
    if not isinstance(words, list):
        text = payload.get("text", "")
        if isinstance(text, str):
            words = [w.strip() for w in text.splitlines() if w.strip()]
        else:
            words = []
    _store.save(words)
    return {"ok": True, "count": len(words)}


@app.get("/api/transcript/{session_id}.srt")
async def get_srt(session_id: str):
    p = resolve_transcript_path(session_id, ".jsonl")
    if not p or not p.exists():
        return HTMLResponse(status_code=404, content="not found")
    lines = p.read_text(encoding="utf-8").splitlines()
    out = []
    idx = 1
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") != "final":
            continue
        text = rec.get("text", "").strip()
        if not text:
            continue
        ts0 = int(rec.get("tsStart", 0))
        ts1 = int(rec.get("tsEnd", ts0))
        spk = rec.get("speaker")
        if spk:
            text = f"{spk}: {text}"
        out.append(str(idx))
        out.append(f"{_fmt_srt_time(ts0)} --> {_fmt_srt_time(ts1)}")
        out.append(text)
        out.append("")
        idx += 1
    content = "\n".join(out)
    return HTMLResponse(status_code=200, content=content, media_type="text/plain; charset=utf-8")


@app.get("/metrics")
async def metrics():
    payload = generate_latest()
    return Response(status_code=200, content=payload, media_type=CONTENT_TYPE_LATEST)

# 静的ファイルは最後にマウント（他のルートを優先させるため）
WEB_DIR = Path("web")
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
