# backend/server.py
"""
Whistx v2 WebSocketサーバー
音声ストリーミング・文字起こし・LLM分析

スレッドセーフ：複数接続に対応
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from transcription import get_transcriber, TranscriptionContext
from llm_service import get_llm_service

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPIアプリ
app = FastAPI(title="Whistx v2")

# 静的ファイルのマウント（オプション）
# app.mount("/static", StaticFiles(directory="static"), name="static")

# 環境変数で設定を上書き可能
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "10"))  # 最大同時接続数
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # セッションタイムアウト（秒）

# グローバル状態（スレッドセーフ）
active_connections: Set[WebSocket] = set()
sessions: Dict[str, dict] = {}
connection_counts: Dict[str, int] = defaultdict(int)  # IPごとの接続数


@app.get("/")
async def get_root():
    """ルートエンドポイント"""
    return {
        "name": "Whistx v2",
        "version": "2.0.0",
        "status": "running",
        "active_connections": len(active_connections),
    }


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    音声文字起こしWebSocketエンドポイント

    メッセージ形式:
    - バイナリ: 音声データ (PCM16, 16kHz, モノラル)
    - JSON: 制御コマンド
      - {"type": "start", "sessionId": "...", "opts": {...}}
      - {"type": "stop"}
      - {"type": "analyze", "sessionId": "..."}
    """
    # クライアントIPアドレスを取得
    client_host = websocket.client.host if websocket.client else "unknown"

    # 接続数制限チェック
    if len(active_connections) >= MAX_CONNECTIONS:
        logger.warning(f"Connection rejected: max connections ({MAX_CONNECTIONS}) reached from {client_host}")
        await websocket.close(code=1013, reason="Server at maximum capacity")
        return

    await websocket.accept()
    active_connections.add(websocket)
    connection_counts[client_host] += 1
    session_id = None

    logger.info(f"Connection accepted from {client_host} (total: {len(active_connections)}/{MAX_CONNECTIONS})")

    try:
        # セッションIDの生成
        session_id = f"sess-{datetime.now().strftime('%Y%m%d%H%M%S')}-{os.urandom(2).hex()}"

        # トランスライバーの初期化（スレッドセーフ）
        transcriber = get_transcriber()
        context = TranscriptionContext(history_size=10)

        # セッション情報の初期化
        sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now().isoformat(),
            "client_host": client_host,
            "transcripts": [],
            "status": "connected",
        }

        # 接続確立通知
        await websocket.send_json({
            "type": "info",
            "sessionId": session_id,
            "message": "ready",
            "backend": "faster-whisper-large-v3",
            "maxConnections": MAX_CONNECTIONS,
            "currentConnections": len(active_connections),
        })

        logger.info(f"Session {session_id} connected from {client_host}")

        # メッセージループ
        async for message in websocket.iter_json() if isinstance(message, dict) else None:
            if message is None:
                continue

            msg_type = message.get("type")

            if msg_type == "start":
                # 録音開始
                await handle_start(websocket, message, transcriber, context, session_id)

            elif msg_type == "stop":
                # 録音停止
                await handle_stop(websocket, session_id)
                break

            elif msg_type == "analyze":
                # LLM分析
                await handle_analyze(websocket, session_id)

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })

    except WebSocketDisconnect:
        logger.info(f"Session {session_id} disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

    finally:
        # 接続クリーンアップ
        active_connections.discard(websocket)
        connection_counts[client_host] = max(0, connection_counts[client_host] - 1)

        if session_id:
            if session_id in sessions:
                sessions[session_id]["status"] = "disconnected"
                sessions[session_id]["disconnected_at"] = datetime.now().isoformat()

            logger.info(
                f"Session {session_id} disconnected from {client_host} "
                f"(remaining: {len(active_connections)}/{MAX_CONNECTIONS})"
            )


async def handle_start(
    websocket: WebSocket,
    message: dict,
    transcriber,
    context: TranscriptionContext,
    session_id: str,
):
    """録音開始処理"""
    opts = message.get("opts", {})
    language = opts.get("language", "ja")

    await websocket.send_json({
        "type": "info",
        "state": "ready",
        "message": "Recording started",
        "language": language,
    })

    logger.info(f"Session {session_id} started recording")


async def handle_stop(websocket: WebSocket, session_id: str):
    """録音停止処理"""
    await websocket.send_json({
        "type": "info",
        "state": "stopped",
        "message": "Recording stopped",
    })

    if session_id in sessions:
        sessions[session_id]["status"] = "stopped"

    logger.info(f"Session {session_id} stopped recording")


async def handle_analyze(websocket: WebSocket, session_id: str):
    """LLM分析処理"""
    try:
        if session_id not in sessions:
            await websocket.send_json({
                "type": "error",
                "message": "Session not found"
            })
            return

        # トランスクリプトを取得
        transcripts = sessions[session_id].get("transcripts", [])

        if not transcripts:
            await websocket.send_json({
                "type": "error",
                "message": "No transcripts to analyze"
            })
            return

        # LLMサービスを取得
        llm_service = get_llm_service()

        # 分析実行
        logger.info(f"Starting LLM analysis for session {session_id} ({len(transcripts)} transcripts)")
        result = llm_service.analyze(transcripts)

        # 結果を送信
        await websocket.send_json({
            "type": "analysis",
            "sessionId": session_id,
            "result": result,
        })

        logger.info(f"LLM analysis completed for session {session_id}")

    except Exception as e:
        logger.error(f"LLM analysis error: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": f"Analysis failed: {str(e)}"
        })


@app.get("/api/sessions")
async def list_sessions():
    """セッション一覧取得"""
    session_list = [
        {
            "session_id": sid,
            "created_at": s["created_at"],
            "client_host": s.get("client_host", "unknown"),
            "status": s["status"],
            "transcript_count": len(s.get("transcripts", [])),
        }
        for sid, s in sessions.items()
    ]
    return {
        "sessions": sorted(session_list, key=lambda x: x["created_at"], reverse=True),
        "total": len(session_list)
    }


@app.get("/api/status")
async def get_status():
    """システム状態取得（監視用）"""
    return {
        "status": "running",
        "active_connections": len(active_connections),
        "max_connections": MAX_CONNECTIONS,
        "total_sessions": len(sessions),
        "connection_by_ip": dict(connection_counts),
        "whisper_model": os.getenv("WHISPER_MODEL", "large-v3"),
        "llm_model": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    }


@app.get("/api/transcript/{session_id}")
async def get_transcript(session_id: str):
    """トランスクリプト取得"""
    if session_id not in sessions:
        return {"error": "Session not found"}

    return sessions[session_id]


def broadcast_connection_count():
    """接続数を全クライアントにブロードキャスト"""
    message = {
        "type": "conn",
        "count": len(active_connections),
    }

    # 全接続に送信
    for ws in list(active_connections):
        try:
            asyncio.create_task(ws.send_json(message))
        except Exception:
            active_connections.discard(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info",
    )
