import { buildWebSocketUrl } from "../transcription/websocket.js";
import { waitForOpen } from "../transcription/websocket.js";

export function createTranscriptionController(appDependencies) {
function wsUrl() {
  return buildWebSocketUrl(location, appDependencies.state.wsPath);
}

async function ensureSocket() {
  if (appDependencies.state.ws && appDependencies.state.ws.readyState === WebSocket.OPEN) {
    return appDependencies.state.ws;
  }

  if (appDependencies.state.ws && appDependencies.state.ws.readyState === WebSocket.CONNECTING) {
    await waitForOpen(appDependencies.state.ws);
    return appDependencies.state.ws;
  }

  const ws = new WebSocket(wsUrl());
  appDependencies.state.ws = ws;
  appDependencies.logWsEvent("connect", { url: wsUrl() });

  ws.addEventListener("message", (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }

    appDependencies.logWsEvent("message", { type: data.type, message: data.message || "", seq: data.seq ?? null });
    const incomingSessionId = String(data.sessionId || "");
    const isReadyMessage = data.type === "info" && data.message === "ready";
    if (
      incomingSessionId &&
      !isReadyMessage &&
      appDependencies.state.runtimeSessionId &&
      incomingSessionId !== appDependencies.state.runtimeSessionId
    ) {
      appDependencies.logWsEvent("ignore_stale_message", {
        incomingSessionId,
        currentSessionId: appDependencies.state.runtimeSessionId,
        type: data.type,
      });
      return;
    }

    if (data.type === "conn") {
      return;
    }

    if (data.type === "info") {
      if (data.message) appDependencies.setStatus(String(data.message));
      if (data.sessionId) {
        appDependencies.state.runtimeSessionId = String(data.sessionId);
        appDependencies.state.runtimeSessionToken = String(data.sessionToken || "");
      }
      if (data.message === "ready") {
        appDependencies.state.runtimeSessionFinalized = false;
      } else if (data.message === "finalized") {
        appDependencies.state.runtimeSessionFinalized = true;
      }
      appDependencies.updateDownloadLinks();
      return;
    }

    if (data.type === "final") {
      appDependencies.addLogLine(
        String(data.text || ""),
        Number(data.tsStart || 0),
        Number(data.tsEnd || 0),
        Number(data.seq),
        String(data.speaker || ""),
        String(data.screenshotPath || ""),
        String(data.rawAudioPath || ""),
        String(data.audioPath || "")
      );
      return;
    }

    if (data.type === "speaker_patch") {
      appDependencies.applySpeakerPatch(data.segments || []);
      return;
    }

    if (data.type === "error") {
      const detail = data.detail ? ` (${data.detail})` : "";
      if (data.message === "server_busy") {
        appDependencies.state.degradedCaptureMode = true;
        appDependencies.sendWsTelemetry("server_busy_acknowledged", {
          detail: data.detail || "",
          backlog: appDependencies.state.pendingOutboundChunks,
        });
        appDependencies.showToast("サーバ処理が詰まっています。画面キャプチャを自動で抑制します", "error", 5000);
      } else if (data.message === "transcription_failed") {
        appDependencies.sendWsTelemetry("transcription_failed_acknowledged", {
          seq: data.seq ?? "",
          buffered: !!data.buffered,
        });
        appDependencies.showToast("文字起こし処理で一時エラーが発生しました", "error", 4000);
      }
      appDependencies.setStatus(`error: ${data.message || "unknown"}${detail}`);
    }
  });

  ws.addEventListener("close", () => {
    appDependencies.logWsEvent("close");
    if (ws.__whistxGracefulStop) {
      if (appDependencies.state.ws === ws) {
        appDependencies.state.ws = null;
      }
      return;
    }
    appDependencies.abortRecordingAfterSocketLoss(ws, "connection_lost");
  });

  ws.addEventListener("error", () => {
    appDependencies.logWsEvent("error");
    appDependencies.abortRecordingAfterSocketLoss(ws, "socket_error");
  });

  await waitForOpen(ws);
  appDependencies.logWsEvent("open");
  return ws;
}

  return { wsUrl, ensureSocket };
}
