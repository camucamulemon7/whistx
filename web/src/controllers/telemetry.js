import { formatAudioSource as formatModeLabel } from "../ui/format.js";

export function createTelemetryController(appDependencies) {
function updateRecordingTelemetry() {
  if (!appDependencies.recordTelemetryEl) return;
  if (!appDependencies.state.recording) {
    appDependencies.recordTelemetryEl.hidden = true;
    appDependencies.recordTelemetryEl.textContent = "";
    return;
  }

  const requested = formatModeLabel(appDependencies.state.recordingRequestedAudioSource || appDependencies.state.recordingAudioSource || "mic");
  const effective = formatModeLabel(appDependencies.state.recordingAudioSource || "mic");
  const sourceText = requested === effective ? `入力: ${effective}` : `入力: ${requested} → ${effective}`;
  const vadFrames = Math.max(0, appDependencies.state.vadFrameCount);
  const speechRatio = vadFrames > 0 ? appDependencies.state.vadSpeechFrameCount / vadFrames : 0;
  const vadText = appDependencies.state.vadAnalyser ? `VAD ${(speechRatio * 100).toFixed(0)}%` : "VAD n/a";
  const gainText = appDependencies.state.captureGainNode ? `GAIN ${appDependencies.state.captureAutoGainLevel.toFixed(1)}x` : "GAIN 1.0x";
  const chunkText = appDependencies.state.recordedChunkCount > 0 ? `CHUNK ${appDependencies.state.recordedChunkCount}` : "CHUNK 0";
  const elapsedMs = appDependencies.state.recordingStartedAt ? Math.max(0, performance.now() - appDependencies.state.recordingStartedAt) : 0;
  const elapsedText = `TIME ${(elapsedMs / 1000).toFixed(1)}s`;
  const fallbackText = appDependencies.state.recordingFallbackReason ? `FALLBACK ${appDependencies.state.recordingFallbackReason}` : "";

  appDependencies.recordTelemetryEl.hidden = false;
  appDependencies.recordTelemetryEl.textContent = [sourceText, vadText, gainText, chunkText, elapsedText, fallbackText].filter(Boolean).join(" / ");
}

function logWsEvent(event, detail = {}) {
  console.info(`[whistx][ws] ${event}`, detail);
}

function logClientEvent(event, detail = {}) {
  console.info(`[whistx][client] ${event}`, detail);
}

function sendWsTelemetry(event, detail = {}) {
  const ws = appDependencies.state.ws;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  try {
    ws.send(JSON.stringify({
      type: "telemetry",
      event,
      detail,
    }));
  } catch {
    // ignore telemetry send failures
  }
}

function bucketizeBacklog(value) {
  if (value >= appDependencies.BACKLOG_DANGER_THRESHOLD) return "danger";
  if (value >= appDependencies.BACKLOG_WARN_THRESHOLD) return "warn";
  return "normal";
}

function emitScreenshotSkipTelemetry(reason, extra = {}) {
  const now = Date.now();
  const sameReason = appDependencies.state.lastScreenshotSkipReason === reason;
  const sentRecently = now - appDependencies.state.lastScreenshotSkipSentAt < 15_000;
  if (sameReason && sentRecently) return;
  appDependencies.state.lastScreenshotSkipReason = reason;
  appDependencies.state.lastScreenshotSkipSentAt = now;
  sendWsTelemetry("screenshot_skipped", {
    reason,
    skippedCount: appDependencies.state.screenshotSkippedCount,
    backlog: appDependencies.state.pendingOutboundChunks,
    ...extra,
  });
}

function updateBackpressureState() {
  appDependencies.state.maxObservedBacklog = Math.max(appDependencies.state.maxObservedBacklog, appDependencies.state.pendingOutboundChunks);
  const nextBucket = bucketizeBacklog(appDependencies.state.pendingOutboundChunks);
  if (nextBucket !== appDependencies.state.lastTelemetryBacklogBucket) {
    appDependencies.state.lastTelemetryBacklogBucket = nextBucket;
    sendWsTelemetry("client_backlog", {
      backlog: appDependencies.state.pendingOutboundChunks,
      maxObservedBacklog: appDependencies.state.maxObservedBacklog,
      bucket: nextBucket,
    });
  }
  const nextDegraded = appDependencies.state.pendingOutboundChunks >= appDependencies.BACKLOG_WARN_THRESHOLD;
  if (nextDegraded !== appDependencies.state.degradedCaptureMode) {
    appDependencies.state.degradedCaptureMode = nextDegraded;
    sendWsTelemetry(nextDegraded ? "degraded_capture_enabled" : "degraded_capture_cleared", {
      backlog: appDependencies.state.pendingOutboundChunks,
      maxObservedBacklog: appDependencies.state.maxObservedBacklog,
    });
  }
}

  return { updateRecordingTelemetry, logWsEvent, logClientEvent, sendWsTelemetry, bucketizeBacklog, emitScreenshotSkipTelemetry, updateBackpressureState };
}
