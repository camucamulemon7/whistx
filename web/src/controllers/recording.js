import { arrayBufferToBase64 } from "../transcription/websocket.js";
import { shouldCutOnSilence } from "../audio/vad.js";
import { VAD_SAMPLE_MS } from "../audio/vad.js";
import { waitForSessionFinalized } from "../transcription/websocket.js";
import { formatTimestamp as formatMs } from "../ui/format.js";
import { LiveCapture } from "../meeting/live-capture.js";
import { applyTranscriptRevision } from "../transcription/revisions.js";
import { readSseJsonStream } from "../api/sse.js";
import { normalizeAudioSource } from "../audio/vad.js";
import { waitForSessionReady } from "../transcription/websocket.js";

export function createRecordingController(appDependencies) {
function selectMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];

  for (const candidate of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(candidate)) {
      return candidate;
    }
  }
  return "";
}

function generateSessionSeed() {
  const t = Date.now().toString(36);
  const r = Math.random().toString(36).slice(2, 6);
  return `sess-${t}-${r}`;
}

function setUiRecording(active) {
  appDependencies.state.recording = active;
  appDependencies.state.recordingPhase = active ? "recording" : "idle";

  if (appDependencies.audioLevelIndicatorEl) {
    appDependencies.audioLevelIndicatorEl.hidden = !active;
  }
  if (!active) {
    appDependencies.renderAudioLevel(0);
  }

  if (active) {
    appDependencies.startBtn.classList.add("is-recording");
    appDependencies.startBtn.querySelector(".record-label").textContent = "停止";
    appDependencies.startBtn.setAttribute("aria-pressed", "true");
    appDependencies.startBtn.setAttribute("aria-label", "録音を停止");
    appDependencies.startBtn.setAttribute("aria-busy", "false");
  } else {
    appDependencies.startBtn.classList.remove("is-recording");
    appDependencies.startBtn.querySelector(".record-label").textContent = "録音開始";
    appDependencies.startBtn.setAttribute("aria-pressed", "false");
    appDependencies.startBtn.setAttribute("aria-label", "録音を開始");
    appDependencies.startBtn.setAttribute("aria-busy", "false");

    if (appDependencies.state.log.length > 0) {
      appDependencies.startBtn.classList.add("is-complete");
      setTimeout(() => appDependencies.startBtn.classList.remove("is-complete"), 800);
    }
  }

  appDependencies.startBtn.disabled = false;
  appDependencies.updateSaveControls();
  appDependencies.updateDownloadLinks();
  appDependencies.setSessionSettingsLocked();
  appDependencies.syncUnloadProtection();
  appDependencies.renderHistoryList();
}

function setUiRecordingStarting() {
  appDependencies.state.recordingPhase = "starting";
  appDependencies.startBtn.disabled = true;
  appDependencies.startBtn.classList.remove("is-recording");
  appDependencies.startBtn.querySelector(".record-label").textContent = "準備中...";
  appDependencies.startBtn.setAttribute("aria-pressed", "false");
  appDependencies.startBtn.setAttribute("aria-label", "録音を準備中");
  appDependencies.startBtn.setAttribute("aria-busy", "true");
  appDependencies.setStatus("starting");
  appDependencies.updateSaveControls();
  appDependencies.updateDownloadLinks();
  appDependencies.setSessionSettingsLocked(true);
  appDependencies.syncUnloadProtection();
  appDependencies.renderHistoryList();
}

function setUiRecordingStopping(finalizing = false) {
  appDependencies.state.recordingPhase = finalizing ? "finalizing" : "stopping";
  appDependencies.startBtn.disabled = true;
  appDependencies.startBtn.querySelector(".record-label").textContent = finalizing ? "最終処理中..." : "停止中...";
  appDependencies.startBtn.setAttribute("aria-pressed", "false");
  appDependencies.startBtn.setAttribute("aria-label", finalizing ? "録音の最終処理中" : "録音を停止中");
  appDependencies.startBtn.setAttribute("aria-busy", "true");
  appDependencies.setStatus(finalizing ? "finalizing" : "stopping");
  appDependencies.updateSaveControls();
  appDependencies.updateDownloadLinks();
  appDependencies.setSessionSettingsLocked(true);
  appDependencies.syncUnloadProtection();
  appDependencies.renderHistoryList();
}

function resetRuntimeSessionState() {
  appDependencies.state.runtimeSessionId = "";
  appDependencies.state.runtimeSessionToken = "";
  appDependencies.state.runtimeSessionFinalized = false;
}

function commitNewRecordingWorkspace() {
  appDependencies.state.historyDetailController?.abort("workspace_changed");
  appDependencies.state.historyDetailController = null;
  appDependencies.state.historyDetailRequestVersion += 1;
  appDependencies.state.history.selectedId = null;
  appDependencies.state.savedHistoryId = null;
  appDependencies.state.viewingHistoryId = null;
  appDependencies.state.log = [];
  appDependencies.state.segments = [];
  appDependencies.markWorkspaceClean();
  appDependencies.state.logAutoScrollEnabled = true;
  if (appDependencies.saveTitleInputEl) {
    appDependencies.saveTitleInputEl.value = "";
  }
  appDependencies.renderEmptyTranscriptState();
  appDependencies.setSummary("", "未実行");
  appDependencies.setProofread("", "未実行");
  appDependencies.updateSegmentCount();
  appDependencies.updateDownloadLinks();
  appDependencies.updateSaveControls();
  appDependencies.renderHistoryList();
}

async function sendChunk(blob, mimeType, durationMsOverride, vadDecision) {
  if (!appDependencies.state.ws || appDependencies.state.ws.readyState !== WebSocket.OPEN) return;

  let durationMs = Number(durationMsOverride);
  if (!Number.isFinite(durationMs)) {
    durationMs = appDependencies.state.chunkMs;
  }
  durationMs = Math.max(200, Math.round(durationMs));

  const offsetMs = appDependencies.state.offsetMs;
  appDependencies.state.offsetMs += durationMs;

  if (!blob || blob.size === 0) return;

  if (appDependencies.shouldSkipChunkByVad(durationMs, vadDecision)) {
    return;
  }

  appDependencies.state.pendingOutboundChunks += 1;
  appDependencies.updateBackpressureState();
  const seq = appDependencies.state.seq++;
  appDependencies.state.recordedChunkCount += 1;
  try {
    const buffer = await blob.arrayBuffer();
    const audio = arrayBufferToBase64(buffer);
    const screenshot = await appDependencies.captureDisplayScreenshot();

    appDependencies.state.ws.send(
      JSON.stringify({
        type: "chunk",
        seq,
        offsetMs,
        durationMs,
        mimeType: mimeType || blob.type || "audio/webm",
        audio,
        screenshot: screenshot?.data || "",
        screenshotMimeType: screenshot?.mimeType || "",
        speechRatio: Number.isFinite(vadDecision?.speechRatio) ? vadDecision.speechRatio : null,
        activeMs: Number.isFinite(vadDecision?.activeMs) ? vadDecision.activeMs : null,
        silenceMs: Number.isFinite(vadDecision?.silenceMs) ? vadDecision.silenceMs : null,
      })
    );
    appDependencies.logWsEvent("send_chunk", {
      seq,
      durationMs,
      offsetMs,
      bytes: blob.size,
      screenshot: screenshot ? "sent" : "skipped",
      backlog: appDependencies.state.pendingOutboundChunks,
    });
  } finally {
    appDependencies.state.pendingOutboundChunks = Math.max(0, appDependencies.state.pendingOutboundChunks - 1);
    appDependencies.updateBackpressureState();
  }
}

function clearChunkTimer() {
  if (appDependencies.state.chunkTimer) {
    clearTimeout(appDependencies.state.chunkTimer);
    appDependencies.state.chunkTimer = null;
  }
}

function chunkHardMaxMs() {
  const mode = appDependencies.currentVadSourceMode();
  const sourceExtra = mode === "display" ? 2_500 : mode === "both" ? 1_500 : 0;
  return Math.max(appDependencies.state.chunkMs + 2_000, appDependencies.state.chunkMs + appDependencies.VAD_SOFT_CUT_GRACE_MS + sourceExtra);
}

function shouldCutChunkOnSilence(options = {}) {
  if (!appDependencies.state.vadAnalyser || !appDependencies.state.segmentStartedAt) return false;

  const now = performance.now();
  const elapsedMs = now - appDependencies.state.segmentStartedAt;
  const silenceMs = Math.max(0, now - (appDependencies.state.vadLastSpeechAt || appDependencies.state.segmentStartedAt));
  return shouldCutOnSilence({
    elapsedMs,
    silenceMs,
    chunkMs: appDependencies.state.chunkMs,
    sourceMode: appDependencies.currentVadSourceMode(),
    relaxed: !!options.relaxed,
    minimumMs: appDependencies.VAD_SEGMENT_MIN_MS,
  });
}

function requestChunkFlush(recorder) {
  if (!appDependencies.state.recording) return;
  if (appDependencies.state.recorder !== recorder) return;
  if (recorder.state !== "recording") return;

  try {
    recorder.stop();
  } catch {
    // ignore
  }
}

function scheduleChunkStop(recorder) {
  clearChunkTimer();
  const check = () => {
    if (!appDependencies.state.recording || appDependencies.state.recorder !== recorder || recorder.state !== "recording") {
      clearChunkTimer();
      return;
    }

    const elapsedMs = Math.max(0, performance.now() - appDependencies.state.segmentStartedAt);
    if (elapsedMs >= chunkHardMaxMs()) {
      requestChunkFlush(recorder);
      return;
    }

    if (shouldCutChunkOnSilence({ relaxed: elapsedMs >= appDependencies.state.chunkMs })) {
      requestChunkFlush(recorder);
      return;
    }

    appDependencies.state.chunkTimer = setTimeout(check, Math.min(250, Math.max(120, VAD_SAMPLE_MS)));
  };

  appDependencies.state.chunkTimer = setTimeout(check, Math.min(250, Math.max(120, VAD_SAMPLE_MS)));
}

function startRecorderCycle() {
  if (!appDependencies.state.recording || !appDependencies.state.stream) return;

  const recorder = new MediaRecorder(appDependencies.state.stream, appDependencies.state.recorderOptions || {});
  appDependencies.state.recorder = recorder;
  const cycleStartedAt = performance.now();
  appDependencies.state.segmentStartedAt = cycleStartedAt;
  appDependencies.state.vadLastSpeechAt = cycleStartedAt;
  const vadSnapshot = appDependencies.snapshotVadCounters();

  recorder.addEventListener("dataavailable", (event) => {
    const eventEndedAt = performance.now();
    const durationMs = Math.max(200, Math.round(eventEndedAt - cycleStartedAt));
    const vadDecision = appDependencies.buildVadDecision(vadSnapshot, eventEndedAt, appDependencies.state.recordingAudioSource);

    appDependencies.state.pendingSendChain = appDependencies.state.pendingSendChain
      .then(() => sendChunk(event.data, appDependencies.state.recorderMimeType, durationMs, vadDecision))
      .catch((err) => {
        appDependencies.setStatus(`chunk_error: ${err.message}`);
      });
  });

  recorder.addEventListener("error", () => {
    appDependencies.setStatus("recorder_error");
    appDependencies.state.recording = false;
    finalizeStop();
  });

  recorder.addEventListener("stop", () => {
    if (appDependencies.state.recorder === recorder) {
      appDependencies.state.recorder = null;
    }
    if (recorder.__whistxAbortWithoutFinalize) {
      return;
    }
    if (appDependencies.state.recording) {
      startRecorderCycle();
      return;
    }
    finalizeStop();
  });

  recorder.start();
  scheduleChunkStop(recorder);
}

async function finalizeStop() {
  if (appDependencies.state.finalizingStop) return;
  appDependencies.state.finalizingStop = true;
  setUiRecordingStopping(true);
  clearChunkTimer();
  let completed = false;
  let finalizeError = null;

  try {
    await appDependencies.state.pendingSendChain;
    const ws = appDependencies.state.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      throw new Error("websocket_unavailable_before_finalize");
    }
    ws.__whistxGracefulStop = true;
    const finalized = waitForSessionFinalized(ws, appDependencies.state.runtimeSessionId);
    ws.send(JSON.stringify({ type: "stop" }));
    appDependencies.logWsEvent("send_stop", { sessionId: appDependencies.state.runtimeSessionId || "" });
    await finalized;
    completed = true;
    if (ws.readyState === WebSocket.OPEN) {
      ws.close(1000, "session_finalized");
    }
    if (appDependencies.state.ws === ws) {
      appDependencies.state.ws = null;
    }
  } catch (error) {
    finalizeError = error;
  } finally {
    appDependencies.state.runtimeSessionFinalized = completed;
    appDependencies.state.pendingSendChain = Promise.resolve();
    appDependencies.cleanupMedia();
    setUiRecording(false);
    appDependencies.state.segmentStartedAt = 0;
    appDependencies.state.recordingStartedAt = 0;
    appDependencies.state.recordingRequestedAudioSource = "mic";
    appDependencies.state.recordingFallbackReason = "";
    appDependencies.state.recordedChunkCount = 0;
    appDependencies.updateRecordingTelemetry();
    appDependencies.state.finalizingStop = false;
    appDependencies.updateSaveControls();
    appDependencies.updateDownloadLinks();
    appDependencies.setSessionSettingsLocked(false);
    appDependencies.syncUnloadProtection();
    appDependencies.renderHistoryList();
    if (completed) {
      appDependencies.setStatus("completed");
      appDependencies.showToast("録音の最終処理が完了しました", "success");
    } else {
      appDependencies.setStatus("finalize_failed");
      appDependencies.showToast(`録音の最終処理に失敗しました: ${finalizeError?.message || "unknown"}`, "error", 7000);
    }
  }
}

function currentMeetingSource() {
  if (!appDependencies.state.meetingInsights || !appDependencies.state.auth.authenticated) return null;
  if (appDependencies.state.liveCapture?.running && appDependencies.state.liveCapture.sessionId) return { runtimeSessionId: appDependencies.state.liveCapture.sessionId };
  const historyId = appDependencies.state.viewingHistoryId || appDependencies.state.savedHistoryId;
  return historyId ? { historyId } : appDependencies.state.runtimeSessionId ? { runtimeSessionId: appDependencies.state.runtimeSessionId } : null;
}

function navigateMeetingSource(source) {
  const rows = [...appDependencies.logEl.querySelectorAll(".log-row")];
  const row = rows.find((item) => item.dataset.segmentId === source.id)
    || rows.find((item) => Number(item.dataset.seq) === Number(source.seq));
  if (!row) {
    appDependencies.showToast(`${formatMs(source.startMs)}: ${source.text}`, "default", 10000);
    return;
  }
  appDependencies.state.logAutoScrollEnabled = false;
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  row.focus({ preventScroll: true });
  row.classList.add("meeting-highlight");
  setTimeout(() => row.classList.remove("meeting-highlight"), 3000);
}

async function startLiveRecording(health, selectedAudioSource) {
  appDependencies.state.liveCapture?.dispose();
  document.querySelector("#hqStatus").hidden = true;
  appDependencies.state.liveCapture = null;
  if (appDependencies.state.ws) {
    appDependencies.state.ws.__whistxGracefulStop = true;
    appDependencies.state.ws.close();
    appDependencies.state.ws = null;
  }
  const stream = await appDependencies.prepareInputStream(selectedAudioSource);
  if (!appDependencies.hasAudioTrack(stream)) throw new Error("audio_track_not_found");
  appDependencies.state.stream = stream;
  await appDependencies.setupVad(stream);
  const separate = appDependencies.state.recordingAudioSource === "both" && appDependencies.hasAudioTrack(appDependencies.state.micStream) && appDependencies.hasAudioTrack(appDependencies.state.displayStream);
  const streams = separate ? { mic: appDependencies.state.micStream, display: appDependencies.state.displayStream } : { [appDependencies.state.recordingAudioSource === "display" ? "display" : "mic"]: stream };
  const superseded = new Set();
  const renderRow = (row) => appDependencies.addLogLine(row.text, row.tsStart, row.tsEnd, row.seq, row.speaker || "", row.screenshotPath || "", row.rawAudioPath || "", row.audioPath || "", row.segmentId, row);
  const reconcileRows = (records) => {
    const wanted = new Map(records.map((row) => [row.segmentId, row]));
    const changed = new Set(appDependencies.state.segments.filter((row) => {
      const replacement = wanted.get(row.segmentId);
      return replacement && ["text", "speaker", "tsStart", "tsEnd", "rawAudioPath", "screenshotPath", "quality"]
        .some((key) => row[key] !== replacement[key]);
    }).map((row) => row.segmentId));
    for (const row of appDependencies.logEl.querySelectorAll(".log-row")) {
      if (!wanted.has(row.dataset.segmentId) || changed.has(row.dataset.segmentId)) row.remove();
    }
    appDependencies.state.segments = appDependencies.state.segments.filter((row) => wanted.has(row.segmentId) && !changed.has(row.segmentId));
    for (const row of records) renderRow(row);
    appDependencies.state.segments = records;
    appDependencies.state.log = records.map((row) => row.text);
    const nodes = new Map([...appDependencies.logEl.querySelectorAll(".log-row")].map((row) => [row.dataset.segmentId, row]));
    for (const row of records) {
      const node = nodes.get(row.segmentId);
      if (node) {
        node.dataset.quality = row.quality || "";
        node.title = row.retainedRealtimeSegmentIds?.length ? "再認識済み・一部の区間は言語の脱落を避けるため速報を維持" : row.quality === "high_accuracy" ? "長区間の音声から再認識済み" : "リアルタイム認識";
        appDependencies.logEl.append(node);
      }
    }
    appDependencies.renderTranscriptParagraphs();
    appDependencies.updateSegmentCount();
    appDependencies.markProofreadStale();
    appDependencies.markWorkspaceDirty();
  };
  const capture = new LiveCapture({
    path: health.liveWsPath,
    packetMs: health.capturePacketMs,
    streams,
    start: { sessionId: generateSessionSeed(), language: appDependencies.selectedLanguage() || "auto", audioSource: appDependencies.state.recordingAudioSource,
      diarizationEnabled: !!appDependencies.diarizationToggleEl?.checked,
      ...appDependencies.resolveDiarizationStartOptions(),
      prompt: appDependencies.promptEl.value.trim(), sharedVocabulary: String(appDependencies.sharedVocabularyEl?.value || appDependencies.state.sharedVocabulary || "").trim() },
    screenshot: appDependencies.captureDisplayScreenshot,
    onEvent: (data) => {
      if (appDependencies.state.liveCapture !== capture) return;
      appDependencies.meetingWorkspace.liveEvent(data);
      if (data.type === "info") {
        appDependencies.state.runtimeSessionId = data.sessionId || appDependencies.state.runtimeSessionId;
        if (data.message === "ready") appDependencies.state.runtimeSessionFinalized = !!data.finalized;
        if (data.message === "finalized") appDependencies.state.runtimeSessionFinalized = true;
        appDependencies.updateDownloadLinks();
      } else if (data.type === "transcript_snapshot") {
        for (const row of data.records) {
          for (const original of row.realtimeSegments || []) superseded.add(original.segmentId);
        }
        reconcileRows(data.records);
      } else if (data.type === "transcript_revision") {
        try {
          const updated = applyTranscriptRevision(appDependencies.state.segments, data);
          for (const id of data.replacesSegmentIds) superseded.add(id);
          reconcileRows(updated);
        } catch {
          document.querySelector("#liveConnection").textContent = "再認識結果の同期に失敗しました。再接続で復元できます。";
        }
      } else if (data.type === "hq_status") {
        const status = document.querySelector("#hqStatus");
        status.hidden = false;
        status.dataset.state = data.state;
        status.textContent = data.state === "running" ? `高精度認識中 ${formatMs(data.tsStart)}–${formatMs(data.tsEnd)}`
          : data.state === "completed" ? `高精度認識を反映 ${formatMs(data.tsStart)}–${formatMs(data.tsEnd)}` : data.message;
      } else if (data.type === "final") {
        if (!superseded.has(data.segmentId)) renderRow(data);
      } else if (data.type === "speaker_patch") {
        appDependencies.applySpeakerPatch(data.segments || []);
      } else if (data.type === "capture_state") {
        appDependencies.state.pendingOutboundChunks = Math.ceil(data.pendingBytes / 32000);
        appDependencies.state.offsetMs = data.samples / 16;
        appDependencies.state.recordedChunkCount = Math.floor(data.samples / 16000);
        if (data.samples) appDependencies.markWorkspaceDirty();
      } else if (data.type === "error") {
        document.querySelector("#liveConnection").textContent = data.message === "transcription_failed"
          ? "認識サービスを再試行しています。受信済みの音声は保存されています。" : data.detail || data.message;
      }
    },
    onFatal: (error) => {
      if (appDependencies.state.liveCapture !== capture) return;
      appDependencies.cleanupMedia();
      setUiRecording(false);
      document.querySelector("#downloadPendingAudio").hidden = !capture.pending.size;
      document.querySelector("#liveConnection").textContent = "録音を停止しました。未送信音声を保存できます。";
      appDependencies.showToast(`ライブ録音を継続できません: ${error.message}`, "error", 7000);
    },
  });
  appDependencies.state.liveCapture = capture;
  try {
    await capture.start();
  } catch (error) {
    capture.dispose();
    appDependencies.state.liveCapture = null;
    throw error;
  }
  commitNewRecordingWorkspace();
  appDependencies.state.recordingStartedAt = performance.now();
  setUiRecording(true);
  appDependencies.setStatus("recording");
  document.querySelector("#liveConnection").textContent = "ライブ文字起こし · 音声を保存中";
  document.querySelector("#retryLiveStop").hidden = true;
  document.querySelector("#downloadPendingAudio").hidden = true;
  appDependencies.meetingWorkspace.setView("transcript");
}

async function finalizeLiveRecording() {
  const capture = appDependencies.state.liveCapture;
  if (!capture || appDependencies.state.finalizingStop) return;
  appDependencies.state.recording = false;
  appDependencies.state.finalizingStop = true;
  setUiRecordingStopping(true);
  document.querySelector("#retryLiveStop").hidden = true;
  try {
    await capture.stop();
    appDependencies.state.runtimeSessionFinalized = true;
    appDependencies.state.liveCapture = null;
    document.querySelector("#liveTranscript").replaceChildren();
    document.querySelector("#liveTranscript").hidden = true;
    document.querySelector("#liveConnection").textContent = "録音完了 · 発話の音声を再生できます";
    appDependencies.setStatus("completed");
    appDependencies.meetingWorkspace.refresh();
  } catch (error) {
    document.querySelector("#retryLiveStop").hidden = capture.closed;
    document.querySelector("#downloadPendingAudio").hidden = !capture.pending.size;
    document.querySelector("#liveConnection").textContent = "最終処理を完了できませんでした。再試行できます。";
    appDependencies.showToast(`最終処理に失敗しました: ${error.message}`, "error", 7000);
  } finally {
    appDependencies.cleanupMedia();
    appDependencies.state.finalizingStop = false;
    setUiRecording(false);
    appDependencies.updateDownloadLinks();
    appDependencies.updateSaveControls();
    appDependencies.updateRecordingTelemetry();
  }
}

async function refineMeetingAudio() {
  if (appDependencies.refinementController) { appDependencies.refinementController.abort(); return; }
  const source = currentMeetingSource();
  if (!source?.runtimeSessionId || !appDependencies.state.runtimeSessionFinalized || appDependencies.isRecordingInteractionLocked() || appDependencies.meetingWorkspace?.busy) return;
  const controller = new AbortController();
  appDependencies.refinementController = controller;
  appDependencies.updateSaveControls();
  appDependencies.setSessionSettingsLocked();
  appDependencies.renderHistoryList();
  const status = document.querySelector("#liveConnection");
  let completed = false;
  try {
    const response = await fetch("/api/meeting/refine", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(source), signal: controller.signal });
    if (!response.ok) throw new Error("音声の再認識を開始できませんでした");
    await readSseJsonStream(response, (event) => {
      if (event.type === "error") throw new Error(event.error);
      if (event.type === "status") status.textContent = event.message;
      if (event.type === "done") {
        completed = true;
        appDependencies.state.log = [];
        appDependencies.state.segments = [];
        appDependencies.renderEmptyTranscriptState();
        for (const row of event.records || []) {
          appDependencies.addLogLine(row.text, row.tsStart, row.tsEnd, row.seq, row.speaker || "", row.screenshotPath || "", row.rawAudioPath || "", row.audioPath || "", row.segmentId);
        }
        appDependencies.markWorkspaceDirty();
        status.textContent = "音声から再認識しました。元の文は保存時のZIPに含まれます。";
        appDependencies.meetingWorkspace.refresh();
      }
    });
    if (!completed) throw new Error("再認識が中断されました");
  } catch (error) {
    status.textContent = controller.signal.aborted ? "再認識をキャンセルしました" : "再認識に失敗しました。原文から再試行できます。";
    if (!controller.signal.aborted) appDependencies.showToast(error.message, "error");
  } finally {
    appDependencies.refinementController = null;
    appDependencies.updateSaveControls();
    appDependencies.setSessionSettingsLocked();
    appDependencies.renderHistoryList();
  }
}

async function startRecording() {
  if (appDependencies.state.recordingPhase !== "idle" || appDependencies.state.finalizingStop || appDependencies.refinementController) return;
  if (!appDependencies.canUseWorkspace()) {
    appDependencies.showToast("ログインが必要です", "error");
    appDependencies.setAppLocked(true);
    appDependencies.loginEmailEl?.focus();
    return;
  }
  if (!appDependencies.confirmWorkspaceDiscard("破棄して新しい録音を開始")) {
    return;
  }
  setUiRecordingStarting();
  let startSent = false;
  const previousRuntimeSessionId = appDependencies.state.runtimeSessionId;
  const previousRuntimeSessionToken = appDependencies.state.runtimeSessionToken;
  const previousRuntimeSessionFinalized = appDependencies.state.runtimeSessionFinalized;

  const selectedChunkSeconds = appDependencies.applyChunkSeconds(appDependencies.chunkSecondsEl.value || appDependencies.CHUNK_DEFAULT_SECONDS);
  appDependencies.state.chunkMs = selectedChunkSeconds * 1000;
  const selectedAudioSource = appDependencies.applyAudioSource(appDependencies.audioSourceEl?.value || "mic");
  appDependencies.state.recordingAudioSource = selectedAudioSource;
  appDependencies.state.recordingRequestedAudioSource = selectedAudioSource;
  appDependencies.state.recordingFallbackReason = "";
  appDependencies.state.recordingStartedAt = performance.now();
  appDependencies.state.recordedChunkCount = 0;
  appDependencies.state.pendingOutboundChunks = 0;
  appDependencies.state.maxObservedBacklog = 0;
  appDependencies.state.degradedCaptureMode = false;
  appDependencies.state.screenshotEncodeInFlight = false;
  appDependencies.state.screenshotLastCapturedAt = 0;
  appDependencies.state.screenshotSkippedCount = 0;
  appDependencies.state.lastTelemetryBacklogBucket = "";
  appDependencies.state.lastScreenshotSkipReason = "";
  appDependencies.state.lastScreenshotSkipSentAt = 0;
  appDependencies.state.seq = 0;
  appDependencies.state.offsetMs = 0;
  appDependencies.state.finalizingStop = false;
  appDependencies.state.pendingSendChain = Promise.resolve();
  clearChunkTimer();

  try {
    const health = await appDependencies.loadCapabilities();
    if (!health?.asrReady || !health?.model) {
      throw new Error("asr_not_ready");
    }

    if (health.asrBackend === "qwen3_vllm" && typeof AudioWorkletNode === "undefined") throw new Error("audio_worklet_required");
    if (health.liveWsPath && (health.asrBackend === "qwen3_vllm" || document.querySelector("#liveTranscriptionEnabled")?.checked) && typeof AudioWorkletNode !== "undefined") {
      await startLiveRecording(health, selectedAudioSource);
      return;
    }

    const ws = await appDependencies.ensureSocket();
    const stream = await appDependencies.prepareInputStream(selectedAudioSource);
    if (!appDependencies.hasAudioTrack(stream)) {
      throw new Error("audio_track_not_found");
    }

    appDependencies.state.stream = stream;
    await appDependencies.setupVad(stream);

    const mimeType = selectMimeType();
    appDependencies.state.recorderMimeType = mimeType || "audio/webm";
    appDependencies.state.recorderOptions = mimeType ? { mimeType } : {};
    const diarizationOptions = appDependencies.resolveDiarizationStartOptions();
    const effectiveAudioSource = normalizeAudioSource(appDependencies.state.recordingAudioSource || selectedAudioSource);

    const readyPromise = waitForSessionReady(ws);
    const startPayload = {
      type: "start",
      sessionId: generateSessionSeed(),
      language: appDependencies.selectedLanguage(),
      audioSource: effectiveAudioSource,
      requestedAudioSource: selectedAudioSource,
      audioSourceFallbackReason: appDependencies.state.recordingFallbackReason || "",
      prompt: appDependencies.promptEl.value.trim(),
      sharedVocabulary: String(appDependencies.sharedVocabularyEl?.value || appDependencies.state.sharedVocabulary || "").trim(),
      diarizationEnabled: !!(appDependencies.state.diarizationAvailable && appDependencies.state.diarizationEnabled),
      diarizationNumSpeakers: diarizationOptions.diarizationNumSpeakers,
      diarizationMinSpeakers: diarizationOptions.diarizationMinSpeakers,
      diarizationMaxSpeakers: diarizationOptions.diarizationMaxSpeakers,
    };
    ws.send(JSON.stringify(startPayload));
    startSent = true;
    appDependencies.logWsEvent("send_start", {
      sessionId: startPayload.sessionId,
      language: startPayload.language || "auto",
      audioSource: startPayload.audioSource,
      requestedAudioSource: startPayload.requestedAudioSource,
      audioSourceFallbackReason: startPayload.audioSourceFallbackReason,
    });
    await readyPromise;

    appDependencies.state.recordingStartedAt = performance.now();
    setUiRecording(true);
    appDependencies.updateRecordingTelemetry();
    if (effectiveAudioSource === "display") {
      appDependencies.setStatus("recording_display_audio");
    } else if (effectiveAudioSource === "both") {
      appDependencies.setStatus("recording_mic_and_display");
    } else {
      appDependencies.setStatus("recording_mic");
    }
    startRecorderCycle();
    commitNewRecordingWorkspace();
  } catch (err) {
    const name = err?.name || "";
    const message = err?.message || "unknown_error";
    if (message === "display_audio_not_found") {
      const diagnostics = err?.diagnostics || {};
      const displaySurface = diagnostics.displaySurface || "unknown";
      const audioTrackCount = Number(diagnostics.audioTrackCount || 0);
      appDependencies.setStatus(
        `start_failed: 画面共有の音声が見つかりません (surface=${displaySurface}, audioTracks=${audioTrackCount})`
      );
      appDependencies.showToast(
        "画面共有の音声トラックを取得できませんでした。Chrome/Edge のタブ共有は通りやすいですが、Skype/Webex のアプリ画面はブラウザ制約で音声が渡らないことがあります。",
        "error",
        9000
      );
    } else if (name === "NotAllowedError") {
      appDependencies.setStatus("start_failed: 権限が拒否されました");
    } else if (message === "asr_not_ready") {
      appDependencies.setStatus("start_failed: /api/health で ASR モデルが確認できません");
    } else {
      appDependencies.setStatus(`start_failed: ${message}`);
    }
    if (startSent && appDependencies.state.ws?.readyState === WebSocket.OPEN) {
      appDependencies.state.ws.send(JSON.stringify({ type: "stop" }));
      appDependencies.logWsEvent("send_stop_after_start_failure");
    }
    appDependencies.cleanupMedia();
    appDependencies.state.runtimeSessionId = previousRuntimeSessionId;
    appDependencies.state.runtimeSessionToken = previousRuntimeSessionToken;
    appDependencies.state.runtimeSessionFinalized = previousRuntimeSessionFinalized;
    appDependencies.state.recordingStartedAt = 0;
    setUiRecording(false);
    appDependencies.updateDownloadLinks();
    appDependencies.updateRecordingTelemetry();
  }
}

function stopRecording() {
  if (!appDependencies.state.recording) return;
  if (appDependencies.state.liveCapture) {
    finalizeLiveRecording();
    return;
  }
  appDependencies.state.recording = false;
  setUiRecordingStopping(false);
  clearChunkTimer();

  try {
    if (appDependencies.state.recorder && appDependencies.state.recorder.state === "recording") {
      appDependencies.state.recorder.stop();
    } else {
      finalizeStop();
    }
  } catch {
    finalizeStop();
  }
}

function abortRecordingAfterSocketLoss(ws, reason) {
  if (ws.__whistxConnectionLossHandled) return;
  ws.__whistxConnectionLossHandled = true;
  if (appDependencies.state.ws === ws) {
    appDependencies.state.ws = null;
  }

  const wasActive =
    appDependencies.state.recordingPhase === "starting" ||
    appDependencies.state.recordingPhase === "recording" ||
    appDependencies.state.finalizingStop;
  if (!wasActive) {
    appDependencies.setStatus(reason === "socket_error" ? "socket_error" : "disconnected");
    return;
  }

  appDependencies.state.recording = false;
  appDependencies.state.recordingPhase = "stopping";
  clearChunkTimer();
  const recorder = appDependencies.state.recorder;
  if (recorder?.state === "recording") {
    try {
      recorder.__whistxAbortWithoutFinalize = true;
      recorder.stop();
    } catch {
      // Continue with media cleanup even when recorder shutdown fails.
    }
  }
  appDependencies.cleanupMedia();
  appDependencies.state.pendingSendChain = Promise.resolve();
  appDependencies.state.pendingOutboundChunks = 0;
  appDependencies.state.finalizingStop = false;
  appDependencies.state.segmentStartedAt = 0;
  appDependencies.state.recordingStartedAt = 0;
  appDependencies.state.recordedChunkCount = 0;
  setUiRecording(false);
  appDependencies.updateRecordingTelemetry();
  appDependencies.setStatus(reason);
  appDependencies.showToast(
    reason === "socket_error"
      ? "サーバー接続エラーのため録音を終了しました。未送信の音声は保存されません"
      : "サーバーとの接続が切れたため録音を終了しました。未送信の音声は保存されません",
    "error",
    7000
  );
}

  return { selectMimeType, generateSessionSeed, setUiRecording, setUiRecordingStarting, setUiRecordingStopping, resetRuntimeSessionState, commitNewRecordingWorkspace, sendChunk, clearChunkTimer, chunkHardMaxMs, shouldCutChunkOnSilence, requestChunkFlush, scheduleChunkStop, startRecorderCycle, finalizeStop, currentMeetingSource, navigateMeetingSource, startLiveRecording, finalizeLiveRecording, refineMeetingAudio, startRecording, stopRecording, abortRecordingAfterSocketLoss };
}
