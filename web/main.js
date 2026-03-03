const $ = (selector) => document.querySelector(selector);

const statusTextEl = $("#statusText");
const connCountEl = $("#connCount");

const languageEl = $("#language");
const audioSourceEl = $("#audioSource");
const chunkSecondsEl = $("#chunkSeconds");
const promptEl = $("#prompt");
const chunkHintEl = $("#chunkHint");
const presetButtons = Array.from(document.querySelectorAll("[data-chunk-preset]"));

const startBtn = $("#startBtn");
const stopBtn = $("#stopBtn");
const summaryBtn = $("#summaryBtn");
const copyBtn = $("#copyBtn");
const clearBtn = $("#clearBtn");

const dlTxt = $("#dlTxt");
const dlJsonl = $("#dlJsonl");
const dlSrt = $("#dlSrt");

const logEl = $("#log");
const segmentCountEl = $("#segmentCount");
const summaryTextEl = $("#summaryText");
const summaryMetaEl = $("#summaryMeta");
const toastContainer = $("#toastContainer");

const CHUNK_MIN_SECONDS = 12;
const CHUNK_MAX_SECONDS = 30;
const CHUNK_DEFAULT_SECONDS = 20;

const VAD_SAMPLE_MS = 80;
const VAD_RMS_THRESHOLD = 0.01;
const VAD_MIN_SPEECH_RATIO = 0.06;
const VAD_MIN_ACTIVE_MS = 160;

const state = {
  ws: null,
  stream: null,
  micStream: null,
  displayStream: null,
  recorder: null,
  recorderOptions: null,
  recorderMimeType: "audio/webm",
  chunkTimer: null,
  finalizingStop: false,
  recording: false,
  runtimeSessionId: "",
  seq: 0,
  offsetMs: 0,
  chunkMs: CHUNK_DEFAULT_SECONDS * 1000,
  pendingSendChain: Promise.resolve(),
  log: [],
  summary: "",
  audioContext: null,
  vadSource: null,
  vadAnalyser: null,
  vadBuffer: null,
  vadTimer: null,
  vadFrameCount: 0,
  vadSpeechFrameCount: 0,
  captureContext: null,
  captureSources: [],
  captureDestination: null,
};

function setStatus(text) {
  statusTextEl.textContent = text;
  statusTextEl.dataset.state = String(text || "").toLowerCase();
}

// Toast notification system
function showToast(message, type = "default", duration = 2500) {
  if (!toastContainer) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  let icon = "";
  if (type === "success") {
    icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>`;
  } else if (type === "error") {
    icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6"/><path d="M9 9l6 6"/></svg>`;
  }

  toast.innerHTML = `${icon}<span>${message}</span>`;
  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("hiding");
    setTimeout(() => toast.remove(), 250);
  }, duration);
}

function updateSegmentCount() {
  segmentCountEl.textContent = `${state.log.length} segments`;
}

function updateDownloadLinks() {
  if (!state.runtimeSessionId) {
    dlTxt.href = "#";
    dlJsonl.href = "#";
    dlSrt.href = "#";
    return;
  }

  dlTxt.href = `/api/transcript/${state.runtimeSessionId}.txt`;
  dlJsonl.href = `/api/transcript/${state.runtimeSessionId}.jsonl`;
  dlSrt.href = `/api/transcript/${state.runtimeSessionId}.srt`;
}

function setSummary(text, meta) {
  state.summary = text || "";

  if (text) {
    summaryTextEl.innerHTML = "";
    summaryTextEl.textContent = text;
  } else {
    summaryTextEl.innerHTML = `
      <div class="empty-state small">
        <p class="empty-description">「要約生成」ボタンでAIによる要約を生成します</p>
      </div>
    `;
  }

  summaryMetaEl.textContent = meta || "未生成";
}

function addLogLine(text, tsStart, tsEnd) {
  // Hide empty state on first log entry
  const emptyState = logEl.querySelector(".empty-state");
  if (emptyState) {
    emptyState.remove();
  }

  const row = document.createElement("div");
  row.className = "log-row new";

  const range = document.createElement("span");
  range.className = "time";
  range.textContent = `${formatMs(tsStart)} - ${formatMs(tsEnd)}`;

  const content = document.createElement("span");
  content.className = "text";
  content.textContent = text;

  row.append(range, content);
  logEl.appendChild(row);
  logEl.scrollTop = logEl.scrollHeight;

  state.log.push(text);
  updateSegmentCount();

  // Remove 'new' class after animation
  setTimeout(() => row.classList.remove("new"), 300);
}

function formatMs(ms) {
  const totalSec = Math.floor(Math.max(0, ms) / 1000);
  const min = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const sec = String(totalSec % 60).padStart(2, "0");
  return `${min}:${sec}`;
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/transcribe`;
}

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

function normalizeChunkSeconds(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return CHUNK_DEFAULT_SECONDS;
  return Math.max(CHUNK_MIN_SECONDS, Math.min(CHUNK_MAX_SECONDS, Math.round(num)));
}

function updateChunkHint(seconds) {
  if (!chunkHintEl) return;
  if (seconds <= 14) {
    chunkHintEl.textContent = "リアルタイム寄り（精度より応答速度優先）";
    return;
  }
  if (seconds <= 19) {
    chunkHintEl.textContent = "バランス（精度と遅延の中間）";
    return;
  }
  if (seconds <= 24) {
    chunkHintEl.textContent = "精度優先（推奨）";
    return;
  }
  chunkHintEl.textContent = "最大精度寄り（遅延増）";
}

function updatePresetActive(seconds) {
  presetButtons.forEach((button) => {
    const raw = button.getAttribute("data-chunk-preset") || "";
    button.classList.toggle("is-active", Number(raw) === seconds);
  });
}

function applyChunkSeconds(value) {
  const seconds = normalizeChunkSeconds(value);
  if (chunkSecondsEl) {
    chunkSecondsEl.value = String(seconds);
  }
  updateChunkHint(seconds);
  updatePresetActive(seconds);
  try {
    localStorage.setItem("whistx_chunk_seconds", String(seconds));
  } catch {
    // ignore
  }
  return seconds;
}

function setUiRecording(active) {
  state.recording = active;

  // Update record button state
  if (active) {
    startBtn.classList.add("is-recording");
    startBtn.querySelector(".record-label").textContent = "停止";
  } else {
    startBtn.classList.remove("is-recording");
    startBtn.querySelector(".record-label").textContent = "録音開始";
  }

  startBtn.disabled = false; // Always enabled to allow stop
  stopBtn.disabled = !active;
}

function normalizeAudioSource(value) {
  if (value === "display" || value === "both") return value;
  return "mic";
}

function applyAudioSource(value) {
  const source = normalizeAudioSource(value);
  if (audioSourceEl) {
    audioSourceEl.value = source;
  }
  try {
    localStorage.setItem("whistx_audio_source", source);
  } catch {
    // ignore
  }
  return source;
}

function hasAudioTrack(stream) {
  return !!stream && stream.getAudioTracks().length > 0;
}

async function requestMicStream() {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
}

async function requestDisplayStream() {
  return navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: true,
  });
}

async function ensureAudioContextResumed(context) {
  if (context.state !== "suspended") return;
  try {
    await context.resume();
  } catch {
    // ignore
  }
}

async function buildMixedAudioStream(streams) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) {
    throw new Error("AudioContext_not_supported");
  }

  const context = new AudioContextCtor();
  await ensureAudioContextResumed(context);

  const destination = context.createMediaStreamDestination();
  const sources = [];

  for (const stream of streams) {
    if (!hasAudioTrack(stream)) continue;
    const source = context.createMediaStreamSource(stream);
    source.connect(destination);
    sources.push(source);
  }

  if (!sources.length) {
    await context.close().catch(() => {
      // ignore
    });
    throw new Error("display_audio_not_found");
  }

  state.captureContext = context;
  state.captureSources = sources;
  state.captureDestination = destination;
  return destination.stream;
}

function bindDisplayEndEvents(displayStream) {
  const tracks = [...displayStream.getVideoTracks(), ...displayStream.getAudioTracks()];
  tracks.forEach((track) => {
    track.addEventListener(
      "ended",
      () => {
        if (!state.recording) return;
        setStatus("display_capture_ended");
        stopRecording();
      },
      { once: true }
    );
  });
}

async function prepareInputStream(sourceMode) {
  const mode = normalizeAudioSource(sourceMode);

  if (mode === "mic") {
    const micStream = await requestMicStream();
    state.micStream = micStream;
    return micStream;
  }

  if (mode === "display") {
    const displayStream = await requestDisplayStream();
    if (!hasAudioTrack(displayStream)) {
      throw new Error("display_audio_not_found");
    }
    bindDisplayEndEvents(displayStream);
    state.displayStream = displayStream;
    return buildMixedAudioStream([displayStream]);
  }

  const displayStream = await requestDisplayStream();
  if (!hasAudioTrack(displayStream)) {
    throw new Error("display_audio_not_found");
  }
  bindDisplayEndEvents(displayStream);

  const micStream = await requestMicStream();
  state.displayStream = displayStream;
  state.micStream = micStream;
  return buildMixedAudioStream([displayStream, micStream]);
}

async function ensureSocket() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    return state.ws;
  }

  if (state.ws && state.ws.readyState === WebSocket.CONNECTING) {
    await waitForOpen(state.ws);
    return state.ws;
  }

  const ws = new WebSocket(wsUrl());
  state.ws = ws;

  ws.addEventListener("message", (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }

    if (data.type === "conn") {
      connCountEl.textContent = String(data.count ?? 0);
      return;
    }

    if (data.type === "info") {
      if (data.message) setStatus(String(data.message));
      if (data.sessionId) {
        state.runtimeSessionId = String(data.sessionId);
        updateDownloadLinks();
      }
      return;
    }

    if (data.type === "final") {
      addLogLine(String(data.text || ""), Number(data.tsStart || 0), Number(data.tsEnd || 0));
      return;
    }

    if (data.type === "error") {
      const detail = data.detail ? ` (${data.detail})` : "";
      setStatus(`error: ${data.message || "unknown"}${detail}`);
    }
  });

  ws.addEventListener("close", () => {
    state.ws = null;
    connCountEl.textContent = "0";
    if (!state.recording) {
      setStatus("disconnected");
    }
  });

  ws.addEventListener("error", () => {
    setStatus("socket_error");
  });

  await waitForOpen(ws);
  return ws;
}

function waitForOpen(ws) {
  return new Promise((resolve, reject) => {
    if (ws.readyState === WebSocket.OPEN) {
      resolve();
      return;
    }

    const onOpen = () => {
      cleanup();
      resolve();
    };

    const onClose = () => {
      cleanup();
      reject(new Error("websocket_closed"));
    };

    const onError = () => {
      cleanup();
      reject(new Error("websocket_error"));
    };

    const cleanup = () => {
      ws.removeEventListener("open", onOpen);
      ws.removeEventListener("close", onClose);
      ws.removeEventListener("error", onError);
    };

    ws.addEventListener("open", onOpen);
    ws.addEventListener("close", onClose);
    ws.addEventListener("error", onError);
  });
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";

  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function sampleVad() {
  if (!state.vadAnalyser || !state.vadBuffer) return;

  state.vadAnalyser.getFloatTimeDomainData(state.vadBuffer);

  let sum = 0;
  for (let i = 0; i < state.vadBuffer.length; i += 1) {
    const value = state.vadBuffer[i];
    sum += value * value;
  }

  const rms = Math.sqrt(sum / state.vadBuffer.length);
  state.vadFrameCount += 1;
  if (rms >= VAD_RMS_THRESHOLD) {
    state.vadSpeechFrameCount += 1;
  }
}

async function setupVad(stream) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) {
    return;
  }

  const context = new AudioContextCtor();
  if (context.state === "suspended") {
    try {
      await context.resume();
    } catch {
      // ignore
    }
  }

  const source = context.createMediaStreamSource(stream);
  const analyser = context.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.05;

  source.connect(analyser);

  state.audioContext = context;
  state.vadSource = source;
  state.vadAnalyser = analyser;
  state.vadBuffer = new Float32Array(analyser.fftSize);
  state.vadFrameCount = 0;
  state.vadSpeechFrameCount = 0;
  state.vadTimer = setInterval(sampleVad, VAD_SAMPLE_MS);
}

function stopVad() {
  if (state.vadTimer) {
    clearInterval(state.vadTimer);
    state.vadTimer = null;
  }

  if (state.vadSource) {
    try {
      state.vadSource.disconnect();
    } catch {
      // ignore
    }
    state.vadSource = null;
  }

  if (state.vadAnalyser) {
    try {
      state.vadAnalyser.disconnect();
    } catch {
      // ignore
    }
    state.vadAnalyser = null;
  }

  if (state.audioContext) {
    state.audioContext.close().catch(() => {
      // ignore
    });
    state.audioContext = null;
  }

  state.vadBuffer = null;
  state.vadFrameCount = 0;
  state.vadSpeechFrameCount = 0;
}

function cleanupCaptureGraph() {
  state.captureSources.forEach((source) => {
    try {
      source.disconnect();
    } catch {
      // ignore
    }
  });
  state.captureSources = [];
  state.captureDestination = null;

  if (state.captureContext) {
    state.captureContext.close().catch(() => {
      // ignore
    });
    state.captureContext = null;
  }
}

function snapshotVadCounters() {
  return {
    frameCount: state.vadFrameCount,
    speechFrameCount: state.vadSpeechFrameCount,
  };
}

function buildVadDecision(snapshot) {
  if (!state.vadAnalyser) {
    return {
      enabled: false,
      speechRatio: 1,
      activeMs: state.chunkMs,
      skip: false,
    };
  }

  const totalFrames = Math.max(1, state.vadFrameCount - snapshot.frameCount);
  const speechFrames = Math.max(0, state.vadSpeechFrameCount - snapshot.speechFrameCount);

  const speechRatio = speechFrames / totalFrames;
  const activeMs = speechFrames * VAD_SAMPLE_MS;
  const skip = speechRatio < VAD_MIN_SPEECH_RATIO && activeMs < VAD_MIN_ACTIVE_MS;

  return {
    enabled: true,
    speechRatio,
    activeMs,
    skip,
  };
}

async function sendChunk(blob, mimeType, durationMsOverride, vadDecision) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;

  let durationMs = Number(durationMsOverride);
  if (!Number.isFinite(durationMs)) {
    durationMs = state.chunkMs;
  }
  durationMs = Math.max(200, Math.round(durationMs));

  const offsetMs = state.offsetMs;
  state.offsetMs += durationMs;

  if (!blob || blob.size === 0) return;

  if (vadDecision?.enabled && vadDecision.skip) {
    return;
  }

  const seq = state.seq++;
  const buffer = await blob.arrayBuffer();
  const audio = arrayBufferToBase64(buffer);

  state.ws.send(
    JSON.stringify({
      type: "chunk",
      seq,
      offsetMs,
      durationMs,
      mimeType: mimeType || blob.type || "audio/webm",
      audio,
    })
  );
}

function clearChunkTimer() {
  if (state.chunkTimer) {
    clearTimeout(state.chunkTimer);
    state.chunkTimer = null;
  }
}

function scheduleChunkStop(recorder) {
  clearChunkTimer();
  state.chunkTimer = setTimeout(() => {
    if (!state.recording) return;
    if (state.recorder !== recorder) return;
    if (recorder.state !== "recording") return;
    try {
      recorder.stop();
    } catch {
      // ignore
    }
  }, state.chunkMs);
}

function startRecorderCycle() {
  if (!state.recording || !state.stream) return;

  const recorder = new MediaRecorder(state.stream, state.recorderOptions || {});
  state.recorder = recorder;
  const cycleStartedAt = performance.now();
  const vadSnapshot = snapshotVadCounters();

  recorder.addEventListener("dataavailable", (event) => {
    const durationMs = Math.max(200, Math.round(performance.now() - cycleStartedAt));
    const vadDecision = buildVadDecision(vadSnapshot);

    state.pendingSendChain = state.pendingSendChain
      .then(() => sendChunk(event.data, state.recorderMimeType, durationMs, vadDecision))
      .catch((err) => {
        setStatus(`chunk_error: ${err.message}`);
      });
  });

  recorder.addEventListener("error", () => {
    setStatus("recorder_error");
    state.recording = false;
    finalizeStop();
  });

  recorder.addEventListener("stop", () => {
    if (state.recorder === recorder) {
      state.recorder = null;
    }
    if (state.recording) {
      startRecorderCycle();
      return;
    }
    finalizeStop();
  });

  recorder.start();
  scheduleChunkStop(recorder);
}

async function finalizeStop() {
  if (state.finalizingStop) return;
  state.finalizingStop = true;
  clearChunkTimer();

  try {
    await state.pendingSendChain;
  } finally {
    state.pendingSendChain = Promise.resolve();
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "stop" }));
    }
    cleanupMedia();
    setUiRecording(false);
    state.finalizingStop = false;
  }
}

async function startRecording() {
  if (state.recording) return;

  const selectedChunkSeconds = applyChunkSeconds(chunkSecondsEl.value || CHUNK_DEFAULT_SECONDS);
  state.chunkMs = selectedChunkSeconds * 1000;
  const selectedAudioSource = applyAudioSource(audioSourceEl?.value || "mic");
  state.seq = 0;
  state.offsetMs = 0;
  state.runtimeSessionId = "";
  state.finalizingStop = false;
  state.pendingSendChain = Promise.resolve();
  clearChunkTimer();

  try {
    const ws = await ensureSocket();
    const stream = await prepareInputStream(selectedAudioSource);
    if (!hasAudioTrack(stream)) {
      throw new Error("audio_track_not_found");
    }

    state.stream = stream;
    await setupVad(stream);

    const mimeType = selectMimeType();
    state.recorderMimeType = mimeType || "audio/webm";
    state.recorderOptions = mimeType ? { mimeType } : {};

    ws.send(
      JSON.stringify({
        type: "start",
        sessionId: generateSessionSeed(),
        language: languageEl.value,
        prompt: promptEl.value.trim(),
      })
    );

    setUiRecording(true);
    if (selectedAudioSource === "display") {
      setStatus("recording_display_audio");
    } else if (selectedAudioSource === "both") {
      setStatus("recording_mic_and_display");
    } else {
      setStatus("recording_mic");
    }
    startRecorderCycle();
  } catch (err) {
    const name = err?.name || "";
    const message = err?.message || "unknown_error";
    if (message === "display_audio_not_found") {
      setStatus("start_failed: 画面共有の音声が見つかりません（音声共有をONにしてください）");
    } else if (name === "NotAllowedError") {
      setStatus("start_failed: 権限が拒否されました");
    } else {
      setStatus(`start_failed: ${message}`);
    }
    cleanupMedia();
    setUiRecording(false);
  }
}

function stopRecording() {
  if (!state.recording) return;
  setStatus("stopping");
  state.recording = false;
  clearChunkTimer();

  try {
    if (state.recorder && state.recorder.state === "recording") {
      state.recorder.stop();
    } else {
      finalizeStop();
    }
  } catch {
    finalizeStop();
  }
}

function cleanupMedia() {
  if (state.recorder) {
    state.recorder.ondataavailable = null;
    state.recorder.onstop = null;
    state.recorder = null;
  }
  state.recorderOptions = null;

  stopVad();
  cleanupCaptureGraph();

  const streams = [state.stream, state.micStream, state.displayStream];
  const seenTracks = new Set();
  streams.forEach((stream) => {
    if (!stream) return;
    stream.getTracks().forEach((track) => {
      if (seenTracks.has(track.id)) return;
      seenTracks.add(track.id);
      try {
        track.stop();
      } catch {
        // ignore
      }
    });
  });

  state.stream = null;
  state.micStream = null;
  state.displayStream = null;
}

async function copyAll() {
  const text = state.log.join("\n").trim();
  if (!text) {
    showToast("コピーする内容がありません", "error");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    showToast("コピーしました", "success");
    setStatus("copied");
  } catch {
    showToast("コピーに失敗しました", "error");
    setStatus("copy_failed");
  }
}

async function summarizeAll() {
  const text = state.log.join("\n").trim();
  if (!text) {
    showToast("要約する文字起こしがありません", "error");
    setStatus("summary_no_text");
    return;
  }

  summaryBtn.disabled = true;
  setStatus("summarizing");
  showToast("要約を生成中...", "default", 5000);

  try {
    const response = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        language: languageEl.value || "ja",
      }),
    });

    let payload = {};
    try {
      payload = await response.json();
    } catch {
      // ignore
    }

    if (!response.ok) {
      const detail = payload.detail || payload.error || `http_${response.status}`;
      throw new Error(String(detail));
    }

    const summaryText = String(payload.summary || "").trim();
    if (!summaryText) {
      throw new Error("empty_summary");
    }

    const metaParts = [];
    if (payload.model) {
      metaParts.push(`model: ${payload.model}`);
    }
    if (payload.truncated) {
      metaParts.push("入力を末尾で切り詰め");
    }

    setSummary(summaryText, metaParts.join(" | ") || "生成完了");
    setStatus("summarized");
    showToast("要約を生成しました", "success");
  } catch (err) {
    showToast(`要約に失敗: ${err.message}`, "error");
    setStatus(`summary_failed: ${err.message}`);
  } finally {
    summaryBtn.disabled = false;
  }
}

function clearView() {
  state.log = [];

  // Restore empty state
  logEl.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" x2="12" y1="19" y2="22"/>
        </svg>
      </div>
      <p class="empty-title">まだ文字起こしがありません</p>
      <p class="empty-description">録音を開始すると、リアルタイムで文字起こしが表示されます</p>
    </div>
  `;

  updateSegmentCount();
  setSummary("", "未生成");
  showToast("クリアしました", "success");
}

chunkSecondsEl.addEventListener("change", () => {
  applyChunkSeconds(chunkSecondsEl.value);
});

chunkSecondsEl.addEventListener("input", () => {
  updateChunkHint(normalizeChunkSeconds(chunkSecondsEl.value));
});

if (audioSourceEl) {
  audioSourceEl.addEventListener("change", () => {
    applyAudioSource(audioSourceEl.value);
  });
}

presetButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const raw = button.getAttribute("data-chunk-preset") || "";
    applyChunkSeconds(raw);
  });
});

startBtn.addEventListener("click", () => {
  if (state.recording) {
    stopRecording();
  } else {
    startRecording();
  }
});

stopBtn.addEventListener("click", () => {
  stopRecording();
});

summaryBtn.addEventListener("click", () => {
  summarizeAll();
});

copyBtn.addEventListener("click", () => {
  copyAll();
});

clearBtn.addEventListener("click", () => {
  clearView();
});

// Initialize empty states
updateSegmentCount();
updateDownloadLinks();
setSummary("", "未生成");
setStatus("idle");

// Show initial empty state for transcript
if (logEl && !logEl.querySelector(".log-row")) {
  logEl.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" x2="12" y1="19" y2="22"/>
        </svg>
      </div>
      <p class="empty-title">まだ文字起こしがありません</p>
      <p class="empty-description">録音を開始すると、リアルタイムで文字起こしが表示されます</p>
    </div>
  `;
}

(() => {
  let initial = CHUNK_DEFAULT_SECONDS;
  try {
    const saved = localStorage.getItem("whistx_chunk_seconds");
    if (saved) initial = normalizeChunkSeconds(saved);
  } catch {
    // ignore
  }
  applyChunkSeconds(initial);
})();

(() => {
  let initial = "mic";
  try {
    const saved = localStorage.getItem("whistx_audio_source");
    if (saved) initial = normalizeAudioSource(saved);
  } catch {
    // ignore
  }
  applyAudioSource(initial);
})();
