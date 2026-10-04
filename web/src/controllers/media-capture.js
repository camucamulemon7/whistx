import { normalizeAudioSource } from "../audio/vad.js";
import { vadThresholdForSource } from "../audio/vad.js";
import { arrayBufferToBase64 } from "../transcription/websocket.js";
import { VAD_SAMPLE_MS } from "../audio/vad.js";
import { buildVadDecision as calculateVadDecision } from "../audio/vad.js";
import { shouldSkipChunkByVad as calculateShouldSkipChunkByVad } from "../audio/vad.js";

export function createMediaCaptureController(appDependencies) {
function currentVadSourceMode() {
  return normalizeAudioSource(appDependencies.state.recordingAudioSource || appDependencies.state.recordingRequestedAudioSource || "mic");
}

function updateVadNoiseFloor(rms, now) {
  if (!Number.isFinite(rms) || rms <= 0) {
    return;
  }

  const startedAt = appDependencies.state.vadNoiseFloorAt || now;
  const elapsed = Math.max(0, now - startedAt);
  const source = currentVadSourceMode();
  const threshold = vadThresholdForSource(source, appDependencies.state.vadNoiseFloor);
  const isWarmup = elapsed <= appDependencies.VAD_NOISE_FLOOR_WARMUP_MS;
  const trackingThreshold = appDependencies.state.vadNoiseFloor === null ? threshold * 0.88 : threshold * (isWarmup ? 0.95 : 0.92);
  const shouldTrack = rms < trackingThreshold;

  if (!shouldTrack) {
    return;
  }

  if (appDependencies.state.vadNoiseFloor === null) {
    appDependencies.state.vadNoiseFloor = rms;
  } else {
    appDependencies.state.vadNoiseFloor = appDependencies.state.vadNoiseFloor * (1 - appDependencies.VAD_NOISE_FLOOR_EWMA) + rms * appDependencies.VAD_NOISE_FLOOR_EWMA;
  }

  appDependencies.state.vadRmsThreshold = vadThresholdForSource(source, appDependencies.state.vadNoiseFloor);
}

function hasAudioTrack(stream) {
  return !!stream && stream.getAudioTracks().length > 0;
}

function setupDisplayCaptureVideo(displayStream) {
  const [videoTrack] = displayStream?.getVideoTracks?.() || [];
  if (!videoTrack) {
    appDependencies.state.displayCaptureVideo = null;
    return;
  }

  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.autoplay = true;
  video.srcObject = new MediaStream([videoTrack]);
  const playPromise = video.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {});
  }
  appDependencies.state.displayCaptureVideo = video;
}

async function captureDisplayScreenshot() {
  if (!appDependencies.state.captureScreenshotsEnabled) return null;
  const video = appDependencies.state.displayCaptureVideo;
  const displayStream = appDependencies.state.displayStream;
  if (!video || !displayStream) return null;
  if (appDependencies.state.screenshotEncodeInFlight) {
    appDependencies.state.screenshotSkippedCount += 1;
    appDependencies.emitScreenshotSkipTelemetry("encode_in_flight");
    return null;
  }

  const [videoTrack] = displayStream.getVideoTracks?.() || [];
  if (!videoTrack) return null;

  const width = Number(video.videoWidth || videoTrack.getSettings?.().width || 0);
  const height = Number(video.videoHeight || videoTrack.getSettings?.().height || 0);
  if (!width || !height) return null;

  const now = performance.now();
  const minIntervalMs = appDependencies.state.degradedCaptureMode ? appDependencies.SCREENSHOT_DEGRADED_INTERVAL_MS : appDependencies.SCREENSHOT_MIN_INTERVAL_MS;
  if (now - appDependencies.state.screenshotLastCapturedAt < minIntervalMs) {
    appDependencies.state.screenshotSkippedCount += 1;
    appDependencies.emitScreenshotSkipTelemetry("interval_guard", { minIntervalMs });
    return null;
  }

  const maxWidth = appDependencies.state.degradedCaptureMode ? appDependencies.SCREENSHOT_DEGRADED_MAX_WIDTH : appDependencies.SCREENSHOT_MAX_WIDTH;
  const targetWidth = Math.min(maxWidth, width);
  const targetHeight = Math.max(1, Math.round((height * targetWidth) / width));

  const canvas = appDependencies.state.screenshotCanvas || document.createElement("canvas");
  appDependencies.state.screenshotCanvas = canvas;
  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

  const signature = buildScreenshotSignature(canvas);
  if (shouldSkipScreenshotByDiff(signature)) {
    appDependencies.state.screenshotSkippedCount += 1;
    appDependencies.emitScreenshotSkipTelemetry("diff_unchanged");
    return null;
  }

  appDependencies.state.screenshotEncodeInFlight = true;
  const blob = await new Promise((resolve) => {
    canvas.toBlob((nextBlob) => resolve(nextBlob), "image/webp", appDependencies.SCREENSHOT_WEBP_QUALITY);
  });
  appDependencies.state.screenshotEncodeInFlight = false;
  if (!blob) {
    appDependencies.emitScreenshotSkipTelemetry("encode_failed");
    return null;
  }

  const buffer = await blob.arrayBuffer();
  appDependencies.state.previousScreenshotSignature = signature;
  appDependencies.state.screenshotLastCapturedAt = now;
  return {
    mimeType: blob.type || "image/webp",
    data: arrayBufferToBase64(buffer),
  };
}

function buildScreenshotSignature(sourceCanvas) {
  const canvas = appDependencies.state.screenshotDiffCanvas || document.createElement("canvas");
  appDependencies.state.screenshotDiffCanvas = canvas;
  canvas.width = appDependencies.SCREENSHOT_DIFF_WIDTH;
  canvas.height = appDependencies.SCREENSHOT_DIFF_HEIGHT;

  const ctx = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(sourceCanvas, 0, 0, appDependencies.SCREENSHOT_DIFF_WIDTH, appDependencies.SCREENSHOT_DIFF_HEIGHT);

  const { data } = ctx.getImageData(0, 0, appDependencies.SCREENSHOT_DIFF_WIDTH, appDependencies.SCREENSHOT_DIFF_HEIGHT);
  const signature = new Uint8Array(appDependencies.SCREENSHOT_DIFF_WIDTH * appDependencies.SCREENSHOT_DIFF_HEIGHT);
  for (let src = 0, dst = 0; src < data.length; src += 4, dst += 1) {
    signature[dst] = ((data[src] * 77) + (data[src + 1] * 150) + (data[src + 2] * 29)) >> 8;
  }
  return signature;
}

function shouldSkipScreenshotByDiff(signature) {
  if (!appDependencies.state.screenshotDiffSkipEnabled || !signature) {
    return false;
  }

  const previous = appDependencies.state.previousScreenshotSignature;
  if (!(previous instanceof Uint8Array) || previous.length !== signature.length) {
    return false;
  }

  let diffSum = 0;
  let changedPixels = 0;
  for (let i = 0; i < signature.length; i += 1) {
    const delta = Math.abs(signature[i] - previous[i]);
    diffSum += delta;
    if (delta >= appDependencies.SCREENSHOT_DIFF_PIXEL_THRESHOLD) {
      changedPixels += 1;
    }
  }

  const meanDiff = diffSum / signature.length;
  const changedRatio = changedPixels / signature.length;
  if (meanDiff < appDependencies.SCREENSHOT_DIFF_MEAN_THRESHOLD || changedRatio < appDependencies.SCREENSHOT_DIFF_CHANGED_RATIO_THRESHOLD) {
    return true;
  }

  return false;
}

async function requestMicStream(sourceMode = "mic") {
  void sourceMode;
  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false,
    },
  });
}

async function requestDisplayStream() {
  const candidates = [
    {
      video: true,
      audio: { suppressLocalAudioPlayback: false },
      systemAudio: "include",
      windowAudio: "system",
      surfaceSwitching: "include",
      selfBrowserSurface: "exclude",
      monitorTypeSurfaces: "include",
    },
    {
      video: true,
      audio: true,
      systemAudio: "include",
      windowAudio: "system",
      surfaceSwitching: "include",
      selfBrowserSurface: "exclude",
      monitorTypeSurfaces: "include",
    },
    {
      video: true,
      audio: true,
    },
  ];

  let lastError = null;
  for (const constraints of candidates) {
    try {
      return await navigator.mediaDevices.getDisplayMedia(constraints);
    } catch (error) {
      lastError = error;
      if (error?.name && error.name !== "TypeError" && error.name !== "OverconstrainedError") {
        throw error;
      }
    }
  }

  throw lastError || new Error("display_capture_not_supported");
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
  const mixBus = context.createGain();
  const outputGain = context.createGain();
  const monitorAnalyser = context.createAnalyser();
  monitorAnalyser.fftSize = 2048;
  monitorAnalyser.smoothingTimeConstant = 0.88;
  const sources = [];

  mixBus.connect(monitorAnalyser);
  mixBus.connect(outputGain);
  outputGain.connect(destination);

  for (const stream of streams) {
    if (!hasAudioTrack(stream)) continue;
    const source = context.createMediaStreamSource(stream);
    source.connect(mixBus);
    sources.push(source);
  }

  if (!sources.length) {
    await context.close().catch(() => {
      // ignore
    });
    throw new Error("display_audio_not_found");
  }

  appDependencies.state.captureContext = context;
  appDependencies.state.captureSources = sources;
  appDependencies.state.captureDestination = destination;
  appDependencies.state.captureGainNode = outputGain;
  appDependencies.state.captureMonitorAnalyser = monitorAnalyser;
  appDependencies.state.captureMonitorBuffer = new Float32Array(monitorAnalyser.fftSize);
  appDependencies.state.captureAutoGainLevel = 1;
  appDependencies.state.captureAutoGainSmoothedRms = 0;
  updateCaptureAutoGainState();
  return destination.stream;
}

function sampleCaptureAutoGainRms() {
  if (!appDependencies.state.captureMonitorAnalyser || !appDependencies.state.captureMonitorBuffer) {
    return 0;
  }

  appDependencies.state.captureMonitorAnalyser.getFloatTimeDomainData(appDependencies.state.captureMonitorBuffer);
  let sum = 0;
  for (let i = 0; i < appDependencies.state.captureMonitorBuffer.length; i += 1) {
    const value = appDependencies.state.captureMonitorBuffer[i];
    sum += value * value;
  }
  return Math.sqrt(sum / appDependencies.state.captureMonitorBuffer.length);
}

function updateCaptureAutoGainState() {
  if (!appDependencies.state.captureGainNode || !appDependencies.state.captureContext) {
    return;
  }

  if (appDependencies.state.captureAutoGainTimer) {
    clearInterval(appDependencies.state.captureAutoGainTimer);
    appDependencies.state.captureAutoGainTimer = null;
  }

  const now = appDependencies.state.captureContext.currentTime;
  appDependencies.state.captureGainNode.gain.cancelScheduledValues(now);

  if (!appDependencies.state.autoGainEnabled) {
    appDependencies.state.captureAutoGainLevel = 1;
    appDependencies.state.captureAutoGainSmoothedRms = 0;
    appDependencies.state.captureGainNode.gain.setTargetAtTime(1, now, 0.35);
    return;
  }

  const tick = () => {
    if (!appDependencies.state.captureGainNode || !appDependencies.state.captureContext) {
      return;
    }

    const rms = sampleCaptureAutoGainRms();
    const smoothed = appDependencies.state.captureAutoGainSmoothedRms > 0
      ? appDependencies.state.captureAutoGainSmoothedRms * (1 - appDependencies.AUTO_GAIN_SMOOTHING) + rms * appDependencies.AUTO_GAIN_SMOOTHING
      : rms;
    appDependencies.state.captureAutoGainSmoothedRms = smoothed;

    let targetGain = 1;
    if (smoothed > 0 && smoothed < appDependencies.AUTO_GAIN_MIN_RMS) {
      targetGain = Math.min(appDependencies.AUTO_GAIN_MAX, appDependencies.AUTO_GAIN_TARGET_RMS / smoothed);
    } else if (smoothed < appDependencies.AUTO_GAIN_TARGET_RMS) {
      const ratio = (appDependencies.AUTO_GAIN_TARGET_RMS - smoothed) / Math.max(0.0001, appDependencies.AUTO_GAIN_TARGET_RMS - appDependencies.AUTO_GAIN_MIN_RMS);
      targetGain = 1 + ratio * 0.75;
    }

    const easedGain = appDependencies.state.captureAutoGainLevel * 0.7 + targetGain * 0.3;
    appDependencies.state.captureAutoGainLevel = Math.max(1, Math.min(appDependencies.AUTO_GAIN_MAX, easedGain));
    const at = appDependencies.state.captureContext.currentTime;
    appDependencies.state.captureGainNode.gain.cancelScheduledValues(at);
    appDependencies.state.captureGainNode.gain.setTargetAtTime(appDependencies.state.captureAutoGainLevel, at, 0.45);
  };

  tick();
  appDependencies.state.captureAutoGainTimer = setInterval(tick, appDependencies.AUTO_GAIN_ANALYZE_MS);
}

function bindDisplayEndEvents(displayStream) {
  const tracks = [...displayStream.getVideoTracks(), ...displayStream.getAudioTracks()];
  tracks.forEach((track) => {
    track.addEventListener(
      "ended",
      () => {
        if (!appDependencies.state.recording) return;
        appDependencies.setStatus("display_capture_ended");
        appDependencies.stopRecording();
      },
      { once: true }
    );
  });
}

function getDisplayCaptureDiagnostics(displayStream) {
  const audioTracks = displayStream?.getAudioTracks?.() || [];
  const videoTracks = displayStream?.getVideoTracks?.() || [];
  const firstVideo = videoTracks[0] || null;
  const settings = firstVideo?.getSettings?.() || {};

  return {
    audioTrackCount: audioTracks.length,
    audioLabels: audioTracks.map((track) => track.label || "unknown"),
    videoLabel: firstVideo?.label || "unknown",
    displaySurface: settings.displaySurface || "unknown",
  };
}

function logDisplayCaptureDiagnostics(displayStream, sourceMode) {
  const diagnostics = getDisplayCaptureDiagnostics(displayStream);
  console.info("[whistx] display capture diagnostics", {
    sourceMode,
    ...diagnostics,
  });
  return diagnostics;
}

async function prepareInputStream(sourceMode) {
  const mode = normalizeAudioSource(sourceMode);

  if (mode === "mic") {
    const micStream = await requestMicStream(mode);
    appDependencies.state.micStream = micStream;
    return buildMixedAudioStream([micStream]);
  }

  if (mode === "display") {
    const displayStream = await requestDisplayStream();
    const diagnostics = logDisplayCaptureDiagnostics(displayStream, mode);
    if (!hasAudioTrack(displayStream)) {
      const error = new Error("display_audio_not_found");
      error.diagnostics = diagnostics;
      throw error;
    }
    bindDisplayEndEvents(displayStream);
    appDependencies.state.displayStream = displayStream;
    setupDisplayCaptureVideo(displayStream);
    return buildMixedAudioStream([displayStream]);
  }

  const displayStream = await requestDisplayStream();
  const diagnostics = logDisplayCaptureDiagnostics(displayStream, mode);
  if (!hasAudioTrack(displayStream)) {
    appDependencies.showToast("画面共有音声が取れないため、マイクのみで開始します", "default", 5000);
    appDependencies.setStatus("recording_mic_fallback");
    appDependencies.state.vadRmsThreshold = vadThresholdForSource("mic");
    appDependencies.state.recordingAudioSource = "mic";
    appDependencies.state.recordingFallbackReason = "display_audio_not_found";
    appDependencies.state.displayStream = displayStream;
    setupDisplayCaptureVideo(displayStream);
    bindDisplayEndEvents(displayStream);
    const micStream = await requestMicStream("mic");
    appDependencies.state.micStream = micStream;
    return buildMixedAudioStream([micStream]);
  }
  bindDisplayEndEvents(displayStream);

  const micStream = await requestMicStream(mode);
  appDependencies.state.displayStream = displayStream;
  setupDisplayCaptureVideo(displayStream);
  appDependencies.state.micStream = micStream;
  return buildMixedAudioStream([displayStream, micStream]);
}

function ensureAudioLevelMatrix() {
  if (!appDependencies.audioLevelMatrixEl) return [];
  if (appDependencies.state.audioLevelColumns.length) return appDependencies.state.audioLevelColumns;

  const columns = [];
  appDependencies.audioLevelMatrixEl.innerHTML = "";

  for (let i = 0; i < appDependencies.AUDIO_LEVEL_COLUMNS; i += 1) {
    const column = document.createElement("div");
    column.className = "audio-level-column";

    const stack = document.createElement("div");
    stack.className = "audio-level-stack";

    const cells = [];

    for (let j = 0; j < appDependencies.AUDIO_LEVEL_SEGMENTS; j += 1) {
      const cell = document.createElement("span");
      cell.className = "audio-level-cell";
      stack.appendChild(cell);
      cells.push(cell);
    }

    column.appendChild(stack);
    appDependencies.audioLevelMatrixEl.appendChild(column);
    columns.push({ cells });
  }

  appDependencies.state.audioLevelColumns = columns;
  return columns;
}

function renderAudioLevel(level) {
  const normalized = Math.max(0, Math.min(1, Number(level) || 0));
  appDependencies.state.audioLevel = normalized;
  const inputLabel = appDependencies.audioLevelIndicatorEl?.querySelector(".audio-level-label");
  if (inputLabel) inputLabel.textContent = normalized > 0.03 ? "入力あり" : "入力待ち";
  const columns = ensureAudioLevelMatrix();
  if (!columns.length) return;

  const now = performance.now();
  const center = (columns.length - 1) / 2;

  columns.forEach((column, index) => {
    const distance = Math.abs(index - center);
    const profile = 1 - distance / Math.max(1, center + 0.5);
    const ripple = 0.84 + 0.24 * Math.sin(now / 240 + index * 0.55);
    const shaped = Math.max(0, Math.min(1, normalized * (0.62 + profile * 0.55) * ripple));
    const activeCount = Math.max(
      normalized > 0.03 ? 1 : 0,
      Math.min(appDependencies.AUDIO_LEVEL_SEGMENTS, Math.round(shaped * appDependencies.AUDIO_LEVEL_SEGMENTS))
    );

    column.cells.forEach((cell, cellIndex) => {
      const active = cellIndex >= appDependencies.AUDIO_LEVEL_SEGMENTS - activeCount;
      cell.classList.toggle("is-active", active);
    });
  });
}

function sampleVad() {
  if (!appDependencies.state.vadAnalyser || !appDependencies.state.vadBuffer) return;

  appDependencies.state.vadAnalyser.getFloatTimeDomainData(appDependencies.state.vadBuffer);

  let sum = 0;
  for (let i = 0; i < appDependencies.state.vadBuffer.length; i += 1) {
    const value = appDependencies.state.vadBuffer[i];
    sum += value * value;
  }

  const rms = Math.sqrt(sum / appDependencies.state.vadBuffer.length);
  const now = performance.now();
  updateVadNoiseFloor(rms, now);
  const threshold = appDependencies.state.vadRmsThreshold || vadThresholdForSource(currentVadSourceMode(), appDependencies.state.vadNoiseFloor);
  const leveled = Math.max(0, rms - appDependencies.AUDIO_LEVEL_NOISE_FLOOR);
  const boosted = Math.min(1, Math.pow(leveled * appDependencies.AUDIO_LEVEL_GAIN, appDependencies.AUDIO_LEVEL_EXPONENT));
  const smoothed = appDependencies.state.audioLevel * 0.72 + boosted * 0.28;
  renderAudioLevel(smoothed);
  appDependencies.state.vadFrameCount += 1;
  if (rms >= threshold) {
    appDependencies.state.vadSpeechFrameCount += 1;
    appDependencies.state.vadLastSpeechAt = now;
  }
  appDependencies.updateRecordingTelemetry();
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

  appDependencies.state.audioContext = context;
  appDependencies.state.vadSource = source;
  appDependencies.state.vadAnalyser = analyser;
  appDependencies.state.vadBuffer = new Float32Array(analyser.fftSize);
  appDependencies.state.vadFrameCount = 0;
  appDependencies.state.vadSpeechFrameCount = 0;
  appDependencies.state.vadLastSpeechAt = performance.now();
  appDependencies.state.vadNoiseFloor = null;
  appDependencies.state.vadNoiseFloorAt = performance.now();
  appDependencies.state.vadRmsThreshold = vadThresholdForSource(currentVadSourceMode(), null);
  appDependencies.state.vadTimer = setInterval(sampleVad, VAD_SAMPLE_MS);
  appDependencies.updateRecordingTelemetry();
}

function stopVad() {
  if (appDependencies.state.vadTimer) {
    clearInterval(appDependencies.state.vadTimer);
    appDependencies.state.vadTimer = null;
  }

  if (appDependencies.state.vadSource) {
    try {
      appDependencies.state.vadSource.disconnect();
    } catch {
      // ignore
    }
    appDependencies.state.vadSource = null;
  }

  if (appDependencies.state.vadAnalyser) {
    try {
      appDependencies.state.vadAnalyser.disconnect();
    } catch {
      // ignore
    }
    appDependencies.state.vadAnalyser = null;
  }

  if (appDependencies.state.audioContext) {
    appDependencies.state.audioContext.close().catch(() => {
      // ignore
    });
    appDependencies.state.audioContext = null;
  }

  appDependencies.state.vadBuffer = null;
  appDependencies.state.vadFrameCount = 0;
  appDependencies.state.vadSpeechFrameCount = 0;
  appDependencies.state.vadLastSpeechAt = 0;
  appDependencies.state.vadNoiseFloor = null;
  appDependencies.state.vadNoiseFloorAt = 0;
  appDependencies.state.vadRmsThreshold = vadThresholdForSource(currentVadSourceMode(), null);
  appDependencies.state.audioLevel = 0;
  renderAudioLevel(0);
  appDependencies.updateRecordingTelemetry();
}

function cleanupCaptureGraph() {
  if (appDependencies.state.captureAutoGainTimer) {
    clearInterval(appDependencies.state.captureAutoGainTimer);
    appDependencies.state.captureAutoGainTimer = null;
  }
  appDependencies.state.captureSources.forEach((source) => {
    try {
      source.disconnect();
    } catch {
      // ignore
    }
  });
  appDependencies.state.captureSources = [];
  appDependencies.state.captureDestination = null;
  appDependencies.state.captureGainNode = null;
  appDependencies.state.captureMonitorAnalyser = null;
  appDependencies.state.captureMonitorBuffer = null;
  appDependencies.state.captureAutoGainLevel = 1;
  appDependencies.state.captureAutoGainSmoothedRms = 0;

  if (appDependencies.state.captureContext) {
    appDependencies.state.captureContext.close().catch(() => {
      // ignore
    });
    appDependencies.state.captureContext = null;
  }
}

function snapshotVadCounters() {
  return {
    frameCount: appDependencies.state.vadFrameCount,
    speechFrameCount: appDependencies.state.vadSpeechFrameCount,
    startedAt: performance.now(),
  };
}

function buildVadDecision(snapshot, endedAt, sourceMode) {
  return calculateVadDecision({
    analyserEnabled: !!appDependencies.state.vadAnalyser,
    snapshot,
    frameCount: appDependencies.state.vadFrameCount,
    speechFrameCount: appDependencies.state.vadSpeechFrameCount,
    endedAt: Number.isFinite(endedAt) ? endedAt : performance.now(),
    lastSpeechAt: appDependencies.state.vadLastSpeechAt,
    segmentStartedAt: appDependencies.state.segmentStartedAt,
    chunkMs: appDependencies.state.chunkMs,
    sourceMode,
  });
}

function shouldSkipChunkByVad(durationMs, vadDecision) {
  return calculateShouldSkipChunkByVad(durationMs, vadDecision, appDependencies.CLIENT_VAD_DROP_ENABLED);
}

function cleanupMedia() {
  if (appDependencies.state.recorder) {
    appDependencies.state.recorder.ondataavailable = null;
    appDependencies.state.recorder.onstop = null;
    appDependencies.state.recorder = null;
  }
  appDependencies.state.recorderOptions = null;

  stopVad();
  cleanupCaptureGraph();

  const streams = [appDependencies.state.stream, appDependencies.state.micStream, appDependencies.state.displayStream];
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

  appDependencies.state.stream = null;
  appDependencies.state.micStream = null;
  appDependencies.state.displayStream = null;
  if (appDependencies.state.displayCaptureVideo) {
    try {
      appDependencies.state.displayCaptureVideo.pause();
      appDependencies.state.displayCaptureVideo.srcObject = null;
    } catch {
      // ignore
    }
  }
  appDependencies.state.displayCaptureVideo = null;
  appDependencies.state.screenshotCanvas = null;
  appDependencies.state.screenshotDiffCanvas = null;
  appDependencies.state.previousScreenshotSignature = null;
}

  return { currentVadSourceMode, updateVadNoiseFloor, hasAudioTrack, setupDisplayCaptureVideo, captureDisplayScreenshot, buildScreenshotSignature, shouldSkipScreenshotByDiff, requestMicStream, requestDisplayStream, ensureAudioContextResumed, buildMixedAudioStream, sampleCaptureAutoGainRms, updateCaptureAutoGainState, bindDisplayEndEvents, getDisplayCaptureDiagnostics, logDisplayCaptureDiagnostics, prepareInputStream, ensureAudioLevelMatrix, renderAudioLevel, sampleVad, setupVad, stopVad, cleanupCaptureGraph, snapshotVadCounters, buildVadDecision, shouldSkipChunkByVad, cleanupMedia };
}
