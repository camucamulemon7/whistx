import { formatStatusText } from "../ui/format.js";
import { clampSpeakerCount as clampSpeakerCountValue } from "../ui/format.js";
import { normalizeSpeakerMode } from "../ui/format.js";
import { normalizeAudioSource } from "../audio/vad.js";
import { vadThresholdForSource } from "../audio/vad.js";

export function createSettingsController(appDependencies) {
function selectedLanguage() {
  const value = String(appDependencies.languageEl?.value || "").trim().toLowerCase();
  if (!value || value === "auto") {
    return null;
  }
  return value;
}

function setStatus(text) {
  const raw = String(text || "").trim();
  appDependencies.state.latestStatus = raw || "idle";
  appDependencies.statusTextEl.textContent = formatStatusText(raw);
  appDependencies.statusTextEl.dataset.state = String(raw || "idle").toLowerCase();
}

function applyBranding(title, tagline) {
  const cleanTitle = String(title || "").trim();
  const cleanTagline = String(tagline || "").trim();

  if (appDependencies.brandTitleEl && cleanTitle) {
    appDependencies.brandTitleEl.textContent = cleanTitle;
    document.title = cleanTitle;
  }

  if (appDependencies.brandTaglineEl && cleanTagline) {
    appDependencies.brandTaglineEl.textContent = cleanTagline;
  }
}

function applyCaptureScreenshotsEnabled(value, options = {}) {
  const persist = options.persist !== false;
  const enabled = !!value;
  appDependencies.state.captureScreenshotsEnabled = enabled;
  const meetingCaptureEnabled = document.querySelector("#meetingCaptureEnabled");
  if (meetingCaptureEnabled) meetingCaptureEnabled.checked = enabled;
  if (appDependencies.captureScreenshotsEnabledEl) {
    appDependencies.captureScreenshotsEnabledEl.checked = enabled;
  }
  if (appDependencies.captureScreenshotsStateTextEl) {
    appDependencies.captureScreenshotsStateTextEl.textContent = enabled ? "ON" : "OFF";
  }
  if (persist) {
    try {
      localStorage.setItem("whistx_capture_screenshots_enabled", enabled ? "1" : "0");
    } catch {
      // ignore
    }
  }
}

function applyScreenshotDiffSkipEnabled(value, options = {}) {
  const persist = options.persist !== false;
  const enabled = !!value;
  appDependencies.state.screenshotDiffSkipEnabled = enabled;
  appDependencies.state.previousScreenshotSignature = null;
  if (appDependencies.screenshotDiffSkipEnabledEl) {
    appDependencies.screenshotDiffSkipEnabledEl.checked = enabled;
  }
  if (appDependencies.screenshotDiffSkipStateTextEl) {
    appDependencies.screenshotDiffSkipStateTextEl.textContent = enabled ? "ON" : "OFF";
  }
  if (persist) {
    try {
      localStorage.setItem("whistx_screenshot_diff_skip_enabled", enabled ? "1" : "0");
    } catch {
      // ignore
    }
  }
}

function applyShowTranscriptAudioEnabled(value, options = {}) {
  const persist = options.persist !== false;
  const enabled = !!value;
  appDependencies.state.showTranscriptAudioEnabled = enabled;
  if (appDependencies.showTranscriptAudioEnabledEl) {
    appDependencies.showTranscriptAudioEnabledEl.checked = enabled;
  }
  if (appDependencies.showTranscriptAudioStateTextEl) {
    appDependencies.showTranscriptAudioStateTextEl.textContent = enabled ? "ON" : "OFF";
  }
  document.body.classList.toggle("is-transcript-audio-hidden", !enabled);
  if (!enabled) {
    document.querySelectorAll(".log-inline-audio").forEach((element) => {
      if (typeof element.pause === "function") {
        element.pause();
      }
    });
  }
  if (persist) {
    try {
      localStorage.setItem("whistx_show_transcript_audio_enabled", enabled ? "1" : "0");
    } catch {
      // ignore
    }
  }
}

function applyAutoGainEnabled(value, options = {}) {
  const persist = options.persist !== false;
  const enabled = !!value;
  appDependencies.state.autoGainEnabled = enabled;
  if (appDependencies.autoGainEnabledEl) {
    appDependencies.autoGainEnabledEl.checked = enabled;
  }
  if (appDependencies.autoGainStateTextEl) {
    appDependencies.autoGainStateTextEl.textContent = enabled ? "ON" : "OFF";
  }
  if (persist) {
    try {
      localStorage.setItem("whistx_auto_gain_enabled", enabled ? "1" : "0");
    } catch {
      // ignore
    }
  }
  appDependencies.updateCaptureAutoGainState();
}

function applyDiarizationEnabled(value, options = {}) {
  const persist = options.persist !== false;
  const enabled = !!value;
  appDependencies.state.diarizationEnabled = enabled;

  if (appDependencies.diarizationToggleEl) {
    appDependencies.diarizationToggleEl.checked = enabled;
  }

  if (appDependencies.diarizationStateTextEl) {
    if (!appDependencies.state.diarizationAvailable) {
      appDependencies.diarizationStateTextEl.textContent = "利用不可";
    } else {
      appDependencies.diarizationStateTextEl.textContent = enabled ? "ON" : "OFF";
    }
  }

  if (persist) {
    try {
      localStorage.setItem("whistx_diarization_enabled", enabled ? "1" : "0");
    } catch {
      // ignore
    }
  }
  updateDiarizationSpeakerUi();
  return enabled;
}

function clampSpeakerCount(value) {
  const cap = Math.max(appDependencies.DIARIZATION_SPEAKER_MIN, Number(appDependencies.state.diarizationSpeakerCap || appDependencies.DIARIZATION_SPEAKER_MAX));
  return clampSpeakerCountValue(value, cap, appDependencies.DIARIZATION_SPEAKER_MIN);
}

function updateDiarizationSpeakerUi() {
  const available = !!appDependencies.state.diarizationAvailable;
  const enabled = !!appDependencies.state.diarizationEnabled;
  const locked = appDependencies.isRecordingInteractionLocked();
  const mode = normalizeSpeakerMode(appDependencies.state.diarizationSpeakerMode);
  const visible = available && enabled;
  const controlsEnabled = visible && !locked;

  const autoMode = mode === "auto";
  const fixedMode = mode === "fixed";
  const rangeMode = mode === "range";

  if (appDependencies.settingsAdvancedToggleEl) {
    appDependencies.settingsAdvancedToggleEl.hidden = !visible;
  }
  if (!visible && appDependencies.state.advancedSettingsOpen) {
    appDependencies.applyAdvancedSettingsOpen(false);
  }

  if (appDependencies.diarizationConfigRowEl) {
    appDependencies.diarizationConfigRowEl.classList.toggle("is-hidden", !visible);
    appDependencies.diarizationConfigRowEl.hidden = !visible;
  }
  if (appDependencies.diarizationSpeakerHintEl) {
    appDependencies.diarizationSpeakerHintEl.classList.toggle("is-hidden", !visible);
    appDependencies.diarizationSpeakerHintEl.hidden = !visible;
  }

  if (appDependencies.diarizationSpeakerModeEl) {
    appDependencies.diarizationSpeakerModeEl.value = mode;
    appDependencies.diarizationSpeakerModeEl.disabled = !controlsEnabled;
  }

  if (appDependencies.diarizationSpeakerCountEl) {
    appDependencies.diarizationSpeakerCountEl.value = String(appDependencies.state.diarizationSpeakerCount);
    appDependencies.diarizationSpeakerCountEl.disabled = !controlsEnabled || !fixedMode;
  }

  if (appDependencies.diarizationMinSpeakersEl) {
    appDependencies.diarizationMinSpeakersEl.value = String(appDependencies.state.diarizationMinSpeakers);
    appDependencies.diarizationMinSpeakersEl.disabled = !controlsEnabled || !rangeMode;
  }

  if (appDependencies.diarizationMaxSpeakersEl) {
    appDependencies.diarizationMaxSpeakersEl.value = String(appDependencies.state.diarizationMaxSpeakers);
    appDependencies.diarizationMaxSpeakersEl.disabled = !controlsEnabled || !rangeMode;
  }

  if (!visible) {
    return;
  }
  if (!appDependencies.diarizationSpeakerHintEl) return;
  if (autoMode) {
    appDependencies.diarizationSpeakerHintEl.textContent = "自動推定: 発話内容から話者人数を推定します。";
    return;
  }
  if (fixedMode) {
    appDependencies.diarizationSpeakerHintEl.textContent = `固定人数: ${appDependencies.state.diarizationSpeakerCount}人として分離します。`;
    return;
  }
  appDependencies.diarizationSpeakerHintEl.textContent = `範囲指定: ${appDependencies.state.diarizationMinSpeakers}〜${appDependencies.state.diarizationMaxSpeakers}人の範囲で推定します。`;
}

function applyDiarizationSpeakerSettings(value, options = {}) {
  const persist = options.persist !== false;
  const nextMode = normalizeSpeakerMode(value?.mode ?? appDependencies.state.diarizationSpeakerMode);
  const nextCount = clampSpeakerCount(value?.count ?? appDependencies.state.diarizationSpeakerCount);
  let nextMin = clampSpeakerCount(value?.min ?? appDependencies.state.diarizationMinSpeakers);
  let nextMax = clampSpeakerCount(value?.max ?? appDependencies.state.diarizationMaxSpeakers);

  if (nextMin > nextMax) {
    const tmp = nextMin;
    nextMin = nextMax;
    nextMax = tmp;
  }

  appDependencies.state.diarizationSpeakerMode = nextMode;
  appDependencies.state.diarizationSpeakerCount = nextCount;
  appDependencies.state.diarizationMinSpeakers = nextMin;
  appDependencies.state.diarizationMaxSpeakers = nextMax;

  updateDiarizationSpeakerUi();

  if (persist) {
    try {
      localStorage.setItem("whistx_diarization_speaker_mode", nextMode);
      localStorage.setItem("whistx_diarization_speaker_count", String(nextCount));
      localStorage.setItem("whistx_diarization_min_speakers", String(nextMin));
      localStorage.setItem("whistx_diarization_max_speakers", String(nextMax));
      appDependencies.state.hasSavedDiarizationSpeakerSettings = true;
    } catch {
      // ignore
    }
  }
}

function resolveDiarizationStartOptions() {
  const mode = normalizeSpeakerMode(appDependencies.state.diarizationSpeakerMode);
  if (!appDependencies.state.diarizationAvailable || !appDependencies.state.diarizationEnabled) {
    return {
      diarizationNumSpeakers: 0,
      diarizationMinSpeakers: 0,
      diarizationMaxSpeakers: 0,
    };
  }
  if (mode === "fixed") {
    return {
      diarizationNumSpeakers: clampSpeakerCount(appDependencies.state.diarizationSpeakerCount),
      diarizationMinSpeakers: 0,
      diarizationMaxSpeakers: 0,
    };
  }
  if (mode === "range") {
    const min = clampSpeakerCount(appDependencies.state.diarizationMinSpeakers);
    const max = clampSpeakerCount(appDependencies.state.diarizationMaxSpeakers);
    return {
      diarizationNumSpeakers: 0,
      diarizationMinSpeakers: Math.min(min, max),
      diarizationMaxSpeakers: Math.max(min, max),
    };
  }
  return {
    diarizationNumSpeakers: 0,
    diarizationMinSpeakers: 0,
    diarizationMaxSpeakers: 0,
  };
}

function normalizeChunkSeconds(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return appDependencies.CHUNK_DEFAULT_SECONDS;
  return Math.max(appDependencies.CHUNK_MIN_SECONDS, Math.min(appDependencies.CHUNK_MAX_SECONDS, Math.round(num)));
}

function updateChunkHint(seconds) {
  if (!appDependencies.chunkHintEl) return;
  if (seconds <= 15) {
    appDependencies.chunkHintEl.textContent = "短め上限。無音で早めに確定";
    return;
  }
  if (seconds <= 30) {
    appDependencies.chunkHintEl.textContent = "バランス。無音で自然に区切る";
    return;
  }
  if (seconds <= 45) {
    appDependencies.chunkHintEl.textContent = "精度優先。長めに文脈を保持";
    return;
  }
  appDependencies.chunkHintEl.textContent = "最大長。無音が少ない会話向け";
}

function updatePresetActive(seconds) {
  appDependencies.presetButtons.forEach((button) => {
    const raw = button.getAttribute("data-chunk-preset") || "";
    button.classList.toggle("is-active", Number(raw) === seconds);
  });
}

function applyChunkSeconds(value) {
  const seconds = normalizeChunkSeconds(value);
  if (appDependencies.chunkSecondsEl) {
    appDependencies.chunkSecondsEl.value = String(seconds);
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

function audioSourceHintText(source) {
  if (source === "display") {
    return "画面共有音声はブラウザ制約で取得できない場合があります。Chrome/Edge のタブ共有が最も安定します。";
  }
  if (source === "both") {
    return "画面共有音声とマイクを混ぜます。画面共有音声は共有面やブラウザ制約の影響を受けます。";
  }
  return "通常のマイク入力を使います。会議アプリ音声は画面共有では取得できない場合があります。";
}

function applyAudioSource(value) {
  const source = normalizeAudioSource(value);
  if (appDependencies.audioSourceEl) {
    appDependencies.audioSourceEl.value = source;
  }
  if (appDependencies.audioSourceHintTextEl || appDependencies.audioSourceHintEl) {
    (appDependencies.audioSourceHintTextEl || appDependencies.audioSourceHintEl).textContent = audioSourceHintText(source);
  }
  appDependencies.state.vadRmsThreshold = vadThresholdForSource(source, appDependencies.state.vadNoiseFloor);
  try {
    localStorage.setItem("whistx_audio_source", source);
  } catch {
    // ignore
  }
  return source;
}

function buildAutoSaveTitle() {
  const explicit = String(appDependencies.saveTitleInputEl?.value || "").trim();
  if (explicit) return explicit;
  const transcript = appDependencies.extractTranscriptText().replace(/\s+/g, " ").trim();
  if (transcript) return transcript.slice(0, 30);
  const language = selectedLanguage() || "Transcript";
  return `${language} ${new Date().toLocaleString("ja-JP")}`;
}

  return { selectedLanguage, setStatus, applyBranding, applyCaptureScreenshotsEnabled, applyScreenshotDiffSkipEnabled, applyShowTranscriptAudioEnabled, applyAutoGainEnabled, applyDiarizationEnabled, clampSpeakerCount, updateDiarizationSpeakerUi, applyDiarizationSpeakerSettings, resolveDiarizationStartOptions, normalizeChunkSeconds, updateChunkHint, updatePresetActive, applyChunkSeconds, audioSourceHintText, applyAudioSource, buildAutoSaveTitle };
}
