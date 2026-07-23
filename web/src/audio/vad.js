export const VAD_SAMPLE_MS = 80;
export const VAD_MIN_SPEECH_RATIO = 0.025;
export const VAD_MIN_ACTIVE_MS = 120;

const SOURCE_CUT_POLICY = {
  mic: { minSilenceMs: 450, maxSilenceMs: 1100, silenceRatio: 0.18, minSegmentRatio: 0.45 },
  display: { minSilenceMs: 550, maxSilenceMs: 1400, silenceRatio: 0.22, minSegmentRatio: 0.5 },
  both: { minSilenceMs: 500, maxSilenceMs: 1500, silenceRatio: 0.2, minSegmentRatio: 0.5 },
};
const NOISE_FLOOR_OFFSET = { mic: 0.0012, display: 0.0008, both: 0.001 };
const NOISE_FLOOR_MULTIPLIER = { mic: 3, display: 3.4, both: 3.2 };

export function normalizeAudioSource(value) {
  if (value === "display" || value === "both") return value;
  return "mic";
}

export function vadSourceCutPolicy(source) {
  return SOURCE_CUT_POLICY[normalizeAudioSource(source)];
}

export function vadThresholdForSource(source, noiseFloor = null) {
  const mode = normalizeAudioSource(source);
  const baseThreshold = mode === "display" ? 0.0045 : mode === "both" ? 0.0065 : 0.0075;
  const floor = Number.isFinite(noiseFloor) ? Math.max(0, Number(noiseFloor)) : 0;
  if (floor <= 0) return baseThreshold;
  return Math.max(baseThreshold, floor * NOISE_FLOOR_MULTIPLIER[mode] + NOISE_FLOOR_OFFSET[mode]);
}

export function buildVadDecision({
  analyserEnabled,
  snapshot,
  frameCount,
  speechFrameCount,
  endedAt,
  lastSpeechAt,
  segmentStartedAt,
  chunkMs,
  sourceMode,
}) {
  if (!analyserEnabled) {
    return { enabled: false, speechRatio: 1, activeMs: chunkMs, silenceMs: 0, skip: false };
  }
  const totalFrames = Math.max(1, frameCount - snapshot.frameCount);
  const speechFrames = Math.max(0, speechFrameCount - snapshot.speechFrameCount);
  const chunkEndAt = Number.isFinite(endedAt) ? endedAt : snapshot.startedAt;
  const lastActiveAt = lastSpeechAt || segmentStartedAt || snapshot.startedAt || chunkEndAt;
  const speechRatio = speechFrames / totalFrames;
  const activeMs = speechFrames * VAD_SAMPLE_MS;
  return {
    enabled: true,
    speechRatio,
    activeMs,
    silenceMs: Math.max(0, chunkEndAt - lastActiveAt),
    sourceMode: normalizeAudioSource(sourceMode),
    policy: vadSourceCutPolicy(sourceMode),
    skip: speechRatio < VAD_MIN_SPEECH_RATIO && activeMs < VAD_MIN_ACTIVE_MS,
  };
}

export function shouldSkipChunkByVad(durationMs, decision, enabled = false) {
  if (!enabled || !decision?.enabled || !decision.skip) return false;
  const safeDurationMs = Number.isFinite(durationMs) ? Math.max(0, durationMs) : 0;
  if (safeDurationMs >= 4000) return false;
  return decision.speechRatio < 0.01 && decision.activeMs < 80;
}

export function minSegmentDuration(chunkMs, sourceMode, minimumMs = 12000) {
  const policy = vadSourceCutPolicy(sourceMode);
  return Math.max(Math.min(minimumMs, chunkMs), Math.round(chunkMs * policy.minSegmentRatio));
}

export function shouldCutOnSilence({
  elapsedMs,
  silenceMs,
  chunkMs,
  sourceMode,
  relaxed = false,
  minimumMs = 12000,
}) {
  if (elapsedMs < minSegmentDuration(chunkMs, sourceMode, minimumMs)) return false;
  const policy = vadSourceCutPolicy(sourceMode);
  const minSilenceMs = relaxed
    ? Math.max(220, Math.min(policy.minSilenceMs, Math.round(policy.minSilenceMs * 0.55)))
    : policy.minSilenceMs;
  if (silenceMs < minSilenceMs) return false;
  const silenceRatio = relaxed ? Math.max(0.1, policy.silenceRatio * 0.72) : policy.silenceRatio;
  return silenceMs >= Math.min(policy.maxSilenceMs, Math.max(minSilenceMs + 160, Math.round(elapsedMs * silenceRatio)));
}
