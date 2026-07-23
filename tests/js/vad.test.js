import test from "node:test";
import assert from "node:assert/strict";

import {
  buildVadDecision,
  minSegmentDuration,
  normalizeAudioSource,
  shouldCutOnSilence,
  shouldSkipChunkByVad,
  vadThresholdForSource,
} from "../../web/src/audio/vad.js";

test("audio source and adaptive threshold are deterministic", () => {
  assert.equal(normalizeAudioSource("display"), "display");
  assert.equal(normalizeAudioSource("invalid"), "mic");
  assert.equal(vadThresholdForSource("mic"), 0.0075);
  assert.ok(vadThresholdForSource("display", 0.01) > 0.03);
});

test("VAD decision identifies a silent short chunk", () => {
  const decision = buildVadDecision({
    analyserEnabled: true,
    snapshot: { frameCount: 10, speechFrameCount: 2, startedAt: 1000 },
    frameCount: 110,
    speechFrameCount: 2,
    endedAt: 3000,
    lastSpeechAt: 1000,
    segmentStartedAt: 1000,
    chunkMs: 2000,
    sourceMode: "mic",
  });
  assert.equal(decision.skip, true);
  assert.equal(decision.speechRatio, 0);
  assert.equal(shouldSkipChunkByVad(2000, decision, true), true);
  assert.equal(shouldSkipChunkByVad(5000, decision, true), false);
});

test("silence cutting observes source policy and minimum duration", () => {
  assert.equal(minSegmentDuration(30000, "mic"), 13500);
  assert.equal(
    shouldCutOnSilence({ elapsedMs: 10000, silenceMs: 2000, chunkMs: 30000, sourceMode: "mic" }),
    false
  );
  assert.equal(
    shouldCutOnSilence({ elapsedMs: 20000, silenceMs: 4000, chunkMs: 30000, sourceMode: "mic" }),
    true
  );
});
