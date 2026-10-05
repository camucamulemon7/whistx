import test from "node:test";
import assert from "node:assert/strict";
import { createAppState } from "../../web/src/state/store.js";
import { createSettingsController } from "../../web/src/controllers/settings.js";
import { createWorkspaceController } from "../../web/src/controllers/workspace.js";
import { createRecordingController } from "../../web/src/controllers/recording.js";

const state = () => createAppState({ chunkDefaultSeconds: 30, speakerMax: 8, defaultPromptTemplate: "synthetic" });

test("application stores do not share recordings, authentication, history or prompt arrays", () => {
  const first = state();
  const second = state();
  first.segments.push({ text: "synthetic" });
  first.auth.authenticated = true;
  first.history.items.push({ id: "synthetic" });
  first.promptTemplates[0].content = "changed";
  assert.deepEqual(second.segments, []);
  assert.equal(second.auth.authenticated, false);
  assert.deepEqual(second.history.items, []);
  assert.equal(second.promptTemplates[0].content, "synthetic");
});

test("settings controller bounds chunks and resolves automatic language without the app", () => {
  const dependencies = { CHUNK_DEFAULT_SECONDS: 30, CHUNK_MIN_SECONDS: 5, CHUNK_MAX_SECONDS: 60, languageEl: { value: " auto " } };
  const settings = createSettingsController(dependencies);
  assert.equal(settings.normalizeChunkSeconds(NaN), 30);
  assert.equal(settings.normalizeChunkSeconds(-100), 5);
  assert.equal(settings.normalizeChunkSeconds(100), 60);
  assert.equal(settings.normalizeChunkSeconds(12.6), 13);
  assert.equal(settings.selectedLanguage(), null);
  dependencies.languageEl.value = " JA ";
  assert.equal(settings.selectedLanguage(), "ja");
});

test("workspace protects pending live audio and all non-idle recording phases", () => {
  const dependencies = { state: state(), refinementController: null };
  const workspace = createWorkspaceController(dependencies);
  assert.equal(workspace.shouldProtectWorkspaceFromUnload(), false);
  for (const phase of ["starting", "recording", "stopping", "finalizing"]) {
    dependencies.state.recordingPhase = phase;
    assert.equal(workspace.isRecordingInteractionLocked(), true);
    assert.equal(workspace.shouldProtectWorkspaceFromUnload(), true);
  }
  dependencies.state.recordingPhase = "idle";
  dependencies.state.liveCapture = { pending: new Map([[1, "synthetic"]]) };
  assert.equal(workspace.shouldProtectWorkspaceFromUnload(), true);
  dependencies.state.liveCapture.pending.clear();
  assert.equal(workspace.shouldProtectWorkspaceFromUnload(), false);
  dependencies.refinementController = {};
  assert.equal(workspace.isRecordingInteractionLocked(), true);
});

test("recording controller resets session handles without discarding transcript state", () => {
  const recordingState = state();
  recordingState.runtimeSessionId = "synthetic-session";
  recordingState.runtimeSessionToken = "synthetic-token";
  recordingState.runtimeSessionFinalized = true;
  recordingState.segments.push({ text: "keep" });
  createRecordingController({ state: recordingState }).resetRuntimeSessionState();
  assert.equal(recordingState.runtimeSessionId, "");
  assert.equal(recordingState.runtimeSessionToken, "");
  assert.equal(recordingState.runtimeSessionFinalized, false);
  assert.deepEqual(recordingState.segments, [{ text: "keep" }]);
});

test("all feature controller constructors are inert until the app wires events", async () => {
  const names = ["ai", "auth", "capabilities", "glossary", "history", "layout", "media-capture", "recording", "screenshot-viewer", "settings", "telemetry", "transcript", "transcription", "workspace"];
  const unavailableDependencies = new Proxy({}, { get() { throw new Error("construction accessed application state or DOM"); } });
  for (const name of names) {
    const module = await import(`../../web/src/controllers/${name}.js`);
    const factory = Object.values(module)[0];
    const controller = factory(unavailableDependencies);
    assert.ok(Object.keys(controller).length > 0, name);
    assert.ok(Object.values(controller).every(action => typeof action === "function"), name);
  }
});


test("recording seeds use secure randomness and fit the server session identifier", () => {
  const recording = createRecordingController({});
  const original = Math.random;
  try {
    Math.random = () => { throw new Error("insecure randomness must not be used"); };
    const first = recording.generateSessionSeed();
    const second = recording.generateSessionSeed();
    assert.match(first, /^sess-[0-9a-f]{32}$/);
    assert.notEqual(first, second);
    assert.ok((first + "_20261004120000_abcd").length <= 96);
  } finally {
    Math.random = original;
  }
});

test("late capabilities responses cannot overwrite the latest notices or settings", async () => {
  const { createCapabilitiesController } = await import("../../web/src/controllers/capabilities.js");
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  const pending = [];
  const rendered = [];
  const dependencies = {
    state: state(), DIARIZATION_SPEAKER_MIN: 1, DIARIZATION_SPEAKER_MAX: 8,
    renderBanners: banners => rendered.push(banners),
  };
  for (const action of ["logClientEvent", "setSessionSettingsLocked", "applyBranding", "renderPromptTemplateButtons", "renderAuthState", "setProofread", "setStatus", "applyDiarizationSpeakerSettings", "applyDiarizationEnabled", "updateDiarizationSpeakerUi"]) dependencies[action] = () => {};
  try {
    globalThis.document = { querySelector: () => ({ textContent: "", checked: false }) };
    globalThis.fetch = () => new Promise(resolve => pending.push(resolve));
    const controller = createCapabilitiesController(dependencies);
    const older = controller.loadCapabilities();
    const latest = controller.loadCapabilities();
    pending[1](new Response(JSON.stringify({ banners: [{ message: "Current notice" }], asrReady: true, model: "current" })));
    await latest;
    pending[0](new Response(JSON.stringify({ banners: [{ message: "Old notice" }], asrReady: false })));
    await older;
    assert.deepEqual(rendered, [[{ message: "Current notice" }]]);
    assert.equal(dependencies.state.asrAvailable, true);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalDocument === undefined) delete globalThis.document;
    else globalThis.document = originalDocument;
  }
});
