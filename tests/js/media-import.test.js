import assert from 'node:assert/strict';
import test from 'node:test';
import { MEDIA_MAX_BYTES, validateMediaFile } from '../../web/src/controllers/media-import.js';
import { createWorkspaceController } from '../../web/src/controllers/workspace.js';
import { createAppState } from '../../web/src/state/store.js';

test('file selection accepts audio/video and rejects empty, oversized or unsupported input', () => {
  for (const name of ['会議.WAV', '会議.mp4', 'test.mov', 'test.webm', 'test.flac']) {
    assert.equal(validateMediaFile({ name, size: 1024 }), '');
  }
  assert.match(validateMediaFile({ name: 'empty.wav', size: 0 }), /空/);
  assert.match(validateMediaFile({ name: 'large.mp4', size: MEDIA_MAX_BYTES + 1 }), /256 MB/);
  assert.match(validateMediaFile({ name: 'remote.m3u8', size: 100 }), /対応/);
  assert.match(validateMediaFile({ name: 'a'.repeat(240) + '.wav', size: 100 }), /240/);
});

test('an active media import locks destructive actions and protects page navigation', () => {
  const state = createAppState({ chunkDefaultSeconds: 30, speakerMax: 8, defaultPromptTemplate: '' });
  const workspace = createWorkspaceController({ state, refinementController: null });
  state.mediaImportController = new AbortController();
  assert.equal(workspace.isRecordingInteractionLocked(), true);
  assert.equal(workspace.shouldProtectWorkspaceFromUnload(), true);
  state.mediaImportController = null;
  assert.equal(workspace.isRecordingInteractionLocked(), false);
});
