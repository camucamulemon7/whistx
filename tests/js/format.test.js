import test from "node:test";
import assert from "node:assert/strict";

import {
  clampSpeakerCount,
  escapeHtml,
  formatAudioSource,
  formatLanguageLabel,
  formatStatusText,
  formatTimestamp,
  normalizeBannerType,
  normalizeProofreadMode,
  normalizeSpeakerMode,
} from "../../web/src/ui/format.js";

test("UI formatters produce stable Japanese labels", () => {
  assert.equal(formatStatusText("recording: 1"), "録音中");
  assert.equal(formatStatusText("summary_error"), "要約失敗");
  assert.equal(formatAudioSource("both"), "両方");
  assert.equal(formatLanguageLabel("JA"), "日本語");
  assert.equal(formatTimestamp(125000), "02:05");
});

test("UI values are escaped and normalized", () => {
  assert.equal(escapeHtml(`<a title="'">&`), "&lt;a title=&quot;&#39;&quot;&gt;&amp;");
  assert.equal(normalizeBannerType("ERROR"), "error");
  assert.equal(normalizeBannerType("unknown"), "info");
  assert.equal(normalizeProofreadMode("translate_en"), "translate_en");
  assert.equal(normalizeProofreadMode("bad"), "proofread");
  assert.equal(normalizeSpeakerMode("fixed"), "fixed");
  assert.equal(normalizeSpeakerMode("bad"), "auto");
  assert.equal(clampSpeakerCount(18, 8), 8);
  assert.equal(clampSpeakerCount("bad", 8), 1);
});
