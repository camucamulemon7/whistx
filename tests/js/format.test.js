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

test("recording connection states are presented in Japanese", () => {
  assert.equal(formatStatusText("starting"), "録音準備中");
  assert.equal(formatStatusText("finalizing"), "最終処理中");
  assert.equal(formatStatusText("completed"), "録音完了");
  assert.equal(formatStatusText("finalize_failed"), "最終処理失敗");
  assert.equal(formatStatusText("connection_lost"), "接続切断・録音終了");
  assert.equal(formatStatusText("socket_error"), "接続エラー・録音終了");
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
