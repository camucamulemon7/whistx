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

test("every application status distinguishes progress, success, failure, and availability", () => {
  const cases = [
    ["idle", "待機中"],
    ["ready", "録音準備完了"],
    ["recording_mic", "録音中"],
    ["display_capture_ended", "画面共有終了"],
    ["start_failed: permission denied", "開始失敗"],
    ["chunk_error: network", "音声送信失敗"],
    ["recorder_error", "録音処理失敗"],
    ["proofread_requested", "校正受付中"],
    ["proofreading", "校正処理中"],
    ["proofread_done", "校正完了"],
    ["proofread_cancelled", "校正キャンセル"],
    ["proofread_failed: timeout", "校正失敗"],
    ["proofread_unavailable", "校正利用不可"],
    ["proofread_no_text", "校正対象なし"],
    ["proofread_copied", "校正結果をコピー済み"],
    ["summarizing", "要約処理中"],
    ["summarized", "要約完了"],
    ["summary_cancelled", "要約キャンセル"],
    ["summary_failed: timeout", "要約失敗"],
    ["summary_no_text", "要約対象なし"],
    ["diarization_started", "話者分離処理中"],
    ["diarization_done", "話者分離完了"],
    ["diarization_failed", "話者分離失敗"],
    ["asr_unavailable", "文字起こし利用不可"],
    ["error: server_busy", "サーバー混雑"],
    ["error: transcription_failed", "文字起こし失敗"],
    ["error: guest_audio_limit", "利用上限に到達"],
    ["error: already_started", "録音は既に開始済み"],
    ["error: chunk_too_large", "音声データが大きすぎます"],
    ["error: diarization_failed", "話者分離失敗"],
    ["error: guest_asr_request_limit", "利用上限に到達"],
    ["error: invalid_chunk", "音声データが不正です"],
    ["error: invalid_chunk_offset", "音声位置情報が不正です"],
    ["error: invalid_chunk_sequence", "音声順序情報が不正です"],
    ["error: invalid_json", "通信データが不正です"],
    ["error: invalid_payload", "通信データが不正です"],
    ["error: not_started", "録音が開始されていません"],
    ["error: rate_limit_exceeded", "利用上限に到達"],
    ["error: session_create_failed", "セッション作成失敗"],
    ["error: unsupported_message", "未対応の通信です"],
    ["unknown_internal_code", "状態更新"],
  ];

  for (const [status, label] of cases) {
    assert.equal(formatStatusText(status), label, status);
  }
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
