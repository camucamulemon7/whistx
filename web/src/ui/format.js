export function formatStatusText(text) {
  const raw = String(text || "").trim();
  const normalized = raw.toLowerCase();
  const code = normalized.split(":", 1)[0].trim();
  const labels = {
    idle: "待機中",
    starting: "録音準備中",
    stopping: "停止処理中",
    finalizing: "最終処理中",
    finalized: "録音完了",
    completed: "録音完了",
    ready: "録音準備完了",
    display_capture_ended: "画面共有終了",
    connection_lost: "接続切断・録音終了",
    disconnected: "未接続",
    socket_error: "接続エラー・録音終了",
    finalize_failed: "最終処理失敗",
    recorder_error: "録音処理失敗",
    chunk_error: "音声送信失敗",
    copied: "コピー済み",
    copy_failed: "コピー失敗",
    proofread_copied: "校正結果をコピー済み",
    proofread_requested: "校正受付中",
    proofread_no_text: "校正対象なし",
    proofread_unavailable: "校正利用不可",
    proofreading: "校正処理中",
    proofread_done: "校正完了",
    proofread_error: "校正失敗",
    proofread_failed: "校正失敗",
    summary_no_text: "要約対象なし",
    summarizing: "要約処理中",
    summarized: "要約完了",
    summary_error: "要約失敗",
    summary_failed: "要約失敗",
    asr_unavailable: "文字起こし利用不可",
    diarization_started: "話者分離処理中",
    diarization_done: "話者分離完了",
    diarization_failed: "話者分離失敗",
  };
  if (!raw) return labels.idle;
  if (labels[code]) return labels[code];
  if (code === "recording" || code.startsWith("recording_")) return "録音中";
  if (code === "start_failed") return "開始失敗";
  if (code === "error") {
    const errorCode = normalized.match(/^error:\s*([a-z0-9_]+)/)?.[1] || "";
    const errorLabels = {
      already_started: "録音は既に開始済み",
      chunk_too_large: "音声データが大きすぎます",
      diarization_failed: "話者分離失敗",
      guest_asr_request_limit: "利用上限に到達",
      guest_audio_limit: "利用上限に到達",
      invalid_chunk: "音声データが不正です",
      invalid_chunk_offset: "音声位置情報が不正です",
      invalid_chunk_sequence: "音声順序情報が不正です",
      invalid_json: "通信データが不正です",
      invalid_payload: "通信データが不正です",
      not_started: "録音が開始されていません",
      rate_limit_exceeded: "利用上限に到達",
      server_busy: "サーバー混雑",
      session_create_failed: "セッション作成失敗",
      transcription_failed: "文字起こし失敗",
      unsupported_message: "未対応の通信です",
    };
    return errorLabels[errorCode] || "エラー";
  }
  if (/^[a-z0-9_:\-./ ()]+$/i.test(raw)) return "状態更新";
  return raw;
}

export function formatAudioSource(mode) {
  if (mode === "both") return "両方";
  if (mode === "display") return "画面共有";
  return "マイク";
}

export function formatLanguageLabel(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "ja") return "日本語";
  if (normalized === "en") return "英語";
  if (!normalized || normalized === "auto") return "自動";
  return String(value);
}

export function formatTimestamp(ms) {
  const totalSec = Math.floor(Math.max(0, Number(ms) || 0) / 1000);
  const min = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const sec = String(totalSec % 60).padStart(2, "0");
  return `${min}:${sec}`;
}

export function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function normalizeBannerType(value) {
  const type = String(value || "info").toLowerCase();
  if (type === "success" || type === "warning" || type === "error") return type;
  return "info";
}

export function normalizeProofreadMode(value) {
  if (value === "translate_ja" || value === "translate_en") return value;
  return "proofread";
}

export function normalizeSpeakerMode(value) {
  if (value === "fixed" || value === "range") return value;
  return "auto";
}

export function clampSpeakerCount(value, cap = 12, minimum = 1) {
  const safeMinimum = Math.max(1, Number(minimum) || 1);
  const safeCap = Math.max(safeMinimum, Number(cap) || 12);
  const num = Number(value);
  if (!Number.isFinite(num)) return safeMinimum;
  return Math.max(safeMinimum, Math.min(safeCap, Math.round(num)));
}

export function isLoginRequiredError(error) {
  return error?.status === 401 || error?.message === "login_required" || error?.payload?.detail === "login_required";
}
