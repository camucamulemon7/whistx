export function formatStatusText(text) {
  const raw = String(text || "").trim();
  const normalized = raw.toLowerCase();
  if (!raw || normalized === "idle") return "待機中";
  if (normalized.startsWith("recording")) return "録音中";
  if (normalized === "stopping") return "停止処理中";
  if (normalized.startsWith("start_failed")) return "開始失敗";
  if (normalized.includes("proofread")) return normalized.includes("error") ? "校正失敗" : "校正処理";
  if (normalized.includes("summary")) return normalized.includes("error") ? "要約失敗" : "要約処理";
  if (normalized === "copied") return "コピー済み";
  if (normalized === "copy_failed") return "コピー失敗";
  if (normalized.includes("unavailable")) return "利用不可";
  if (normalized.includes("error")) return "エラー";
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
