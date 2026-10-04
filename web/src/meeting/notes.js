import { fetchJson } from "../api/client.js";

export function notesErrorMessage(error) {
  const messages = {
    notes_auth_required: "OpenWebUIの認証またはNotes利用権限を確認してください。",
    notes_account_mismatch: "Whistxと同じメールアドレスのOpenWebUI認証を使用してください。",
    openwebui_not_configured: "管理者設定でOpenWebUI URLを設定してください。",
    notes_current_recap_required: "現在の会議の議事録を作成してから保存してください。",
    notes_save_uncertain: "保存結果を確認できません。確認・再試行で保存済みNoteを照合できます。結果不明の間は追加送信しません。",
    notes_save_rejected: "OpenWebUIが保存を拒否しました。設定を確認して再試行できます。",
  };
  if (error.code === "timeout" || error.code === "network") return messages.notes_save_uncertain;
  if (error.status === 401 && error.message !== "notes_auth_required") return "Whistxにログインしてください。";
  return messages[error.message] || "Notes保存に失敗しました。確認・再試行できます。既存のNoteは保持されています。";
}

export function createNotesExport({ getSource, getAccess, request = fetchJson, document = globalThis.document }) {
  const token = document.querySelector("#notesToken");
  const title = document.querySelector("#notesTitle");
  const button = document.querySelector("#notesSave");
  const status = document.querySelector("#notesStatus");
  let inFlight = false;
  let version = 0;

  function sync() {
    button.disabled = inFlight || getAccess() !== "ready" || !getSource();
    button.setAttribute("aria-busy", String(inFlight));
    button.textContent = inFlight ? "保存結果を確認中…" : "Notesに保存 / 確認・再試行";
    if (getAccess() !== "ready") token.value = "";
  }
  function reset() {
    version += 1;
    token.value = "";
    status.textContent = "";
    sync();
  }
  async function save() {
    if (inFlight || getAccess() !== "ready" || !getSource()) return;
    const credential = token.value.trim();
    if (!credential) { status.textContent = "自分の既存OpenWebUI認証を入力してください。"; token.focus(); return; }
    const current = version;
    inFlight = true;
    sync();
    status.textContent = "OpenWebUIに議事録を保存しています…";
    try {
      const result = await request("/api/meeting/notes", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...getSource(), title: title.value.trim(), token: credential }), timeoutMs: 90_000 });
      if (current !== version) return;
      status.textContent = result.alreadySaved ? "この議事録は既にNotesに保存されています。" : "自分の非公開Noteに議事録を保存しました。";
    } catch (error) {
      if (current === version) status.textContent = notesErrorMessage(error);
    } finally {
      token.value = "";
      inFlight = false;
      sync();
    }
  }
  button.addEventListener("click", save);
  window.addEventListener("pagehide", reset);
  sync();
  return { sync, reset, save };
}
