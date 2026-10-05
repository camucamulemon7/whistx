import { formatLanguageLabel } from "../ui/format.js";
import { escapeHtml } from "../ui/format.js";
import { clearHistoryState } from "../history/state.js";
import { fetchHistoryList as fetchHistoryListRequest } from "../history/api.js";
import { applyHistoryListPayload } from "../history/state.js";
import { applyHistoryDetailPayload } from "../history/state.js";
import { fetchHistoryDetail } from "../history/api.js";
import { deleteHistoryRequest } from "../history/api.js";
import { saveHistoryRequest } from "../history/api.js";

export function createHistoryController(appDependencies) {
function setHistorySearchQuery(value) {
  appDependencies.state.history.query = String(value || "").trim();
  appDependencies.state.history.offset = 0;
  if (appDependencies.state.historySearchTimer) {
    clearTimeout(appDependencies.state.historySearchTimer);
    appDependencies.state.historySearchTimer = null;
  }
  appDependencies.state.historySearchTimer = setTimeout(() => {
    appDependencies.state.historySearchTimer = null;
    loadHistoryList();
  }, appDependencies.HISTORY_SEARCH_DEBOUNCE_MS);
}

function formatHistoryMeta(item) {
  const parts = [];
  if (item.savedAt) parts.push(new Date(item.savedAt).toLocaleString("ja-JP"));
  if (item.language) parts.push(formatLanguageLabel(item.language));
  return parts.join(" / ");
}

function formatHistoryDaysRemaining(item) {
  const retentionDays = Math.max(0, Number(appDependencies.state.auth.historyRetentionDays ?? 0));
  if (!retentionDays) return "無期限";
  if (!item?.savedAt) return `${retentionDays}日で削除`;
  const savedAtMs = new Date(item.savedAt).getTime();
  if (!Number.isFinite(savedAtMs)) return `${retentionDays}日で削除`;
  const expiresAtMs = savedAtMs + retentionDays * 24 * 60 * 60 * 1000;
  const remainingMs = expiresAtMs - Date.now();
  if (remainingMs <= 0) return "まもなく削除";
  return `あと${Math.ceil(remainingMs / (24 * 60 * 60 * 1000))}日`;
}

function updateHistoryEmptyState(message = "") {
  if (!appDependencies.historyEmptyEl) return;
  if (message) {
    appDependencies.historyEmptyEl.hidden = false;
    appDependencies.historyEmptyEl.textContent = message;
    return;
  }
  if (appDependencies.state.auth.isGuest) {
    appDependencies.historyEmptyEl.hidden = false;
    appDependencies.historyEmptyEl.textContent = "ゲストでは履歴は利用できません";
    return;
  }
  if (!appDependencies.state.auth.authenticated) {
    appDependencies.historyEmptyEl.hidden = false;
    appDependencies.historyEmptyEl.textContent = appDependencies.state.auth.bootstrapAdminRequired
      ? "初回管理者アカウントを作成すると履歴がここに表示されます"
      : "ログインすると保存済み履歴がここに表示されます";
    return;
  }
  appDependencies.historyEmptyEl.hidden = appDependencies.state.history.items.length > 0;
  appDependencies.historyEmptyEl.textContent = "保存済み履歴はまだありません";
}

function renderHistoryList() {
  if (!appDependencies.historyListEl) return;
  appDependencies.historyListEl.innerHTML = "";
  const historyCount = appDependencies.state.history.total || appDependencies.state.history.items.length || 0;
  if (appDependencies.historyCountBadgeEl) {
    appDependencies.historyCountBadgeEl.textContent = historyCount + "件";
  }
  appDependencies.state.history.items.forEach((item) => {
    const article = document.createElement("article");
    article.className = "history-item";
    if (appDependencies.state.history.selectedId === item.id) {
      article.classList.add("is-active");
    }
    const savedAt = item.savedAt
      ? new Date(item.savedAt).toLocaleString("ja-JP", {
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
        })
      : "日時不明";
    const language = formatLanguageLabel(item.language || "auto");
    const daysRemaining = formatHistoryDaysRemaining(item);
    article.innerHTML = `
      <div class="history-item-main" role="button" tabindex="0" aria-label="履歴を開く">
        <div class="history-item-header">
          <div class="history-item-title">${escapeHtml(item.title || "無題")}</div>
        </div>
        <div class="history-item-meta-row">
          <span class="history-item-badge">${escapeHtml(language)}</span>
          <span class="history-item-badge">${escapeHtml(daysRemaining)}</span>
        </div>
        <div class="history-item-meta">${escapeHtml(formatHistoryMeta(item))}</div>
        <div class="history-item-preview">${escapeHtml(item.preview || "")}</div>
      </div>
      <div class="history-item-actions">
        <button type="button" class="history-item-delete" aria-label="履歴を削除">削除</button>
      </div>
    `;
    const historyLocked = appDependencies.isRecordingInteractionLocked();
    const mainAction = article.querySelector(".history-item-main");
    const deleteAction = article.querySelector(".history-item-delete");
    mainAction?.setAttribute("aria-disabled", String(historyLocked));
    if (deleteAction) {
      deleteAction.disabled = historyLocked;
    }
    article.classList.toggle("is-disabled", historyLocked);
    mainAction?.addEventListener("click", () => {
      openHistoryDetail(item.id);
    });
    mainAction?.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openHistoryDetail(item.id);
      }
    });
    deleteAction?.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      deleteHistory(item.id);
    });
    appDependencies.historyListEl.appendChild(article);
  });
  updateHistoryEmptyState();
}

async function loadHistoryList() {
  if (!appDependencies.state.auth.authenticated) {
    clearHistoryState(appDependencies.state);
    renderHistoryList();
    appDependencies.logClientEvent("history.load.skipped");
    return;
  }

  appDependencies.logClientEvent("history.load.start", { query: appDependencies.state.history.query, offset: appDependencies.state.history.offset, limit: appDependencies.state.history.limit });
  appDependencies.state.historyListController?.abort("history_request_replaced");
  const controller = new AbortController();
  appDependencies.state.historyListController = controller;
  try {
    const payload = await fetchHistoryListRequest({
      limit: appDependencies.state.history.limit,
      offset: appDependencies.state.history.offset,
      query: appDependencies.state.history.query,
      signal: controller.signal,
    });
    applyHistoryListPayload(appDependencies.state, payload);
    renderHistoryList();
    appDependencies.logClientEvent("history.load.success", { total: appDependencies.state.history.total, items: appDependencies.state.history.items.length });
  } catch (error) {
    if (error?.code === "aborted") return;
    updateHistoryEmptyState("履歴の取得に失敗しました");
    appDependencies.logClientEvent("history.load.failed");
  } finally {
    if (appDependencies.state.historyListController === controller) {
      appDependencies.state.historyListController = null;
    }
  }
}

function renderHistoryDetail(payload) {
  applyHistoryDetailPayload(appDependencies.state, payload);
  appDependencies.renderEmptyTranscriptState();
  appDependencies.logEl.innerHTML = "";
  appDependencies.state.logAutoScrollEnabled = true;
  (payload.segments || []).forEach((segment) => {
    appDependencies.addLogLine(
      String(segment.text || ""),
      Number(segment.tsStart || 0),
      Number(segment.tsEnd || 0),
      Number(segment.seq),
      String(segment.speaker || ""),
      String(segment.screenshotUrl || ""),
      String(segment.rawAudioUrl || segment.rawAudioPath || ""),
      String(segment.audioUrl || segment.audioPath || ""),
      String(segment.segmentId || "")
    );
  });
  if (!payload.segments || payload.segments.length === 0) {
    appDependencies.renderEmptyTranscriptState();
  }
  appDependencies.setSummary(payload.summaryText || "", payload.savedAt ? `保存: ${new Date(payload.savedAt).toLocaleString("ja-JP")}` : "履歴");
  appDependencies.setProofread(payload.proofreadText || "", payload.savedAt ? `保存: ${new Date(payload.savedAt).toLocaleString("ja-JP")}` : "履歴");
  if (appDependencies.saveTitleInputEl) {
    appDependencies.saveTitleInputEl.value = String(payload.title || "");
  }
  appDependencies.markWorkspaceClean();
  appDependencies.updateDownloadLinks();
  appDependencies.updateSaveControls();
  renderHistoryList();
}

async function openHistoryDetail(historyId) {
  if (!appDependencies.state.auth.authenticated) {
    appDependencies.showToast("ログインが必要です", "error");
    appDependencies.setAppLocked(true);
    appDependencies.loginEmailEl?.focus();
    return;
  }
  if (appDependencies.isRecordingInteractionLocked()) {
    appDependencies.showRecordingInteractionBlocked("履歴を開くことが");
    return;
  }
  if (appDependencies.state.viewingHistoryId !== historyId && !appDependencies.confirmWorkspaceDiscard("破棄して別の履歴を表示")) {
    return;
  }
  const requestVersion = appDependencies.state.historyDetailRequestVersion + 1;
  appDependencies.state.historyDetailRequestVersion = requestVersion;
  const runtimeSessionIdAtRequest = appDependencies.state.runtimeSessionId;
  appDependencies.state.historyDetailController?.abort("history_request_replaced");
  const controller = new AbortController();
  appDependencies.state.historyDetailController = controller;
  try {
    const payload = await fetchHistoryDetail(historyId, { signal: controller.signal });
    if (
      requestVersion !== appDependencies.state.historyDetailRequestVersion ||
      appDependencies.isRecordingInteractionLocked() ||
      appDependencies.state.runtimeSessionId !== runtimeSessionIdAtRequest
    ) {
      return;
    }
    renderHistoryDetail(payload);
    if (window.innerWidth <= 1100) {
      appDependencies.applyHistoryDrawerOpen(false);
    }
  } catch (error) {
    if (error?.code === "aborted") return;
    appDependencies.showToast("履歴の取得に失敗しました", "error");
  } finally {
    if (appDependencies.state.historyDetailController === controller) {
      appDependencies.state.historyDetailController = null;
    }
  }
}

async function deleteHistory(historyId) {
  if (!historyId || !appDependencies.state.auth.authenticated) return;
  if (appDependencies.isRecordingInteractionLocked()) {
    appDependencies.showRecordingInteractionBlocked("履歴を削除することが");
    return;
  }
  const target = appDependencies.state.history.items.find((item) => item.id === historyId);
  const confirmed = window.confirm(`履歴「${target?.title || historyId}」を削除しますか？`);
  if (!confirmed) return;

  try {
    await deleteHistoryRequest(historyId);
  } catch {
    appDependencies.showToast("履歴の削除に失敗しました", "error");
    return;
  }

  if (appDependencies.state.viewingHistoryId === historyId || appDependencies.state.savedHistoryId === historyId || appDependencies.state.history.selectedId === historyId) {
    appDependencies.clearView({ skipConfirmation: true });
  }
  await loadHistoryList();
  appDependencies.showToast("履歴を削除しました", "success");
}

async function saveCurrentHistory() {
  if (appDependencies.state.auth.isGuest) {
    appDependencies.showToast("ゲストでは履歴保存できません", "error");
    return;
  }
  if (!appDependencies.state.auth.authenticated) {
    appDependencies.showToast("ログインが必要です", "error");
    appDependencies.loginEmailEl?.focus();
    return;
  }
  if (appDependencies.state.recording || appDependencies.state.finalizingStop) {
    appDependencies.showToast("録音中は保存できません", "error");
    return;
  }
  if (!appDependencies.state.runtimeSessionId || appDependencies.state.segments.length === 0) {
    appDependencies.showToast("保存できる文字起こしがありません", "error");
    return;
  }

  appDependencies.state.saveInFlight = true;
  appDependencies.updateSaveControls();
  let payload;
  try {
    payload = await saveHistoryRequest({
      runtimeSessionId: appDependencies.state.runtimeSessionId,
      runtimeSessionToken: appDependencies.state.runtimeSessionToken,
      title: appDependencies.buildAutoSaveTitle(),
      summaryText: appDependencies.state.summary || null,
      proofreadText: appDependencies.state.proofread || null,
    });
  } catch (error) {
    appDependencies.state.saveInFlight = false;
    appDependencies.updateSaveControls();
    const responsePayload = error?.payload || {};
    appDependencies.showToast(
      responsePayload.error === "runtime_session_not_finalized"
        ? "録音停止後に保存してください"
        : responsePayload.error === "history_already_saved"
          ? "このセッションは既に保存済みです"
          : "保存に失敗しました",
      "error"
    );
    return;
  }
  appDependencies.state.saveInFlight = false;
  appDependencies.updateSaveControls();
  appDependencies.state.savedHistoryId = payload.history?.id || null;
  appDependencies.state.viewingHistoryId = appDependencies.state.savedHistoryId;
  appDependencies.state.history.selectedId = appDependencies.state.savedHistoryId;
  appDependencies.markWorkspaceClean();
  appDependencies.updateDownloadLinks();
  appDependencies.updateSaveControls();
  await loadHistoryList();
  appDependencies.showToast("保存しました", "success");
}

  return { setHistorySearchQuery, formatHistoryMeta, formatHistoryDaysRemaining, updateHistoryEmptyState, renderHistoryList, loadHistoryList, renderHistoryDetail, openHistoryDetail, deleteHistory, saveCurrentHistory };
}
