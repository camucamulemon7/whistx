import { normalizeProofreadMode } from "../ui/format.js";
import { readSseJsonStream } from "../api/sse.js";
import { isLoginRequiredError } from "../ui/format.js";
import { fetchJson } from "../api/client.js";

export function createAiController(appDependencies) {
function setProofreadButtonBusy(busy) {
  if (!appDependencies.proofreadBtn) return;
  appDependencies.proofreadBtn.disabled = false;
  if (appDependencies.proofreadBtnLabelEl) {
    appDependencies.proofreadBtnLabelEl.textContent = busy ? "キャンセル" : `${proofreadActionLabel()}する`;
  }
  appDependencies.proofreadBtn.setAttribute("aria-busy", busy ? "true" : "false");
  appDependencies.proofreadBtn.setAttribute("aria-label", busy ? `${proofreadActionLabel()}をキャンセル` : `${proofreadActionLabel()}を開始`);
}

function proofreadActionLabel() {
  const mode = normalizeProofreadMode(appDependencies.state.proofreadMode);
  if (mode === "translate_ja") return "日本語訳";
  if (mode === "translate_en") return "英語訳";
  return "校正";
}

function applyProofreadMode(value) {
  appDependencies.state.proofreadMode = normalizeProofreadMode(value);
  if (appDependencies.proofreadModeEl) {
    appDependencies.proofreadModeEl.value = appDependencies.state.proofreadMode;
  }
  if (appDependencies.proofreadBtn && !appDependencies.state.proofreadInFlight) {
    if (appDependencies.proofreadBtnLabelEl) {
      appDependencies.proofreadBtnLabelEl.textContent = `${proofreadActionLabel()}する`;
    }
    appDependencies.proofreadBtn.setAttribute("aria-label", `${proofreadActionLabel()}を開始`);
    appDependencies.proofreadBtn.title = proofreadActionLabel();
  }
}

function applySummaryPromptEditorOpen(value) {
  appDependencies.state.summaryPromptEditorOpen = !!value;
  if (appDependencies.summaryPromptEditorEl) {
    appDependencies.summaryPromptEditorEl.hidden = !appDependencies.state.summaryPromptEditorOpen;
    appDependencies.summaryPromptEditorEl.classList.toggle("is-open", appDependencies.state.summaryPromptEditorOpen);
  }
  if (appDependencies.summaryPromptToggleBtn) {
    appDependencies.summaryPromptToggleBtn.setAttribute("aria-expanded", appDependencies.state.summaryPromptEditorOpen ? "true" : "false");
    appDependencies.summaryPromptToggleBtn.textContent = appDependencies.state.summaryPromptEditorOpen ? "議事録の作成方針を閉じる" : "議事録の作成方針";
  }
}

function setSummary(text, meta) {
  appDependencies.state.summary = text || "";

  if (text) {
    appDependencies.summaryTextEl.innerHTML = "";
    appDependencies.summaryTextEl.textContent = text;
  } else {
    appDependencies.summaryTextEl.innerHTML = `
      <div class="empty-state small">
        <p class="empty-description">文字起こしから、議論の経緯・決定事項・次のアクションを画像付きの議事録として表示します。</p>
      </div>
    `;
  }

  appDependencies.summaryMetaEl.textContent = meta || "未実行";
}

async function copySummaryText() {
  if (!appDependencies.canUseWorkspace()) {
    appDependencies.showToast("ログインが必要です", "error");
    appDependencies.setAppLocked(true);
    appDependencies.loginEmailEl?.focus();
    return;
  }
  const text = String(appDependencies.state.summary || "").trim();
  if (!text) {
    appDependencies.showToast("要約がありません", "error");
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    appDependencies.showToast("要約をコピーしました", "success");
  } catch {
    appDependencies.showToast("要約のコピーに失敗しました", "error");
  }
}

function setProofread(text, meta) {
  appDependencies.state.proofread = text || "";

  if (text) {
    appDependencies.proofreadTextEl.innerHTML = "";
    appDependencies.proofreadTextEl.textContent = text;
  } else {
    appDependencies.proofreadTextEl.innerHTML = `
      <div class="empty-state small">
        <p class="empty-description">文字起こしができたら、ここで表記や文章を整えられます</p>
      </div>
    `;
  }

  appDependencies.proofreadMetaEl.textContent = meta || "未実行";
}

function markProofreadStale() {
  if (!appDependencies.state.proofread) return;
  appDependencies.proofreadMetaEl.textContent = "更新が必要";
}

async function copyProofread() {
  const text = appDependencies.state.proofread.trim();
  if (!text) {
    appDependencies.showToast("コピーする校正結果がありません", "error");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    appDependencies.copyProofreadBtn.classList.add("is-success");
    appDependencies.showToast("校正結果をコピーしました", "success");
    appDependencies.setStatus("proofread_copied");
    setTimeout(() => appDependencies.copyProofreadBtn.classList.remove("is-success"), 1200);
  } catch {
    appDependencies.showToast("コピーに失敗しました", "error");
    appDependencies.setStatus("copy_failed");
  }
}

async function proofreadAll() {
  if (appDependencies.state.proofreadInFlight) {
    appDependencies.state.proofreadController?.abort("user_cancelled");
    return;
  }
  if (!appDependencies.canUseWorkspace()) {
    appDependencies.showToast("ログインが必要です", "error");
    appDependencies.setAppLocked(true);
    appDependencies.loginEmailEl?.focus();
    return;
  }
  appDependencies.setStatus("proofread_requested");

  const text = appDependencies.extractTranscriptText();
  if (!text) {
    appDependencies.showToast("校正する文字起こしがありません", "error");
    appDependencies.setStatus("proofread_no_text");
    return;
  }

  if (!appDependencies.state.proofreadAvailable) {
    appDependencies.showToast("校正機能がサーバーで無効です", "error");
    setProofread("", "利用不可");
    appDependencies.setStatus("proofread_unavailable");
    return;
  }

  appDependencies.state.proofreadInFlight = true;
  setProofreadButtonBusy(true);
  appDependencies.proofreadMetaEl.textContent = "処理中...";
  appDependencies.setStatus("proofreading");
  appDependencies.showToast(`${proofreadActionLabel()}中...`, "default", 5000);
  const controller = new AbortController();
  appDependencies.state.proofreadController = controller;
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort("request_timeout");
  }, 300000);
  const proofreadStartedAt = performance.now();

  try {
    console.info("[whistx][api] request", { method: "POST", url: "/api/proofread/stream" });
    const response = await fetch("/api/proofread/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        language: appDependencies.selectedLanguage(),
        mode: appDependencies.state.proofreadMode,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      let payload = {};
      try {
        payload = await response.json();
      } catch {
        // ignore
      }
      const detail = payload.detail || payload.error || `http_${response.status}`;
      console.error("[whistx][api] error", {
        method: "POST",
        url: "/api/proofread/stream",
        status: response.status,
        durationMs: Math.round(performance.now() - proofreadStartedAt),
        detail: String(detail),
      });
      throw new Error(String(detail));
    }
    console.info("[whistx][api] response", {
      method: "POST",
      url: "/api/proofread/stream",
      status: response.status,
      durationMs: Math.round(performance.now() - proofreadStartedAt),
    });

    let correctedText = "";
    let modelName = "";
    let chunkCount = 0;
    let currentChunk = 0;
    let lastRenderAt = 0;

    await readSseJsonStream(response, (event) => {
      const eventType = String(event?.type || "");

      if (eventType === "start") {
        modelName = String(event.model || "");
        chunkCount = Number(event.chunkCount || 0);
        appDependencies.proofreadMetaEl.textContent = chunkCount > 1 ? `処理中... 0/${chunkCount}` : "処理中...";
        return;
      }

      if (eventType === "chunk_start") {
        currentChunk = Number(event.chunkIndex || currentChunk || 0);
        appDependencies.proofreadMetaEl.textContent = chunkCount > 1 ? `処理中... ${currentChunk}/${chunkCount}` : "処理中...";
        return;
      }

      if (eventType === "delta") {
        correctedText += String(event.delta || "");
        if (correctedText) appDependencies.markWorkspaceDirty();
        const now = Date.now();
        if (now - lastRenderAt >= 120 || correctedText.endsWith("\n")) {
          setProofread(correctedText, chunkCount > 1 ? `処理中... ${currentChunk}/${chunkCount}` : "処理中...");
          lastRenderAt = now;
        }
        return;
      }

      if (eventType === "final_text") {
        correctedText = String(event.text || "").trim();
        if (correctedText) appDependencies.markWorkspaceDirty();
        setProofread(correctedText, chunkCount > 1 ? `処理中... ${currentChunk}/${chunkCount}` : "処理中...");
        lastRenderAt = Date.now();
        return;
      }

      if (eventType === "error") {
        throw new Error(String(event.detail || event.message || "proofread_stream_failed"));
      }
    });

    correctedText = correctedText.trim();
    if (!correctedText) {
      throw new Error("empty_corrected");
    }

    const metaParts = [];
    if (modelName) {
      metaParts.push(`model: ${modelName}`);
    }
    if (chunkCount > 1) {
      metaParts.push(`chunks: ${chunkCount}`);
    }

    setProofread(correctedText, metaParts.join(" | ") || "完了");
    appDependencies.markWorkspaceDirty();
    appDependencies.setStatus("proofread_done");
    appDependencies.showToast(`${proofreadActionLabel()}が完了しました`, "success");
  } catch (err) {
    if (isLoginRequiredError(err)) {
      appDependencies.showToast("ログインが必要です", "error");
      appDependencies.setAppLocked(true);
      appDependencies.loginEmailEl?.focus();
      return;
    }
    const message = err?.name === "AbortError"
      ? timedOut
        ? "request_timeout"
        : "request_cancelled"
      : err?.message || "unknown_error";
    if (message === "request_cancelled") {
      appDependencies.showToast(`${proofreadActionLabel()}をキャンセルしました`, "default");
      appDependencies.proofreadMetaEl.textContent = "キャンセル";
      appDependencies.setStatus("proofread_cancelled");
    } else {
      appDependencies.showToast(
        message === "request_timeout"
          ? `${proofreadActionLabel()}がタイムアウトしました`
          : `${proofreadActionLabel()}に失敗: ${message}`,
        "error"
      );
      setProofread(`${proofreadActionLabel()}に失敗しました。\n${message}`, "エラー");
      appDependencies.setStatus(`proofread_failed: ${message}`);
    }
  } finally {
    clearTimeout(timeoutId);
    if (appDependencies.state.proofreadController === controller) {
      appDependencies.state.proofreadController = null;
    }
    appDependencies.state.proofreadInFlight = false;
    setProofreadButtonBusy(false);
  }
}

async function summarizeAll() {
  if (appDependencies.state.meetingInsights && appDependencies.currentMeetingSource()) {
    await appDependencies.meetingWorkspace.generateRecap();
    return;
  }
  if (appDependencies.state.summaryInFlight) {
    appDependencies.state.summaryController?.abort("user_cancelled");
    return;
  }
  if (!appDependencies.canUseWorkspace()) {
    appDependencies.showToast("ログインが必要です", "error");
    appDependencies.setAppLocked(true);
    appDependencies.loginEmailEl?.focus();
    return;
  }
  const text = appDependencies.extractTranscriptText();
  if (!text) {
    appDependencies.showToast("要約する文字起こしがありません", "error");
    appDependencies.setStatus("summary_no_text");
    return;
  }

  const controller = new AbortController();
  appDependencies.state.summaryController = controller;
  appDependencies.state.summaryInFlight = true;
  appDependencies.summaryBtn.disabled = false;
  appDependencies.summaryBtn.setAttribute("aria-busy", "true");
  appDependencies.summaryBtn.setAttribute("aria-label", "要約をキャンセル");
  if (appDependencies.summaryBtnLabelEl) {
    appDependencies.summaryBtnLabelEl.textContent = "キャンセル";
  }
  appDependencies.setStatus("summarizing");
  appDependencies.showToast("要約しています...", "default", 5000);

  try {
    const payload = await fetchJson("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        language: appDependencies.selectedLanguage(),
        prompt: String(appDependencies.summaryPromptEl?.value || "").trim(),
      }),
      signal: controller.signal,
      timeoutMs: 120000,
    });
    const summaryText = String(payload.summary || "").trim();
    if (!summaryText) {
      throw new Error("empty_summary");
    }

    const metaParts = [];
    if (payload.model) {
      metaParts.push(`model: ${payload.model}`);
    }
    if (payload.chunkCount > 1) {
      metaParts.push(`chunks: ${payload.chunkCount}`);
    }
    if (payload.reduced) {
      metaParts.push("統合済み");
    }

    setSummary(summaryText, metaParts.join(" | ") || "完了");
    appDependencies.markWorkspaceDirty();
    appDependencies.setStatus("summarized");
    appDependencies.showToast("要約が完了しました", "success");
  } catch (err) {
    if (isLoginRequiredError(err)) {
      appDependencies.showToast("ログインが必要です", "error");
      appDependencies.setAppLocked(true);
      appDependencies.loginEmailEl?.focus();
      return;
    }
    const message = err?.code === "timeout"
      ? "request_timeout"
      : err?.code === "aborted"
        ? "request_cancelled"
        : err?.code === "offline"
          ? "offline"
          : err?.message || "unknown_error";
    appDependencies.showToast(
      message === "request_cancelled"
        ? "要約をキャンセルしました"
        : message === "request_timeout"
          ? "要約がタイムアウトしました"
          : message === "offline"
            ? "オフラインのため要約できません"
            : `要約に失敗: ${message}`,
      message === "request_cancelled" ? "default" : "error"
    );
    appDependencies.setStatus(message === "request_cancelled" ? "summary_cancelled" : `summary_failed: ${message}`);
  } finally {
    if (appDependencies.state.summaryController === controller) {
      appDependencies.state.summaryController = null;
    }
    appDependencies.state.summaryInFlight = false;
    appDependencies.summaryBtn.disabled = false;
    appDependencies.summaryBtn.setAttribute("aria-busy", "false");
    appDependencies.summaryBtn.setAttribute("aria-label", "要約を開始");
    if (appDependencies.summaryBtnLabelEl) {
      appDependencies.summaryBtnLabelEl.textContent = "要約する";
    }
  }
}

  return { setProofreadButtonBusy, proofreadActionLabel, applyProofreadMode, applySummaryPromptEditorOpen, setSummary, copySummaryText, setProofread, markProofreadStale, copyProofread, proofreadAll, summarizeAll };
}
