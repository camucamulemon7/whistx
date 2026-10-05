import { fetchSharedGlossary as fetchSharedGlossaryRequest } from "../api/glossary.js";
import { saveSharedGlossary as saveSharedGlossaryRequest } from "../api/glossary.js";

export function createGlossaryController(appDependencies) {
function renderPromptTemplateButtons(rawTemplates) {
  if (!appDependencies.promptTemplateButtonsEl || !appDependencies.promptEl) return;
  const templates = Array.isArray(rawTemplates) && rawTemplates.length > 0
    ? rawTemplates
    : appDependencies.state.promptTemplates;

  appDependencies.promptTemplateButtonsEl.innerHTML = "";
  templates.forEach((template, index) => {
    const label = String(template?.label || `Template ${index + 1}`).trim();
    const content = String(template?.content || "").trim();
    if (!content) return;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "prompt-template-btn";
    button.textContent = label;
    button.disabled = appDependencies.isRecordingInteractionLocked();
    button.title = button.disabled ? "録音中・停止処理中は変更できません" : "";
    button.addEventListener("click", () => {
      if (appDependencies.isRecordingInteractionLocked()) {
        appDependencies.showRecordingInteractionBlocked("プロンプトを変更することが");
        return;
      }
      appDependencies.promptEl.value = content;
      appDependencies.promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      appDependencies.promptEl.focus();
      appDependencies.showToast(`${label}を入力しました`, "success");
    });
    appDependencies.promptTemplateButtonsEl.appendChild(button);
  });
}

function applySharedVocabulary(payload, options = {}) {
  const preserveDraft = options.preserveDraft === true;
  const text = String(payload?.items || "").trim();
  appDependencies.state.sharedVocabulary = text;
  appDependencies.state.sharedVocabularyUpdatedAt = String(payload?.updatedAt || "").trim();
  appDependencies.state.sharedVocabularyUpdatedBy = String(payload?.updatedBy || "").trim();
  if (appDependencies.sharedVocabularyEl && !preserveDraft) {
    appDependencies.sharedVocabularyEl.value = text;
  }
  updateSharedVocabularyMeta();
  appDependencies.meetingWorkspace?.syncSource();
}

function updateSharedVocabularyMeta() {
  const authenticated = !!appDependencies.state.auth.authenticated;
  const isGuest = !!appDependencies.state.auth.isGuest;
  const recordingLocked = appDependencies.isRecordingInteractionLocked();
  if (appDependencies.sharedVocabularySaveBtn) {
    appDependencies.sharedVocabularySaveBtn.disabled = !authenticated || isGuest || recordingLocked || appDependencies.state.sharedVocabularySaving;
    appDependencies.sharedVocabularySaveBtn.textContent = appDependencies.state.sharedVocabularySaving ? "保存中..." : "全体に保存";
    appDependencies.sharedVocabularySaveBtn.title = recordingLocked
      ? "録音中・停止処理中は変更できません"
      : isGuest
        ? "ゲストでは全体用語辞典を更新できません"
        : authenticated
          ? ""
          : "ログインが必要です";
  }
  if (!appDependencies.sharedVocabularyMetaEl) return;
  if (!appDependencies.state.sharedVocabulary) {
    appDependencies.sharedVocabularyMetaEl.textContent = "全体用語辞典は未設定です";
    return;
  }
  const parts = [];
  if (appDependencies.state.sharedVocabularyUpdatedAt) {
    parts.push(`更新: ${new Date(appDependencies.state.sharedVocabularyUpdatedAt).toLocaleString("ja-JP")}`);
  }
  if (appDependencies.state.sharedVocabularyUpdatedBy) {
    parts.push(`更新者: ${appDependencies.state.sharedVocabularyUpdatedBy}`);
  }
  appDependencies.sharedVocabularyMetaEl.textContent = parts.join(" / ") || "全体用語辞典を使用します";
}

async function loadSharedGlossary() {
  appDependencies.logClientEvent("shared_glossary.load.start");
  try {
    const payload = await fetchSharedGlossaryRequest();
    applySharedVocabulary(payload);
    appDependencies.logClientEvent("shared_glossary.load.success", { hasItems: !!String(payload?.items || "").trim() });
  } catch {
    applySharedVocabulary({ items: "", updatedAt: "", updatedBy: "" });
    appDependencies.logClientEvent("shared_glossary.load.fallback");
  }
}

async function saveSharedGlossary() {
  if (appDependencies.isRecordingInteractionLocked()) {
    appDependencies.showRecordingInteractionBlocked("全体用語辞典を変更することが");
    return;
  }
  if (appDependencies.state.auth.isGuest) {
    appDependencies.showToast("ゲストでは全体用語辞典を更新できません", "error");
    return;
  }
  if (!appDependencies.state.auth.authenticated) {
    appDependencies.showToast("ログインが必要です", "error");
    appDependencies.loginEmailEl?.focus();
    return;
  }
  appDependencies.state.sharedVocabularySaving = true;
  updateSharedVocabularyMeta();
  try {
    const payload = await saveSharedGlossaryRequest(String(appDependencies.sharedVocabularyEl?.value || "").trim());
    applySharedVocabulary(payload);
    appDependencies.showToast("全体用語辞典を保存しました", "success");
  } catch (error) {
    appDependencies.showToast(`全体用語辞典の保存に失敗: ${error.message}`, "error");
  } finally {
    appDependencies.state.sharedVocabularySaving = false;
    updateSharedVocabularyMeta();
  }
}

  return { renderPromptTemplateButtons, applySharedVocabulary, updateSharedVocabularyMeta, loadSharedGlossary, saveSharedGlossary };
}
