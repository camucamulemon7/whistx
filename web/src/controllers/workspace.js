export function createWorkspaceController(appDependencies) {
function setSaveBadge(label, saved = false) {
  if (!appDependencies.saveStateBadgeEl) return;
  appDependencies.saveStateBadgeEl.textContent = label;
  appDependencies.saveStateBadgeEl.classList.toggle("is-saved", !!saved);
}

function isRecordingInteractionLocked() {
  return appDependencies.state.recordingPhase !== "idle" || !!appDependencies.state.finalizingStop || Boolean(appDependencies.refinementController);
}

function shouldProtectWorkspaceFromUnload() {
  return appDependencies.state.workspaceDirty || isRecordingInteractionLocked() || Boolean(appDependencies.state.liveCapture?.pending.size);
}

function handleBeforeUnload(event) {
  if (!shouldProtectWorkspaceFromUnload()) return;
  event.preventDefault();
  event.returnValue = "";
}

function syncUnloadProtection() {
  const shouldProtect = shouldProtectWorkspaceFromUnload();
  if (shouldProtect && !appDependencies.state.unloadProtectionRegistered) {
    window.addEventListener("beforeunload", handleBeforeUnload);
    appDependencies.state.unloadProtectionRegistered = true;
  } else if (!shouldProtect && appDependencies.state.unloadProtectionRegistered) {
    window.removeEventListener("beforeunload", handleBeforeUnload);
    appDependencies.state.unloadProtectionRegistered = false;
  }
  document.documentElement.dataset.unsavedTranscript = shouldProtect ? "true" : "false";
}

function markWorkspaceDirty() {
  appDependencies.state.workspaceDirty = true;
  syncUnloadProtection();
}

function markWorkspaceClean() {
  appDependencies.state.workspaceDirty = false;
  syncUnloadProtection();
}

function confirmWorkspaceDiscard(action) {
  if (!appDependencies.state.workspaceDirty) return true;
  return window.confirm(`保存されていない文字起こしや生成結果があります。${action}してよいですか？`);
}

function showRecordingInteractionBlocked(action) {
  const finalizing = appDependencies.state.recordingPhase === "stopping" || appDependencies.state.recordingPhase === "finalizing" || appDependencies.state.finalizingStop;
  appDependencies.showToast(finalizing ? `録音の停止処理中は${action}できません` : `録音中は${action}できません`, "error");
}

function updateSaveControls() {
  const hasSegments = appDependencies.state.segments.length > 0;
  const authenticated = !!appDependencies.state.auth.authenticated;
  const isGuest = !!appDependencies.state.auth.isGuest;
  const saved = !!appDependencies.state.savedHistoryId;
  const viewingHistory = !!appDependencies.state.viewingHistoryId;
  const recordingLocked = isRecordingInteractionLocked();
  const refineButton = document.querySelector("#refineAudioBtn");
  if (refineButton) {
    refineButton.disabled = !appDependencies.refinementController && (!authenticated || isGuest || recordingLocked || !hasSegments || !appDependencies.state.runtimeSessionFinalized || saved || viewingHistory || Boolean(appDependencies.meetingWorkspace?.busy));
    refineButton.textContent = appDependencies.refinementController ? "再認識をキャンセル" : "音声から再認識";
  }

  if (appDependencies.saveBtn) {
    appDependencies.saveBtn.disabled =
      !authenticated || isGuest || recordingLocked || !hasSegments || appDependencies.state.saveInFlight || saved || viewingHistory || Boolean(appDependencies.meetingWorkspace?.busy) || Boolean(appDependencies.state.liveCapture && !appDependencies.state.runtimeSessionFinalized);
    appDependencies.saveBtn.textContent = appDependencies.state.saveInFlight ? "保存中..." : "保存";
    appDependencies.saveBtn.title = recordingLocked
      ? "録音中は保存できません"
      : isGuest
        ? "ゲストでは保存できません"
        : authenticated
          ? ""
          : "ログインが必要です";
  }
  if (appDependencies.saveTitleInputEl) {
    appDependencies.saveTitleInputEl.disabled = viewingHistory || appDependencies.state.saveInFlight || recordingLocked;
  }
  if (appDependencies.clearBtn) {
    appDependencies.clearBtn.disabled = appDependencies.runtimeUi.appLocked || recordingLocked;
    appDependencies.clearBtn.title = recordingLocked ? "録音中・停止処理中はクリアできません" : "クリア";
  }
  setSaveBadge(
    saved ? "保存済み" : recordingLocked ? "録音中は保存不可" : isGuest ? "ゲストでは保存不可" : authenticated ? "未保存" : "ログインが必要",
    saved
  );
}

function setSessionSettingsLocked(locked = isRecordingInteractionLocked()) {
  const sessionInputs = [
    document.querySelector("#liveTranscriptionEnabled"),
    appDependencies.languageEl,
    appDependencies.audioSourceEl,
    appDependencies.chunkSecondsEl,
    appDependencies.promptEl,
    appDependencies.sharedVocabularyEl,
  ];
  sessionInputs.forEach((element) => {
    if (!element) return;
    const qwenFixed = appDependencies.state.asrBackend === "qwen3_vllm" && element.id === "liveTranscriptionEnabled";
    element.disabled = !!locked || qwenFixed;
    element.title = qwenFixed ? "日本語・英語を自動認識するライブ文字起こしを使用します" : locked ? "録音中・停止処理中は変更できません" : "";
  });
  appDependencies.presetButtons.forEach((button) => {
    button.disabled = !!locked;
    button.title = locked ? "録音中・停止処理中は変更できません" : "";
  });
  appDependencies.promptTemplateButtonsEl?.querySelectorAll("button").forEach((button) => {
    button.disabled = !!locked;
    button.title = locked ? "録音中・停止処理中は変更できません" : "";
  });
  if (appDependencies.diarizationToggleEl) {
    appDependencies.diarizationToggleEl.disabled = !!locked || !appDependencies.state.diarizationAvailable;
    appDependencies.diarizationToggleEl.title = locked
      ? "録音中・停止処理中は変更できません"
      : appDependencies.state.diarizationAvailable
        ? "話者分離を有効/無効"
        : "サーバーで話者分離は無効";
  }
  document.body.classList.toggle("session-settings-locked", !!locked);
  appDependencies.updateDiarizationSpeakerUi();
  appDependencies.updateSharedVocabularyMeta();
  appDependencies.meetingWorkspace?.syncSource();
}

function hasDiscardableWorkspaceData() {
  return (
    appDependencies.state.segments.length > 0 ||
    String(appDependencies.state.summary || "").trim().length > 0 ||
    String(appDependencies.state.proofread || "").trim().length > 0
  );
}

function clearView(options = {}) {
  if (!appDependencies.canUseWorkspace()) {
    appDependencies.showToast("ログインが必要です", "error");
    appDependencies.setAppLocked(true);
    appDependencies.loginEmailEl?.focus();
    return;
  }
  if (isRecordingInteractionLocked()) {
    showRecordingInteractionBlocked("クリア");
    return;
  }
  if (
    !options.skipConfirmation &&
    hasDiscardableWorkspaceData() &&
    !window.confirm("表示中の文字起こしや生成結果が消えます。クリアしてよいですか？")
  ) {
    return;
  }
  appDependencies.state.historyDetailRequestVersion += 1;
  appDependencies.state.historyDetailController?.abort("workspace_cleared");
  appDependencies.state.historyDetailController = null;
  appDependencies.state.log = [];
  appDependencies.state.segments = [];
  appDependencies.state.logAutoScrollEnabled = true;
  appDependencies.state.history.selectedId = null;
  appDependencies.state.viewingHistoryId = null;
  appDependencies.state.savedHistoryId = null;
  appDependencies.resetRuntimeSessionState();
  markWorkspaceClean();

  appDependencies.renderEmptyTranscriptState();

  appDependencies.updateSegmentCount();
  appDependencies.setSummary("", "未実行");
  appDependencies.setProofread("", "未実行");
  appDependencies.updateDownloadLinks();
  updateSaveControls();
  appDependencies.renderHistoryList();
  appDependencies.showToast("クリアしました", "success");
}

  return { setSaveBadge, isRecordingInteractionLocked, shouldProtectWorkspaceFromUnload, handleBeforeUnload, syncUnloadProtection, markWorkspaceDirty, markWorkspaceClean, confirmWorkspaceDiscard, showRecordingInteractionBlocked, updateSaveControls, setSessionSettingsLocked, hasDiscardableWorkspaceData, clearView };
}
