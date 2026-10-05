import { fetchCapabilities } from "../capabilities/api.js";
import { normalizeWsPath } from "../transcription/websocket.js";

export function createCapabilitiesController(appDependencies) {
let capabilitiesGeneration = 0;
async function loadCapabilities() {
  const generation = ++capabilitiesGeneration;
  appDependencies.logClientEvent("capabilities.load.start");
  try {
    const health = await fetchCapabilities();
    if (generation !== capabilitiesGeneration) return;
    appDependencies.state.wsPath = normalizeWsPath(health.wsPath || appDependencies.state.wsPath);
    appDependencies.state.meetingInsights = !!health.meetingInsights;
    appDependencies.state.asrBackend = health.asrBackend || "whisper";
    document.querySelector('#hqIntervalHint').textContent = health.highAccuracyWindowSeconds
      ? `高精度認識は通常${health.highAccuracyWindowSeconds}秒分ごと（発話の区切りでは早めに実行）`
      : '翻訳は高精度認識の確定後、または録音終了後に実行';
    if (appDependencies.state.asrBackend === "qwen3_vllm") {
      document.querySelector("#liveTranscriptionEnabled").checked = true;
    }
    appDependencies.setSessionSettingsLocked();
    appDependencies.renderBanners(health.banners);
    appDependencies.applyBranding(health.uiBrandTitle, health.uiBrandTagline);
    if (Array.isArray(health.uiPromptTemplates) && health.uiPromptTemplates.length > 0) {
      appDependencies.state.promptTemplates = health.uiPromptTemplates
        .map((template, index) => ({
          id: String(template?.id || `template-${index + 1}`),
          label: String(template?.label || `Template ${index + 1}`),
          content: String(template?.content || "").trim(),
        }))
        .filter((template) => template.content);
    }
    appDependencies.renderPromptTemplateButtons(appDependencies.state.promptTemplates);
    appDependencies.state.asrAvailable = !!health.asrReady && !!health.model;
    appDependencies.state.proofreadAvailable = !!health.proofreadModel;
    appDependencies.state.diarizationAvailable = !!health.diarizationEnabled;
    appDependencies.state.selfSignupEnabled = !!health.selfSignupEnabled;
    appDependencies.state.auth.keycloakEnabled = !!health.keycloakEnabled;
    appDependencies.state.auth.keycloakButtonLabel = String(health.keycloakButtonLabel || "Keycloakでログイン");
    appDependencies.renderAuthState();

    if (!appDependencies.state.proofreadAvailable) {
      appDependencies.setProofread(
        "校正・翻訳機能が無効です。\nサーバーの API キー設定（PROOFREAD_API_KEY / SUMMARY_API_KEY / ASR_API_KEY）を確認してください。",
        "利用不可"
      );
      if (appDependencies.proofreadBtn) {
        appDependencies.proofreadBtn.title = "校正・翻訳機能はサーバーで無効";
      }
    } else if (appDependencies.proofreadBtn) {
      appDependencies.proofreadBtn.title = appDependencies.proofreadActionLabel();
    }

    if (!appDependencies.state.asrAvailable) {
      appDependencies.setStatus("asr_unavailable");
    }

    if (appDependencies.diarizationToggleEl) {
      const recordingLocked = appDependencies.isRecordingInteractionLocked();
      appDependencies.diarizationToggleEl.disabled = recordingLocked || !appDependencies.state.diarizationAvailable;
      appDependencies.diarizationToggleEl.title = recordingLocked
        ? "録音中・停止処理中は変更できません"
        : appDependencies.state.diarizationAvailable
          ? "話者分離を有効/無効"
          : "サーバーで話者分離は無効";
    }
    appDependencies.state.diarizationSpeakerCap = Math.max(
      appDependencies.DIARIZATION_SPEAKER_MIN,
      Number(health.diarizationSpeakerCap || appDependencies.DIARIZATION_SPEAKER_MAX)
    );

    if (!appDependencies.state.hasSavedDiarizationSpeakerSettings) {
      const defaultNum = Number(health.diarizationDefaultNumSpeakers || 0);
      const defaultMin = Number(health.diarizationDefaultMinSpeakers || 0);
      const defaultMax = Number(health.diarizationDefaultMaxSpeakers || 0);

      let mode = "auto";
      if (defaultNum > 0) {
        mode = "fixed";
      } else if (defaultMin > 0 || defaultMax > 0) {
        mode = "range";
      }

      appDependencies.applyDiarizationSpeakerSettings(
        {
          mode,
          count: defaultNum > 0 ? defaultNum : appDependencies.state.diarizationSpeakerCount,
          min: defaultMin > 0 ? defaultMin : appDependencies.state.diarizationMinSpeakers,
          max: defaultMax > 0 ? defaultMax : appDependencies.state.diarizationMaxSpeakers,
        },
        { persist: false }
      );
    } else {
      appDependencies.updateDiarizationSpeakerUi();
    }

    appDependencies.applyDiarizationEnabled(appDependencies.state.diarizationEnabled, { persist: false });
    appDependencies.logClientEvent("capabilities.load.success", {
      asrReady: !!health.asrReady,
      proofreadReady: !!health.proofreadModel,
      diarizationEnabled: !!health.diarizationEnabled,
    });
    return health;
  } catch {
    if (generation !== capabilitiesGeneration) return;
    // ignore capability check errors
    appDependencies.logClientEvent("capabilities.load.failed");
    return null;
  }
}

  return { loadCapabilities };
}
