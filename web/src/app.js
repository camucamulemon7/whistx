import { createAppState } from "./state/store.js";
import { createAuthController } from "./controllers/auth.js";
import { createSettingsController } from "./controllers/settings.js";
import { createAiController } from "./controllers/ai.js";
import { createLayoutController } from "./controllers/layout.js";
import { createHistoryController } from "./controllers/history.js";
import { createTelemetryController } from "./controllers/telemetry.js";
import { createScreenshotViewerController } from "./controllers/screenshot-viewer.js";
import { createTranscriptController } from "./controllers/transcript.js";
import { createWorkspaceController } from "./controllers/workspace.js";
import { createTranscriptionController } from "./controllers/transcription.js";
import { createRecordingController } from "./controllers/recording.js";
import { createMediaCaptureController } from "./controllers/media-capture.js";
import { createGlossaryController } from "./controllers/glossary.js";
import { createCapabilitiesController } from "./controllers/capabilities.js";
import { createModalController } from "./ui/modals.js";
import { createBannerController, createToastController } from "./ui/notifications.js";
import { fetchCapabilities } from "./capabilities/api.js";
import { fetchAuthState, loginRequest, bootstrapAdminRequest, registerRequest, logoutRequest, updateDisplayNameRequest } from "./auth/api.js";
import { canUseWorkspace as canUseWorkspaceForAuth, persistGuestMode, readGuestMode, serializeUserLabel } from "./auth/session.js";
import { deleteHistoryRequest, fetchHistoryList as fetchHistoryListRequest, fetchHistoryDetail, saveHistoryRequest } from "./history/api.js";
import { fetchSharedGlossary as fetchSharedGlossaryRequest, saveSharedGlossary as saveSharedGlossaryRequest } from "./api/glossary.js";
import { fetchJson } from "./api/client.js";
import { readSseJsonStream } from "./api/sse.js";
import { applyTranscriptRevision } from "./transcription/revisions.js";
import { LiveCapture } from "./meeting/live-capture.js";
import { createMeetingWorkspace } from "./meeting/workspace.js";
import { transcriptParagraphs, transcriptJoiner } from "./meeting/paragraphs.js";
import { applyHistoryDetailPayload, applyHistoryListPayload, clearHistoryState } from "./history/state.js";
import { readStoredJson, readStoredValue, writeStoredValue } from "./state/storage.js";
import { normalizeTheme, resolveInitialTheme, themeMetaColor } from "./ui/theme.js";
import {
  clampSpeakerCount as clampSpeakerCountValue,
  escapeHtml,
  formatAudioSource as formatModeLabel,
  formatLanguageLabel,
  formatStatusText,
  formatTimestamp as formatMs,
  isLoginRequiredError,
  normalizeProofreadMode,
  normalizeSpeakerMode,
} from "./ui/format.js";
import {
  arrayBufferToBase64,
  buildWebSocketUrl,
  normalizeWsPath,
  waitForOpen,
  waitForSessionFinalized,
  waitForSessionReady,
} from "./transcription/websocket.js";
import {
  VAD_SAMPLE_MS,
  buildVadDecision as calculateVadDecision,
  normalizeAudioSource,
  shouldCutOnSilence,
  shouldSkipChunkByVad as calculateShouldSkipChunkByVad,
  vadSourceCutPolicy,
  vadThresholdForSource,
} from "./audio/vad.js";

const $ = (selector) => document.querySelector(selector);
let meetingWorkspace = null;
let refinementController = null;

const statusTextEl = $("#statusText");
const connCountEl = $("#connCount");
const bannersContainerEl = $("#bannersContainer");
const brandTitleEl = $("#brandTitle");
const brandTaglineEl = $("#brandTagline");
const themeToggleBtn = $("#themeToggle");
const audioLevelIndicatorEl = $("#audioLevelIndicator");
const audioLevelMatrixEl = $("#audioLevelMatrix");

const languageEl = $("#language");
try {
  const savedLanguage = localStorage.getItem("whistx_language");
  if (savedLanguage !== null && [...languageEl.options].some((option) => option.value === savedLanguage)) {
    languageEl.value = savedLanguage;
  }
} catch {
  // Storage may be unavailable in restricted browser sessions.
}
languageEl?.addEventListener("change", () => {
  try {
    localStorage.setItem("whistx_language", languageEl.value);
  } catch {
    // Keep the current selection usable even when storage is unavailable.
  }
});
const audioSourceEl = $("#audioSource");
const audioSourceHintEl = $("#audioSourceHint");
const audioSourceHintTextEl = $("#audioSourceHintText");
const diarizationToggleEl = $("#diarizationEnabled");
const diarizationStateTextEl = $("#diarizationStateText");
const diarizationConfigRowEl = $("#diarizationConfigRow");
const diarizationSpeakerModeEl = $("#diarizationSpeakerMode");
const diarizationSpeakerCountEl = $("#diarizationSpeakerCount");
const diarizationMinSpeakersEl = $("#diarizationMinSpeakers");
const diarizationMaxSpeakersEl = $("#diarizationMaxSpeakers");
const diarizationSpeakerHintEl = $("#diarizationSpeakerHint");
const captureScreenshotsEnabledEl = $("#captureScreenshotsEnabled");
const captureScreenshotsStateTextEl = $("#captureScreenshotsStateText");
const screenshotDiffSkipEnabledEl = $("#screenshotDiffSkipEnabled");
const screenshotDiffSkipStateTextEl = $("#screenshotDiffSkipStateText");
const showTranscriptAudioEnabledEl = $("#showTranscriptAudioEnabled");
const showTranscriptAudioStateTextEl = $("#showTranscriptAudioStateText");
const autoGainEnabledEl = $("#autoGainEnabled");
const autoGainStateTextEl = $("#autoGainStateText");
const chunkSecondsEl = $("#chunkSeconds");
const promptEl = $("#prompt");
const sharedVocabularyEl = $("#sharedVocabulary");
const sharedVocabularySaveBtn = $("#sharedVocabularySaveBtn");
const sharedVocabularyMetaEl = $("#sharedVocabularyMeta");
const summaryPromptEl = $("#summaryPrompt");
const summaryPromptEditorEl = $("#summaryPromptEditor");
const summaryPromptToggleBtn = $("#summaryPromptToggleBtn");
const promptTemplateButtonsEl = $("#promptTemplateButtons");
const workspacePanelsEl = $("#workspacePanels");
const panelResizerEls = Array.from(document.querySelectorAll("[data-resizer]"));
const panelToggleEls = Array.from(document.querySelectorAll("[data-panel-toggle]"));
const WORKSPACE_STACK_BREAKPOINT = 1280;
const WORKSPACE_DUAL_AI_BREAKPOINT = 1439;
const chunkHintEl = $("#chunkHint");
const presetButtons = Array.from(document.querySelectorAll("[data-chunk-preset]"));
const settingsAdvancedToggleEl = $("#settingsAdvancedToggle");
const inputAdvancedSettingsEl = $("#inputAdvancedSettings");
const aiTabEls = Array.from(document.querySelectorAll("[data-ai-tab]"));

const startBtn = $("#startBtn");
const summaryBtn = $("#summaryBtn");
const summaryBtnLabelEl = $("#summaryBtnLabel");
const proofreadBtn = $("#proofreadBtn");
const proofreadBtnLabelEl = $("#proofreadBtnLabel");
const proofreadModeEl = $("#proofreadMode");
const copyBtn = $("#copyBtn");
const copyProofreadBtn = $("#copyProofreadBtn");
const clearBtn = $("#clearBtn");
const saveBtn = $("#saveBtn");
const saveTitleInputEl = $("#saveTitleInput");
const saveStateBadgeEl = $("#saveStateBadge");
const loginBtn = $("#loginBtn");
const logoutBtn = $("#logoutBtn");
const authProfileEditBtn = $("#authProfileEditBtn");
const authProfileEditorEl = $("#authProfileEditor");
const authProfileDisplayNameEl = $("#authProfileDisplayName");
const authProfileSaveBtn = $("#authProfileSaveBtn");
const authProfileCancelBtn = $("#authProfileCancelBtn");
const historyCollapseBtn = $("#historyCollapseBtn");
const historyDrawerOpenEl = $("#historyDrawerOpen");
const historyDrawerCloseEl = document.querySelector("#historyDrawerClose");
const historyDrawerBackdropEl = $("#historyDrawerBackdrop");
const authUserLabelEl = document.querySelector("#authUserLabel");
const authGuestViewEl = $("#authGuestView");
const authUserViewEl = $("#authUserView");
const adminQueueBtn = $("#adminQueueBtn");
const adminQueueBadgeEl = $("#adminQueueBadge");
const loginEmailEl = $("#loginEmail");
const loginPasswordEl = $("#loginPassword");
const loginSubmitBtn = $("#loginSubmitBtn");
const keycloakLoginBtnEl = $("#keycloakLoginBtn");
const guestLoginBtn = $("#guestLoginBtn");
const bootstrapDisplayNameEl = $("#bootstrapDisplayName");
const bootstrapEmailEl = $("#bootstrapEmail");
const bootstrapPasswordEl = $("#bootstrapPassword");
const bootstrapAdminBtnEl = $("#bootstrapAdminBtn");
const authBootstrapSectionEl = $("#authBootstrapSection");
const authLoginSectionEl = $("#authLoginSection");
const authRegisterSectionEl = $("#authRegisterSection");
const registerDisplayNameEl = $("#registerDisplayName");
const registerEmailEl = $("#registerEmail");
const registerPasswordEl = $("#registerPassword");
const registerBtn = $("#registerBtn");
const registerHintEl = $("#registerHint");
const historySearchInputEl = $("#historySearchInput");
const historyListEl = $("#historyList");
const historyEmptyEl = $("#historyEmpty");
const historyCountBadgeEl = $("#historyCountBadge");
const recordTelemetryEl = $("#recordTelemetry");
const copySummaryBtnEl = $("#copySummaryBtn");
const screenshotModalEl = $("#screenshotModal");
const screenshotModalCloseEl = $("#screenshotModalClose");
const screenshotModalImageEl = $("#screenshotModalImage");
const screenshotModalViewportEl = $("#screenshotModalViewport");
const screenshotModalStageEl = $("#screenshotModalStage");
const screenshotZoomOutBtnEl = $("#screenshotZoomOutBtn");
const screenshotZoomResetBtnEl = $("#screenshotZoomResetBtn");
const screenshotZoomInBtnEl = $("#screenshotZoomInBtn");

const dlTxt = $("#dlTxt");
const dlJsonl = $("#dlJsonl");
const dlZip = $("#dlZip");

const logEl = $("#log");
const segmentCountEl = $("#segmentCount");
const summaryTextEl = $("#summaryText");
const summaryMetaEl = $("#summaryMeta");
const proofreadTextEl = $("#proofreadText");
const proofreadMetaEl = $("#proofreadMeta");
const toastContainer = $("#toastContainer");
const appEl = document.querySelector(".app");
const mainContentEl = document.querySelector(".main-content");
const workspaceShellEl = $("#workspaceShell");
const historyRailEl = document.querySelector(".history-rail");
const transcriptPanelEl = document.querySelector(".transcript-panel");
const proofreadPanelEl = document.querySelector(".proofread-panel");
const summaryPanelEl = document.querySelector(".summary-panel");
const sidePanelSections = Array.from(document.querySelectorAll(".side-panel-section"));

const CHUNK_MIN_SECONDS = 15;
const CHUNK_MAX_SECONDS = 60;
const CHUNK_DEFAULT_SECONDS = 30;
const DIARIZATION_SPEAKER_MIN = 1;
const DIARIZATION_SPEAKER_MAX = 12;

const VAD_SEGMENT_MIN_MS = 12_000;
const VAD_SOFT_CUT_GRACE_MS = 6_000;
const VAD_NOISE_FLOOR_WARMUP_MS = 1_800;
const VAD_NOISE_FLOOR_EWMA = 0.18;
const AUDIO_LEVEL_NOISE_FLOOR = 0.0025;
const AUDIO_LEVEL_GAIN = 22;
const AUDIO_LEVEL_EXPONENT = 0.65;
const AUDIO_LEVEL_COLUMNS = 21;
const AUDIO_LEVEL_SEGMENTS = 10;
const AUTO_GAIN_ANALYZE_MS = 1200;
const AUTO_GAIN_MIN_RMS = 0.018;
const AUTO_GAIN_TARGET_RMS = 0.04;
const AUTO_GAIN_MAX = 2.8;
const AUTO_GAIN_SMOOTHING = 0.22;
const SCREENSHOT_DIFF_WIDTH = 64;
const SCREENSHOT_DIFF_HEIGHT = 36;
const SCREENSHOT_DIFF_PIXEL_THRESHOLD = 12;
const SCREENSHOT_DIFF_MEAN_THRESHOLD = 4;
const SCREENSHOT_DIFF_CHANGED_RATIO_THRESHOLD = 0.06;
const SCREENSHOT_ZOOM_MIN = 1;
const SCREENSHOT_ZOOM_MAX = 5;
const SCREENSHOT_ZOOM_STEP = 0.25;
const SCREENSHOT_MAX_WIDTH = 1600;
const SCREENSHOT_DEGRADED_MAX_WIDTH = 1280;
const SCREENSHOT_WEBP_QUALITY = 0.93;
const CLIENT_VAD_DROP_ENABLED = false;
const HISTORY_SEARCH_DEBOUNCE_MS = 180;
const BACKLOG_WARN_THRESHOLD = 2;
const BACKLOG_DANGER_THRESHOLD = 4;
const SCREENSHOT_MIN_INTERVAL_MS = 45_000;
const SCREENSHOT_DEGRADED_INTERVAL_MS = 45_000;
const DEFAULT_SOC_PROMPT_TEMPLATE = `SoC, ASIC, chiplet, CPU, GPU, NPU, DSP, ISP, VPU, DPU, MCU, PMU, NoC, interconnect, AXI, AXI4, AXI-Lite, AHB, APB, ACE, CHI, UCIe, PCIe, CXL, DDR, DDR4, DDR5, LPDDR4, LPDDR5, HBM, SRAM, ROM, eMMC, UFS, PHY, SerDes, PLL, DLL, RC oscillator, clock, clock tree, clock gating, reset, async reset, sync reset, power domain, voltage island, retention, isolation, level shifter, DVFS, AVS, UPF, CPF, RTL, SystemVerilog, Verilog, VHDL, UVM, testbench, assertion, SVA, lint, SpyGlass, CDC, RDC, STA, MCMM, OCV, AOCV, POCV, derate, setup, hold, recovery, removal, skew, jitter, uncertainty, timing closure, timing path, false path, multicycle path, path group, endpoint, startpoint, slack, WNS, TNS, violating path, critical path, synthesis, logic synthesis, Design Compiler, Genus, netlist, mapped netlist, unmapped netlist, compile, incremental compile, retiming, boundary optimization, datapath optimization, resource sharing, register balancing, ECO, formal, equivalence check, LEC, Conformal, Formality, gate-level simulation, GLS, SDF, back annotation, place and route, place-and-route, PnR, floorplan, floorplanning, macro placement, standard cell, utilization, density, congestion, global placement, detailed placement, legalization, CTS, clock tree synthesis, useful skew, hold fixing, setup fixing, routing, global route, detailed route, track assignment, antenna, filler cell, decap, tap cell, endcap, spare cell, spare gate, metal fill, density fill, ECO route, route guide, signoff, sign-off, DRC, LVS, ERC, extraction, parasitic extraction, RC extraction, SPEF, DEF, LEF, Liberty, .lib, TLU+, QRC, StarRC, Quantus, IR drop, dynamic IR drop, static IR drop, EM, electromigration, voltage drop, power integrity, signal integrity, SI, crosstalk, noise, glitch, overshoot, undershoot, hotspot, thermal, leakage, dynamic power, switching power, internal power, leakage power, power analysis, PrimeTime PX, PrimePower, Voltus, RedHawk, vectorless, VCD, FSDB, SAIF, toggle rate, activity factor, inrush current, rush current, decoupling capacitor, decap cell, package model, bump, substrate, interposer, TSV, process node, 28nm, 16nm, 12nm, 7nm, 5nm, 4nm, 3nm, FinFET, GAA, foundry, TSMC, Samsung, Intel, PDK, DFM, manufacturability, yield, wafer, lot, mask, reticle, tape-out, respin, metal fix, MPW, shuttle, bring-up, validation, characterization, errata, workaround, DFT, scan, scan chain, scan compression, EDT, ATPG, stuck-at, transition fault, path delay fault, bridging fault, JTAG, boundary scan, MBIST, LBIST, BISR, repair, fuse, eFuse, OTP, secure boot, TrustZone, TEE, firmware, bootloader, NAND, NAND flash, Toggle NAND, ONFI, raw NAND, managed NAND, SLC, MLC, TLC, QLC, PLC, 3D NAND, V-NAND, charge trap, floating gate, page, block, plane, die, LUN, bad block, bad block management, BBT, ECC, BCH, LDPC, RAID, read disturb, program disturb, erase disturb, wear leveling, garbage collection, overprovisioning, endurance, retention, BER, bit error rate, read retry, soft decoding, threshold voltage, ISPP, incremental step pulse programming, erase verify, program verify, copyback, cache read, cache program, multi-plane, interleaving, channel, CE, RE, WE, ALE, CLE, R/B, spare area, OOB, metadata, FTL, flash translation layer, NVMe, SATA, controller, queue depth, throughput, latency, bandwidth, QoS, arbiter, scheduler, mux, demux, crossbar, SRAM compiler, memory compiler, register file, dual port RAM, single port RAM, SRAM macro, macro, hard macro, soft macro, black box, hierarchy, partition, block-level, top-level, full-chip, chip top, top module, hierarchy flattening, dont_touch, set_false_path, set_multicycle_path, create_clock, generated clock, propagated clock, ideal clock, set_input_delay, set_output_delay, set_clock_uncertainty, set_clock_groups, operating condition, corner, slow corner, fast corner, typical corner, SS, FF, TT, RCmax, RCmin, setup view, hold view.`;

const runtimeUi = {
  injectedStyle: null,
  overlayEl: null,
  loginOverlayEl: null,
  historyDrawerEl: null,
  screenshotModalEl: null,
  screenshotModalImageEl: null,
  summaryCopyBtnEl: null,
  appLocked: true,
};
const { openManagedModal, closeManagedModal, topmostModal, trapModalFocus } = createModalController({
  document,
  window,
  getBlockingModals: () => [runtimeUi.screenshotModalEl],
});
const { renderBanners } = createBannerController({ container: bannersContainerEl, document });
const { showToast } = createToastController({ container: toastContainer, document });

const state = createAppState({
  chunkDefaultSeconds: CHUNK_DEFAULT_SECONDS,
  speakerMax: DIARIZATION_SPEAKER_MAX,
  defaultPromptTemplate: DEFAULT_SOC_PROMPT_TEMPLATE,
});

const controllerContext = {
  get AUDIO_LEVEL_COLUMNS() { return AUDIO_LEVEL_COLUMNS; },
  get AUDIO_LEVEL_EXPONENT() { return AUDIO_LEVEL_EXPONENT; },
  get AUDIO_LEVEL_GAIN() { return AUDIO_LEVEL_GAIN; },
  get AUDIO_LEVEL_NOISE_FLOOR() { return AUDIO_LEVEL_NOISE_FLOOR; },
  get AUDIO_LEVEL_SEGMENTS() { return AUDIO_LEVEL_SEGMENTS; },
  get AUTO_GAIN_ANALYZE_MS() { return AUTO_GAIN_ANALYZE_MS; },
  get AUTO_GAIN_MAX() { return AUTO_GAIN_MAX; },
  get AUTO_GAIN_MIN_RMS() { return AUTO_GAIN_MIN_RMS; },
  get AUTO_GAIN_SMOOTHING() { return AUTO_GAIN_SMOOTHING; },
  get AUTO_GAIN_TARGET_RMS() { return AUTO_GAIN_TARGET_RMS; },
  get BACKLOG_DANGER_THRESHOLD() { return BACKLOG_DANGER_THRESHOLD; },
  get BACKLOG_WARN_THRESHOLD() { return BACKLOG_WARN_THRESHOLD; },
  get CHUNK_DEFAULT_SECONDS() { return CHUNK_DEFAULT_SECONDS; },
  get CHUNK_MAX_SECONDS() { return CHUNK_MAX_SECONDS; },
  get CHUNK_MIN_SECONDS() { return CHUNK_MIN_SECONDS; },
  get CLIENT_VAD_DROP_ENABLED() { return CLIENT_VAD_DROP_ENABLED; },
  get DIARIZATION_SPEAKER_MAX() { return DIARIZATION_SPEAKER_MAX; },
  get DIARIZATION_SPEAKER_MIN() { return DIARIZATION_SPEAKER_MIN; },
  get HISTORY_SEARCH_DEBOUNCE_MS() { return HISTORY_SEARCH_DEBOUNCE_MS; },
  get SCREENSHOT_DEGRADED_INTERVAL_MS() { return SCREENSHOT_DEGRADED_INTERVAL_MS; },
  get SCREENSHOT_DEGRADED_MAX_WIDTH() { return SCREENSHOT_DEGRADED_MAX_WIDTH; },
  get SCREENSHOT_DIFF_CHANGED_RATIO_THRESHOLD() { return SCREENSHOT_DIFF_CHANGED_RATIO_THRESHOLD; },
  get SCREENSHOT_DIFF_HEIGHT() { return SCREENSHOT_DIFF_HEIGHT; },
  get SCREENSHOT_DIFF_MEAN_THRESHOLD() { return SCREENSHOT_DIFF_MEAN_THRESHOLD; },
  get SCREENSHOT_DIFF_PIXEL_THRESHOLD() { return SCREENSHOT_DIFF_PIXEL_THRESHOLD; },
  get SCREENSHOT_DIFF_WIDTH() { return SCREENSHOT_DIFF_WIDTH; },
  get SCREENSHOT_MAX_WIDTH() { return SCREENSHOT_MAX_WIDTH; },
  get SCREENSHOT_MIN_INTERVAL_MS() { return SCREENSHOT_MIN_INTERVAL_MS; },
  get SCREENSHOT_WEBP_QUALITY() { return SCREENSHOT_WEBP_QUALITY; },
  get SCREENSHOT_ZOOM_MAX() { return SCREENSHOT_ZOOM_MAX; },
  get SCREENSHOT_ZOOM_MIN() { return SCREENSHOT_ZOOM_MIN; },
  get VAD_NOISE_FLOOR_EWMA() { return VAD_NOISE_FLOOR_EWMA; },
  get VAD_NOISE_FLOOR_WARMUP_MS() { return VAD_NOISE_FLOOR_WARMUP_MS; },
  get VAD_SEGMENT_MIN_MS() { return VAD_SEGMENT_MIN_MS; },
  get VAD_SOFT_CUT_GRACE_MS() { return VAD_SOFT_CUT_GRACE_MS; },
  get WORKSPACE_DUAL_AI_BREAKPOINT() { return WORKSPACE_DUAL_AI_BREAKPOINT; },
  get WORKSPACE_STACK_BREAKPOINT() { return WORKSPACE_STACK_BREAKPOINT; },
  get abortRecordingAfterSocketLoss() { return abortRecordingAfterSocketLoss; },
  get addLogLine() { return addLogLine; },
  get adminQueueBadgeEl() { return adminQueueBadgeEl; },
  get adminQueueBtn() { return adminQueueBtn; },
  get aiTabEls() { return aiTabEls; },
  get applyAdvancedSettingsOpen() { return applyAdvancedSettingsOpen; },
  get applyAudioSource() { return applyAudioSource; },
  get applyBranding() { return applyBranding; },
  get applyChunkSeconds() { return applyChunkSeconds; },
  get applyDiarizationEnabled() { return applyDiarizationEnabled; },
  get applyDiarizationSpeakerSettings() { return applyDiarizationSpeakerSettings; },
  get applyHistoryDrawerOpen() { return applyHistoryDrawerOpen; },
  get applySpeakerPatch() { return applySpeakerPatch; },
  get audioLevelIndicatorEl() { return audioLevelIndicatorEl; },
  get audioLevelMatrixEl() { return audioLevelMatrixEl; },
  get audioSourceEl() { return audioSourceEl; },
  get audioSourceHintEl() { return audioSourceHintEl; },
  get audioSourceHintTextEl() { return audioSourceHintTextEl; },
  get authBootstrapSectionEl() { return authBootstrapSectionEl; },
  get authGuestViewEl() { return authGuestViewEl; },
  get authLoginSectionEl() { return authLoginSectionEl; },
  get authProfileCancelBtn() { return authProfileCancelBtn; },
  get authProfileDisplayNameEl() { return authProfileDisplayNameEl; },
  get authProfileEditBtn() { return authProfileEditBtn; },
  get authProfileEditorEl() { return authProfileEditorEl; },
  get authProfileSaveBtn() { return authProfileSaveBtn; },
  get authRegisterSectionEl() { return authRegisterSectionEl; },
  get authUserLabelEl() { return authUserLabelEl; },
  get authUserViewEl() { return authUserViewEl; },
  get autoGainEnabledEl() { return autoGainEnabledEl; },
  get autoGainStateTextEl() { return autoGainStateTextEl; },
  get bootstrapDisplayNameEl() { return bootstrapDisplayNameEl; },
  get bootstrapEmailEl() { return bootstrapEmailEl; },
  get bootstrapPasswordEl() { return bootstrapPasswordEl; },
  get brandTaglineEl() { return brandTaglineEl; },
  get brandTitleEl() { return brandTitleEl; },
  get buildAutoSaveTitle() { return buildAutoSaveTitle; },
  get buildVadDecision() { return buildVadDecision; },
  get canUseWorkspace() { return canUseWorkspace; },
  get captureDisplayScreenshot() { return captureDisplayScreenshot; },
  get captureScreenshotsEnabledEl() { return captureScreenshotsEnabledEl; },
  get captureScreenshotsStateTextEl() { return captureScreenshotsStateTextEl; },
  get chunkHintEl() { return chunkHintEl; },
  get chunkSecondsEl() { return chunkSecondsEl; },
  get cleanupMedia() { return cleanupMedia; },
  get clearBtn() { return clearBtn; },
  get clearView() { return clearView; },
  get closeManagedModal() { return closeManagedModal; },
  get confirmWorkspaceDiscard() { return confirmWorkspaceDiscard; },
  get connCountEl() { return connCountEl; },
  get copyBtn() { return copyBtn; },
  get copyProofreadBtn() { return copyProofreadBtn; },
  get currentMeetingSource() { return currentMeetingSource; },
  get currentVadSourceMode() { return currentVadSourceMode; },
  get diarizationConfigRowEl() { return diarizationConfigRowEl; },
  get diarizationMaxSpeakersEl() { return diarizationMaxSpeakersEl; },
  get diarizationMinSpeakersEl() { return diarizationMinSpeakersEl; },
  get diarizationSpeakerCountEl() { return diarizationSpeakerCountEl; },
  get diarizationSpeakerHintEl() { return diarizationSpeakerHintEl; },
  get diarizationSpeakerModeEl() { return diarizationSpeakerModeEl; },
  get diarizationStateTextEl() { return diarizationStateTextEl; },
  get diarizationToggleEl() { return diarizationToggleEl; },
  get dlJsonl() { return dlJsonl; },
  get dlTxt() { return dlTxt; },
  get dlZip() { return dlZip; },
  get emitScreenshotSkipTelemetry() { return emitScreenshotSkipTelemetry; },
  get ensureSocket() { return ensureSocket; },
  get extractTranscriptText() { return extractTranscriptText; },
  get guestLoginBtn() { return guestLoginBtn; },
  get hasAudioTrack() { return hasAudioTrack; },
  get historyCollapseBtn() { return historyCollapseBtn; },
  get historyCountBadgeEl() { return historyCountBadgeEl; },
  get historyDrawerBackdropEl() { return historyDrawerBackdropEl; },
  get historyDrawerCloseEl() { return historyDrawerCloseEl; },
  get historyDrawerOpenEl() { return historyDrawerOpenEl; },
  get historyEmptyEl() { return historyEmptyEl; },
  get historyListEl() { return historyListEl; },
  get historyRailEl() { return historyRailEl; },
  get inputAdvancedSettingsEl() { return inputAdvancedSettingsEl; },
  get isRecordingInteractionLocked() { return isRecordingInteractionLocked; },
  get keycloakLoginBtnEl() { return keycloakLoginBtnEl; },
  get languageEl() { return languageEl; },
  get loadCapabilities() { return loadCapabilities; },
  get loadHistoryList() { return loadHistoryList; },
  get logClientEvent() { return logClientEvent; },
  get logEl() { return logEl; },
  get logWsEvent() { return logWsEvent; },
  get loginBtn() { return loginBtn; },
  get loginEmailEl() { return loginEmailEl; },
  get loginPasswordEl() { return loginPasswordEl; },
  get logoutBtn() { return logoutBtn; },
  get markProofreadStale() { return markProofreadStale; },
  get markWorkspaceClean() { return markWorkspaceClean; },
  get markWorkspaceDirty() { return markWorkspaceDirty; },
  get meetingWorkspace() { return meetingWorkspace; },
  set meetingWorkspace(value) { meetingWorkspace = value; },
  get openManagedModal() { return openManagedModal; },
  get panelResizerEls() { return panelResizerEls; },
  get panelToggleEls() { return panelToggleEls; },
  get prepareInputStream() { return prepareInputStream; },
  get presetButtons() { return presetButtons; },
  get promptEl() { return promptEl; },
  get promptTemplateButtonsEl() { return promptTemplateButtonsEl; },
  get proofreadActionLabel() { return proofreadActionLabel; },
  get proofreadBtn() { return proofreadBtn; },
  get proofreadBtnLabelEl() { return proofreadBtnLabelEl; },
  get proofreadMetaEl() { return proofreadMetaEl; },
  get proofreadModeEl() { return proofreadModeEl; },
  get proofreadPanelEl() { return proofreadPanelEl; },
  get proofreadTextEl() { return proofreadTextEl; },
  get recordTelemetryEl() { return recordTelemetryEl; },
  get refinementController() { return refinementController; },
  set refinementController(value) { refinementController = value; },
  get registerBtn() { return registerBtn; },
  get registerDisplayNameEl() { return registerDisplayNameEl; },
  get registerEmailEl() { return registerEmailEl; },
  get registerHintEl() { return registerHintEl; },
  get registerPasswordEl() { return registerPasswordEl; },
  get renderAudioLevel() { return renderAudioLevel; },
  get renderAuthState() { return renderAuthState; },
  get renderBanners() { return renderBanners; },
  get renderEmptyTranscriptState() { return renderEmptyTranscriptState; },
  get renderHistoryList() { return renderHistoryList; },
  get renderPromptTemplateButtons() { return renderPromptTemplateButtons; },
  get renderTranscriptParagraphs() { return renderTranscriptParagraphs; },
  get resetRuntimeSessionState() { return resetRuntimeSessionState; },
  get resolveDiarizationStartOptions() { return resolveDiarizationStartOptions; },
  get runtimeUi() { return runtimeUi; },
  get saveBtn() { return saveBtn; },
  get saveStateBadgeEl() { return saveStateBadgeEl; },
  get saveTitleInputEl() { return saveTitleInputEl; },
  get screenshotDiffSkipEnabledEl() { return screenshotDiffSkipEnabledEl; },
  get screenshotDiffSkipStateTextEl() { return screenshotDiffSkipStateTextEl; },
  get screenshotModalCloseEl() { return screenshotModalCloseEl; },
  get screenshotModalImageEl() { return screenshotModalImageEl; },
  get screenshotModalStageEl() { return screenshotModalStageEl; },
  get screenshotModalViewportEl() { return screenshotModalViewportEl; },
  get screenshotZoomInBtnEl() { return screenshotZoomInBtnEl; },
  get screenshotZoomOutBtnEl() { return screenshotZoomOutBtnEl; },
  get screenshotZoomResetBtnEl() { return screenshotZoomResetBtnEl; },
  get segmentCountEl() { return segmentCountEl; },
  get selectedLanguage() { return selectedLanguage; },
  get sendWsTelemetry() { return sendWsTelemetry; },
  get setAppLocked() { return setAppLocked; },
  get setProofread() { return setProofread; },
  get setSessionSettingsLocked() { return setSessionSettingsLocked; },
  get setStatus() { return setStatus; },
  get setSummary() { return setSummary; },
  get settingsAdvancedToggleEl() { return settingsAdvancedToggleEl; },
  get setupVad() { return setupVad; },
  get sharedVocabularyEl() { return sharedVocabularyEl; },
  get sharedVocabularyMetaEl() { return sharedVocabularyMetaEl; },
  get sharedVocabularySaveBtn() { return sharedVocabularySaveBtn; },
  get shouldSkipChunkByVad() { return shouldSkipChunkByVad; },
  get showRecordingInteractionBlocked() { return showRecordingInteractionBlocked; },
  get showScreenshotModal() { return showScreenshotModal; },
  get showToast() { return showToast; },
  get showTranscriptAudioEnabledEl() { return showTranscriptAudioEnabledEl; },
  get showTranscriptAudioStateTextEl() { return showTranscriptAudioStateTextEl; },
  get snapshotVadCounters() { return snapshotVadCounters; },
  get startBtn() { return startBtn; },
  get state() { return state; },
  get statusTextEl() { return statusTextEl; },
  get stopRecording() { return stopRecording; },
  get summaryBtn() { return summaryBtn; },
  get summaryBtnLabelEl() { return summaryBtnLabelEl; },
  get summaryMetaEl() { return summaryMetaEl; },
  get summaryPanelEl() { return summaryPanelEl; },
  get summaryPromptEditorEl() { return summaryPromptEditorEl; },
  get summaryPromptEl() { return summaryPromptEl; },
  get summaryPromptToggleBtn() { return summaryPromptToggleBtn; },
  get summaryTextEl() { return summaryTextEl; },
  get syncUnloadProtection() { return syncUnloadProtection; },
  get updateBackpressureState() { return updateBackpressureState; },
  get updateCaptureAutoGainState() { return updateCaptureAutoGainState; },
  get updateDiarizationSpeakerUi() { return updateDiarizationSpeakerUi; },
  get updateDownloadLinks() { return updateDownloadLinks; },
  get updateHistoryEmptyState() { return updateHistoryEmptyState; },
  get updateRecordingTelemetry() { return updateRecordingTelemetry; },
  get updateSaveControls() { return updateSaveControls; },
  get updateSegmentCount() { return updateSegmentCount; },
  get updateSharedVocabularyMeta() { return updateSharedVocabularyMeta; },
  get workspacePanelsEl() { return workspacePanelsEl; },
  get workspaceShellEl() { return workspaceShellEl; },
};
const { canUseWorkspace, setAppLocked, renderAuthState, syncAuthProfileEditor, setAuthProfileEditorOpen, handleAuthErrorFromLocation, loadAuthState, login, bootstrapAdmin, registerAccount, saveDisplayName, logout, loginAsGuest } = createAuthController(controllerContext);
const { selectedLanguage, setStatus, applyBranding, applyCaptureScreenshotsEnabled, applyScreenshotDiffSkipEnabled, applyShowTranscriptAudioEnabled, applyAutoGainEnabled, applyDiarizationEnabled, clampSpeakerCount, updateDiarizationSpeakerUi, applyDiarizationSpeakerSettings, resolveDiarizationStartOptions, normalizeChunkSeconds, updateChunkHint, updatePresetActive, applyChunkSeconds, audioSourceHintText, applyAudioSource, buildAutoSaveTitle } = createSettingsController(controllerContext);
const { setProofreadButtonBusy, proofreadActionLabel, applyProofreadMode, applySummaryPromptEditorOpen, setSummary, copySummaryText, setProofread, markProofreadStale, copyProofread, proofreadAll, summarizeAll } = createAiController(controllerContext);
const { applyHistoryDrawerOpen, applyHistoryCollapsed, updateHistoryControls, applyAdvancedSettingsOpen, syncAiResponsiveState, applyActiveAiPanel, applyWorkspaceRatios, updateWorkspaceGridTemplate, applyPanelCollapseState, setupWorkspaceResizers, setupPanelToggles, applyTheme, initTheme, updateThemeColorMeta, toggleTheme } = createLayoutController(controllerContext);
const { setHistorySearchQuery, formatHistoryMeta, formatHistoryDaysRemaining, updateHistoryEmptyState, renderHistoryList, loadHistoryList, renderHistoryDetail, openHistoryDetail, deleteHistory, saveCurrentHistory } = createHistoryController(controllerContext);
const { updateRecordingTelemetry, logWsEvent, logClientEvent, sendWsTelemetry, bucketizeBacklog, emitScreenshotSkipTelemetry, updateBackpressureState } = createTelemetryController(controllerContext);
const { clampScreenshotZoom, updateScreenshotZoomUi, recalculateScreenshotBaseSize, centerScreenshotViewport, setScreenshotZoom, setScreenshotZoomAt, zoomScreenshot, stopScreenshotDrag, beginScreenshotDrag, handleScreenshotDrag, resetScreenshotZoom, showScreenshotModal, hideScreenshotModal } = createScreenshotViewerController(controllerContext);
const { renderEmptyTranscriptState, updateSegmentCount, extractTranscriptText, updateDownloadLinks, renderTranscriptText, isLogNearBottom, scrollLogToBottom, updateTranscriptLatestButton, renderTranscriptParagraphs, addLogLine, applySpeakerPatch, copyAll } = createTranscriptController(controllerContext);
const { setSaveBadge, isRecordingInteractionLocked, shouldProtectWorkspaceFromUnload, handleBeforeUnload, syncUnloadProtection, markWorkspaceDirty, markWorkspaceClean, confirmWorkspaceDiscard, showRecordingInteractionBlocked, updateSaveControls, setSessionSettingsLocked, hasDiscardableWorkspaceData, clearView } = createWorkspaceController(controllerContext);
const { wsUrl, ensureSocket } = createTranscriptionController(controllerContext);
const { selectMimeType, generateSessionSeed, setUiRecording, setUiRecordingStarting, setUiRecordingStopping, resetRuntimeSessionState, commitNewRecordingWorkspace, sendChunk, clearChunkTimer, chunkHardMaxMs, shouldCutChunkOnSilence, requestChunkFlush, scheduleChunkStop, startRecorderCycle, finalizeStop, currentMeetingSource, navigateMeetingSource, startLiveRecording, finalizeLiveRecording, refineMeetingAudio, startRecording, stopRecording, abortRecordingAfterSocketLoss } = createRecordingController(controllerContext);
const { currentVadSourceMode, updateVadNoiseFloor, hasAudioTrack, setupDisplayCaptureVideo, captureDisplayScreenshot, buildScreenshotSignature, shouldSkipScreenshotByDiff, requestMicStream, requestDisplayStream, ensureAudioContextResumed, buildMixedAudioStream, sampleCaptureAutoGainRms, updateCaptureAutoGainState, bindDisplayEndEvents, getDisplayCaptureDiagnostics, logDisplayCaptureDiagnostics, prepareInputStream, ensureAudioLevelMatrix, renderAudioLevel, sampleVad, setupVad, stopVad, cleanupCaptureGraph, snapshotVadCounters, buildVadDecision, shouldSkipChunkByVad, cleanupMedia } = createMediaCaptureController(controllerContext);
const { renderPromptTemplateButtons, applySharedVocabulary, updateSharedVocabularyMeta, loadSharedGlossary, saveSharedGlossary } = createGlossaryController(controllerContext);
const { loadCapabilities } = createCapabilitiesController(controllerContext);

function buildRuntimeUi() {
  runtimeUi.loginOverlayEl = authGuestViewEl || null;
  runtimeUi.historyDrawerEl = historyRailEl || null;
  runtimeUi.screenshotModalEl = screenshotModalEl || null;
  runtimeUi.screenshotModalImageEl = screenshotModalImageEl || null;
  runtimeUi.summaryCopyBtnEl = copySummaryBtnEl || null;

  // Hide the old auth/history side-panel sections entirely.
  sidePanelSections.forEach((section) => {
    if (section.querySelector("#loginEmail") || section.querySelector("#registerEmail")) {
      section.classList.add("whistx-auth-hidden");
      section.hidden = true;
      section.setAttribute("aria-hidden", "true");
    }
    if (section.querySelector("#historySearchInput") || section.querySelector("#historyRefreshBtn")) {
      section.classList.add("whistx-history-hidden");
      section.hidden = true;
      section.setAttribute("aria-hidden", "true");
    }
  });

  if (loginBtn) loginBtn.hidden = true;
  if (authUserLabelEl) authUserLabelEl.textContent = "";

  if (runtimeUi.summaryCopyBtnEl && !runtimeUi.summaryCopyBtnEl.dataset.bound) {
    runtimeUi.summaryCopyBtnEl.dataset.bound = "1";
    runtimeUi.summaryCopyBtnEl.addEventListener("click", () => {
      copySummaryText();
    });
  }

  if (logEl && !logEl.dataset.boundScroll) {
    logEl.dataset.boundScroll = "1";
    logEl.addEventListener("scroll", () => {
      state.logAutoScrollEnabled = isLogNearBottom();
      updateTranscriptLatestButton();
    });
  }

  if (runtimeUi.screenshotModalEl && !runtimeUi.screenshotModalEl.dataset.bound) {
    runtimeUi.screenshotModalEl.dataset.bound = "1";
    runtimeUi.screenshotModalEl.addEventListener("click", (event) => {
      if (event.target === runtimeUi.screenshotModalEl || event.target?.matches?.("[data-modal-close]")) {
        hideScreenshotModal();
      }
    });
    screenshotModalCloseEl?.addEventListener("click", hideScreenshotModal);
    screenshotModalImageEl?.addEventListener("load", () => {
      recalculateScreenshotBaseSize();
      resetScreenshotZoom();
    });
    screenshotZoomOutBtnEl?.addEventListener("click", () => {
      zoomScreenshot(-SCREENSHOT_ZOOM_STEP);
    });
    screenshotZoomResetBtnEl?.addEventListener("click", () => {
      resetScreenshotZoom();
    });
    screenshotZoomInBtnEl?.addEventListener("click", () => {
      zoomScreenshot(SCREENSHOT_ZOOM_STEP);
    });
    screenshotModalViewportEl?.addEventListener("pointerdown", (event) => {
      beginScreenshotDrag(event);
    });
    screenshotModalViewportEl?.addEventListener("pointermove", (event) => {
      handleScreenshotDrag(event);
    });
    screenshotModalViewportEl?.addEventListener("pointerup", () => {
      stopScreenshotDrag();
    });
    screenshotModalViewportEl?.addEventListener("pointercancel", () => {
      stopScreenshotDrag();
    });
    screenshotModalViewportEl?.addEventListener("lostpointercapture", () => {
      stopScreenshotDrag();
    });
    screenshotModalViewportEl?.addEventListener("wheel", (event) => {
      if (runtimeUi.screenshotModalEl?.hidden) return;
      event.preventDefault();
      zoomScreenshot(event.deltaY < 0 ? SCREENSHOT_ZOOM_STEP : -SCREENSHOT_ZOOM_STEP, event);
    }, { passive: false });
    window.addEventListener("resize", () => {
      if (runtimeUi.screenshotModalEl?.hidden || !screenshotModalImageEl?.src) return;
      const currentZoom = state.screenshotZoom;
      recalculateScreenshotBaseSize();
      setScreenshotZoom(currentZoom, { recenter: currentZoom <= SCREENSHOT_ZOOM_MIN });
    });
  }
}

document.querySelector("#transcriptLatestBtn").addEventListener("click", () => {
  state.logAutoScrollEnabled = true;
  logEl.focus({ preventScroll: true });
  scrollLogToBottom();
});

document.querySelector("#transcriptDetailsToggle").addEventListener("click", (event) => {
  const expanded = logEl.classList.toggle("show-transcript-details");
  event.currentTarget.setAttribute("aria-pressed", String(expanded));
});

chunkSecondsEl.addEventListener("change", () => {
  applyChunkSeconds(chunkSecondsEl.value);
});

chunkSecondsEl.addEventListener("input", () => {
  updateChunkHint(normalizeChunkSeconds(chunkSecondsEl.value));
});

if (audioSourceEl) {
  audioSourceEl.addEventListener("change", () => {
    applyAudioSource(audioSourceEl.value);
  });
}

if (autoGainEnabledEl) {
  autoGainEnabledEl.addEventListener("change", () => {
    applyAutoGainEnabled(autoGainEnabledEl.checked);
  });
}

if (guestLoginBtn) {
  guestLoginBtn.addEventListener("click", () => {
    loginAsGuest();
  });
}

if (diarizationToggleEl) {
  diarizationToggleEl.addEventListener("change", () => {
    applyDiarizationEnabled(!!diarizationToggleEl.checked);
  });
}

if (diarizationSpeakerModeEl) {
  diarizationSpeakerModeEl.addEventListener("change", () => {
    applyDiarizationSpeakerSettings({
      mode: diarizationSpeakerModeEl.value,
    });
  });
}

if (diarizationSpeakerCountEl) {
  diarizationSpeakerCountEl.addEventListener("change", () => {
    applyDiarizationSpeakerSettings({
      count: diarizationSpeakerCountEl.value,
    });
  });
}

if (diarizationMinSpeakersEl) {
  diarizationMinSpeakersEl.addEventListener("change", () => {
    applyDiarizationSpeakerSettings({
      min: diarizationMinSpeakersEl.value,
    });
  });
}

if (diarizationMaxSpeakersEl) {
  diarizationMaxSpeakersEl.addEventListener("change", () => {
    applyDiarizationSpeakerSettings({
      max: diarizationMaxSpeakersEl.value,
    });
  });
}

presetButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const raw = button.getAttribute("data-chunk-preset") || "";
    applyChunkSeconds(raw);
  });
});

startBtn.addEventListener("click", () => {
  if (state.recordingPhase === "recording") {
    stopRecording();
  } else if (state.recordingPhase === "idle" && !state.finalizingStop) {
    startRecording();
  }
});

if (summaryBtn) {
  summaryBtn.addEventListener("click", () => {
    summarizeAll();
  });
}

if (proofreadBtn) {
  proofreadBtn.addEventListener("click", () => {
    proofreadAll();
  });
}

if (proofreadModeEl) {
  proofreadModeEl.addEventListener("change", () => {
    applyProofreadMode(proofreadModeEl.value);
  });
}

if (logoutBtn) {
  logoutBtn.addEventListener("click", () => {
    logout();
  });
}

if (authProfileEditBtn) {
  authProfileEditBtn.addEventListener("click", () => {
    setAuthProfileEditorOpen(!state.auth.profileEditorOpen);
  });
}

if (authProfileSaveBtn) {
  authProfileSaveBtn.addEventListener("click", () => {
    saveDisplayName();
  });
}

if (authProfileCancelBtn) {
  authProfileCancelBtn.addEventListener("click", () => {
    setAuthProfileEditorOpen(false);
  });
}

if (adminQueueBtn) {
  adminQueueBtn.addEventListener("click", () => {
    window.location.href = "/admin";
  });
}

if (historyDrawerCloseEl) {
  historyDrawerCloseEl.addEventListener("click", () => {
    applyHistoryDrawerOpen(false);
  });
}

if (historyDrawerOpenEl) {
  historyDrawerOpenEl.addEventListener("click", () => {
    applyHistoryDrawerOpen(true);
  });
}

if (historyDrawerBackdropEl) {
  historyDrawerBackdropEl.addEventListener("click", () => {
    applyHistoryDrawerOpen(false);
  });
}

if (historyCollapseBtn) {
  historyCollapseBtn.addEventListener("click", () => {
    applyHistoryCollapsed(!state.historyCollapsed);
  });
}

if (loginSubmitBtn) {
  loginSubmitBtn.addEventListener("click", () => {
    login();
  });
}

if (bootstrapAdminBtnEl) {
  bootstrapAdminBtnEl.addEventListener("click", () => {
    bootstrapAdmin();
  });
}

[loginEmailEl, loginPasswordEl].forEach((element) => {
  if (!element) return;
  element.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      login();
    }
  });
});

[authProfileDisplayNameEl].forEach((element) => {
  if (!element) return;
  element.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      saveDisplayName();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      setAuthProfileEditorOpen(false);
    }
  });
});

[bootstrapDisplayNameEl, bootstrapEmailEl, bootstrapPasswordEl].forEach((element) => {
  if (!element) return;
  element.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      bootstrapAdmin();
    }
  });
});

if (registerBtn) {
  registerBtn.addEventListener("click", () => {
    registerAccount();
  });
}

if (settingsAdvancedToggleEl) {
  settingsAdvancedToggleEl.addEventListener("click", () => {
    applyAdvancedSettingsOpen(!state.advancedSettingsOpen);
  });
}

aiTabEls.forEach((button) => {
  button.addEventListener("click", () => {
    applyActiveAiPanel(button.dataset.aiTab);
  });
});

if (historySearchInputEl) {
  historySearchInputEl.addEventListener("input", () => {
    setHistorySearchQuery(historySearchInputEl.value);
  });
}

if (saveBtn) {
  saveBtn.addEventListener("click", () => {
    saveCurrentHistory();
  });
}

if (captureScreenshotsEnabledEl) {
  const meetingCaptureEnabled = document.querySelector("#meetingCaptureEnabled");
  meetingCaptureEnabled.checked = state.captureScreenshotsEnabled;
  meetingCaptureEnabled.addEventListener("change", () => applyCaptureScreenshotsEnabled(meetingCaptureEnabled.checked));
  captureScreenshotsEnabledEl.addEventListener("change", () => {
    applyCaptureScreenshotsEnabled(captureScreenshotsEnabledEl.checked);
    meetingCaptureEnabled.checked = captureScreenshotsEnabledEl.checked;
  });
}

if (screenshotDiffSkipEnabledEl) {
  screenshotDiffSkipEnabledEl.addEventListener("change", () => {
    applyScreenshotDiffSkipEnabled(screenshotDiffSkipEnabledEl.checked);
  });
}

if (showTranscriptAudioEnabledEl) {
  showTranscriptAudioEnabledEl.addEventListener("change", () => {
    applyShowTranscriptAudioEnabled(showTranscriptAudioEnabledEl.checked);
  });
}

document.addEventListener("keydown", (event) => {
  const modal = topmostModal();
  if (modal && event.key === "Tab") {
    trapModalFocus(event, modal);
    return;
  }
  if (modal && event.key === "Escape") {
    event.preventDefault();
    event.stopPropagation();
    if (modal === runtimeUi.screenshotModalEl) {
      hideScreenshotModal();

    }
    return;
  }
  if (event.key === "Escape") {
    applyHistoryDrawerOpen(false);
  }
});

if (summaryPromptToggleBtn) {
  summaryPromptToggleBtn.addEventListener("click", () => {
    applySummaryPromptEditorOpen(!state.summaryPromptEditorOpen);
  });
}

window.addEventListener("resize", () => {
  updateWorkspaceGridTemplate();
  if (window.innerWidth > 1100) {
    applyHistoryDrawerOpen(false);
  }
  applyHistoryCollapsed(state.historyCollapsed, { persist: false });
  applyActiveAiPanel(state.activeAiPanel);
});

window.addEventListener("pagehide", () => {
  state.summaryController?.abort("page_hidden");
  state.proofreadController?.abort("page_hidden");
  state.historyListController?.abort("page_hidden");
  state.historyDetailController?.abort("page_hidden");
});

setupWorkspaceResizers();
setupPanelToggles();

try {
  const savedRatios = readStoredJson("whistx_workspace_ratios", null);
  if (savedRatios) {
    applyWorkspaceRatios(savedRatios.left, savedRatios.center, savedRatios.right, { persist: false });
  } else {
    applyWorkspaceRatios(state.panelLeftRatio, state.panelCenterRatio, state.panelRightRatio, { persist: false });
  }
} catch {
  applyWorkspaceRatios(state.panelLeftRatio, state.panelCenterRatio, state.panelRightRatio, { persist: false });
}

try {
  const savedCollapsed = readStoredJson("whistx_panel_collapsed", null);
  if (savedCollapsed) {
    applyPanelCollapseState("transcript", !!savedCollapsed.transcript, { persist: false });
    applyPanelCollapseState("proofread", !!savedCollapsed.proofread, { persist: false });
    applyPanelCollapseState("summary", !!savedCollapsed.summary, { persist: false });
  }
} catch {
  // ignore
}

try {
  const savedHistoryCollapsed = readStoredValue("whistx_history_collapsed", null);
  applyHistoryCollapsed(savedHistoryCollapsed === "1", { persist: false });
} catch {
  applyHistoryCollapsed(false, { persist: false });
}

try {
  const savedCaptureScreenshots = readStoredValue("whistx_capture_screenshots_enabled", null);
  applyCaptureScreenshotsEnabled(savedCaptureScreenshots !== "0", { persist: false });
} catch {
  applyCaptureScreenshotsEnabled(true, { persist: false });
}

try {
  const savedScreenshotDiffSkip = readStoredValue("whistx_screenshot_diff_skip_enabled", null);
  applyScreenshotDiffSkipEnabled(savedScreenshotDiffSkip !== "0", { persist: false });
} catch {
  applyScreenshotDiffSkipEnabled(true, { persist: false });
}

try {
  const savedShowTranscriptAudio = readStoredValue("whistx_show_transcript_audio_enabled", null);
  applyShowTranscriptAudioEnabled(savedShowTranscriptAudio !== "0", { persist: false });
} catch {
  applyShowTranscriptAudioEnabled(true, { persist: false });
}

copyBtn.addEventListener("click", () => {
  copyAll();
});

if (copyProofreadBtn) {
  copyProofreadBtn.addEventListener("click", () => {
    copyProofread();
  });
}

clearBtn.addEventListener("click", () => {
  clearView();
});

// Download link feedback
[dlTxt, dlJsonl, dlZip].forEach((link) => {
  link.addEventListener("click", (event) => {
    const href = link.getAttribute("href");
    if (link.getAttribute("aria-disabled") === "true" || !href) {
      event.preventDefault();
      showToast(
        isRecordingInteractionLocked()
          ? "録音の最終処理が完了してから書き出してください"
          : "書き出せる文字起こしがありません",
        "error"
      );
      return;
    }
    link.classList.add("is-downloaded");
    setTimeout(() => link.classList.remove("is-downloaded"), 800);
  });
});

/* --------------------------------------------------------------------------
   Theme Toggle - Light/Dark Mode
   -------------------------------------------------------------------------- */

if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", toggleTheme);
}

if (sharedVocabularySaveBtn) {
  sharedVocabularySaveBtn.addEventListener("click", () => {
    void saveSharedGlossary();
  });
}

// Initialize theme on load
initTheme();
handleAuthErrorFromLocation();

renderPromptTemplateButtons(state.promptTemplates);
buildRuntimeUi();
setAppLocked(true);

// Initialize empty states
updateSegmentCount();
updateDownloadLinks();
setSummary("", "未実行");
setProofread("", "未実行");
renderAuthState();
updateSharedVocabularyMeta();
applyProofreadMode(proofreadModeEl?.value || "proofread");
if (summaryBtnLabelEl) {
  summaryBtnLabelEl.textContent = "要約する";
}
applyAdvancedSettingsOpen(false);
applySummaryPromptEditorOpen(false);
applyActiveAiPanel(state.activeAiPanel);
applyHistoryDrawerOpen(false);
setStatus("idle");
document.documentElement.dataset.whistxReady = "true";

// Show initial empty state for transcript
if (logEl && !logEl.querySelector(".log-row")) {
  renderEmptyTranscriptState();
}

(() => {
  let initial = CHUNK_DEFAULT_SECONDS;
  try {
    const saved = readStoredValue("whistx_chunk_seconds", null);
    if (saved) initial = normalizeChunkSeconds(saved);
  } catch {
    // ignore
  }
  applyChunkSeconds(initial);
})();

(() => {
  let initial = "mic";
  try {
    const saved = readStoredValue("whistx_audio_source", null);
    if (saved) initial = normalizeAudioSource(saved);
  } catch {
    // ignore
  }
  applyAudioSource(initial);
})();

(() => {
  let initial = true;
  try {
    initial = readStoredValue("whistx_auto_gain_enabled", "1") !== "0";
  } catch {
    // ignore
  }
  applyAutoGainEnabled(initial, { persist: false });
})();

(() => {
  let initial = true;
  try {
    const saved = readStoredValue("whistx_diarization_enabled", null);
    if (saved !== null) {
      initial = saved === "1" || saved.toLowerCase() === "true";
    }
  } catch {
    // ignore
  }
  applyDiarizationEnabled(initial, { persist: false });
})();

(() => {
  let hasSaved = false;
  let mode = "auto";
  let count = state.diarizationSpeakerCount;
  let min = state.diarizationMinSpeakers;
  let max = state.diarizationMaxSpeakers;

  try {
    const savedMode = readStoredValue("whistx_diarization_speaker_mode", null);
    const savedCount = readStoredValue("whistx_diarization_speaker_count", null);
    const savedMin = readStoredValue("whistx_diarization_min_speakers", null);
    const savedMax = readStoredValue("whistx_diarization_max_speakers", null);

    if (savedMode !== null || savedCount !== null || savedMin !== null || savedMax !== null) {
      hasSaved = true;
      mode = savedMode || mode;
      if (savedCount !== null) count = Number(savedCount);
      if (savedMin !== null) min = Number(savedMin);
      if (savedMax !== null) max = Number(savedMax);
    }
  } catch {
    // ignore
  }

  state.hasSavedDiarizationSpeakerSettings = hasSaved;
  applyDiarizationSpeakerSettings(
    {
      mode,
      count,
      min,
      max,
    },
    { persist: false }
  );
})();

meetingWorkspace = createMeetingWorkspace({
  getSource: currentMeetingSource,
  getAccess: () => !state.auth.authenticated ? "login" : !state.meetingInsights ? "unavailable" : "ready",
  onNavigate: navigateMeetingSource,
  onRecap: (text, meta) => { setSummary(text, meta); markWorkspaceDirty(); },
  onImage: showScreenshotModal,
  onBusy: () => updateSaveControls(),
});
meetingWorkspace.syncSource();
document.querySelector("#refineAudioBtn")?.addEventListener("click", refineMeetingAudio);
document.querySelector("#retryLiveStop")?.addEventListener("click", finalizeLiveRecording);
document.querySelector("#downloadPendingAudio")?.addEventListener("click", () => state.liveCapture?.downloadPending());
const liveToggle = document.querySelector("#liveTranscriptionEnabled");
if (liveToggle) {
  liveToggle.checked = readStoredValue("whistx_live_transcription", "1") !== "0";
  liveToggle.addEventListener("change", () => writeStoredValue("whistx_live_transcription", liveToggle.checked ? "1" : "0"));
}

(async () => {
  logClientEvent("bootstrap.start");
  await loadCapabilities();
  await loadSharedGlossary();
  await loadAuthState();
  logClientEvent("bootstrap.done");
})();
