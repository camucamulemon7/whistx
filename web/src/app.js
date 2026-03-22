import { fetchCapabilities } from "./capabilities/api.js";
import { fetchAuthState, loginRequest, bootstrapAdminRequest, registerRequest, logoutRequest, fetchPendingUsers as fetchPendingUsersRequest, approvePendingUserRequest } from "./auth/api.js";
import { canUseWorkspace as canUseWorkspaceForAuth, persistGuestMode, readGuestMode, serializeUserLabel } from "./auth/session.js";
import { deleteHistoryRequest, fetchHistoryList as fetchHistoryListRequest, fetchHistoryDetail, saveHistoryRequest } from "./history/api.js";
import { fetchSharedGlossary as fetchSharedGlossaryRequest, saveSharedGlossary as saveSharedGlossaryRequest } from "./api/glossary.js";
import { fetchJson } from "./api/client.js";
import { applyHistoryDetailPayload, applyHistoryListPayload, clearHistoryState } from "./history/state.js";
import { readStoredJson, readStoredValue, writeStoredValue } from "./state/storage.js";
import { normalizeTheme, resolveInitialTheme, themeMetaColor } from "./ui/theme.js";

const $ = (selector) => document.querySelector(selector);

const statusTextEl = $("#statusText");
const connCountEl = $("#connCount");
const bannersContainerEl = $("#bannersContainer");
const brandTitleEl = $("#brandTitle");
const brandTaglineEl = $("#brandTagline");
const themeToggleBtn = $("#themeToggle");
const sidebarToggleBtn = $("#sidebarToggle");
const sidebarCloseBtn = $("#sidebarClose");
const sidebarBackdropEl = $("#sidebarBackdrop");
const sidePanelEl = $("#sidePanel");
const audioLevelIndicatorEl = $("#audioLevelIndicator");
const audioLevelMatrixEl = $("#audioLevelMatrix");

const languageEl = $("#language");
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
const helpBtn = $("#helpBtn");
const loginBtn = $("#loginBtn");
const logoutBtn = $("#logoutBtn");
const historyCollapseBtn = $("#historyCollapseBtn");
const historyDrawerCloseEl = document.querySelector("#historyDrawerClose");
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
const adminQueueModalEl = $("#adminQueueModal");
const adminQueueCloseEl = $("#adminQueueClose");
const adminPendingListEl = $("#adminPendingList");
const helpModalEl = $("#helpModal");
const helpModalCloseEl = $("#helpModalClose");
const helpModalFrameEl = $("#helpModalFrame");

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

const VAD_SAMPLE_MS = 80;
const VAD_MIN_SPEECH_RATIO = 0.025;
const VAD_MIN_ACTIVE_MS = 120;
const VAD_SEGMENT_MIN_MS = 12_000;
const VAD_SEGMENT_MAX_SILENCE_MS = 1_100;
const VAD_SEGMENT_MIN_SILENCE_MS = 450;
const VAD_SOFT_CUT_GRACE_MS = 6_000;
const VAD_NOISE_FLOOR_WARMUP_MS = 1_800;
const VAD_NOISE_FLOOR_EWMA = 0.18;
const VAD_NOISE_FLOOR_OFFSET = {
  mic: 0.0012,
  display: 0.0008,
  both: 0.0010,
};
const VAD_NOISE_FLOOR_MULTIPLIER = {
  mic: 3.0,
  display: 3.4,
  both: 3.2,
};
const VAD_SOURCE_CUT_POLICY = {
  mic: {
    minSilenceMs: 450,
    maxSilenceMs: 1_100,
    silenceRatio: 0.18,
    minSegmentRatio: 0.45,
  },
  display: {
    minSilenceMs: 550,
    maxSilenceMs: 1_400,
    silenceRatio: 0.22,
    minSegmentRatio: 0.50,
  },
  both: {
    minSilenceMs: 500,
    maxSilenceMs: 1_500,
    silenceRatio: 0.20,
    minSegmentRatio: 0.50,
  },
};
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
const SCREENSHOT_DIFF_CHANGED_RATIO_THRESHOLD = 0.015;
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
const SCREENSHOT_MIN_INTERVAL_MS = 18_000;
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
  bodyScrollLocks: 0,
};

const state = {
  ws: null,
  stream: null,
  micStream: null,
  displayStream: null,
  recorder: null,
  recorderOptions: null,
  recorderMimeType: "audio/webm",
  chunkTimer: null,
  finalizingStop: false,
  recording: false,
  recordingAudioSource: "mic",
  recordingRequestedAudioSource: "mic",
  recordingFallbackReason: "",
  recordingStartedAt: 0,
  logAutoScrollEnabled: true,
  screenshotZoom: 1,
  screenshotBaseWidth: 0,
  screenshotBaseHeight: 0,
  screenshotDragging: false,
  screenshotDragPointerId: null,
  screenshotDragStartX: 0,
  screenshotDragStartY: 0,
  screenshotDragScrollLeft: 0,
  screenshotDragScrollTop: 0,
  recordedChunkCount: 0,
  runtimeSessionId: "",
  runtimeSessionToken: "",
  savedHistoryId: null,
  viewingHistoryId: null,
  saveInFlight: false,
  seq: 0,
  offsetMs: 0,
  chunkMs: CHUNK_DEFAULT_SECONDS * 1000,
  pendingSendChain: Promise.resolve(),
  log: [],
  segments: [],
  summary: "",
  proofread: "",
  asrAvailable: true,
  proofreadAvailable: true,
  proofreadInFlight: false,
  proofreadMode: "proofread",
  activeAiPanel: "proofread",
  advancedSettingsOpen: false,
  latestStatus: "idle",
  sidebarOpen: false,
  historyCollapsed: false,
  activeResizer: null,
  panelLeftRatio: 1.35,
  panelCenterRatio: 0.95,
  panelRightRatio: 0.9,
  diarizationAvailable: true,
  diarizationEnabled: true,
  diarizationSpeakerMode: "auto",
  diarizationSpeakerCount: 2,
  diarizationMinSpeakers: 2,
  diarizationMaxSpeakers: 4,
  diarizationSpeakerCap: DIARIZATION_SPEAKER_MAX,
  hasSavedDiarizationSpeakerSettings: false,
  banners: [],
  audioContext: null,
  vadSource: null,
  vadAnalyser: null,
  vadBuffer: null,
  vadTimer: null,
  vadFrameCount: 0,
  vadSpeechFrameCount: 0,
  vadLastSpeechAt: 0,
  vadRmsThreshold: 0.01,
  vadNoiseFloor: null,
  vadNoiseFloorAt: 0,
  audioLevel: 0,
  audioLevelColumns: [],
  segmentStartedAt: 0,
  captureContext: null,
  captureSources: [],
  captureDestination: null,
  captureGainNode: null,
  captureMonitorAnalyser: null,
  captureMonitorBuffer: null,
  captureAutoGainTimer: null,
  captureAutoGainLevel: 1,
  captureAutoGainSmoothedRms: 0,
  displayCaptureVideo: null,
  screenshotCanvas: null,
  captureScreenshotsEnabled: true,
  screenshotDiffSkipEnabled: true,
  autoGainEnabled: true,
  screenshotDiffCanvas: null,
  previousScreenshotSignature: null,
  screenshotEncodeInFlight: false,
  screenshotLastCapturedAt: 0,
  screenshotSkippedCount: 0,
  pendingOutboundChunks: 0,
  maxObservedBacklog: 0,
  degradedCaptureMode: false,
  lastTelemetryBacklogBucket: "",
  lastScreenshotSkipReason: "",
  lastScreenshotSkipSentAt: 0,
  panelCollapsed: {
    transcript: false,
    proofread: false,
    summary: false,
  },
  summaryPromptEditorOpen: false,
  promptTemplates: [
    {
      id: "soc-design",
      label: "SoC設計テンプレート",
      content: DEFAULT_SOC_PROMPT_TEMPLATE,
    },
  ],
  sharedVocabulary: "",
  sharedVocabularyUpdatedAt: "",
  sharedVocabularyUpdatedBy: "",
  sharedVocabularySaving: false,
  auth: {
    authenticated: false,
    isGuest: false,
    user: null,
    bootstrapAdminRequired: false,
    pendingApprovalCount: 0,
    keycloakEnabled: false,
    keycloakButtonLabel: "Keycloakでログイン",
    historyRetentionDays: 7,
  },
  history: {
    items: [],
    loading: false,
    selectedId: null,
    total: 0,
    limit: 20,
    offset: 0,
    query: "",
  },
  selfSignupEnabled: false,
  adminPendingUsers: [],
};

function canUseWorkspace() {
  return canUseWorkspaceForAuth(state.auth);
}

function selectedLanguage() {
  const value = String(languageEl?.value || "").trim().toLowerCase();
  if (!value || value === "auto") {
    return null;
  }
  return value;
}

function formatStatusText(text) {
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

function setStatus(text) {
  const raw = String(text || "").trim();
  state.latestStatus = raw || "idle";
  statusTextEl.textContent = formatStatusText(raw);
  statusTextEl.dataset.state = String(raw || "idle").toLowerCase();
}

function applyBranding(title, tagline) {
  const cleanTitle = String(title || "").trim();
  const cleanTagline = String(tagline || "").trim();

  if (brandTitleEl && cleanTitle) {
    brandTitleEl.textContent = cleanTitle;
    document.title = cleanTitle;
  }

  if (brandTaglineEl && cleanTagline) {
    brandTaglineEl.textContent = cleanTagline;
  }
}

function setProofreadButtonBusy(busy) {
  if (!proofreadBtn) return;
  proofreadBtn.disabled = busy;
  if (proofreadBtnLabelEl) {
    proofreadBtnLabelEl.textContent = busy ? "生成中..." : "生成";
  }
  proofreadBtn.setAttribute("aria-busy", busy ? "true" : "false");
}

function normalizeProofreadMode(value) {
  if (value === "translate_ja" || value === "translate_en") return value;
  return "proofread";
}

function proofreadActionLabel() {
  const mode = normalizeProofreadMode(state.proofreadMode);
  if (mode === "translate_ja") return "日本語訳";
  if (mode === "translate_en") return "英語訳";
  return "校正";
}

function applyProofreadMode(value) {
  state.proofreadMode = normalizeProofreadMode(value);
  if (proofreadModeEl) {
    proofreadModeEl.value = state.proofreadMode;
  }
  if (proofreadBtn && !state.proofreadInFlight) {
    if (proofreadBtnLabelEl) {
      proofreadBtnLabelEl.textContent = "生成";
    }
    proofreadBtn.setAttribute("aria-label", `${proofreadActionLabel()}を開始`);
    proofreadBtn.title = proofreadActionLabel();
  }
}

function applySidebarOpen(value) {
  state.sidebarOpen = !!value;
  if (sidePanelEl) {
    sidePanelEl.classList.toggle("is-open", state.sidebarOpen);
    sidePanelEl.setAttribute("aria-hidden", state.sidebarOpen ? "false" : "true");
  }
  if (sidebarBackdropEl) {
    sidebarBackdropEl.hidden = !state.sidebarOpen;
    sidebarBackdropEl.classList.toggle("is-open", state.sidebarOpen);
  }
  if (sidebarToggleBtn) {
    sidebarToggleBtn.setAttribute("aria-label", state.sidebarOpen ? "詳細設定を閉じる" : "詳細設定を開く");
    sidebarToggleBtn.title = state.sidebarOpen ? "詳細設定を閉じる" : "詳細設定";
  }
}

function applyHistoryDrawerOpen(open) {
  if (!historyRailEl) return;
  const isMobile = window.innerWidth <= 1100;
  const isOpen = !!open && isMobile;
  historyRailEl.classList.toggle("is-open", isOpen);
  historyRailEl.setAttribute("aria-hidden", isMobile ? (isOpen ? "false" : "true") : "false");
  document.body.classList.toggle("is-history-drawer-open", isOpen);
  updateHistoryControls();
}

function applyHistoryCollapsed(value, options = {}) {
  const persist = options.persist !== false;
  state.historyCollapsed = !!value;
  const isDesktop = window.innerWidth > 1100;
  if (workspaceShellEl) {
    workspaceShellEl.classList.toggle("is-history-collapsed", isDesktop && state.historyCollapsed);
  }
  if (historyRailEl) {
    historyRailEl.classList.toggle("is-collapsed", isDesktop && state.historyCollapsed);
  }
  if (persist) {
    try {
      localStorage.setItem("whistx_history_collapsed", state.historyCollapsed ? "1" : "0");
    } catch {
      // ignore
    }
  }
  updateHistoryControls();
}

function updateHistoryControls() {
  const isMobile = window.innerWidth <= 1100;
  if (historyCollapseBtn) {
    historyCollapseBtn.hidden = isMobile;
    historyCollapseBtn.classList.toggle("is-collapsed", !isMobile && state.historyCollapsed);
    historyCollapseBtn.setAttribute("aria-label", state.historyCollapsed ? "履歴を展開" : "履歴をたたむ");
    historyCollapseBtn.title = state.historyCollapsed ? "展開" : "たたむ";
  }
}

function applySummaryPromptEditorOpen(value) {
  state.summaryPromptEditorOpen = !!value;
  if (summaryPromptEditorEl) {
    summaryPromptEditorEl.hidden = !state.summaryPromptEditorOpen;
    summaryPromptEditorEl.classList.toggle("is-open", state.summaryPromptEditorOpen);
  }
  if (summaryPromptToggleBtn) {
    summaryPromptToggleBtn.setAttribute("aria-expanded", state.summaryPromptEditorOpen ? "true" : "false");
    summaryPromptToggleBtn.textContent = state.summaryPromptEditorOpen ? "要約方針を閉じる" : "要約方針";
  }
}

function applyAdvancedSettingsOpen(open) {
  state.advancedSettingsOpen = !!open;
  if (inputAdvancedSettingsEl) {
    inputAdvancedSettingsEl.hidden = !state.advancedSettingsOpen;
    inputAdvancedSettingsEl.classList.toggle("is-open", state.advancedSettingsOpen);
  }
  if (settingsAdvancedToggleEl) {
    settingsAdvancedToggleEl.setAttribute("aria-expanded", state.advancedSettingsOpen ? "true" : "false");
    settingsAdvancedToggleEl.textContent = state.advancedSettingsOpen ? "話者設定を閉じる" : "話者設定";
  }
}

function syncAiResponsiveState() {
  if (window.innerWidth <= 1439) {
    ["proofread", "summary"].forEach((key) => {
      document.querySelector(`.${key}-panel`)?.classList.remove("is-collapsed");
      document.querySelector(`[data-panel-toggle="${key}"]`)?.classList.remove("is-collapsed");
    });
    return;
  }
  applyPanelCollapseState("proofread", !!state.panelCollapsed.proofread, { persist: false });
  applyPanelCollapseState("summary", !!state.panelCollapsed.summary, { persist: false });
}

function applyActiveAiPanel(panel) {
  const nextPanel = panel === "summary" ? "summary" : "proofread";
  state.activeAiPanel = nextPanel;
  if (proofreadPanelEl) {
    proofreadPanelEl.classList.toggle("is-ai-active", nextPanel === "proofread");
  }
  if (summaryPanelEl) {
    summaryPanelEl.classList.toggle("is-ai-active", nextPanel === "summary");
  }
  aiTabEls.forEach((button) => {
    const active = button.dataset.aiTab === nextPanel;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  syncAiResponsiveState();
}

function applyWorkspaceRatios(left, center, right, options = {}) {
  const persist = options.persist !== false;
  const safeLeft = Math.max(0.65, Math.min(2.4, Number(left) || state.panelLeftRatio));
  const safeCenter = Math.max(0.6, Math.min(2.2, Number(center) || state.panelCenterRatio));
  const safeRight = Math.max(0.6, Math.min(2.2, Number(right) || state.panelRightRatio));

  state.panelLeftRatio = safeLeft;
  state.panelCenterRatio = safeCenter;
  state.panelRightRatio = safeRight;

  if (workspacePanelsEl) {
    workspacePanelsEl.style.setProperty("--panel-left", `${safeLeft}fr`);
    workspacePanelsEl.style.setProperty("--panel-center", `${safeCenter}fr`);
    workspacePanelsEl.style.setProperty("--panel-right", `${safeRight}fr`);
  }
  updateWorkspaceGridTemplate();

  if (persist) {
    try {
      localStorage.setItem(
        "whistx_workspace_ratios",
        JSON.stringify({ left: safeLeft, center: safeCenter, right: safeRight })
      );
    } catch {
      // ignore
    }
  }
}

function updateWorkspaceGridTemplate() {
  if (!workspacePanelsEl) return;
  if (window.innerWidth <= 1100) {
    workspacePanelsEl.style.gridTemplateColumns = "1fr";
    return;
  }
  if (window.innerWidth <= 1439) {
    const left = state.panelCollapsed.transcript ? "92px" : `minmax(320px, ${state.panelLeftRatio + 0.2}fr)`;
    const right = `minmax(280px, ${state.panelCenterRatio + state.panelRightRatio}fr)`;
    workspacePanelsEl.style.gridTemplateColumns = `${left} ${right}`;
    return;
  }

  const left = state.panelCollapsed.transcript ? "88px" : `minmax(280px, ${state.panelLeftRatio}fr)`;
  const center = state.panelCollapsed.proofread ? "88px" : `minmax(240px, ${state.panelCenterRatio}fr)`;
  const right = state.panelCollapsed.summary ? "88px" : `minmax(240px, ${state.panelRightRatio}fr)`;
  workspacePanelsEl.style.gridTemplateColumns = `${left} 10px ${center} 10px ${right}`;
}

function applyPanelCollapseState(panel, collapsed, options = {}) {
  const persist = options.persist !== false;
  const key = panel === "proofread" || panel === "summary" ? panel : "transcript";
  state.panelCollapsed[key] = !!collapsed;

  const panelEl = document.querySelector(`.${key}-panel`);
  if (panelEl) {
    panelEl.classList.toggle("is-collapsed", !!collapsed);
  }

  const toggleBtn = document.querySelector(`[data-panel-toggle="${key}"]`);
  if (toggleBtn) {
    toggleBtn.classList.toggle("is-collapsed", !!collapsed);
    const labelMap = { transcript: "文字起こし", proofread: "校正", summary: "要約" };
    const label = labelMap[key] || "パネル";
    toggleBtn.setAttribute("aria-label", collapsed ? `${label}を展開` : `${label}をたたむ`);
    toggleBtn.title = collapsed ? "展開" : "たたむ";
  }

  if (persist) {
    try {
      localStorage.setItem("whistx_panel_collapsed", JSON.stringify(state.panelCollapsed));
    } catch {
      // ignore
    }
  }
  updateWorkspaceGridTemplate();
}

function applyCaptureScreenshotsEnabled(value, options = {}) {
  const persist = options.persist !== false;
  const enabled = !!value;
  state.captureScreenshotsEnabled = enabled;
  if (captureScreenshotsEnabledEl) {
    captureScreenshotsEnabledEl.checked = enabled;
  }
  if (captureScreenshotsStateTextEl) {
    captureScreenshotsStateTextEl.textContent = enabled ? "ON" : "OFF";
  }
  if (persist) {
    try {
      localStorage.setItem("whistx_capture_screenshots_enabled", enabled ? "1" : "0");
    } catch {
      // ignore
    }
  }
}

function applyScreenshotDiffSkipEnabled(value, options = {}) {
  const persist = options.persist !== false;
  const enabled = !!value;
  state.screenshotDiffSkipEnabled = enabled;
  state.previousScreenshotSignature = null;
  if (screenshotDiffSkipEnabledEl) {
    screenshotDiffSkipEnabledEl.checked = enabled;
  }
  if (screenshotDiffSkipStateTextEl) {
    screenshotDiffSkipStateTextEl.textContent = enabled ? "ON" : "OFF";
  }
  if (persist) {
    try {
      localStorage.setItem("whistx_screenshot_diff_skip_enabled", enabled ? "1" : "0");
    } catch {
      // ignore
    }
  }
}

function applyAutoGainEnabled(value, options = {}) {
  const persist = options.persist !== false;
  const enabled = !!value;
  state.autoGainEnabled = enabled;
  if (autoGainEnabledEl) {
    autoGainEnabledEl.checked = enabled;
  }
  if (autoGainStateTextEl) {
    autoGainStateTextEl.textContent = enabled ? "ON" : "OFF";
  }
  if (persist) {
    try {
      localStorage.setItem("whistx_auto_gain_enabled", enabled ? "1" : "0");
    } catch {
      // ignore
    }
  }
  updateCaptureAutoGainState();
}

function ensureRuntimeUi() {
  if (runtimeUi.injectedStyle) return;

  runtimeUi.injectedStyle = document.createElement("style");
  runtimeUi.injectedStyle.id = "whistx-runtime-ui";
  runtimeUi.injectedStyle.textContent = `
    body.whistx-auth-locked {
      overflow: hidden;
    }

    body.whistx-auth-locked .app {
      filter: blur(10px) saturate(0.85);
      pointer-events: none;
      user-select: none;
    }

    .whistx-auth-overlay {
      position: fixed;
      inset: 0;
      z-index: 2000;
      display: grid;
      place-items: center;
      padding: 24px;
      background:
        radial-gradient(circle at top left, rgba(47, 125, 115, 0.22), transparent 36%),
        radial-gradient(circle at bottom right, rgba(138, 90, 31, 0.16), transparent 34%),
        rgba(10, 10, 10, 0.68);
      backdrop-filter: blur(14px);
    }

    .whistx-auth-card {
      width: min(440px, calc(100vw - 32px));
      padding: 28px;
      border-radius: 24px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-primary);
      box-shadow: var(--shadow-2xl);
      display: grid;
      gap: 16px;
    }

    .whistx-auth-card h2 {
      font-size: 28px;
      letter-spacing: -0.03em;
    }

    .whistx-auth-card p {
      color: var(--text-secondary);
      line-height: 1.6;
    }

    .whistx-auth-field {
      display: grid;
      gap: 8px;
    }

    .whistx-auth-field input {
      width: 100%;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid var(--border-primary);
      background: var(--bg-secondary);
      color: var(--text-primary);
      font-size: 15px;
    }

    .whistx-auth-actions {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .whistx-auth-actions .btn-action,
    .whistx-auth-actions .btn-secondary {
      flex: 1;
      justify-content: center;
    }

    .whistx-history-drawer,
    .history-rail {
      display: flex;
      flex-direction: column;
      gap: 14px;
      min-height: 100%;
      padding: 20px;
      position: relative;
      overflow: hidden;
    }

    .whistx-history-drawer::before,
    .history-rail::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(47, 125, 115, 0.06), transparent 20%);
      pointer-events: none;
    }

    .whistx-history-header,
    .history-rail-header {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 12px;
      position: relative;
      z-index: 1;
    }

    .whistx-history-title,
    .history-rail-title {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .whistx-history-subtitle,
    .history-rail-description {
      color: var(--text-secondary);
      font-size: 13px;
      margin-top: 4px;
    }

    .whistx-history-toolbar {
      display: grid;
      gap: 10px;
      position: relative;
      z-index: 1;
    }

    .whistx-history-search,
    #historySearchInput {
      width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--border-primary);
      background: var(--bg-elevated);
      color: var(--text-primary);
    }

    .whistx-history-list,
    #historyList {
      display: grid;
      gap: 12px;
      overflow: auto;
      padding-right: 2px;
      position: relative;
      z-index: 1;
    }

    .whistx-history-item,
    .history-item {
      display: grid;
      gap: 8px;
      padding: 14px;
      border-radius: 18px;
      border: 1px solid var(--border-primary);
      background: var(--bg-elevated);
      text-align: left;
      cursor: pointer;
      transition: transform 150ms var(--ease-out-quart), border-color 150ms var(--ease-out-quart), box-shadow 150ms var(--ease-out-quart);
    }

    .whistx-history-item:hover,
    .history-item:hover {
      transform: translateY(-1px);
      border-color: var(--accent-300);
      box-shadow: var(--shadow-md);
    }

    .whistx-history-item.is-active,
    .history-item.is-active {
      border-color: var(--accent-500);
      box-shadow: var(--shadow-lg);
    }

    .whistx-history-item-title,
    .history-item-title {
      font-size: 15px;
      font-weight: 700;
      line-height: 1.4;
    }

    .whistx-history-item-meta,
    .history-item-meta {
      color: var(--text-tertiary);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }

    .whistx-history-item-preview,
    .history-item-preview {
      color: var(--text-secondary);
      font-size: 13px;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .whistx-history-empty,
    #historyEmpty {
      color: var(--text-secondary);
      font-size: 13px;
      padding: 8px 2px 0;
      position: relative;
      z-index: 1;
    }

    .whistx-summary-copy-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      white-space: nowrap;
    }

    .whistx-image-modal {
      position: fixed;
      inset: 0;
      z-index: 2100;
      display: none;
      place-items: center;
      padding: 24px;
      background: rgba(10, 10, 10, 0.76);
      backdrop-filter: blur(10px);
    }

    .whistx-image-modal.is-open {
      display: grid;
    }

    .whistx-image-modal-card {
      position: relative;
      max-width: min(96vw, 1400px);
      max-height: 92vh;
      padding: 14px;
      border-radius: 20px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-primary);
      box-shadow: var(--shadow-2xl);
    }

    .whistx-image-modal-card img {
      display: block;
      max-width: 100%;
      max-height: calc(92vh - 28px);
      border-radius: 14px;
      object-fit: contain;
    }

    .whistx-image-modal-close {
      position: absolute;
      top: 12px;
      right: 12px;
      width: 34px;
      height: 34px;
      border-radius: 999px;
      border: 1px solid var(--border-primary);
      background: var(--bg-elevated);
      color: var(--text-primary);
      display: grid;
      place-items: center;
      cursor: pointer;
    }

    .whistx-image-modal-close:hover {
      background: var(--bg-secondary);
    }

    .whistx-auth-hidden,
    .whistx-history-hidden {
      display: none !important;
    }

    .whistx-history-drawer .side-panel-input,
    .history-rail .side-panel-input {
      width: 100%;
    }

    .whistx-history-drawer .whistx-history-footer,
    .history-rail .whistx-history-footer {
      margin-top: auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      position: relative;
      z-index: 1;
    }

    .whistx-history-drawer .whistx-history-user,
    .history-rail .whistx-history-user {
      color: var(--text-secondary);
      font-size: 13px;
    }

    @media (max-width: 1100px) {
      .whistx-history-drawer {
        min-height: auto;
      }

      .whistx-auth-card {
        width: min(480px, calc(100vw - 24px));
      }
    }
  `;
  document.head.appendChild(runtimeUi.injectedStyle);
}

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

function setAppLocked(locked) {
  runtimeUi.appLocked = !!locked;
  document.body.classList.toggle("whistx-auth-locked", runtimeUi.appLocked);
  if (runtimeUi.loginOverlayEl) {
    runtimeUi.loginOverlayEl.hidden = !runtimeUi.appLocked;
  }
  if (startBtn) {
    startBtn.disabled = runtimeUi.appLocked;
  }
  if (summaryBtn) {
    summaryBtn.disabled = runtimeUi.appLocked;
  }
  if (proofreadBtn) {
    proofreadBtn.disabled = runtimeUi.appLocked;
  }
  if (clearBtn) {
    clearBtn.disabled = runtimeUi.appLocked;
  }
  if (saveBtn && runtimeUi.appLocked) {
    saveBtn.disabled = true;
  }
}

function hasOpenBlockingModal() {
  return [runtimeUi.screenshotModalEl, helpModalEl, adminQueueModalEl].some((element) => element && !element.hidden);
}

function syncBodyScrollLock() {
  const locked = hasOpenBlockingModal();
  runtimeUi.bodyScrollLocks = locked ? 1 : 0;
  document.body.style.overflow = locked ? "hidden" : "";
  document.body.classList.toggle("is-modal-open", locked);
}

function lockBodyScroll() {
  syncBodyScrollLock();
}

function unlockBodyScroll() {
  syncBodyScrollLock();
}

function setHistorySearchQuery(value) {
  state.history.query = String(value || "").trim();
  state.history.offset = 0;
  if (state.historySearchTimer) {
    clearTimeout(state.historySearchTimer);
    state.historySearchTimer = null;
  }
  state.historySearchTimer = setTimeout(() => {
    state.historySearchTimer = null;
    loadHistoryList();
  }, HISTORY_SEARCH_DEBOUNCE_MS);
}

function formatModeLabel(mode) {
  if (mode === "both") return "両方";
  if (mode === "display") return "画面共有";
  return "マイク";
}

function updateRecordingTelemetry() {
  if (!recordTelemetryEl) return;
  if (!state.recording) {
    recordTelemetryEl.hidden = true;
    recordTelemetryEl.textContent = "";
    return;
  }

  const requested = formatModeLabel(state.recordingRequestedAudioSource || state.recordingAudioSource || "mic");
  const effective = formatModeLabel(state.recordingAudioSource || "mic");
  const sourceText = requested === effective ? `入力: ${effective}` : `入力: ${requested} → ${effective}`;
  const vadFrames = Math.max(0, state.vadFrameCount);
  const speechRatio = vadFrames > 0 ? state.vadSpeechFrameCount / vadFrames : 0;
  const vadText = state.vadAnalyser ? `VAD ${(speechRatio * 100).toFixed(0)}%` : "VAD n/a";
  const gainText = state.captureGainNode ? `GAIN ${state.captureAutoGainLevel.toFixed(1)}x` : "GAIN 1.0x";
  const chunkText = state.recordedChunkCount > 0 ? `CHUNK ${state.recordedChunkCount}` : "CHUNK 0";
  const elapsedMs = state.recordingStartedAt ? Math.max(0, performance.now() - state.recordingStartedAt) : 0;
  const elapsedText = `TIME ${(elapsedMs / 1000).toFixed(1)}s`;
  const fallbackText = state.recordingFallbackReason ? `FALLBACK ${state.recordingFallbackReason}` : "";

  recordTelemetryEl.hidden = false;
  recordTelemetryEl.textContent = [sourceText, vadText, gainText, chunkText, elapsedText, fallbackText].filter(Boolean).join(" / ");
}

function clampScreenshotZoom(value) {
  return Math.min(SCREENSHOT_ZOOM_MAX, Math.max(SCREENSHOT_ZOOM_MIN, Number(value) || SCREENSHOT_ZOOM_MIN));
}

function updateScreenshotZoomUi() {
  if (!screenshotModalImageEl || !screenshotModalViewportEl || !screenshotModalStageEl) return;

  const zoom = clampScreenshotZoom(state.screenshotZoom);
  const viewportWidth = Math.max(1, screenshotModalViewportEl.clientWidth || 1);
  const viewportHeight = Math.max(1, screenshotModalViewportEl.clientHeight || 1);
  const baseWidth = Math.max(1, state.screenshotBaseWidth || viewportWidth);
  const baseHeight = Math.max(1, state.screenshotBaseHeight || viewportHeight);
  const renderedWidth = Math.max(1, Math.round(baseWidth * zoom));
  const renderedHeight = Math.max(1, Math.round(baseHeight * zoom));

  screenshotModalStageEl.style.width = Math.max(viewportWidth, renderedWidth) + "px";
  screenshotModalStageEl.style.height = Math.max(viewportHeight, renderedHeight) + "px";
  screenshotModalImageEl.style.width = renderedWidth + "px";
  screenshotModalImageEl.style.height = renderedHeight + "px";
  screenshotModalViewportEl.classList.toggle("is-zoomed", zoom > SCREENSHOT_ZOOM_MIN);
  screenshotModalViewportEl.classList.toggle("is-dragging", state.screenshotDragging);

  if (screenshotZoomOutBtnEl) screenshotZoomOutBtnEl.disabled = zoom <= SCREENSHOT_ZOOM_MIN;
  if (screenshotZoomInBtnEl) screenshotZoomInBtnEl.disabled = zoom >= SCREENSHOT_ZOOM_MAX;
  if (screenshotZoomResetBtnEl) screenshotZoomResetBtnEl.textContent = Math.round(zoom * 100) + "%";
}

function recalculateScreenshotBaseSize() {
  if (!screenshotModalImageEl || !screenshotModalViewportEl) return;
  if (!screenshotModalImageEl.naturalWidth || !screenshotModalImageEl.naturalHeight) return;

  const viewportWidth = Math.max(1, screenshotModalViewportEl.clientWidth || 1);
  const viewportHeight = Math.max(1, screenshotModalViewportEl.clientHeight || 1);
  const fitScale = Math.min(viewportWidth / screenshotModalImageEl.naturalWidth, viewportHeight / screenshotModalImageEl.naturalHeight);

  state.screenshotBaseWidth = Math.max(1, Math.round(screenshotModalImageEl.naturalWidth * fitScale));
  state.screenshotBaseHeight = Math.max(1, Math.round(screenshotModalImageEl.naturalHeight * fitScale));
}

function centerScreenshotViewport() {
  if (!screenshotModalViewportEl) return;
  const maxScrollLeft = Math.max(0, screenshotModalViewportEl.scrollWidth - screenshotModalViewportEl.clientWidth);
  const maxScrollTop = Math.max(0, screenshotModalViewportEl.scrollHeight - screenshotModalViewportEl.clientHeight);
  screenshotModalViewportEl.scrollLeft = maxScrollLeft / 2;
  screenshotModalViewportEl.scrollTop = maxScrollTop / 2;
}

function setScreenshotZoom(nextZoom, { recenter = false } = {}) {
  if (!screenshotModalViewportEl) return;

  const previousScrollableX = Math.max(0, screenshotModalViewportEl.scrollWidth - screenshotModalViewportEl.clientWidth);
  const previousScrollableY = Math.max(0, screenshotModalViewportEl.scrollHeight - screenshotModalViewportEl.clientHeight);
  const ratioX = previousScrollableX > 0 ? screenshotModalViewportEl.scrollLeft / previousScrollableX : 0.5;
  const ratioY = previousScrollableY > 0 ? screenshotModalViewportEl.scrollTop / previousScrollableY : 0.5;

  state.screenshotZoom = clampScreenshotZoom(nextZoom);
  updateScreenshotZoomUi();

  if (recenter) {
    centerScreenshotViewport();
    return;
  }

  const nextScrollableX = Math.max(0, screenshotModalViewportEl.scrollWidth - screenshotModalViewportEl.clientWidth);
  const nextScrollableY = Math.max(0, screenshotModalViewportEl.scrollHeight - screenshotModalViewportEl.clientHeight);
  screenshotModalViewportEl.scrollLeft = nextScrollableX * ratioX;
  screenshotModalViewportEl.scrollTop = nextScrollableY * ratioY;
}

function setScreenshotZoomAt(nextZoom, clientX, clientY) {
  if (!screenshotModalViewportEl) return;

  const currentZoom = clampScreenshotZoom(state.screenshotZoom);
  const resolvedZoom = clampScreenshotZoom(nextZoom);
  if (resolvedZoom === currentZoom) return;

  const rect = screenshotModalViewportEl.getBoundingClientRect();
  const offsetX = Math.max(0, Math.min(rect.width, clientX - rect.left));
  const offsetY = Math.max(0, Math.min(rect.height, clientY - rect.top));
  const baseWidth = Math.max(1, state.screenshotBaseWidth || rect.width || 1);
  const baseHeight = Math.max(1, state.screenshotBaseHeight || rect.height || 1);
  const currentWidth = Math.max(1, Math.round(baseWidth * currentZoom));
  const currentHeight = Math.max(1, Math.round(baseHeight * currentZoom));
  const contentRatioX = (screenshotModalViewportEl.scrollLeft + offsetX) / currentWidth;
  const contentRatioY = (screenshotModalViewportEl.scrollTop + offsetY) / currentHeight;

  state.screenshotZoom = resolvedZoom;
  updateScreenshotZoomUi();

  if (resolvedZoom <= SCREENSHOT_ZOOM_MIN) {
    centerScreenshotViewport();
    return;
  }

  const nextWidth = Math.max(1, Math.round(baseWidth * resolvedZoom));
  const nextHeight = Math.max(1, Math.round(baseHeight * resolvedZoom));
  const nextScrollableX = Math.max(0, screenshotModalViewportEl.scrollWidth - screenshotModalViewportEl.clientWidth);
  const nextScrollableY = Math.max(0, screenshotModalViewportEl.scrollHeight - screenshotModalViewportEl.clientHeight);
  screenshotModalViewportEl.scrollLeft = Math.max(0, Math.min(nextScrollableX, contentRatioX * nextWidth - offsetX));
  screenshotModalViewportEl.scrollTop = Math.max(0, Math.min(nextScrollableY, contentRatioY * nextHeight - offsetY));
}

function zoomScreenshot(delta, anchorEvent = null) {
  if (anchorEvent && screenshotModalViewportEl) {
    setScreenshotZoomAt(state.screenshotZoom + delta, anchorEvent.clientX, anchorEvent.clientY);
    return;
  }
  setScreenshotZoom(state.screenshotZoom + delta);
}

function stopScreenshotDrag() {
  if (screenshotModalViewportEl && state.screenshotDragPointerId !== null) {
    try {
      if (screenshotModalViewportEl.hasPointerCapture?.(state.screenshotDragPointerId)) {
        screenshotModalViewportEl.releasePointerCapture(state.screenshotDragPointerId);
      }
    } catch {
      // ignore pointer capture release failures
    }
  }
  state.screenshotDragging = false;
  state.screenshotDragPointerId = null;
  updateScreenshotZoomUi();
}

function beginScreenshotDrag(event) {
  if (!screenshotModalViewportEl || state.screenshotZoom <= SCREENSHOT_ZOOM_MIN) return;
  state.screenshotDragging = true;
  state.screenshotDragPointerId = event.pointerId;
  state.screenshotDragStartX = event.clientX;
  state.screenshotDragStartY = event.clientY;
  state.screenshotDragScrollLeft = screenshotModalViewportEl.scrollLeft;
  state.screenshotDragScrollTop = screenshotModalViewportEl.scrollTop;
  screenshotModalViewportEl.setPointerCapture?.(event.pointerId);
  updateScreenshotZoomUi();
  event.preventDefault();
}

function handleScreenshotDrag(event) {
  if (!screenshotModalViewportEl || !state.screenshotDragging) return;
  if (state.screenshotDragPointerId !== null && event.pointerId !== state.screenshotDragPointerId) return;
  const deltaX = event.clientX - state.screenshotDragStartX;
  const deltaY = event.clientY - state.screenshotDragStartY;
  screenshotModalViewportEl.scrollLeft = state.screenshotDragScrollLeft - deltaX;
  screenshotModalViewportEl.scrollTop = state.screenshotDragScrollTop - deltaY;
  event.preventDefault();
}

function resetScreenshotZoom() {
  setScreenshotZoom(SCREENSHOT_ZOOM_MIN, { recenter: true });
}

function showScreenshotModal(src, alt = "スクリーンショット") {
  if (!runtimeUi.screenshotModalEl || !runtimeUi.screenshotModalImageEl) return;
  state.screenshotZoom = SCREENSHOT_ZOOM_MIN;
  state.screenshotBaseWidth = 0;
  state.screenshotBaseHeight = 0;
  stopScreenshotDrag();
  runtimeUi.screenshotModalImageEl.src = src;
  runtimeUi.screenshotModalImageEl.alt = alt;
  runtimeUi.screenshotModalEl.hidden = false;
  runtimeUi.screenshotModalEl.classList.add("is-open");
  lockBodyScroll();

  if (runtimeUi.screenshotModalImageEl.complete) {
    recalculateScreenshotBaseSize();
    resetScreenshotZoom();
  } else {
    updateScreenshotZoomUi();
  }
}

function hideScreenshotModal() {
  if (!runtimeUi.screenshotModalEl || !runtimeUi.screenshotModalImageEl) return;
  runtimeUi.screenshotModalEl.classList.remove("is-open");
  runtimeUi.screenshotModalEl.hidden = true;
  runtimeUi.screenshotModalImageEl.src = "";
  runtimeUi.screenshotModalImageEl.style.width = "";
  runtimeUi.screenshotModalImageEl.style.height = "";
  screenshotModalStageEl?.style.removeProperty("width");
  screenshotModalStageEl?.style.removeProperty("height");
  state.screenshotZoom = SCREENSHOT_ZOOM_MIN;
  state.screenshotBaseWidth = 0;
  state.screenshotBaseHeight = 0;
  stopScreenshotDrag();
  updateScreenshotZoomUi();
  unlockBodyScroll();
}

function openHelpModal() {
  if (!helpModalEl || !helpModalFrameEl) return;
  if (!helpModalFrameEl.src) {
    helpModalFrameEl.src = "/help.html";
  }
  helpModalEl.hidden = false;
  helpModalEl.classList.add("is-open");
  lockBodyScroll();
}

function closeHelpModal() {
  if (!helpModalEl) return;
  helpModalEl.classList.remove("is-open");
  helpModalEl.hidden = true;
  unlockBodyScroll();
}

function logWsEvent(event, detail = {}) {
  console.info(`[whistx][ws] ${event}`, detail);
}

function logClientEvent(event, detail = {}) {
  console.info(`[whistx][client] ${event}`, detail);
}

function sendWsTelemetry(event, detail = {}) {
  const ws = state.ws;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  try {
    ws.send(JSON.stringify({
      type: "telemetry",
      event,
      detail,
    }));
  } catch {
    // ignore telemetry send failures
  }
}

function bucketizeBacklog(value) {
  if (value >= BACKLOG_DANGER_THRESHOLD) return "danger";
  if (value >= BACKLOG_WARN_THRESHOLD) return "warn";
  return "normal";
}

function emitScreenshotSkipTelemetry(reason, extra = {}) {
  const now = Date.now();
  const sameReason = state.lastScreenshotSkipReason === reason;
  const sentRecently = now - state.lastScreenshotSkipSentAt < 15_000;
  if (sameReason && sentRecently) return;
  state.lastScreenshotSkipReason = reason;
  state.lastScreenshotSkipSentAt = now;
  sendWsTelemetry("screenshot_skipped", {
    reason,
    skippedCount: state.screenshotSkippedCount,
    backlog: state.pendingOutboundChunks,
    ...extra,
  });
}

function updateBackpressureState() {
  state.maxObservedBacklog = Math.max(state.maxObservedBacklog, state.pendingOutboundChunks);
  const nextBucket = bucketizeBacklog(state.pendingOutboundChunks);
  if (nextBucket !== state.lastTelemetryBacklogBucket) {
    state.lastTelemetryBacklogBucket = nextBucket;
    sendWsTelemetry("client_backlog", {
      backlog: state.pendingOutboundChunks,
      maxObservedBacklog: state.maxObservedBacklog,
      bucket: nextBucket,
    });
  }
  const nextDegraded = state.pendingOutboundChunks >= BACKLOG_WARN_THRESHOLD;
  if (nextDegraded !== state.degradedCaptureMode) {
    state.degradedCaptureMode = nextDegraded;
    sendWsTelemetry(nextDegraded ? "degraded_capture_enabled" : "degraded_capture_cleared", {
      backlog: state.pendingOutboundChunks,
      maxObservedBacklog: state.maxObservedBacklog,
    });
  }
}

function setupWorkspaceResizers() {
  if (!workspacePanelsEl || panelResizerEls.length === 0) return;

  const stopDrag = () => {
    state.activeResizer = null;
    document.body.classList.remove("is-resizing-panels");
  };

  const onPointerMove = (event) => {
    if (!state.activeResizer || window.innerWidth <= 1439) return;
    const rect = workspacePanelsEl.getBoundingClientRect();
    const totalWidth = rect.width;
    if (totalWidth <= 0) return;

    const currentLeft = state.panelLeftRatio;
    const currentCenter = state.panelCenterRatio;
    const currentRight = state.panelRightRatio;
    const totalRatio = currentLeft + currentCenter + currentRight;
    const ratioPerPixel = totalRatio / totalWidth;

    if (state.activeResizer === "left") {
      const pointerRatio = (event.clientX - rect.left) * ratioPerPixel;
      const nextLeft = Math.max(0.75, Math.min(totalRatio - currentRight - 0.7, pointerRatio));
      const nextCenter = totalRatio - currentRight - nextLeft;
      applyWorkspaceRatios(nextLeft, nextCenter, currentRight);
      return;
    }

    const pointerRatio = (rect.right - event.clientX) * ratioPerPixel;
    const nextRight = Math.max(0.7, Math.min(totalRatio - currentLeft - 0.7, pointerRatio));
    const nextCenter = totalRatio - currentLeft - nextRight;
    applyWorkspaceRatios(currentLeft, nextCenter, nextRight);
  };

  panelResizerEls.forEach((handle) => {
    handle.addEventListener("pointerdown", (event) => {
      if (window.innerWidth <= 1439) return;
      state.activeResizer = String(handle.dataset.resizer || "");
      document.body.classList.add("is-resizing-panels");
      handle.setPointerCapture?.(event.pointerId);
      event.preventDefault();
    });
  });

  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", stopDrag);
  window.addEventListener("pointercancel", stopDrag);
}

function setupPanelToggles() {
  panelToggleEls.forEach((button) => {
    button.addEventListener("click", () => {
      const panel = String(button.dataset.panelToggle || "");
      applyPanelCollapseState(panel, !state.panelCollapsed[panel]);
    });
  });
}

async function readSseJsonStream(response, onEvent) {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("stream_not_supported");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let boundaryIndex = buffer.indexOf("\n\n");
    while (boundaryIndex !== -1) {
      const rawEvent = buffer.slice(0, boundaryIndex);
      buffer = buffer.slice(boundaryIndex + 2);

      const data = rawEvent
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n")
        .trim();

      if (data) {
        onEvent(JSON.parse(data));
      }

      boundaryIndex = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }
}

function applyDiarizationEnabled(value, options = {}) {
  const persist = options.persist !== false;
  const enabled = !!value;
  state.diarizationEnabled = enabled;

  if (diarizationToggleEl) {
    diarizationToggleEl.checked = enabled;
  }

  if (diarizationStateTextEl) {
    if (!state.diarizationAvailable) {
      diarizationStateTextEl.textContent = "利用不可";
    } else {
      diarizationStateTextEl.textContent = enabled ? "ON" : "OFF";
    }
  }

  if (persist) {
    try {
      localStorage.setItem("whistx_diarization_enabled", enabled ? "1" : "0");
    } catch {
      // ignore
    }
  }
  updateDiarizationSpeakerUi();
  return enabled;
}

function clampSpeakerCount(value) {
  const num = Number(value);
  const cap = Math.max(DIARIZATION_SPEAKER_MIN, Number(state.diarizationSpeakerCap || DIARIZATION_SPEAKER_MAX));
  if (!Number.isFinite(num)) return DIARIZATION_SPEAKER_MIN;
  return Math.max(DIARIZATION_SPEAKER_MIN, Math.min(cap, Math.round(num)));
}

function normalizeSpeakerMode(value) {
  if (value === "fixed" || value === "range") return value;
  return "auto";
}

function updateDiarizationSpeakerUi() {
  const available = !!state.diarizationAvailable;
  const enabled = !!state.diarizationEnabled;
  const mode = normalizeSpeakerMode(state.diarizationSpeakerMode);
  const controlsEnabled = available && enabled;
  const visible = controlsEnabled;

  const autoMode = mode === "auto";
  const fixedMode = mode === "fixed";
  const rangeMode = mode === "range";

  if (settingsAdvancedToggleEl) {
    settingsAdvancedToggleEl.hidden = !visible;
  }
  if (!visible && state.advancedSettingsOpen) {
    applyAdvancedSettingsOpen(false);
  }

  if (diarizationConfigRowEl) {
    diarizationConfigRowEl.classList.toggle("is-hidden", !visible);
    diarizationConfigRowEl.hidden = !visible;
  }
  if (diarizationSpeakerHintEl) {
    diarizationSpeakerHintEl.classList.toggle("is-hidden", !visible);
    diarizationSpeakerHintEl.hidden = !visible;
  }

  if (diarizationSpeakerModeEl) {
    diarizationSpeakerModeEl.value = mode;
    diarizationSpeakerModeEl.disabled = !controlsEnabled;
  }

  if (diarizationSpeakerCountEl) {
    diarizationSpeakerCountEl.value = String(state.diarizationSpeakerCount);
    diarizationSpeakerCountEl.disabled = !controlsEnabled || !fixedMode;
  }

  if (diarizationMinSpeakersEl) {
    diarizationMinSpeakersEl.value = String(state.diarizationMinSpeakers);
    diarizationMinSpeakersEl.disabled = !controlsEnabled || !rangeMode;
  }

  if (diarizationMaxSpeakersEl) {
    diarizationMaxSpeakersEl.value = String(state.diarizationMaxSpeakers);
    diarizationMaxSpeakersEl.disabled = !controlsEnabled || !rangeMode;
  }

  if (!visible) {
    return;
  }
  if (!diarizationSpeakerHintEl) return;
  if (autoMode) {
    diarizationSpeakerHintEl.textContent = "自動推定: 発話内容から話者人数を推定します。";
    return;
  }
  if (fixedMode) {
    diarizationSpeakerHintEl.textContent = `固定人数: ${state.diarizationSpeakerCount}人として分離します。`;
    return;
  }
  diarizationSpeakerHintEl.textContent = `範囲指定: ${state.diarizationMinSpeakers}〜${state.diarizationMaxSpeakers}人の範囲で推定します。`;
}

function applyDiarizationSpeakerSettings(value, options = {}) {
  const persist = options.persist !== false;
  const nextMode = normalizeSpeakerMode(value?.mode ?? state.diarizationSpeakerMode);
  const nextCount = clampSpeakerCount(value?.count ?? state.diarizationSpeakerCount);
  let nextMin = clampSpeakerCount(value?.min ?? state.diarizationMinSpeakers);
  let nextMax = clampSpeakerCount(value?.max ?? state.diarizationMaxSpeakers);

  if (nextMin > nextMax) {
    const tmp = nextMin;
    nextMin = nextMax;
    nextMax = tmp;
  }

  state.diarizationSpeakerMode = nextMode;
  state.diarizationSpeakerCount = nextCount;
  state.diarizationMinSpeakers = nextMin;
  state.diarizationMaxSpeakers = nextMax;

  updateDiarizationSpeakerUi();

  if (persist) {
    try {
      localStorage.setItem("whistx_diarization_speaker_mode", nextMode);
      localStorage.setItem("whistx_diarization_speaker_count", String(nextCount));
      localStorage.setItem("whistx_diarization_min_speakers", String(nextMin));
      localStorage.setItem("whistx_diarization_max_speakers", String(nextMax));
      state.hasSavedDiarizationSpeakerSettings = true;
    } catch {
      // ignore
    }
  }
}

function resolveDiarizationStartOptions() {
  const mode = normalizeSpeakerMode(state.diarizationSpeakerMode);
  if (!state.diarizationAvailable || !state.diarizationEnabled) {
    return {
      diarizationNumSpeakers: 0,
      diarizationMinSpeakers: 0,
      diarizationMaxSpeakers: 0,
    };
  }
  if (mode === "fixed") {
    return {
      diarizationNumSpeakers: clampSpeakerCount(state.diarizationSpeakerCount),
      diarizationMinSpeakers: 0,
      diarizationMaxSpeakers: 0,
    };
  }
  if (mode === "range") {
    const min = clampSpeakerCount(state.diarizationMinSpeakers);
    const max = clampSpeakerCount(state.diarizationMaxSpeakers);
    return {
      diarizationNumSpeakers: 0,
      diarizationMinSpeakers: Math.min(min, max),
      diarizationMaxSpeakers: Math.max(min, max),
    };
  }
  return {
    diarizationNumSpeakers: 0,
    diarizationMinSpeakers: 0,
    diarizationMaxSpeakers: 0,
  };
}

function normalizeBannerType(value) {
  const type = String(value || "info").toLowerCase();
  if (type === "success" || type === "warning" || type === "error") return type;
  return "info";
}

function renderBanners(rawBanners) {
  if (!bannersContainerEl) return;

  const banners = Array.isArray(rawBanners) ? rawBanners : [];
  state.banners = banners;
  bannersContainerEl.innerHTML = "";

  banners.forEach((banner, index) => {
    const record = banner && typeof banner === "object" ? banner : {};
    const id = String(record.id || `banner-${index + 1}`).trim() || `banner-${index + 1}`;
    const type = normalizeBannerType(record.type);
    const title = String(record.title || "").trim();
    const message = String(record.message || "").trim();
    const dismissible = record.dismissible !== false;

    if (!message) return;
    const node = document.createElement("article");
    node.className = `notice-banner notice-${type}`;
    node.setAttribute("role", "status");

    const header = document.createElement("div");
    header.className = "notice-banner-header";

    const heading = document.createElement("strong");
    heading.className = "notice-banner-title";
    heading.textContent = title || type.toUpperCase();
    header.appendChild(heading);

    if (dismissible) {
      const closeBtn = document.createElement("button");
      closeBtn.type = "button";
      closeBtn.className = "notice-banner-close";
      closeBtn.setAttribute("aria-label", "バナーを閉じる");
      closeBtn.textContent = "×";
      closeBtn.addEventListener("click", () => {
        node.remove();
        bannersContainerEl.hidden = bannersContainerEl.childElementCount === 0;
      });
      header.appendChild(closeBtn);
    }

    const body = document.createElement("p");
    body.className = "notice-banner-body";
    body.textContent = message;

    node.appendChild(header);
    node.appendChild(body);
    bannersContainerEl.appendChild(node);
  });

  bannersContainerEl.hidden = bannersContainerEl.childElementCount === 0;
}

// Toast notification system
function showToast(message, type = "default", duration = 2500) {
  if (!toastContainer) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  const iconEl = createToastIcon(type);
  if (iconEl) {
    toast.appendChild(iconEl);
  }
  const textEl = document.createElement("span");
  textEl.textContent = String(message || "");
  toast.appendChild(textEl);
  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("hiding");
    setTimeout(() => toast.remove(), 250);
  }, duration);
}

function createToastIcon(type) {
  if (type !== "success" && type !== "error") {
    return null;
  }
  const svgNs = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNs, "svg");
  svg.setAttribute("width", "16");
  svg.setAttribute("height", "16");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");

  if (type === "success") {
    const path = document.createElementNS(svgNs, "path");
    path.setAttribute("d", "M20 6L9 17l-5-5");
    svg.appendChild(path);
    return svg;
  }

  const circle = document.createElementNS(svgNs, "circle");
  circle.setAttribute("cx", "12");
  circle.setAttribute("cy", "12");
  circle.setAttribute("r", "10");
  const path1 = document.createElementNS(svgNs, "path");
  path1.setAttribute("d", "M15 9l-6 6");
  const path2 = document.createElementNS(svgNs, "path");
  path2.setAttribute("d", "M9 9l6 6");
  svg.append(circle, path1, path2);
  return svg;
}

function renderEmptyTranscriptState() {
  logEl.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
          <line x1="12" x2="12" y1="19" y2="22"/>
        </svg>
      </div>
      <p class="empty-title">まだ文字起こしがありません</p>
      <p class="empty-description">録音を開始すると、リアルタイムで文字起こしが表示されます</p>
    </div>
  `;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function isLoginRequiredError(error) {
  return error?.status === 401 || error?.message === "login_required" || error?.payload?.detail === "login_required";
}

function setSaveBadge(label, saved = false) {
  if (!saveStateBadgeEl) return;
  saveStateBadgeEl.textContent = label;
  saveStateBadgeEl.classList.toggle("is-saved", !!saved);
}

function updateSaveControls() {
  const hasSegments = state.segments.length > 0;
  const authenticated = !!state.auth.authenticated;
  const isGuest = !!state.auth.isGuest;
  const saved = !!state.savedHistoryId;
  const viewingHistory = !!state.viewingHistoryId;
  const recordingLocked = !!state.recording || !!state.finalizingStop;

  if (saveBtn) {
    saveBtn.disabled =
      !authenticated || isGuest || recordingLocked || !hasSegments || state.saveInFlight || saved || viewingHistory;
    saveBtn.textContent = state.saveInFlight ? "保存中..." : "保存";
    saveBtn.title = recordingLocked
      ? "録音中は保存できません"
      : isGuest
        ? "ゲストでは保存できません"
        : authenticated
          ? ""
          : "ログインが必要です";
  }
  if (saveTitleInputEl) {
    saveTitleInputEl.disabled = viewingHistory || state.saveInFlight || recordingLocked;
  }
  setSaveBadge(
    saved ? "保存済み" : recordingLocked ? "録音中は保存不可" : isGuest ? "ゲストでは保存不可" : authenticated ? "未保存" : "ログインが必要",
    saved
  );
}

function formatHistoryMeta(item) {
  const parts = [];
  if (item.savedAt) parts.push(new Date(item.savedAt).toLocaleString("ja-JP"));
  if (item.language) parts.push(formatLanguageLabel(item.language));
  return parts.join(" / ");
}

function formatHistoryDaysRemaining(item) {
  const retentionDays = Math.max(1, Number(state.auth.historyRetentionDays || 7));
  if (!item?.savedAt) return `${retentionDays}日で削除`;
  const savedAtMs = new Date(item.savedAt).getTime();
  if (!Number.isFinite(savedAtMs)) return `${retentionDays}日で削除`;
  const expiresAtMs = savedAtMs + retentionDays * 24 * 60 * 60 * 1000;
  const remainingMs = expiresAtMs - Date.now();
  if (remainingMs <= 0) return "まもなく削除";
  return `あと${Math.ceil(remainingMs / (24 * 60 * 60 * 1000))}日`;
}

function formatLanguageLabel(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "ja") return "日本語";
  if (normalized === "en") return "英語";
  if (!normalized || normalized === "auto") return "自動";
  return String(value);
}

function updateHistoryEmptyState(message = "") {
  if (!historyEmptyEl) return;
  if (message) {
    historyEmptyEl.hidden = false;
    historyEmptyEl.textContent = message;
    return;
  }
  if (state.auth.isGuest) {
    historyEmptyEl.hidden = false;
    historyEmptyEl.textContent = "ゲストでは履歴は利用できません";
    return;
  }
  if (!state.auth.authenticated) {
    historyEmptyEl.hidden = false;
    historyEmptyEl.textContent = state.auth.bootstrapAdminRequired
      ? "初回管理者アカウントを作成すると履歴がここに表示されます"
      : "ログインすると保存済み履歴がここに表示されます";
    return;
  }
  historyEmptyEl.hidden = state.history.items.length > 0;
  historyEmptyEl.textContent = "保存済み履歴はまだありません";
}

function renderHistoryList() {
  if (!historyListEl) return;
  historyListEl.innerHTML = "";
  const historyCount = state.history.total || state.history.items.length || 0;
  if (historyCountBadgeEl) {
    historyCountBadgeEl.textContent = historyCount + "件";
  }
  state.history.items.forEach((item) => {
    const article = document.createElement("article");
    article.className = "history-item";
    if (state.history.selectedId === item.id) {
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
    article.querySelector(".history-item-main")?.addEventListener("click", () => {
      openHistoryDetail(item.id);
    });
    article.querySelector(".history-item-main")?.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openHistoryDetail(item.id);
      }
    });
    article.querySelector(".history-item-delete")?.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      deleteHistory(item.id);
    });
    historyListEl.appendChild(article);
  });
  updateHistoryEmptyState();
}

function closeAdminQueueModal() {
  if (!adminQueueModalEl) return;
  adminQueueModalEl.hidden = true;
  unlockBodyScroll();
}

function renderAdminPendingUsers() {
  if (!adminPendingListEl) return;
  adminPendingListEl.innerHTML = "";
  if (!state.adminPendingUsers.length) {
    adminPendingListEl.innerHTML = `
      <div class="admin-pending-empty">
        <p>承認待ちユーザーはありません。</p>
      </div>
    `;
    return;
  }
  state.adminPendingUsers.forEach((item) => {
    const article = document.createElement("article");
    article.className = "admin-pending-item";
    const createdAt = item.createdAt
      ? new Date(item.createdAt).toLocaleString("ja-JP")
      : "";
    article.innerHTML = `
      <div class="admin-pending-copy">
        <div class="admin-pending-name">${escapeHtml(item.displayName || item.email || "pending")}</div>
        <div class="admin-pending-email">${escapeHtml(item.email || "")}</div>
        <div class="admin-pending-date">${escapeHtml(createdAt)}</div>
      </div>
      <button type="button" class="btn-action admin-pending-approve">承認</button>
    `;
    article.querySelector(".admin-pending-approve")?.addEventListener("click", () => {
      approvePendingUser(item.id);
    });
    adminPendingListEl.appendChild(article);
  });
}

function renderAuthState() {
  const authenticated = !!state.auth.authenticated;
  const isGuest = !!state.auth.isGuest;
  const workspaceEnabled = authenticated || isGuest;
  const bootstrapRequired = !!state.auth.bootstrapAdminRequired;
  const isAdmin = !!state.auth.user?.isAdmin;
  const userLabel = authenticated ? serializeUserLabel(state.auth.user) : isGuest ? "ゲスト利用中" : "未ログイン";
  if (authUserLabelEl) {
    authUserLabelEl.textContent = userLabel;
  }
  if (loginBtn) {
    loginBtn.hidden = workspaceEnabled;
  }
  if (logoutBtn) {
    logoutBtn.hidden = !workspaceEnabled;
  }
  if (authGuestViewEl) {
    authGuestViewEl.hidden = workspaceEnabled;
  }
  if (authUserViewEl) {
    authUserViewEl.hidden = !workspaceEnabled;
  }
  if (authBootstrapSectionEl) {
    authBootstrapSectionEl.hidden = !bootstrapRequired;
  }
  if (authLoginSectionEl) {
    authLoginSectionEl.hidden = bootstrapRequired;
  }
  if (authRegisterSectionEl) {
    authRegisterSectionEl.hidden = bootstrapRequired || !state.selfSignupEnabled;
  }
  if (keycloakLoginBtnEl) {
    keycloakLoginBtnEl.hidden = bootstrapRequired || !state.auth.keycloakEnabled;
    keycloakLoginBtnEl.textContent = state.auth.keycloakButtonLabel || "Keycloakでログイン";
  }
  if (adminQueueBtn) {
    adminQueueBtn.hidden = !authenticated || !isAdmin;
  }
  if (adminQueueBadgeEl) {
    adminQueueBadgeEl.textContent = String(state.auth.pendingApprovalCount || 0);
  }
  if (registerBtn) {
    registerBtn.disabled = !state.selfSignupEnabled;
  }
  if (registerHintEl) {
    if (bootstrapRequired) {
      registerHintEl.textContent = "";
    } else if (state.selfSignupEnabled) {
      registerHintEl.textContent = "申請後は管理者の承認が完了するまでログインできません";
    } else {
      registerHintEl.textContent = "新規登録は現在無効です";
    }
  }
  updateHistoryEmptyState();
  updateSaveControls();
  updateSharedVocabularyMeta();
}

function resetCurrentSaveState() {
  state.runtimeSessionToken = "";
  state.savedHistoryId = null;
  state.viewingHistoryId = null;
  if (saveTitleInputEl) {
    saveTitleInputEl.value = "";
  }
  updateDownloadLinks();
  updateSaveControls();
}

function updateSegmentCount() {
  const count = state.segments.length;
  segmentCountEl.textContent = `${count}件`;
  if (connCountEl) {
    connCountEl.textContent = String(count);
  }

  segmentCountEl.classList.remove("updated");
  void segmentCountEl.offsetWidth;
  segmentCountEl.classList.add("updated");
}

function extractTranscriptText() {
  const fromState = state.segments
    .map((segment) => renderTranscriptText(segment.text, segment.speaker))
    .join("\n")
    .trim();
  if (fromState) return fromState;

  const rows = Array.from(logEl.querySelectorAll(".log-row .text"));
  const fromDom = rows.map((node) => node.textContent || "").join("\n").trim();
  return fromDom;
}

function updateDownloadLinks() {
  if (state.viewingHistoryId) {
    dlTxt.href = `/api/history/${state.viewingHistoryId}/download.txt`;
    dlJsonl.href = `/api/history/${state.viewingHistoryId}/download.jsonl`;
    dlZip.href = `/api/history/${state.viewingHistoryId}/download.zip`;
    return;
  }

  if (!state.runtimeSessionId) {
    dlTxt.href = "#";
    dlJsonl.href = "#";
    dlZip.href = "#";
    return;
  }

  const suffix = state.runtimeSessionToken ? `?token=${encodeURIComponent(state.runtimeSessionToken)}` : "";
  dlTxt.href = `/api/transcript/${state.runtimeSessionId}.txt${suffix}`;
  dlJsonl.href = `/api/transcript/${state.runtimeSessionId}.jsonl${suffix}`;
  dlZip.href = `/api/transcript/${state.runtimeSessionId}.zip${suffix}`;
}

function setSummary(text, meta) {
  state.summary = text || "";

  if (text) {
    summaryTextEl.innerHTML = "";
    summaryTextEl.textContent = text;
  } else {
    summaryTextEl.innerHTML = `
      <div class="empty-state small">
        <p class="empty-description">「要約生成」ボタンでAIによる要約を生成します</p>
      </div>
    `;
  }

  summaryMetaEl.textContent = meta || "未生成";
}

async function copySummaryText() {
  if (!canUseWorkspace()) {
    showToast("ログインが必要です", "error");
    setAppLocked(true);
    loginEmailEl?.focus();
    return;
  }
  const text = String(state.summary || summaryTextEl?.textContent || "").trim();
  if (!text) {
    showToast("要約がありません", "error");
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    showToast("要約をコピーしました", "success");
  } catch {
    showToast("要約のコピーに失敗しました", "error");
  }
}

function setProofread(text, meta) {
  state.proofread = text || "";

  if (text) {
    proofreadTextEl.innerHTML = "";
    proofreadTextEl.textContent = text;
  } else {
    proofreadTextEl.innerHTML = `
      <div class="empty-state small">
        <p class="empty-description">「校正」ボタンで校正済みテキストを生成します</p>
      </div>
    `;
  }

  proofreadMetaEl.textContent = meta || "未生成";
}

function markProofreadStale() {
  if (!state.proofread) return;
  proofreadMetaEl.textContent = "更新が必要";
}

function renderTranscriptText(rawText, speaker) {
  const clean = String(rawText || "").trim();
  if (!clean) return "";
  const label = String(speaker || "").trim();
  if (!label) return clean;
  return `[${label}] ${clean}`;
}

function isLogNearBottom() {
  if (!logEl) return true;
  const distance = logEl.scrollHeight - logEl.clientHeight - logEl.scrollTop;
  return distance <= 48;
}

function scrollLogToBottom() {
  if (!logEl) return;
  requestAnimationFrame(() => {
    logEl.scrollTop = logEl.scrollHeight;
  });
}

function addLogLine(text, tsStart, tsEnd, seq, speaker, screenshotPath = "", rawAudioPath = "", audioPath = "") {
  const shouldAutoScroll = state.logAutoScrollEnabled || isLogNearBottom();

  // Hide empty state on first log entry
  const emptyState = logEl.querySelector(".empty-state");
  if (emptyState) {
    emptyState.remove();
  }

  const row = document.createElement("div");
  row.className = "log-row new";

  const range = document.createElement("span");
  range.className = "time";
  range.textContent = `${formatMs(tsStart)} - ${formatMs(tsEnd)}`;

  const content = document.createElement("span");
  content.className = "text";
  content.dataset.rawText = text;
  if (speaker) {
    content.dataset.speaker = speaker;
  }
  content.textContent = renderTranscriptText(text, speaker);

  if (Number.isFinite(Number(seq))) {
    row.dataset.seq = String(Number(seq));
  }

  const body = document.createElement("div");
  body.className = "log-body";
  body.append(content);

  let mediaGroup = null;
  if (screenshotPath || rawAudioPath || audioPath) {
    mediaGroup = document.createElement("div");
    mediaGroup.className = "log-media-group";
  }

  if (screenshotPath && mediaGroup) {
    const link = document.createElement("a");
    link.className = "log-screenshot-link";
    link.href = screenshotPath;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.title = "スクリーンショットを開く";
    link.addEventListener("click", (event) => {
      event.preventDefault();
      showScreenshotModal(screenshotPath, `${formatMs(tsStart)} の画面キャプチャ`);
    });

    const image = document.createElement("img");
    image.className = "log-screenshot";
    image.src = screenshotPath;
    image.alt = `${formatMs(tsStart)} の画面キャプチャ`;
    image.loading = "lazy";
    image.addEventListener("click", (event) => {
      event.preventDefault();
      showScreenshotModal(screenshotPath, image.alt);
    });

    link.append(image);
    mediaGroup.append(link);
  }

  if ((rawAudioPath || audioPath) && mediaGroup) {
    const audioActions = document.createElement("div");
    audioActions.className = "log-audio-actions";
    const inlineAudio = document.createElement("audio");
    inlineAudio.className = "log-inline-audio";
    inlineAudio.controls = true;
    inlineAudio.preload = "none";
    inlineAudio.hidden = true;
    inlineAudio.addEventListener("play", () => {
      document.querySelectorAll(".log-inline-audio").forEach((element) => {
        if (element !== inlineAudio && typeof element.pause === "function") {
          element.pause();
        }
      });
    });

    const createAudioButton = (label, url, extraClass = "") => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `log-audio-link ${extraClass}`.trim();
      button.title = `${label}を再生`;
      button.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="6 3 20 12 6 21 6 3"></polygon>
        </svg>
        <span>${label}</span>
      `;
      button.addEventListener("click", async () => {
        const isSameSource = inlineAudio.dataset.src === url;
        if (isSameSource && !inlineAudio.hidden) {
          if (inlineAudio.paused) {
            try {
              await inlineAudio.play();
            } catch {
              // ignore
            }
          } else {
            inlineAudio.pause();
          }
          return;
        }
        inlineAudio.pause();
        inlineAudio.src = url;
        inlineAudio.dataset.src = url;
        inlineAudio.hidden = false;
        try {
          await inlineAudio.play();
        } catch {
          // controls stay visible for manual playback
        }
      });
      return button;
    };

    if (rawAudioPath) {
      audioActions.append(createAudioButton("元音声", rawAudioPath));
    }

    if (audioPath) {
      audioActions.append(createAudioButton("加工後", audioPath, "is-processed"));
    }

    mediaGroup.append(audioActions);
    mediaGroup.append(inlineAudio);
  }

  if (mediaGroup) {
    body.append(mediaGroup);
  }

  row.append(range, body);
  logEl.appendChild(row);
  if (shouldAutoScroll) {
    scrollLogToBottom();
  }

  state.log.push(text);
  state.segments.push({
    text,
    tsStart,
    tsEnd,
    seq,
    speaker,
    screenshotPath,
    rawAudioPath,
    audioPath,
  });
  updateSegmentCount();
  markProofreadStale();
  updateSaveControls();

  // Remove 'new' class after animation
  setTimeout(() => row.classList.remove("new"), 300);
}

function applySpeakerPatch(segments) {
  if (!Array.isArray(segments) || !segments.length) return;

  for (const seg of segments) {
    const seq = Number(seg?.seq);
    const speaker = String(seg?.speaker || "").trim();
    if (!Number.isFinite(seq) || !speaker) continue;

    const row = logEl.querySelector(`.log-row[data-seq="${seq}"]`);
    if (!row) continue;

    const textNode = row.querySelector(".text");
    if (!textNode) continue;

    const raw = String(textNode.dataset.rawText || textNode.textContent || "").trim();
    textNode.dataset.rawText = raw;
    textNode.dataset.speaker = speaker;
    textNode.textContent = renderTranscriptText(raw, speaker);

    const target = state.segments.find((item) => Number(item.seq) === seq);
    if (target) {
      target.speaker = speaker;
    }
  }
}

function formatMs(ms) {
  const totalSec = Math.floor(Math.max(0, ms) / 1000);
  const min = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const sec = String(totalSec % 60).padStart(2, "0");
  return `${min}:${sec}`;
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/transcribe`;
}

function selectMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];

  for (const candidate of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(candidate)) {
      return candidate;
    }
  }
  return "";
}

function generateSessionSeed() {
  const t = Date.now().toString(36);
  const r = Math.random().toString(36).slice(2, 6);
  return `sess-${t}-${r}`;
}

function normalizeChunkSeconds(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return CHUNK_DEFAULT_SECONDS;
  return Math.max(CHUNK_MIN_SECONDS, Math.min(CHUNK_MAX_SECONDS, Math.round(num)));
}

function updateChunkHint(seconds) {
  if (!chunkHintEl) return;
  if (seconds <= 15) {
    chunkHintEl.textContent = "短め上限。無音で早めに確定";
    return;
  }
  if (seconds <= 30) {
    chunkHintEl.textContent = "バランス。無音で自然に区切る";
    return;
  }
  if (seconds <= 45) {
    chunkHintEl.textContent = "精度優先。長めに文脈を保持";
    return;
  }
  chunkHintEl.textContent = "最大長。無音が少ない会話向け";
}

function updatePresetActive(seconds) {
  presetButtons.forEach((button) => {
    const raw = button.getAttribute("data-chunk-preset") || "";
    button.classList.toggle("is-active", Number(raw) === seconds);
  });
}

function applyChunkSeconds(value) {
  const seconds = normalizeChunkSeconds(value);
  if (chunkSecondsEl) {
    chunkSecondsEl.value = String(seconds);
  }
  updateChunkHint(seconds);
  updatePresetActive(seconds);
  try {
    localStorage.setItem("whistx_chunk_seconds", String(seconds));
  } catch {
    // ignore
  }
  return seconds;
}

function setUiRecording(active) {
  state.recording = active;

  if (audioLevelIndicatorEl) {
    audioLevelIndicatorEl.hidden = !active;
  }
  if (!active) {
    renderAudioLevel(0);
  }

  if (active) {
    startBtn.classList.add("is-recording");
    startBtn.querySelector(".record-label").textContent = "停止";
    startBtn.setAttribute("aria-pressed", "true");
    startBtn.setAttribute("aria-label", "録音を停止");
  } else {
    startBtn.classList.remove("is-recording");
    startBtn.querySelector(".record-label").textContent = "録音開始";
    startBtn.setAttribute("aria-pressed", "false");
    startBtn.setAttribute("aria-label", "録音を開始");

    if (state.log.length > 0) {
      startBtn.classList.add("is-complete");
      setTimeout(() => startBtn.classList.remove("is-complete"), 800);
    }
  }

  startBtn.disabled = false;
  updateSaveControls();
}

function normalizeAudioSource(value) {
  if (value === "display" || value === "both") return value;
  return "mic";
}

function resetRuntimeSessionState() {
  state.runtimeSessionId = "";
  state.runtimeSessionToken = "";
}

function audioSourceHintText(source) {
  if (source === "display") {
    return "画面共有音声はブラウザ制約で取得できない場合があります。Chrome/Edge のタブ共有が最も安定します。";
  }
  if (source === "both") {
    return "画面共有音声とマイクを混ぜます。画面共有音声は共有面やブラウザ制約の影響を受けます。";
  }
  return "通常のマイク入力を使います。会議アプリ音声は画面共有では取得できない場合があります。";
}

function applyAudioSource(value) {
  const source = normalizeAudioSource(value);
  if (audioSourceEl) {
    audioSourceEl.value = source;
  }
  if (audioSourceHintTextEl || audioSourceHintEl) {
    (audioSourceHintTextEl || audioSourceHintEl).textContent = audioSourceHintText(source);
  }
  state.vadRmsThreshold = vadThresholdForSource(source, state.vadNoiseFloor);
  try {
    localStorage.setItem("whistx_audio_source", source);
  } catch {
    // ignore
  }
  return source;
}

function vadThresholdForSource(source, noiseFloor = null) {
  const mode = normalizeAudioSource(source);
  const baseThreshold = mode === "display" ? 0.0045 : mode === "both" ? 0.0065 : 0.0075;
  const floor = Number.isFinite(noiseFloor) ? Math.max(0, Number(noiseFloor)) : 0;
  if (floor <= 0) return baseThreshold;

  const multiplier = VAD_NOISE_FLOOR_MULTIPLIER[mode] || VAD_NOISE_FLOOR_MULTIPLIER.mic;
  const offset = VAD_NOISE_FLOOR_OFFSET[mode] || VAD_NOISE_FLOOR_OFFSET.mic;
  return Math.max(baseThreshold, floor * multiplier + offset);
}

function vadSourceCutPolicy(source) {
  return VAD_SOURCE_CUT_POLICY[normalizeAudioSource(source)] || VAD_SOURCE_CUT_POLICY.mic;
}

function currentVadSourceMode() {
  return normalizeAudioSource(state.recordingAudioSource || state.recordingRequestedAudioSource || "mic");
}

function updateVadNoiseFloor(rms, now) {
  if (!Number.isFinite(rms) || rms <= 0) {
    return;
  }

  const startedAt = state.vadNoiseFloorAt || now;
  const elapsed = Math.max(0, now - startedAt);
  const source = currentVadSourceMode();
  const threshold = vadThresholdForSource(source, state.vadNoiseFloor);
  const isWarmup = elapsed <= VAD_NOISE_FLOOR_WARMUP_MS;
  const trackingThreshold = state.vadNoiseFloor === null ? threshold * 0.88 : threshold * (isWarmup ? 0.95 : 0.92);
  const shouldTrack = rms < trackingThreshold;

  if (!shouldTrack) {
    return;
  }

  if (state.vadNoiseFloor === null) {
    state.vadNoiseFloor = rms;
  } else {
    state.vadNoiseFloor = state.vadNoiseFloor * (1 - VAD_NOISE_FLOOR_EWMA) + rms * VAD_NOISE_FLOOR_EWMA;
  }

  state.vadRmsThreshold = vadThresholdForSource(source, state.vadNoiseFloor);
}

function hasAudioTrack(stream) {
  return !!stream && stream.getAudioTracks().length > 0;
}

function setupDisplayCaptureVideo(displayStream) {
  const [videoTrack] = displayStream?.getVideoTracks?.() || [];
  if (!videoTrack) {
    state.displayCaptureVideo = null;
    return;
  }

  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.autoplay = true;
  video.srcObject = new MediaStream([videoTrack]);
  const playPromise = video.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {});
  }
  state.displayCaptureVideo = video;
}

async function captureDisplayScreenshot() {
  if (!state.captureScreenshotsEnabled) return null;
  const video = state.displayCaptureVideo;
  const displayStream = state.displayStream;
  if (!video || !displayStream) return null;
  if (state.screenshotEncodeInFlight) {
    state.screenshotSkippedCount += 1;
    emitScreenshotSkipTelemetry("encode_in_flight");
    return null;
  }

  const [videoTrack] = displayStream.getVideoTracks?.() || [];
  if (!videoTrack) return null;

  const width = Number(video.videoWidth || videoTrack.getSettings?.().width || 0);
  const height = Number(video.videoHeight || videoTrack.getSettings?.().height || 0);
  if (!width || !height) return null;

  const now = performance.now();
  const minIntervalMs = state.degradedCaptureMode ? SCREENSHOT_DEGRADED_INTERVAL_MS : SCREENSHOT_MIN_INTERVAL_MS;
  if (now - state.screenshotLastCapturedAt < minIntervalMs) {
    state.screenshotSkippedCount += 1;
    emitScreenshotSkipTelemetry("interval_guard", { minIntervalMs });
    return null;
  }

  const maxWidth = state.degradedCaptureMode ? SCREENSHOT_DEGRADED_MAX_WIDTH : SCREENSHOT_MAX_WIDTH;
  const targetWidth = Math.min(maxWidth, width);
  const targetHeight = Math.max(1, Math.round((height * targetWidth) / width));

  const canvas = state.screenshotCanvas || document.createElement("canvas");
  state.screenshotCanvas = canvas;
  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

  const signature = buildScreenshotSignature(canvas);
  if (shouldSkipScreenshotByDiff(signature)) {
    state.screenshotSkippedCount += 1;
    emitScreenshotSkipTelemetry("diff_unchanged");
    return null;
  }

  state.screenshotEncodeInFlight = true;
  const blob = await new Promise((resolve) => {
    canvas.toBlob((nextBlob) => resolve(nextBlob), "image/webp", SCREENSHOT_WEBP_QUALITY);
  });
  state.screenshotEncodeInFlight = false;
  if (!blob) {
    emitScreenshotSkipTelemetry("encode_failed");
    return null;
  }

  const buffer = await blob.arrayBuffer();
  state.screenshotLastCapturedAt = now;
  return {
    mimeType: blob.type || "image/webp",
    data: arrayBufferToBase64(buffer),
  };
}

function buildScreenshotSignature(sourceCanvas) {
  const canvas = state.screenshotDiffCanvas || document.createElement("canvas");
  state.screenshotDiffCanvas = canvas;
  canvas.width = SCREENSHOT_DIFF_WIDTH;
  canvas.height = SCREENSHOT_DIFF_HEIGHT;

  const ctx = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(sourceCanvas, 0, 0, SCREENSHOT_DIFF_WIDTH, SCREENSHOT_DIFF_HEIGHT);

  const { data } = ctx.getImageData(0, 0, SCREENSHOT_DIFF_WIDTH, SCREENSHOT_DIFF_HEIGHT);
  const signature = new Uint8Array(SCREENSHOT_DIFF_WIDTH * SCREENSHOT_DIFF_HEIGHT);
  for (let src = 0, dst = 0; src < data.length; src += 4, dst += 1) {
    signature[dst] = ((data[src] * 77) + (data[src + 1] * 150) + (data[src + 2] * 29)) >> 8;
  }
  return signature;
}

function shouldSkipScreenshotByDiff(signature) {
  if (!state.screenshotDiffSkipEnabled || !signature) {
    if (signature) {
      state.previousScreenshotSignature = signature;
    }
    return false;
  }

  const previous = state.previousScreenshotSignature;
  if (!(previous instanceof Uint8Array) || previous.length !== signature.length) {
    state.previousScreenshotSignature = signature;
    return false;
  }

  let diffSum = 0;
  let changedPixels = 0;
  for (let i = 0; i < signature.length; i += 1) {
    const delta = Math.abs(signature[i] - previous[i]);
    diffSum += delta;
    if (delta >= SCREENSHOT_DIFF_PIXEL_THRESHOLD) {
      changedPixels += 1;
    }
  }

  const meanDiff = diffSum / signature.length;
  const changedRatio = changedPixels / signature.length;
  if (meanDiff < SCREENSHOT_DIFF_MEAN_THRESHOLD && changedRatio < SCREENSHOT_DIFF_CHANGED_RATIO_THRESHOLD) {
    return true;
  }

  state.previousScreenshotSignature = signature;
  return false;
}

async function requestMicStream(sourceMode = "mic") {
  void sourceMode;
  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false,
    },
  });
}

async function requestDisplayStream() {
  const candidates = [
    {
      video: true,
      audio: { suppressLocalAudioPlayback: false },
      systemAudio: "include",
      windowAudio: "system",
      surfaceSwitching: "include",
      selfBrowserSurface: "exclude",
      monitorTypeSurfaces: "include",
    },
    {
      video: true,
      audio: true,
      systemAudio: "include",
      windowAudio: "system",
      surfaceSwitching: "include",
      selfBrowserSurface: "exclude",
      monitorTypeSurfaces: "include",
    },
    {
      video: true,
      audio: true,
    },
  ];

  let lastError = null;
  for (const constraints of candidates) {
    try {
      return await navigator.mediaDevices.getDisplayMedia(constraints);
    } catch (error) {
      lastError = error;
      if (error?.name && error.name !== "TypeError" && error.name !== "OverconstrainedError") {
        throw error;
      }
    }
  }

  throw lastError || new Error("display_capture_not_supported");
}

async function ensureAudioContextResumed(context) {
  if (context.state !== "suspended") return;
  try {
    await context.resume();
  } catch {
    // ignore
  }
}

async function buildMixedAudioStream(streams) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) {
    throw new Error("AudioContext_not_supported");
  }

  const context = new AudioContextCtor();
  await ensureAudioContextResumed(context);

  const destination = context.createMediaStreamDestination();
  const mixBus = context.createGain();
  const outputGain = context.createGain();
  const monitorAnalyser = context.createAnalyser();
  monitorAnalyser.fftSize = 2048;
  monitorAnalyser.smoothingTimeConstant = 0.88;
  const sources = [];

  mixBus.connect(monitorAnalyser);
  mixBus.connect(outputGain);
  outputGain.connect(destination);

  for (const stream of streams) {
    if (!hasAudioTrack(stream)) continue;
    const source = context.createMediaStreamSource(stream);
    source.connect(mixBus);
    sources.push(source);
  }

  if (!sources.length) {
    await context.close().catch(() => {
      // ignore
    });
    throw new Error("display_audio_not_found");
  }

  state.captureContext = context;
  state.captureSources = sources;
  state.captureDestination = destination;
  state.captureGainNode = outputGain;
  state.captureMonitorAnalyser = monitorAnalyser;
  state.captureMonitorBuffer = new Float32Array(monitorAnalyser.fftSize);
  state.captureAutoGainLevel = 1;
  state.captureAutoGainSmoothedRms = 0;
  updateCaptureAutoGainState();
  return destination.stream;
}

function sampleCaptureAutoGainRms() {
  if (!state.captureMonitorAnalyser || !state.captureMonitorBuffer) {
    return 0;
  }

  state.captureMonitorAnalyser.getFloatTimeDomainData(state.captureMonitorBuffer);
  let sum = 0;
  for (let i = 0; i < state.captureMonitorBuffer.length; i += 1) {
    const value = state.captureMonitorBuffer[i];
    sum += value * value;
  }
  return Math.sqrt(sum / state.captureMonitorBuffer.length);
}

function updateCaptureAutoGainState() {
  if (!state.captureGainNode || !state.captureContext) {
    return;
  }

  if (state.captureAutoGainTimer) {
    clearInterval(state.captureAutoGainTimer);
    state.captureAutoGainTimer = null;
  }

  const now = state.captureContext.currentTime;
  state.captureGainNode.gain.cancelScheduledValues(now);

  if (!state.autoGainEnabled) {
    state.captureAutoGainLevel = 1;
    state.captureAutoGainSmoothedRms = 0;
    state.captureGainNode.gain.setTargetAtTime(1, now, 0.35);
    return;
  }

  const tick = () => {
    if (!state.captureGainNode || !state.captureContext) {
      return;
    }

    const rms = sampleCaptureAutoGainRms();
    const smoothed = state.captureAutoGainSmoothedRms > 0
      ? state.captureAutoGainSmoothedRms * (1 - AUTO_GAIN_SMOOTHING) + rms * AUTO_GAIN_SMOOTHING
      : rms;
    state.captureAutoGainSmoothedRms = smoothed;

    let targetGain = 1;
    if (smoothed > 0 && smoothed < AUTO_GAIN_MIN_RMS) {
      targetGain = Math.min(AUTO_GAIN_MAX, AUTO_GAIN_TARGET_RMS / smoothed);
    } else if (smoothed < AUTO_GAIN_TARGET_RMS) {
      const ratio = (AUTO_GAIN_TARGET_RMS - smoothed) / Math.max(0.0001, AUTO_GAIN_TARGET_RMS - AUTO_GAIN_MIN_RMS);
      targetGain = 1 + ratio * 0.75;
    }

    const easedGain = state.captureAutoGainLevel * 0.7 + targetGain * 0.3;
    state.captureAutoGainLevel = Math.max(1, Math.min(AUTO_GAIN_MAX, easedGain));
    const at = state.captureContext.currentTime;
    state.captureGainNode.gain.cancelScheduledValues(at);
    state.captureGainNode.gain.setTargetAtTime(state.captureAutoGainLevel, at, 0.45);
  };

  tick();
  state.captureAutoGainTimer = setInterval(tick, AUTO_GAIN_ANALYZE_MS);
}

function bindDisplayEndEvents(displayStream) {
  const tracks = [...displayStream.getVideoTracks(), ...displayStream.getAudioTracks()];
  tracks.forEach((track) => {
    track.addEventListener(
      "ended",
      () => {
        if (!state.recording) return;
        setStatus("display_capture_ended");
        stopRecording();
      },
      { once: true }
    );
  });
}

function getDisplayCaptureDiagnostics(displayStream) {
  const audioTracks = displayStream?.getAudioTracks?.() || [];
  const videoTracks = displayStream?.getVideoTracks?.() || [];
  const firstVideo = videoTracks[0] || null;
  const settings = firstVideo?.getSettings?.() || {};

  return {
    audioTrackCount: audioTracks.length,
    audioLabels: audioTracks.map((track) => track.label || "unknown"),
    videoLabel: firstVideo?.label || "unknown",
    displaySurface: settings.displaySurface || "unknown",
  };
}

function logDisplayCaptureDiagnostics(displayStream, sourceMode) {
  const diagnostics = getDisplayCaptureDiagnostics(displayStream);
  console.info("[whistx] display capture diagnostics", {
    sourceMode,
    ...diagnostics,
  });
  return diagnostics;
}

async function prepareInputStream(sourceMode) {
  const mode = normalizeAudioSource(sourceMode);

  if (mode === "mic") {
    const micStream = await requestMicStream(mode);
    state.micStream = micStream;
    return buildMixedAudioStream([micStream]);
  }

  if (mode === "display") {
    const displayStream = await requestDisplayStream();
    const diagnostics = logDisplayCaptureDiagnostics(displayStream, mode);
    if (!hasAudioTrack(displayStream)) {
      const error = new Error("display_audio_not_found");
      error.diagnostics = diagnostics;
      throw error;
    }
    bindDisplayEndEvents(displayStream);
    state.displayStream = displayStream;
    setupDisplayCaptureVideo(displayStream);
    return buildMixedAudioStream([displayStream]);
  }

  const displayStream = await requestDisplayStream();
  const diagnostics = logDisplayCaptureDiagnostics(displayStream, mode);
  if (!hasAudioTrack(displayStream)) {
    showToast("画面共有音声が取れないため、マイクのみで開始します", "default", 5000);
    setStatus("recording_mic_fallback");
    state.vadRmsThreshold = vadThresholdForSource("mic");
    state.recordingAudioSource = "mic";
    state.recordingFallbackReason = "display_audio_not_found";
    state.displayStream = displayStream;
    setupDisplayCaptureVideo(displayStream);
    bindDisplayEndEvents(displayStream);
    const micStream = await requestMicStream("mic");
    state.micStream = micStream;
    return buildMixedAudioStream([micStream]);
  }
  bindDisplayEndEvents(displayStream);

  const micStream = await requestMicStream(mode);
  state.displayStream = displayStream;
  setupDisplayCaptureVideo(displayStream);
  state.micStream = micStream;
  return buildMixedAudioStream([displayStream, micStream]);
}

async function ensureSocket() {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    return state.ws;
  }

  if (state.ws && state.ws.readyState === WebSocket.CONNECTING) {
    await waitForOpen(state.ws);
    return state.ws;
  }

  const ws = new WebSocket(wsUrl());
  state.ws = ws;
  logWsEvent("connect", { url: wsUrl() });

  ws.addEventListener("message", (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }

    logWsEvent("message", { type: data.type, message: data.message || "", seq: data.seq ?? null });

    if (data.type === "conn") {
      return;
    }

    if (data.type === "info") {
      if (data.message) setStatus(String(data.message));
      if (data.sessionId) {
        state.runtimeSessionId = String(data.sessionId);
        state.runtimeSessionToken = String(data.sessionToken || "");
        updateDownloadLinks();
      }
      return;
    }

    if (data.type === "final") {
      addLogLine(
        String(data.text || ""),
        Number(data.tsStart || 0),
        Number(data.tsEnd || 0),
        Number(data.seq),
        String(data.speaker || ""),
        String(data.screenshotPath || ""),
        String(data.rawAudioPath || ""),
        String(data.audioPath || "")
      );
      return;
    }

    if (data.type === "speaker_patch") {
      applySpeakerPatch(data.segments || []);
      return;
    }

    if (data.type === "error") {
      const detail = data.detail ? ` (${data.detail})` : "";
      if (data.message === "server_busy") {
        state.degradedCaptureMode = true;
        sendWsTelemetry("server_busy_acknowledged", {
          detail: data.detail || "",
          backlog: state.pendingOutboundChunks,
        });
        showToast("サーバ処理が詰まっています。画面キャプチャを自動で抑制します", "error", 5000);
      } else if (data.message === "transcription_failed") {
        sendWsTelemetry("transcription_failed_acknowledged", {
          seq: data.seq ?? "",
          buffered: !!data.buffered,
        });
        showToast("文字起こし処理で一時エラーが発生しました", "error", 4000);
      }
      setStatus(`error: ${data.message || "unknown"}${detail}`);
    }
  });

  ws.addEventListener("close", () => {
    state.ws = null;
    logWsEvent("close");
    if (!state.recording) {
      setStatus("disconnected");
    }
  });

  ws.addEventListener("error", () => {
    logWsEvent("error");
    setStatus("socket_error");
  });

  await waitForOpen(ws);
  logWsEvent("open");
  return ws;
}

function waitForSessionReady(ws) {
  return new Promise((resolve, reject) => {
    const onMessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }

      if (data.type === "info" && data.message === "ready") {
        cleanup();
        resolve(data);
        return;
      }

      if (data.type === "error" && (data.message === "session_create_failed" || data.message === "not_started")) {
        cleanup();
        reject(new Error(String(data.detail || data.message || "session_start_failed")));
      }
    };

    const onClose = () => {
      cleanup();
      reject(new Error("websocket_closed"));
    };

    const onError = () => {
      cleanup();
      reject(new Error("websocket_error"));
    };

    const cleanup = () => {
      ws.removeEventListener("message", onMessage);
      ws.removeEventListener("close", onClose);
      ws.removeEventListener("error", onError);
    };

    ws.addEventListener("message", onMessage);
    ws.addEventListener("close", onClose);
    ws.addEventListener("error", onError);
  });
}

function waitForOpen(ws) {
  return new Promise((resolve, reject) => {
    if (ws.readyState === WebSocket.OPEN) {
      resolve();
      return;
    }

    const onOpen = () => {
      cleanup();
      resolve();
    };

    const onClose = () => {
      cleanup();
      reject(new Error("websocket_closed"));
    };

    const onError = () => {
      cleanup();
      reject(new Error("websocket_error"));
    };

    const cleanup = () => {
      ws.removeEventListener("open", onOpen);
      ws.removeEventListener("close", onClose);
      ws.removeEventListener("error", onError);
    };

    ws.addEventListener("open", onOpen);
    ws.addEventListener("close", onClose);
    ws.addEventListener("error", onError);
  });
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";

  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function ensureAudioLevelMatrix() {
  if (!audioLevelMatrixEl) return [];
  if (state.audioLevelColumns.length) return state.audioLevelColumns;

  const columns = [];
  audioLevelMatrixEl.innerHTML = "";

  for (let i = 0; i < AUDIO_LEVEL_COLUMNS; i += 1) {
    const column = document.createElement("div");
    column.className = "audio-level-column";

    const stack = document.createElement("div");
    stack.className = "audio-level-stack";

    const cells = [];

    for (let j = 0; j < AUDIO_LEVEL_SEGMENTS; j += 1) {
      const cell = document.createElement("span");
      cell.className = "audio-level-cell";
      stack.appendChild(cell);
      cells.push(cell);
    }

    column.appendChild(stack);
    audioLevelMatrixEl.appendChild(column);
    columns.push({ cells });
  }

  state.audioLevelColumns = columns;
  return columns;
}

function renderAudioLevel(level) {
  const normalized = Math.max(0, Math.min(1, Number(level) || 0));
  state.audioLevel = normalized;
  const columns = ensureAudioLevelMatrix();
  if (!columns.length) return;

  const now = performance.now();
  const center = (columns.length - 1) / 2;

  columns.forEach((column, index) => {
    const distance = Math.abs(index - center);
    const profile = 1 - distance / Math.max(1, center + 0.5);
    const ripple = 0.84 + 0.24 * Math.sin(now / 240 + index * 0.55);
    const shaped = Math.max(0, Math.min(1, normalized * (0.62 + profile * 0.55) * ripple));
    const activeCount = Math.max(
      normalized > 0.03 ? 1 : 0,
      Math.min(AUDIO_LEVEL_SEGMENTS, Math.round(shaped * AUDIO_LEVEL_SEGMENTS))
    );

    column.cells.forEach((cell, cellIndex) => {
      const active = cellIndex >= AUDIO_LEVEL_SEGMENTS - activeCount;
      cell.classList.toggle("is-active", active);
    });
  });
}

function sampleVad() {
  if (!state.vadAnalyser || !state.vadBuffer) return;

  state.vadAnalyser.getFloatTimeDomainData(state.vadBuffer);

  let sum = 0;
  for (let i = 0; i < state.vadBuffer.length; i += 1) {
    const value = state.vadBuffer[i];
    sum += value * value;
  }

  const rms = Math.sqrt(sum / state.vadBuffer.length);
  const now = performance.now();
  updateVadNoiseFloor(rms, now);
  const threshold = state.vadRmsThreshold || vadThresholdForSource(currentVadSourceMode(), state.vadNoiseFloor);
  const leveled = Math.max(0, rms - AUDIO_LEVEL_NOISE_FLOOR);
  const boosted = Math.min(1, Math.pow(leveled * AUDIO_LEVEL_GAIN, AUDIO_LEVEL_EXPONENT));
  const smoothed = state.audioLevel * 0.72 + boosted * 0.28;
  renderAudioLevel(smoothed);
  state.vadFrameCount += 1;
  if (rms >= threshold) {
    state.vadSpeechFrameCount += 1;
    state.vadLastSpeechAt = now;
  }
  updateRecordingTelemetry();
}

async function setupVad(stream) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) {
    return;
  }

  const context = new AudioContextCtor();
  if (context.state === "suspended") {
    try {
      await context.resume();
    } catch {
      // ignore
    }
  }

  const source = context.createMediaStreamSource(stream);
  const analyser = context.createAnalyser();
  analyser.fftSize = 2048;
  analyser.smoothingTimeConstant = 0.05;

  source.connect(analyser);

  state.audioContext = context;
  state.vadSource = source;
  state.vadAnalyser = analyser;
  state.vadBuffer = new Float32Array(analyser.fftSize);
  state.vadFrameCount = 0;
  state.vadSpeechFrameCount = 0;
  state.vadLastSpeechAt = performance.now();
  state.vadNoiseFloor = null;
  state.vadNoiseFloorAt = performance.now();
  state.vadRmsThreshold = vadThresholdForSource(currentVadSourceMode(), null);
  state.vadTimer = setInterval(sampleVad, VAD_SAMPLE_MS);
  updateRecordingTelemetry();
}

function stopVad() {
  if (state.vadTimer) {
    clearInterval(state.vadTimer);
    state.vadTimer = null;
  }

  if (state.vadSource) {
    try {
      state.vadSource.disconnect();
    } catch {
      // ignore
    }
    state.vadSource = null;
  }

  if (state.vadAnalyser) {
    try {
      state.vadAnalyser.disconnect();
    } catch {
      // ignore
    }
    state.vadAnalyser = null;
  }

  if (state.audioContext) {
    state.audioContext.close().catch(() => {
      // ignore
    });
    state.audioContext = null;
  }

  state.vadBuffer = null;
  state.vadFrameCount = 0;
  state.vadSpeechFrameCount = 0;
  state.vadLastSpeechAt = 0;
  state.vadNoiseFloor = null;
  state.vadNoiseFloorAt = 0;
  state.vadRmsThreshold = vadThresholdForSource(currentVadSourceMode(), null);
  state.audioLevel = 0;
  renderAudioLevel(0);
  updateRecordingTelemetry();
}

function cleanupCaptureGraph() {
  if (state.captureAutoGainTimer) {
    clearInterval(state.captureAutoGainTimer);
    state.captureAutoGainTimer = null;
  }
  state.captureSources.forEach((source) => {
    try {
      source.disconnect();
    } catch {
      // ignore
    }
  });
  state.captureSources = [];
  state.captureDestination = null;
  state.captureGainNode = null;
  state.captureMonitorAnalyser = null;
  state.captureMonitorBuffer = null;
  state.captureAutoGainLevel = 1;
  state.captureAutoGainSmoothedRms = 0;

  if (state.captureContext) {
    state.captureContext.close().catch(() => {
      // ignore
    });
    state.captureContext = null;
  }
}

function snapshotVadCounters() {
  return {
    frameCount: state.vadFrameCount,
    speechFrameCount: state.vadSpeechFrameCount,
    startedAt: performance.now(),
  };
}

function buildVadDecision(snapshot, endedAt, sourceMode) {
  if (!state.vadAnalyser) {
    return {
      enabled: false,
      speechRatio: 1,
      activeMs: state.chunkMs,
      silenceMs: 0,
      skip: false,
    };
  }

  const totalFrames = Math.max(1, state.vadFrameCount - snapshot.frameCount);
  const speechFrames = Math.max(0, state.vadSpeechFrameCount - snapshot.speechFrameCount);
  const policy = vadSourceCutPolicy(sourceMode);
  const chunkEndAt = Number.isFinite(endedAt) ? endedAt : performance.now();
  const lastSpeechAt = state.vadLastSpeechAt || state.segmentStartedAt || snapshot.startedAt || chunkEndAt;

  const speechRatio = speechFrames / totalFrames;
  const activeMs = speechFrames * VAD_SAMPLE_MS;
  const silenceMs = Math.max(0, chunkEndAt - lastSpeechAt);
  const skip = speechRatio < VAD_MIN_SPEECH_RATIO && activeMs < VAD_MIN_ACTIVE_MS;

  return {
    enabled: true,
    speechRatio,
    activeMs,
    silenceMs,
    sourceMode: normalizeAudioSource(sourceMode),
    policy,
    skip,
  };
}

function shouldSkipChunkByVad(durationMs, vadDecision) {
  if (!CLIENT_VAD_DROP_ENABLED) {
    return false;
  }
  if (!vadDecision?.enabled || !vadDecision.skip) {
    return false;
  }

  const safeDurationMs = Number.isFinite(durationMs) ? Math.max(0, durationMs) : 0;
  if (safeDurationMs >= 4_000) {
    return false;
  }

  return vadDecision.speechRatio < 0.01 && vadDecision.activeMs < 80;
}

async function sendChunk(blob, mimeType, durationMsOverride, vadDecision) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;

  let durationMs = Number(durationMsOverride);
  if (!Number.isFinite(durationMs)) {
    durationMs = state.chunkMs;
  }
  durationMs = Math.max(200, Math.round(durationMs));

  const offsetMs = state.offsetMs;
  state.offsetMs += durationMs;

  if (!blob || blob.size === 0) return;

  if (shouldSkipChunkByVad(durationMs, vadDecision)) {
    return;
  }

  state.pendingOutboundChunks += 1;
  updateBackpressureState();
  const seq = state.seq++;
  state.recordedChunkCount += 1;
  try {
    const buffer = await blob.arrayBuffer();
    const audio = arrayBufferToBase64(buffer);
    const screenshot = await captureDisplayScreenshot();

    state.ws.send(
      JSON.stringify({
        type: "chunk",
        seq,
        offsetMs,
        durationMs,
        mimeType: mimeType || blob.type || "audio/webm",
        audio,
        screenshot: screenshot?.data || "",
        screenshotMimeType: screenshot?.mimeType || "",
        speechRatio: Number.isFinite(vadDecision?.speechRatio) ? vadDecision.speechRatio : null,
        activeMs: Number.isFinite(vadDecision?.activeMs) ? vadDecision.activeMs : null,
        silenceMs: Number.isFinite(vadDecision?.silenceMs) ? vadDecision.silenceMs : null,
      })
    );
    logWsEvent("send_chunk", {
      seq,
      durationMs,
      offsetMs,
      bytes: blob.size,
      screenshot: screenshot ? "sent" : "skipped",
      backlog: state.pendingOutboundChunks,
    });
  } finally {
    state.pendingOutboundChunks = Math.max(0, state.pendingOutboundChunks - 1);
    updateBackpressureState();
  }
}

function clearChunkTimer() {
  if (state.chunkTimer) {
    clearTimeout(state.chunkTimer);
    state.chunkTimer = null;
  }
}

function minSegmentMs() {
  const policy = vadSourceCutPolicy(currentVadSourceMode());
  return Math.max(Math.min(VAD_SEGMENT_MIN_MS, state.chunkMs), Math.round(state.chunkMs * policy.minSegmentRatio));
}

function chunkHardMaxMs() {
  const mode = currentVadSourceMode();
  const sourceExtra = mode === "display" ? 2_500 : mode === "both" ? 1_500 : 0;
  return Math.max(state.chunkMs + 2_000, state.chunkMs + VAD_SOFT_CUT_GRACE_MS + sourceExtra);
}

function shouldCutChunkOnSilence(options = {}) {
  if (!state.vadAnalyser || !state.segmentStartedAt) return false;

  const policy = vadSourceCutPolicy(currentVadSourceMode());
  const relaxed = !!options.relaxed;
  const now = performance.now();
  const elapsedMs = now - state.segmentStartedAt;
  if (elapsedMs < minSegmentMs()) return false;

  const silenceMs = Math.max(0, now - (state.vadLastSpeechAt || state.segmentStartedAt));
  const minSilenceMs = relaxed
    ? Math.max(220, Math.min(policy.minSilenceMs, Math.round(policy.minSilenceMs * 0.55)))
    : policy.minSilenceMs;
  if (silenceMs < minSilenceMs) return false;

  const silenceRatio = relaxed ? Math.max(0.1, policy.silenceRatio * 0.72) : policy.silenceRatio;
  return silenceMs >= Math.min(policy.maxSilenceMs, Math.max(minSilenceMs + 160, Math.round(elapsedMs * silenceRatio)));
}

function requestChunkFlush(recorder) {
  if (!state.recording) return;
  if (state.recorder !== recorder) return;
  if (recorder.state !== "recording") return;

  try {
    recorder.stop();
  } catch {
    // ignore
  }
}

function scheduleChunkStop(recorder) {
  clearChunkTimer();
  const check = () => {
    if (!state.recording || state.recorder !== recorder || recorder.state !== "recording") {
      clearChunkTimer();
      return;
    }

    const elapsedMs = Math.max(0, performance.now() - state.segmentStartedAt);
    if (elapsedMs >= chunkHardMaxMs()) {
      requestChunkFlush(recorder);
      return;
    }

    if (shouldCutChunkOnSilence({ relaxed: elapsedMs >= state.chunkMs })) {
      requestChunkFlush(recorder);
      return;
    }

    state.chunkTimer = setTimeout(check, Math.min(250, Math.max(120, VAD_SAMPLE_MS)));
  };

  state.chunkTimer = setTimeout(check, Math.min(250, Math.max(120, VAD_SAMPLE_MS)));
}

function startRecorderCycle() {
  if (!state.recording || !state.stream) return;

  const recorder = new MediaRecorder(state.stream, state.recorderOptions || {});
  state.recorder = recorder;
  const cycleStartedAt = performance.now();
  state.segmentStartedAt = cycleStartedAt;
  state.vadLastSpeechAt = cycleStartedAt;
  const vadSnapshot = snapshotVadCounters();

  recorder.addEventListener("dataavailable", (event) => {
    const eventEndedAt = performance.now();
    const durationMs = Math.max(200, Math.round(eventEndedAt - cycleStartedAt));
    const vadDecision = buildVadDecision(vadSnapshot, eventEndedAt, state.recordingAudioSource);

    state.pendingSendChain = state.pendingSendChain
      .then(() => sendChunk(event.data, state.recorderMimeType, durationMs, vadDecision))
      .catch((err) => {
        setStatus(`chunk_error: ${err.message}`);
      });
  });

  recorder.addEventListener("error", () => {
    setStatus("recorder_error");
    state.recording = false;
    finalizeStop();
  });

  recorder.addEventListener("stop", () => {
    if (state.recorder === recorder) {
      state.recorder = null;
    }
    if (state.recording) {
      startRecorderCycle();
      return;
    }
    finalizeStop();
  });

  recorder.start();
  scheduleChunkStop(recorder);
}

async function finalizeStop() {
  if (state.finalizingStop) return;
  state.finalizingStop = true;
  clearChunkTimer();

  try {
    await state.pendingSendChain;
  } finally {
    state.pendingSendChain = Promise.resolve();
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "stop" }));
      logWsEvent("send_stop", { sessionId: state.runtimeSessionId || "" });
    }
    cleanupMedia();
    setUiRecording(false);
    state.segmentStartedAt = 0;
    state.recordingStartedAt = 0;
    state.recordingRequestedAudioSource = "mic";
    state.recordingFallbackReason = "";
    state.recordedChunkCount = 0;
    updateRecordingTelemetry();
    state.finalizingStop = false;
    updateSaveControls();
  }
}

async function startRecording() {
  if (state.recording) return;
  if (!canUseWorkspace()) {
    showToast("ログインが必要です", "error");
    setAppLocked(true);
    loginEmailEl?.focus();
    return;
  }

  const selectedChunkSeconds = applyChunkSeconds(chunkSecondsEl.value || CHUNK_DEFAULT_SECONDS);
  state.chunkMs = selectedChunkSeconds * 1000;
  const selectedAudioSource = applyAudioSource(audioSourceEl?.value || "mic");
  state.recordingAudioSource = selectedAudioSource;
  state.recordingRequestedAudioSource = selectedAudioSource;
  state.recordingFallbackReason = "";
  state.recordingStartedAt = performance.now();
  state.recordedChunkCount = 0;
  state.pendingOutboundChunks = 0;
  state.maxObservedBacklog = 0;
  state.degradedCaptureMode = false;
  state.screenshotEncodeInFlight = false;
  state.screenshotLastCapturedAt = 0;
  state.screenshotSkippedCount = 0;
  state.lastTelemetryBacklogBucket = "";
  state.lastScreenshotSkipReason = "";
  state.lastScreenshotSkipSentAt = 0;
  state.seq = 0;
  state.offsetMs = 0;
  state.runtimeSessionId = "";
  state.runtimeSessionToken = "";
  state.history.selectedId = null;
  state.finalizingStop = false;
  state.pendingSendChain = Promise.resolve();
  state.log = [];
  state.segments = [];
  state.logAutoScrollEnabled = true;
  renderEmptyTranscriptState();
  setSummary("", "未生成");
  setProofread("", "未生成");
  resetCurrentSaveState();
  renderHistoryList();
  clearChunkTimer();

  try {
    const health = await loadCapabilities();
    if (!health?.asrReady || !health?.model) {
      throw new Error("asr_not_ready");
    }

    const ws = await ensureSocket();
    const stream = await prepareInputStream(selectedAudioSource);
    if (!hasAudioTrack(stream)) {
      throw new Error("audio_track_not_found");
    }

    state.stream = stream;
    await setupVad(stream);

    const mimeType = selectMimeType();
    state.recorderMimeType = mimeType || "audio/webm";
    state.recorderOptions = mimeType ? { mimeType } : {};
    const diarizationOptions = resolveDiarizationStartOptions();

    const readyPromise = waitForSessionReady(ws);
    const startPayload = {
      type: "start",
      sessionId: generateSessionSeed(),
      language: selectedLanguage(),
      audioSource: selectedAudioSource,
      prompt: promptEl.value.trim(),
      sharedVocabulary: String(sharedVocabularyEl?.value || state.sharedVocabulary || "").trim(),
      diarizationEnabled: !!(state.diarizationAvailable && state.diarizationEnabled),
      diarizationNumSpeakers: diarizationOptions.diarizationNumSpeakers,
      diarizationMinSpeakers: diarizationOptions.diarizationMinSpeakers,
      diarizationMaxSpeakers: diarizationOptions.diarizationMaxSpeakers,
    };
    ws.send(JSON.stringify(startPayload));
    logWsEvent("send_start", {
      sessionId: startPayload.sessionId,
      language: startPayload.language || "auto",
      audioSource: startPayload.audioSource,
    });
    await readyPromise;

    state.recordingStartedAt = performance.now();
    setUiRecording(true);
    const effectiveAudioSource = normalizeAudioSource(state.recordingAudioSource || selectedAudioSource);
    updateRecordingTelemetry();
    if (effectiveAudioSource === "display") {
      setStatus("recording_display_audio");
    } else if (effectiveAudioSource === "both") {
      setStatus("recording_mic_and_display");
    } else {
      setStatus("recording_mic");
    }
    startRecorderCycle();
  } catch (err) {
    const name = err?.name || "";
    const message = err?.message || "unknown_error";
    if (message === "display_audio_not_found") {
      const diagnostics = err?.diagnostics || {};
      const displaySurface = diagnostics.displaySurface || "unknown";
      const audioTrackCount = Number(diagnostics.audioTrackCount || 0);
      setStatus(
        `start_failed: 画面共有の音声が見つかりません (surface=${displaySurface}, audioTracks=${audioTrackCount})`
      );
      showToast(
        "画面共有の音声トラックを取得できませんでした。Chrome/Edge のタブ共有は通りやすいですが、Skype/Webex のアプリ画面はブラウザ制約で音声が渡らないことがあります。",
        "error",
        9000
      );
    } else if (name === "NotAllowedError") {
      setStatus("start_failed: 権限が拒否されました");
    } else if (message === "asr_not_ready") {
      setStatus("start_failed: /api/health で ASR モデルが確認できません");
    } else {
      setStatus(`start_failed: ${message}`);
    }
    cleanupMedia();
    setUiRecording(false);
    updateRecordingTelemetry();
  }
}

function stopRecording() {
  if (!state.recording) return;
  setStatus("stopping");
  state.recording = false;
  clearChunkTimer();

  try {
    if (state.recorder && state.recorder.state === "recording") {
      state.recorder.stop();
    } else {
      finalizeStop();
    }
  } catch {
    finalizeStop();
  }
}

function cleanupMedia() {
  if (state.recorder) {
    state.recorder.ondataavailable = null;
    state.recorder.onstop = null;
    state.recorder = null;
  }
  state.recorderOptions = null;

  stopVad();
  cleanupCaptureGraph();

  const streams = [state.stream, state.micStream, state.displayStream];
  const seenTracks = new Set();
  streams.forEach((stream) => {
    if (!stream) return;
    stream.getTracks().forEach((track) => {
      if (seenTracks.has(track.id)) return;
      seenTracks.add(track.id);
      try {
        track.stop();
      } catch {
        // ignore
      }
    });
  });

  state.stream = null;
  state.micStream = null;
  state.displayStream = null;
  if (state.displayCaptureVideo) {
    try {
      state.displayCaptureVideo.pause();
      state.displayCaptureVideo.srcObject = null;
    } catch {
      // ignore
    }
  }
  state.displayCaptureVideo = null;
  state.screenshotCanvas = null;
  state.screenshotDiffCanvas = null;
  state.previousScreenshotSignature = null;
}

async function copyAll() {
  const text = extractTranscriptText();
  if (!text) {
    showToast("コピーする内容がありません", "error");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    copyBtn.classList.add("is-success");
    showToast("コピーしました", "success");
    setStatus("copied");
    setTimeout(() => copyBtn.classList.remove("is-success"), 1200);
  } catch {
    showToast("コピーに失敗しました", "error");
    setStatus("copy_failed");
  }
}

async function copyProofread() {
  const text = state.proofread.trim();
  if (!text) {
    showToast("コピーする校正結果がありません", "error");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    copyProofreadBtn.classList.add("is-success");
    showToast("校正結果をコピーしました", "success");
    setStatus("proofread_copied");
    setTimeout(() => copyProofreadBtn.classList.remove("is-success"), 1200);
  } catch {
    showToast("コピーに失敗しました", "error");
    setStatus("copy_failed");
  }
}

async function proofreadAll() {
  if (state.proofreadInFlight) {
    return;
  }
  if (!canUseWorkspace()) {
    showToast("ログインが必要です", "error");
    setAppLocked(true);
    loginEmailEl?.focus();
    return;
  }
  setStatus("proofread_requested");

  const text = extractTranscriptText();
  if (!text) {
    showToast("校正する文字起こしがありません", "error");
    setStatus("proofread_no_text");
    return;
  }

  if (!state.proofreadAvailable) {
    showToast("校正機能がサーバーで無効です", "error");
    setProofread("", "利用不可");
    setStatus("proofread_unavailable");
    return;
  }

  state.proofreadInFlight = true;
  setProofreadButtonBusy(true);
  proofreadMetaEl.textContent = "処理中...";
  setStatus("proofreading");
  showToast(`${proofreadActionLabel()}中...`, "default", 5000);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000);
  const proofreadStartedAt = performance.now();

  try {
    console.info("[whistx][api] request", { method: "POST", url: "/api/proofread/stream" });
    const response = await fetch("/api/proofread/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        language: selectedLanguage(),
        mode: state.proofreadMode,
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
        proofreadMetaEl.textContent = chunkCount > 1 ? `処理中... 0/${chunkCount}` : "処理中...";
        return;
      }

      if (eventType === "chunk_start") {
        currentChunk = Number(event.chunkIndex || currentChunk || 0);
        proofreadMetaEl.textContent = chunkCount > 1 ? `処理中... ${currentChunk}/${chunkCount}` : "処理中...";
        return;
      }

      if (eventType === "delta") {
        correctedText += String(event.delta || "");
        const now = Date.now();
        if (now - lastRenderAt >= 120 || correctedText.endsWith("\n")) {
          setProofread(correctedText, chunkCount > 1 ? `処理中... ${currentChunk}/${chunkCount}` : "処理中...");
          lastRenderAt = now;
        }
        return;
      }

      if (eventType === "final_text") {
        correctedText = String(event.text || "").trim();
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

    setProofread(correctedText, metaParts.join(" | ") || "生成完了");
    setStatus("proofread_done");
    showToast(`${proofreadActionLabel()}を生成しました`, "success");
  } catch (err) {
    if (isLoginRequiredError(err)) {
      showToast("ログインが必要です", "error");
      setAppLocked(true);
      loginEmailEl?.focus();
      return;
    }
    const message = err?.name === "AbortError" ? "request_timeout" : err?.message || "unknown_error";
    showToast(`${proofreadActionLabel()}に失敗: ${message}`, "error");
    setProofread(`${proofreadActionLabel()}に失敗しました。\n${message}`, "エラー");
    setStatus(`proofread_failed: ${message}`);
  } finally {
    clearTimeout(timeoutId);
    state.proofreadInFlight = false;
    setProofreadButtonBusy(false);
  }
}

async function summarizeAll() {
  if (!canUseWorkspace()) {
    showToast("ログインが必要です", "error");
    setAppLocked(true);
    loginEmailEl?.focus();
    return;
  }
  const text = extractTranscriptText();
  if (!text) {
    showToast("要約する文字起こしがありません", "error");
    setStatus("summary_no_text");
    return;
  }

  summaryBtn.disabled = true;
  if (summaryBtnLabelEl) {
    summaryBtnLabelEl.textContent = "生成中...";
  }
  setStatus("summarizing");
  showToast("要約を生成中...", "default", 5000);

  try {
    const payload = await fetchJson("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        language: selectedLanguage(),
        prompt: String(summaryPromptEl?.value || "").trim(),
      }),
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

    setSummary(summaryText, metaParts.join(" | ") || "生成完了");
    setStatus("summarized");
    showToast("要約を生成しました", "success");
  } catch (err) {
    if (isLoginRequiredError(err)) {
      showToast("ログインが必要です", "error");
      setAppLocked(true);
      loginEmailEl?.focus();
      return;
    }
    showToast(`要約に失敗: ${err.message}`, "error");
    setStatus(`summary_failed: ${err.message}`);
  } finally {
    summaryBtn.disabled = false;
    if (summaryBtnLabelEl) {
      summaryBtnLabelEl.textContent = "生成";
    }
  }
}

function clearView() {
  if (!canUseWorkspace()) {
    showToast("ログインが必要です", "error");
    setAppLocked(true);
    loginEmailEl?.focus();
    return;
  }
  state.log = [];
  state.segments = [];
  state.logAutoScrollEnabled = true;
  state.history.selectedId = null;
  state.viewingHistoryId = null;
  state.savedHistoryId = null;

  renderEmptyTranscriptState();

  updateSegmentCount();
  setSummary("", "未生成");
  setProofread("", "未生成");
  updateDownloadLinks();
  updateSaveControls();
  renderHistoryList();
  showToast("クリアしました", "success");
}

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
  if (state.recording) {
    stopRecording();
  } else {
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

if (sidebarToggleBtn) {
  sidebarToggleBtn.addEventListener("click", () => {
    applySidebarOpen(!state.sidebarOpen);
  });
}

if (sidebarCloseBtn) {
  sidebarCloseBtn.addEventListener("click", () => {
    applySidebarOpen(false);
  });
}

if (sidebarBackdropEl) {
  sidebarBackdropEl.addEventListener("click", () => {
    applySidebarOpen(false);
  });
}

if (loginBtn) {
  loginBtn.addEventListener("click", () => {
    applySidebarOpen(true);
    loginEmailEl?.focus();
  });
}

if (logoutBtn) {
  logoutBtn.addEventListener("click", () => {
    logout();
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
  captureScreenshotsEnabledEl.addEventListener("change", () => {
    applyCaptureScreenshotsEnabled(captureScreenshotsEnabledEl.checked);
  });
}

if (screenshotDiffSkipEnabledEl) {
  screenshotDiffSkipEnabledEl.addEventListener("change", () => {
    applyScreenshotDiffSkipEnabled(screenshotDiffSkipEnabledEl.checked);
  });
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.sidebarOpen) {
    applySidebarOpen(false);
  }
  if (event.key === "Escape") {
    applyHistoryDrawerOpen(false);
    hideScreenshotModal();
    closeAdminQueueModal();
  }
});

if (helpBtn) {
  helpBtn.addEventListener("click", () => {
    openHelpModal();
  });
}

if (helpModalCloseEl) {
  helpModalCloseEl.addEventListener("click", () => {
    closeHelpModal();
  });
}

if (adminQueueCloseEl) {
  adminQueueCloseEl.addEventListener("click", () => {
    closeAdminQueueModal();
  });
}

if (adminQueueModalEl) {
  adminQueueModalEl.addEventListener("click", (event) => {
    if (event.target === adminQueueModalEl || event.target?.matches?.("[data-admin-modal-close]")) {
      closeAdminQueueModal();
    }
  });
}

if (helpModalEl) {
  helpModalEl.addEventListener("click", (event) => {
    if (event.target === helpModalEl || event.target?.matches?.("[data-help-modal-close]")) {
      closeHelpModal();
    }
  });
}

if (summaryPromptToggleBtn) {
  summaryPromptToggleBtn.addEventListener("click", () => {
    applySummaryPromptEditorOpen(!state.summaryPromptEditorOpen);
  });
}

window.addEventListener("message", (event) => {
  if (event.origin !== window.location.origin) return;
  if (event.data?.type === "whistx:help-close") {
    closeHelpModal();
  }
});

window.addEventListener("resize", () => {
  updateWorkspaceGridTemplate();
  if (window.innerWidth > 1100) {
    applyHistoryDrawerOpen(false);
  }
  applyHistoryCollapsed(state.historyCollapsed, { persist: false });
  applyActiveAiPanel(state.activeAiPanel);
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

function renderPromptTemplateButtons(rawTemplates) {
  if (!promptTemplateButtonsEl || !promptEl) return;
  const templates = Array.isArray(rawTemplates) && rawTemplates.length > 0
    ? rawTemplates
    : state.promptTemplates;

  promptTemplateButtonsEl.innerHTML = "";
  templates.forEach((template, index) => {
    const label = String(template?.label || `Template ${index + 1}`).trim();
    const content = String(template?.content || "").trim();
    if (!content) return;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "prompt-template-btn";
    button.textContent = label;
    button.addEventListener("click", () => {
      promptEl.value = content;
      promptEl.dispatchEvent(new Event("input", { bubbles: true }));
      promptEl.focus();
      showToast(`${label}を入力しました`, "success");
    });
    promptTemplateButtonsEl.appendChild(button);
  });
}

function applySharedVocabulary(payload, options = {}) {
  const preserveDraft = options.preserveDraft === true;
  const text = String(payload?.items || "").trim();
  state.sharedVocabulary = text;
  state.sharedVocabularyUpdatedAt = String(payload?.updatedAt || "").trim();
  state.sharedVocabularyUpdatedBy = String(payload?.updatedBy || "").trim();
  if (sharedVocabularyEl && !preserveDraft) {
    sharedVocabularyEl.value = text;
  }
  updateSharedVocabularyMeta();
}

function updateSharedVocabularyMeta() {
  const authenticated = !!state.auth.authenticated;
  const isGuest = !!state.auth.isGuest;
  if (sharedVocabularySaveBtn) {
    sharedVocabularySaveBtn.disabled = !authenticated || isGuest || state.sharedVocabularySaving;
    sharedVocabularySaveBtn.textContent = state.sharedVocabularySaving ? "保存中..." : "全体に保存";
    sharedVocabularySaveBtn.title = isGuest ? "ゲストでは全体用語辞典を更新できません" : authenticated ? "" : "ログインが必要です";
  }
  if (!sharedVocabularyMetaEl) return;
  if (!state.sharedVocabulary) {
    sharedVocabularyMetaEl.textContent = "全体用語辞典は未設定です";
    return;
  }
  const parts = [];
  if (state.sharedVocabularyUpdatedAt) {
    parts.push(`更新: ${new Date(state.sharedVocabularyUpdatedAt).toLocaleString("ja-JP")}`);
  }
  if (state.sharedVocabularyUpdatedBy) {
    parts.push(`更新者: ${state.sharedVocabularyUpdatedBy}`);
  }
  sharedVocabularyMetaEl.textContent = parts.join(" / ") || "全体用語辞典を使用します";
}

async function loadSharedGlossary() {
  logClientEvent("shared_glossary.load.start");
  try {
    const payload = await fetchSharedGlossaryRequest();
    applySharedVocabulary(payload);
    logClientEvent("shared_glossary.load.success", { hasItems: !!String(payload?.items || "").trim() });
  } catch {
    applySharedVocabulary({ items: "", updatedAt: "", updatedBy: "" });
    logClientEvent("shared_glossary.load.fallback");
  }
}

async function saveSharedGlossary() {
  if (state.auth.isGuest) {
    showToast("ゲストでは全体用語辞典を更新できません", "error");
    return;
  }
  if (!state.auth.authenticated) {
    showToast("ログインが必要です", "error");
    loginEmailEl?.focus();
    return;
  }
  state.sharedVocabularySaving = true;
  updateSharedVocabularyMeta();
  try {
    const payload = await saveSharedGlossaryRequest(String(sharedVocabularyEl?.value || "").trim());
    applySharedVocabulary(payload);
    showToast("全体用語辞典を保存しました", "success");
  } catch (error) {
    showToast(`全体用語辞典の保存に失敗: ${error.message}`, "error");
  } finally {
    state.sharedVocabularySaving = false;
    updateSharedVocabularyMeta();
  }
}

// Download link feedback
[dlTxt, dlJsonl, dlZip].forEach((link) => {
  link.addEventListener("click", () => {
    const format = link.textContent;
    if (link.href && link.href !== "#") {
      link.classList.add("is-downloaded");
      setTimeout(() => link.classList.remove("is-downloaded"), 800);
    }
  });
});

/* --------------------------------------------------------------------------
   Theme Toggle - Light/Dark Mode
   -------------------------------------------------------------------------- */
function applyTheme(theme, options = {}) {
  const persist = options.persist !== false;
  const normalized = normalizeTheme(theme);
  document.documentElement.setAttribute("data-theme", normalized);
  if (persist) {
    try {
      writeStoredValue("whistx_theme", normalized);
    } catch {
      // ignore
    }
  }
  updateThemeColorMeta(normalized);
  return normalized;
}

function handleAuthErrorFromLocation() {
  const url = new URL(window.location.href);
  const authError = String(url.searchParams.get("authError") || "");
  if (!authError) return;
  if (authError === "approval_required") {
    showToast("Keycloak ログイン後も管理者承認が必要です", "error");
  } else if (authError === "keycloak_state") {
    showToast("Keycloak ログインの状態確認に失敗しました", "error");
  } else if (authError === "keycloak_email_not_verified") {
    showToast("Keycloak 側でメールアドレス確認が完了していません", "error");
  } else if (authError === "keycloak_account_link_required") {
    showToast("同じメールアドレスの既存アカウントがあります。管理者に Keycloak 連携を依頼してください", "error", 7000);
  } else if (authError === "keycloak_identity_conflict") {
    showToast("Keycloak アカウントの紐付けに競合があります", "error", 7000);
  } else if (authError === "keycloak_failed") {
    showToast("Keycloak ログインに失敗しました", "error");
  }
  url.searchParams.delete("authError");
  window.history.replaceState({}, document.title, `${url.pathname}${url.search}${url.hash}`);
}

function initTheme() {
  let savedTheme = "";
  try {
    savedTheme = readStoredValue("whistx_theme", "") || "";
  } catch {
    savedTheme = "";
  }
  applyTheme(resolveInitialTheme(savedTheme, window.matchMedia("(prefers-color-scheme: dark)").matches), { persist: false });
}

function updateThemeColorMeta(theme) {
  const color = themeMetaColor(theme);
  const metaThemeColors = document.querySelectorAll('meta[name="theme-color"]');
  metaThemeColors.forEach((meta) => {
    meta.setAttribute("content", color);
  });
}

function toggleTheme() {
  const currentTheme = normalizeTheme(document.documentElement.getAttribute("data-theme") || "light");
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  applyTheme(newTheme, { persist: true });

  // Show toast
  const themeName = newTheme === "dark" ? "ダークモード" : "ライトモード";
  showToast(`${themeName}に切り替えました`, "success");
}

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

async function loadCapabilities() {
  logClientEvent("capabilities.load.start");
  try {
    const health = await fetchCapabilities();
    renderBanners(health.banners);
    applyBranding(health.uiBrandTitle, health.uiBrandTagline);
    if (Array.isArray(health.uiPromptTemplates) && health.uiPromptTemplates.length > 0) {
      state.promptTemplates = health.uiPromptTemplates
        .map((template, index) => ({
          id: String(template?.id || `template-${index + 1}`),
          label: String(template?.label || `Template ${index + 1}`),
          content: String(template?.content || "").trim(),
        }))
        .filter((template) => template.content);
    }
    renderPromptTemplateButtons(state.promptTemplates);
    state.asrAvailable = !!health.asrReady && !!health.model;
    state.proofreadAvailable = !!health.proofreadModel;
    state.diarizationAvailable = !!health.diarizationEnabled;
    state.selfSignupEnabled = !!health.selfSignupEnabled;
    state.auth.keycloakEnabled = !!health.keycloakEnabled;
    state.auth.keycloakButtonLabel = String(health.keycloakButtonLabel || "Keycloakでログイン");
    renderAuthState();

    if (!state.proofreadAvailable) {
      setProofread(
        "校正・翻訳機能が無効です。\nサーバーの API キー設定（PROOFREAD_API_KEY / SUMMARY_API_KEY / ASR_API_KEY）を確認してください。",
        "利用不可"
      );
      if (proofreadBtn) {
        proofreadBtn.title = "校正・翻訳機能はサーバーで無効";
      }
    } else if (proofreadBtn) {
      proofreadBtn.title = proofreadActionLabel();
    }

    if (!state.asrAvailable) {
      setStatus("asr_unavailable");
    }

    if (diarizationToggleEl) {
      diarizationToggleEl.disabled = !state.diarizationAvailable;
      diarizationToggleEl.title = state.diarizationAvailable
        ? "話者分離を有効/無効"
        : "サーバーで話者分離は無効";
    }
    state.diarizationSpeakerCap = Math.max(
      DIARIZATION_SPEAKER_MIN,
      Number(health.diarizationSpeakerCap || DIARIZATION_SPEAKER_MAX)
    );

    if (!state.hasSavedDiarizationSpeakerSettings) {
      const defaultNum = Number(health.diarizationDefaultNumSpeakers || 0);
      const defaultMin = Number(health.diarizationDefaultMinSpeakers || 0);
      const defaultMax = Number(health.diarizationDefaultMaxSpeakers || 0);

      let mode = "auto";
      if (defaultNum > 0) {
        mode = "fixed";
      } else if (defaultMin > 0 || defaultMax > 0) {
        mode = "range";
      }

      applyDiarizationSpeakerSettings(
        {
          mode,
          count: defaultNum > 0 ? defaultNum : state.diarizationSpeakerCount,
          min: defaultMin > 0 ? defaultMin : state.diarizationMinSpeakers,
          max: defaultMax > 0 ? defaultMax : state.diarizationMaxSpeakers,
        },
        { persist: false }
      );
    } else {
      updateDiarizationSpeakerUi();
    }

    applyDiarizationEnabled(state.diarizationEnabled, { persist: false });
    logClientEvent("capabilities.load.success", {
      asrReady: !!health.asrReady,
      proofreadReady: !!health.proofreadModel,
      diarizationEnabled: !!health.diarizationEnabled,
    });
    return health;
  } catch {
    // ignore capability check errors
    logClientEvent("capabilities.load.failed");
    return null;
  }
}

async function loadAuthState() {
  logClientEvent("auth_state.load.start");
  try {
    const payload = await fetchAuthState();
    if (!payload?.authenticated) {
      state.auth.bootstrapAdminRequired = !!payload?.bootstrapAdminRequired;
      state.selfSignupEnabled = !!payload?.selfSignupEnabled;
      state.auth.historyRetentionDays = Math.max(1, Number(payload?.historyRetentionDays || 7));
      state.auth.keycloakEnabled = !!payload?.keycloakEnabled;
      state.auth.keycloakButtonLabel = String(payload?.keycloakButtonLabel || "Keycloakでログイン");
      try {
        state.auth.isGuest = readGuestMode();
      } catch {
        state.auth.isGuest = false;
      }
      if (!state.auth.isGuest && !state.auth.bootstrapAdminRequired) {
        state.auth.isGuest = true;
        persistGuestMode(true);
      }
      setAppLocked(!canUseWorkspace());
      renderAuthState();
      logClientEvent("auth_state.load.success", { authenticated: false, guest: !!state.auth.isGuest });
      return;
    }
    state.auth.authenticated = !!payload.authenticated;
    state.auth.isGuest = false;
    state.auth.user = payload.user || null;
    state.selfSignupEnabled = !!payload.selfSignupEnabled;
    state.auth.historyRetentionDays = Math.max(1, Number(payload.historyRetentionDays || 7));
    state.auth.bootstrapAdminRequired = !!payload.bootstrapAdminRequired;
    state.auth.pendingApprovalCount = Number(payload.pendingApprovalCount || 0);
    state.auth.keycloakEnabled = !!payload.keycloakEnabled;
    state.auth.keycloakButtonLabel = String(payload.keycloakButtonLabel || "Keycloakでログイン");
    if (state.auth.authenticated) {
      persistGuestMode(false);
    } else {
      try {
        state.auth.isGuest = readGuestMode();
      } catch {
        state.auth.isGuest = false;
      }
    }
    setAppLocked(!canUseWorkspace());
    renderAuthState();
    if (state.auth.authenticated) {
      await loadHistoryList();
      if (state.auth.user?.isAdmin) {
        await loadPendingUsers();
      }
    } else if (state.auth.isGuest) {
      state.history.items = [];
      state.history.selectedId = null;
      renderHistoryList();
    } else {
      state.history.items = [];
      state.history.selectedId = null;
      renderHistoryList();
      if (state.auth.bootstrapAdminRequired) {
        bootstrapDisplayNameEl?.focus();
      } else {
        loginEmailEl?.focus();
      }
    }
    logClientEvent("auth_state.load.success", {
      authenticated: !!state.auth.authenticated,
      guest: !!state.auth.isGuest,
      isAdmin: !!state.auth.user?.isAdmin,
    });
  } catch {
    // ignore auth bootstrap errors
    try {
      state.auth.isGuest = readGuestMode();
      if (!state.auth.isGuest && !state.auth.bootstrapAdminRequired) {
        state.auth.isGuest = true;
        persistGuestMode(true);
      }
      setAppLocked(!canUseWorkspace());
      renderAuthState();
      logClientEvent("auth_state.load.fallback", { guest: !!state.auth.isGuest });
    } catch {
      // ignore
      logClientEvent("auth_state.load.failed");
    }
  }
}

async function login() {
  const email = String(loginEmailEl?.value || "").trim();
  const password = String(loginPasswordEl?.value || "");
  if (!email || !password) {
    showToast("メールアドレスとパスワードを入力してください", "error");
    return;
  }

  try {
    const payload = await loginRequest({ email, password });
    state.auth.authenticated = true;
    state.auth.isGuest = false;
    state.auth.user = payload.user || null;
    state.auth.bootstrapAdminRequired = false;
    state.auth.pendingApprovalCount = Number(payload.pendingApprovalCount || 0);
    persistGuestMode(false);
    setAppLocked(false);
    renderAuthState();
    await loadHistoryList();
    if (state.auth.user?.isAdmin) {
      await loadPendingUsers();
    }
    showToast("ログインしました", "success");
  } catch (error) {
    const payload = error?.payload || {};
    if (payload.error === "approval_required") {
      showToast("管理者の承認後にログインできます", "error");
    } else if (payload.error === "too_many_login_attempts") {
      showToast(`ログイン試行が多すぎます。${Number(payload.retryAfterSec || 0)}秒後に再試行してください`, "error", 5000);
    } else {
      showToast("ログインに失敗しました", "error");
    }
  }
}

async function bootstrapAdmin() {
  const displayName = String(bootstrapDisplayNameEl?.value || "").trim();
  const email = String(bootstrapEmailEl?.value || "").trim();
  const password = String(bootstrapPasswordEl?.value || "");
  if (!email || password.length < 8) {
    showToast("メールアドレスと8文字以上のパスワードを入力してください", "error");
    return;
  }

  try {
    const payload = await bootstrapAdminRequest({ email, password, displayName });
    state.auth.authenticated = true;
    state.auth.isGuest = false;
    state.auth.user = payload.user || null;
    state.auth.bootstrapAdminRequired = false;
    state.auth.pendingApprovalCount = 0;
    persistGuestMode(false);
    setAppLocked(false);
    renderAuthState();
    await loadHistoryList();
    await loadPendingUsers();
    showToast("管理者アカウントを作成しました", "success");
  } catch (error) {
    const payload = error?.payload || {};
    showToast(payload.error === "email_already_exists" ? "既に存在するメールアドレスです" : "管理者作成に失敗しました", "error");
  }
}

async function registerAccount() {
  if (!state.selfSignupEnabled) {
    showToast("新規登録は無効です", "error");
    return;
  }
  const email = String(registerEmailEl?.value || "").trim();
  const password = String(registerPasswordEl?.value || "");
  const displayName = String(registerDisplayNameEl?.value || "").trim();
  if (!email || password.length < 8) {
    showToast("メールアドレスと8文字以上のパスワードを入力してください", "error");
    return;
  }

  try {
    await registerRequest({ email, password, displayName });
    showToast("登録申請を受け付けました。管理者の承認後にログインできます", "success");
    if (registerPasswordEl) registerPasswordEl.value = "";
  } catch (error) {
    const payload = error?.payload || {};
    showToast(payload.error === "email_already_exists" ? "既に存在するメールアドレスです" : "新規登録に失敗しました", "error");
  }
}

async function logout() {
  await logoutRequest().catch(() => null);
  if (state.recording || state.finalizingStop) {
    stopRecording();
    for (let i = 0; i < 60 && (state.recording || state.finalizingStop); i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }
  state.auth.authenticated = false;
  state.auth.isGuest = false;
  state.auth.user = null;
  state.auth.pendingApprovalCount = 0;
  clearHistoryState(state);
  state.log = [];
  state.segments = [];
  state.logAutoScrollEnabled = true;
  renderEmptyTranscriptState();
  setSummary("", "未生成");
  setProofread("", "未生成");
  resetRuntimeSessionState();
  persistGuestMode(false);
  setAppLocked(true);
  renderAuthState();
  renderHistoryList();
  updateDownloadLinks();
  loginEmailEl?.focus();
  showToast("ログアウトしました", "success");
}

function loginAsGuest() {
  state.auth.authenticated = false;
  state.auth.isGuest = true;
  state.auth.user = null;
  state.auth.pendingApprovalCount = 0;
  clearHistoryState(state);
  resetRuntimeSessionState();
  persistGuestMode(true);
  setAppLocked(false);
  renderAuthState();
  renderHistoryList();
  updateDownloadLinks();
  showToast("ゲストモードで開始しました", "success");
}

async function loadPendingUsers() {
  if (!state.auth.authenticated || !state.auth.user?.isAdmin) return;
  logClientEvent("pending_users.load.start");
  try {
    const payload = await fetchPendingUsersRequest();
    state.adminPendingUsers = Array.isArray(payload.items) ? payload.items : [];
    state.auth.pendingApprovalCount = state.adminPendingUsers.length;
    renderAuthState();
    renderAdminPendingUsers();
    logClientEvent("pending_users.load.success", { count: state.adminPendingUsers.length });
  } catch {
    showToast("承認待ち一覧の取得に失敗しました", "error");
    logClientEvent("pending_users.load.failed");
  }
}

async function openAdminQueueModal() {
  if (!state.auth.user?.isAdmin) return;
  await loadPendingUsers();
  if (!adminQueueModalEl) return;
  adminQueueModalEl.hidden = false;
  lockBodyScroll();
}

async function approvePendingUser(userId) {
  try {
    await approvePendingUserRequest(userId);
  } catch {
    showToast("ユーザー承認に失敗しました", "error");
    return;
  }
  showToast("ユーザーを承認しました", "success");
  await loadPendingUsers();
}

async function loadHistoryList() {
  if (!state.auth.authenticated) {
    clearHistoryState(state);
    renderHistoryList();
    logClientEvent("history.load.skipped");
    return;
  }

  logClientEvent("history.load.start", { query: state.history.query, offset: state.history.offset, limit: state.history.limit });
  try {
    const payload = await fetchHistoryListRequest({
      limit: state.history.limit,
      offset: state.history.offset,
      query: state.history.query,
    });
    applyHistoryListPayload(state, payload);
    renderHistoryList();
    logClientEvent("history.load.success", { total: state.history.total, items: state.history.items.length });
  } catch {
    updateHistoryEmptyState("履歴の取得に失敗しました");
    logClientEvent("history.load.failed");
  }
}

function renderHistoryDetail(payload) {
  applyHistoryDetailPayload(state, payload);
  renderEmptyTranscriptState();
  logEl.innerHTML = "";
  state.logAutoScrollEnabled = true;
  (payload.segments || []).forEach((segment) => {
    addLogLine(
      String(segment.text || ""),
      Number(segment.tsStart || 0),
      Number(segment.tsEnd || 0),
      Number(segment.seq),
      String(segment.speaker || ""),
      String(segment.screenshotUrl || ""),
      String(segment.rawAudioUrl || segment.rawAudioPath || ""),
      String(segment.audioUrl || segment.audioPath || "")
    );
  });
  if (!payload.segments || payload.segments.length === 0) {
    renderEmptyTranscriptState();
  }
  setSummary(payload.summaryText || "", payload.savedAt ? `保存: ${new Date(payload.savedAt).toLocaleString("ja-JP")}` : "履歴");
  setProofread(payload.proofreadText || "", payload.savedAt ? `保存: ${new Date(payload.savedAt).toLocaleString("ja-JP")}` : "履歴");
  if (saveTitleInputEl) {
    saveTitleInputEl.value = String(payload.title || "");
  }
  updateDownloadLinks();
  updateSaveControls();
  renderHistoryList();
}

async function openHistoryDetail(historyId) {
  if (!state.auth.authenticated) {
    showToast("ログインが必要です", "error");
    setAppLocked(true);
    loginEmailEl?.focus();
    return;
  }
  if (state.recording) {
    showToast("録音中は履歴を開けません", "error");
    return;
  }
  try {
    const payload = await fetchHistoryDetail(historyId);
    renderHistoryDetail(payload);
    if (window.innerWidth <= 1100) {
      applyHistoryDrawerOpen(false);
    }
  } catch {
    showToast("履歴の取得に失敗しました", "error");
  }
}

async function deleteHistory(historyId) {
  if (!historyId || !state.auth.authenticated) return;
  const target = state.history.items.find((item) => item.id === historyId);
  const confirmed = window.confirm(`履歴「${target?.title || historyId}」を削除しますか？`);
  if (!confirmed) return;

  try {
    await deleteHistoryRequest(historyId);
  } catch {
    showToast("履歴の削除に失敗しました", "error");
    return;
  }

  if (state.viewingHistoryId === historyId || state.savedHistoryId === historyId || state.history.selectedId === historyId) {
    clearView();
  }
  await loadHistoryList();
  showToast("履歴を削除しました", "success");
}

function buildAutoSaveTitle() {
  const explicit = String(saveTitleInputEl?.value || "").trim();
  if (explicit) return explicit;
  const transcript = extractTranscriptText().replace(/\s+/g, " ").trim();
  if (transcript) return transcript.slice(0, 30);
  const language = selectedLanguage() || "Transcript";
  return `${language} ${new Date().toLocaleString("ja-JP")}`;
}

async function saveCurrentHistory() {
  if (state.auth.isGuest) {
    showToast("ゲストでは履歴保存できません", "error");
    return;
  }
  if (!state.auth.authenticated) {
    showToast("ログインが必要です", "error");
    loginEmailEl?.focus();
    return;
  }
  if (state.recording || state.finalizingStop) {
    showToast("録音中は保存できません", "error");
    return;
  }
  if (!state.runtimeSessionId || state.segments.length === 0) {
    showToast("保存できる文字起こしがありません", "error");
    return;
  }

  state.saveInFlight = true;
  updateSaveControls();
  let payload;
  try {
    payload = await saveHistoryRequest({
      runtimeSessionId: state.runtimeSessionId,
      runtimeSessionToken: state.runtimeSessionToken,
      title: buildAutoSaveTitle(),
      summaryText: state.summary || null,
      proofreadText: state.proofread || null,
    });
  } catch (error) {
    state.saveInFlight = false;
    updateSaveControls();
    const responsePayload = error?.payload || {};
    showToast(
      responsePayload.error === "runtime_session_not_finalized"
        ? "録音停止後に保存してください"
        : responsePayload.error === "history_already_saved"
          ? "このセッションは既に保存済みです"
          : "保存に失敗しました",
      "error"
    );
    return;
  }
  state.saveInFlight = false;
  updateSaveControls();
  state.savedHistoryId = payload.history?.id || null;
  state.viewingHistoryId = state.savedHistoryId;
  state.history.selectedId = state.savedHistoryId;
  updateDownloadLinks();
  updateSaveControls();
  await loadHistoryList();
  showToast("保存しました", "success");
}

renderPromptTemplateButtons(state.promptTemplates);
buildRuntimeUi();
setAppLocked(true);

// Initialize empty states
updateSegmentCount();
updateDownloadLinks();
setSummary("", "未生成");
setProofread("", "未生成");
renderAuthState();
updateSharedVocabularyMeta();
applyProofreadMode(proofreadModeEl?.value || "proofread");
if (summaryBtnLabelEl) {
  summaryBtnLabelEl.textContent = "生成";
}
applyAdvancedSettingsOpen(false);
applySummaryPromptEditorOpen(false);
applyActiveAiPanel(state.activeAiPanel);
applySidebarOpen(false);
updateHistoryControls();
setStatus("idle");

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

(async () => {
  logClientEvent("bootstrap.start");
  await loadCapabilities();
  await loadSharedGlossary();
  await loadAuthState();
  logClientEvent("bootstrap.done");
})();
