import { normalizeTheme } from "../ui/theme.js";
import { writeStoredValue } from "../state/storage.js";
import { readStoredValue } from "../state/storage.js";
import { resolveInitialTheme } from "../ui/theme.js";
import { themeMetaColor } from "../ui/theme.js";

export function createLayoutController(appDependencies) {
function applyHistoryDrawerOpen(open) {
  if (!appDependencies.historyRailEl) return;
  const isMobile = window.innerWidth <= 1100;
  const isOpen = !!open && isMobile;
  const wasOpen = appDependencies.state.historyDrawerOpen;
  appDependencies.state.historyDrawerOpen = isOpen;
  appDependencies.historyRailEl.classList.toggle("is-open", isOpen);
  appDependencies.historyRailEl.classList.toggle("is-collapsed", !isMobile && appDependencies.state.historyCollapsed);
  appDependencies.historyRailEl.setAttribute("aria-hidden", isMobile ? (isOpen ? "false" : "true") : "false");
  if (isMobile) {
    appDependencies.historyRailEl.setAttribute("role", "dialog");
    appDependencies.historyRailEl.setAttribute("aria-modal", "true");
  } else {
    appDependencies.historyRailEl.removeAttribute("role");
    appDependencies.historyRailEl.removeAttribute("aria-modal");
  }
  if (appDependencies.historyDrawerOpenEl) {
    appDependencies.historyDrawerOpenEl.setAttribute("aria-expanded", isOpen ? "true" : "false");
  }
  if (appDependencies.historyDrawerBackdropEl) {
    appDependencies.historyDrawerBackdropEl.hidden = !isOpen;
    appDependencies.historyDrawerBackdropEl.classList.toggle("is-open", isOpen);
  }
  document.body.classList.toggle("is-history-drawer-open", isOpen);
  updateHistoryControls();
  if (isOpen && !wasOpen) {
    requestAnimationFrame(() => {
      appDependencies.historyDrawerCloseEl?.focus();
    });
  } else if (!isOpen && wasOpen && isMobile) {
    appDependencies.historyDrawerOpenEl?.focus();
  }
}

function applyHistoryCollapsed(value, options = {}) {
  const persist = options.persist !== false;
  appDependencies.state.historyCollapsed = !!value;
  const isDesktop = window.innerWidth > 1100;
  if (appDependencies.workspaceShellEl) {
    appDependencies.workspaceShellEl.classList.toggle("is-history-collapsed", isDesktop && appDependencies.state.historyCollapsed);
  }
  if (appDependencies.historyRailEl) {
    appDependencies.historyRailEl.classList.toggle("is-collapsed", isDesktop && appDependencies.state.historyCollapsed);
  }
  if (persist) {
    try {
      localStorage.setItem("whistx_history_collapsed", appDependencies.state.historyCollapsed ? "1" : "0");
    } catch {
      // ignore
    }
  }
  updateHistoryControls();
}

function updateHistoryControls() {
  const isMobile = window.innerWidth <= 1100;
  if (appDependencies.historyDrawerOpenEl) {
    appDependencies.historyDrawerOpenEl.hidden = !isMobile;
    appDependencies.historyDrawerOpenEl.setAttribute("aria-expanded", appDependencies.state.historyDrawerOpen ? "true" : "false");
  }
  if (appDependencies.historyCollapseBtn) {
    appDependencies.historyCollapseBtn.hidden = isMobile;
    appDependencies.historyCollapseBtn.classList.toggle("is-collapsed", appDependencies.state.historyCollapsed);
    appDependencies.historyCollapseBtn.setAttribute("aria-label", appDependencies.state.historyCollapsed ? "履歴を展開" : "履歴をたたむ");
    appDependencies.historyCollapseBtn.title = appDependencies.state.historyCollapsed ? "展開" : "たたむ";
  }
}

function applyAdvancedSettingsOpen(open) {
  appDependencies.state.advancedSettingsOpen = !!open;
  if (appDependencies.inputAdvancedSettingsEl) {
    appDependencies.inputAdvancedSettingsEl.hidden = !appDependencies.state.advancedSettingsOpen;
    appDependencies.inputAdvancedSettingsEl.classList.toggle("is-open", appDependencies.state.advancedSettingsOpen);
  }
  if (appDependencies.settingsAdvancedToggleEl) {
    appDependencies.settingsAdvancedToggleEl.setAttribute("aria-expanded", appDependencies.state.advancedSettingsOpen ? "true" : "false");
    appDependencies.settingsAdvancedToggleEl.textContent = appDependencies.state.advancedSettingsOpen ? "話者設定を閉じる" : "話者設定";
  }
}

function syncAiResponsiveState() {
  applyPanelCollapseState("transcript", !!appDependencies.state.panelCollapsed.transcript, { persist: false });
  applyPanelCollapseState("proofread", !!appDependencies.state.panelCollapsed.proofread, { persist: false });
  applyPanelCollapseState("summary", !!appDependencies.state.panelCollapsed.summary, { persist: false });
}

function applyActiveAiPanel(panel) {
  const nextPanel = panel === "summary" ? "summary" : "proofread";
  appDependencies.state.activeAiPanel = nextPanel;
  if (appDependencies.proofreadPanelEl) {
    appDependencies.proofreadPanelEl.classList.toggle("is-ai-active", nextPanel === "proofread");
  }
  if (appDependencies.summaryPanelEl) {
    appDependencies.summaryPanelEl.classList.toggle("is-ai-active", nextPanel === "summary");
  }
  appDependencies.aiTabEls.forEach((button) => {
    const active = button.dataset.aiTab === nextPanel;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  syncAiResponsiveState();
}

function applyWorkspaceRatios(left, center, right, options = {}) {
  const persist = options.persist !== false;
  const safeLeft = Math.max(0.65, Math.min(2.4, Number(left) || appDependencies.state.panelLeftRatio));
  const safeCenter = Math.max(0.6, Math.min(2.2, Number(center) || appDependencies.state.panelCenterRatio));
  const safeRight = Math.max(0.6, Math.min(2.2, Number(right) || appDependencies.state.panelRightRatio));

  appDependencies.state.panelLeftRatio = safeLeft;
  appDependencies.state.panelCenterRatio = safeCenter;
  appDependencies.state.panelRightRatio = safeRight;

  if (appDependencies.workspacePanelsEl) {
    appDependencies.workspacePanelsEl.style.setProperty("--panel-left", `${safeLeft}fr`);
    appDependencies.workspacePanelsEl.style.setProperty("--panel-center", `${safeCenter}fr`);
    appDependencies.workspacePanelsEl.style.setProperty("--panel-right", `${safeRight}fr`);
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
  if (!appDependencies.workspacePanelsEl) return;
  if (window.innerWidth <= appDependencies.WORKSPACE_STACK_BREAKPOINT) {
    appDependencies.workspacePanelsEl.style.gridTemplateColumns = "1fr";
    return;
  }
  if (window.innerWidth <= appDependencies.WORKSPACE_DUAL_AI_BREAKPOINT) {
    const activeAiPanel = appDependencies.state.activeAiPanel === "summary" ? "summary" : "proofread";
    const left = appDependencies.state.panelCollapsed.transcript ? "92px" : `minmax(280px, ${appDependencies.state.panelLeftRatio + 0.2}fr)`;
    const right = appDependencies.state.panelCollapsed[activeAiPanel]
      ? "88px"
      : `minmax(240px, ${appDependencies.state.panelCenterRatio + appDependencies.state.panelRightRatio}fr)`;
    appDependencies.workspacePanelsEl.style.gridTemplateColumns = `${left} ${right}`;
    return;
  }

  const left = appDependencies.state.panelCollapsed.transcript ? "88px" : `minmax(280px, ${appDependencies.state.panelLeftRatio}fr)`;
  const center = appDependencies.state.panelCollapsed.proofread ? "88px" : `minmax(240px, ${appDependencies.state.panelCenterRatio}fr)`;
  const right = appDependencies.state.panelCollapsed.summary ? "88px" : `minmax(240px, ${appDependencies.state.panelRightRatio}fr)`;
  appDependencies.workspacePanelsEl.style.gridTemplateColumns = `${left} 10px ${center} 10px ${right}`;
}

function applyPanelCollapseState(panel, collapsed, options = {}) {
  const persist = options.persist !== false;
  const key = panel === "proofread" || panel === "summary" ? panel : "transcript";
  appDependencies.state.panelCollapsed[key] = !!collapsed;
  const collapseAvailable = !appDependencies.workspacePanelsEl?.classList.contains("meeting-layout") && window.innerWidth > appDependencies.WORKSPACE_STACK_BREAKPOINT;
  const visuallyCollapsed = collapseAvailable && !!collapsed;

  const panelEl = document.querySelector(`.${key}-panel`);
  if (panelEl) {
    panelEl.classList.toggle("is-collapsed", visuallyCollapsed);
  }

  const toggleBtn = document.querySelector(`[data-panel-toggle="${key}"]`);
  if (toggleBtn) {
    toggleBtn.classList.toggle("is-collapsed", visuallyCollapsed);
    toggleBtn.hidden = !collapseAvailable;
    toggleBtn.disabled = !collapseAvailable;
    toggleBtn.setAttribute("aria-hidden", String(!collapseAvailable));
    const labelMap = { transcript: "文字起こし", proofread: "校正", summary: "要約" };
    const label = labelMap[key] || "パネル";
    toggleBtn.setAttribute("aria-label", visuallyCollapsed ? `${label}を展開` : `${label}をたたむ`);
    toggleBtn.title = visuallyCollapsed ? "展開" : "たたむ";
  }

  if (persist) {
    try {
      localStorage.setItem("whistx_panel_collapsed", JSON.stringify(appDependencies.state.panelCollapsed));
    } catch {
      // ignore
    }
  }
  updateWorkspaceGridTemplate();
}

function setupWorkspaceResizers() {
  if (!appDependencies.workspacePanelsEl || appDependencies.panelResizerEls.length === 0) return;

  const stopDrag = () => {
    appDependencies.state.activeResizer = null;
    document.body.classList.remove("is-resizing-panels");
  };

  const onPointerMove = (event) => {
    if (!appDependencies.state.activeResizer || window.innerWidth <= 1439) return;
    const rect = appDependencies.workspacePanelsEl.getBoundingClientRect();
    const totalWidth = rect.width;
    if (totalWidth <= 0) return;

    const currentLeft = appDependencies.state.panelLeftRatio;
    const currentCenter = appDependencies.state.panelCenterRatio;
    const currentRight = appDependencies.state.panelRightRatio;
    const totalRatio = currentLeft + currentCenter + currentRight;
    const ratioPerPixel = totalRatio / totalWidth;

    if (appDependencies.state.activeResizer === "left") {
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

  appDependencies.panelResizerEls.forEach((handle) => {
    handle.addEventListener("pointerdown", (event) => {
      if (window.innerWidth <= 1439) return;
      appDependencies.state.activeResizer = String(handle.dataset.resizer || "");
      document.body.classList.add("is-resizing-panels");
      try {
        handle.setPointerCapture?.(event.pointerId);
      } catch {
        // Pointer capture can be unavailable for synthetic or interrupted pointer sequences.
      }
      event.preventDefault();
    });
  });

  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", stopDrag);
  window.addEventListener("pointercancel", stopDrag);
}

function setupPanelToggles() {
  appDependencies.panelToggleEls.forEach((button) => {
    button.addEventListener("click", () => {
      const panel = String(button.dataset.panelToggle || "");
      applyPanelCollapseState(panel, !appDependencies.state.panelCollapsed[panel]);
    });
  });
}

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
  appDependencies.showToast(`${themeName}に切り替えました`, "success");
}

  return { applyHistoryDrawerOpen, applyHistoryCollapsed, updateHistoryControls, applyAdvancedSettingsOpen, syncAiResponsiveState, applyActiveAiPanel, applyWorkspaceRatios, updateWorkspaceGridTemplate, applyPanelCollapseState, setupWorkspaceResizers, setupPanelToggles, applyTheme, initTheme, updateThemeColorMeta, toggleTheme };
}
