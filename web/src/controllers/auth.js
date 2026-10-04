import { canUseWorkspace as canUseWorkspaceForAuth } from "../auth/session.js";
import { serializeUserLabel } from "../auth/session.js";
import { fetchAuthState } from "../auth/api.js";
import { readGuestMode } from "../auth/session.js";
import { persistGuestMode } from "../auth/session.js";
import { loginRequest } from "../auth/api.js";
import { bootstrapAdminRequest } from "../auth/api.js";
import { registerRequest } from "../auth/api.js";
import { updateDisplayNameRequest } from "../auth/api.js";
import { logoutRequest } from "../auth/api.js";
import { clearHistoryState } from "../history/state.js";

export function createAuthController(appDependencies) {
function canUseWorkspace() {
  return canUseWorkspaceForAuth(appDependencies.state.auth);
}

function setAppLocked(locked) {
  appDependencies.runtimeUi.appLocked = !!locked;
  document.body.classList.toggle("whistx-auth-locked", appDependencies.runtimeUi.appLocked);
  if (appDependencies.runtimeUi.loginOverlayEl) {
    appDependencies.runtimeUi.loginOverlayEl.hidden = !appDependencies.runtimeUi.appLocked;
  }
  if (appDependencies.startBtn) {
    appDependencies.startBtn.disabled = appDependencies.runtimeUi.appLocked;
  }
  if (appDependencies.summaryBtn) {
    appDependencies.summaryBtn.disabled = appDependencies.runtimeUi.appLocked;
  }
  if (appDependencies.proofreadBtn) {
    appDependencies.proofreadBtn.disabled = appDependencies.runtimeUi.appLocked;
  }
  if (appDependencies.clearBtn) {
    appDependencies.clearBtn.disabled = appDependencies.runtimeUi.appLocked;
  }
  if (appDependencies.saveBtn && appDependencies.runtimeUi.appLocked) {
    appDependencies.saveBtn.disabled = true;
  }
}

function renderAuthState() {
  const authenticated = !!appDependencies.state.auth.authenticated;
  const isGuest = !!appDependencies.state.auth.isGuest;
  const workspaceEnabled = authenticated || isGuest;
  const bootstrapRequired = !!appDependencies.state.auth.bootstrapAdminRequired;
  const isAdmin = !!appDependencies.state.auth.user?.isAdmin;
  const userLabel = authenticated ? serializeUserLabel(appDependencies.state.auth.user) : isGuest ? "ゲスト利用中" : "未ログイン";
  if (appDependencies.authUserLabelEl) {
    appDependencies.authUserLabelEl.textContent = userLabel;
  }
  if (appDependencies.loginBtn) {
    appDependencies.loginBtn.hidden = workspaceEnabled;
  }
  if (appDependencies.logoutBtn) {
    appDependencies.logoutBtn.hidden = !workspaceEnabled;
  }
  if (appDependencies.authProfileEditBtn) {
    appDependencies.authProfileEditBtn.hidden = !authenticated;
    appDependencies.authProfileEditBtn.disabled = !!appDependencies.state.auth.profileSaving;
    appDependencies.authProfileEditBtn.textContent = appDependencies.state.auth.profileEditorOpen ? "表示名編集を閉じる" : "表示名変更";
  }
  if (appDependencies.authGuestViewEl) {
    appDependencies.authGuestViewEl.hidden = workspaceEnabled;
  }
  if (appDependencies.guestLoginBtn) {
    appDependencies.guestLoginBtn.hidden = !appDependencies.state.auth.guestTranscriptionAllowed;
  }
  if (appDependencies.authUserViewEl) {
    appDependencies.authUserViewEl.hidden = !workspaceEnabled;
  }
  if (appDependencies.authProfileEditorEl) {
    appDependencies.authProfileEditorEl.hidden = !authenticated || !appDependencies.state.auth.profileEditorOpen;
  }
  if (appDependencies.authProfileDisplayNameEl) {
    appDependencies.authProfileDisplayNameEl.disabled = !authenticated || !!appDependencies.state.auth.profileSaving;
  }
  if (appDependencies.authProfileSaveBtn) {
    appDependencies.authProfileSaveBtn.disabled = !authenticated || !!appDependencies.state.auth.profileSaving;
    appDependencies.authProfileSaveBtn.textContent = appDependencies.state.auth.profileSaving ? "保存中..." : "保存";
  }
  if (appDependencies.authProfileCancelBtn) {
    appDependencies.authProfileCancelBtn.disabled = !!appDependencies.state.auth.profileSaving;
  }
  if (appDependencies.authBootstrapSectionEl) {
    appDependencies.authBootstrapSectionEl.hidden = !bootstrapRequired;
  }
  if (appDependencies.authLoginSectionEl) {
    appDependencies.authLoginSectionEl.hidden = bootstrapRequired;
  }
  if (appDependencies.authRegisterSectionEl) {
    appDependencies.authRegisterSectionEl.hidden = bootstrapRequired || !appDependencies.state.selfSignupEnabled;
  }
  if (appDependencies.keycloakLoginBtnEl) {
    appDependencies.keycloakLoginBtnEl.hidden = bootstrapRequired || !appDependencies.state.auth.keycloakEnabled;
    appDependencies.keycloakLoginBtnEl.textContent = appDependencies.state.auth.keycloakButtonLabel || "Keycloakでログイン";
  }
  if (appDependencies.adminQueueBtn) {
    appDependencies.adminQueueBtn.hidden = !authenticated || !isAdmin;
  }
  if (appDependencies.adminQueueBadgeEl) {
    appDependencies.adminQueueBadgeEl.textContent = String(appDependencies.state.auth.pendingApprovalCount || 0);
  }
  if (appDependencies.registerBtn) {
    appDependencies.registerBtn.disabled = !appDependencies.state.selfSignupEnabled;
  }
  if (appDependencies.registerHintEl) {
    if (bootstrapRequired) {
      appDependencies.registerHintEl.textContent = "";
    } else if (appDependencies.state.selfSignupEnabled) {
      appDependencies.registerHintEl.textContent = "申請後は管理者の承認が完了するまでログインできません";
    } else {
      appDependencies.registerHintEl.textContent = "新規登録は現在無効です";
    }
  }
  appDependencies.updateHistoryEmptyState();
  appDependencies.updateSaveControls();
  appDependencies.updateSharedVocabularyMeta();
  appDependencies.meetingWorkspace?.syncSource();
}

function syncAuthProfileEditor() {
  if (!appDependencies.authProfileDisplayNameEl) return;
  appDependencies.authProfileDisplayNameEl.value = String(appDependencies.state.auth.user?.displayName || "");
}

function setAuthProfileEditorOpen(open) {
  appDependencies.state.auth.profileEditorOpen = !!open;
  if (appDependencies.state.auth.profileEditorOpen) {
    syncAuthProfileEditor();
  }
  renderAuthState();
  if (appDependencies.state.auth.profileEditorOpen) {
    appDependencies.authProfileDisplayNameEl?.focus();
    appDependencies.authProfileDisplayNameEl?.select();
  }
}

function handleAuthErrorFromLocation() {
  const url = new URL(window.location.href);
  const authError = String(url.searchParams.get("authError") || "");
  if (!authError) return;
  if (authError === "approval_required") {
    appDependencies.showToast("Keycloak ログイン後も管理者承認が必要です", "error");
  } else if (authError === "keycloak_state") {
    appDependencies.showToast("Keycloak ログインの状態確認に失敗しました", "error");
  } else if (authError === "keycloak_email_not_verified") {
    appDependencies.showToast("Keycloak 側でメールアドレス確認が完了していません", "error");
  } else if (authError === "keycloak_account_link_required") {
    appDependencies.showToast("同じメールアドレスの既存アカウントがあります。管理者に Keycloak 連携を依頼してください", "error", 7000);
  } else if (authError === "keycloak_identity_conflict") {
    appDependencies.showToast("Keycloak アカウントの紐付けに競合があります", "error", 7000);
  } else if (authError === "keycloak_failed") {
    appDependencies.showToast("Keycloak ログインに失敗しました", "error");
  }
  url.searchParams.delete("authError");
  window.history.replaceState({}, document.title, `${url.pathname}${url.search}${url.hash}`);
}

async function loadAuthState() {
  appDependencies.logClientEvent("auth_state.load.start");
  try {
    const payload = await fetchAuthState();
    if (!payload?.authenticated) {
      appDependencies.state.auth.authenticated = false;
      appDependencies.state.auth.user = null;
      appDependencies.state.auth.sessionInvalid = !!payload?.sessionInvalid;
      appDependencies.state.auth.bootstrapAdminRequired = !!payload?.bootstrapAdminRequired;
      appDependencies.state.selfSignupEnabled = !!payload?.selfSignupEnabled;
      appDependencies.state.auth.guestTranscriptionAllowed = !!payload?.guestTranscriptionAllowed;
      appDependencies.state.auth.profileEditorOpen = false;
      appDependencies.state.auth.profileSaving = false;
      appDependencies.state.auth.historyRetentionDays = Math.max(0, Number(payload?.historyRetentionDays ?? 0));
      appDependencies.state.auth.keycloakEnabled = !!payload?.keycloakEnabled;
      appDependencies.state.auth.keycloakButtonLabel = String(payload?.keycloakButtonLabel || "Keycloakでログイン");
      try {
        appDependencies.state.auth.isGuest = appDependencies.state.auth.guestTranscriptionAllowed && readGuestMode();
      } catch {
        appDependencies.state.auth.isGuest = false;
      }
      if (appDependencies.state.auth.sessionInvalid) {
        appDependencies.state.auth.isGuest = false;
        persistGuestMode(false);
      }
      setAppLocked(!canUseWorkspace());
      renderAuthState();
      if (appDependencies.state.auth.sessionInvalid) {
        appDependencies.showToast("ログインセッションが切れました。再ログインしてください", "error", 7000);
        appDependencies.loginEmailEl?.focus();
      }
      appDependencies.logClientEvent("auth_state.load.success", { authenticated: false, guest: !!appDependencies.state.auth.isGuest });
      return;
    }
    appDependencies.state.auth.authenticated = !!payload.authenticated;
    appDependencies.state.auth.isGuest = false;
    appDependencies.state.auth.sessionInvalid = false;
    appDependencies.state.auth.user = payload.user || null;
    appDependencies.state.auth.profileEditorOpen = false;
    appDependencies.state.auth.profileSaving = false;
    appDependencies.state.selfSignupEnabled = !!payload.selfSignupEnabled;
    appDependencies.state.auth.guestTranscriptionAllowed = !!payload.guestTranscriptionAllowed;
    appDependencies.state.auth.historyRetentionDays = Math.max(0, Number(payload.historyRetentionDays ?? 0));
    appDependencies.state.auth.bootstrapAdminRequired = !!payload.bootstrapAdminRequired;
    appDependencies.state.auth.pendingApprovalCount = Number(payload.pendingApprovalCount || 0);
    appDependencies.state.auth.keycloakEnabled = !!payload.keycloakEnabled;
    appDependencies.state.auth.keycloakButtonLabel = String(payload.keycloakButtonLabel || "Keycloakでログイン");
    if (appDependencies.state.auth.authenticated) {
      persistGuestMode(false);
    } else {
      try {
        appDependencies.state.auth.isGuest = appDependencies.state.auth.guestTranscriptionAllowed && readGuestMode();
      } catch {
        appDependencies.state.auth.isGuest = false;
      }
    }
    setAppLocked(!canUseWorkspace());
    renderAuthState();
    if (appDependencies.state.auth.authenticated) {
      await appDependencies.loadHistoryList();
    } else if (appDependencies.state.auth.isGuest) {
      appDependencies.state.history.items = [];
      appDependencies.state.history.selectedId = null;
      appDependencies.renderHistoryList();
    } else {
      appDependencies.state.history.items = [];
      appDependencies.state.history.selectedId = null;
      appDependencies.renderHistoryList();
      if (appDependencies.state.auth.bootstrapAdminRequired) {
        appDependencies.bootstrapDisplayNameEl?.focus();
      } else {
        appDependencies.loginEmailEl?.focus();
      }
    }
    appDependencies.logClientEvent("auth_state.load.success", {
      authenticated: !!appDependencies.state.auth.authenticated,
      guest: !!appDependencies.state.auth.isGuest,
      isAdmin: !!appDependencies.state.auth.user?.isAdmin,
    });
  } catch {
    try {
      setAppLocked(!canUseWorkspace());
      renderAuthState();
      appDependencies.showToast("認証状態を確認できません。通信を確認して再読み込みしてください", "error", 7000);
      appDependencies.logClientEvent("auth_state.load.fallback", { guest: !!appDependencies.state.auth.isGuest });
    } catch {
      // ignore
      appDependencies.logClientEvent("auth_state.load.failed");
    }
  }
}

async function login() {
  const email = String(appDependencies.loginEmailEl?.value || "").trim();
  const password = String(appDependencies.loginPasswordEl?.value || "");
  if (!email || !password) {
    appDependencies.showToast("メールアドレスとパスワードを入力してください", "error");
    return;
  }

  try {
    const payload = await loginRequest({ email, password });
    appDependencies.state.auth.authenticated = true;
    appDependencies.state.auth.isGuest = false;
    appDependencies.state.auth.sessionInvalid = false;
    appDependencies.state.auth.user = payload.user || null;
    appDependencies.state.auth.profileEditorOpen = false;
    appDependencies.state.auth.profileSaving = false;
    appDependencies.state.auth.bootstrapAdminRequired = false;
    appDependencies.state.auth.pendingApprovalCount = Number(payload.pendingApprovalCount || 0);
    persistGuestMode(false);
    setAppLocked(false);
    renderAuthState();
    await loadAuthState();
    appDependencies.showToast("ログインしました", "success");
  } catch (error) {
    const payload = error?.payload || {};
    if (payload.error === "approval_required") {
      appDependencies.showToast("管理者の承認後にログインできます", "error");
    } else if (payload.error === "too_many_login_attempts") {
      appDependencies.showToast(`ログイン試行が多すぎます。${Number(payload.retryAfterSec || 0)}秒後に再試行してください`, "error", 5000);
    } else {
      appDependencies.showToast("ログインに失敗しました", "error");
    }
  }
}

async function bootstrapAdmin() {
  const displayName = String(appDependencies.bootstrapDisplayNameEl?.value || "").trim();
  const email = String(appDependencies.bootstrapEmailEl?.value || "").trim();
  const password = String(appDependencies.bootstrapPasswordEl?.value || "");
  if (!email || password.length < 8) {
    appDependencies.showToast("メールアドレスと8文字以上のパスワードを入力してください", "error");
    return;
  }

  try {
    const payload = await bootstrapAdminRequest({ email, password, displayName });
    appDependencies.state.auth.authenticated = true;
    appDependencies.state.auth.isGuest = false;
    appDependencies.state.auth.sessionInvalid = false;
    appDependencies.state.auth.user = payload.user || null;
    appDependencies.state.auth.profileEditorOpen = false;
    appDependencies.state.auth.profileSaving = false;
    appDependencies.state.auth.bootstrapAdminRequired = false;
    appDependencies.state.auth.pendingApprovalCount = 0;
    persistGuestMode(false);
    setAppLocked(false);
    renderAuthState();
    await loadAuthState();
    appDependencies.showToast("管理者アカウントを作成しました", "success");
  } catch (error) {
    const payload = error?.payload || {};
    appDependencies.showToast(payload.error === "email_already_exists" ? "既に存在するメールアドレスです" : "管理者作成に失敗しました", "error");
  }
}

async function registerAccount() {
  if (!appDependencies.state.selfSignupEnabled) {
    appDependencies.showToast("新規登録は無効です", "error");
    return;
  }
  const email = String(appDependencies.registerEmailEl?.value || "").trim();
  const password = String(appDependencies.registerPasswordEl?.value || "");
  const displayName = String(appDependencies.registerDisplayNameEl?.value || "").trim();
  if (!email || password.length < 8) {
    appDependencies.showToast("メールアドレスと8文字以上のパスワードを入力してください", "error");
    return;
  }

  try {
    await registerRequest({ email, password, displayName });
    appDependencies.showToast("登録申請を受け付けました。管理者の承認後にログインできます", "success");
    if (appDependencies.registerPasswordEl) appDependencies.registerPasswordEl.value = "";
  } catch (error) {
    const payload = error?.payload || {};
    appDependencies.showToast(payload.error === "email_already_exists" ? "既に存在するメールアドレスです" : "新規登録に失敗しました", "error");
  }
}

async function saveDisplayName() {
  if (!appDependencies.state.auth.authenticated) {
    appDependencies.showToast("ログインが必要です", "error");
    return;
  }
  const displayName = String(appDependencies.authProfileDisplayNameEl?.value || "").trim();
  appDependencies.state.auth.profileSaving = true;
  renderAuthState();
  try {
    const payload = await updateDisplayNameRequest(displayName);
    appDependencies.state.auth.user = payload.user || appDependencies.state.auth.user;
    appDependencies.state.auth.profileSaving = false;
    appDependencies.state.auth.profileEditorOpen = false;
    syncAuthProfileEditor();
    renderAuthState();
    appDependencies.showToast("表示名を更新しました", "success");
  } catch {
    appDependencies.state.auth.profileSaving = false;
    renderAuthState();
    appDependencies.showToast("表示名の更新に失敗しました", "error");
  }
}

async function logout() {
  if (!appDependencies.confirmWorkspaceDiscard("破棄してログアウト")) {
    return;
  }
  await logoutRequest().catch(() => null);
  if (appDependencies.state.recording || appDependencies.state.finalizingStop) {
    appDependencies.stopRecording();
    for (let i = 0; i < 60 && (appDependencies.state.recording || appDependencies.state.finalizingStop); i += 1) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }
  appDependencies.state.auth.authenticated = false;
  appDependencies.state.auth.isGuest = false;
  appDependencies.state.auth.sessionInvalid = false;
  appDependencies.state.auth.user = null;
  appDependencies.state.auth.profileEditorOpen = false;
  appDependencies.state.auth.profileSaving = false;
  appDependencies.state.auth.pendingApprovalCount = 0;
  clearHistoryState(appDependencies.state);
  appDependencies.state.log = [];
  appDependencies.state.segments = [];
  appDependencies.state.logAutoScrollEnabled = true;
  appDependencies.renderEmptyTranscriptState();
  appDependencies.setSummary("", "未実行");
  appDependencies.setProofread("", "未実行");
  appDependencies.resetRuntimeSessionState();
  appDependencies.markWorkspaceClean();
  persistGuestMode(false);
  setAppLocked(true);
  renderAuthState();
  appDependencies.renderHistoryList();
  appDependencies.updateDownloadLinks();
  appDependencies.loginEmailEl?.focus();
  appDependencies.showToast("ログアウトしました", "success");
}

function loginAsGuest() {
  if (!appDependencies.state.auth.guestTranscriptionAllowed) {
    appDependencies.showToast("ゲスト文字起こしは無効です", "error");
    return;
  }
  appDependencies.state.auth.authenticated = false;
  appDependencies.state.auth.isGuest = true;
  appDependencies.state.auth.sessionInvalid = false;
  appDependencies.state.auth.user = null;
  appDependencies.state.auth.profileEditorOpen = false;
  appDependencies.state.auth.profileSaving = false;
  appDependencies.state.auth.pendingApprovalCount = 0;
  clearHistoryState(appDependencies.state);
  appDependencies.resetRuntimeSessionState();
  appDependencies.markWorkspaceClean();
  persistGuestMode(true);
  setAppLocked(false);
  renderAuthState();
  appDependencies.renderHistoryList();
  appDependencies.updateDownloadLinks();
  appDependencies.showToast("ゲストモードで開始しました", "success");
}

  return { canUseWorkspace, setAppLocked, renderAuthState, syncAuthProfileEditor, setAuthProfileEditorOpen, handleAuthErrorFromLocation, loadAuthState, login, bootstrapAdmin, registerAccount, saveDisplayName, logout, loginAsGuest };
}
