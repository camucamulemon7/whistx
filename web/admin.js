import { fetchJson } from "./src/api/client.js";

const pendingPanelEl = document.querySelector("#pendingPanel");
const pendingListEl = document.querySelector("#pendingList");
const pendingCountEl = document.querySelector("#pendingCount");
const userCountEl = document.querySelector("#userCount");
const userTableBodyEl = document.querySelector("#userTableBody");
const userCardsEl = document.querySelector("#userCards");
const statusEl = document.querySelector("#adminStatus");
const refreshBtnEl = document.querySelector("#adminRefreshBtn");
const userSearchFormEl = document.querySelector("#userSearchForm");
const userSearchInputEl = document.querySelector("#userSearchInput");
const userSearchClearBtnEl = document.querySelector("#userSearchClearBtn");

const state = {
  userQuery: "",
  usersOffset: 0,
  pendingOffset: 0,
  requestId: 0,
};

function normalizeTheme(value) {
  return value === "dark" ? "dark" : "light";
}

function applyTheme(theme) {
  const normalized = normalizeTheme(theme);
  document.documentElement.setAttribute("data-theme", normalized);
  document.documentElement.style.colorScheme = normalized;
  const color = normalized === "dark" ? "#0a0a0a" : "#f5f5f7";
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", color);
}

function initTheme() {
  try {
    applyTheme(localStorage.getItem("whistx_theme") || "light");
  } catch {
    applyTheme("light");
  }
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString("ja-JP");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function roleLabel(isAdmin) {
  return isAdmin ? "管理者" : "メンバー";
}

function statusLabel(isActive) {
  return isActive ? "有効" : "承認待ち";
}

function roleBadge(isAdmin) {
  return `<span class="admin-badge ${isAdmin ? "is-admin" : ""}">${escapeHtml(roleLabel(isAdmin))}</span>`;
}

function statusBadge(isActive) {
  return `<span class="admin-badge ${isActive ? "is-active" : "is-pending"}">${escapeHtml(statusLabel(isActive))}</span>`;
}

async function approveUser(userId) {
  await fetchJson(`/api/admin/pending-users/${userId}/approve`, { method: "POST" });
  await loadAdminData();
}

async function updateUserRole(userId, role) {
  await fetchJson(`/api/admin/users/${userId}/role`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role }),
  });
  await loadAdminData();
}

function currentUserQuery() {
  return String(state.userQuery || "").trim();
}

function bindApproveButtons(root) {
  root.querySelectorAll(".admin-approve-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        await approveUser(button.dataset.userId);
      } catch (error) {
        statusEl.textContent = error?.message || "承認に失敗しました";
        button.disabled = false;
      }
    });
  });
}

function bindRoleSelects(root) {
  root.querySelectorAll(".admin-role-select").forEach((selectEl) => {
    selectEl.addEventListener("change", async () => {
      selectEl.disabled = true;
      try {
        await updateUserRole(selectEl.dataset.userId, selectEl.value);
      } catch (error) {
        statusEl.textContent = error?.message || "権限更新に失敗しました";
        selectEl.disabled = false;
      }
    });
  });
}

function renderPending(items) {
  if (!pendingListEl) return;
  const count = items.length;
  pendingCountEl.textContent = String(count);
  pendingPanelEl?.classList.toggle("is-empty", count === 0);

  if (!count) {
    pendingListEl.innerHTML = '<div class="admin-empty">承認待ちはありません。</div>';
    return;
  }

  pendingListEl.innerHTML = items.map((item) => `
    <article class="admin-request">
      <div class="admin-request-copy">
        <div class="admin-request-name">${escapeHtml(item.displayName || item.email || "pending")}</div>
        <div class="admin-request-email">${escapeHtml(item.email || "")}</div>
        <div class="admin-request-date">申請日時: ${escapeHtml(formatDate(item.createdAt))}</div>
      </div>
      <button type="button" class="admin-approve-btn" data-user-id="${escapeHtml(item.id)}">承認</button>
    </article>
  `).join("");

  bindApproveButtons(pendingListEl);
}

function renderUsers(items) {
  userCountEl.textContent = String(items.length);

  if (userTableBodyEl) {
    if (!items.length) {
      userTableBodyEl.innerHTML = '<tr><td colspan="6" class="admin-empty">ユーザーがありません。</td></tr>';
    } else {
      userTableBodyEl.innerHTML = items.map((item) => `
        <tr>
          <td>
            <div class="admin-user-name">${escapeHtml(item.displayName || "-")}</div>
            <div class="admin-user-email">${escapeHtml(item.email || "")}</div>
          </td>
          <td>${roleBadge(!!item.isAdmin)}</td>
          <td>${statusBadge(!!item.isActive)}</td>
          <td>${escapeHtml(formatDate(item.createdAt))}</td>
          <td>${escapeHtml(formatDate(item.lastLoginAt))}</td>
          <td class="admin-actions-cell">
            <select class="admin-role-select" data-user-id="${escapeHtml(item.id)}">
              <option value="member" ${item.isAdmin ? "" : "selected"}>メンバー</option>
              <option value="admin" ${item.isAdmin ? "selected" : ""}>管理者</option>
            </select>
          </td>
        </tr>
      `).join("");
      bindRoleSelects(userTableBodyEl);
    }
  }

  if (userCardsEl) {
    if (!items.length) {
      userCardsEl.innerHTML = '<div class="admin-empty">ユーザーがありません。</div>';
      return;
    }

    userCardsEl.innerHTML = items.map((item) => `
      <article class="admin-user-card">
        <div class="admin-user-card-copy">
          <div class="admin-user-name">${escapeHtml(item.displayName || "-")}</div>
          <div class="admin-user-email">${escapeHtml(item.email || "")}</div>
        </div>
        <div class="admin-user-meta-grid">
          <div class="admin-user-meta-item">
            <span class="admin-muted">権限</span>
            ${roleBadge(!!item.isAdmin)}
          </div>
          <div class="admin-user-meta-item">
            <span class="admin-muted">状態</span>
            ${statusBadge(!!item.isActive)}
          </div>
          <div class="admin-user-meta-item">
            <span class="admin-muted">作成日</span>
            <span class="admin-user-meta">${escapeHtml(formatDate(item.createdAt))}</span>
          </div>
          <div class="admin-user-meta-item">
            <span class="admin-muted">最終ログイン</span>
            <span class="admin-user-meta">${escapeHtml(formatDate(item.lastLoginAt))}</span>
          </div>
        </div>
        <select class="admin-role-select" data-user-id="${escapeHtml(item.id)}">
          <option value="member" ${item.isAdmin ? "" : "selected"}>メンバー</option>
          <option value="admin" ${item.isAdmin ? "selected" : ""}>管理者</option>
        </select>
      </article>
    `).join("");
    bindRoleSelects(userCardsEl);
  }
}

const PAGE_SIZE = 50;
const pagerButtons = ['pendingPrev', 'pendingNext', 'usersPrev', 'usersNext'].map((id) => document.getElementById(id));

function renderPager(prefix, data) {
  const total = Number(data.total ?? data.items.length);
  const offset = Number(data.offset || 0);
  const range = total ? `${offset + 1}–${offset + data.items.length} / ${total} 件` : '0 件';
  document.getElementById(`${prefix}Range`).textContent = range;
  document.getElementById(`${prefix}Prev`).disabled = offset === 0;
  document.getElementById(`${prefix}Next`).disabled = offset + PAGE_SIZE >= total;
}

async function loadAdminData({ usersOffset = state.usersOffset, pendingOffset = state.pendingOffset } = {}) {
  const requestId = ++state.requestId;
  statusEl.textContent = "読み込み中...";
  pagerButtons.forEach((button) => { button.disabled = true; });
  const userQuery = currentUserQuery();
  try {
    const [pending, users] = await Promise.all([
      fetchJson(`/api/admin/pending-users?limit=${PAGE_SIZE}&offset=${pendingOffset}`),
      fetchJson(`/api/admin/users?limit=${PAGE_SIZE}&offset=${usersOffset}&q=${encodeURIComponent(userQuery)}`),
    ]);
    if (requestId !== state.requestId) return;
    const lastOffset = (data, requested) => Math.min(requested, Math.max(0, Math.ceil(data.total / PAGE_SIZE) - 1) * PAGE_SIZE);
    const lastUsers = lastOffset(users, usersOffset);
    const lastPending = lastOffset(pending, pendingOffset);
    if (lastUsers < usersOffset || lastPending < pendingOffset) {
      return await loadAdminData({ usersOffset: lastUsers, pendingOffset: lastPending });
    }
    state.usersOffset = usersOffset;
    state.pendingOffset = pendingOffset;
    renderPending(pending.items);
    renderUsers(users.items);
    pendingCountEl.textContent = String(pending.total);
    userCountEl.textContent = String(users.total);
    renderPager('pending', pending);
    renderPager('users', users);
    statusEl.textContent = `承認待ち ${pending.total} 件 / ${userQuery ? '検索結果' : 'ユーザー'} ${users.total} 件`;
  } catch (error) {
    if (requestId !== state.requestId) return;
    pagerButtons.forEach((button) => { button.disabled = false; });
    throw error;
  }
}

for (const prefix of ['pending', 'users']) {
  for (const [direction, step] of [['Prev', -1], ['Next', 1]]) {
    document.getElementById(`${prefix}${direction}`)?.addEventListener('click', () => {
      const key = `${prefix}Offset`;
      loadAdminData({ [key]: Math.max(0, state[key] + step * PAGE_SIZE) }).catch((error) => {
        statusEl.textContent = error?.message || '読み込みに失敗しました';
      });
    });
  }
}

refreshBtnEl?.addEventListener("click", () => {
  loadAdminData().catch((error) => {
    statusEl.textContent = error?.message || "読み込みに失敗しました";
  });
});

userSearchFormEl?.addEventListener("submit", (event) => {
  event.preventDefault();
  state.userQuery = userSearchInputEl?.value || "";
  state.usersOffset = 0;
  loadAdminData().catch((error) => {
    statusEl.textContent = error?.message || "読み込みに失敗しました";
  });
});

userSearchInputEl?.addEventListener("search", () => {
  state.userQuery = userSearchInputEl.value || "";
  state.usersOffset = 0;
  loadAdminData().catch((error) => {
    statusEl.textContent = error?.message || "読み込みに失敗しました";
  });
});

userSearchClearBtnEl?.addEventListener("click", () => {
  state.userQuery = "";
  state.usersOffset = 0;
  if (userSearchInputEl) {
    userSearchInputEl.value = "";
    userSearchInputEl.focus();
  }
  loadAdminData().catch((error) => {
    statusEl.textContent = error?.message || "読み込みに失敗しました";
  });
});

window.addEventListener("storage", (event) => {
  if (event.key === "whistx_theme") {
    applyTheme(event.newValue || "light");
  }
});

initTheme();
loadAdminData().catch((error) => {
  statusEl.textContent = error?.message || "読み込みに失敗しました";
});

const serverForm = document.querySelector('#serverSettingsForm');
const serverStatus = document.querySelector('#serverSettingsStatus');
const secretFields = ['ASR_API_KEY', 'SUMMARY_API_KEY', 'OPENWEBUI_API_KEY'];
let settingsVersion = 0;
function showConfiguredSecrets(values) {
  for (const key of secretFields) {
    serverForm.elements.namedItem(key).placeholder = values[key] ? '設定済み（変更時のみ入力）' : '未設定';
  }
}
async function loadServerSettings() {
  const version = settingsVersion;
  try {
    const result = await fetchJson('/api/admin/settings');
    if (version !== settingsVersion) return;
    for (const [key, value] of Object.entries(result.values)) serverForm.elements.namedItem(key).value = value;
    showConfiguredSecrets(result.configuredSecrets);
  } catch { if (version === settingsVersion) serverStatus.textContent = '設定を読み込めませんでした'; }
}
serverForm.addEventListener('submit', async event => {
  event.preventDefault();
  const button = serverForm.querySelector('button');
  if (button.disabled) return;
  settingsVersion++;
  const submitted = Object.fromEntries(new FormData(serverForm));
  let saved = false;
  button.disabled = true;
  try {
    const result = await fetchJson('/api/admin/settings', {method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify(submitted)});
    if (!result?.ok) throw new Error('settings_save_unconfirmed');
    saved = true;
    for (const key of secretFields) {
      const input = serverForm.elements.namedItem(key);
      if (input.value === submitted[key]) input.value = '';
    }
    const current = await fetchJson('/api/admin/settings');
    showConfiguredSecrets(current.configuredSecrets);
    serverStatus.textContent = '保存しました。コンテナ再起動後に適用されます。';
  } catch {
    serverStatus.textContent = saved ? '保存しました。保存状態の表示を確認できませんでした。画面を再読み込みして確認してください。'
      : '保存できませんでした。入力内容を確認してください。';
  }
  finally { button.disabled = false; }
});
loadServerSettings();
