import { fetchJson } from "./src/api/client.js";

const pendingPanelEl = document.querySelector("#pendingPanel");
const pendingListEl = document.querySelector("#pendingList");
const pendingCountEl = document.querySelector("#pendingCount");
const userCountEl = document.querySelector("#userCount");
const userTableBodyEl = document.querySelector("#userTableBody");
const userCardsEl = document.querySelector("#userCards");
const statusEl = document.querySelector("#adminStatus");
const refreshBtnEl = document.querySelector("#adminRefreshBtn");

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

async function loadAdminData() {
  statusEl.textContent = "読み込み中...";
  const [pending, users] = await Promise.all([
    fetchJson("/api/admin/pending-users"),
    fetchJson("/api/admin/users"),
  ]);
  const pendingItems = Array.isArray(pending.items) ? pending.items : [];
  const userItems = Array.isArray(users.items) ? users.items : [];
  renderPending(pendingItems);
  renderUsers(userItems);
  statusEl.textContent = `承認待ち ${pendingItems.length} 件 / ユーザー ${userItems.length} 件`;
}

refreshBtnEl?.addEventListener("click", () => {
  loadAdminData().catch((error) => {
    statusEl.textContent = error?.message || "読み込みに失敗しました";
  });
});

loadAdminData().catch((error) => {
  statusEl.textContent = error?.message || "読み込みに失敗しました";
});
