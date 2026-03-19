const $ = (selector) => document.querySelector(selector);

const pendingListEl = $("#pendingList");
const userTableBodyEl = $("#userTableBody");
const pendingCountEl = $("#pendingCount");
const userCountEl = $("#userCount");
const refreshBtnEl = $("#adminRefreshBtn");

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (response.status === 401) {
    window.location.href = "/";
    throw new Error("login_required");
  }
  if (response.status === 403) {
    document.body.innerHTML = '<div class="admin-app"><div class="admin-card"><h1>管理者のみアクセスできます</h1><p class="admin-description"><a href="/" class="admin-link-btn">アプリへ戻る</a></p></div></div>';
    throw new Error("admin_required");
  }
  if (!response.ok) {
    throw new Error(`http_${response.status}`);
  }
  return response.json();
}

function formatDate(value) {
  if (!value) return "未設定";
  try {
    return new Date(value).toLocaleString("ja-JP");
  } catch {
    return String(value);
  }
}

function renderPending(items) {
  pendingCountEl.textContent = String(items.length);
  pendingListEl.innerHTML = "";
  if (!items.length) {
    pendingListEl.innerHTML = '<div class="admin-empty">承認待ちユーザーはありません。</div>';
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("article");
    row.className = "admin-request";
    row.innerHTML = `
      <div class="admin-request-copy">
        <div class="admin-request-name">${escapeHtml(item.displayName || item.email || "pending")}</div>
        <div class="admin-request-email">${escapeHtml(item.email || "")}</div>
        <div class="admin-request-date">申請: ${escapeHtml(formatDate(item.createdAt))}</div>
      </div>
      <button type="button" class="admin-approve-btn">承認</button>
    `;
    row.querySelector(".admin-approve-btn")?.addEventListener("click", async () => {
      await fetchJson(`/api/admin/pending-users/${item.id}/approve`, { method: "POST" });
      await loadAdminData();
    });
    pendingListEl.appendChild(row);
  });
}

function badge(label, className = "") {
  return `<span class="admin-badge ${className}">${escapeHtml(label)}</span>`;
}

function renderUsers(items) {
  userCountEl.textContent = String(items.length);
  userTableBodyEl.innerHTML = "";
  items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>
        <div class="admin-user-name">${escapeHtml(item.displayName || item.email || "-")}</div>
        <div class="admin-user-email">${escapeHtml(item.email || "")}</div>
      </td>
      <td>
        <div class="admin-role-cell">
          ${item.isAdmin ? badge("admin", "is-admin") : badge("member")}
          <select class="admin-role-select" aria-label="権限変更">
            <option value="member" ${item.isAdmin ? "" : "selected"}>通常</option>
            <option value="admin" ${item.isAdmin ? "selected" : ""}>管理者</option>
          </select>
        </div>
      </td>
      <td>${item.isActive ? badge("active", "is-active") : badge("pending", "is-pending")}</td>
      <td class="admin-muted">${escapeHtml(formatDate(item.createdAt))}</td>
      <td class="admin-muted">${escapeHtml(formatDate(item.lastLoginAt))}</td>
    `;
    const selectEl = tr.querySelector(".admin-role-select");
    selectEl?.addEventListener("change", async () => {
      const previous = item.isAdmin ? "admin" : "member";
      const next = String(selectEl.value || previous);
      if (next === previous) return;
      try {
        await fetchJson(`/api/admin/users/${item.id}/role`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ role: next }),
        });
        await loadAdminData();
      } catch (error) {
        if (String(error?.message || "").includes("http_409")) {
          window.alert("最後の管理者は通常ユーザーに変更できません。");
        }
        selectEl.value = previous;
      }
    });
    userTableBodyEl.appendChild(tr);
  });
}

async function loadAdminData() {
  const [pending, users] = await Promise.all([
    fetchJson("/api/admin/pending-users"),
    fetchJson("/api/admin/users"),
  ]);
  renderPending(Array.isArray(pending.items) ? pending.items : []);
  renderUsers(Array.isArray(users.items) ? users.items : []);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

refreshBtnEl?.addEventListener("click", () => {
  loadAdminData().catch(() => {});
});

loadAdminData().catch(() => {});
