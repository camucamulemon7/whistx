import { fetchJson } from "./src/api/client.js";

const pendingTableBodyEl = document.querySelector("#pendingTable tbody");
const usersTableBodyEl = document.querySelector("#usersTable tbody");
const statusEl = document.querySelector("#adminStatus");
const refreshBtnEl = document.querySelector("#refreshBtn");

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function badge(label, className = "") {
  return `<span class="admin-badge ${className}">${escapeHtml(label)}</span>`;
}

function renderPending(items) {
  if (!pendingTableBodyEl) return;
  if (!items.length) {
    pendingTableBodyEl.innerHTML = '<tr><td colspan="4" class="admin-empty">承認待ちはありません</td></tr>';
    return;
  }
  pendingTableBodyEl.innerHTML = items.map((item) => `
    <tr>
      <td>${escapeHtml(item.displayName || "-")}</td>
      <td>${escapeHtml(item.email)}</td>
      <td>${formatDate(item.createdAt)}</td>
      <td><button class="admin-approve-btn" data-user-id="${item.id}">承認</button></td>
    </tr>
  `).join("");
  pendingTableBodyEl.querySelectorAll(".admin-approve-btn").forEach((button) => {
    button.addEventListener("click", async () => {
      await fetchJson(`/api/admin/pending-users/${button.dataset.userId}/approve`, { method: "POST" });
      await loadAdminData();
    });
  });
}

function renderUsers(items) {
  if (!usersTableBodyEl) return;
  usersTableBodyEl.innerHTML = items.map((item) => `
    <tr>
      <td>${escapeHtml(item.displayName || "-")}</td>
      <td>${escapeHtml(item.email)}</td>
      <td>${item.isAdmin ? badge("admin", "is-admin") : badge("member")}</td>
      <td>${item.isActive ? badge("active", "is-active") : badge("pending", "is-pending")}</td>
      <td>${formatDate(item.createdAt)}</td>
      <td>${formatDate(item.lastLoginAt)}</td>
      <td>
        <select class="admin-role-select" data-user-id="${item.id}">
          <option value="member" ${item.isAdmin ? "" : "selected"}>member</option>
          <option value="admin" ${item.isAdmin ? "selected" : ""}>admin</option>
        </select>
      </td>
    </tr>
  `).join("");
  usersTableBodyEl.querySelectorAll(".admin-role-select").forEach((selectEl) => {
    selectEl.addEventListener("change", async () => {
      await fetchJson(`/api/admin/users/${selectEl.dataset.userId}/role`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: selectEl.value }),
      });
      await loadAdminData();
    });
  });
}

async function loadAdminData() {
  statusEl.textContent = "読み込み中...";
  const [pending, users] = await Promise.all([
    fetchJson("/api/admin/pending-users"),
    fetchJson("/api/admin/users"),
  ]);
  renderPending(pending.items || []);
  renderUsers(users.items || []);
  statusEl.textContent = `承認待ち ${pending.items?.length || 0} 件 / ユーザー ${users.items?.length || 0} 件`;
}

refreshBtnEl?.addEventListener("click", () => {
  loadAdminData().catch((error) => {
    statusEl.textContent = error?.message || "読み込みに失敗しました";
  });
});

loadAdminData().catch((error) => {
  statusEl.textContent = error?.message || "読み込みに失敗しました";
});
