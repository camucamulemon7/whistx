import { fetchJson } from "../api/client.js";

export function fetchAuthState() {
  return fetchJson("/api/auth/me");
}

export function loginRequest({ email, password }) {
  return fetchJson("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export function bootstrapAdminRequest({ email, password, displayName }) {
  return fetchJson("/api/auth/bootstrap-admin", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, display_name: displayName || null }),
  });
}

export function registerRequest({ email, password, displayName }) {
  return fetchJson("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, display_name: displayName || null }),
  });
}

export function logoutRequest() {
  return fetchJson("/api/auth/logout", { method: "POST" });
}

export function fetchPendingUsers() {
  return fetchJson("/api/admin/pending-users");
}

export function approvePendingUserRequest(userId) {
  return fetchJson(`/api/admin/pending-users/${userId}/approve`, { method: "POST" });
}
