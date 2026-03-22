import { fetchJson } from "../api/client.js";

export function fetchHistoryList({ limit, offset, query }) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (query) {
    params.set("q", query);
  }
  return fetchJson(`/api/history?${params.toString()}`);
}

export function fetchHistoryDetail(historyId) {
  return fetchJson(`/api/history/${historyId}`);
}

export function saveHistoryRequest(payload) {
  return fetchJson("/api/history", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function deleteHistoryRequest(historyId) {
  return fetchJson(`/api/history/${historyId}`, {
    method: "DELETE",
  });
}
