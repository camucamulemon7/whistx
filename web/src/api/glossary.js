import { fetchJson } from "./client.js";

export function fetchSharedGlossary() {
  return fetchJson("/api/glossary/shared");
}

export function saveSharedGlossary(text) {
  return fetchJson("/api/glossary/shared", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}
