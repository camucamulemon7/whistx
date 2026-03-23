import { fetchJson } from "../api/client.js";

export function fetchCapabilities() {
  return fetchJson("/api/health");
}
