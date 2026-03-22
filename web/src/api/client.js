export async function fetchJson(url, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  const startedAt = performance.now();
  console.info("[whistx][api] request", { method, url });

  const response = await fetch(url, options);
  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }
  const durationMs = Math.round(performance.now() - startedAt);
  if (!response.ok) {
    const error = new Error((payload && (payload.error || payload.detail)) || response.statusText || "request_failed");
    error.status = response.status;
    error.payload = payload;
    console.error("[whistx][api] error", {
      method,
      url,
      status: response.status,
      durationMs,
      detail: error.message,
    });
    throw error;
  }
  console.info("[whistx][api] response", { method, url, status: response.status, durationMs });
  return payload;
}
