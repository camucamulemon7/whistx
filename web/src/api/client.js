export const API_TIMEOUT_MS = Object.freeze({
  read: 15_000,
  write: 30_000,
  long: 120_000,
});

function requestError(message, code, name = "Error", cause = null) {
  const error = new Error(message);
  error.name = name;
  error.code = code;
  if (cause) error.cause = cause;
  return error;
}

function defaultTimeoutMs(method) {
  return method === "GET" || method === "HEAD" ? API_TIMEOUT_MS.read : API_TIMEOUT_MS.write;
}

export async function fetchJson(url, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  const startedAt = performance.now();
  const timeoutMs = Math.max(1, Number(options.timeoutMs) || defaultTimeoutMs(method));
  const externalSignal = options.signal || null;
  const controller = new AbortController();
  let timedOut = false;
  const relayAbort = () => controller.abort(externalSignal?.reason);
  if (externalSignal?.aborted) {
    relayAbort();
  } else {
    externalSignal?.addEventListener("abort", relayAbort, { once: true });
  }
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort("request_timeout");
  }, timeoutMs);
  const fetchOptions = { ...options, signal: controller.signal };
  delete fetchOptions.timeoutMs;

  console.info("[whistx][api] request", { method, url, timeoutMs });

  try {
    const response = await fetch(url, fetchOptions);
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
  } catch (error) {
    if (timedOut) {
      throw requestError("request_timeout", "timeout", "TimeoutError", error);
    }
    if (externalSignal?.aborted || controller.signal.aborted) {
      throw requestError("request_cancelled", "aborted", "AbortError", error);
    }
    if (typeof navigator !== "undefined" && navigator.onLine === false) {
      throw requestError("offline", "offline", "NetworkError", error);
    }
    if (error instanceof TypeError) {
      throw requestError("network_error", "network", "NetworkError", error);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
    externalSignal?.removeEventListener("abort", relayAbort);
  }
}
