export async function fetchJson(url, options = {}) {
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
  if (!response.ok) {
    const error = new Error((payload && (payload.error || payload.detail)) || response.statusText || "request_failed");
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}
