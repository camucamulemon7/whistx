export function normalizeWsPath(value) {
  const trimmed = String(value || "").trim();
  if (!trimmed) return "/ws/transcribe";
  return trimmed.startsWith("/") ? trimmed : `/${trimmed}`;
}

export function buildWebSocketUrl(locationLike, path) {
  const protocol = locationLike?.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${locationLike?.host || ""}${normalizeWsPath(path)}`;
}

export function waitForOpen(ws, openState = 1) {
  return new Promise((resolve, reject) => {
    if (ws.readyState === openState) {
      resolve();
      return;
    }

    const cleanup = () => {
      ws.removeEventListener("open", onOpen);
      ws.removeEventListener("close", onClose);
      ws.removeEventListener("error", onError);
    };
    const onOpen = () => {
      cleanup();
      resolve();
    };
    const onClose = () => {
      cleanup();
      reject(new Error("websocket_closed"));
    };
    const onError = () => {
      cleanup();
      reject(new Error("websocket_error"));
    };

    ws.addEventListener("open", onOpen);
    ws.addEventListener("close", onClose);
    ws.addEventListener("error", onError);
  });
}

export function waitForSessionReady(ws) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      ws.removeEventListener("message", onMessage);
      ws.removeEventListener("close", onClose);
      ws.removeEventListener("error", onError);
    };
    const onMessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }
      if (data.type === "info" && data.message === "ready") {
        cleanup();
        resolve(data);
      } else if (
        data.type === "error" &&
        (data.message === "session_create_failed" || data.message === "not_started")
      ) {
        cleanup();
        reject(new Error(String(data.detail || data.message || "session_start_failed")));
      }
    };
    const onClose = () => {
      cleanup();
      reject(new Error("websocket_closed"));
    };
    const onError = () => {
      cleanup();
      reject(new Error("websocket_error"));
    };

    ws.addEventListener("message", onMessage);
    ws.addEventListener("close", onClose);
    ws.addEventListener("error", onError);
  });
}

export function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}
