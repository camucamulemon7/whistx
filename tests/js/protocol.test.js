import test from "node:test";
import assert from "node:assert/strict";

import { readSseJsonStream } from "../../web/src/api/sse.js";
import {
  arrayBufferToBase64,
  buildWebSocketUrl,
  normalizeWsPath,
  waitForOpen,
  waitForSessionFinalized,
  waitForSessionReady,
} from "../../web/src/transcription/websocket.js";

class FakeSocket extends EventTarget {
  constructor(readyState = 0) {
    super();
    this.readyState = readyState;
  }
}

test("WebSocket URL respects page protocol and normalizes path", () => {
  assert.equal(normalizeWsPath("custom"), "/custom");
  assert.equal(buildWebSocketUrl({ protocol: "https:", host: "example.test" }, "custom"), "wss://example.test/custom");
  assert.equal(buildWebSocketUrl({ protocol: "http:", host: "localhost:8005" }, ""), "ws://localhost:8005/ws/transcribe");
});

test("WebSocket lifecycle helpers resolve on protocol readiness", async () => {
  const socket = new FakeSocket();
  const opened = waitForOpen(socket);
  socket.readyState = 1;
  socket.dispatchEvent(new Event("open"));
  await opened;

  const ready = waitForSessionReady(socket);
  const event = new Event("message");
  event.data = JSON.stringify({ type: "info", message: "ready", sessionId: "session-1" });
  socket.dispatchEvent(event);
  assert.equal((await ready).sessionId, "session-1");
});

test("session readiness rejects every server error including duplicate starts", async () => {
  const ws = new EventTarget();
  const ready = waitForSessionReady(ws);
  const event = new Event("message");
  event.data = JSON.stringify({ type: "error", message: "already_started" });
  ws.dispatchEvent(event);

  await assert.rejects(ready, /already_started/);
});

test("session finalization waits for the matching server acknowledgement", async () => {
  const ws = new EventTarget();
  const finalized = waitForSessionFinalized(ws, "session-2", 100);
  const stale = new Event("message");
  stale.data = JSON.stringify({ type: "info", message: "finalized", sessionId: "session-1" });
  ws.dispatchEvent(stale);
  const matching = new Event("message");
  matching.data = JSON.stringify({ type: "info", message: "finalized", sessionId: "session-2" });
  ws.dispatchEvent(matching);

  assert.equal((await finalized).sessionId, "session-2");
});

test("SSE JSON parser handles events split across chunks", async () => {
  const encoder = new TextEncoder();
  const chunks = [encoder.encode('data: {"text":"hel'), encoder.encode('lo"}\n\ndata: {"done":true}\n\n')];
  const response = new Response(new ReadableStream({
    pull(controller) {
      const chunk = chunks.shift();
      if (chunk) controller.enqueue(chunk);
      else controller.close();
    },
  }));
  const events = [];
  await readSseJsonStream(response, (event) => events.push(event));
  assert.deepEqual(events, [{ text: "hello" }, { done: true }]);
});

test("binary payload conversion is stable", () => {
  assert.equal(arrayBufferToBase64(Uint8Array.from([0, 1, 2, 255]).buffer), "AAEC/w==");
});
