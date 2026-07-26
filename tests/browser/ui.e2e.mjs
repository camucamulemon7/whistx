import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const webRoot = path.join(root, "web");

async function findChrome() {
  const candidates = [
    process.env.CHROME_BIN,
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next supported executable.
    }
  }
  throw new Error("Chrome/Chromium was not found. Set CHROME_BIN to run browser tests.");
}

function contentType(filePath) {
  const extension = path.extname(filePath);
  if (extension === ".html") return "text/html; charset=utf-8";
  if (extension === ".css") return "text/css; charset=utf-8";
  if (extension === ".js") return "text/javascript; charset=utf-8";
  if (extension === ".svg") return "image/svg+xml";
  return "application/octet-stream";
}

async function startStaticServer() {
  const server = http.createServer(async (request, response) => {
    const pathname = new URL(request.url || "/", "http://localhost").pathname;
    const relativePath = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
    const filePath = path.resolve(webRoot, relativePath);
    if (!filePath.startsWith(`${webRoot}${path.sep}`) && filePath !== path.join(webRoot, "index.html")) {
      response.writeHead(403).end();
      return;
    }
    try {
      const body = await readFile(filePath);
      response.writeHead(200, { "Content-Type": contentType(filePath), "Cache-Control": "no-store" });
      response.end(body);
    } catch {
      response.writeHead(404, { "Content-Type": "application/json" });
      response.end("{}");
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  return {
    server,
    url: `http://127.0.0.1:${address.port}/`,
  };
}

function waitForDevTools(child) {
  return new Promise((resolve, reject) => {
    let stderr = "";
    const timeout = setTimeout(() => reject(new Error(`Chrome DevTools did not start:\n${stderr}`)), 10_000);
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
      const match = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (match) {
        clearTimeout(timeout);
        resolve(match[1]);
      }
    });
    child.once("exit", (code) => {
      clearTimeout(timeout);
      reject(new Error(`Chrome exited before DevTools was ready (${code}):\n${stderr}`));
    });
  });
}

class CdpClient {
  constructor(webSocketUrl) {
    this.socket = new WebSocket(webSocketUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.events = [];
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (!message.id) {
        this.events.push(message);
        return;
      }
      if (!this.pending.has(message.id)) return;
      const { resolve, reject } = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) reject(new Error(message.error.message));
      else resolve(message.result);
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    this.socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
  }

  close() {
    this.socket.close();
  }
}

async function connectPage(browserWebSocketUrl, pageUrl) {
  const browserUrl = new URL(browserWebSocketUrl);
  const response = await fetch(
    `http://${browserUrl.host}/json/new?${encodeURIComponent(pageUrl)}`,
    { method: "PUT" },
  );
  assert.equal(response.ok, true, "Chrome should create a page target");
  const target = await response.json();
  const client = new CdpClient(target.webSocketDebuggerUrl);
  await client.open();
  await client.send("Page.enable");
  await client.send("Runtime.enable");
  return client;
}

async function evaluate(client, expression) {
  const result = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || "Browser evaluation failed");
  }
  return result.result.value;
}

async function waitForApp(client) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const ready = await evaluate(
      client,
      `Boolean(
        document.querySelector("#historyDrawerOpen") &&
        document.querySelector("#historyRail") &&
        document.documentElement.dataset.whistxReady === "true"
      )`,
    );
    if (ready) return;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  const diagnostics = await evaluate(
    client,
    `({
      url: location.href,
      readyState: document.readyState,
      scripts: [...document.scripts].map((script) => script.src || "inline"),
      resources: performance.getEntriesByType("resource").map((entry) => entry.name),
      appReady: document.documentElement.dataset.whistxReady
    })`,
  );
  const exceptions = client.events
    .filter((event) => event.method === "Runtime.exceptionThrown")
    .map((event) => event.params?.exceptionDetails?.exception?.description || event.params?.exceptionDetails?.text);
  throw new Error(`Application DOM did not become ready: ${JSON.stringify({ diagnostics, exceptions })}`);
}

async function drawerState(client) {
  return evaluate(
    client,
    `(() => {
      const trigger = document.querySelector("#historyDrawerOpen");
      const rail = document.querySelector("#historyRail");
      const backdrop = document.querySelector("#historyDrawerBackdrop");
      return {
        triggerVisible: getComputedStyle(trigger).display !== "none" && !trigger.hidden,
        expanded: trigger.getAttribute("aria-expanded"),
        railOpen: rail.classList.contains("is-open"),
        railCollapsed: rail.classList.contains("is-collapsed"),
        railHidden: rail.getAttribute("aria-hidden"),
        backdropHidden: backdrop.hidden,
        bodyLocked: document.body.classList.contains("is-history-drawer-open"),
      };
    })()`,
  );
}

async function verifyHistoryDrawerAtWidth(client, width) {
  await client.send("Emulation.setDeviceMetricsOverride", {
    width,
    height: 844,
    deviceScaleFactor: 1,
    mobile: width <= 640,
  });
  await evaluate(client, `window.dispatchEvent(new Event("resize"))`);

  let state = await drawerState(client);
  assert.equal(state.triggerVisible, true, `${width}px: history trigger should be visible`);
  assert.equal(state.railCollapsed, false, `${width}px: desktop collapse must not constrain the drawer`);

  await evaluate(client, `document.querySelector("#historyDrawerOpen").click()`);
  state = await drawerState(client);
  assert.deepEqual(
    {
      expanded: state.expanded,
      railOpen: state.railOpen,
      railHidden: state.railHidden,
      backdropHidden: state.backdropHidden,
      bodyLocked: state.bodyLocked,
    },
    {
      expanded: "true",
      railOpen: true,
      railHidden: "false",
      backdropHidden: false,
      bodyLocked: true,
    },
    `${width}px: opening the history drawer should synchronize visible and accessible state`,
  );

  await evaluate(client, `document.querySelector("#historyDrawerBackdrop").click()`);
  state = await drawerState(client);
  assert.equal(state.railOpen, false, `${width}px: backdrop click should close the drawer`);
  assert.equal(state.expanded, "false", `${width}px: backdrop close should update aria-expanded`);

  await evaluate(client, `document.querySelector("#historyDrawerOpen").click()`);
  await client.send("Input.dispatchKeyEvent", { type: "keyDown", key: "Escape", code: "Escape" });
  await client.send("Input.dispatchKeyEvent", { type: "keyUp", key: "Escape", code: "Escape" });
  state = await drawerState(client);
  assert.equal(state.railOpen, false, `${width}px: Escape should close the drawer`);

  await evaluate(client, `document.querySelector("#historyDrawerOpen").click()`);
  await evaluate(client, `document.querySelector("#historyDrawerClose").click()`);
  state = await drawerState(client);
  assert.equal(state.railOpen, false, `${width}px: close button should close the drawer`);
}

const recordingMocks = String.raw`
  (() => {
    window.__recordingTest = {
      mediaRequests: 0,
      startMessages: 0,
      stopMessages: 0,
      trackStops: 0,
      contextCloses: 0,
      socket: null
    };

    const jsonResponse = (payload) => Promise.resolve(new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    }));
    window.fetch = (input) => {
      const url = String(input);
      if (url.includes("/api/health")) {
        return jsonResponse({
          asrReady: true,
          model: "browser-test",
          wsPath: "/ws/transcribe",
          diarizationEnabled: false,
          proofreadModel: ""
        });
      }
      if (url.includes("/api/auth/me")) {
        return jsonResponse({
          authenticated: false,
          guestTranscriptionAllowed: true,
          bootstrapAdminRequired: false,
          selfSignupEnabled: false
        });
      }
      if (url.includes("/api/glossary/shared")) {
        return jsonResponse({ text: "" });
      }
      return jsonResponse({});
    };

    const audioTrack = {
      id: "audio-track",
      kind: "audio",
      stop() { window.__recordingTest.trackStops += 1; },
      addEventListener() {},
      getSettings() { return {}; }
    };
    const stream = {
      getTracks() { return [audioTrack]; },
      getAudioTracks() { return [audioTrack]; },
      getVideoTracks() { return []; }
    };
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        async getUserMedia() {
          window.__recordingTest.mediaRequests += 1;
          await new Promise((resolve) => setTimeout(resolve, 40));
          return stream;
        }
      }
    });

    class AudioNodeMock {
      connect() {}
      disconnect() {}
    }
    class GainNodeMock extends AudioNodeMock {
      constructor() {
        super();
        this.gain = {
          cancelScheduledValues() {},
          setTargetAtTime() {}
        };
      }
    }
    class AnalyserMock extends AudioNodeMock {
      constructor() {
        super();
        this.fftSize = 2048;
        this.smoothingTimeConstant = 0;
      }
      getFloatTimeDomainData(buffer) {
        buffer.fill(0);
      }
    }
    class AudioContextMock {
      constructor() {
        this.state = "running";
        this.currentTime = 0;
      }
      async resume() {}
      async close() { window.__recordingTest.contextCloses += 1; }
      createMediaStreamDestination() { return { stream }; }
      createMediaStreamSource() { return new AudioNodeMock(); }
      createGain() { return new GainNodeMock(); }
      createAnalyser() { return new AnalyserMock(); }
    }
    window.AudioContext = AudioContextMock;
    window.webkitAudioContext = AudioContextMock;

    class MediaRecorderMock extends EventTarget {
      constructor() {
        super();
        this.state = "inactive";
      }
      start() {
        this.state = "recording";
      }
      stop() {
        if (this.state === "inactive") return;
        this.state = "inactive";
        queueMicrotask(() => this.dispatchEvent(new Event("stop")));
      }
    }
    MediaRecorderMock.isTypeSupported = () => true;
    window.MediaRecorder = MediaRecorderMock;

    class WebSocketMock extends EventTarget {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSING = 2;
      static CLOSED = 3;
      constructor() {
        super();
        this.readyState = WebSocketMock.CONNECTING;
        window.__recordingTest.socket = this;
        queueMicrotask(() => {
          this.readyState = WebSocketMock.OPEN;
          this.dispatchEvent(new Event("open"));
        });
      }
      send(raw) {
        const message = JSON.parse(raw);
        if (message.type === "start") {
          window.__recordingTest.startMessages += 1;
          setTimeout(() => {
            const event = new Event("message");
            event.data = JSON.stringify({ type: "info", message: "ready", sessionId: "browser-test" });
            this.dispatchEvent(event);
          }, 60);
        } else if (message.type === "stop") {
          window.__recordingTest.stopMessages += 1;
        }
      }
      close() {
        this.readyState = WebSocketMock.CLOSED;
        this.dispatchEvent(new Event("close"));
      }
    }
    window.WebSocket = WebSocketMock;
  })();
`;

async function verifyRecordingStartIsSingleFlight(client) {
  await client.send("Page.addScriptToEvaluateOnNewDocument", { source: recordingMocks });
  await client.send("Page.reload", { ignoreCache: true });
  await waitForApp(client);
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const unlocked = await evaluate(client, `!document.body.classList.contains("whistx-auth-locked")`);
    if (unlocked) break;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }

  await evaluate(
    client,
    `(() => {
      const button = document.querySelector("#startBtn");
      button.click();
      button.click();
    })()`,
  );

  const starting = await evaluate(
    client,
    `({
      disabled: document.querySelector("#startBtn").disabled,
      busy: document.querySelector("#startBtn").getAttribute("aria-busy"),
      label: document.querySelector("#startBtn .record-label").textContent
    })`,
  );
  assert.deepEqual(
    starting,
    { disabled: true, busy: "true", label: "準備中..." },
    "record button should expose and lock the starting state",
  );

  await new Promise((resolve) => setTimeout(resolve, 180));
  const result = await evaluate(
    client,
    `({
      ...window.__recordingTest,
      disabled: document.querySelector("#startBtn").disabled,
      busy: document.querySelector("#startBtn").getAttribute("aria-busy"),
      pressed: document.querySelector("#startBtn").getAttribute("aria-pressed")
    })`,
  );
  assert.equal(result.mediaRequests, 1, "double click should request one input stream");
  assert.equal(result.startMessages, 1, "double click should send one WebSocket start message");
  assert.equal(result.disabled, false, "record button should be enabled after startup");
  assert.equal(result.busy, "false", "record button should clear aria-busy after startup");
  assert.equal(result.pressed, "true", "record button should enter recording state");
}

async function verifySocketLossStopsRecording(client) {
  await evaluate(
    client,
    `(() => {
      const event = new Event("message");
      event.data = JSON.stringify({
        type: "final",
        text: "保持すべき文字起こし",
        tsStart: 0,
        tsEnd: 1000,
        seq: 1
      });
      window.__recordingTest.socket.dispatchEvent(event);
    })()`,
  );
  await evaluate(client, `window.__recordingTest.socket.close()`);
  await new Promise((resolve) => setTimeout(resolve, 50));
  const result = await evaluate(
    client,
    `({
      trackStops: window.__recordingTest.trackStops,
      contextCloses: window.__recordingTest.contextCloses,
      pressed: document.querySelector("#startBtn").getAttribute("aria-pressed"),
      label: document.querySelector("#startBtn .record-label").textContent,
      status: document.querySelector("#statusText").textContent
    })`,
  );
  assert.ok(result.trackStops >= 1, "socket loss should stop captured media tracks");
  assert.ok(result.contextCloses >= 2, "socket loss should close capture and VAD audio contexts");
  assert.equal(result.pressed, "false", "socket loss should leave recording UI");
  assert.equal(result.label, "録音開始", "socket loss should restore the start action");
  assert.equal(result.status, "接続切断・録音終了", "socket loss should explain why recording stopped");
}

async function verifyFailedStartPreservesTranscript(client) {
  await evaluate(
    client,
    `(() => {
      window.confirm = () => true;
      window.fetch = (input) => {
        const url = String(input);
        const payload = url.includes("/api/health")
          ? { asrReady: false, model: "" }
          : url.includes("/api/auth/me")
            ? {
                authenticated: false,
                guestTranscriptionAllowed: true,
                bootstrapAdminRequired: false
              }
            : { text: "" };
        return Promise.resolve(new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" }
        }));
      };
      document.querySelector("#startBtn").click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 60));
  const result = await evaluate(
    client,
    `({
      transcript: document.querySelector(".log-row .text")?.textContent || "",
      segmentCount: document.querySelector("#segmentCount").textContent,
      label: document.querySelector("#startBtn .record-label").textContent,
      status: document.querySelector("#statusText").textContent
    })`,
  );
  assert.equal(result.transcript, "保持すべき文字起こし", "failed start should retain transcript text");
  assert.equal(result.segmentCount, "1件", "failed start should retain transcript state");
  assert.equal(result.label, "録音開始", "failed start should restore the start action");
  assert.equal(result.status, "開始失敗", "failed start should be reported without clearing results");

  await evaluate(
    client,
    `(() => {
      window.fetch = (input) => {
        const url = String(input);
        const payload = url.includes("/api/health")
          ? {
              asrReady: true,
              model: "browser-test",
              wsPath: "/ws/transcribe",
              diarizationEnabled: false
            }
          : url.includes("/api/auth/me")
            ? {
                authenticated: false,
                guestTranscriptionAllowed: true,
                bootstrapAdminRequired: false
              }
            : { text: "" };
        return Promise.resolve(new Response(JSON.stringify(payload), {
          status: 200,
          headers: { "Content-Type": "application/json" }
        }));
      };
      navigator.mediaDevices.getUserMedia = async () => {
        throw new DOMException("Permission denied", "NotAllowedError");
      };
      document.querySelector("#startBtn").click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 60));
  const permissionResult = await evaluate(
    client,
    `({
      transcript: document.querySelector(".log-row .text")?.textContent || "",
      segmentCount: document.querySelector("#segmentCount").textContent,
      label: document.querySelector("#startBtn .record-label").textContent
    })`,
  );
  assert.deepEqual(
    permissionResult,
    {
      transcript: "保持すべき文字起こし",
      segmentCount: "1件",
      label: "録音開始",
    },
    "permission rejection should retain the current transcript",
  );

  await evaluate(
    client,
    `(() => {
      window.__recordingTest.socket?.close();
      class FailingWebSocket extends EventTarget {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;
        constructor() {
          super();
          this.readyState = FailingWebSocket.CONNECTING;
          queueMicrotask(() => {
            this.readyState = FailingWebSocket.CLOSED;
            this.dispatchEvent(new Event("close"));
          });
        }
        send() {}
      }
      window.WebSocket = FailingWebSocket;
      document.querySelector("#startBtn").click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 60));
  const socketResult = await evaluate(
    client,
    `({
      transcript: document.querySelector(".log-row .text")?.textContent || "",
      segmentCount: document.querySelector("#segmentCount").textContent,
      label: document.querySelector("#startBtn .record-label").textContent
    })`,
  );
  assert.deepEqual(
    socketResult,
    {
      transcript: "保持すべき文字起こし",
      segmentCount: "1件",
      label: "録音開始",
    },
    "WebSocket startup failure should retain the current transcript",
  );
}

const chrome = await findChrome();
const profileDir = await mkdtemp(path.join(os.tmpdir(), "whistx-chrome-"));
const { server, url } = await startStaticServer();
const chromeProcess = spawn(
  chrome,
  [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--remote-debugging-port=0",
    `--user-data-dir=${profileDir}`,
    "about:blank",
  ],
  { stdio: ["ignore", "ignore", "pipe"] },
);

let client;
try {
  const browserWebSocketUrl = await waitForDevTools(chromeProcess);
  client = await connectPage(browserWebSocketUrl, url);
  await waitForApp(client);
  await evaluate(
    client,
    `(() => {
      document.body.classList.remove("whistx-auth-locked");
      const overlay = document.querySelector("#authGuestView");
      if (overlay) overlay.hidden = true;
    })()`,
  );
  for (const width of [390, 640, 1100]) {
    await verifyHistoryDrawerAtWidth(client, width);
  }
  await verifyRecordingStartIsSingleFlight(client);
  await verifySocketLossStopsRecording(client);
  await verifyFailedStartPreservesTranscript(client);
  process.stdout.write("Browser UI and recording lifecycle checks passed.\n");
} finally {
  client?.close();
  if (chromeProcess.exitCode === null) {
    const exited = new Promise((resolve) => chromeProcess.once("exit", resolve));
    chromeProcess.kill("SIGTERM");
    await exited;
  }
  await new Promise((resolve) => server.close(resolve));
  await rm(profileDir, { recursive: true, force: true });
}
