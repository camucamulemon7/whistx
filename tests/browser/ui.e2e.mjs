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

async function unloadProtectionState(client) {
  return evaluate(
    client,
    `(() => {
      const event = new Event("beforeunload", { cancelable: true });
      const dispatched = window.dispatchEvent(event);
      return {
        dirty: document.documentElement.dataset.unsavedTranscript || "false",
        prevented: event.defaultPrevented,
        dispatched
      };
    })()`,
  );
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
  const transcriptState = await evaluate(
    client,
    `({
      collapsed: document.querySelector(".transcript-panel").classList.contains("is-collapsed"),
      contentVisible: getComputedStyle(document.querySelector("#log")).display !== "none"
    })`,
  );
  assert.deepEqual(
    transcriptState,
    { collapsed: false, contentVisible: true },
    `${width}px: desktop transcript collapse must not hide narrow-layout content`,
  );

  await evaluate(client, `document.querySelector("#historyDrawerOpen").click()`);
  state = await drawerState(client);
  const drawerLayout = await evaluate(
    client,
    `(() => {
      const rail = document.querySelector("#historyRail");
      const rect = rail.getBoundingClientRect();
      return {
        position: getComputedStyle(rail).position,
        top: rect.top,
        left: rect.left,
        height: rect.height,
        viewportHeight: window.innerHeight
      };
    })()`,
  );
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
  assert.equal(drawerLayout.position, "fixed", `${width}px: history must remain a viewport drawer`);
  assert.ok(Math.abs(drawerLayout.top) <= 1, `${width}px: history drawer must align with the viewport top`);
  assert.ok(Math.abs(drawerLayout.left) <= 1, `${width}px: history drawer must align with the viewport left`);
  assert.ok(
    Math.abs(drawerLayout.height - drawerLayout.viewportHeight) <= 1,
    `${width}px: history drawer must fill the viewport height`,
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

async function verifyDesktopPanelLayout(client) {
  await client.send("Emulation.setDeviceMetricsOverride", {
    width: 1280,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await evaluate(client, `window.dispatchEvent(new Event("resize"))`);
  const desktopHistoryPosition = await evaluate(
    client,
    `(() => {
      const rail = document.querySelector("#historyRail");
      const main = document.querySelector(".workspace-main");
      const railRect = rail.getBoundingClientRect();
      const mainRect = main.getBoundingClientRect();
      return {
        position: getComputedStyle(rail).position,
        topDifference: Math.abs(railRect.top - mainRect.top)
      };
    })()`,
  );
  assert.equal(desktopHistoryPosition.position, "sticky", "desktop history must retain its sticky sidebar position");
  assert.ok(desktopHistoryPosition.topDifference <= 1, "desktop history must align with the workspace content");
  let layout = await evaluate(
    client,
    `({
      columns: getComputedStyle(document.querySelector("#workspacePanels")).gridTemplateColumns.split(" ").length,
      resizerDisplay: getComputedStyle(document.querySelector('[data-resizer="left"]')).display
    })`,
  );
  assert.equal(layout.columns, 1, "1280px should use a single workspace column");
  assert.equal(layout.resizerDisplay, "none", "1280px should hide desktop-only resizers");

  await client.send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await evaluate(client, `window.dispatchEvent(new Event("resize"))`);
  const historyLayout = await evaluate(
    client,
    `(() => {
      const rail = document.querySelector("#historyRail");
      const workspace = document.querySelector(".workspace-main");
      const railRect = rail.getBoundingClientRect();
      const workspaceRect = workspace.getBoundingClientRect();
      return {
        railTop: railRect.top,
        workspaceTop: workspaceRect.top,
        railHeight: railRect.height,
        workspaceHeight: workspaceRect.height,
        minHeight: getComputedStyle(rail).minHeight,
        position: getComputedStyle(rail).position
      };
    })()`,
  );
  assert.ok(
    Math.abs(historyLayout.railTop - historyLayout.workspaceTop) <= 1,
    "desktop history rail should align with the workspace top",
  );
  assert.equal(historyLayout.position, "sticky", "desktop history rail should retain its sticky positioning");
  assert.notEqual(historyLayout.minHeight, "100%", "runtime styles must not stretch the static history rail");
  assert.ok(
    historyLayout.railHeight < historyLayout.workspaceHeight,
    "history rail should size to its contents instead of the full workspace",
  );
  const beforeCollapse = await evaluate(
    client,
    `({
      transcript: document.querySelector(".transcript-panel").getBoundingClientRect().width,
      proofread: document.querySelector(".proofread-panel").getBoundingClientRect().width,
      resizerDisplay: getComputedStyle(document.querySelector('[data-resizer="left"]')).display
    })`,
  );
  assert.notEqual(beforeCollapse.resizerDisplay, "none", "1440px should expose panel resizers");
  await evaluate(client, `document.querySelector('[data-panel-toggle="transcript"]').click()`);
  const afterCollapse = await evaluate(
    client,
    `({
      transcript: document.querySelector(".transcript-panel").getBoundingClientRect().width,
      proofread: document.querySelector(".proofread-panel").getBoundingClientRect().width
    })`,
  );
  assert.ok(afterCollapse.transcript <= 92, "collapsed transcript should shrink to its compact desktop width");
  assert.ok(
    afterCollapse.transcript < beforeCollapse.transcript - 100,
    "collapsing transcript should release substantial workspace width",
  );
  assert.ok(afterCollapse.proofread > beforeCollapse.proofread, "adjacent panels should use released width");

  await client.send("Emulation.setDeviceMetricsOverride", {
    width: 1280,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await evaluate(client, `window.dispatchEvent(new Event("resize"))`);
  const narrowAfterCollapse = await evaluate(
    client,
    `({
      collapsedClass: document.querySelector(".transcript-panel").classList.contains("is-collapsed"),
      transcriptVisible: getComputedStyle(document.querySelector("#log")).display !== "none",
      toggleHidden: document.querySelector('[data-panel-toggle="transcript"]').hidden,
      columns: getComputedStyle(document.querySelector("#workspacePanels")).gridTemplateColumns.split(" ").length
    })`,
  );
  assert.deepEqual(
    narrowAfterCollapse,
    { collapsedClass: false, transcriptVisible: true, toggleHidden: true, columns: 1 },
    "one-column layout should suspend desktop collapse without hiding transcript content",
  );

  await client.send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await evaluate(client, `window.dispatchEvent(new Event("resize"))`);
  const restoredCollapse = await evaluate(
    client,
    `({
      collapsedClass: document.querySelector(".transcript-panel").classList.contains("is-collapsed"),
      width: document.querySelector(".transcript-panel").getBoundingClientRect().width,
      toggleHidden: document.querySelector('[data-panel-toggle="transcript"]').hidden
    })`,
  );
  assert.equal(restoredCollapse.collapsedClass, true, "desktop collapse preference should return after widening");
  assert.ok(restoredCollapse.width <= 92, "restored desktop collapse should use compact width");
  assert.equal(restoredCollapse.toggleHidden, false, "desktop collapse control should return after widening");
  await evaluate(client, `document.querySelector('[data-panel-toggle="transcript"]').click()`);

  await client.send("Emulation.setDeviceMetricsOverride", {
    width: 1920,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await evaluate(client, `window.dispatchEvent(new Event("resize"))`);
  const beforeResize = await evaluate(
    client,
    `(() => {
      const transcript = document.querySelector(".transcript-panel").getBoundingClientRect();
      const proofread = document.querySelector(".proofread-panel").getBoundingClientRect();
      const handle = document.querySelector('[data-resizer="left"]').getBoundingClientRect();
      return {
        transcript: transcript.width,
        proofread: proofread.width,
        handleX: handle.left + handle.width / 2,
        handleY: handle.top + Math.min(80, handle.height / 2)
      };
    })()`,
  );
  await evaluate(
    client,
    `(() => {
      const handle = document.querySelector('[data-resizer="left"]');
      handle.dispatchEvent(new PointerEvent("pointerdown", {
        bubbles: true,
        pointerId: 1,
        clientX: ${beforeResize.handleX},
        clientY: ${beforeResize.handleY}
      }));
      window.dispatchEvent(new PointerEvent("pointermove", {
        bubbles: true,
        pointerId: 1,
        clientX: ${beforeResize.handleX + 120},
        clientY: ${beforeResize.handleY}
      }));
      window.dispatchEvent(new PointerEvent("pointerup", {
        bubbles: true,
        pointerId: 1,
        clientX: ${beforeResize.handleX + 120},
        clientY: ${beforeResize.handleY}
      }));
    })()`,
  );
  const afterResize = await evaluate(
    client,
    `({
      transcript: document.querySelector(".transcript-panel").getBoundingClientRect().width,
      proofread: document.querySelector(".proofread-panel").getBoundingClientRect().width
    })`,
  );
  assert.ok(
    afterResize.transcript > beforeResize.transcript + 40,
    `dragging should widen the transcript panel: ${JSON.stringify({ beforeResize, afterResize })}`,
  );
  assert.ok(
    afterResize.proofread < beforeResize.proofread - 20,
    `dragging should resize the adjacent panel: ${JSON.stringify({ beforeResize, afterResize })}`,
  );
}

const recordingMocks = String.raw`
  (() => {
    window.__recordingTest = {
      instanceId: crypto.randomUUID(),
      mediaRequests: 0,
      startMessages: 0,
      startPayloads: [],
      stopMessages: 0,
      trackStops: 0,
      contextCloses: 0,
      historyDetailRequests: 0,
      clipboardWrites: [],
      authRequests: [],
      socket: null,
      sockets: []
    };
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        async writeText(text) {
          window.__recordingTest.clipboardWrites.push(String(text));
        }
      }
    });

    const jsonResponse = (payload) => Promise.resolve(new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    }));
    window.fetch = (input, options = {}) => {
      const url = String(input);
      if (url.includes("/api/health")) {
        return jsonResponse({
          asrReady: true,
          model: "browser-test",
          wsPath: "/ws/transcribe",
          diarizationEnabled: true,
          proofreadModel: ""
        });
      }
      if (url.includes("/api/auth/me")) {
        window.__recordingTest.authRequests.push({
          cache: options.cache || "",
          credentials: options.credentials || ""
        });
        return jsonResponse({
          authenticated: true,
          user: {
            id: "browser-user",
            email: "browser@example.com",
            displayName: "Browser Test",
            isAdmin: false
          },
          guestTranscriptionAllowed: true,
          bootstrapAdminRequired: false,
          selfSignupEnabled: false
        });
      }
      if (url.includes("/api/history/history-1")) {
        window.__recordingTest.historyDetailRequests += 1;
        return jsonResponse({
          id: "history-1",
          title: "既存の履歴",
          savedAt: "2026-01-01T00:00:00Z",
          segments: [{ text: "履歴の文字起こし", tsStart: 0, tsEnd: 1000, seq: 1 }]
        });
      }
      if (url.includes("/api/history?")) {
        return jsonResponse({
          total: 1,
          items: [{
            id: "history-1",
            title: "既存の履歴",
            preview: "履歴の文字起こし",
            language: "ja",
            savedAt: "2026-01-01T00:00:00Z"
          }]
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
        window.__recordingTest.sockets.push(this);
        queueMicrotask(() => {
          this.readyState = WebSocketMock.OPEN;
          this.dispatchEvent(new Event("open"));
        });
      }
      send(raw) {
        const message = JSON.parse(raw);
        if (message.type === "start") {
          window.__recordingTest.startMessages += 1;
          window.__recordingTest.startPayloads.push(message);
          this.sessionId = "browser-test-" + window.__recordingTest.startMessages;
          setTimeout(() => {
            const event = new Event("message");
            event.data = JSON.stringify({
              type: "info",
              message: "ready",
              sessionId: this.sessionId
            });
            this.dispatchEvent(event);
          }, 60);
        } else if (message.type === "stop") {
          window.__recordingTest.stopMessages += 1;
          const stopping = new Event("message");
          stopping.data = JSON.stringify({
            type: "info",
            message: "stopping",
            sessionId: this.sessionId
          });
          this.dispatchEvent(stopping);
          setTimeout(() => {
            const final = new Event("message");
            final.data = JSON.stringify({
              type: "final",
              sessionId: this.sessionId,
              text: "停止直前の文字起こし",
              tsStart: 1000,
              tsEnd: 2000,
              seq: 2
            });
            this.dispatchEvent(final);
          }, 35);
          setTimeout(() => {
            const finalized = new Event("message");
            finalized.data = JSON.stringify({
              type: "info",
              message: "finalized",
              state: "completed",
              sessionId: this.sessionId
            });
            this.dispatchEvent(finalized);
          }, 90);
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
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const ready = await evaluate(
      client,
      `document.documentElement.dataset.whistxReady === "true" && Boolean(window.__recordingTest?.instanceId)`,
    );
    if (ready) break;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const unlocked = await evaluate(client, `!document.body.classList.contains("whistx-auth-locked")`);
    if (unlocked) break;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  await verifyEmptySummaryIsNotCopied(client);
  await verifyEmptyDownloadsAreDisabled(client);
  assert.deepEqual(
    await evaluate(client, `window.__recordingTest.authRequests[0]`),
    { cache: "no-store", credentials: "same-origin" },
    "browser auth bootstrap should bypass caches and send same-origin credentials",
  );
  assert.deepEqual(
    await unloadProtectionState(client),
    { dirty: "false", prevented: false, dispatched: true },
    "empty workspace should not register unload protection",
  );

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
      label: document.querySelector("#startBtn .record-label").textContent,
      dirty: document.documentElement.dataset.unsavedTranscript,
      settings: {
        language: document.querySelector("#language").disabled,
        audioSource: document.querySelector("#audioSource").disabled,
        chunkSeconds: document.querySelector("#chunkSeconds").disabled,
        prompt: document.querySelector("#prompt").disabled,
        sharedVocabulary: document.querySelector("#sharedVocabulary").disabled,
        chunkPreset: document.querySelector("[data-chunk-preset]").disabled,
        promptTemplate: document.querySelector(".prompt-template-btn").disabled,
        diarization: document.querySelector("#diarizationEnabled").disabled
      }
    })`,
  );
  assert.deepEqual(
    starting,
    {
      disabled: true,
      busy: "true",
      label: "準備中...",
      dirty: "true",
      settings: {
        language: true,
        audioSource: true,
        chunkSeconds: true,
        prompt: true,
        sharedVocabulary: true,
        chunkPreset: true,
        promptTemplate: true,
        diarization: true,
      },
    },
    "record button should expose and lock the starting state",
  );

  await new Promise((resolve) => setTimeout(resolve, 180));
  const result = await evaluate(
    client,
    `({
      ...window.__recordingTest,
      disabled: document.querySelector("#startBtn").disabled,
      busy: document.querySelector("#startBtn").getAttribute("aria-busy"),
      pressed: document.querySelector("#startBtn").getAttribute("aria-pressed"),
      downloadHref: document.querySelector("#dlTxt").getAttribute("href"),
      downloadDisabled: document.querySelector("#dlTxt").getAttribute("aria-disabled"),
      settingsLocked: [
        "#language",
        "#audioSource",
        "#chunkSeconds",
        "#prompt",
        "#sharedVocabulary",
        "[data-chunk-preset]",
        ".prompt-template-btn",
        "#diarizationEnabled",
        "#diarizationSpeakerMode"
      ].every((selector) => document.querySelector(selector).disabled)
    })`,
  );
  assert.equal(result.mediaRequests, 1, "double click should request one input stream");
  assert.equal(result.startMessages, 1, "double click should send one WebSocket start message");
  assert.equal(result.disabled, false, "record button should be enabled after startup");
  assert.equal(result.busy, "false", "record button should clear aria-busy after startup");
  assert.equal(result.pressed, "true", "record button should enter recording state");
  assert.equal(result.downloadHref, null, "recording should not expose a partial artifact URL");
  assert.equal(result.downloadDisabled, "true", "recording exports should remain disabled");
  assert.equal(result.settingsLocked, true, "session settings should remain locked while recording");
  assert.deepEqual(
    await unloadProtectionState(client),
    { dirty: "true", prevented: true, dispatched: false },
    "recording should protect against page unload even before finalization",
  );
}

async function verifyEmptyDownloadsAreDisabled(client) {
  const beforeUrl = await evaluate(client, `location.href`);
  await evaluate(client, `document.querySelector("#dlTxt").click()`);
  const result = await evaluate(
    client,
    `({
      href: document.querySelector("#dlTxt").getAttribute("href"),
      disabled: document.querySelector("#dlTxt").getAttribute("aria-disabled"),
      downloaded: document.querySelector("#dlTxt").classList.contains("is-downloaded"),
      url: location.href,
      toast: document.querySelector("#toastContainer .toast:last-child")?.textContent || ""
    })`,
  );
  assert.equal(result.href, null, "empty workspace exports should have no href");
  assert.equal(result.disabled, "true", "empty workspace exports should be exposed as disabled");
  assert.equal(result.downloaded, false, "disabled export must not show success feedback");
  assert.equal(result.url, beforeUrl, "disabled export must not navigate the current page");
  assert.match(result.toast, /書き出せる文字起こしがありません/, "disabled export should explain why");
}

async function verifyEmptySummaryIsNotCopied(client) {
  await evaluate(
    client,
    `(() => {
      window.__summaryClipboardWrites = [];
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: {
          async writeText(text) {
            window.__summaryClipboardWrites.push(String(text));
          }
        }
      });
      document.querySelector("#copySummaryBtn").click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 20));
  const result = await evaluate(
    client,
    `({
      clipboardWrites: window.__summaryClipboardWrites,
      toast: document.querySelector("#toastContainer .toast:last-child")?.textContent || ""
    })`,
  );
  assert.deepEqual(result.clipboardWrites, [], "empty summary placeholder must not be written to the clipboard");
  assert.match(result.toast, /要約がありません/, "empty summary copy should explain that no result exists");
}

async function verifyDestructiveActionsAreLocked(client) {
  await evaluate(
    client,
    `(() => {
      const event = new Event("message");
      event.data = JSON.stringify({
        type: "final",
        sessionId: "browser-test-1",
        text: "録音中に保持する文字起こし https://example.com/very/long/path/that/must/wrap/without/causing/horizontal/overflow/abcdefghijklmnopqrstuvwxyz0123456789",
        tsStart: 0,
        tsEnd: 1000,
        seq: 1,
        screenshotPath: "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==",
        rawAudioPath: "data:audio/wav;base64,UklGRg==",
        audioPath: "data:audio/wav;base64,UklGRg=="
      });
      window.__recordingTest.socket.dispatchEvent(event);
      const audioOnly = new Event("message");
      audioOnly.data = JSON.stringify({
        type: "final",
        sessionId: "browser-test-1",
        text: "音声のみの行",
        tsStart: 1000,
        tsEnd: 2000,
        seq: 2,
        rawAudioPath: "data:audio/wav;base64,UklGRg=="
      });
      window.__recordingTest.socket.dispatchEvent(audioOnly);
      const imageOnly = new Event("message");
      imageOnly.data = JSON.stringify({
        type: "final",
        sessionId: "browser-test-1",
        text: "画像のみの行",
        tsStart: 2000,
        tsEnd: 3000,
        seq: 3,
        screenshotPath: "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
      });
      window.__recordingTest.socket.dispatchEvent(imageOnly);
      document.querySelector("#clearBtn").click();
      document.querySelector(".history-item-main").click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 20));
  const result = await evaluate(
    client,
    `({
      clearDisabled: document.querySelector("#clearBtn").disabled,
      historyDisabled: document.querySelector(".history-item-main").getAttribute("aria-disabled"),
      historyDetailRequests: window.__recordingTest.historyDetailRequests,
      transcript: document.querySelector("#log").textContent
    })`,
  );
  assert.equal(result.clearDisabled, true, "clear must be disabled while recording");
  assert.equal(result.historyDisabled, "true", "history switching must be marked disabled while recording");
  assert.equal(result.historyDetailRequests, 0, "recording must block history detail requests");
  assert.match(result.transcript, /録音中に保持する文字起こし/, "blocked clear must preserve the transcript");
}

async function verifyTranscriptMediaResponsive(client) {
  for (const width of [390, 640, 768, 900]) {
    await client.send("Emulation.setDeviceMetricsOverride", {
      width,
      height: 900,
      deviceScaleFactor: 1,
      mobile: width <= 640,
    });
    await evaluate(client, `window.dispatchEvent(new Event("resize"))`);
    const result = await evaluate(
      client,
      `(() => {
        document.querySelectorAll(".log-inline-audio").forEach((audio) => {
          audio.hidden = false;
        });
        const log = document.querySelector("#log");
        const logRect = log.getBoundingClientRect();
        const rows = [...document.querySelectorAll(".log-row")];
        return {
          logOverflow: log.scrollWidth - log.clientWidth,
          rowOverflows: rows.map((row) => row.scrollWidth - row.clientWidth),
          outsideRows: rows.filter((row) => row.getBoundingClientRect().right > logRect.right + 1).length,
          outsideMedia: [...document.querySelectorAll(".log-media-group")].filter(
            (media) => media.getBoundingClientRect().right > logRect.right + 1
          ).length,
          bodyDirections: [...document.querySelectorAll(".log-body")].map(
            (body) => getComputedStyle(body).flexDirection
          ),
          mediaTypes: [...document.querySelectorAll(".log-media-group")].map(
            (media) => [media.dataset.hasAudio, media.dataset.hasScreenshot].join("/")
          ),
          textWidths: [...document.querySelectorAll(".log-row .text")].map(
            (text) => text.getBoundingClientRect().width
          )
        };
      })()`,
    );
    assert.ok(result.logOverflow <= 1, `${width}px: transcript container must not overflow horizontally`);
    assert.ok(result.rowOverflows.every((overflow) => overflow <= 1), `${width}px: rows must stay within their width`);
    assert.equal(result.outsideRows, 0, `${width}px: rows must stay inside the transcript panel`);
    assert.equal(result.outsideMedia, 0, `${width}px: media controls must stay inside the transcript panel`);
    assert.ok(result.bodyDirections.every((direction) => direction === "column"), `${width}px: text and media should stack`);
    assert.ok(result.textWidths.every((textWidth) => textWidth > 80), `${width}px: transcript text needs readable width`);
    assert.ok(result.mediaTypes.includes("true/true"), `${width}px: audio+image row should be covered`);
    assert.ok(result.mediaTypes.includes("true/false"), `${width}px: audio-only row should be covered`);
    assert.ok(result.mediaTypes.includes("false/true"), `${width}px: image-only row should be covered`);
  }
}

async function verifyTranscriptAutoScroll(client) {
  await evaluate(
    client,
    `(() => {
      for (let index = 0; index < 36; index += 1) {
        const event = new Event("message");
        event.data = JSON.stringify({
          type: "final",
          sessionId: "browser-test-1",
          text: \`自動スクロール確認用の文字起こし \${index + 1}\`,
          tsStart: (index + 3) * 1000,
          tsEnd: (index + 4) * 1000,
          seq: index + 4
        });
        window.__recordingTest.socket.dispatchEvent(event);
      }
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 30));

  for (const width of [1440, 390]) {
    await client.send("Emulation.setDeviceMetricsOverride", {
      width,
      height: 900,
      deviceScaleFactor: 1,
      mobile: width <= 640,
    });
    await evaluate(client, `window.dispatchEvent(new Event("resize"))`);

    let result = await evaluate(
      client,
      `(() => {
        const log = document.querySelector("#log");
        const downloads = document.querySelector(".downloads-bar");
        log.scrollTop = log.scrollHeight;
        log.dispatchEvent(new Event("scroll"));
        return {
          scrollable: log.scrollHeight > log.clientHeight + 10,
          downloadsAfterLog: downloads.getBoundingClientRect().top >= log.getBoundingClientRect().bottom - 1,
          horizontalOverflow: log.scrollWidth - log.clientWidth
        };
      })()`,
    );
    assert.equal(result.scrollable, true, `${width}px: transcript must have its own vertical scroll area`);
    assert.equal(result.downloadsAfterLog, true, `${width}px: save and export controls must remain outside the transcript scroll area`);
    assert.ok(result.horizontalOverflow <= 1, `${width}px: scrollable transcript must not overflow horizontally`);

    await evaluate(
      client,
      `(() => {
        const event = new Event("message");
        event.data = JSON.stringify({
          type: "final",
          sessionId: "browser-test-1",
          text: "末尾追従確認 ${width}px",
          tsStart: ${width} * 1000,
          tsEnd: (${width} + 1) * 1000,
          seq: ${width}
        });
        window.__recordingTest.socket.dispatchEvent(event);
      })()`,
    );
    await new Promise((resolve) => setTimeout(resolve, 30));
    result = await evaluate(
      client,
      `(() => {
        const log = document.querySelector("#log");
        return {
          distanceFromBottom: log.scrollHeight - log.clientHeight - log.scrollTop,
          hasLatest: log.textContent.includes("末尾追従確認 ${width}px")
        };
      })()`,
    );
    assert.ok(result.distanceFromBottom <= 2, `${width}px: new transcript rows should follow the bottom`);
    assert.equal(result.hasLatest, true, `${width}px: followed transcript row should be rendered`);

    await evaluate(
      client,
      `(() => {
        const log = document.querySelector("#log");
        log.scrollTop = 0;
        log.dispatchEvent(new Event("scroll"));
        const event = new Event("message");
        event.data = JSON.stringify({
          type: "final",
          sessionId: "browser-test-1",
          text: "過去閲覧位置維持 ${width}px",
          tsStart: (${width} + 2) * 1000,
          tsEnd: (${width} + 3) * 1000,
          seq: ${width} + 1
        });
        window.__recordingTest.socket.dispatchEvent(event);
      })()`,
    );
    await new Promise((resolve) => setTimeout(resolve, 30));
    result = await evaluate(
      client,
      `(() => {
        const log = document.querySelector("#log");
        return {
          scrollTop: log.scrollTop,
          hasLatest: log.textContent.includes("過去閲覧位置維持 ${width}px")
        };
      })()`,
    );
    assert.equal(result.scrollTop, 0, `${width}px: new rows must not steal the position while reading older transcript`);
    assert.equal(result.hasLatest, true, `${width}px: transcript should still update while auto-follow is paused`);
  }
}

async function verifyModalKeyboardManagement(client) {
  await evaluate(
    client,
    `(() => {
      const help = document.querySelector("#helpBtn");
      help.focus();
      help.click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 30));
  let state = await evaluate(
    client,
    `({
      helpHidden: document.querySelector("#helpModal").hidden,
      activeId: document.activeElement?.id || "",
      bodyLocked: document.body.classList.contains("is-modal-open")
    })`,
  );
  assert.equal(state.helpHidden, false, "help modal should open");
  assert.equal(state.activeId, "helpModalClose", "help modal should receive initial focus");
  assert.equal(state.bodyLocked, true, "open modal should lock background scrolling");

  await evaluate(client, `document.querySelector(".log-screenshot-link").click()`);
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (await evaluate(client, `document.activeElement?.id === "screenshotModalClose"`)) break;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  state = await evaluate(
    client,
    `({
      helpHidden: document.querySelector("#helpModal").hidden,
      screenshotHidden: document.querySelector("#screenshotModal").hidden,
      activeId: document.activeElement?.id || ""
    })`,
  );
  assert.equal(state.helpHidden, false, "opening a second modal should keep the underlying modal");
  assert.equal(state.screenshotHidden, false, "screenshot modal should open above help");
  assert.equal(state.activeId, "screenshotModalClose", "topmost modal should receive focus");

  await client.send("Input.dispatchKeyEvent", { type: "keyDown", key: "Escape", code: "Escape" });
  await client.send("Input.dispatchKeyEvent", { type: "keyUp", key: "Escape", code: "Escape" });
  state = await evaluate(
    client,
    `({
      helpHidden: document.querySelector("#helpModal").hidden,
      screenshotHidden: document.querySelector("#screenshotModal").hidden,
      activeId: document.activeElement?.id || "",
      bodyLocked: document.body.classList.contains("is-modal-open")
    })`,
  );
  assert.equal(state.screenshotHidden, true, "Escape should close only the topmost modal");
  assert.equal(state.helpHidden, false, "underlying modal should remain open after one Escape");
  assert.equal(state.activeId, "helpModalClose", "focus should return to the underlying modal");
  assert.equal(state.bodyLocked, true, "background should remain locked while another modal is open");

  await evaluate(
    client,
    `document.activeElement.dispatchEvent(new KeyboardEvent("keydown", {
      key: "Tab",
      shiftKey: true,
      bubbles: true
    }))`,
  );
  assert.equal(
    await evaluate(client, `document.activeElement?.id || document.activeElement?.tagName || ""`),
    "helpModalFrame",
    "Shift+Tab from the first control should wrap to the last modal control",
  );
  await evaluate(
    client,
    `document.activeElement.dispatchEvent(new KeyboardEvent("keydown", {
      key: "Tab",
      bubbles: true
    }))`,
  );
  assert.equal(
    await evaluate(client, `document.activeElement?.id || ""`),
    "helpModalClose",
    "Tab from the last control should wrap to the first modal control",
  );

  await client.send("Input.dispatchKeyEvent", { type: "keyDown", key: "Escape", code: "Escape" });
  await client.send("Input.dispatchKeyEvent", { type: "keyUp", key: "Escape", code: "Escape" });
  state = await evaluate(
    client,
    `({
      helpHidden: document.querySelector("#helpModal").hidden,
      activeId: document.activeElement?.id || "",
      bodyLocked: document.body.classList.contains("is-modal-open")
    })`,
  );
  assert.equal(state.helpHidden, true, "second Escape should close the remaining modal");
  assert.equal(state.activeId, "helpBtn", "closing should restore focus to the opener");
  assert.equal(state.bodyLocked, false, "closing the final modal should unlock background scrolling");
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

async function verifyGracefulStopIsSerialized(client) {
  await evaluate(
    client,
    `(() => {
      const button = document.querySelector("#startBtn");
      button.click();
      button.click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 20));
  const finalizing = await evaluate(
    client,
    `({
      startMessages: window.__recordingTest.startMessages,
      stopMessages: window.__recordingTest.stopMessages,
      disabled: document.querySelector("#startBtn").disabled,
      busy: document.querySelector("#startBtn").getAttribute("aria-busy"),
      label: document.querySelector("#startBtn .record-label").textContent,
      clearDisabled: document.querySelector("#clearBtn").disabled,
      historyDisabled: document.querySelector(".history-item-main").getAttribute("aria-disabled"),
      historyDetailRequests: window.__recordingTest.historyDetailRequests,
      downloadHref: document.querySelector("#dlTxt").getAttribute("href"),
      downloadDisabled: document.querySelector("#dlTxt").getAttribute("aria-disabled"),
      settingsLocked: [
        "#language",
        "#audioSource",
        "#chunkSeconds",
        "#prompt",
        "#sharedVocabulary",
        "[data-chunk-preset]",
        ".prompt-template-btn",
        "#diarizationEnabled",
        "#diarizationSpeakerMode"
      ].every((selector) => document.querySelector(selector).disabled)
    })`,
  );
  assert.equal(finalizing.startMessages, 1, "stop/start click sequence must not start a second session");
  assert.equal(finalizing.stopMessages, 1, "graceful stop should send one stop message");
  assert.equal(finalizing.disabled, true, "record action should remain locked during server finalization");
  assert.equal(finalizing.busy, "true", "finalization should remain exposed as busy");
  assert.equal(finalizing.label, "最終処理中...", "finalization should have a distinct action label");
  assert.equal(finalizing.clearDisabled, true, "clear must remain disabled during server finalization");
  assert.equal(finalizing.historyDisabled, "true", "history switching must remain disabled during finalization");
  assert.equal(finalizing.historyDetailRequests, 0, "finalization must not load another history view");
  assert.equal(finalizing.downloadHref, null, "finalization must not expose an unfinished artifact");
  assert.equal(finalizing.downloadDisabled, "true", "finalization exports should remain disabled");
  assert.equal(finalizing.settingsLocked, true, "session settings should remain locked during finalization");

  await new Promise((resolve) => setTimeout(resolve, 120));
  const completed = await evaluate(
    client,
    `({
      startMessages: window.__recordingTest.startMessages,
      disabled: document.querySelector("#startBtn").disabled,
      busy: document.querySelector("#startBtn").getAttribute("aria-busy"),
      label: document.querySelector("#startBtn .record-label").textContent,
      status: document.querySelector("#statusText").textContent,
      clearDisabled: document.querySelector("#clearBtn").disabled,
      historyDisabled: document.querySelector(".history-item-main").getAttribute("aria-disabled"),
      downloadHref: document.querySelector("#dlTxt").getAttribute("href"),
      downloadDisabled: document.querySelector("#dlTxt").getAttribute("aria-disabled"),
      settingsUnlocked: [
        "#language",
        "#audioSource",
        "#chunkSeconds",
        "#prompt",
        "#sharedVocabulary",
        "[data-chunk-preset]",
        ".prompt-template-btn",
        "#diarizationEnabled",
        "#diarizationSpeakerMode"
      ].every((selector) => !document.querySelector(selector).disabled),
      transcript: [...document.querySelectorAll(".log-row .text")].map((node) => node.textContent)
    })`,
  );
  assert.equal(completed.startMessages, 1, "server finalization must finish before another start");
  assert.equal(completed.disabled, false, "record action should unlock after finalized acknowledgement");
  assert.equal(completed.busy, "false", "completed recording should clear busy state");
  assert.equal(completed.label, "録音開始", "completed recording should restore start label");
  assert.equal(completed.status, "録音完了", "normal stop should be distinct from disconnect");
  assert.equal(completed.clearDisabled, false, "clear should unlock after finalization");
  assert.equal(completed.historyDisabled, "false", "history switching should unlock after finalization");
  assert.equal(
    completed.downloadHref,
    "/api/transcript/browser-test-1.txt",
    "finalized runtime session should expose its TXT artifact",
  );
  assert.equal(completed.downloadDisabled, "false", "exports should unlock after finalization");
  assert.equal(completed.settingsUnlocked, true, "session settings should unlock after finalization");
  assert.deepEqual(
    await unloadProtectionState(client),
    { dirty: "true", prevented: true, dispatched: false },
    "completed but unsaved transcript should keep unload protection",
  );
  assert.ok(
    completed.transcript.includes("停止直前の文字起こし"),
    "final segment arriving before finalized acknowledgement should remain in the completed session",
  );
}

async function startSecondRecordingAndRejectStaleMessages(client) {
  await evaluate(
    client,
    `(() => {
      window.confirm = () => true;
      document.querySelector("#startBtn").click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 180));
  await evaluate(
    client,
    `(() => {
      const stale = new Event("message");
      stale.data = JSON.stringify({
        type: "final",
        sessionId: "browser-test-1",
        text: "混入してはいけない旧セッション",
        tsStart: 0,
        tsEnd: 1000,
        seq: 99
      });
      window.__recordingTest.sockets[0].dispatchEvent(stale);
    })()`,
  );
  const result = await evaluate(
    client,
    `({
      startMessages: window.__recordingTest.startMessages,
      pressed: document.querySelector("#startBtn").getAttribute("aria-pressed"),
      transcript: document.querySelector("#log").textContent
    })`,
  );
  assert.equal(result.startMessages, 2, "new recording should start only after previous finalization");
  assert.equal(result.pressed, "true", "second recording should be active");
  assert.doesNotMatch(result.transcript, /混入してはいけない旧セッション/, "stale session messages must be ignored");
}

async function verifyIdleDestructiveActions(client) {
  await evaluate(
    client,
    `(() => {
      window.confirm = () => false;
      document.querySelector("#clearBtn").click();
    })()`,
  );
  let result = await evaluate(
    client,
    `({
      transcript: document.querySelector("#log").textContent,
      historyDetailRequests: window.__recordingTest.historyDetailRequests,
      downloadHref: document.querySelector("#dlTxt").getAttribute("href")
    })`,
  );
  assert.match(result.transcript, /停止直前の文字起こし/, "cancelled clear must preserve completed transcript");
  assert.equal(result.historyDetailRequests, 0, "cancelled clear should not affect history state");
  assert.deepEqual(
    await unloadProtectionState(client),
    { dirty: "true", prevented: true, dispatched: false },
    "cancelling a destructive clear should retain unload protection",
  );

  await evaluate(
    client,
    `(() => {
      window.confirm = () => true;
      document.querySelector(".history-item-main").click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 20));
  result = await evaluate(
    client,
    `({
      transcript: document.querySelector("#log").textContent,
      historyDetailRequests: window.__recordingTest.historyDetailRequests,
      downloadHref: document.querySelector("#dlTxt").getAttribute("href")
    })`,
  );
  assert.equal(result.historyDetailRequests, 1, "history should load only after recording finalization");
  assert.match(result.transcript, /履歴の文字起こし/, "unlocked history action should render the selected history");
  assert.equal(result.downloadHref, "/api/history/history-1/download.txt", "selected history should expose its artifact");
  assert.deepEqual(
    await unloadProtectionState(client),
    { dirty: "false", prevented: false, dispatched: true },
    "saved history view should remove unload protection",
  );
}

async function verifySummaryCanBeCancelled(client) {
  await evaluate(
    client,
    `(() => {
      const originalFetch = window.fetch;
      window.__restoreSummaryFetch = () => {
        window.fetch = originalFetch;
      };
      window.fetch = (input, options = {}) => {
        if (!String(input).includes("/api/summarize")) {
          return originalFetch(input, options);
        }
        return new Promise((_resolve, reject) => {
          options.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          }, { once: true });
        });
      };
      document.querySelector("#summaryBtn").click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 20));
  let result = await evaluate(
    client,
    `({
      label: document.querySelector("#summaryBtnLabel").textContent,
      busy: document.querySelector("#summaryBtn").getAttribute("aria-busy"),
      disabled: document.querySelector("#summaryBtn").disabled
    })`,
  );
  assert.deepEqual(
    result,
    { label: "キャンセル", busy: "true", disabled: false },
    "hung summary should expose a usable cancel action",
  );

  await evaluate(client, `document.querySelector("#summaryBtn").click()`);
  await new Promise((resolve) => setTimeout(resolve, 20));
  result = await evaluate(
    client,
    `({
      label: document.querySelector("#summaryBtnLabel").textContent,
      busy: document.querySelector("#summaryBtn").getAttribute("aria-busy"),
      disabled: document.querySelector("#summaryBtn").disabled,
      status: document.querySelector("#statusText").textContent,
      toast: document.querySelector("#toastContainer .toast:last-child")?.textContent || ""
    })`,
  );
  await evaluate(client, `window.__restoreSummaryFetch()`);
  assert.equal(result.label, "生成", "cancelled summary should restore its action label");
  assert.equal(result.busy, "false", "cancelled summary should clear busy state");
  assert.equal(result.disabled, false, "cancelled summary should remain usable");
  assert.equal(result.status, "要約キャンセル", "cancelled summary should leave processing state");
  assert.match(result.toast, /要約をキャンセルしました/, "cancellation should be explained to the user");

  await evaluate(
    client,
    `(() => {
      const originalFetch = window.fetch;
      const originalSetTimeout = window.setTimeout;
      window.__restoreSummaryTimeoutTest = () => {
        window.fetch = originalFetch;
        window.setTimeout = originalSetTimeout;
      };
      window.setTimeout = (callback, delay, ...args) =>
        originalSetTimeout(callback, delay === 120000 ? 10 : delay, ...args);
      window.fetch = (input, options = {}) => {
        if (!String(input).includes("/api/summarize")) {
          return originalFetch(input, options);
        }
        return new Promise((_resolve, reject) => {
          options.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          }, { once: true });
        });
      };
      document.querySelector("#summaryBtn").click();
    })()`,
  );
  await new Promise((resolve) => setTimeout(resolve, 40));
  result = await evaluate(
    client,
    `({
      label: document.querySelector("#summaryBtnLabel").textContent,
      busy: document.querySelector("#summaryBtn").getAttribute("aria-busy"),
      disabled: document.querySelector("#summaryBtn").disabled,
      status: document.querySelector("#statusText").textContent,
      toast: document.querySelector("#toastContainer .toast:last-child")?.textContent || ""
    })`,
  );
  await evaluate(client, `window.__restoreSummaryTimeoutTest()`);
  assert.equal(result.label, "生成", "timed-out summary should restore its action label");
  assert.equal(result.busy, "false", "timed-out summary should clear busy state");
  assert.equal(result.disabled, false, "timed-out summary should remain usable");
  assert.equal(result.status, "要約失敗", "timed-out summary should leave processing state");
  assert.match(result.toast, /要約がタイムアウトしました/, "timeout should be distinguished from cancellation");
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

async function verifyEffectiveAudioSourceFallback(client) {
  const previousInstanceId = await evaluate(client, `window.__recordingTest?.instanceId || ""`);
  await client.send("Page.reload", { ignoreCache: true });
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const ready = await evaluate(
      client,
      `document.documentElement.dataset.whistxReady === "true" &&
       window.__recordingTest?.instanceId &&
       window.__recordingTest.instanceId !== ${JSON.stringify(previousInstanceId)}`,
    );
    if (ready) break;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const unlocked = await evaluate(client, `!document.body.classList.contains("whistx-auth-locked")`);
    if (unlocked) break;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  await evaluate(
    client,
    `(() => {
      const displayWithoutAudio = {
        getTracks() { return []; },
        getAudioTracks() { return []; },
        getVideoTracks() { return []; }
      };
      navigator.mediaDevices.getDisplayMedia = async () => displayWithoutAudio;
      const source = document.querySelector("#audioSource");
      source.value = "both";
      source.dispatchEvent(new Event("change", { bubbles: true }));
      document.querySelector("#startBtn").click();
    })()`,
  );
  for (let attempt = 0; attempt < 50; attempt += 1) {
    const started = await evaluate(
      client,
      `document.querySelector("#startBtn").getAttribute("aria-pressed") === "true"`,
    );
    if (started) break;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  const result = await evaluate(
    client,
    `({
      payload: window.__recordingTest.startPayloads[0],
      pressed: document.querySelector("#startBtn").getAttribute("aria-pressed"),
      telemetry: document.querySelector("#recordTelemetry").textContent,
      status: document.querySelector("#statusText").textContent,
      startMessages: window.__recordingTest.startMessages,
      mediaRequests: window.__recordingTest.mediaRequests,
      locked: document.body.classList.contains("whistx-auth-locked"),
      startDisabled: document.querySelector("#startBtn").disabled,
      audioSourceDisabled: document.querySelector("#audioSource").disabled,
      toast: document.querySelector("#toastContainer").textContent
    })`,
  );
  assert.equal(result.pressed, "true", `fallback recording should still start: ${JSON.stringify(result)}`);
  assert.equal(result.payload.audioSource, "mic", "server payload should use the effective microphone source");
  assert.equal(result.payload.requestedAudioSource, "both", "payload should preserve the requested mixed source");
  assert.equal(
    result.payload.audioSourceFallbackReason,
    "display_audio_not_found",
    "payload should expose why the effective source changed",
  );
  assert.match(result.telemetry, /両方 → マイク/, "UI telemetry should distinguish requested and effective sources");

  await evaluate(client, `document.querySelector("#startBtn").click()`);
  await new Promise((resolve) => setTimeout(resolve, 140));
  await evaluate(
    client,
    `(() => {
      window.confirm = () => true;
      document.querySelector("#clearBtn").click();
    })()`,
  );
  assert.deepEqual(
    await unloadProtectionState(client),
    { dirty: "false", prevented: false, dispatched: true },
    "explicitly discarded transcript should remove unload protection",
  );
}

async function verifyInvalidSessionDoesNotBecomeGuest(client) {
  const previousInstanceId = await evaluate(client, `window.__recordingTest?.instanceId || ""`);
  await client.send("Page.addScriptToEvaluateOnNewDocument", {
    source: String.raw`
      (() => {
        localStorage.setItem("whistx_guest_mode", "1");
        const originalFetch = window.fetch;
        window.fetch = (input, options = {}) => {
          if (String(input).includes("/api/auth/me")) {
            return Promise.resolve(new Response(JSON.stringify({
              authenticated: false,
              sessionInvalid: true,
              guestTranscriptionAllowed: true,
              bootstrapAdminRequired: false,
              selfSignupEnabled: false
            }), {
              status: 200,
              headers: { "Content-Type": "application/json" }
            }));
          }
          return originalFetch(input, options);
        };
      })();
    `,
  });
  await client.send("Page.reload", { ignoreCache: true });
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const ready = await evaluate(
      client,
      `document.documentElement.dataset.whistxReady === "true" &&
       window.__recordingTest?.instanceId &&
       window.__recordingTest.instanceId !== ${JSON.stringify(previousInstanceId)}`,
    );
    if (ready) break;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const ready = await evaluate(
      client,
      `document.body.classList.contains("whistx-auth-locked") &&
       document.querySelector("#toastContainer").textContent.includes("ログインセッションが切れました")`,
    );
    if (ready) break;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  let result = await evaluate(
    client,
    `({
      locked: document.body.classList.contains("whistx-auth-locked"),
      guestMode: localStorage.getItem("whistx_guest_mode"),
      toast: document.querySelector("#toastContainer").textContent
    })`,
  );
  assert.equal(result.locked, true, "invalid session should require login instead of opening the workspace");
  assert.equal(result.guestMode, null, "invalid session should clear stale guest-mode cache");
  assert.match(result.toast, /ログインセッションが切れました/, "invalid session should explain that re-login is required");

  await evaluate(client, `document.querySelector("#guestLoginBtn").click()`);
  result = await evaluate(
    client,
    `({
      locked: document.body.classList.contains("whistx-auth-locked"),
      guestMode: localStorage.getItem("whistx_guest_mode")
    })`,
  );
  assert.deepEqual(
    result,
    { locked: false, guestMode: "1" },
    "guest mode should begin only after the explicit guest action",
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
  await verifyDesktopPanelLayout(client);
  for (const width of [390, 640, 768, 1100]) {
    await verifyHistoryDrawerAtWidth(client, width);
  }
  await verifyRecordingStartIsSingleFlight(client);
  await verifyDestructiveActionsAreLocked(client);
  await verifyTranscriptMediaResponsive(client);
  await verifyTranscriptAutoScroll(client);
  await verifyModalKeyboardManagement(client);
  await verifyGracefulStopIsSerialized(client);
  await verifyIdleDestructiveActions(client);
  await verifySummaryCanBeCancelled(client);
  await startSecondRecordingAndRejectStaleMessages(client);
  await verifySocketLossStopsRecording(client);
  await verifyFailedStartPreservesTranscript(client);
  await verifyEffectiveAudioSourceFallback(client);
  await verifyInvalidSessionDoesNotBecomeGuest(client);
  process.stdout.write("Browser UI and recording lifecycle checks passed.\n");
} finally {
  client?.close();
  if (chromeProcess.exitCode === null) {
    const exited = new Promise((resolve) => chromeProcess.once("exit", resolve));
    chromeProcess.kill("SIGTERM");
    await exited;
  }
  await new Promise((resolve) => server.close(resolve));
  await rm(profileDir, { recursive: true, force: true, maxRetries: 5, retryDelay: 50 });
}
