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
        document.body.classList.contains("whistx-auth-locked")
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
      authInitialized: document.body.classList.contains("whistx-auth-locked")
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
  process.stdout.write("Browser UI checks passed at 390px, 640px, and 1100px.\n");
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
