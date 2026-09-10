import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const root = new URL("../../", import.meta.url);

test("workspace exposes a keyboard skip target and labelled live transcript", async () => {
  const html = await readFile(new URL("web/index.html", root), "utf8");
  assert.match(html, /class="skip-link" href="#workspaceMain"/);
  assert.match(html, /<main id="workspaceMain"[^>]*tabindex="-1"/);
  assert.match(html, /id="log"[^>]*role="log"[^>]*tabindex="0"/);
  assert.match(html, /aria-labelledby="transcriptPanelTitle"/);
});

test("mobile history drawer has an accessible trigger and backdrop", async () => {
  const [html, css, app] = await Promise.all([
    readFile(new URL("web/index.html", root), "utf8"),
    readFile(new URL("web/style.css", root), "utf8"),
    readFile(new URL("web/src/app.js", root), "utf8"),
  ]);

  assert.match(html, /id="historyDrawerOpen"[^>]*aria-controls="historyRail"[^>]*aria-expanded="false"/s);
  assert.match(html, /id="historyRail"[^>]*tabindex="-1"/);
  assert.match(html, /id="historyDrawerBackdrop"[^>]*hidden/);
  assert.match(css, /\.history-drawer-backdrop/);
  assert.match(app, /historyDrawerOpenEl\.setAttribute\("aria-expanded"/);
  assert.match(app, /historyDrawerBackdropEl\.addEventListener\("click"/);
});

test("runtime UI styles do not override the static history rail layout", async () => {
  const [html, css, runtimeCss] = await Promise.all([
    readFile(new URL("web/index.html", root), "utf8"),
    readFile(new URL("web/style.css", root), "utf8"),
    readFile(new URL("web/runtime-ui.css", root), "utf8"),
  ]);

  assert.match(css, /\.history-rail\s*\{[^}]*position:\s*sticky/s);
  assert.doesNotMatch(runtimeCss, /(?:\.history-rail|#history(?:SearchInput|List|Empty))/);
  assert.match(html, /runtime-ui\.css\?v=20260728/);
});

test("workspace uses a direct task hierarchy and responsive design tokens", async () => {
  const [html, css] = await Promise.all([
    readFile(new URL("web/index.html", root), "utf8"),
    readFile(new URL("web/style.css", root), "utf8"),
  ]);
  assert.doesNotMatch(html, /data-journey-step=/);
  assert.match(html, /class="settings-intro-title">録音</);
  assert.match(html, /id="transcriptPanelTitle"[^>]*>文字起こし</);
  assert.match(html, /id="proofreadBtnLabel">校正する</);
  assert.match(html, /id="summaryBtnLabel">要約する</);
  assert.match(css, /--surface-canvas:/);
  assert.match(css, /@media \(min-width: 1440px\)/);
  assert.match(css, /@media \(max-width: 640px\)/);
});
