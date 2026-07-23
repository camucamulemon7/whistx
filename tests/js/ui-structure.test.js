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

test("guided journey and responsive design tokens are present", async () => {
  const [html, css, app] = await Promise.all([
    readFile(new URL("web/index.html", root), "utf8"),
    readFile(new URL("web/style.css", root), "utf8"),
    readFile(new URL("web/src/app.js", root), "utf8"),
  ]);
  assert.equal((html.match(/data-journey-step=/g) || []).length, 3);
  assert.match(css, /--surface-canvas:/);
  assert.match(css, /@media \(min-width: 1440px\)/);
  assert.match(css, /@media \(max-width: 640px\)/);
  assert.match(app, /function syncJourneyStage/);
  assert.match(app, /setAttribute\("aria-current", "step"\)/);
});
