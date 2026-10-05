import assert from "node:assert/strict";
import test from "node:test";
import { createNotesExport } from "../../web/src/meeting/notes.js";

function fixture(request) {
  const nodes = Object.fromEntries(["notesToken", "notesTitle", "notesSave", "notesStatus"].map(id => [id, {
    value: id === "notesToken" ? "synthetic-existing-token" : "Synthetic meeting", textContent: "", disabled: false,
    setAttribute() {}, addEventListener() {}, focus() {},
  }]));
  globalThis.window = { addEventListener() {} };
  let access = "ready";
  const controller = createNotesExport({ getSource: () => ({ runtimeSessionId: "synthetic" }), getAccess: () => access,
    document: { querySelector: selector => nodes[selector.slice(1)] }, request });
  return { controller, nodes, setAccess(value) { access = value; controller.sync(); } };
}

test("Notes sends only on explicit save, blocks double clicks and clears credentials", async () => {
  const calls = [];
  let finish;
  const done = new Promise(resolve => { finish = resolve; });
  const { controller, nodes } = fixture(async (...args) => { calls.push(args); return done; });
  assert.equal(calls.length, 0);
  const first = controller.save();
  await controller.save();
  assert.equal(calls.length, 1);
  assert.equal(nodes.notesSave.disabled, true);
  assert.equal(JSON.parse(calls[0][1].body).token, "synthetic-existing-token");
  finish({ ok: true, alreadySaved: false });
  await first;
  assert.equal(nodes.notesToken.value, "");
  assert.match(nodes.notesStatus.textContent, /非公開Note/);
});

test("Notes 401, timeout and failed save allow explicit retry without automatic transmission", async () => {
  for (const error of [{ status: 401, message: "notes_auth_required" }, { code: "timeout" }, { message: "notes_save_rejected" }]) {
    let calls = 0;
    const { controller, nodes } = fixture(async () => { calls += 1; throw error; });
    await controller.save();
    assert.equal(calls, 1);
    assert.equal(nodes.notesToken.value, "");
    assert.equal(nodes.notesSave.disabled, false);
    assert.ok(nodes.notesStatus.textContent);
    nodes.notesToken.value = "synthetic-retry-token";
    await controller.save();
    assert.equal(calls, 2);
  }
});

test("Notes logout clears credential and blocks guest export; source switch ignores old response", async () => {
  let calls = 0, finish;
  const { controller, nodes, setAccess } = fixture(async () => { calls += 1; return new Promise(resolve => { finish = resolve; }); });
  setAccess("login");
  assert.equal(nodes.notesToken.value, "");
  await controller.save();
  assert.equal(calls, 0);
  setAccess("ready");
  nodes.notesToken.value = "synthetic";
  const pending = controller.save();
  controller.reset();
  finish({ ok: true });
  await pending;
  assert.equal(nodes.notesStatus.textContent, "");
  assert.equal(nodes.notesToken.value, "");
});
