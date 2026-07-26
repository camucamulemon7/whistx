import test from "node:test";
import assert from "node:assert/strict";

import { fetchJson } from "../../web/src/api/client.js";

test("fetchJson aborts a hung request at the configured timeout", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  let forwardedOptions;
  globalThis.fetch = (_url, options) => {
    forwardedOptions = options;
    return new Promise((_resolve, reject) => {
      options.signal.addEventListener(
        "abort",
        () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })),
        { once: true },
      );
    });
  };

  await assert.rejects(
    fetchJson("/hung", { timeoutMs: 10 }),
    (error) => error.name === "TimeoutError" && error.code === "timeout" && error.message === "request_timeout",
  );
  assert.equal("timeoutMs" in forwardedOptions, false, "internal timeout option must not leak into fetch");
});

test("fetchJson distinguishes caller cancellation from timeout", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = (_url, options) =>
    new Promise((_resolve, reject) => {
      options.signal.addEventListener(
        "abort",
        () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })),
        { once: true },
      );
    });

  const controller = new AbortController();
  const request = fetchJson("/cancelled", { signal: controller.signal, timeoutMs: 1000 });
  controller.abort("user_cancelled");
  await assert.rejects(
    request,
    (error) => error.name === "AbortError" && error.code === "aborted" && error.message === "request_cancelled",
  );
});
