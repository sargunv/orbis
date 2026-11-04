import assert from "node:assert/strict";
import { test } from "node:test";

// A deterministic animation clock shared by both browser host modes.
const callbacks = [];
globalThis.requestAnimationFrame = (callback) => callbacks.push(callback);
const { frameLoop } = await import("./frames.js");

function harness(render) {
  callbacks.length = 0;
  const reports = [];
  const frames = frameLoop((ack) => reports.push(ack), (e) => { throw e; });
  frames.start({ render });
  return { frames, reports, tick() { assert.ok(callbacks.length, "no frame scheduled"); callbacks.shift()(); } };
}
const info = (idle, tiles = 0) => ({ idle, tiles_drawn: tiles, free() {} });

test("a loading map keeps drawing until an arriving tile is published, without camera input", () => {
  let arrived = false;
  const h = harness(() => info(arrived, arrived ? 1 : 0));
  h.tick();
  assert.equal(h.reports[0].idle, false);
  // The host is still loading. Its next frame must already be scheduled.
  arrived = true;
  h.tick();
  assert.equal(h.reports[1].tilesDrawn, 1);
  assert.equal(h.reports[1].idle, true);
  assert.equal(callbacks.length, 0);
});

test("a result wakes an idle map and repeated wakes coalesce", () => {
  const h = harness(() => info(true, 1));
  h.tick();
  h.frames.request();
  h.frames.request();
  assert.equal(callbacks.length, 1);
  h.tick();
  assert.equal(h.reports.length, 2);
  assert.equal(callbacks.length, 0);
});

test("a synchronous result wake during render is preserved", () => {
  let count = 0;
  const h = harness(() => {
    if (count++ === 0) h.frames.request();
    return info(true, count);
  });
  h.tick();
  h.tick();
  assert.equal(h.reports.length, 2);
  assert.equal(callbacks.length, 0);
});

test("workers without requestAnimationFrame keep loading frames running on the timer", { timeout: 1000 }, async () => {
  const raf = globalThis.requestAnimationFrame;
  delete globalThis.requestAnimationFrame;
  const { frameLoop: workerLoop } = await import("./frames.js?timer");
  globalThis.requestAnimationFrame = raf;
  let renders = 0;
  await new Promise((resolve, reject) => {
    const frames = workerLoop((ack) => {
      if (ack.idle) resolve();
    }, reject);
    frames.start({ render: () => info(++renders === 2, renders) });
  });
  assert.equal(renders, 2);
});
