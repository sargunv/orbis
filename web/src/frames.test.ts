import assert from "node:assert/strict";
import { test } from "node:test";

// A deterministic animation clock.
const callbacks: ((now: number) => void)[] = [];
globalThis.requestAnimationFrame = ((callback: (now: number) => void) =>
  callbacks.push(callback)) as typeof requestAnimationFrame;
const { frameLoop } = await import("./frames.ts");

const tick = () => {
  assert.ok(callbacks.length, "no frame scheduled");
  callbacks.shift()!(0);
};

test("frames run only while asked for: a loading map keeps drawing, an idle one stops", () => {
  callbacks.length = 0;
  let loading = 2;
  let frames = 0;
  const loop = frameLoop(
    () => {
      frames++;
      return --loading > 0;
    },
    (e) => {
      throw e;
    },
  );
  assert.equal(callbacks.length, 0, "nothing until requested");
  loop.request();
  loop.request();
  assert.equal(callbacks.length, 1, "requests coalesce into one frame");
  tick();
  tick();
  assert.equal(frames, 2);
  assert.equal(callbacks.length, 0, "idle: no more frames");
});

test("a request during a frame schedules the next", () => {
  callbacks.length = 0;
  let again = true;
  const loop = frameLoop(
    () => {
      if (again) {
        again = false;
        loop.request();
      }
      return false;
    },
    (e) => {
      throw e;
    },
  );
  loop.request();
  tick();
  assert.equal(callbacks.length, 1);
  tick();
  assert.equal(callbacks.length, 0);
});

test("errors reach the handler and stop the loop", () => {
  callbacks.length = 0;
  const errors: unknown[] = [];
  const loop = frameLoop(
    () => {
      throw new Error("boom");
    },
    (e) => errors.push(e),
  );
  loop.request();
  tick();
  assert.equal(errors.length, 1);
  assert.equal(callbacks.length, 0);
});
