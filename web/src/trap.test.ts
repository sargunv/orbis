import assert from "node:assert/strict";
import { test } from "node:test";

import { DeferredLoadError, guard, isTrap, trapError } from "./trap.ts";

/** A stand-in for the wasm map: its own borrow, as wasm-bindgen's. */
class Fake {
  borrowed = false;
  calls = 0;
  frame = new Promise<string>(() => {});
  #enter(): void {
    if (this.borrowed) throw new Error("recursive use of an object");
    this.calls++;
  }
  camera(): number {
    this.#enter();
    return 1;
  }
  render(): void {
    this.#enter();
    // A panic traps with the borrow still held.
    this.borrowed = true;
    throw new WebAssembly.RuntimeError("unreachable");
  }
  readNextFrame(): Promise<string> {
    this.#enter();
    return this.frame;
  }
  query(): void {
    this.#enter();
    throw new Error("no such layer");
  }
}

test("a trap stops the map with one error, which every call after throws without entering it", async () => {
  const fake = new Fake();
  const heard: Error[] = [];
  const { core } = guard(fake, (e) => heard.push(e));
  const quiet = console.error;
  console.error = () => {};
  try {
    assert.equal(core.camera(), 1);
    assert.throws(() => core.query(), /no such layer/, "other errors pass");
    const waiting = core.readNextFrame();
    let error: unknown;
    try {
      core.render();
    } catch (e) {
      error = e;
    }
    assert.ok(error instanceof Error);
    assert.match(
      error.message,
      /^the map stopped: its wasm trapped \(unreachable\)/,
    );
    assert.ok(isTrap(error));
    assert.deepEqual(heard, [error], "heard once");
    const calls = fake.calls;
    assert.throws(
      () => core.camera(),
      (e) => e === error,
    );
    assert.throws(
      () => core.render(),
      (e) => e === error,
    );
    assert.equal(fake.calls, calls, "no call reaches the trapped map");
    await assert.rejects(waiting, (e) => e === error, "a waiting call ends");
    assert.equal(heard.length, 1);
  } finally {
    console.error = quiet;
  }
});

test("a trap outside a call stops the map too", () => {
  const heard: Error[] = [];
  const { core, stop } = guard(new Fake(), (e) => heard.push(e));
  const quiet = console.error;
  console.error = () => {};
  try {
    const error = stop(new WebAssembly.RuntimeError("unreachable"));
    assert.equal(stop(new WebAssembly.RuntimeError("again")), error);
    assert.throws(
      () => core.camera(),
      (e) => e === error,
    );
    assert.deepEqual(heard, [error]);
  } finally {
    console.error = quiet;
  }
});

test("a deferred half that didn't load stops the map as a trap, not a bug", () => {
  const url = new URL("http://127.0.0.1/assets/core-deferred-1a2b.wasm");
  const failed = new DeferredLoadError(url, new Error("NetworkError"));
  assert.ok(isTrap(failed));
  assert.equal(
    trapError(failed).message,
    "the map stopped: core-deferred-1a2b.wasm didn't load: NetworkError",
  );
});
