import assert from "node:assert/strict";
import { test } from "node:test";

import { wheelDelta } from "./input.ts";

test("wheels are read as notches, trackpads as pixels", () => {
  const event = (deltaMode: number, deltaY: number, wheelDeltaY?: number) =>
    wheelDelta({ deltaMode, deltaY, wheelDeltaY });
  // Firefox: lines, 3 a notch, and pages.
  assert.deepEqual(event(1, 3), [1, 1]);
  assert.deepEqual(event(1, -6), [1, -2]);
  assert.deepEqual(event(2, 1), [1, 1]);
  // Safari: 4.000244140625 px a notch, and whole multiples.
  assert.deepEqual(event(0, -4.000244140625), [1, -1]);
  assert.deepEqual(event(0, 8.00048828125), [1, 2]);
  // Chromium: a detent's legacy wheelDeltaY is 120, a trackpad's -3 ×
  // deltaY.
  assert.deepEqual(event(0, 100, -120), [1, 1]);
  assert.deepEqual(event(0, -2, 6), [0, -2]);
  // Playwright's synthetic wheel in headless Chromium at 2x.
  assert.deepEqual(event(0, 50, -60), [0, 50]);
  // No legacy delta (Firefox's pixels): continuous.
  assert.deepEqual(event(0, 12), [0, 12]);
});
