import assert from "node:assert/strict";
import { test } from "node:test";

import { wheelDelta } from "./input.ts";

const PIXELS = 0;
const LINES = 1;
const PAGES = 2;

const read = (
  deltaMode: number,
  deltaY: number,
  wheelDeltaY?: number,
  ctrlKey = false,
) => wheelDelta({ deltaMode, deltaY, wheelDeltaY, ctrlKey });
const notches = (amount: number) => [1, amount];
const pixels = (amount: number) => [0, amount];

test("Firefox on Windows and Linux: a notch is 6 lines and ∓120 ticks", () => {
  assert.deepEqual(read(LINES, 6, -120), notches(1));
  assert.deepEqual(read(LINES, -12, 240), notches(-2));
  assert.deepEqual(read(LINES, 6, -120, true), notches(1));
});

test("Firefox on Windows and Linux: a fraction of a tick is that much of a notch", () => {
  // A high-resolution wheel's 30 of a detent's 120, with and without the
  // doubled lines.
  assert.deepEqual(read(LINES, 1.5, -30), notches(0.25));
  assert.deepEqual(read(LINES, 3, -60), notches(0.5));
  assert.deepEqual(read(LINES, -1, 40), notches(-1 / 3));
});

test("Firefox on macOS: a notch is its lines, with no whole ticks", () => {
  // wheelDeltaY is −round(3 × deltaY in px), with a line of 19 px.
  assert.deepEqual(read(LINES, 1, -57), notches(1));
  assert.deepEqual(read(LINES, -3, 171), notches(-3));
});

test("pages are notches", () => {
  assert.deepEqual(read(PAGES, 1, -120), notches(1));
});

test("Safari and Chromium on macOS: a notch is 4.000244140625 px", () => {
  assert.deepEqual(read(PIXELS, 4.000244140625, -12), notches(1));
  assert.deepEqual(read(PIXELS, -12.000732421875, 36), notches(-3));
  assert.deepEqual(read(PIXELS, 8.00048828125), notches(2));
  assert.deepEqual(read(PIXELS, 4.000244140625, -12, true), notches(1));
});

test("Chromium on macOS: an accelerated notch off that multiple counts its ticks", () => {
  assert.deepEqual(read(PIXELS, 12, -120), notches(1));
});

test("Chromium on Windows and Linux: a notch is ±120 ticks, with or without Ctrl", () => {
  assert.deepEqual(read(PIXELS, 100, -120), notches(1));
  assert.deepEqual(read(PIXELS, -120, 120), notches(-1));
  assert.deepEqual(read(PIXELS, 33.333333, -120), notches(1));
  assert.deepEqual(read(PIXELS, 200, -240), notches(2));
  assert.deepEqual(read(PIXELS, 100, -120, true), notches(1));
});

test("a trackpad pinch is pixels in every browser", () => {
  // Chromium: 100 × ln(scale) an update, with ±120 ticks.
  const step = 100 * Math.log(1.05);
  assert.deepEqual(read(PIXELS, -step, 120, true), pixels(-step));
  assert.deepEqual(read(PIXELS, step, -120, true), pixels(step));
  // Firefox and Safari: −3 × deltaY ticks.
  assert.deepEqual(read(PIXELS, -5, 15, true), pixels(-5));
  assert.deepEqual(read(PIXELS, 40, -120, true), pixels(40));
});

test("precise scrolling is pixels", () => {
  // Chromium on macOS, Firefox and Safari: −3 × deltaY ticks.
  assert.deepEqual(read(PIXELS, 40, -120), pixels(40));
  assert.deepEqual(read(PIXELS, -2, 6), pixels(-2));
  // Chromium on Windows and Linux: about −deltaY.
  assert.deepEqual(read(PIXELS, 37, -37), pixels(37));
  // Playwright's synthetic wheel in headless Chromium at 2x.
  assert.deepEqual(read(PIXELS, 50, -60), pixels(50));
  // No legacy delta: continuous.
  assert.deepEqual(read(PIXELS, 12), pixels(12));
});
