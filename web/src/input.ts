// DOM input to the map's gestures (orb-camera's input): pointer events for
// mouse, pen and touch, wheel, keys and focus, sent to the map wherever it
// runs. Each carries the event's timestamp, which release velocities and
// double taps measure, on the clock frames' timestamps use. A Shift-drag's
// box shows over the canvas while it is drawn.

import type { InputKind } from "./engine.ts";

const modifiers = (e: KeyboardEvent | MouseEvent) =>
  (e.shiftKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.altKey ? 4 : 0) | (e.metaKey ? 8 : 0);

/** The keys the map uses (orb-web's `key`); Escape only ends a box zoom. */
const KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "_"]);

/** Where input goes: `send(kind, args, time)` while the map takes it (not
 * while it recovers its GPU). */
export type InputSink = (kind: InputKind, args: (number | string)[], time: number) => void;

/** Feeds `canvas`'s input to `send`. Moves arrive as `pointerrawupdate`s
 * with `raw` (as soon as the browser has them, for a map in a worker:
 * `pointermove` waits for the page's next frame, and the worker's would
 * already have begun). Returns the detach function and `showBox`, which
 * shows a Shift-drag's box ([x0, y0, x1, y1], or null to hide it). */
export function attachInput(canvas: HTMLCanvasElement, send: InputSink, raw: boolean) {
  canvas.tabIndex = 0;
  canvas.style.touchAction = "none";
  const box = document.createElement("div");
  Object.assign(box.style, {
    position: "absolute", display: "none", pointerEvents: "none", boxSizing: "border-box",
    border: "2px dotted #202020", background: "rgba(255, 255, 255, 0.5)",
  });
  canvas.after(box);
  const showBox = (corners: number[] | null) => {
    if (!corners) {
      box.style.display = "none";
      return;
    }
    const [x0, y0, x1, y1] = corners;
    Object.assign(box.style, {
      display: "block",
      left: `${canvas.offsetLeft + Math.min(x0, x1)}px`, top: `${canvas.offsetTop + Math.min(y0, y1)}px`,
      width: `${Math.abs(x1 - x0)}px`, height: `${Math.abs(y1 - y0)}px`,
    });
  };
  // Where the canvas is while pointers are captured: `offsetX` would lay
  // the page out again on every move once overlays have moved.
  let origin: DOMRect | null = null;
  let captured = 0;
  const at = (e: PointerEvent): [number, number] => {
    if (!captured) return [e.offsetX, e.offsetY];
    origin ??= canvas.getBoundingClientRect();
    return [e.clientX - origin.left, e.clientY - origin.top];
  };
  const down = (e: PointerEvent) => {
    canvas.focus({ preventScroll: true });
    origin = null;
    const [x, y] = at(e);
    send("pointerDown", [e.pointerId, e.pointerType, e.button, x, y, modifiers(e)], e.timeStamp);
    // The map drags with the primary and secondary buttons.
    if (e.button === 0 || e.button === 2) {
      canvas.setPointerCapture(e.pointerId);
      captured++;
    }
  };
  const move = (e: PointerEvent) => {
    const [x, y] = at(e);
    send("pointerMove", [e.pointerId, x, y], e.timeStamp);
  };
  const up = (e: PointerEvent) => {
    const [x, y] = at(e);
    send("pointerUp", [e.pointerId, x, y], e.timeStamp);
  };
  const lost = () => {
    captured = Math.max(0, captured - 1);
    origin = null;
  };
  const cancel = (e: PointerEvent) => send("pointerCancel", [e.pointerId], e.timeStamp);
  // The map zooms on every wheel event and uses its keys.
  const wheel = (e: WheelEvent) => {
    e.preventDefault();
    send("wheel", [e.offsetX, e.offsetY, e.deltaX, e.deltaY, e.deltaMode, modifiers(e)], e.timeStamp);
  };
  const key = (e: KeyboardEvent) => {
    if (KEYS.has(e.key)) e.preventDefault();
    if (KEYS.has(e.key) || e.key === "Escape") send("key", [e.key, modifiers(e)], e.timeStamp);
  };
  const blur = (e: FocusEvent) => send("blur", [], e.timeStamp);
  const menu = (e: Event) => e.preventDefault();
  const listeners: [string, EventListener, AddEventListenerOptions?][] = [
    ["pointerdown", down as EventListener],
    [raw ? "pointerrawupdate" : "pointermove", move as EventListener],
    ["pointerup", up as EventListener],
    ["pointercancel", cancel as EventListener],
    ["lostpointercapture", lost],
    ["wheel", wheel as EventListener, { passive: false }],
    ["keydown", key as EventListener],
    ["blur", blur as EventListener],
    ["contextmenu", menu],
  ];
  for (const [type, listener, options] of listeners) canvas.addEventListener(type, listener, options);
  return {
    showBox,
    detach: () => {
      for (const [type, listener, options] of listeners) canvas.removeEventListener(type, listener, options);
      box.remove();
    },
  };
}
