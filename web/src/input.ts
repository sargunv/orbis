// DOM input to the map's gestures (orb-camera's input): pointer events for
// mouse, pen and touch, wheel, keys and focus, sent to the map wherever it
// runs. Each carries the event's timestamp, which release velocities and
// double taps measure, on the clock frames' timestamps use. A Shift-drag's
// box shows over the canvas while it is drawn.

import type { InputKind } from "./engine.ts";
import type { GestureOptions } from "./types.ts";

/** Which movements the map's gestures make: the wheel and keys of those
 * that don't scroll the page as they would without the map. */
export interface Allowed {
  pan: boolean;
  zoom: boolean;
  rotate: boolean;
  pitch: boolean;
}

/** What `options` allows. */
export const allowed = (options: GestureOptions = {}): Allowed => {
  const on = (m: GestureOptions["pan"]) =>
    typeof m === "boolean" ? m : (m?.enabled ?? true);
  return {
    pan: on(options.pan),
    zoom: on(options.zoom),
    rotate: on(options.rotate),
    pitch: on(options.pitch),
  };
};

const modifiers = (e: KeyboardEvent | MouseEvent) =>
  (e.shiftKey ? 1 : 0) |
  (e.ctrlKey ? 2 : 0) |
  (e.altKey ? 4 : 0) |
  (e.metaKey ? 8 : 0);

/** Whether the map uses a key (orb-web's `key`) with what it allows;
 * Escape only ends a box zoom. */
const usesKey = (e: KeyboardEvent, allowed: Allowed) => {
  switch (e.key) {
    case "ArrowLeft":
    case "ArrowRight":
      return e.shiftKey ? allowed.rotate : allowed.pan;
    case "ArrowUp":
    case "ArrowDown":
      return e.shiftKey ? allowed.pitch : allowed.pan;
    case "+":
    case "=":
    case "-":
    case "_":
      return allowed.zoom;
    default:
      return false;
  }
};

/** Where input goes: `send(kind, args, time)` while the map takes it (not
 * while it recovers its GPU). */
export type InputSink = (
  kind: InputKind,
  args: (number | string)[],
  time: number,
) => void;

/** Feeds `canvas`'s input to `send`. Moves arrive as `pointerrawupdate`s
 * with `raw` (as soon as the browser has them, for a map in a worker:
 * `pointermove` waits for the page's next frame, and the worker's would
 * already have begun). Returns the detach function, `showBox`, which
 * shows a Shift-drag's box ([x0, y0, x1, y1], or null to hide it), and
 * `allow`, which says what the map's gestures move now. */
export function attachInput(
  canvas: HTMLCanvasElement,
  send: InputSink,
  raw: boolean,
) {
  let allowedNow = allowed();
  canvas.tabIndex = 0;
  canvas.style.touchAction = "none";
  const box = document.createElement("div");
  Object.assign(box.style, {
    position: "absolute",
    display: "none",
    pointerEvents: "none",
    boxSizing: "border-box",
    border: "2px dotted #202020",
    background: "rgba(255, 255, 255, 0.5)",
  });
  canvas.after(box);
  const showBox = (corners: number[] | null) => {
    const [x0, y0, x1, y1] = corners ?? [];
    if (
      x0 === undefined ||
      y0 === undefined ||
      x1 === undefined ||
      y1 === undefined
    ) {
      box.style.display = "none";
      return;
    }
    Object.assign(box.style, {
      display: "block",
      left: `${canvas.offsetLeft + Math.min(x0, x1)}px`,
      top: `${canvas.offsetTop + Math.min(y0, y1)}px`,
      width: `${Math.abs(x1 - x0)}px`,
      height: `${Math.abs(y1 - y0)}px`,
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
    send(
      "pointerDown",
      [e.pointerId, e.pointerType, e.button, x, y, modifiers(e)],
      e.timeStamp,
    );
    // The map drags with the primary and secondary buttons.
    if (e.button === 0 || e.button === 2) {
      canvas.setPointerCapture(e.pointerId);
      captured++;
    }
  };
  // A drag's every sample with its own time, for the release's velocity:
  // `pointermove` comes once a frame with the samples since folded in, and
  // `pointerrawupdate` can fold too. Hovers send the latest alone.
  const move = (e: PointerEvent) => {
    const samples = (captured && e.getCoalescedEvents?.()) || [];
    for (const s of samples.length ? samples : [e]) {
      const [x, y] = at(s);
      send("pointerMove", [e.pointerId, x, y], s.timeStamp);
    }
  };
  const up = (e: PointerEvent) => {
    const [x, y] = at(e);
    send("pointerUp", [e.pointerId, x, y], e.timeStamp);
  };
  const lost = () => {
    captured = Math.max(0, captured - 1);
    origin = null;
  };
  const cancel = (e: PointerEvent) =>
    send("pointerCancel", [e.pointerId], e.timeStamp);
  // The map zooms on every wheel event, unless it doesn't zoom, and uses
  // its keys.
  const wheel = (e: WheelEvent) => {
    if (!allowedNow.zoom) return;
    e.preventDefault();
    send(
      "wheel",
      [e.offsetX, e.offsetY, e.deltaX, e.deltaY, e.deltaMode, modifiers(e)],
      e.timeStamp,
    );
  };
  const key = (e: KeyboardEvent) => {
    const uses = usesKey(e, allowedNow);
    if (uses) e.preventDefault();
    if (uses || e.key === "Escape")
      send("key", [e.key, modifiers(e)], e.timeStamp);
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
  for (const [type, listener, options] of listeners)
    canvas.addEventListener(type, listener, options);
  return {
    showBox,
    allow: (next: Allowed) => {
      allowedNow = next;
    },
    detach: () => {
      for (const [type, listener, options] of listeners)
        canvas.removeEventListener(type, listener, options);
      box.remove();
    },
  };
}
