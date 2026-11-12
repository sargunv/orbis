// DOM input to the map's gestures (orb-camera's input): pointer events for
// mouse, pen and touch, wheel, Safari's trackpad pinches, keys and focus,
// sent to the map wherever it runs. Each carries the event's timestamp,
// which release velocities and double taps measure, on the clock frames'
// timestamps use. A Shift-drag's box shows over the canvas while it is
// drawn.

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

/** Safari's wheel notch in pixels, and a faster turn's a whole multiple. */
const SAFARI_NOTCH = 4.000244140625;

/** A wheel event as the gestures take it (orb-web's `wheel`): its notches
 * (`notches` 1), for a mouse wheel's detents, which the map eases, or its
 * pixels (0), for continuous scrolling (a trackpad, a precise wheel),
 * which it applies at once. Positive scrolls down. Browsers don't say
 * which device scrolled, so this reads it from how each reports it, as
 * mln-rs's page does, `deltaMode` first:
 * - Firefox reports wheels in lines, 3 a notch, and pages.
 * - Safari reports a notch as 4.000244140625 px, and a faster turn as a
 *   whole multiple of it (GL JS's test): one notch per multiple.
 * - Chromium's legacy `wheelDeltaY` is 120 a detent, and -3 × `deltaY`
 *   for precise scrolling.
 * A misread zooms the same way by a similar amount, with or without the
 * easing. */
export const wheelDelta = (
  e: Pick<WheelEvent, "deltaMode" | "deltaY"> & { wheelDeltaY?: number },
): [notches: number, amount: number] => {
  if (e.deltaMode === 1) return [1, e.deltaY / 3];
  if (e.deltaMode === 2) return [1, e.deltaY];
  if (e.deltaY !== 0 && e.deltaY % SAFARI_NOTCH === 0)
    return [1, e.deltaY / SAFARI_NOTCH];
  const legacy = e.wheelDeltaY;
  if (
    typeof legacy === "number" &&
    legacy !== 0 &&
    Math.abs(legacy + 3 * e.deltaY) >= 1 &&
    legacy % 120 === 0
  )
    return [1, -legacy / 120];
  return [0, e.deltaY];
};

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
    // Classified before anything reads a delta: Firefox reports a wheel's
    // lines only to a page that asks for `deltaMode` first, and pixels
    // otherwise.
    const [notches, amount] = wheelDelta(e);
    if (amount === 0) return;
    send(
      "wheel",
      [e.offsetX, e.offsetY, notches, amount, modifiers(e)],
      e.timeStamp,
    );
  };
  // Safari's trackpad pinch (`GestureEvent`, which only it has): `scale`
  // and `rotation` (degrees clockwise) since the gesture began, handed on
  // as the change since the last. Any Ctrl-wheel events beside it don't
  // zoom again (orb-camera drops them during a pinch).
  type Gesture = Event & {
    scale: number;
    rotation: number;
    clientX: number;
    clientY: number;
  };
  let pinch = { scale: 1, rotation: 0 };
  const gesture = (phase: number) => (event: Event) => {
    const e = event as Gesture;
    if (!allowedNow.zoom && !allowedNow.rotate) return;
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const [scale, rotation] = phase === 0 ? [1, 0] : [e.scale, e.rotation];
    send(
      "pinch",
      [
        e.clientX - rect.left,
        e.clientY - rect.top,
        scale / pinch.scale,
        rotation - pinch.rotation,
        phase,
      ],
      e.timeStamp,
    );
    pinch = { scale, rotation };
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
    ["gesturestart", gesture(0)],
    ["gesturechange", gesture(1)],
    ["gestureend", gesture(2)],
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
