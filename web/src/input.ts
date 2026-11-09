// DOM input to the map's gestures (orb-camera's input): pointer events for
// mouse, pen and touch, wheel, keys and focus. Each call carries the
// event's timestamp, which release velocities and double taps measure, on
// the clock frames' timestamps use. A Shift-drag's box shows over the
// canvas while it is drawn.

import type { Core } from "./hosts.ts";

const modifiers = (e: KeyboardEvent | MouseEvent) =>
  (e.shiftKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.altKey ? 4 : 0) | (e.metaKey ? 8 : 0);

/** Feeds `canvas`'s input to the map `core()` gives, while it gives one
 * (not while the map recovers its GPU); `changed` asks for a frame.
 * Returns a detach function. */
export function attachInput(canvas: HTMLCanvasElement, core: () => Core | undefined, changed: () => void): () => void {
  canvas.tabIndex = 0;
  canvas.style.touchAction = "none";
  const box = document.createElement("div");
  Object.assign(box.style, {
    position: "absolute", display: "none", pointerEvents: "none", boxSizing: "border-box",
    border: "2px dotted #202020", background: "rgba(255, 255, 255, 0.5)",
  });
  canvas.after(box);
  // Shows the box a Shift-drag zooms to, if any, over the canvas.
  const showBox = (ui: Core | undefined) => {
    const corners = ui?.boxZoom();
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
    const ui = core();
    if (ui?.pointerDown(e.pointerId, e.pointerType, e.button, x, y, modifiers(e), e.timeStamp)) {
      canvas.setPointerCapture(e.pointerId);
      captured++;
      changed();
    }
  };
  const move = (e: PointerEvent) => {
    const [x, y] = at(e);
    const ui = core();
    if (ui?.pointerMove(e.pointerId, x, y, e.timeStamp)) {
      showBox(ui);
      changed();
    }
  };
  const up = (e: PointerEvent) => {
    const [x, y] = at(e);
    const ui = core();
    if (ui?.pointerUp(e.pointerId, x, y, e.timeStamp)) changed();
    showBox(ui);
  };
  const lost = () => {
    captured = Math.max(0, captured - 1);
    origin = null;
  };
  const cancel = (e: PointerEvent) => {
    const ui = core();
    if (ui?.pointerCancel(e.pointerId, e.timeStamp)) changed();
    showBox(ui);
  };
  const wheel = (e: WheelEvent) => {
    if (core()?.wheel(e.offsetX, e.offsetY, e.deltaX, e.deltaY, e.deltaMode, modifiers(e), e.timeStamp)) {
      e.preventDefault();
      changed();
    }
  };
  const key = (e: KeyboardEvent) => {
    const ui = core();
    if (ui?.key(e.key, modifiers(e), e.timeStamp)) {
      e.preventDefault();
      showBox(ui);
      changed();
    }
  };
  const blur = (e: FocusEvent) => {
    const ui = core();
    ui?.blur(e.timeStamp);
    showBox(ui);
    changed();
  };
  const menu = (e: Event) => e.preventDefault();
  const listeners: [string, EventListener, AddEventListenerOptions?][] = [
    ["pointerdown", down as EventListener],
    ["pointermove", move as EventListener],
    ["pointerup", up as EventListener],
    ["pointercancel", cancel as EventListener],
    ["lostpointercapture", lost],
    ["wheel", wheel as EventListener, { passive: false }],
    ["keydown", key as EventListener],
    ["blur", blur as EventListener],
    ["contextmenu", menu],
  ];
  for (const [type, listener, options] of listeners) canvas.addEventListener(type, listener, options);
  return () => {
    for (const [type, listener, options] of listeners) canvas.removeEventListener(type, listener, options);
    box.remove();
  };
}
