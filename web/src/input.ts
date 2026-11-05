// DOM input to the ui role's gestures (orb-interact): pointer events for
// mouse, pen and touch, wheel, keys and focus. Each call carries the
// event's timestamp, which inertia and double taps measure.

import type { UiRole } from "../pkg/core.js";

const modifiers = (e: KeyboardEvent | MouseEvent) =>
  (e.shiftKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.altKey ? 4 : 0) | (e.metaKey ? 8 : 0);

/** Feeds `canvas`'s input to `ui`; `changed` asks for a frame. Returns a detach function. */
export function attachInput(canvas: HTMLCanvasElement, ui: UiRole, changed: () => void): () => void {
  canvas.tabIndex = 0;
  canvas.style.touchAction = "none";
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
    if (ui.pointerDown(e.pointerId, e.pointerType, e.button, x, y, modifiers(e), e.timeStamp)) {
      canvas.setPointerCapture(e.pointerId);
      captured++;
      changed();
    }
  };
  const move = (e: PointerEvent) => {
    const [x, y] = at(e);
    if (ui.pointerMove(e.pointerId, x, y, e.timeStamp)) changed();
  };
  const up = (e: PointerEvent) => {
    const [x, y] = at(e);
    if (ui.pointerUp(e.pointerId, x, y, e.timeStamp)) changed();
  };
  const lost = () => {
    captured = Math.max(0, captured - 1);
    origin = null;
  };
  const cancel = (e: PointerEvent) => {
    if (ui.pointerCancel(e.pointerId, e.timeStamp)) changed();
  };
  const wheel = (e: WheelEvent) => {
    if (ui.wheel(e.offsetX, e.offsetY, e.deltaX, e.deltaY, e.deltaMode, modifiers(e), e.timeStamp)) {
      e.preventDefault();
      changed();
    }
  };
  const key = (e: KeyboardEvent) => {
    if (ui.key(e.key, modifiers(e), e.timeStamp)) {
      e.preventDefault();
      changed();
    }
  };
  const blur = (e: FocusEvent) => { ui.blur(e.timeStamp); changed(); };
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
  };
}
