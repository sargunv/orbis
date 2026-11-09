// MapLibre Native, through maplibre-native-ffi's browser build (its WebGL 2
// backend, MapLibre Native's reference renderer), as an engine for the
// comparison (compare.js). It loads only when a side picks it. Its threads
// share memory, which needs a cross-origin isolated page (isolation.js).
//
// The browser module isn't in the `@maplibre/native-ffi` package: the build
// serves `@maplibre/native-ffi-opengl-wasm32`'s `.mjs` and `.wasm` under
// `native/<version>/webgl/` (vite.config.js), and `loadModule` imports it
// from there on first use. Native draws on the page's thread when the page
// services it, and takes its gestures from the page, as in the binding's
// browser example (examples/browser in maplibre-native-ffi), which this
// follows.
import {
  CameraUpdateMode, FrameDemandFlag, GesturePhase, loadModule, MapMode, registerWebGlContext,
  LogSeverity, RenderDriverKind, RenderResult, ResourceErrorReason, ResourceKindFlag, ResourceResponseStatus,
  RuntimeEventFlag, RuntimeEventType, RuntimeHandle, setLogCallback, UrlMatchFlag, WebGlContextKind,
} from "@maplibre/native-ffi";
import moduleUrl, { version } from "virtual:native-module";
import { h } from "./dom.js";

export const label = `MapLibre Native ${version}`;

// One runtime for every map, and its events for each map by id.
let runtime;
const handlers = new Map();
function start() {
  runtime ??= (async () => {
    await loadModule(new URL(moduleUrl, location.href));
    // Native logs everything as an error by default; its warnings and errors
    // also come as map events, which open() reports.
    setLogCallback({ callback: (severity, _event, _code, message) => {
      if (severity === LogSeverity.info) console.debug(`MapLibre Native: ${message}`);
    } });
    const created = await RuntimeHandle.create();
    created.discarding.setEventHandler({
      callback: (events) => { for (const event of events) handlers.get(event.source)?.(event); },
    });
    // A route matches nothing without a URL pattern.
    await created.setResourceProvider({
      routes: [{ kinds: ResourceKindFlag.glyphs, url: "**", flags: UrlMatchFlag.glob }], callback: glyphs,
    });
    return created;
  })();
  runtime.catch(() => { runtime = undefined; });
  return runtime;
}

/** A Native map of `style` in `container`, at `camera` within `constraints`. Resolves once it draws. */
export async function open(container, { style, camera, constraints = {} }) {
  if (!crossOriginIsolated) throw new Error("MapLibre Native needs a cross-origin isolated page");
  const canvas = h("canvas", { class: "native-canvas", tabindex: "0", "aria-label": "MapLibre Native's map",
    style: { display: "block", width: "100%", height: "100%", touchAction: "none" } });
  container.append(canvas);
  const context = canvas.getContext("webgl2", { depth: true, stencil: true, antialias: true });
  if (!context) throw new Error("MapLibre Native needs WebGL 2");
  const rt = await start();

  // The map's scale factor stays fixed for its life.
  const scaleFactor = devicePixelRatio;
  const logicalSize = () => ({ width: Math.max(1, Math.round(canvas.clientWidth)),
    height: Math.max(1, Math.round(canvas.clientHeight)) });
  const sizeCanvas = () => {
    const size = logicalSize();
    canvas.width = Math.round(size.width * scaleFactor);
    canvas.height = Math.round(size.height * scaleFactor);
    return size;
  };

  const map = rt.createMap({
    initialExtent: { ...logicalSize(), scaleFactor },
    mapMode: MapMode.continuous,
    eventMask: RuntimeEventFlag.mapLoadingFailed | RuntimeEventFlag.mapRenderError
      | RuntimeEventFlag.mapCameraIsChanging | RuntimeEventFlag.mapCameraDidChange,
  });
  const moves = new Set();
  handlers.set(map.id, (event) => {
    switch (event.type) {
      case RuntimeEventType.mapCameraIsChanging:
      case RuntimeEventType.mapCameraDidChange:
        for (const listener of moves) listener();
        break;
      case RuntimeEventType.mapLoadingFailed:
      case RuntimeEventType.mapRenderError:
        console.error(`MapLibre Native: ${event.message}`);
        break;
    }
  });
  const removing = new AbortController();
  let session;
  let registration;
  const remove = () => {
    if (removing.signal.aborted) return;
    removing.abort();
    handlers.delete(map.id);
    moves.clear();
    const graphics = session;
    session = undefined;
    // The session goes before its map, and the context's registration after both.
    (async () => {
      await graphics?.release();
      await map.release();
      registration?.close();
    })().catch((e) => console.error(e));
    canvas.remove();
  };
  try {
    await map.ready;
    const { minZoom, maxZoom, minPitch, maxPitch, bounds } = constraints;
    map.discarding.setBounds({
      // Orbis lets pitch reach 85 degrees by default.
      minZoom, maxZoom, minPitch, maxPitch: maxPitch ?? 85,
      ...(bounds ? { bounds: { southwest: { longitude: bounds[0], latitude: bounds[1] },
        northeast: { longitude: bounds[2], latitude: bounds[3] } } } : { unbounded: true }),
    });
    jumpTo(map, camera);
    map.discarding.setStyleJson(absolute(style));

    registration = registerWebGlContext(context);
    const wakes = { frame: () => drain(), driverWork: () => session?.serviceDriverWork() };
    session = map.attachOpenGlSurface(
      { logicalSize: sizeCanvas(),
        context: { data: { kind: "webGl", value: { kind: WebGlContextKind.existing, context: registration.context } } } },
      { driver: RenderDriverKind.callerGraphicsThread, frameWake: { callback: wakes.frame },
        driverWorkWake: { callback: wakes.driverWork } },
    );
    await session.ready;
  } catch (e) {
    remove();
    throw e;
  }
  // Each finished frame asks for the next one on the next animation frame,
  // which waits until the map has something new to draw.
  const demand = () => session?.demandFrame({ flags: FrameDemandFlag.waitForUpdate });
  const drain = () => {
    const results = session?.drainFrameResults() ?? [];
    if (results.length === 0) return;
    if (results.some(({ disposition }) => disposition === RenderResult.failed)) {
      console.error("MapLibre Native failed to draw a frame");
    } else requestAnimationFrame(demand);
  };
  demand();
  const resized = new ResizeObserver(() => session?.discarding.resize(sizeCanvas()));
  resized.observe(canvas);
  removing.signal.addEventListener("abort", () => resized.disconnect());
  gestures(canvas, map, removing.signal);

  return {
    camera: () => {
      const { center, zoom, bearing, pitch, roll } = map.getSnapshot().view.camera;
      return { center: [center.longitude, center.latitude], zoom, bearing, pitch, roll };
    },
    jumpTo: (next) => jumpTo(map, next),
    // Native reports camera changes after they happen, a frame or so later.
    onMove: (listener) => { moves.add(listener); return () => moves.delete(listener); },
    stop: () => map.discarding.cancelCameraTransitions(),
    setStyle: (next) => map.discarding.setStyleJson(absolute(next)),
    loaded: () => {
      const { fullyLoaded, gestureInProgress, transitionInProgress } = map.getSnapshot();
      return fullyLoaded && !gestureInProgress && !transitionInProgress;
    },
    remove,
  };
}

/**
 * A glyph range, where a range the server doesn't have is an empty one.
 * Native's labels wait for every range their text needs, and a range that
 * fails holds back the whole tile; GL JS and Orbis draw the tile and leave
 * out what the range would have drawn. The Tokyo extract's glyphs leave out
 * the ranges that GL JS and Orbis draw from local fonts (CJK and Hangul), so
 * without this Native drew none of its tiles; with it, Native leaves those
 * names out, having no local fonts here.
 */
async function glyphs(request, handle) {
  const fail = (errorReason, errorMessage) => handle.complete({ status: ResourceResponseStatus.error, errorReason, errorMessage });
  try {
    const response = await fetch(request.resolvedUrl);
    if (handle.isCancelled()) return;
    if (response.status === 404) handle.complete({ status: ResourceResponseStatus.noContent });
    else if (!response.ok) fail(ResourceErrorReason.server, `HTTP ${response.status}`);
    else handle.complete({ status: ResourceResponseStatus.ok, bytes: new Uint8Array(await response.arrayBuffer()) });
  } catch (e) {
    if (!handle.isCancelled()) fail(ResourceErrorReason.other, String(e));
  }
}

function jumpTo(map, { center: [longitude, latitude], zoom, bearing = 0, pitch = 0, roll = 0 }) {
  map.discarding.updateCamera({ mode: CameraUpdateMode.jump,
    camera: { center: { latitude, longitude }, zoom, bearing, pitch, roll } });
}

/**
 * `style` with its URLs absolute: Native resolves none against the page,
 * and the demo's own styles point at paths under the site.
 */
function absolute(style) {
  const page = new URL(".", location.href).href;
  const url = (u) => {
    if (typeof u !== "string") return u;
    if (u.startsWith("pmtiles://")) return `pmtiles://${url(u.slice("pmtiles://".length))}`;
    if (/^[a-z][a-z\d+.-]*:/i.test(u)) return u;
    // Not `new URL`, which would escape a template's braces.
    return u.startsWith("/") ? `${location.origin}${u}` : `${page}${u.replace(/^\.\//, "")}`;
  };
  const sources = Object.fromEntries(Object.entries(style.sources ?? {}).map(([id, source]) => [id, {
    ...source,
    ...(source.url && { url: url(source.url) }),
    ...(source.tiles && { tiles: source.tiles.map(url) }),
    ...(typeof source.data === "string" && { data: url(source.data) }),
    ...(source.urls && { urls: source.urls.map(url) }),
  }]));
  const sprite = Array.isArray(style.sprite) ? style.sprite.map((s) => ({ ...s, url: url(s.url) })) : url(style.sprite);
  return { ...style, sources, ...(style.glyphs && { glyphs: url(style.glyphs) }), ...(style.sprite && { sprite }) };
}

/**
 * The gestures of maplibre-native-ffi's browser example: a drag pans, a
 * right or Ctrl drag turns and tilts, two pointers pinch, turn and pan, the
 * wheel zooms, a double click zooms in (out with Shift), and the arrow and
 * +/- keys move. A drag or pinch keeps moving after release at the speed it
 * moved over its last 100 ms, and a press stops it.
 */
function gestures(canvas, map, signal) {
  const WINDOW = 100;
  const on = (type, listener, options = {}) => canvas.addEventListener(type, listener, { ...options, signal });
  const pointers = new Map();
  let rotating = false;
  let motion = [];
  let anchor;
  const apply = (delta) => map.discarding.applyCameraDelta(delta);
  const point = (e) => {
    const box = canvas.getBoundingClientRect();
    return { x: e.clientX - box.left, y: e.clientY - box.top };
  };
  const trim = (time) => { while (motion.length > 1 && motion[1].time <= time - WINDOW) motion.shift(); };
  const move = (e, delta) => {
    apply({ ...delta, gesturePhase: GesturePhase.update });
    const last = motion.at(-1);
    motion.push({ time: e.timeStamp, pan: { x: last.pan.x + (delta.offset?.x ?? 0), y: last.pan.y + (delta.offset?.y ?? 0) },
      zoom: last.zoom + Math.log2(delta.scale ?? 1), bearing: last.bearing + (delta.bearing ?? 0),
      pitch: last.pitch + (delta.pitch ?? 0) });
    trim(e.timeStamp);
  };

  on("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    canvas.focus({ preventScroll: true });
    if (pointers.size === 0) {
      rotating = e.pointerType === "mouse" && (e.button === 2 || e.ctrlKey);
      motion = [{ time: e.timeStamp, pan: { x: 0, y: 0 }, zoom: 0, bearing: 0, pitch: 0 }];
      anchor = undefined;
      apply({ gesturePhase: GesturePhase.begin });
    }
    pointers.set(e.pointerId, point(e));
  });
  on("pointermove", (e) => {
    const previous = pointers.get(e.pointerId);
    if (!previous) return;
    const current = point(e);
    const other = [...pointers].find(([id]) => id !== e.pointerId)?.[1];
    pointers.set(e.pointerId, current);
    const dx = current.x - previous.x;
    const dy = current.y - previous.y;
    if (other) {
      // The midpoint moves by half this pointer's movement; the span scales and turns about the other pointer.
      const before = { x: previous.x - other.x, y: previous.y - other.y };
      const after = { x: current.x - other.x, y: current.y - other.y };
      const scale = Math.hypot(after.x, after.y) / Math.hypot(before.x, before.y);
      const turn = (Math.atan2(after.y, after.x) - Math.atan2(before.y, before.x)) * (180 / Math.PI);
      anchor = { x: (current.x + other.x) / 2, y: (current.y + other.y) / 2 };
      move(e, { offset: { x: dx / 2, y: dy / 2 }, scale: Number.isFinite(scale) && scale > 0 ? scale : 1,
        bearing: -wrap(turn), anchor });
    } else if (rotating) move(e, { bearing: dx * 0.5, pitch: -dy * 0.5 });
    else move(e, { offset: { x: dx, y: dy } });
  });
  on("pointerup", (e) => {
    if (!pointers.delete(e.pointerId) || pointers.size > 0) return;
    trim(e.timeStamp);
    const first = motion[0];
    const last = motion.at(-1);
    const seconds = Math.min(e.timeStamp - first.time, WINDOW) / 1000;
    const speed = (from, to) => (seconds > 0 ? (to - from) / seconds : 0);
    apply({
      momentum: {
        pan: { velocity: { x: speed(first.pan.x, last.pan.x), y: speed(first.pan.y, last.pan.y) } },
        zoom: { velocity: speed(first.zoom, last.zoom), anchor },
        // Only a turn snaps the bearing to north.
        bearing: { velocity: speed(first.bearing, last.bearing), anchor, snap: last.bearing === 0 ? undefined : {} },
        pitch: { velocity: speed(first.pitch, last.pitch) },
      },
      gesturePhase: GesturePhase.end,
    });
  });
  on("pointercancel", (e) => {
    if (pointers.delete(e.pointerId) && pointers.size === 0) apply({ gesturePhase: GesturePhase.cancel });
  });
  on("contextmenu", (e) => e.preventDefault());
  on("wheel", (e) => {
    e.preventDefault();
    // A trackpad pinch arrives as a wheel event with Ctrl held, in smaller steps.
    const pixels = e.deltaMode === WheelEvent.DOM_DELTA_LINE ? e.deltaY * 40
      : e.deltaMode === WheelEvent.DOM_DELTA_PAGE ? e.deltaY * canvas.clientHeight : e.deltaY;
    apply({ scale: 2 ** (-pixels * (e.ctrlKey ? 1 / 100 : 1 / 300)), anchor: point(e) });
  }, { passive: false });
  on("dblclick", (e) => apply({ scale: e.shiftKey ? 0.5 : 2, anchor: point(e), animation: { durationMs: 300 } }));
  on("keydown", (e) => {
    const step = 100;
    const animation = { durationMs: 300 };
    const delta = {
      ArrowLeft: { offset: { x: step, y: 0 } }, ArrowRight: { offset: { x: -step, y: 0 } },
      ArrowUp: { offset: { x: 0, y: step } }, ArrowDown: { offset: { x: 0, y: -step } },
      "+": { scale: 2 }, "=": { scale: 2 }, "-": { scale: 0.5 },
    }[e.key];
    if (!delta) return;
    e.preventDefault();
    apply({ ...delta, animation });
  });
}

/** An angle in degrees, wrapped to [-180, 180). */
const wrap = (degrees) => ((((degrees + 180) % 360) + 360) % 360) - 180;
