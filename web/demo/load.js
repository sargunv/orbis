// The load-timing page (`web/scripts/load.mjs` and `memory.mjs` drive it):
// one map of `?style=` (a URL or a catalogue key, such as `alps`; with
// `?globe`, on the globe under the demo's sky) at
// `?at=lon,lat,zoom,bearing,pitch` in Orbis (`?engine=orbis`), the pinned
// GL JS (`gljs`), MapLibre Native's browser build (`native`, which needs
// the page cross-origin isolated) or mln-rs's web build (`mlnrs`, its
// module at `?mlnrs=<url>`), timed from this script's start to the
// engine's first `idle`. The title becomes `done {json}` with
// `performance.now()` at the script's start, the map's creation and idle
// (and `performance.timeOrigin`, to place them against the harness's
// request times), or `error <message>`. `window.tour()` then flies the map
// through a gesture tour, a camera a display frame, and resolves once it
// is idle again.
import { demoStyleWithScene } from "./styles.js";

const t0 = performance.now();
const params = new URLSearchParams(location.search);
const engine = params.get("engine") ?? "orbis";
const [lon, lat, zoom, bearing, pitch] = (params.get("at") ?? "0,0,0,0,0")
  .split(",")
  .map(Number);
const container = document.getElementById("map");
const nextFrame = () =>
  new Promise((resolve) => requestAnimationFrame(resolve));
const json = async (style) =>
  typeof style === "string" ? (await fetch(style)).json() : style;

// Each resolves to the map's creation time, a promise of its first idle,
// and how to jump its camera and wait for idle again.
const engines = {
  async orbis(style) {
    const { createMap } = await import("orbis");
    const map = await createMap({
      container,
      style,
      camera: { center: [lon, lat], zoom, bearing, pitch },
    });
    // GPU bytes the map allocated, by its own count (memory.mjs).
    window.gpuBytes = () => map.stats?.memory.allocated;
    return {
      created: performance.now(),
      idle: () => map.once("idle"),
      jumpTo: (c) => map.camera.jumpTo(c),
    };
  },
  async gljs(style) {
    const [gl, { default: workerUrl }] = await Promise.all([
      import("maplibre-gl"),
      import("maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url"),
      import("maplibre-gl/dist/maplibre-gl.css"),
    ]);
    const maplibregl = gl.default ?? gl;
    maplibregl.setWorkerUrl(workerUrl);
    // GL JS stops pitch at 60° unless told otherwise; Orbis and Native go to 85°.
    const map = new maplibregl.Map({
      container,
      style,
      center: [lon, lat],
      zoom,
      bearing,
      pitch,
      maxPitch: 85,
      attributionControl: false,
    });
    map.on("error", (e) => console.error(String(e.error)));
    return {
      created: performance.now(),
      idle: () => new Promise((resolve) => map.once("idle", resolve)),
      jumpTo: (c) => map.jumpTo(c),
    };
  },
  async native(style) {
    const { open } = await import("./app/native.js");
    const map = await open(container, {
      style: await json(style),
      camera: { center: [lon, lat], zoom, bearing, pitch },
    });
    // Native has no idle event: its snapshot says when it's fully loaded and still.
    const idle = async () => {
      do await nextFrame();
      while (!map.loaded());
    };
    return { created: performance.now(), idle, jumpTo: map.jumpTo };
  },
  async mlnrs(style) {
    const { Map } = await import(/* @vite-ignore */ params.get("mlnrs"));
    const map = new Map({
      container,
      style,
      center: [lon, lat],
      zoom,
      bearing,
      pitch,
      interactive: false,
    });
    map.on("error", (e) => console.error(String(e.error)));
    return {
      created: performance.now(),
      idle: () => map.once("idle"),
      jumpTo: (c) => map.jumpTo(c),
    };
  },
};

/** The tour's camera at `f` in [0, 1]: two screens east and back, two zooms in and back, a half turn tilting to 60°. */
function tourCamera(f) {
  const wave = Math.sin(f * 2 * Math.PI);
  return {
    center: [lon + (720 / 2 ** zoom) * wave, lat],
    zoom: zoom + 2 * Math.sin(f * Math.PI),
    bearing: bearing + 180 * f,
    pitch: Math.max(pitch, 60 * Math.sin(f * Math.PI)),
  };
}

try {
  const map = await engines[engine](await demoStyleWithScene(params));
  await map.idle();
  window.tour = async (frames = 360) => {
    for (let f = 1; f < frames; f++) {
      map.jumpTo(tourCamera(f / frames));
      await nextFrame();
    }
    // Waiting from before the last camera: an engine may go idle in the frame that shows it.
    const idle = map.idle();
    map.jumpTo(tourCamera(1));
    await idle;
  };
  document.title =
    "done " +
    JSON.stringify({
      origin: performance.timeOrigin,
      script: t0,
      created: map.created,
      idle: performance.now(),
    });
} catch (e) {
  document.title = "error " + e;
}
