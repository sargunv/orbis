// The load-timing page (`web/scripts/load.mjs` and `memory.mjs` drive it):
// one map of `?style=` (a URL or a catalogue key, such as `alps`; with
// `?globe`, on the globe under the demo's sky) at
// `?at=lon,lat,zoom,bearing,pitch` in Orbis (`?engine=orbis`), the pinned
// GL JS (`gljs`), MapLibre Native's browser build (`native`, which needs
// the page cross-origin isolated) or mln-rs's browser build (`mlnrs`, its
// module at `?mlnrs=<url>`: `scripts/mlnrs-map.js`, which has no `tour`),
// timed from this script's start to the engine's first `idle`. `?nonce=`
// compiles Orbis's pipelines cold (`cold.js`). The title becomes `done {json}` with `performance.now()` at the script's
// start, the map's creation and idle
// (and `performance.timeOrigin`, to place them against the harness's
// request times), or `error <message>`. `window.tour()` then flies the map
// through a gesture tour, a camera a display frame, and resolves once it
// is idle again. With `?frames`, `window.mapFrames` holds every frame the
// map draws, by `performance.now()` as the page hears of it, with Orbis's
// counters (session.mjs); `window.mapIdle()` waits for the map's next idle and
// `window.mapCamera()` gives Orbis's or GL JS's camera.
import { bustShaderCaches } from "./cold.js";
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
window.mapFrames = [];
const recording = params.has("frames");
const json = async (style) =>
  typeof style === "string" ? (await fetch(style)).json() : style;

// Each resolves to the map's creation time, a promise of its first idle,
// and how to jump its camera and wait for idle again.
const engines = {
  async orbis(style) {
    const { createMap } = await import("orbis");
    const busted = bustShaderCaches(params);
    const backend = params.get("backend") ?? undefined;
    const map = await createMap({
      container,
      style,
      camera: { center: [lon, lat], zoom, bearing, pitch },
      ...(backend && { backend }),
    });
    // GPU bytes the map allocated, by its own count, and what WebGL 2's
    // CPU driver keeps in wasm memory (memory.mjs).
    window.gpuBytes = () => map.stats?.memory.allocated;
    window.cpuCopies = () => map.stats?.memory.cpuCopies;
    // GPU timestamps of each submission's passes (session.mjs's `--gpu`).
    window.mapGpu = {
      start: () => map.setGpuTiming(true),
      take: () => map.gpuTimings(),
      repaint: () => map.triggerRepaint(),
    };
    // The shaders `?nonce=` made cold, and the backend (load.mjs).
    window.loadDetails = () => ({ busted: busted(), backend: map.backend });
    if (recording)
      map.on("render", (s) =>
        mapFrames.push({
          t: performance.now(),
          at: at(map.camera.current),
          ...counters(s),
        }),
      );
    return {
      created: performance.now(),
      idle: () => map.once("idle"),
      jumpTo: (c) => map.camera.jumpTo(c),
      camera: () => map.camera.current,
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
    window.mapGpu = glGpuTiming(map);
    if (recording) {
      // Whether the tiles under the view's centre are in, as Orbis's
      // `centreLoaded` says: looked at once a frame until they are, and
      // again once the camera moves.
      let centre = false;
      map.on("move", () => (centre = false));
      map.on("render", () => {
        centre ||= centreLoaded(map, maplibregl);
        const [lng, lat] = map.getCenter().toArray();
        mapFrames.push({
          t: performance.now(),
          at: at({ center: [lng, lat], zoom: map.getZoom() }),
          centre,
        });
      });
    }
    return {
      created: performance.now(),
      idle: () => new Promise((resolve) => map.once("idle", resolve)),
      jumpTo: (c) => map.jumpTo(c),
      camera: () => ({
        center: map.getCenter().toArray(),
        zoom: map.getZoom(),
        bearing: map.getBearing(),
        pitch: map.getPitch(),
      }),
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

/** A frame's camera, as session.mjs dumps it: lon, lat, zoom. */
const at = (c) => c && [...c.center, c.zoom].map((x) => +x.toFixed(4));

/** What session.mjs reads of an Orbis frame's counters: enough to say why a frame came late. */
const counters = (s) => ({
  ms: s.ms,
  needs: s.needsFrame,
  times: Object.fromEntries(STAGES.map((k) => [k, s.times[k]])),
  arrived: s.tilesArrived,
  drawn: s.tilesDrawn,
  loading: s.tilesLoading,
  meshes: s.meshesUploaded,
  uploaded:
    s.uploadedFromWasm + s.uploadedFromJs + s.texelsFromWasm + s.texelsFromJs,
  jobs: s.jobsDone,
  pending: s.pipelinesPending,
  waiting: s.layersWaiting,
  gpuBytes: s.memory.allocated,
  rebuilt: s.candidatesRebuilt,
  // What the device did since the last frame: pipelines asked for, memory
  // made, uploads and their calls, texture-write submissions.
  device: s.gpu,
  drapeSlots: s.drapeSlots,
  input: s.input,
  moved: s.moved,
  centre: s.centreLoaded,
  // session.mjs's running tally of WebGPU calls, where it wraps them.
  gpu: globalThis.sessionGpuCalls?.(),
});
const STAGES = [
  "total",
  "arrivals",
  "upload",
  "terrain",
  "plan",
  "drape",
  "place",
  "placement",
  "cull",
  "bundles",
  "drapePasses",
  "encode",
  "submit",
];

/** GL JS's GPU time a frame, as Orbis's `mapGpu` gives its own (session.mjs's
 * `--gpu`): a WebGL 2 timer query around each `painter.render`, read once
 * the GPU has it, each a submission of one pass, `frame`. Null where the
 * browser has no timer queries. */
function glGpuTiming(map) {
  const gl = map.painter.context.gl;
  const ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
  if (!ext) return null;
  let timing = false;
  const pending = [];
  const frames = [];
  const read = () => {
    while (
      pending.length &&
      gl.getQueryParameter(pending[0], gl.QUERY_RESULT_AVAILABLE)
    ) {
      const query = pending.shift();
      // A disjoint run (the GPU's clock changed) times nothing.
      if (!gl.getParameter(ext.GPU_DISJOINT_EXT)) {
        const ms = gl.getQueryParameter(query, gl.QUERY_RESULT) / 1e6;
        frames.push({ passes: [["frame", ms]], span: ms });
      }
      gl.deleteQuery(query);
    }
  };
  const render = map.painter.render.bind(map.painter);
  map.painter.render = (...args) => {
    if (!timing) return render(...args);
    read();
    const query = gl.createQuery();
    gl.beginQuery(ext.TIME_ELAPSED_EXT, query);
    try {
      return render(...args);
    } finally {
      gl.endQuery(ext.TIME_ELAPSED_EXT);
      pending.push(query);
    }
  };
  return {
    start: () => (timing = true),
    take: async () => {
      timing = false;
      for (let i = 0; pending.length && i < 60; i++) {
        await nextFrame();
        read();
      }
      return frames.splice(0);
    },
    repaint: () => map.triggerRepaint(),
  };
}

/** Whether every source GL JS draws has the ideal tile under the view's
 * centre loaded (or failed): Orbis's `centreLoaded`. */
function centreLoaded(map, maplibregl) {
  const style = map.style;
  if (!style?._loaded) return false;
  const at = maplibregl.MercatorCoordinate.fromLngLat(map.getCenter());
  return Object.values(style.tileManagers ?? style.sourceCaches ?? {}).every(
    (tm) => {
      if (!tm.used) return true;
      const source = tm.getSource();
      if (!source.loaded?.() && source.tileSize === undefined) return false;
      const ideal = map
        .coveringTiles({
          tileSize: source.tileSize ?? 512,
          minzoom: source.minzoom,
          maxzoom: source.maxzoom,
          roundZoom: source.roundZoom,
        })
        .find(({ canonical: c }) => {
          const n = 2 ** c.z;
          return Math.floor(at.x * n) === c.x && Math.floor(at.y * n) === c.y;
        });
      if (!ideal) return true;
      const state = tm.getTile(ideal)?.state;
      return state !== undefined && state !== "loading";
    },
  );
}

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
  // The style as the engines load it, for a native twin to load the same.
  const style = (window.mapStyle = await demoStyleWithScene(params));
  const map = await engines[engine](style);
  window.mapIdle = map.idle;
  window.mapCamera = map.camera;
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
