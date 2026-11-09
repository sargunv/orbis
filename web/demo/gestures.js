// The gestures bench in the browser (`bench.html?experiment=gestures`):
// `orb-bench`'s rotate-and-tilt and zoom paths, driven through the orbis
// package's camera once a display frame. Where the
// native bench times `Map::render` alone, this times what a user sees: the
// interval between frames reaching the page, the render CPU time of each,
// main-thread long tasks, and how long the map takes to reach `idle` after
// the path ends (a backlog of frames shows there). `paths=alps` runs the
// terrain demo's view instead: rotate and pan at z11.6, pitch 72, and its
// 60 s flight (`data/alps-flight.json`), flown as the demo flies it.
// `engine=gl` drives MapLibre GL JS along the same paths instead and
// counts its labels' blinks and returns as the native bench counts
// Orbis's, from `style.placement`'s opacities after each frame.
import { createMap } from "orbis";
import alpsFlight from "./data/alps-flight.json";

/** Frames per path: 6 s at 60 Hz. */
const PATH_FRAMES = 360;
const CITY_PATHS = [
  {
    name: "rotate and tilt at z14",
    camera: (center, f) => ({ center, zoom: 14, bearing: 360 * f, pitch: 30 - 30 * Math.cos(f * 2 * Math.PI) }),
  },
  {
    name: "zoom z12 → z16 → z12",
    camera: (center, f) => ({ center, zoom: 14 - 2 * Math.cos(f * 2 * Math.PI), bearing: 0, pitch: 0 }),
  },
  {
    name: "zoom z12 → z16 → z12, 3 times",
    camera: (center, f) => ({ center, zoom: 14 - 2 * Math.cos(3 * f * 2 * Math.PI), bearing: 0, pitch: 0 }),
  },
];
/** The terrain demo's view: z11.6, bearing −32, pitch 72. */
const alpsView = (center, lon, bearing) => ({ center: [center[0] + lon, center[1]], zoom: 11.6, bearing, pitch: 72 });
const ALPS_PATHS = [
  { name: "rotate at z11.6, pitch 72", camera: (center, f) => alpsView(center, 0, -32 + 360 * f) },
  { name: "pan at z11.6, pitch 72", camera: (center, f) => alpsView(center, 0.15 * Math.sin(f * 2 * Math.PI), -32) },
  { name: "Alps flight (60 s)", camera: (center) => alpsView(center, 0, -32), flight: alpsFlight.legs },
];
const PATH_SETS = { city: CITY_PATHS, alps: ALPS_PATHS };
const LOAD_TIMEOUT = 120_000;
/** Within this many frames, a label that all but vanished (under 0.1) and is back in full (0.9) blinked, or returned. */
const FLASH_FRAMES = 10;
const RETURN_FRAMES = 60;

const percentile = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.round((s.length - 1) * p))] : NaN;
};
const ms = (x) => (Number.isFinite(x) ? x.toFixed(1) : "—");
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));

function idle(map, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label}: not idle after ${LOAD_TIMEOUT / 1000} s`)), LOAD_TIMEOUT);
    map.once("idle").then(() => { clearTimeout(timer); resolve(); });
  });
}

/** Runs one path on a fresh map; resolves to its row. */
async function runPath(container, style, jobWorkers, center, path, show) {
  const label = path.name;
  show(`${label}: loading`);
  const created = performance.now();
  const map = await createMap({ container, style, jobWorkers, camera: path.camera(center, 0), interactive: false });
  try {
    await idle(map, label);
    const load = performance.now() - created;
    show(`${label}: moving`);
    const acks = [];
    const off = map.on("render", (ack) => acks.push({ t: performance.now(), ms: ack.ms, placement: ack.placementMs }));
    const longs = [];
    const observer = typeof PerformanceObserver !== "undefined" &&
      PerformanceObserver.supportedEntryTypes?.includes("longtask")
      ? new PerformanceObserver((list) => { for (const e of list.getEntries()) longs.push({ start: e.startTime, duration: e.duration }); })
      : null;
    observer?.observe({ type: "longtask" });
    const started = await nextFrame();
    if (path.flight) {
      for (const leg of path.flight) await map.camera.flyTo(leg);
    } else {
      for (let f = 1; f <= PATH_FRAMES; f++) {
        map.camera.jumpTo(path.camera(center, f / PATH_FRAMES));
        await nextFrame();
      }
    }
    const ended = performance.now();
    await idle(map, label);
    const settle = performance.now() - ended;
    observer?.disconnect();
    off();
    const moving = acks.filter((a) => a.t > started && a.t <= ended);
    const after = acks.filter((a) => a.t > ended);
    const longsAfter = longs.filter((l) => l.start > ended).map((l) => l.duration);
    const gaps = moving.slice(1).map((a, i) => a.t - moving[i].t);
    const cpu = moving.map((a) => a.ms);
    return {
      path: path.name, frames: moving.length, load,
      interval: [0.5, 0.95, 0.99, 1].map((p) => percentile(gaps, p)),
      cpu: [0.5, 0.95, 1].map((p) => percentile(cpu, p)),
      placement: [0.5, 0.95].map((p) => percentile(moving.map((a) => a.placement), p)),
      missed: gaps.filter((g) => g > 50).length,
      longs: longs.length, longest: Math.max(0, ...longs.map((l) => l.duration)),
      settle, settleCpu: Math.max(0, ...after.map((a) => a.ms)), settleLong: Math.max(0, ...longsAfter),
    };
  } finally {
    map.remove();
  }
}

/**
 * Per label text, the greatest opacity GL JS draws its text with now: each
 * drawn symbol tile's instances, their `style.placement` opacity moved on by
 * the fade since its commit (what the symbol shader computes).
 */
function glShown(map, maplibregl, texts) {
  const style = map.style;
  const placement = style.placement;
  const shown = new Map();
  if (!placement) return shown;
  const change = placement.symbolFadeChange(maplibregl.now());
  for (const id of style._order) {
    const layer = style._layers[id];
    if (layer.type !== "symbol") continue;
    const tiles = style.tileManagers[layer.source];
    for (const key of tiles.getRenderableIds(true)) {
      const tile = tiles.getTileByID(key);
      const bucket = tile.getBucket(layer);
      if (!bucket?.symbolInstances) continue;
      let names = texts.get(bucket);
      if (!names) {
        // The texts the layer's `text-field` evaluates to for each instance's feature.
        const index = tile.latestFeatureIndex;
        const source = index.loadVTLayers()[index.sourceLayerCoder.decode(bucket.sourceLayerIndex)];
        const field = layer.layout.get("text-field");
        names = [];
        for (let i = 0; i < bucket.symbolInstances.length; i++) {
          const f = source.feature(bucket.symbolInstances.get(i).featureIndex);
          const text = field.evaluate({ type: f.type, id: f.id, properties: f.properties, geometry: [] }, {}, tile.tileID.canonical);
          names.push(text ? text.toString() : "");
        }
        texts.set(bucket, names);
      }
      for (let i = 0; i < names.length; i++) {
        if (!names[i]) continue;
        const o = placement.opacities[bucket.symbolInstances.get(i).crossTileID];
        const v = o ? Math.min(1, Math.max(0, o.text.opacity + (o.text.placed ? change : -change))) : 0;
        if (v > (shown.get(names[i]) ?? 0)) shown.set(names[i], v);
      }
    }
  }
  return shown;
}

/** Counts labels' blinks and returns frame by frame, as `orb-bench` does. */
function labelCounter() {
  const labels = new Map();
  const counts = { blinks: 0, blinkFrames: 0, returns: 0, drops: 0 };
  return {
    counts,
    frame(f, shown) {
      for (const [name, since] of labels) {
        const opacity = shown.get(name) ?? 0;
        if (since === null && opacity < 0.1) { labels.set(name, f); counts.drops++; }
        else if (since !== null && opacity >= 0.9) {
          const gone = f - since;
          if (gone <= FLASH_FRAMES) { counts.blinks++; counts.blinkFrames += gone; }
          if (gone <= RETURN_FRAMES) counts.returns++;
          labels.set(name, null);
        }
      }
      for (const [name, opacity] of shown) if (opacity >= 0.9 && !labels.has(name)) labels.set(name, null);
    },
  };
}

let glModule;
async function loadGl() {
  if (!glModule) {
    const [gl, { Protocol }, { default: workerUrl }] = await Promise.all([
      import("maplibre-gl"), import("pmtiles"), import("maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url"),
    ]);
    glModule = gl.default ?? gl;
    glModule.setWorkerUrl(workerUrl);
    glModule.addProtocol("pmtiles", new Protocol().tile);
  }
  return glModule;
}

/** Runs one path through GL JS on a fresh map; resolves to its row. */
async function runGlPath(container, styleUrl, center, path, show) {
  const maplibregl = await loadGl();
  const style = await (await fetch(styleUrl)).json();
  // GL JS wants absolute URLs.
  const absolute = (url) => new URL(url, new URL(styleUrl, location.href)).href;
  if (style.sprite) style.sprite = absolute(style.sprite);
  if (style.glyphs) style.glyphs = absolute(style.glyphs).replace(/%7B/g, "{").replace(/%7D/g, "}");
  for (const source of Object.values(style.sources)) {
    if (source.url?.startsWith("pmtiles://")) source.url = `pmtiles://${absolute(source.url.slice(10))}`;
  }
  show(`${path.name}: loading (GL JS)`);
  const created = performance.now();
  const map = new maplibregl.Map({ container, style, ...path.camera(center, 0), interactive: false, attributionControl: false });
  try {
    await idle(map, path.name);
    const load = performance.now() - created;
    show(`${path.name}: moving (GL JS)`);
    const texts = new WeakMap();
    const counter = labelCounter();
    const times = [];
    for (let f = 1; f <= PATH_FRAMES; f++) {
      map.jumpTo(path.camera(center, f / PATH_FRAMES));
      times.push(await nextFrame());
      counter.frame(f, glShown(map, maplibregl, texts));
    }
    const ended = performance.now();
    // A repaint ends in `idle` once nothing is left to do.
    const settled = idle(map, path.name);
    map.triggerRepaint();
    await settled;
    const gaps = times.slice(1).map((t, i) => t - times[i]);
    return { path: path.name, load, frames: times.length, interval: [0.5, 0.95, 1].map((p) => percentile(gaps, p)),
      ...counter.counts, settle: performance.now() - ended };
  } finally {
    map.remove();
  }
}

/** The GL JS run's Markdown report: the paths but a flight. */
async function runGlGestures({ style, center, run, size, about, show, container }) {
  const rows = [];
  for (const path of run.filter((p) => !p.flight)) rows.push(await runGlPath(container, style, center, path, show));
  return [
    "### Gestures bench (page, MapLibre GL JS)", "",
    ...about.map((line) => `- ${line}`),
    `- ${size[0]}×${size[1]} CSS px at ${center[0]}, ${center[1]}; ${PATH_FRAMES} frames a path, ` +
      "the camera moved once a display frame (`jumpTo`), each path from a fresh map at rest",
    "",
    "| path | load to idle ms | frames | interval p50 | p95 | max | label blinks (frames) | label returns (drops) | idle after ms |",
    "|---|---|---|---|---|---|---|---|---|",
    ...rows.map((r) => `| ${r.path} | ${r.load.toFixed(0)} | ${r.frames} | ${r.interval.map(ms).join(" | ")} | ` +
      `${r.blinks} (${r.blinkFrames}) | ${r.returns} (${r.drops}) | ${r.settle.toFixed(0)} |`),
    "",
    `Label blinks: label texts drawn in full that all but vanished (opacity under 0.1) and were back in full within ${FLASH_FRAMES} ` +
      `frames, and the frames they were gone; label returns: back in full within ${RETURN_FRAMES} frames (1 s), blinks included, of the label texts drawn in full that all but vanished. ` +
      "Opacities are `style.placement`'s, moved on by the fade since its commit, over the tiles GL JS draws.",
  ].join("\n") + "\n";
}

/** Runs every path of `paths` (`city` or `alps`); resolves to the Markdown report. */
export async function runGestures({ style, center, paths, jobWorkers, size, about, show, engine = "orbis" }) {
  const run = PATH_SETS[paths];
  if (!run) throw new Error("paths must be city or alps");
  if (engine === "gl") {
    const container = document.createElement("div");
    Object.assign(container.style, { width: `${size[0]}px`, height: `${size[1]}px`, position: "relative" });
    document.body.append(container);
    try {
      return await runGlGestures({ style, center, run, size, about, show, container });
    } finally {
      container.remove();
    }
  }
  const container = document.createElement("div");
  Object.assign(container.style, { width: `${size[0]}px`, height: `${size[1]}px`, position: "relative" });
  document.body.append(container);
  const rows = [];
  try {
    for (const path of run) rows.push(await runPath(container, style, jobWorkers, center, path, show));
  } finally {
    container.remove();
  }
  const lines = [
    "### Gestures bench (page)", "",
    ...about.map((line) => `- ${line}`),
    `- ${size[0]}×${size[1]} CSS px at ${center[0]}, ${center[1]}; ${PATH_FRAMES} frames a path, ` +
      "the camera moved once a display frame (`jumpTo`; a flight's legs through `flyTo`), each path from a fresh map at rest; " +
      `${jobWorkers ?? "default"} job workers`,
    "",
    "| path | load to idle ms | frames | interval p50 | p95 | p99 | max | gaps > 50 ms | render CPU p50 | p95 | max | long tasks (max ms) | idle after ms | after: render CPU max | page long task max | placement p50 | p95 |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ...rows.map((r) => `| ${r.path} | ${r.load.toFixed(0)} | ${r.frames} | ${r.interval.map(ms).join(" | ")} | ${r.missed} | ` +
      `${r.cpu.map(ms).join(" | ")} | ${r.longs} (${ms(r.longest)}) | ${r.settle.toFixed(0)} | ${ms(r.settleCpu)} | ${ms(r.settleLong)} | ${r.placement.map(ms).join(" | ")} |`),
    "",
    "Times in ms. Interval: between frames reaching the page while the camera moves; render CPU: the frame's " +
      "`render`; long tasks: the page's, over 50 ms; idle after: " +
      "from the path's end until the map's `idle` event; after: the frames and the page's long tasks from the path's end to `idle`; " +
      "placement: the page thread's share of render CPU (building the placement worker's frame, applying commits; 0.2 ms).",
  ];
  return `${lines.join("\n")}\n`;
}
