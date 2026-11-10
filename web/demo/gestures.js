// The gestures bench in the browser (`bench.html?experiment=gestures`):
// `orb-bench`'s rotate-and-tilt and zoom paths, driven through the orbis
// package's camera once a display frame. Where the
// native bench times `Map::render` alone, this times what a user sees: the
// interval between frames reaching the page, the render CPU time of each,
// main-thread long tasks, and how long the map takes to reach `idle` after
// the path ends (a backlog of frames shows there). Frame pacing: each
// frame is stamped with its animation frame's time, as both engines draw
// in one, and gaps over 16.7 and 25 ms between them count; marks around
// each path let `pnpm bench` count the GPU process's presents, which show
// a GPU running behind where animation frames don't. `paths=dense`
// pans, zooms and turns at pitch 60 around z15 over central Paris;
// `paths=alps` runs the Alps terrain style's view instead: rotate and pan at
// z11.6, pitch 72, three fast turns at pitch 80, and its 60 s flight
// (`data/alps-flight.json`), flown as the demo flies it; `paths=globe`
// spins, zooms and turns the globe.
// `engine=gl` drives MapLibre GL JS along the same paths and counts its
// labels' blinks and returns as the native bench counts Orbis's, from
// `style.placement`'s opacities after each frame; `engine=orbis,gl` runs
// both, path by path.
import { createMap } from "orbis";

import alpsFlight from "./data/alps-flight.json";

/** Frames per path: 6 s at 60 Hz. */
const PATH_FRAMES = 360;
/** A path's length with `pace=wall`. */
const PATH_MS = 6000;
/** Gaps between frames over these (ms) count, each with 1 ms for the browser's coarsened clock. */
const LATE = [16.7, 25];
const LATE_SLACK = 1;
const CITY_PATHS = [
  {
    name: "rotate and tilt at z14",
    camera: (center, f) => ({
      center,
      zoom: 14,
      bearing: 360 * f,
      pitch: 30 - 30 * Math.cos(f * 2 * Math.PI),
    }),
  },
  {
    name: "zoom z12 → z16 → z12",
    camera: (center, f) => ({
      center,
      zoom: 14 - 2 * Math.cos(f * 2 * Math.PI),
      bearing: 0,
      pitch: 0,
    }),
  },
  {
    name: "zoom z12 → z16 → z12, 3 times",
    camera: (center, f) => ({
      center,
      zoom: 14 - 2 * Math.cos(3 * f * 2 * Math.PI),
      bearing: 0,
      pitch: 0,
    }),
  },
];
/** The Alps terrain style's view: z11.6, bearing −32, pitch 72. */
const alpsView = (center, lon, bearing, pitch = 72) => ({
  center: [center[0] + lon, center[1]],
  zoom: 11.6,
  bearing,
  pitch,
});
const ALPS_PATHS = [
  {
    name: "rotate at z11.6, pitch 72",
    camera: (center, f) => alpsView(center, 0, -32 + 360 * f),
  },
  {
    name: "fast turns at z11.6, pitch 80",
    camera: (center, f) => alpsView(center, 0, -32 + 3 * 360 * f, 80),
  },
  {
    name: "pan at z11.6, pitch 72",
    camera: (center, f) =>
      alpsView(center, 0.15 * Math.sin(f * 2 * Math.PI), -32),
  },
  {
    name: "Alps flight (60 s)",
    camera: (center) => alpsView(center, 0, -32),
    flight: alpsFlight.legs,
  },
];
/** Dense city views at pitch 60: what a user's pan, pinch and twist do in a city. */
const DENSE_PATHS = [
  {
    name: "pan at z15, pitch 60",
    camera: ([lon, lat], f) => ({
      center: [lon + 0.012 * Math.sin(f * 2 * Math.PI), lat],
      zoom: 15,
      bearing: 30,
      pitch: 60,
    }),
  },
  {
    name: "zoom z14 → z17 → z14, pitch 60",
    camera: (center, f) => ({
      center,
      zoom: 15.5 - 1.5 * Math.cos(f * 2 * Math.PI),
      bearing: 30,
      pitch: 60,
    }),
  },
  {
    name: "rotate at z15, pitch 60",
    camera: (center, f) => ({
      center,
      zoom: 15,
      bearing: 30 + 360 * f,
      pitch: 60,
    }),
  },
];
const GLOBE_PATHS = [
  {
    name: "spin the globe at z1.5",
    camera: ([lon, lat], f) => ({
      center: [lon + 360 * f, lat],
      zoom: 1.5,
      bearing: 0,
      pitch: 0,
    }),
  },
  {
    name: "zoom z1.5 → z6 → z1.5",
    camera: (center, f) => ({
      center,
      zoom: 3.75 - 2.25 * Math.cos(f * 2 * Math.PI),
      bearing: 0,
      pitch: 0,
    }),
  },
  {
    name: "turn and tilt at z3",
    camera: (center, f) => ({
      center,
      zoom: 3,
      bearing: 360 * f,
      pitch: 20 - 20 * Math.cos(f * 2 * Math.PI),
    }),
  },
];
const PATH_SETS = {
  city: CITY_PATHS,
  dense: DENSE_PATHS,
  alps: ALPS_PATHS,
  globe: GLOBE_PATHS,
};
/** Where each path set runs unless the page says otherwise. */
export const PATH_CENTERS = {
  city: [139.74, 35.685],
  dense: [2.35, 48.857],
  alps: [7.7491, 46.0207],
  globe: [10, 30],
};
const LOAD_TIMEOUT = 120_000;
/** Within this many frames, a label that all but vanished (under 0.1) and is back in full (0.9) blinked, or returned. */
const FLASH_FRAMES = 10;
const RETURN_FRAMES = 60;

const percentile = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length
    ? s[Math.min(s.length - 1, Math.round((s.length - 1) * p))]
    : NaN;
};
const ms = (x) => (Number.isFinite(x) ? x.toFixed(1) : "—");
const whole = (x) => (Number.isFinite(x) ? x.toFixed(0) : "—");
const nextFrame = () =>
  new Promise((resolve) => requestAnimationFrame(resolve));
/** The current animation frame's time: a frame drawn in it reaches the screen with that frame. */
const frameTime = () => document.timeline.currentTime;
/**
 * Where along a path (0 to 1) display frame `i` (from 1) puts the camera:
 * `i` of PATH_FRAMES, or with `wall`, the time since the clock started
 * over PATH_MS, as a hand moves whatever the frame rate. On a device that
 * draws slower than the display, a frame-paced path slows the camera down
 * with it, and the tiles it asks for: only a wall-paced one shows whether
 * tile loads keep up with a gesture.
 */
function pathClock(wall) {
  const start = performance.now();
  return (i) =>
    wall ? Math.min(1, (performance.now() - start) / PATH_MS) : i / PATH_FRAMES;
}
/** How the reports say a path was paced. */
const pace = (wall) =>
  wall
    ? `${PATH_MS / 1000} s a path (\`pace=wall\`), the camera moved once a display frame to where the time puts it`
    : `${PATH_FRAMES} frames a path, the camera moved once a display frame`;

/** Frame pacing from frames' times: the gaps' percentiles and how many are late. */
function pacing(times) {
  const gaps = times.slice(1).map((t, i) => t - times[i]);
  return {
    interval: [0.5, 0.95, 0.99, 1].map((p) => percentile(gaps, p)),
    late: LATE.map(
      (limit) => gaps.filter((g) => g > limit + LATE_SLACK).length,
    ),
  };
}
const PACING_HEAD = `interval p50 | p95 | p99 | max | ${LATE.map((l) => `gaps > ${l} ms`).join(" | ")}`;
const pacingCells = (r) =>
  `${r.interval.map(ms).join(" | ")} | ${r.late.join(" | ")}`;
const PACING_NOTE =
  "Interval: between the animation frames that drew a frame while the camera moved, each a display frame " +
  `when the map keeps up; gaps over ${LATE.join(" and ")} ms (with ${LATE_SLACK} ms for the coarsened clock) are frames that came late ` +
  "on the page's thread. Animation frames keep their pace while the GPU runs behind: `pnpm bench` adds the GPU process's presents.";

/** The page's long tasks from now until `stop()`, as `{ start, duration }`. */
function longTasks() {
  const longs = [];
  const add = (entries) => {
    for (const e of entries)
      longs.push({ start: e.startTime, duration: e.duration });
  };
  const observer =
    typeof PerformanceObserver !== "undefined" &&
    PerformanceObserver.supportedEntryTypes?.includes("longtask")
      ? new PerformanceObserver((list) => add(list.getEntries()))
      : null;
  observer?.observe({ type: "longtask" });
  return {
    longs,
    stop: () => {
      add(observer?.takeRecords() ?? []);
      observer?.disconnect();
    },
  };
}

function idle(map, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () =>
        reject(new Error(`${label}: not idle after ${LOAD_TIMEOUT / 1000} s`)),
      LOAD_TIMEOUT,
    );
    map.once("idle").then(() => {
      clearTimeout(timer);
      resolve();
    });
  });
}

/** Runs one path on a fresh map; resolves to its row. */
async function runPath(
  container,
  style,
  jobWorkers,
  center,
  path,
  show,
  { gpu, wall },
) {
  const label = path.name;
  show(`${label}: loading`);
  const created = performance.now();
  const map = await createMap({
    container,
    style,
    jobWorkers,
    camera: path.camera(center, 0),
    interactive: false,
    // As the demo draws it: the Globe entry's light and fills are previews.
    previews: "all",
  });
  try {
    await idle(map, label);
    const load = performance.now() - created;
    show(`${label}: moving`);
    const acks = [];
    const off = map.on("render", (ack) =>
      acks.push({
        t: frameTime(),
        now: performance.now(),
        ms: ack.ms,
        placement: ack.times.placement,
        loading: ack.tilesLoading,
        loaded: ack.loaded,
        rebuilt: ack.candidatesRebuilt,
      }),
    );
    if (gpu) map.setGpuTiming(true);
    const { longs, stop } = longTasks();
    const started = await nextFrame();
    performance.mark(`gestures:orbis:${path.name}:start`);
    if (path.flight) {
      for (const leg of path.flight) await map.camera.flyTo(leg);
    } else {
      const at = pathClock(wall);
      for (let i = 1, f = 0; f < 1; i++) {
        f = at(i);
        map.camera.jumpTo(path.camera(center, f));
        await nextFrame();
      }
    }
    const ended = performance.now();
    performance.mark(`gestures:orbis:${path.name}:end`);
    await idle(map, label);
    const settle = performance.now() - ended;
    stop();
    off();
    // GPU passes frame by frame, split by whether the frame laid the
    // candidate lists out again: a render's submission is the one with a
    // main pass (others upload).
    let passes;
    if (gpu) {
      const submissions = (await map.gpuTimings()).filter((s) =>
        s.passes.some(([l]) => l === "main"),
      );
      map.setGpuTiming(false);
      if (submissions.length === acks.length) {
        const of = (rebuilt, label) =>
          submissions
            .filter((_, i) => acks[i].rebuilt === rebuilt)
            .map((s) => s.passes.find(([l]) => l === label)?.[1])
            .filter((t) => t !== undefined);
        passes = [true, false].map((rebuilt) => ({
          frames: acks.filter((a) => a.rebuilt === rebuilt).length,
          cull: of(rebuilt, "cull"),
          main: of(rebuilt, "main"),
          span: submissions
            .filter((_, i) => acks[i].rebuilt === rebuilt)
            .map((s) => s.span),
        }));
      }
    }
    const moving = acks.filter((a) => a.t > started && a.t <= ended);
    const after = acks.filter((a) => a.t > ended);
    const longsMoving = longs
      .filter((l) => l.start <= ended)
      .map((l) => l.duration);
    const longsAfter = longs
      .filter((l) => l.start > ended)
      .map((l) => l.duration);
    const cpu = moving.map((a) => a.ms);
    // A backlog of tile loads: how many are in flight while the camera
    // moves, and how long after it stops the view's own tiles take to
    // arrive and all loads to drain.
    const since = (a) => (a ? a.now - ended : NaN);
    const loads = {
      max: Math.max(0, ...moving.map((a) => a.loading)),
      end: moving.at(-1)?.loading ?? 0,
      loaded: since(after.find((a) => a.loaded)),
      drained: since(after.find((a) => a.loading === 0)),
    };
    return {
      passes,
      loads,
      path: path.name,
      frames: moving.length,
      load,
      ...pacing(moving.map((a) => a.t)),
      cpu: [0.5, 0.95, 1].map((p) => percentile(cpu, p)),
      placement: [0.5, 0.95].map((p) =>
        percentile(
          moving.map((a) => a.placement),
          p,
        ),
      ),
      longs: longsMoving.length,
      longest: Math.max(0, ...longsMoving),
      settle,
      settleCpu: Math.max(0, ...after.map((a) => a.ms)),
      settleLong: Math.max(0, ...longsAfter),
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
        const source =
          index.loadVTLayers()[
            index.sourceLayerCoder.decode(bucket.sourceLayerIndex)
          ];
        const field = layer.layout.get("text-field");
        names = [];
        for (let i = 0; i < bucket.symbolInstances.length; i++) {
          const f = source.feature(bucket.symbolInstances.get(i).featureIndex);
          const text = field.evaluate(
            { type: f.type, id: f.id, properties: f.properties, geometry: [] },
            {},
            tile.tileID.canonical,
          );
          names.push(text ? text.toString() : "");
        }
        texts.set(bucket, names);
      }
      for (let i = 0; i < names.length; i++) {
        if (!names[i]) continue;
        const o =
          placement.opacities[bucket.symbolInstances.get(i).crossTileID];
        const v = o
          ? Math.min(
              1,
              Math.max(0, o.text.opacity + (o.text.placed ? change : -change)),
            )
          : 0;
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
        if (since === null && opacity < 0.1) {
          labels.set(name, f);
          counts.drops++;
        } else if (since !== null && opacity >= 0.9) {
          const gone = f - since;
          if (gone <= FLASH_FRAMES) {
            counts.blinks++;
            counts.blinkFrames += gone;
          }
          if (gone <= RETURN_FRAMES) counts.returns++;
          labels.set(name, null);
        }
      }
      for (const [name, opacity] of shown)
        if (opacity >= 0.9 && !labels.has(name)) labels.set(name, null);
    },
  };
}

let glModule;
/** MapLibre GL JS, with its worker and the pmtiles protocol. */
export async function loadGl() {
  if (!glModule) {
    const [gl, { Protocol }, { default: workerUrl }] = await Promise.all([
      import("maplibre-gl"),
      import("pmtiles"),
      import("maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url"),
    ]);
    glModule = gl.default ?? gl;
    glModule.setWorkerUrl(workerUrl);
    glModule.addProtocol("pmtiles", new Protocol().tile);
  }
  return glModule;
}

/** `style`, a URL or a style, as GL JS takes it: with absolute URLs. */
export async function glStyle(style) {
  const base =
    typeof style === "string"
      ? new URL(style, location.href)
      : new URL(location.href);
  const doc =
    typeof style === "string"
      ? await (await fetch(style)).json()
      : structuredClone(style);
  const absolute = (url) => new URL(url, base).href;
  if (typeof doc.sprite === "string") doc.sprite = absolute(doc.sprite);
  if (doc.glyphs)
    doc.glyphs = absolute(doc.glyphs).replace(/%7B/g, "{").replace(/%7D/g, "}");
  for (const source of Object.values(doc.sources)) {
    if (source.url?.startsWith("pmtiles://"))
      source.url = `pmtiles://${absolute(source.url.slice(10))}`;
  }
  return doc;
}

/** Runs one path through GL JS on a fresh map; resolves to its row. */
async function runGlPath(container, style, center, path, show, wall) {
  const maplibregl = await loadGl();
  show(`${path.name}: loading (GL JS)`);
  const created = performance.now();
  // GL JS stops pitch at 60° unless told otherwise; Orbis goes to 85°.
  const map = new maplibregl.Map({
    container,
    style: await glStyle(style),
    ...path.camera(center, 0),
    maxPitch: 85,
    interactive: false,
    attributionControl: false,
  });
  try {
    await idle(map, path.name);
    const load = performance.now() - created;
    show(`${path.name}: moving (GL JS)`);
    const texts = new WeakMap();
    const counter = labelCounter();
    const times = [];
    const drawn = () => times.push(frameTime());
    map.on("render", drawn);
    const { longs, stop } = longTasks();
    performance.mark(`gestures:gl:${path.name}:start`);
    const at = pathClock(wall);
    for (let i = 1, f = 0; f < 1; i++) {
      f = at(i);
      map.jumpTo(path.camera(center, f));
      await nextFrame();
      counter.frame(i, glShown(map, maplibregl, texts));
    }
    map.off("render", drawn);
    stop();
    const ended = performance.now();
    performance.mark(`gestures:gl:${path.name}:end`);
    // A repaint ends in `idle` once nothing is left to do.
    const settled = idle(map, path.name);
    map.triggerRepaint();
    await settled;
    return {
      path: path.name,
      load,
      frames: times.length,
      ...pacing(times),
      ...counter.counts,
      settle: performance.now() - ended,
      longs: longs.length,
      longest: Math.max(0, ...longs.map((l) => l.duration)),
    };
  } finally {
    map.remove();
  }
}

/** The GL JS rows' Markdown report. */
function glReport({ rows, center, size, about, wall }) {
  return (
    [
      "### Gestures bench (page, MapLibre GL JS)",
      "",
      ...about.map((line) => `- ${line}`),
      `- ${size[0]}×${size[1]} CSS px at ${center[0]}, ${center[1]}; ${pace(wall)}` +
        " (`jumpTo`), each path from a fresh map at rest; flights left out",
      "",
      `| path | load to idle ms | frames | ${PACING_HEAD} | long tasks (max ms) | label blinks (frames) | label returns (drops) | idle after ms |`,
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
      ...rows.map(
        (r) =>
          `| ${r.path} | ${r.load.toFixed(0)} | ${r.frames} | ${pacingCells(r)} | ${r.longs} (${ms(r.longest)}) | ` +
          `${r.blinks} (${r.blinkFrames}) | ${r.returns} (${r.drops}) | ${r.settle.toFixed(0)} |`,
      ),
      "",
      `Times in ms. ${PACING_NOTE} Long tasks: the page's, over 50 ms, while the camera moved. Label blinks: label texts drawn in full ` +
        `that all but vanished (opacity under 0.1) and were back in full within ${FLASH_FRAMES} frames, and the frames they were gone; ` +
        `label returns: back in full within ${RETURN_FRAMES} frames (1 s), blinks included, of the label texts drawn in full that all but vanished. ` +
        "Opacities are `style.placement`'s, moved on by the fade since its commit, over the tiles GL JS draws, read after each frame " +
        "on the page's thread (some 0.2 ms a frame at p95 on the Tokyo extract), which GL JS's pacing includes.",
    ].join("\n") + "\n"
  );
}

/**
 * Runs every path of `paths` (a key of PATH_SETS) in each of `engines`
 * (`orbis`, `gl`; GL JS leaves flights out), interleaved: path by path,
 * the engines' order turning each path, so a busy spell on the machine
 * falls on both. Resolves to each engine's Markdown report.
 */
export async function runGestures({
  style,
  center,
  paths,
  jobWorkers,
  size,
  about,
  show,
  engines = ["orbis"],
  gpu = false,
  wall = false,
}) {
  const run = PATH_SETS[paths];
  if (!run)
    throw new Error(
      `paths must be one of ${Object.keys(PATH_SETS).join(", ")}`,
    );
  for (const engine of engines)
    if (!["orbis", "gl"].includes(engine))
      throw new Error(`engine must be orbis or gl, not ${engine}`);
  const container = document.createElement("div");
  Object.assign(container.style, {
    width: `${size[0]}px`,
    height: `${size[1]}px`,
    position: "relative",
  });
  document.body.append(container);
  const rows = { orbis: [], gl: [] };
  try {
    for (const [i, path] of run.entries()) {
      for (const engine of i % 2 ? [...engines].reverse() : engines) {
        if (engine === "orbis")
          rows.orbis.push(
            await runPath(container, style, jobWorkers, center, path, show, {
              gpu,
              wall,
            }),
          );
        else if (!path.flight)
          rows.gl.push(
            await runGlPath(container, style, center, path, show, wall),
          );
      }
    }
  } finally {
    container.remove();
  }
  const report = { orbis: orbisReport, gl: glReport };
  return engines
    .map((engine) =>
      report[engine]({
        rows: rows[engine],
        center,
        size,
        about,
        jobWorkers,
        wall,
      }),
    )
    .join("\n");
}

/** The Orbis rows' Markdown report. */
function orbisReport({ rows, center, size, about, jobWorkers, wall }) {
  const lines = [
    "### Gestures bench (page)",
    "",
    ...about.map((line) => `- ${line}`),
    `- ${size[0]}×${size[1]} CSS px at ${center[0]}, ${center[1]}; ${pace(wall)}` +
      " (`jumpTo`; a flight's legs through `flyTo`), each path from a fresh map at rest; " +
      `${jobWorkers ?? "default"} job workers`,
    "",
    `| path | load to idle ms | frames | ${PACING_HEAD} | render CPU p50 | p95 | max | long tasks (max ms) | idle after ms | after: render CPU max | page long task max | placement p50 | p95 | loads in flight max | at end | loaded after ms | drained after ms |`,
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ...rows.map(
      (r) =>
        `| ${r.path} | ${r.load.toFixed(0)} | ${r.frames} | ${pacingCells(r)} | ` +
        `${r.cpu.map(ms).join(" | ")} | ${r.longs} (${ms(r.longest)}) | ${r.settle.toFixed(0)} | ${ms(r.settleCpu)} | ${ms(r.settleLong)} | ${r.placement.map(ms).join(" | ")} | ` +
        `${r.loads.max} | ${r.loads.end} | ${whole(r.loads.loaded)} | ${whole(r.loads.drained)} |`,
    ),
    "",
    `Times in ms. ${PACING_NOTE} Render CPU: the frame's ` +
      "`render`; long tasks: the page's, over 50 ms, while the camera moved; idle after: " +
      "from the path's end until the map's `idle` event; after: the frames and the page's long tasks from the path's end to `idle`; " +
      "placement: the page thread's share of render CPU (building the placement worker's frame, applying commits; 0.2 ms); " +
      "loads in flight: the map's tile loads (`tilesLoading`), most while the camera moved and at its last frame; " +
      "loaded after: from the path's end to the first frame the map reports loaded (the view's tiles in, its labels placed); drained after: to the first with no load in flight (—: the map was idle with no frame after the path).",
  ];
  if (rows.some((r) => r.passes)) {
    const mean = (xs) =>
      xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
    const max = (xs) => (xs.length ? Math.max(...xs) : NaN);
    lines.push(
      "",
      "| path | frames rebuilt | cull mean | max | main mean | GPU frame mean | frames kept | cull mean | max | main mean | GPU frame mean |",
      "|---|---|---|---|---|---|---|---|---|---|---|",
      ...rows
        .filter((r) => r.passes)
        .map(
          (r) =>
            `| ${r.path} | ` +
            r.passes
              .map(
                (p) =>
                  `${p.frames} | ${ms(mean(p.cull))} | ${ms(max(p.cull))} | ${ms(mean(p.main))} | ${ms(mean(p.span))}`,
              )
              .join(" | ") +
            " |",
        ),
      "",
      "GPU passes (`gpu`: timestamp queries around every pass, which cost a little themselves) from the path's start to `idle`, " +
        "in ms: frames whose candidate lists were laid out again (a tile came or went), where the cull rewrites every indexed node's " +
        "compacted indices, and frames that kept them.",
    );
  }
  return `${lines.join("\n")}\n`;
}
