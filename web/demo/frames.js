// The frame bench in the browser (`bench.html`, the default experiment):
// `orb-bench`'s views through the orbis package. One map visits each view,
// waits for `idle`, then draws `rounds` runs of `warmup` + `frames` repaints
// of the still view, back to back (a display frame each, or as fast as the
// browser draws them where it runs unpaced), timing the GPU's passes with
// timestamp queries (`setGpuTiming`, where the backend has them), the
// render CPU and the interval between repaints, and counting draws.
// `engine=orbis,gl` adds MapLibre GL JS at the same views, the engines
// taking turns view by view: its render CPU is its `_render`, its draws the
// WebGL draw calls that makes.
import { createMap } from "orbis";

import { glStyle, instrumentGl, loadGl } from "./gestures.js";

/** Name, zoom, pitch, and whether the style gains `BUILDINGS_3D` (when
 * it has OpenMapTiles' buildings and no extrusions): the native bench's
 * views. */
const VIEWS = [
  ["z12", 12, 0, false],
  ["z15", 15, 0, false],
  ["z15 pitch 60", 15, 60, false],
  ["z17", 17, 0, false],
  ["z16 pitch 60, 3D buildings", 16, 60, true],
];
/** OpenFreeMap Liberty's extrusion layer, which the reduced style leaves
 * out, added before the labels. */
const BUILDINGS_3D = {
  id: "building-3d",
  type: "fill-extrusion",
  source: "openmaptiles",
  "source-layer": "building",
  minzoom: 14,
  filter: ["!", ["has", "hide_3d"]],
  paint: {
    "fill-extrusion-base": ["get", "render_min_height"],
    "fill-extrusion-color": "hsl(35,8%,85%)",
    "fill-extrusion-height": ["get", "render_height"],
    "fill-extrusion-opacity": 0.8,
  },
};
const LOAD_TIMEOUT = 120_000;
/** How long without a frame `settle` takes to mean none is due. */
const QUIET_MS = 250;

/** The median, p90 and mean of `xs`, or null without any. */
function summary(xs) {
  if (!xs?.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const at = (p) => s[Math.round((s.length - 1) * p)];
  return {
    median: at(0.5),
    p90: at(0.9),
    mean: s.reduce((a, b) => a + b, 0) / s.length,
  };
}
const ms = (x) => (x == null ? "—" : x.toFixed(3));

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

/**
 * Waits until no Orbis frame is drawing or due: `QUIET_MS` without one
 * after a repaint, with no pipeline compiling. At rest the map compiles
 * the style's other pipelines ahead of need and wakes for a frame as each
 * arrives (on a phone, through the first view's rounds). A frame drawn
 * before a repaint asked for one answers it, every later repaint then
 * takes its predecessor's frame, and the submissions, one per frame,
 * miscount the repaints by one.
 */
async function settle(map, repaint) {
  const until = performance.now() + LOAD_TIMEOUT;
  for (;;) {
    let last = await repaint();
    await new Promise((resolve) => {
      let timer;
      const quiet = () => {
        clearTimeout(timer);
        timer = setTimeout(() => {
          map.off("render", drawn);
          resolve();
        }, QUIET_MS);
      };
      const drawn = (stats) => {
        last = stats;
        quiet();
      };
      map.on("render", drawn);
      quiet();
    });
    if (last.pipelinesPending === 0) return;
    if (performance.now() > until) {
      throw new Error(`still compiling after ${LOAD_TIMEOUT / 1000} s`);
    }
  }
}

/** Whether a style draws as it is at a view that asks for 3D buildings: one with extrusions of its own (the Globe entry's), or without OpenMapTiles' buildings. */
function needsBuildings(style) {
  return (
    style.sources.openmaptiles &&
    !style.layers.some((l) => l.type === "fill-extrusion")
  );
}

/** Times `repaint` (a still view's frame, resolving to its CPU ms and draws) back to back: `rounds` × `warmup` + `frames`. */
async function repaints(repaint, { rounds, warmup, frames }, show) {
  const s = { cpu: [], interval: [], draws: [] };
  for (let round = 0; round < rounds; round++) {
    show(round);
    let last = null;
    for (let f = 0; f < warmup + frames; f++) {
      const stats = await repaint();
      const now = performance.now();
      if (f >= warmup) {
        s.cpu.push(stats.ms);
        s.draws.push(stats.draws);
        if (last !== null) s.interval.push(now - last);
      }
      last = now;
    }
  }
  return s;
}

/** The CSS pixels of a view's snapshot for its picture. */
const PICTURE_SIZE = [256, 192];
/** Snapshot pixels a picture's cell averages, on a side. */
const PICTURE_CELL = 8;

/** `image` as the mean RGB of each `PICTURE_CELL`-square cell, row by row:
 * coarse enough that two GPUs' antialiasing agree, fine enough that a
 * view drawn blank or with missing layers doesn't. */
function picture({ width, height, data }) {
  const cells = [];
  for (let cy = 0; cy + PICTURE_CELL <= height; cy += PICTURE_CELL) {
    for (let cx = 0; cx + PICTURE_CELL <= width; cx += PICTURE_CELL) {
      const sum = [0, 0, 0];
      for (let y = cy; y < cy + PICTURE_CELL; y++) {
        for (let x = cx; x < cx + PICTURE_CELL; x++) {
          const i = (y * width + x) * 4;
          for (let c = 0; c < 3; c++) sum[c] += data[i + c];
        }
      }
      cells.push(...sum.map((v) => Math.round(v / PICTURE_CELL ** 2)));
    }
  }
  return cells;
}

/** An Orbis map at `center`. */
async function orbisEngine(
  container,
  style,
  center,
  ratio,
  { backend, worker, keepAll },
) {
  // Half a zoom off the first view, so that its `jumpTo` moves the camera
  // and ends in `idle` however long the other engine took to start.
  const [zoom, pitch] = [VIEWS[0][1] - 0.5, VIEWS[0][2]];
  const map = await createMap({
    container,
    style,
    interactive: false,
    pixelRatio: ratio,
    camera: { center, zoom, pitch, bearing: 0 },
    // As the demo draws it, with every preview on (main.js).
    previews: "all",
    ...(backend && { backend }),
    ...(worker === false && { worker }),
  });
  if (keepAll) map.setKeepAllCull(true);
  const warnings = new Set();
  map.on("warning", ({ message }) => warnings.add(message));
  return {
    name: `Orbis (${map.backend}${keepAll ? ", the cull keeping every candidate" : ""})`,
    warnings,
    remove: () => map.remove(),
    async view(name, zoom, pitch, buildings) {
      const loaded = idle(map, name);
      map.camera.jumpTo({ center, zoom, pitch, bearing: 0 });
      await loaded;
      const now = await map.getStyle();
      if (buildings && needsBuildings(now)) {
        const before = now.layers.find((l) => l.type === "symbol")?.id;
        const built = idle(map, name);
        await map.edit((tx) => tx.addLayer(BUILDINGS_3D, before));
        await built;
      }
    },
    /** The view's snapshot as a `PICTURE` grid (`picture`). */
    async picture() {
      return picture(await map.snapshot(...PICTURE_SIZE, 1));
    },
    async measure(counts, show) {
      const s = { gpu: [], cull: [], main: [], extrusion: [], submit: [] };
      let stats = null;
      const repaint = async () => {
        const drawn = map.once("render");
        map.triggerRepaint();
        return (stats = await drawn);
      };
      map.setGpuTiming(true);
      await settle(map, repaint);
      // What timing caught while settling.
      await map.gpuTimings();
      const timed = await repaints(repaint, counts, show);
      // A backend without timestamp queries gives no submissions.
      const submissions = await map.gpuTimings();
      map.setGpuTiming(false);
      const each = counts.warmup + counts.frames;
      if (submissions.length && submissions.length !== counts.rounds * each) {
        throw new Error(
          `${submissions.length} GPU submissions for ${counts.rounds * each} frames`,
        );
      }
      for (const [f, { passes, span, cpu }] of submissions.entries()) {
        if (f % each < counts.warmup || !passes.length) continue;
        s.gpu.push(span);
        for (const [key, label] of [
          ["cull", "cull"],
          ["main", "main"],
          ["extrusion", "3d layer"],
        ]) {
          const pass = passes.find(([l]) => l === label);
          if (pass) s[key].push(pass[1]);
        }
        s.submit.push(cpu);
      }
      return { ...s, ...timed, stats };
    },
  };
}

/** A GL JS map at `center`, its draw calls counted. */
async function glEngine(container, style, center, ratio) {
  const maplibregl = await loadGl();
  // Half a zoom off the first view, so that its `jumpTo` moves the camera
  // and ends in `idle` however long the other engine took to start.
  const [zoom, pitch] = [VIEWS[0][1] - 0.5, VIEWS[0][2]];
  const map = new maplibregl.Map({
    container,
    style: await glStyle(style),
    center,
    zoom,
    pitch,
    bearing: 0,
    pixelRatio: ratio,
    interactive: false,
    attributionControl: false,
  });
  const gl = instrumentGl(map);
  const warnings = new Set();
  await idle(map, "GL JS");
  return {
    name: "GL JS",
    warnings,
    remove: () => map.remove(),
    async view(name, zoom, pitch, buildings) {
      const loaded = idle(map, name);
      map.jumpTo({ center, zoom, pitch, bearing: 0 });
      await loaded;
      const now = map.getStyle();
      if (buildings && needsBuildings(now)) {
        const before = now.layers.find((l) => l.type === "symbol")?.id;
        const built = idle(map, name);
        map.addLayer(BUILDINGS_3D, before);
        await built;
      }
    },
    async measure(counts, show) {
      const timed = await repaints(
        () =>
          new Promise((resolve) => {
            map.once("render", () => resolve(gl.last()));
            map.triggerRepaint();
          }),
        counts,
        show,
      );
      return { gpu: [], ...timed, stats: null };
    },
  };
}

/**
 * Runs every view (or those whose indices `views` lists) in each engine of
 * `engines` (`orbis`, `gl`), with the style's layers of the types `drop`
 * lists left out (what those families cost); resolves to the Markdown
 * report.
 */
export async function runFrames({
  style,
  center,
  size,
  ratio,
  frames,
  warmup,
  rounds,
  about,
  show,
  engines = ["orbis"],
  worker,
  backend,
  views,
  drop,
}) {
  if (drop?.length) {
    const doc = await glStyle(style);
    doc.layers = doc.layers.filter((l) => !drop.includes(l.type));
    style = doc;
    about.push(`layers left out: ${drop.join(", ")}`);
  }
  // The engines' maps lie on one another, the one measured on top: each
  // is shown, and so paced, as one map on its own would be. A canvas
  // scrolled out of view isn't presented, and its frames don't wait for
  // the GPU.
  const stack = document.createElement("div");
  Object.assign(stack.style, {
    width: `${size[0]}px`,
    height: `${size[1]}px`,
    position: "relative",
  });
  document.body.prepend(stack);
  const containers = new Map();
  const made = [];
  const counts = { frames, warmup, rounds };
  const results = new Map();
  // Each Orbis view's picture, for `bench.mjs` to check a phone's against
  // this machine's.
  globalThis.benchPictures = {};
  try {
    for (const engine of engines) {
      const container = document.createElement("div");
      Object.assign(container.style, {
        width: `${size[0]}px`,
        height: `${size[1]}px`,
        position: "absolute",
        inset: "0",
      });
      stack.append(container);
      show(`${engine}: starting`);
      if (engine === "orbis")
        made.push(
          await orbisEngine(container, style, center, ratio, {
            backend,
            worker,
          }),
        );
      else if (engine === "gl")
        made.push(await glEngine(container, style, center, ratio));
      else if (engine === "keep-all")
        made.push(
          await orbisEngine(container, style, center, ratio, {
            backend,
            worker,
            keepAll: true,
          }),
        );
      else
        throw new Error(`engine must be orbis, keep-all or gl, not ${engine}`);
      results.set(made.at(-1), []);
      containers.set(made.at(-1), container);
    }
    for (const [v, [name, zoom, pitch, buildings]] of VIEWS.entries()) {
      if (views && !views.includes(v)) continue;
      // The engines' order turns each view, so a busy spell falls on both.
      for (const engine of v % 2 ? [...made].reverse() : made) {
        show(`${engine.name}, ${name}: loading`);
        for (const [other, container] of containers)
          container.style.zIndex = other === engine ? "1" : "0";
        await engine.view(name, zoom, pitch, buildings);
        // Before the measured frames, whose GPU submissions are counted:
        // the snapshot's own lands while the view settles.
        if (engine.picture) {
          (globalThis.benchPictures[engine.name] ??= {})[name] =
            await engine.picture();
        }
        const s = await engine.measure(counts, (round) =>
          show(`${engine.name}, ${name}: round ${round + 1} of ${rounds}`),
        );
        results.get(engine).push({ name, ...s });
      }
    }
  } finally {
    for (const engine of made) engine.remove();
    stack.remove();
  }
  const [w, h] = [Math.round(size[0] * ratio), Math.round(size[1] * ratio)];
  const lines = [
    "### Frame bench (page)",
    "",
    ...about.map((line) => `- ${line}`),
    `- ${w}×${h} px (${size[0]}×${size[1]} @${ratio}x) at ${center[0]}, ${center[1]}; ${rounds} rounds × ${frames} ` +
      `repaints after ${warmup} warm-up per view, back to back`,
  ];
  for (const [engine, views] of results) {
    const timestamps = views.some((v) => v.gpu.length);
    const mean = (xs) => ms(summary(xs)?.mean);
    lines.push(
      "",
      `#### ${engine.name}`,
      "",
      timestamps
        ? "- GPU: timestamp queries around every pass (`setGpuTiming`), a frame's GPU time from its first pass's start to its last one's end"
        : "- GPU: **no timestamp queries: no GPU times**",
      "",
      "| view | interval mean | p50 | p90 | GPU mean | p90 | cull pass | main pass | extrusion pass | render CPU | encode+submit CPU | draws |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|",
      ...views.map(
        (v) =>
          `| ${v.name} | ${mean(v.interval)} | ${ms(summary(v.interval)?.median)} | ${ms(summary(v.interval)?.p90)} | ` +
          `${mean(v.gpu)} | ${ms(summary(v.gpu)?.p90)} | ${mean(v.cull)} | ${mean(v.main)} | ${mean(v.extrusion)} | ` +
          `${mean(v.cpu)} | ${mean(v.submit)} | ${summary(v.draws)?.median ?? "—"} |`,
      ),
      "",
      ...views
        .filter((v) => v.stats)
        .map(
          (v) =>
            `- ${v.name}: ${v.stats.tilesDrawn} tiles drawn, ${v.stats.candidates} candidates`,
        ),
      ...[...engine.warnings].map((w) => `- warning: ${w}`),
    );
  }
  lines.push(
    "",
    "Times in ms, means over every measured frame: browsers coarsen timestamps (Chrome to 65.5 µs on a Mac) " +
      "and `performance.now()` (to 100 µs), which means don't bias. Interval: from one repaint's `render` event on the page " +
      "to the next's, a display frame where the browser paces frames. Render CPU: Orbis's frame `render` (in its worker " +
      "where it runs in one), GL JS's `_render` on the page. Draws: Orbis's recorded draws, GL JS's WebGL draw calls; the median frame's.",
  );
  return `${lines.join("\n")}\n`;
}
