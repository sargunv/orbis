// The frame bench in the browser (`bench.html`, the default experiment):
// `orb-bench`'s views through the orbis package. One map visits each view,
// waits for `idle`, then draws `rounds` runs of `warmup` + `frames` repaints
// of the still view, a display frame each, timing the GPU's passes with
// timestamp queries (`setGpuTiming`) and the page's render CPU per frame.
import { createMap } from "orbis";

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

/** The median, p90 and mean of `xs`, or null without any. */
function summary(xs) {
  if (!xs.length) return null;
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

/** A repaint of the still view, and its stats. */
function repaint(map) {
  const drawn = map.once("render");
  map.triggerRepaint();
  return drawn;
}

/** Runs every view; resolves to the Markdown report. */
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
}) {
  const container = document.createElement("div");
  Object.assign(container.style, {
    width: `${size[0]}px`,
    height: `${size[1]}px`,
    position: "relative",
  });
  document.body.append(container);
  const views = [];
  const warnings = new Set();
  const [first] = VIEWS;
  const map = await createMap({
    container,
    style,
    interactive: false,
    pixelRatio: ratio,
    camera: { center, zoom: first[1], pitch: first[2], bearing: 0 },
  });
  map.on("warning", ({ message }) => warnings.add(message));
  let timestamps = false;
  try {
    for (const [name, zoom, pitch, buildings] of VIEWS) {
      show(`${name}: loading`);
      const loaded = idle(map, name);
      map.camera.jumpTo({ center, zoom, pitch, bearing: 0 });
      await loaded;
      // A style with extrusions of its own (the Globe entry's), or without
      // OpenMapTiles' buildings, draws as it is.
      const now = await map.getStyle();
      if (
        buildings &&
        now.sources.openmaptiles &&
        !now.layers.some((l) => l.type === "fill-extrusion")
      ) {
        const before = now.layers.find((l) => l.type === "symbol")?.id;
        const built = idle(map, name);
        await map.edit((tx) => tx.addLayer(BUILDINGS_3D, before));
        await built;
      }
      const s = {
        gpu: [],
        cull: [],
        main: [],
        extrusion: [],
        cpu: [],
        submit: [],
        stats: null,
      };
      map.setGpuTiming(true);
      for (let round = 0; round < rounds; round++) {
        show(`${name}: round ${round + 1} of ${rounds}`);
        const cpu = [];
        for (let f = 0; f < warmup + frames; f++) {
          s.stats = await repaint(map);
          cpu.push(s.stats.ms);
        }
        const submissions = await map.gpuTimings();
        if (submissions.length !== cpu.length) {
          throw new Error(
            `${submissions.length} GPU submissions for ${cpu.length} frames`,
          );
        }
        for (let f = warmup; f < cpu.length; f++) {
          const passes = submissions[f].passes;
          timestamps ||= passes.length > 0;
          s.gpu.push(submissions[f].span);
          for (const label of ["cull", "main", "extrusion"]) {
            const pass = passes.find(([l]) => l === label);
            if (pass) s[label].push(pass[1]);
          }
          s.cpu.push(cpu[f]);
          s.submit.push(submissions[f].cpu);
        }
      }
      map.setGpuTiming(false);
      views.push({ name, ...s });
    }
  } finally {
    map.remove();
    container.remove();
  }
  const [w, h] = [Math.round(size[0] * ratio), Math.round(size[1] * ratio)];
  const lines = [
    "### Frame bench (page)",
    "",
    ...about.map((line) => `- ${line}`),
    `- ${w}×${h} px (${size[0]}×${size[1]} @${ratio}x) at ${center[0]}, ${center[1]}; ${rounds} rounds × ${frames} ` +
      `repaints after ${warmup} warm-up per view, one a display frame`,
    timestamps
      ? "- GPU: timestamp queries around every pass (`setGpuTiming`), a frame's GPU time from its first pass's start to its last one's end"
      : "- GPU: **no timestamp queries: no GPU times**",
    "",
    "| view | GPU mean | p90 | cull pass | main pass | extrusion pass | render CPU | encode+submit CPU | draws | dispatches |",
    "|---|---|---|---|---|---|---|---|---|---|",
    ...views.map(
      (v) =>
        `| ${v.name} | ${ms(summary(v.gpu)?.mean)} | ${ms(summary(v.gpu)?.p90)} | ` +
        `${ms(summary(v.cull)?.mean)} | ${ms(summary(v.main)?.mean)} | ${ms(summary(v.extrusion)?.mean)} | ` +
        `${ms(summary(v.cpu)?.mean)} | ${ms(summary(v.submit)?.mean)} | ${v.stats.draws} | ${v.stats.dispatches} |`,
    ),
    "",
    "Times in ms, means over every measured frame: browsers coarsen timestamps (Chrome to 65.5 µs on a Mac) " +
      "and `performance.now()` (to 100 µs), which means don't bias. Render CPU: the frame's `render` on the page.",
    ...views.map(
      (v) =>
        `- ${v.name}: ${v.stats.tilesDrawn} tiles drawn, ${v.stats.candidates} candidates`,
    ),
    ...[...warnings].map((w) => `- warning: ${w}`),
  ];
  return `${lines.join("\n")}\n`;
}
