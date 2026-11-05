// The gestures bench in the browser (`bench.html?experiment=gestures`):
// `orb-bench`'s rotate-and-tilt and zoom paths, driven through the orbis
// package's camera once a display frame. Where the
// native bench times `Map::render` alone, this times what a user sees: the
// interval between frames reaching the page, the render CPU time of each,
// main-thread long tasks, and how long the map takes to reach `idle` after
// the path ends (a backlog of frames shows there). `paths=alps` runs the
// terrain demo's view instead: rotate and pan at z11.6, pitch 72, and its
// 60 s flight (`data/alps-flight.json`), flown as the demo flies it.
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

/** Runs every path of `paths` (`city` or `alps`); resolves to the Markdown report. */
export async function runGestures({ style, center, paths, jobWorkers, size, about, show }) {
  const run = PATH_SETS[paths];
  if (!run) throw new Error("paths must be city or alps");
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
