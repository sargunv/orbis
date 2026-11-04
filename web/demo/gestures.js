// The gestures bench in the browser (`bench.html?experiment=gestures`):
// `orb-bench`'s rotate-and-tilt and zoom paths, driven through the orbis
// package's camera once a display frame, in each thread mode. Where the
// native bench times `Map::render` alone, this times what a user sees: the
// interval between frames reaching the page, the render CPU time of each,
// main-thread long tasks, and how long the map takes to reach `idle` after
// the path ends (a backlog of frames shows there).
import { createMap } from "orbis";

/** Frames per path: 6 s at 60 Hz. */
const PATH_FRAMES = 360;
const PATHS = [
  {
    name: "rotate and tilt at z14",
    camera: (center, f) => ({ center, zoom: 14, bearing: 360 * f, pitch: 30 - 30 * Math.cos(f * 2 * Math.PI) }),
  },
  {
    name: "zoom z12 → z16 → z12",
    camera: (center, f) => ({ center, zoom: 14 - 2 * Math.cos(f * 2 * Math.PI), bearing: 0, pitch: 0 }),
  },
];
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
async function runPath(container, style, thread, center, path, show) {
  const label = `${thread}: ${path.name}`;
  show(`${label}: loading`);
  const map = await createMap({ container, style, thread, camera: path.camera(center, 0), interactive: false });
  try {
    await idle(map, label);
    show(`${label}: moving`);
    const acks = [];
    const off = map.on("render", (ack) => acks.push({ t: performance.now(), ms: ack.ms }));
    const longs = [];
    const observer = typeof PerformanceObserver !== "undefined" &&
      PerformanceObserver.supportedEntryTypes?.includes("longtask")
      ? new PerformanceObserver((list) => { for (const e of list.getEntries()) longs.push(e.duration); })
      : null;
    observer?.observe({ type: "longtask" });
    const started = await nextFrame();
    for (let f = 1; f <= PATH_FRAMES; f++) {
      map.camera.jumpTo(path.camera(center, f / PATH_FRAMES));
      await nextFrame();
    }
    const ended = performance.now();
    await idle(map, label);
    const settle = performance.now() - ended;
    observer?.disconnect();
    off();
    const moving = acks.filter((a) => a.t > started && a.t <= ended);
    const gaps = moving.slice(1).map((a, i) => a.t - moving[i].t);
    const cpu = moving.map((a) => a.ms);
    return {
      thread: map.thread, path: path.name, frames: moving.length,
      interval: [0.5, 0.95, 0.99, 1].map((p) => percentile(gaps, p)),
      cpu: [0.5, 0.95, 1].map((p) => percentile(cpu, p)),
      missed: gaps.filter((g) => g > 50).length,
      longs: longs.length, longest: Math.max(0, ...longs),
      settle,
    };
  } finally {
    map.remove();
  }
}

/** Runs every path in each of `threads`; resolves to the Markdown report. */
export async function runGestures({ style, center, threads, size, about, show }) {
  const container = document.createElement("div");
  Object.assign(container.style, { width: `${size[0]}px`, height: `${size[1]}px`, position: "relative" });
  document.body.append(container);
  const rows = [];
  try {
    for (const thread of threads) {
      for (const path of PATHS) rows.push(await runPath(container, style, thread, center, path, show));
    }
  } finally {
    container.remove();
  }
  const lines = [
    "### Gestures bench (page)", "",
    ...about.map((line) => `- ${line}`),
    `- ${size[0]}×${size[1]} CSS px at ${center[0]}, ${center[1]}; ${PATH_FRAMES} frames a path, ` +
      "the camera moved once a display frame (`jumpTo`), each path from a fresh map at rest",
    "",
    "| thread | path | frames | interval p50 | p95 | p99 | max | gaps > 50 ms | render CPU p50 | p95 | max | long tasks (max ms) | idle after ms |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ...rows.map((r) => `| ${r.thread} | ${r.path} | ${r.frames} | ${r.interval.map(ms).join(" | ")} | ${r.missed} | ` +
      `${r.cpu.map(ms).join(" | ")} | ${r.longs} (${ms(r.longest)}) | ${r.settle.toFixed(0)} |`),
    "",
    "Times in ms. Interval: between frames reaching the page while the camera moves; render CPU: the frame's " +
      "`render` where it drew (the render worker in worker mode); long tasks: the page's, over 50 ms; idle after: " +
      "from the path's end until the map's `idle` event.",
  ];
  return `${lines.join("\n")}\n`;
}
