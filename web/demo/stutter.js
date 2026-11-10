// The stutter bench in the browser (`bench.html?experiment=stutter`):
// `orb-bench`'s stutter paths, the camera moved once a display frame
// through the orbis package. Per frame: the interval since the last one
// reached the page, `render`'s CPU time by stage (`FrameStats.times`), the
// work `prepare` did on the page between it and the last (applying tiles
// that arrived between frames), the bytes uploaded and the tiles that
// arrived; the GPU's submissions where timestamp queries exist. `engine=gl`
// drives MapLibre GL JS along the same paths and records its frame pacing,
// the CPU time of its `_render` and the tiles that load each frame;
// `engine=both` runs each path through both, interleaved after an untimed
// pass of each, and sets their pacing side by side.
import { createMap } from "orbis";

import alpsFlight from "./data/alps-flight.json";
import alps from "./data/alps.json?url";
import { glStyle, loadGl } from "./gestures.js";

const LOAD_TIMEOUT = 120_000;
const FRAME = 1000 / 60;
const WORST = 8;
/** The stage histogram's bucket edges, in ms. */
const BUCKETS = [0.25, 1, 2, 4, 8, 16.7, 33.3];
const MANHATTAN = [-73.9855, 40.758];
const MANHATTAN_SOUTH = [-73.9975, 40.731];
const TOKYO_STATION = [139.767, 35.681];
const ASIA = [105, 30];
const ATLANTIC = [-40, 30];
const ALPS = [7.7491, 46.0207];

const ease = (f) => f * f * (3 - 2 * f);
const lerp = (a, b, f) => a + (b - a) * f;

/** Liberty's buildings lit as the demo's Globe entry lights its own, ground and night side included (orb-bench's GLOBE_LIT). */
const lit = (tx) =>
  tx
    .setLight({
      anchor: "viewport",
      position: [1.15, 210, 60],
      intensity: 0.35,
      "dev.sargunv.orbis:shadow-intensity": 0.5,
      "dev.sargunv.orbis:ground-lighting": 1,
      "dev.sargunv.orbis:ambient-color": "#8d9fd6",
      "dev.sargunv.orbis:ambient-intensity": 0.5,
      "dev.sargunv.orbis:twilight": 14,
    })
    .setPaintProperty(
      "building-3d",
      "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity",
      0.6,
    )
    .setPaintProperty(
      "building-3d",
      "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-intensity",
      0.5,
    );

/** 6 s from the whole globe down to Midtown at z16, tilting to pitch 60 as the buildings come in, then 2 s turning. */
function globeManhattan(f) {
  const zoomIn = Math.min(1, f / 0.75),
    e = ease(zoomIn),
    g = ease(Math.min(1, zoomIn / 0.6));
  const turn = Math.min(1, Math.max(0, (f - 0.75) / 0.25));
  return {
    center: [
      lerp(ATLANTIC[0], MANHATTAN[0], g),
      lerp(ATLANTIC[1], MANHATTAN[1], g),
    ],
    zoom: lerp(1.5, 16, e),
    bearing: 29 + 30 * turn,
    pitch: lerp(0, 60, e),
  };
}

/** The native bench's scenarios (crates/orb-bench/src/stutter.rs). */
const SCENARIOS = [
  {
    name: "buildings: zoom z12 → z16 at pitch 60, then pan, Manhattan",
    style: "liberty",
    frames: 600,
    camera: (f) => {
      const pan = Math.min(1, Math.max(0, 2 * f - 1));
      return {
        center: [
          lerp(MANHATTAN[0], MANHATTAN_SOUTH[0], pan),
          lerp(MANHATTAN[1], MANHATTAN_SOUTH[1], pan),
        ],
        zoom: lerp(12, 16, ease(Math.min(1, 2 * f))),
        bearing: 29,
        pitch: 60,
      };
    },
  },
  {
    name: "buildings, shadows and AO: zoom z12 → z16 at pitch 60, then pan, Manhattan",
    style: "liberty",
    frames: 600,
    edit: (tx) =>
      tx
        .setLight({
          anchor: "viewport",
          position: [1.15, 210, 30],
          intensity: 0.5,
          "dev.sargunv.orbis:shadow-intensity": 0.35,
        })
        .setPaintProperty(
          "building-3d",
          "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity",
          0.5,
        )
        .setPaintProperty(
          "building-3d",
          "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-intensity",
          0.4,
        ),
    camera: (f) => SCENARIOS[0].camera(f),
  },
  {
    name: "terrain: fast turns at z11.6, pitch 80, Alps",
    style: "alps",
    frames: 360,
    camera: (f) => ({
      center: ALPS,
      zoom: 11.6,
      bearing: -32 + 3 * 360 * f,
      pitch: 80,
    }),
  },
  {
    name: "terrain: the Alps flight",
    style: "alps",
    flight: true,
    camera: () => ({ center: ALPS, zoom: 11.6, bearing: -32, pitch: 72 }),
  },
  {
    name: "globe: zoom z1.5 → z13 onto Tokyo",
    style: "liberty",
    globe: true,
    frames: 480,
    camera: (f) => {
      const e = ease(f),
        g = ease(Math.min(f, 0.6) / 0.6);
      return {
        center: [
          lerp(ASIA[0], TOKYO_STATION[0], g),
          lerp(ASIA[1], TOKYO_STATION[1], g),
        ],
        zoom: lerp(1.5, 13, e),
        bearing: 0,
        pitch: lerp(0, 40, e),
      };
    },
  },
  {
    name: "globe, buildings: zoom z1.5 → z16 onto Manhattan",
    style: "liberty",
    globe: true,
    frames: 480,
    camera: globeManhattan,
  },
  {
    name: "globe, buildings, shadows and AO: zoom z1.5 → z16 onto Manhattan",
    style: "liberty",
    globe: true,
    edit: lit,
    frames: 480,
    camera: globeManhattan,
  },
];

/** `FrameStats.times`' stages, in the report's order, and what the runtime's total leaves. */
const STAGES = [
  ["arrivals", "arrivals"],
  ["upload", "upload"],
  ["terrain", "terrain"],
  ["plan", "plan"],
  ["drape plan", "drape"],
  ["place", "place"],
  ["placement", "placement"],
  ["cull", "cull"],
  ["bundles", "bundles"],
  ["drape passes", "drapePasses"],
  ["encode", "encode"],
  ["submit", "submit"],
];
const STAGE_NAMES = [...STAGES.map(([name]) => name), "runtime other", "page"];

const sorted = (xs) => [...xs].sort((a, b) => a - b);
const at = (s, p) =>
  s.length ? s[Math.min(s.length - 1, Math.round((s.length - 1) * p))] : NaN;
const ms = (x) => (Number.isFinite(x) ? x.toFixed(2) : "—");
const mib = (bytes) => (bytes / 2 ** 20).toFixed(1);
const over = (xs, limit) => xs.filter((x) => x > limit).length;
const nextFrame = () =>
  new Promise((resolve) => requestAnimationFrame(resolve));

function idle(once, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () =>
        reject(new Error(`${label}: not idle after ${LOAD_TIMEOUT / 1000} s`)),
      LOAD_TIMEOUT,
    );
    once().then(() => {
      clearTimeout(timer);
      resolve();
    });
  });
}

/** Moves along `scenario` once a display frame (a flight: through `fly`); resolves when it ends. */
async function move(scenario, legs, jump, fly) {
  if (scenario.flight) {
    for (const leg of legs) await fly(leg);
  } else {
    for (let f = 1; f <= scenario.frames; f++) {
      jump(scenario.camera(f / scenario.frames));
      await nextFrame();
    }
  }
}

/** Long tasks while `observer` runs, or null where the browser doesn't report them. */
function longTasks() {
  const tasks = [];
  if (!PerformanceObserver.supportedEntryTypes?.includes("longtask"))
    return { tasks, stop() {} };
  const observer = new PerformanceObserver((list) => {
    for (const e of list.getEntries()) tasks.push(e.duration);
  });
  observer.observe({ type: "longtask" });
  return { tasks, stop: () => observer.disconnect() };
}

/** The GPU API Orbis's maps draw with: `runStutter`'s `backend`. */
let orbisBackend;

/** One scenario through the orbis package on a fresh map; resolves to its frames. */
async function runOrbis(container, scenario, style, legs, show) {
  show(`${scenario.name}: loading`);
  const map = await createMap({
    container,
    style,
    camera: scenario.camera(0),
    interactive: false,
    // The lit paths' shadows and ground lighting are previews, as in orb-bench.
    previews: "all",
    ...(orbisBackend && { backend: orbisBackend }),
  });
  try {
    await idle(() => map.once("idle"), scenario.name);
    if (scenario.globe || scenario.edit) {
      const loaded = idle(() => map.once("idle"), scenario.name);
      map.edit((tx) => {
        if (scenario.globe) tx.setProjection({ type: "globe" });
        scenario.edit?.(tx);
      });
      await loaded;
    }
    show(`${scenario.name}: moving`);
    const frames = [];
    // GPU timings each second, before the timestamp queries run out.
    const batches = [];
    let uploaded;
    const off = map.on("render", (s) => {
      if (frames.length % 60 === 59) batches.push(map.gpuTimings());
      const bytes =
        s.uploadedFromWasm +
        s.uploadedFromJs +
        s.texelsFromWasm +
        s.texelsFromJs;
      const t = s.times;
      const stages = STAGES.map(([, key]) => t[key]);
      const named = stages.reduce((a, b) => a + b, 0);
      stages.push(Math.max(0, t.total - named), Math.max(0, s.ms - t.total));
      frames.push({
        t: performance.now(),
        zoom: s.zoom,
        cpu: s.ms,
        stages,
        between: s.prepared.total,
        betweenStages: STAGES.map(([, key]) => s.prepared[key]),
        arrived: s.tilesArrived,
        uploaded: uploaded === undefined ? 0 : bytes - uploaded,
      });
      uploaded = bytes;
    });
    map.setGpuTiming(true);
    const timedFrom = frames.length;
    await map.gpuTimings();
    const long = longTasks();
    const started = await nextFrame();
    await move(
      scenario,
      legs,
      (camera) => map.camera.jumpTo(camera),
      (leg) => map.camera.flyTo(leg),
    );
    const ended = performance.now();
    long.stop();
    const timed = frames.slice(timedFrom);
    batches.push(map.gpuTimings());
    // One submission a frame has passes; the others only copy.
    const submissions = (await Promise.all(batches))
      .flat()
      .filter((s) => s.passes.length);
    map.setGpuTiming(false);
    const gpuMatched = submissions.length === timed.length;
    if (gpuMatched) {
      timed.forEach((f, i) => {
        f.passes = submissions[i].passes;
        f.gpu = f.passes.reduce((sum, [, d]) => sum + d, 0);
      });
    }
    const settled = performance.now();
    await idle(() => map.once("idle"), scenario.name);
    off();
    const moving = frames.filter((f) => f.t > started && f.t <= ended);
    const gpu = submissions.map((s) =>
      s.passes.reduce((sum, [, d]) => sum + d, 0),
    );
    return {
      frames: moving,
      longTasks: long.tasks,
      settle: performance.now() - settled,
      gpuMatched,
      gpu,
      timedFrames: timed.length,
    };
  } finally {
    map.remove();
  }
}

/** One scenario through GL JS on a fresh map; resolves to its frames. */
async function runGl(container, scenario, style, legs, show) {
  const maplibregl = await loadGl();
  show(`${scenario.name}: loading (GL JS)`);
  const json =
    style.startsWith("http") && !style.startsWith(location.origin)
      ? style
      : await glStyle(style);
  const map = new maplibregl.Map({
    container,
    style: json,
    ...scenario.camera(0),
    maxPitch: 85,
    interactive: false,
    attributionControl: false,
  });
  try {
    await idle(() => map.once("idle"), scenario.name);
    if (scenario.globe) {
      const loaded = idle(() => map.once("idle"), scenario.name);
      map.setProjection({ type: "globe" });
      map.triggerRepaint();
      await loaded;
    }
    show(`${scenario.name}: moving (GL JS)`);
    // `_render` is a frame's work on the page: its CPU time.
    const frames = [];
    let arrived = 0;
    const onData = (e) => {
      if (e.tile && e.dataType === "source") arrived++;
    };
    map.on("sourcedata", onData);
    const render = map._render.bind(map);
    map._render = (t) => {
      const begun = performance.now();
      const result = render(t);
      const end = performance.now();
      frames.push({ t: end, zoom: map.getZoom(), cpu: end - begun, arrived });
      arrived = 0;
      return result;
    };
    const long = longTasks();
    const started = await nextFrame();
    await move(
      scenario,
      legs,
      (camera) => map.jumpTo(camera),
      (leg) =>
        new Promise((resolve) => {
          map.once("moveend", resolve);
          map.flyTo(leg);
        }),
    );
    const ended = performance.now();
    long.stop();
    const settled = performance.now();
    const done = idle(() => map.once("idle"), scenario.name);
    map.triggerRepaint();
    await done;
    map.off("sourcedata", onData);
    return {
      frames: frames.filter((f) => f.t > started && f.t <= ended),
      longTasks: long.tasks,
      settle: performance.now() - settled,
    };
  } finally {
    map.remove();
  }
}

/** `names`' shares of the times `of` gives over `frames`. */
function share(frames, names, of) {
  const totals = names.map((name, i) => [
    name,
    frames.reduce((sum, f) => sum + of(f)[i], 0),
  ]);
  const all = totals.reduce((sum, [, t]) => sum + t, 0) || 1;
  return totals
    .sort((a, b) => b[1] - a[1])
    .filter(([, t]) => t / all >= 0.005)
    .map(([name, t]) => `${name} ${((100 * t) / all).toFixed(0)}%`)
    .join(", ");
}

/** Each stage's share of the frames over 16.7 ms of render CPU, and of the work between frames. */
function shares(frames) {
  const slow = frames.filter((f) => f.cpu > FRAME);
  const between = frames.reduce((sum, f) => sum + f.between, 0);
  return [
    slow.length
      ? `- the ${slow.length} frames over 16.7 ms of render CPU spent it on: ${share(slow, STAGE_NAMES, (f) => f.stages)}`
      : "- no frame's render took over 16.7 ms of CPU",
    `- between frames, \`prepare\` spent ${between.toFixed(0)} ms on: ${share(
      frames,
      STAGES.map(([name]) => name),
      (f) => f.betweenStages,
    )}`,
  ];
}

/** The slowest frames, by the interval up to them, with their stages. */
function worstTable(frames, gl) {
  const withGaps = frames
    .slice(1)
    .map((f, i) => ({ ...f, index: i + 2, interval: f.t - frames[i].t }));
  const worst = withGaps
    .sort((a, b) => Math.max(b.interval, b.cpu) - Math.max(a.interval, a.cpu))
    .slice(0, WORST);
  const columns = gl
    ? []
    : [
        "between frames",
        "GPU",
        "GPU's slowest passes",
        "KiB uploaded",
        ...STAGE_NAMES,
      ];
  const head = [
    "frame",
    "zoom",
    "interval",
    "render CPU",
    "tiles arrived",
    ...columns,
  ];
  return [
    `| ${head.join(" | ")} |`,
    `|${"---|".repeat(head.length)}`,
    ...worst.map((f) => {
      const cells = [
        f.index,
        f.zoom.toFixed(2),
        ms(f.interval),
        ms(f.cpu),
        f.arrived,
      ];
      if (!gl) {
        const passes = (f.passes ?? [])
          .slice()
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([label, d]) => `${label} ${d.toFixed(1)}`)
          .join(", ");
        cells.push(
          ms(f.between),
          ms(f.gpu),
          passes,
          (f.uploaded / 1024).toFixed(0),
          ...f.stages.map(ms),
        );
      }
      return `| ${cells.join(" | ")} |`;
    }),
  ];
}

/** Per stage, and for the work between frames, how many frames spent each
 * bucket's time on it, its p99, max and share of render CPU. */
function histogram(frames) {
  const head = [
    "stage",
    `< ${BUCKETS[0]}`,
    ...BUCKETS.slice(1).map((e, i) => `${BUCKETS[i]}–${e}`),
    `> ${BUCKETS.at(-1)}`,
    "p99",
    "max",
    "share",
  ];
  const all = frames.reduce((sum, f) => sum + f.cpu, 0) || 1;
  const rows = [
    ...STAGE_NAMES.map((name, i) => [name, (f) => f.stages[i]]),
    ["between frames", (f) => f.between],
  ];
  return [
    `| ${head.join(" | ")} |`,
    `|${"---|".repeat(head.length)}`,
    ...rows.map(([name, of]) => {
      const times = sorted(frames.map(of));
      const counts = Array.from({ length: BUCKETS.length + 1 }, () => 0);
      for (const t of times) counts[BUCKETS.filter((e) => t >= e).length]++;
      const total = times.reduce((a, b) => a + b, 0);
      return `| ${name} | ${counts.join(" | ")} | ${ms(at(times, 0.99))} | ${ms(at(times, 1))} | ${((100 * total) / all).toFixed(0)}% |`;
    }),
  ];
}

const ENGINES = {
  orbis: { name: "Orbis", run: runOrbis },
  gl: { name: "MapLibre GL JS", run: runGl },
};

/** Intervals between a path's frames, sorted. */
const intervals = (f) => sorted(f.slice(1).map((x, i) => x.t - f[i].t));

/** One engine's table and per-path details. */
function engineReport(rows, gl) {
  const lines = [
    `| path | frames | interval p50 | p99 | max | > 25 ms | > 33.3 ms | render CPU p50 | p99 | max | > 16.7 ms |${gl ? "" : " between frames p99 | max | GPU p50 | p99 | max | > 16.7 ms |"} tiles arrived |${gl ? "" : " MiB uploaded |"} long tasks (max ms) | settle ms |`,
    `|${"---|".repeat(gl ? 14 : 21)}`,
  ];
  for (const r of rows) {
    const f = r.frames;
    const gaps = intervals(f);
    const cpu = sorted(f.map((x) => x.cpu));
    // Each submission's, paired with its frame or not.
    const gpu = sorted(r.gpu ?? []);
    const cells = [
      r.scenario.name,
      f.length,
      ms(at(gaps, 0.5)),
      ms(at(gaps, 0.99)),
      ms(at(gaps, 1)),
      over(gaps, 1.5 * FRAME),
      over(gaps, 2 * FRAME),
      ms(at(cpu, 0.5)),
      ms(at(cpu, 0.99)),
      ms(at(cpu, 1)),
      over(cpu, FRAME),
    ];
    if (!gl) {
      const between = sorted(f.map((x) => x.between));
      cells.push(
        ms(at(between, 0.99)),
        ms(at(between, 1)),
        ms(at(gpu, 0.5)),
        ms(at(gpu, 0.99)),
        ms(at(gpu, 1)),
        over(gpu, FRAME),
      );
    }
    cells.push(f.reduce((sum, x) => sum + x.arrived, 0));
    if (!gl) cells.push(mib(f.reduce((sum, x) => sum + x.uploaded, 0)));
    cells.push(
      `${r.longTasks.length} (${ms(Math.max(0, ...r.longTasks))})`,
      r.settle.toFixed(0),
    );
    lines.push(`| ${cells.join(" | ")} |`);
  }
  lines.push(
    "",
    "Times in ms. Interval: between frames reaching the page while the camera moves (over 25 ms: a refresh missed; " +
      "a GPU running behind doesn't show here, the page's frames go on); render CPU: " +
      (gl ? "GL JS's `_render`" : "the frame's `render`") +
      "; long tasks: the page's, over 50 ms; settle: from the " +
      "path's end until the map is idle." +
      (gl
        ? ""
        : " Between frames: `prepare`'s work on the page since the frame " +
          "before (arrived tiles applied and asked for between frames); GPU: the frame's passes, where the device has timestamp " +
          "queries; uploaded and tiles arrived: since the frame before, between frames included."),
  );
  for (const r of rows) {
    lines.push("", `#### ${r.scenario.name}`, "", `- style: ${r.style}`);
    if (!gl) {
      lines.push(...shares(r.frames));
      if (!r.gpuMatched) {
        lines.push(
          `- GPU: ${r.gpu.length} submissions drew for ${r.timedFrames} frames, unpaired: the table's GPU ` +
            "columns are over the submissions, the worst frames have none",
        );
      }
    }
    lines.push("", ...worstTable(r.frames, gl));
    if (!gl) lines.push("", ...histogram(r.frames));
  }
  if (!gl) {
    lines.push(
      "",
      "Stages as `cargo xtask bench stutter` names them; page: the package's `render` around the runtime's frame.",
    );
  }
  return lines;
}

/** Each path's pacing through each engine, side by side. */
function comparison(results) {
  const lines = [
    "| path | engine | interval p50 | p99 | max | > 25 ms | > 33.3 ms | render CPU p99 | max | tiles arrived | long tasks |",
    `|${"---|".repeat(11)}`,
  ];
  results.orbis.forEach((_, i) => {
    for (const [key, rows] of Object.entries(results)) {
      const { scenario, frames, longTasks } = rows[i];
      const gaps = intervals(frames);
      const cpu = sorted(frames.map((x) => x.cpu));
      const cells = [
        scenario.name,
        ENGINES[key].name,
        ms(at(gaps, 0.5)),
        ms(at(gaps, 0.99)),
        ms(at(gaps, 1)),
        over(gaps, 1.5 * FRAME),
        over(gaps, 2 * FRAME),
        ms(at(cpu, 0.99)),
        ms(at(cpu, 1)),
        frames.reduce((sum, x) => sum + x.arrived, 0),
        longTasks.length,
      ];
      lines.push(`| ${cells.join(" | ")} |`);
    }
  });
  lines.push(
    "",
    "Times in ms, as in each engine's table below. Neither interval shows a GPU running behind: Orbis's GPU " +
      "columns below do; GL JS isn't GPU-timed here.",
  );
  return lines;
}

/** Runs each scenario `filter` keeps through `engine`: orbis, gl, or both
 * (per path an untimed pass of each, then each timed in turn, so the HTTP
 * cache, the shader caches and drift treat them alike); resolves to the
 * Markdown report. */
export async function runStutter({
  liberty,
  size,
  about,
  show,
  filter,
  legs,
  engine = "orbis",
  backend,
}) {
  orbisBackend = backend;
  const keys = engine === "both" ? ["orbis", "gl"] : [engine];
  if (!keys.every((key) => ENGINES[key]))
    throw new Error("engine must be orbis, gl or both");
  const container = document.createElement("div");
  Object.assign(container.style, {
    width: `${size[0]}px`,
    height: `${size[1]}px`,
    position: "relative",
  });
  document.body.append(container);
  const results = Object.fromEntries(keys.map((key) => [key, []]));
  const flight = alpsFlight.legs.slice(0, legs ?? alpsFlight.legs.length);
  try {
    // GL JS has no shadows or ambient occlusion: the paths that light the buildings are Orbis's alone.
    const chosen = SCENARIOS.filter(
      (s) =>
        (!filter || s.name.includes(filter)) &&
        !(s.edit && keys.includes("gl")),
    );
    for (const scenario of chosen) {
      const style = scenario.style === "alps" ? alps : liberty;
      if (keys.length > 1) {
        for (const key of keys) {
          await ENGINES[key].run(container, scenario, style, flight, (line) =>
            show(`${line}, untimed`),
          );
        }
      }
      for (const key of keys) {
        results[key].push({
          scenario,
          style,
          ...(await ENGINES[key].run(container, scenario, style, flight, show)),
        });
      }
    }
  } finally {
    container.remove();
  }
  if (!results[keys[0]].length)
    throw new Error("no scenario matches the filter");
  const both = keys.length > 1;
  const lines = [
    `### Stutter bench (page, ${keys.map((key) => ENGINES[key].name).join(" and ")})`,
    "",
    ...about.map((line) => `- ${line}`),
    `- ${size[0]}×${size[1]} CSS px; the camera moved once a display frame (\`jumpTo\`; a flight's legs through ` +
      "`flyTo`), each path from a fresh map at rest" +
      (both
        ? "; per path an untimed pass of each engine, then each timed in turn"
        : ""),
  ];
  if (both) lines.push("", ...comparison(results));
  for (const key of keys) {
    if (both) lines.push("", `### ${ENGINES[key].name}`);
    lines.push("", ...engineReport(results[key], key === "gl"));
  }
  return `${lines.join("\n")}\n`;
}
