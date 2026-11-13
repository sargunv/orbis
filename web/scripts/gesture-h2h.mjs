// Gestures head to head: the same mouse drags, flings, right-drag
// rotations and wheel zooms on Orbis's demo and on mln-rs's browser page
// (its examples/web, as its `just web` builds it into examples/web/dist),
// both on live Liberty from the same camera, at 1280x800 @2x on WebGPU.
//
// In headless Chromium, under a Chrome trace: for each gesture, the GPU
// process's presents from its start until its glide or ease is over (their
// interval's median and p95, the worst, and how many came over 25 and 50 ms
// apart), when the first came, and the busiest page and worker threads'
// time within it. `--warm` makes the gestures once untimed first, so the timed pass
// takes its tiles from the HTTP cache. Each gesture's camera trajectory
// goes beside it: every 100 ms from its start, the change in zoom,
// bearing, pitch and centre (pixels at the start's zoom, x east and y
// south), read from each page's camera as its frames report it.
//
// `--browser=firefox` makes them in the installed Firefox over Marionette
// (firefox.mjs), headed, on a fresh profile, input through Firefox's widget
// layer: for each gesture, how long Firefox took to dispatch it, and from
// its composition recorder the composites that showed a new map frame, with
// their intervals (and stale frames, Firefox bug 2091844).
//
//   node scripts/gesture-h2h.mjs --mlnrs=<dist> [--engines=orbis,mlnrs]
//     [--scene=paris|manhattan] [--runs=n] [--warm] [--browser=firefox]
import { resolve } from "node:path";

import { chromium } from "playwright";

import { frameReport, openFirefox } from "./firefox.mjs";
import { chromiumPlatform, serveDemo } from "./serve.mjs";

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.match(/^--([a-z]+)(?:=(.+))?$/);
    if (!m) throw new Error(`unknown argument ${a}`);
    return [m[1], m[2] ?? "1"];
  }),
);
const engines = (args.engines ?? "orbis,mlnrs").split(",");
const scene = args.scene ?? "paris";
const runs = Number(args.runs ?? 1);
const warm = !!args.warm;
const inFirefox = args.browser === "firefox";
const [width, height] = [1280, 800];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const CAMERAS = {
  // zoom/lat/lon/bearing/pitch
  paris: "15/48.857/2.35/0/60",
  manhattan: "11/40.758/-73.9855/0/50",
};

async function mark(page, name) {
  await page.evaluate((n) => {
    performance.mark(n);
    globalThis.__marks[n] = performance.now();
  }, name);
}

/** Records each camera the page's map shows, as [time, lat, lon, zoom,
 * bearing, pitch]: mln-rs's map worker posts one per frame that moved it,
 * and Orbis's is read on each animation frame. */
function recordCameras() {
  globalThis.__cameras = [];
  globalThis.__marks = {};
  const Base = globalThis.Worker;
  globalThis.Worker = class extends Base {
    constructor(...args) {
      super(...args);
      this.addEventListener("message", (e) => {
        if (e.data?.type === "camera" && !globalThis.orbis)
          globalThis.__cameras.push([performance.now(), ...e.data.camera]);
      });
    }
  };
  const poll = () => {
    const camera = globalThis.orbis?.map?.camera?.current;
    if (camera) {
      const { center, zoom, bearing, pitch } = camera;
      const last = globalThis.__cameras.at(-1);
      const next = [center[1], center[0], zoom, bearing, pitch];
      if (!last || next.some((v, i) => v !== last[i + 1]))
        globalThis.__cameras.push([performance.now(), ...next]);
    }
    requestAnimationFrame(poll);
  };
  requestAnimationFrame(poll);
}

/** Each gesture's camera every 100 ms from its start, as changes from its
 * start: [ms, zoom, bearing, pitch, x px, y px]. */
function trajectories({ cameras, marks }) {
  const mercatorY = (lat) =>
    Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
  const at = (t) => {
    const i = cameras.findLastIndex(([time]) => time <= t);
    return cameras[Math.max(0, i)];
  };
  const out = {};
  for (const [name] of GESTURES) {
    const start = marks[`g:${name}:start`];
    const end = marks[`g:${name}:after`];
    if (start === undefined || end === undefined || !cameras.length) continue;
    const [, lat0, lon0, zoom0, bearing0, pitch0] = at(start);
    const scale = 512 * 2 ** zoom0;
    const steps = [];
    for (let t = 0; start + t <= end + 1; t += 100) {
      const [, lat, lon, zoom, bearing, pitch] = at(start + t);
      const turn = ((((bearing - bearing0 + 180) % 360) + 360) % 360) - 180;
      steps.push(
        [
          t,
          zoom - zoom0,
          turn,
          pitch - pitch0,
          ((lon - lon0) / 360) * scale,
          ((mercatorY(lat0) - mercatorY(lat)) / (2 * Math.PI)) * scale,
        ].map((v) => +v.toFixed(2)),
      );
    }
    out[name] = steps;
  }
  return out;
}

/** A drag by (dx, dy) over `ms`, a move every 8 ms, released at speed. */
async function fling(page, [x0, y0], [dx, dy], ms, button = "left") {
  await page.mouse.move(x0, y0);
  await page.mouse.down({ button });
  const steps = Math.round(ms / 8);
  for (let i = 1; i <= steps; i++) {
    await page.mouse.move(x0 + (dx * i) / steps, y0 + (dy * i) / steps);
    await sleep(8);
  }
  await page.mouse.up({ button });
}

async function wheel(page, [x, y], delta, n, ms) {
  await page.mouse.move(x, y);
  for (let i = 0; i < n; i++) {
    await page.mouse.wheel(0, delta);
    await sleep(ms);
  }
}

/** Each gesture: its name, what it does, how long the map moves after it. */
const GESTURES = [
  ["pan", (p, c) => fling(p, c, [-500, 0], 500), 1200],
  ["panDiagonal", (p, c) => fling(p, c, [300, 250], 400), 1200],
  ["panBack", (p, c) => fling(p, c, [200, -250], 400), 1200],
  ["rotate", (p, c) => fling(p, c, [300, -60], 800, "right"), 600],
  ["wheelIn", (p, c) => wheel(p, c, -100, 8, 50), 400],
  ["wheelOut", (p, c) => wheel(p, c, 100, 8, 50), 400],
];

async function play(page, timed) {
  const c = [width / 2, height / 2 + 100];
  for (const [name, gesture, after] of GESTURES) {
    if (timed) await mark(page, `g:${name}:start`);
    await gesture(page, c);
    if (timed) await mark(page, `g:${name}:end`);
    await sleep(after);
    if (timed) await mark(page, `g:${name}:after`);
    await sleep(400);
  }
}

async function run(engine, base) {
  const { executablePath, args: flags } = chromiumPlatform();
  const browser = await chromium.launch({
    headless: true,
    executablePath,
    args: ["--enable-unsafe-webgpu", ...flags],
  });
  try {
    const page = await browser.newPage({
      viewport: { width, height },
      deviceScaleFactor: 2,
    });
    page.on("pageerror", (e) => console.error(`${engine}: ${e.message}`));
    await page.addInitScript(recordCameras);
    const url =
      engine === "orbis"
        ? `${base}?style=liberty&backend=webgpu#${CAMERAS[scene]}`
        : `${base}mlnrs/index.html?style=liberty&backend=webgpu#${CAMERAS[scene]}`;
    const idle = () =>
      engine === "orbis"
        ? page.evaluate(
            () =>
              new Promise((r) => {
                globalThis.orbis.map.once("idle").then(r);
                setTimeout(r, 60_000);
              }),
          )
        : page.evaluate(() =>
            Promise.race([
              window.mln.idle(),
              new Promise((r) => setTimeout(r, 60_000)),
            ]),
          );
    const open = async () => {
      await page.goto(url);
      if (engine === "orbis")
        await page.waitForFunction(() => globalThis.orbis?.map, null, {
          timeout: 120_000,
        });
      else
        await page.waitForFunction(() => window.mln, null, {
          timeout: 120_000,
        });
      if (engine === "mlnrs") await page.evaluate(() => window.mln.loaded);
      await idle();
      await sleep(1000);
    };
    await open();
    if (warm) {
      await play(page, false);
      await open();
    }
    await page.mouse.click(width / 2, height / 2 + 100);
    await sleep(800);
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Tracing.start", {
      categories:
        "toplevel,gpu,viz,blink.user_timing,disabled-by-default-devtools.timeline",
      transferMode: "ReturnAsStream",
    });
    await play(page, true);
    const cameras = await page.evaluate(() => ({
      cameras: globalThis.__cameras,
      marks: globalThis.__marks,
    }));
    const traced = new Promise((r) => cdp.once("Tracing.tracingComplete", r));
    await cdp.send("Tracing.end");
    const { stream } = await traced;
    const chunks = [];
    for (;;) {
      const { data, eof } = await cdp.send("IO.read", { handle: stream });
      chunks.push(data);
      if (eof) break;
    }
    return { trace: JSON.parse(chunks.join("")), cameras };
  } finally {
    await browser.close();
  }
}

/** The gestures as WebDriver actions at canvas point `c`. */
const FIREFOX_GESTURES = {
  pan: (c) => flingActions(c, [-500, 0], 500),
  panDiagonal: (c) => flingActions(c, [300, 250], 400),
  panBack: (c) => flingActions(c, [200, -250], 400),
  rotate: (c) => flingActions(c, [300, -60], 800, 2),
  wheelIn: (c) => wheelActions(c, -100, 8, 50),
  wheelOut: (c) => wheelActions(c, 100, 8, 50),
};
function flingActions([x0, y0], [dx, dy], ms, button = 0) {
  const steps = Math.round(ms / 8);
  const moves = Array.from({ length: steps }, (_, i) => ({
    type: "pointerMove",
    duration: 8,
    origin: "viewport",
    x: Math.round(x0 + (dx * (i + 1)) / steps),
    y: Math.round(y0 + (dy * (i + 1)) / steps),
  }));
  return [
    {
      type: "pointer",
      id: "mouse",
      parameters: { pointerType: "mouse" },
      actions: [
        { type: "pointerMove", duration: 0, origin: "viewport", x: x0, y: y0 },
        { type: "pointerDown", button },
        ...moves,
        { type: "pointerUp", button },
      ],
    },
  ];
}
function wheelActions([x, y], delta, n, ms) {
  const actions = [];
  for (let i = 0; i < n; i++)
    actions.push(
      {
        type: "scroll",
        x,
        y,
        deltaX: 0,
        deltaY: delta,
        duration: 0,
        origin: "viewport",
      },
      { type: "pause", duration: ms },
    );
  return [{ type: "wheel", id: "wheel", actions }];
}

async function runFirefox(engine, base) {
  const page = await openFirefox({ width, height });
  try {
    const url =
      engine === "orbis"
        ? `${base}?style=liberty&backend=webgpu#${CAMERAS[scene]}`
        : `${base}mlnrs/index.html?style=liberty&backend=webgpu#${CAMERAS[scene]}`;
    await page.goto(url);
    if (engine === "orbis") {
      await page.waitForFunction(() => globalThis.orbis?.map, null, {
        timeout: 120_000,
      });
      await page.evaluate(
        () =>
          new Promise((r) => {
            globalThis.orbis.map.once("idle").then(r);
            setTimeout(r, 60_000);
          }),
      );
    } else {
      await page.waitForFunction(() => window.mln, null, { timeout: 120_000 });
      await page.evaluate(() => window.mln.loaded);
      await page.evaluate(() =>
        Promise.race([
          window.mln.idle(),
          new Promise((r) => setTimeout(r, 60_000)),
        ]),
      );
    }
    await sleep(1000);
    const content = await page.content();
    const canvas = await page.evaluate(() => {
      const r = [...document.querySelectorAll("canvas")]
        .map((c) => c.getBoundingClientRect())
        .sort((a, b) => b.width * b.height - a.width * a.height)[0];
      return { x: r.x, y: r.y, width: r.width, height: r.height };
    });
    const rect = {
      x: (content.x + canvas.x) * content.ratio,
      y: (content.y + canvas.y) * content.ratio,
      width: canvas.width * content.ratio,
      height: canvas.height * content.ratio,
    };
    const c = [
      Math.round(canvas.x + canvas.width / 2),
      Math.round(canvas.y + canvas.height / 2 + 100),
    ];
    const out = {};
    for (const [name, , after] of GESTURES) {
      const recording = await page.record();
      const started = Date.now();
      await page.perform(FIREFOX_GESTURES[name](c));
      const gestureMs = Date.now() - started;
      await sleep(after);
      const report = frameReport(await recording.stop(), rect);
      out[name] = {
        gestureMs,
        fresh: report.fresh,
        stale: report.stale,
        ...report.newFrameInterval,
      };
      await sleep(400);
    }
    return out;
  } finally {
    await page.close();
  }
}

const pct = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length
    ? s[Math.min(s.length - 1, Math.round((s.length - 1) * p))]
    : 0;
};

/** Each gesture's presents and its page's and workers' time in a Chrome
 * trace. */
function analyse(trace) {
  const events = Array.isArray(trace) ? trace : trace.traceEvents;
  const presents = events
    .filter((e) => e.name === "ImageTransportSurfaceOverlayMac::Present")
    .map((e) => e.ts)
    .sort((a, b) => a - b);
  const marks = new Map();
  const workers = new Set();
  for (const e of events) {
    if (e.name?.startsWith("g:") && !marks.has(e.name)) marks.set(e.name, e.ts);
    if (
      e.ph === "M" &&
      e.name === "thread_name" &&
      /Worker|CrRendererMain/.test(e.args.name)
    )
      workers.add(`${e.pid}:${e.tid}`);
  }
  // Their outermost tasks: [thread, start, end].
  const tasks = [];
  const ends = new Map();
  for (const e of events
    .filter((e) => e.ph === "X" && e.dur)
    .sort((a, b) => a.ts - b.ts)) {
    const thread = `${e.pid}:${e.tid}`;
    if (!workers.has(thread) || e.ts < (ends.get(thread) ?? 0)) continue;
    ends.set(thread, e.ts + e.dur);
    tasks.push([thread, e.ts, e.ts + e.dur]);
  }
  const out = {};
  for (const [name] of GESTURES) {
    const start = marks.get(`g:${name}:start`);
    const end = marks.get(`g:${name}:after`);
    if (start === undefined || end === undefined) continue;
    const inside = presents.filter((t) => t >= start && t <= end);
    const gaps = inside.slice(1).map((t, i) => (t - inside[i]) / 1000);
    const busy = new Map();
    for (const [thread, from, to] of tasks) {
      const ms = (Math.min(to, end) - Math.max(from, start)) / 1000;
      if (ms > 0) busy.set(thread, (busy.get(thread) ?? 0) + ms);
    }
    out[name] = {
      windowMs: Math.round((end - start) / 1000),
      frames: inside.length,
      firstMs: inside.length ? +((inside[0] - start) / 1000).toFixed(1) : null,
      p50: +pct(gaps, 0.5).toFixed(1),
      p95: +pct(gaps, 0.95).toFixed(1),
      worst: +Math.max(0, ...gaps).toFixed(1),
      over25: gaps.filter((g) => g > 25).length,
      over50: gaps.filter((g) => g > 50).length,
      // The two busiest threads: the map's (Orbis's page, mln-rs's map
      // worker) and its helpers'.
      busyMs: [...busy.values()]
        .sort((x, y) => y - x)
        .slice(0, 2)
        .map(Math.round),
    };
  }
  return out;
}

const mlnrs = args.mlnrs && resolve(args.mlnrs);
if (engines.includes("mlnrs") && !mlnrs)
  throw new Error("--mlnrs=<dist> is needed");
const server = await serveDemo({
  mounts: mlnrs ? { "mlnrs/": mlnrs + "/" } : {},
});
try {
  for (let r = 0; r < runs; r++)
    for (const engine of engines) {
      if (inFirefox) {
        const out = await runFirefox(engine, server.url);
        console.log(
          JSON.stringify({ engine, browser: "firefox", run: r, scene, ...out }),
        );
        continue;
      }
      const { trace, cameras } = await run(engine, server.url);
      const camera = trajectories(cameras);
      const out = analyse(trace);
      for (const name of Object.keys(out)) out[name].camera = camera[name];
      console.log(JSON.stringify({ engine, run: r, scene, warm, ...out }));
    }
} finally {
  server.close();
}
