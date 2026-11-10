// Real mouse drags and wheel zooms on the demo in headless Chromium, with
// a Chrome trace of every process, and what the GPU process made of them:
// its main thread's busy time and long tasks, the intervals between its
// presents, and the render CPU and uploads of the map's frames. For
// stutter that lives in the GPU process rather than the map (ANGLE's
// waits on WebGL 2, #140).
//
//   pnpm drag [--scene=globe|manhattan] [--backend=webgl2|webgpu]
//     [--pkgs=a,b] [--runs=n] [--size=WxH] [--ratio=r] [--warm] [--deep]
//     [--trace=dir]
//
// `globe`: the demo's Globe entry, dragged and wheel-zoomed for about 16 s.
// `manhattan`: Liberty over Midtown at pitch 50, wheel-zoomed z11 to z14.
// `--pkgs` serves each build of `core.wasm` (copies of web/pkg) in turn,
// `--runs` times, interleaved. `--warm` makes the gestures once untimed
// first, so the timed pass takes its tiles from the HTTP cache. `--deep` traces the GPU process's decoder
// too (`--enable-gpu-service-tracing`) and names what its long tasks did.
import {
  cpSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

import { chromiumPlatform, serveDemo } from "./serve.mjs";

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.match(/^--([a-z]+)(?:=(.+))?$/);
    if (!m) throw new Error(`unknown argument ${a}`);
    return [m[1], m[2] ?? "1"];
  }),
);
const scene = args.scene ?? "globe";
const backend = args.backend ?? "webgl2";
const [width, height] = (args.size ?? "1440x900").split("x").map(Number);
const ratio = Number(args.ratio ?? 2);
const runs = Number(args.runs ?? 1);
const deep = !!args.deep;
const warm = !!args.warm;
const pkg = fileURLToPath(new URL("../pkg/", import.meta.url));
const pkgs = args.pkgs ? args.pkgs.split(",").map((p) => resolve(p)) : [null];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Drags from (x0, y0) by (dx, dy) over `ms`, a move a display frame. */
async function drag(page, [x0, y0], [dx, dy], ms) {
  await page.mouse.move(x0, y0);
  await page.mouse.down();
  const steps = Math.round(ms / 16);
  for (let i = 1; i <= steps; i++) {
    await page.mouse.move(x0 + (dx * i) / steps, y0 + (dy * i) / steps);
    await sleep(16);
  }
  await sleep(100);
  await page.mouse.up();
}

/** `n` wheel notches at (x, y), `ms` apart: negative `delta` zooms in. */
async function wheel(page, [x, y], delta, n, ms = 40) {
  await page.mouse.move(x, y);
  for (let i = 0; i < n; i++) {
    await page.mouse.wheel(0, delta);
    await sleep(ms);
  }
}

const GESTURES = {
  async globe(page) {
    const c = [width / 2, height / 2];
    await drag(page, c, [-400, 0], 1000);
    await sleep(300);
    await wheel(page, c, -100, 12);
    await sleep(600);
    await drag(page, c, [300, 200], 1000);
    await sleep(300);
    await wheel(page, c, -100, 12);
    await sleep(600);
    await drag(page, c, [-300, 150], 1000);
    await sleep(300);
    await wheel(page, c, -100, 10);
    await sleep(600);
    await drag(page, c, [250, -250], 1000);
    await sleep(300);
    await wheel(page, c, 100, 24);
    await sleep(600);
    await drag(page, c, [-400, 0], 1000);
    await sleep(1000);
  },
  async manhattan(page) {
    const c = [width / 2, height / 2];
    for (let i = 0; i < 6; i++) {
      await wheel(page, c, -100, 4, 60);
      await sleep(700);
    }
    await drag(page, c, [-300, 0], 1000);
    await sleep(1500);
  },
};
const URLS = {
  globe: `?backend=${backend}`,
  manhattan: `?style=liberty&backend=${backend}#11/40.758/-73.9855/0/50`,
};

async function run(dir, at) {
  const { executablePath, args: flags } = chromiumPlatform();
  const browser = await chromium.launch({
    headless: true,
    executablePath,
    args: [
      "--enable-unsafe-webgpu",
      ...flags,
      ...(deep ? ["--enable-gpu-service-tracing"] : []),
    ],
  });
  try {
    const page = await browser.newPage({
      viewport: { width, height },
      deviceScaleFactor: ratio,
    });
    page.on("pageerror", (e) => console.error(`page: ${e.message}`));
    await page.addInitScript(() => {
      globalThis.__frames = [];
      addEventListener("orbis-frame", (e) => {
        const s = e.detail;
        globalThis.__frames.push({
          t: performance.now(),
          ms: s.ms,
          up: (s.uploadedFromWasm ?? 0) + (s.uploadedFromJs ?? 0),
          needs: s.needsFrame,
          compacted: [s.memory?.compactedUsed, s.memory?.compactedAllocated],
        });
      });
    });
    const open = async () => {
      await page.goto(`${at}${URLS[scene]}`);
      await page.waitForFunction(() => globalThis.orbis?.map, null, {
        timeout: 120_000,
      });
      const got = await page.evaluate(() => globalThis.orbis.map.backend);
      if (got !== backend) throw new Error(`map drew with ${got}`);
      await page.evaluate(
        () =>
          new Promise((r) => {
            globalThis.orbis.map.once("idle").then(r);
            setTimeout(r, 60_000);
          }),
      );
      await sleep(1000);
    };
    await open();
    if (warm) {
      // The same gestures once untimed, so the tiles come from the HTTP
      // cache in the timed pass, the same in every run.
      await GESTURES[scene](page);
      await open();
    }
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Tracing.start", {
      categories: [
        "toplevel",
        "gpu",
        "viz",
        ...(deep
          ? [
              "gpu.service",
              "gpu.angle",
              "disabled-by-default-gpu.service",
              "disabled-by-default-gpu.decoder",
              "disabled-by-default-gpu_decoder",
            ]
          : []),
      ].join(","),
      transferMode: "ReturnAsStream",
    });
    const started = await page.evaluate(() => performance.now());
    await GESTURES[scene](page);
    const ended = await page.evaluate(() => performance.now());
    const traced = new Promise((r) => cdp.once("Tracing.tracingComplete", r));
    await cdp.send("Tracing.end");
    const { stream } = await traced;
    const chunks = [];
    for (;;) {
      const { data, eof } = await cdp.send("IO.read", { handle: stream });
      chunks.push(data);
      if (eof) break;
    }
    const path = join(dir, `trace-${Date.now()}.json`);
    writeFileSync(path, chunks.join(""));
    const frames = (await page.evaluate(() => globalThis.__frames)).filter(
      (f) => f.t > started && f.t <= ended,
    );
    return { path, frames, seconds: (ended - started) / 1000 };
  } finally {
    await browser.close();
  }
}

/** The GPU process's main thread in `trace`: busy time, long tasks (and with
 * `--deep`, the events that took the most of them), present gaps. */
function analyse(path) {
  const d = JSON.parse(readFileSync(path, "utf8"));
  const events = Array.isArray(d) ? d : d.traceEvents;
  const meta = (name) => events.filter((e) => e.ph === "M" && e.name === name);
  const gpuPid = meta("process_name").find(
    (e) => e.args.name === "GPU Process",
  )?.pid;
  const main = meta("thread_name").find(
    (e) => e.pid === gpuPid && e.args.name === "CrGpuMain",
  )?.tid;
  const onMain = events.filter(
    (e) => e.pid === gpuPid && e.tid === main && e.ph === "X",
  );
  const tasks = onMain.filter(
    (e) => e.name === "ThreadControllerImpl::RunTask",
  );
  const busy = tasks.reduce((s, e) => s + e.dur, 0) / 1000;
  const long = tasks.filter((e) => e.dur > 50_000);
  const presents = events
    .filter((e) => e.name === "ImageTransportSurfaceOverlayMac::Present")
    .map((e) => e.ts)
    .sort((a, b) => a - b);
  const gaps = presents.slice(1).map((t, i) => (t - presents[i]) / 1000);
  const report = {
    busyMs: Math.round(busy),
    over50: long.length,
    longestMs: Math.round(Math.max(0, ...tasks.map((e) => e.dur)) / 1000),
    presents: presents.length,
    worstGapMs: Math.round(Math.max(0, ...gaps)),
    gapsOver50: gaps.filter((g) => g > 50).length,
  };
  if (deep) {
    // What took the most time inside tasks over 16 ms, by name.
    const slow = tasks.filter((e) => e.dur > 16_000);
    const by = {};
    for (const e of onMain) {
      if (e === undefined || e.name === "ThreadControllerImpl::RunTask")
        continue;
      if (!slow.some((t) => e.ts >= t.ts && e.ts < t.ts + t.dur)) continue;
      const b = (by[e.name] ??= { n: 0, ms: 0, max: 0 });
      b.n++;
      b.ms += e.dur / 1000;
      b.max = Math.max(b.max, e.dur / 1000);
    }
    report.inSlowTasks = Object.entries(by)
      .sort((a, b) => b[1].ms - a[1].ms)
      .slice(0, 16)
      .map(
        ([name, b]) =>
          `${name}: ${b.n}× ${b.ms.toFixed(0)} ms (max ${b.max.toFixed(1)})`,
      );
    // GL calls over the whole run, by name.
    const gl = {};
    for (const e of onMain) {
      if (!e.name.startsWith("TraceGLAPI::")) continue;
      const b = (gl[e.name.slice(12)] ??= { n: 0, ms: 0, max: 0 });
      b.n++;
      b.ms += e.dur / 1000;
      b.max = Math.max(b.max, e.dur / 1000);
    }
    report.gl = Object.entries(gl)
      .sort((a, b) => b[1].ms - a[1].ms)
      .slice(0, 10)
      .map(
        ([name, b]) =>
          `${name}: ${b.n}× ${b.ms.toFixed(0)} ms (max ${b.max.toFixed(1)})`,
      );
    // The GPU main thread's work within the worst present gaps.
    report.worstGaps = gaps
      .map((g, i) => ({ g, from: presents[i], to: presents[i + 1] }))
      .sort((a, b) => b.g - a.g)
      .slice(0, 4)
      .map(({ g, from, to }) => {
        const inside = tasks.filter((t) => t.ts < to && t.ts + t.dur > from);
        const busy = inside.reduce((s, t) => s + t.dur, 0) / 1000;
        const longest = inside.reduce((m, t) => Math.max(m, t.dur), 0) / 1000;
        return `${g.toFixed(0)} ms: GPU main busy ${busy.toFixed(0)} ms, longest task ${longest.toFixed(0)} ms`;
      });
  }
  return report;
}

/** Intervals between the map's frames that follow one needing another:
 * the worst, and those over 50 ms. */
function pageGaps(frames) {
  const gaps = [];
  for (let i = 1; i < frames.length; i++)
    if (frames[i - 1].needs) gaps.push(frames[i].t - frames[i - 1].t);
  return {
    worstFrameGapMs: Math.round(Math.max(0, ...gaps)),
    frameGapsOver50: gaps.filter((g) => g > 50).length,
  };
}

const pct = (xs, p) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length
    ? s[Math.min(s.length - 1, Math.round((s.length - 1) * p))]
    : 0;
};

if (args.analyse) {
  console.log(JSON.stringify(analyse(args.analyse), null, 1));
  process.exit(0);
}
const dir = args.trace ?? mkdtempSync(join(tmpdir(), "orbis-drag-"));
mkdirSync(dir, { recursive: true });
const saved = mkdtempSync(join(tmpdir(), "orbis-pkg-"));
cpSync(pkg, saved, { recursive: true });
try {
  for (let r = 0; r < runs; r++) {
    for (const p of pkgs) {
      if (p) cpSync(p, pkg, { recursive: true });
      let server;
      try {
        server = await serveDemo();
      } finally {
        cpSync(saved, pkg, { recursive: true });
      }
      try {
        const { path, frames, seconds } = await run(dir, server.url);
        const ms = frames.map((f) => f.ms);
        console.log(
          JSON.stringify({
            pkg: p ?? "web/pkg",
            run: r,
            scene,
            backend,
            seconds: +seconds.toFixed(1),
            ...analyse(path),
            frames: frames.length,
            renderP99: +pct(ms, 0.99).toFixed(1),
            ...pageGaps(frames),
            compactedMiB: frames
              .at(-1)
              ?.compacted.map((b) => +(b / (1 << 20)).toFixed(1)),
            renderTotalS: +(ms.reduce((a, b) => a + b, 0) / 1000).toFixed(2),
            uploadedMiB: Math.round(
              ((frames.at(-1)?.up ?? 0) - (frames[0]?.up ?? 0)) / (1 << 20),
            ),
          }),
        );
        if (!args.trace) rmSync(path);
      } finally {
        server.close();
      }
    }
  }
} finally {
  rmSync(saved, { recursive: true });
  if (!args.trace) rmSync(dir, { recursive: true });
}
