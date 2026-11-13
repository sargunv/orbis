// The demo's Earthquake entry (`?style=earthquake`) with its ground held
// still and then rippling, the camera turning slowly through both so that
// every display frame draws: the intervals between frames reaching the
// page and between the GPU process's presents (Chromium), render CPU, GPU
// time per submission (timestamp queries), the steps of heights shown a
// second, the bytes uploaded a frame, and any memory made or moved while
// the camera moves.
//
//   pnpm quake [--browser=chromium|firefox] [--seconds=n] [--size=WxH]
//     [--ratio=r] [--query=backend=webgl2] [--shot=prefix]
//
// `--shot` keeps a screenshot of each phase's end as `<prefix>-still.png`
// and `<prefix>-quake.png` (Chromium).
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { openFirefox } from "./firefox.mjs";
import { launchChromium, serveDemo } from "./serve.mjs";

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.match(/^--([a-z]+)=(.+)$/s);
    if (!m) throw new Error(`unknown argument ${a}`);
    return [m[1], m[2]];
  }),
);
const seconds = Number(args.seconds ?? 10);
const [width, height] = (args.size ?? "1600x1000").split("x").map(Number);
const ratio = Number(args.ratio ?? 2);
const browserName = args.browser ?? "chromium";

const server = await serveDemo();
const firefox =
  browserName === "firefox" ? await openFirefox({ width, height }) : null;
const browser = firefox ? null : await launchChromium({ unsafe: false });
const dir = mkdtempSync(join(tmpdir(), "orbis-quake-"));
try {
  const page =
    firefox ??
    (await browser.newPage({
      viewport: { width, height },
      deviceScaleFactor: ratio,
    }));
  page.on?.("pageerror", (e) => console.error(`page: ${e.message}`));
  await page.goto(
    `${server.url}?style=earthquake${args.query ? `&${args.query}` : ""}`,
  );
  await page.waitForFunction(() => globalThis.orbis?.map, null, {
    timeout: 120_000,
  });
  // Tiles in and the quake running for a while before either phase.
  await page.waitForFunction(() => globalThis.quakeStats?.tiles > 0, null, {
    timeout: 120_000,
  });
  await page.evaluate(() => new Promise((r) => setTimeout(r, 5000)));
  const where = await page.evaluate(() => ({
    backend: globalThis.orbis.map.backend,
  }));
  const report = { browser: browserName, ...where, seconds };
  if (args.query) report.query = args.query;
  for (const phase of ["still", "quake"]) {
    const cdp =
      browserName === "chromium"
        ? await page.context().newCDPSession(page)
        : null;
    const trace = join(dir, `${phase}.json`);
    await page.evaluate(async (paused) => {
      globalThis.quake.pause(paused);
      // The last step in, and the ground settled.
      await new Promise((r) => setTimeout(r, 2000));
    }, phase === "still");
    await cdp?.send("Tracing.start", {
      categories: "toplevel,gpu",
      transferMode: "ReturnAsStream",
    });
    const run = await page.evaluate(measure, seconds * 1000);
    if (cdp) {
      const traced = new Promise((r) => cdp.once("Tracing.tracingComplete", r));
      await cdp.send("Tracing.end");
      const { stream } = await traced;
      const chunks = [];
      for (;;) {
        const { data, eof } = await cdp.send("IO.read", { handle: stream });
        chunks.push(data);
        if (eof) break;
      }
      writeFileSync(trace, chunks.join(""));
      run.presents = presents(trace);
    }
    if (args.shot && page.screenshot)
      await page.screenshot({ path: `${args.shot}-${phase}.png` });
    report[phase] = run;
  }
  console.log(JSON.stringify(report, null, 1));
} finally {
  await (firefox ?? browser).close();
  server.close();
  rmSync(dir, { recursive: true });
}

/**
 * On the page: turns the camera by 15° over `ms` and records every frame's
 * report and GPU submission meanwhile.
 */
async function measure(ms) {
  const map = globalThis.orbis.map;
  const frames = [];
  const listener = (e) => frames.push({ t: performance.now(), s: e.detail });
  addEventListener("orbis-frame", listener);
  map.setGpuTiming(true);
  await map.gpuTimings();
  const before = globalThis.quakeStats;
  const started = performance.now();
  const { bearing } = map.camera.current;
  // Turned a frame at a time: an easing function can't reach a worker.
  await new Promise((resolve) => {
    const turn = (now) => {
      const t = Math.min(1, (now - started) / ms);
      map.camera.jumpTo({ bearing: bearing + 15 * t });
      if (t < 1) requestAnimationFrame(turn);
      else resolve();
    };
    requestAnimationFrame(turn);
  });
  const ended = performance.now();
  removeEventListener("orbis-frame", listener);
  const submissions = await map.gpuTimings();
  map.setGpuTiming(false);
  const quake = globalThis.quakeStats;
  const stat = (xs) => {
    const s = [...xs].sort((a, b) => a - b);
    if (!s.length) return null;
    const at = (p) =>
      +s[Math.min(s.length - 1, Math.round((s.length - 1) * p))].toFixed(2);
    return {
      n: s.length,
      p50: at(0.5),
      p95: at(0.95),
      p99: at(0.99),
      max: at(1),
    };
  };
  const gaps = frames.slice(1).map((f, i) => f.t - frames[i].t);
  const sum = (f) => frames.reduce((n, x) => n + f(x.s), 0);
  return {
    frames: frames.length,
    fps: +((frames.length / (ended - started)) * 1000).toFixed(1),
    intervalMs: stat(gaps),
    over25: gaps.filter((g) => g > 25).length,
    renderCpuMs: stat(frames.map((f) => f.s.ms)),
    // Each stage's mean CPU a frame.
    stagesMs: Object.fromEntries(
      Object.keys(frames[0]?.s.times ?? {}).map((k) => [
        k,
        +(sum((s) => s.times[k]) / Math.max(1, frames.length)).toFixed(2),
      ]),
    ),
    gpuSpanMs: stat(submissions.map((s) => s.span)),
    uploadBytesPerFrame: stat(frames.map((f) => f.s.gpu.uploadBytes)),
    drapeSlotsPerFrame: stat(frames.map((f) => f.s.drapeSlots)),
    bytesAllocated: sum((s) => s.gpu.bytesAllocated),
    poolMoves: sum((s) => s.gpu.poolMoves),
    texturesCreated: sum((s) => s.gpu.texturesCreated),
    buffersCreated: sum((s) => s.gpu.buffersCreated),
    stepsShownPerSecond: quake?.shownPerSecond,
    heightsMiBPerSecond: quake && +quake.megabytesPerSecond.toFixed(1),
    quakeTiles: quake?.tiles,
    statsBefore: before,
  };
}

/** The intervals between the GPU process's presents in a Chromium trace. */
function presents(trace) {
  const d = JSON.parse(readFileSync(trace, "utf8"));
  const events = Array.isArray(d) ? d : d.traceEvents;
  const named = (name) =>
    events
      .filter((e) => e.name === name)
      .map((e) => e.ts)
      .sort((a, b) => a - b);
  const mac = named("ImageTransportSurfaceOverlayMac::Present");
  const ts = mac.length ? mac : named("Display::DrawAndSwap");
  const gaps = ts.slice(1).map((t, i) => (t - ts[i]) / 1000);
  const s = gaps.sort((a, b) => a - b);
  const at = (p) =>
    s.length
      ? +s[Math.min(s.length - 1, Math.round((s.length - 1) * p))].toFixed(1)
      : null;
  return {
    count: ts.length,
    p50: at(0.5),
    p95: at(0.95),
    max: at(1),
    over25: s.filter((g) => g > 25).length,
  };
}
