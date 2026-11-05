// Plan §7.2's measurements in Chrome: input-to-photon latency for pan,
// pinch and rotate, DOM overlay skew, main-thread busy time and frames when
// idle, for Orbis and the pinned GL JS, on
// web/dist/measure.html. In headless Chromium, or in Chrome on Android (keep
// its screen on).
//
//   pnpm measure [--android [--physical] | --headed] [--runs=3]
//     [--configs=orbis,gljs] [--no-capture] [--no-annotation]
//   pnpm measure --pipelines [--runs=3] [--android [--physical]]
//
// `--pipelines` measures pipeline creation instead (#26): measure.html
// `?pipelines` (web/demo/pipelines.js) loads the Tokyo extract `--runs`
// times in a fresh profile (cold: the browser's shader cache empty) and as
// often again in that profile reopened (warm), reporting pipelines made and
// the GPU queue's wait from map creation to idle; on Android, cold after
// `chrome://gpuclean` empties its GPU disk cache, warm in a tab after.
//
// `--android` runs on this worktree's Android emulator (`scripts/android-serial`;
// `ANDROID_SERIAL` overrides); `--physical` on a phone that the caller assigned
// to this worktree with `devices-cli assign`, released after the measurement.
//
// `--headed` measures in a Chromium window on this machine's display, at
// its refresh rate, rather than headless.
//
// Method: DevTools dispatches touch gestures (8 ms apart, one second each)
// while it captures composited frames (Page.startScreencast, PNG). Each
// frame's blue circle (drawn by the map) is located; the page's log of where
// that circle should be after each input (from the camera on the page)
// gives the input it shows, so latency is the frame's swap time minus that
// input's timestamp, both on the browser's clock. The cyan square is the
// DOM annotation at the same point: its offset from the circle, as input
// time, is the overlay skew. Main busy time is DevTools' TaskDuration for the
// page's main thread over the load and over each gesture.
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";
import { decodePng } from "./png.mjs";
import { launchChromium, openAndroid, serveDemo } from "./serve.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const physical = args.includes("--physical");
const android = physical || args.includes("--android");
const runs = Number(option("runs", "3"));
const configs = option("configs", "orbis,gljs").split(",");
// Busy time and idle frames only, without frame capture (whose PNG encoding
// also loads the device).
const capture = !args.includes("--no-capture");
// Without the DOM annotation, for what it costs the main thread.
const annotation = !args.includes("--no-annotation");
const GESTURES = ["pan", "pinch", "rotate"];

/** The centroid of pixels matching `test`, or null. */
function centroid(image, test) {
  let [sx, sy, n] = [0, 0, 0];
  const { width, height, channels, pixels } = image;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * channels;
      if (test(pixels[i], pixels[i + 1], pixels[i + 2])) { sx += x; sy += y; n++; }
    }
  }
  return n >= 12 ? [sx / n + 0.5, sy / n + 0.5] : null;
}
const isBlue = (r, g, b) => r < 70 && g < 70 && b > 190;
const isCyan = (r, g, b) => r < 70 && g > 190 && b > 190;

const percentile = (values, p) => {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Touch points over time for a gesture around the viewport's centre. */
function gesture(kind, [cx, cy]) {
  return (t) => {
    // The circle starts 120 px right of the centre; each gesture keeps it
    // in view.
    if (kind === "pan") return [[cx + 100 - 300 * t, cy]];
    if (kind === "pinch") {
      const d = 60 + 30 * t;
      return [[cx - d, cy], [cx + d, cy]];
    }
    const angle = (t * Math.PI) / 2;
    const [dx, dy] = [150 * Math.cos(angle), 150 * Math.sin(angle)];
    return [[cx - dx, cy - dy], [cx + dx, cy + dy]];
  };
}

/** The gesture's progress shown by a circle position: monotonic in time. */
function parameter(kind, [x, y], [cx, cy]) {
  if (kind === "pan") return x;
  if (kind === "pinch") return Math.hypot(x - cx, y - cy);
  return Math.atan2(y - cy, x - cx);
}

/** The log with each run of equal positions cut to its last input: the
 * one a frame showing that position shows (gestures arm before they move). */
function monotonic(log) {
  return log.filter((e, i) => i === log.length - 1 || Math.abs(log[i + 1].p - e.p) > 1e-6);
}

/** The input time at which the logged circle position reached `value`. */
function inputTime(log, value) {
  for (let i = 1; i < log.length; i++) {
    const [a, b] = [log[i - 1], log[i]];
    if ((a.p - value) * (b.p - value) <= 0 && a.p !== b.p) {
      return a.t + ((value - a.p) / (b.p - a.p)) * (b.t - a.t);
    }
  }
  return NaN;
}

async function run(page, config, viewport) {
  const engine = config;
  // A session per page: Android's Chrome delivers synthetic touches only to
  // the first page a session navigates.
  const cdp = await page.context().newCDPSession(page);
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await cdp.send("Performance.enable");
  const busy = async () => (await cdp.send("Performance.getMetrics")).metrics
    .find((m) => m.name === "TaskDuration").value * 1000;
  const query = new URLSearchParams({ engine, ...(annotation ? {} : { annotation: "0" }) });
  await page.goto("about:blank");
  const before = await busy();
  await page.goto(`${server.url}measure.html?${query}`);
  // measure.html imports measure.js, which sets `window.measure`, after load.
  await page.waitForFunction(() => window.measure);
  const info = await page.evaluate(async () => {
    const m = await window.measure;
    return { engine: m.engine, loadMs: m.loadMs, timeOrigin: m.timeOrigin };
  });
  const loadBusy = (await busy()) - before;
  const idleQuiet = await page.evaluate(() => window.measure.then((m) => m.quiet(2000)));
  // The page's own viewport: a blank page's can differ on phones.
  Object.assign(viewport, await page.evaluate(() => ({ width: innerWidth, height: innerHeight })));
  const center = [viewport.width / 2, viewport.height / 2];
  const result = { config, loadMs: info.loadMs, loadBusyMs: loadBusy, idle: idleQuiet, gestures: {} };
  for (const kind of GESTURES) {
    await page.evaluate(() => window.measure.then(async (m) => { await m.home(); await new Promise((r) => setTimeout(r, 300)); m.reset(); }));
    const frames = [];
    // A session per capture: Android's Chrome stops sending frames to one
    // that captured before.
    const session = capture ? await page.context().newCDPSession(page) : null;
    session?.on("Page.screencastFrame", ({ data, metadata, sessionId }) => {
      frames.push({ data, metadata });
      session.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
    });
    await session?.send("Page.startScreencast", { format: "png", everyNthFrame: 1, maxWidth: Math.min(viewport.width, 800) });
    await sleep(300);
    const busyStart = await busy();
    const path = gesture(kind, center);
    const touches = (t) => path(t).map(([x, y], id) => ({ x, y, id, radiusX: 2, radiusY: 2, force: 1 }));
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: touches(0) });
    const started = performance.now();
    for (;;) {
      const t = (performance.now() - started) / 1000;
      if (t >= 1) break;
      await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: touches(t) });
      await sleep(8);
    }
    // Android's Chrome keeps a gesture open unless the end names its points.
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: touches(1) });
    await sleep(700);
    const gestureBusy = (await busy()) - busyStart;
    await session?.send("Page.stopScreencast");
    await session?.detach();
    const log = monotonic((await page.evaluate(() => window.measure.then((m) => m.log())))
      .map((e) => ({ t: e.t, p: parameter(kind, [e.x, e.y], center) })));
    const latencies = [];
    const skews = [];
    for (const { data, metadata } of frames) {
      const image = decodePng(Buffer.from(data, "base64"));
      const scale = metadata.deviceWidth / image.width;
      const toCss = (p) => p && [p[0] * scale, p[1] * scale - (metadata.offsetTop ?? 0)];
      const blue = toCss(centroid(image, isBlue));
      const cyan = toCss(centroid(image, isCyan));
      if (!blue) continue;
      const value = parameter(kind, blue, center);
      // Not moved yet: which input it shows is undetermined.
      if (!log.length || Math.abs(value - log[0].p) < (kind === "rotate" ? 0.005 : 0.5)) continue;
      const shown = inputTime(log, value);
      if (!Number.isFinite(shown)) continue;
      latencies.push(metadata.timestamp * 1000 - (info.timeOrigin + shown));
      if (cyan) {
        const annotation = inputTime(log, parameter(kind, cyan, center));
        if (Number.isFinite(annotation)) skews.push(annotation - shown);
      }
    }
    result.gestures[kind] = {
      inputs: log.length, frames: frames.length, measured: latencies.length,
      latencyP50: percentile(latencies, 0.5), latencyP95: percentile(latencies, 0.95),
      skewP50: percentile(skews.map(Math.abs), 0.5), skewP95: percentile(skews.map(Math.abs), 0.95),
      skewMean: skews.reduce((a, b) => a + b, 0) / Math.max(skews.length, 1),
      busyMs: gestureBusy,
    };
    await page.evaluate(() => window.measure.then((m) => m.idle()));
  }
  result.afterQuiet = await page.evaluate(() => window.measure.then((m) => m.quiet(3000)));
  result.errors = errors;
  await cdp.detach();
  return result;
}

const server = await serveDemo();
if (args.includes("--pipelines")) {
  await pipelines(server);
  process.exit(0);
}
let browser, page, close;
if (android) {
  ({ browser, page, close } = await openAndroid(server, { physical }));
} else {
  browser = await launchChromium({ unsafe: false, headed: args.includes("--headed") });
  page = await browser.newPage({ viewport: { width: 800, height: 600 }, hasTouch: true });
  close = () => browser.close();
}
try {
  const viewport = await page.evaluate(() => ({ width: innerWidth, height: innerHeight, ratio: devicePixelRatio,
    refresh: null, ua: navigator.userAgent }));
  // The display's frame interval, from animation frames.
  viewport.frameMs = await page.evaluate(() => new Promise((resolve) => {
    const times = [];
    const step = (t) => { times.push(t); if (times.length < 61) requestAnimationFrame(step); else resolve((times[60] - times[0]) / 60); };
    requestAnimationFrame(step);
  }));
  console.error(`measure: ${viewport.ua}; ${viewport.width}×${viewport.height} @${viewport.ratio}, frame ${viewport.frameMs.toFixed(2)} ms`);
  const results = [];
  for (let r = 0; r < runs; r++) {
    for (const config of configs) {
      // Android's Chrome delivers synthetic touches to one page per tab
      // only: each config gets a tab of its own.
      if (android && results.length) {
        await close();
        ({ browser, page, close } = await openAndroid(server, { physical }));
      }
      const result = await run(page, config, viewport);
      console.error(`measure: run ${r + 1} ${config}: ${JSON.stringify(result)}`);
      results.push(result);
    }
  }
  // Medians over runs, per config and gesture.
  const median = (values) => percentile(values, 0.5);
  const fmt = (v) => (Number.isFinite(v) ? v.toFixed(1) : "–");
  const rows = [];
  for (const config of configs) {
    const mine = results.filter((r) => r.config === config);
    const g = (kind, key) => median(mine.map((r) => r.gestures[kind][key]));
    rows.push({
      config,
      loadBusy: median(mine.map((r) => r.loadBusyMs)),
      gestureBusy: median(mine.map((r) => GESTURES.reduce((s, k) => s + r.gestures[k].busyMs, 0))),
      idle: mine.map((r) => `${r.idle.rafs}/${r.idle.renders}, ${r.afterQuiet.rafs}/${r.afterQuiet.renders}`).join("; "),
      ...Object.fromEntries(GESTURES.flatMap((k) => [
        [`${k}P50`, g(k, "latencyP50")], [`${k}P95`, g(k, "latencyP95")],
        [`${k}Skew`, g(k, "skewP95")], [`${k}Frames`, g(k, "measured")],
      ])),
    });
  }
  console.log(`\n${viewport.ua}\n${viewport.width}×${viewport.height} CSS px @${viewport.ratio}, display frame ${viewport.frameMs.toFixed(2)} ms; medians of ${runs} runs\n`);
  console.log("| config | pan p50/p95 ms | pinch p50/p95 ms | rotate p50/p95 ms | skew p95 ms (pan/pinch/rotate) | main busy: load / gestures ms | idle frames (main rAF / map), before and after gestures |");
  console.log("|---|---|---|---|---|---|---|");
  for (const r of rows) {
    console.log(`| ${r.config} | ${fmt(r.panP50)} / ${fmt(r.panP95)} | ${fmt(r.pinchP50)} / ${fmt(r.pinchP95)} | ` +
      `${fmt(r.rotateP50)} / ${fmt(r.rotateP95)} | ${fmt(r.panSkew)} / ${fmt(r.pinchSkew)} / ${fmt(r.rotateSkew)} | ` +
      `${fmt(r.loadBusy)} / ${fmt(r.gestureBusy)} | ${r.idle} |`);
  }
  console.log(`\nframes measured per gesture: ${rows.map((r) => `${r.config} ${GESTURES.map((k) => r[`${k}Frames`]).join("/")}`).join("; ")}`);
  console.log(`\n${JSON.stringify(results)}`);
} finally {
  await close();
  server.close();
}

/** One load in a profile, the browser started for it and closed after. */
async function load(server, profile) {
  const context = await chromium.launchPersistentContext(profile, { args: ["--enable-features=Vulkan,WebGPU", "--use-angle=metal"], viewport: { width: 1024, height: 768 } });
  try {
    const page = context.pages()[0] ?? await context.newPage();
    await page.goto(`${server.url}measure.html?pipelines`);
    await page.waitForFunction(() => window.pipelines);
    return await page.evaluate(() => window.pipelines);
  } finally {
    await context.close();
  }
}

/** One load in a tab of Chrome on the device, after emptying its GPU cache if `clean`. */
async function loadAndroid(server, clean) {
  if (clean) {
    const { page, close } = await openAndroid(server, { physical });
    await page.goto("chrome://gpuclean").catch(() => {});
    await close();
  }
  const { page, close } = await openAndroid(server, { physical });
  try {
    await page.goto(`${server.url}measure.html?pipelines`);
    await page.waitForFunction(() => window.pipelines);
    return await page.evaluate(() => window.pipelines);
  } finally {
    await close();
  }
}

/** Pipeline creation, cold and warm (#26). */
async function pipelines(server) {
  const rows = [];
  try {
    for (let r = 0; r < runs; r++) {
      if (android) {
        const cold = await loadAndroid(server, true);
        const warm = await loadAndroid(server, false);
        rows.push({ cold, warm });
        console.error(`pipelines: run ${r + 1}: ${JSON.stringify({ cold, warm })}`);
        continue;
      }
      const profile = await mkdtemp(join(tmpdir(), "orbis-pipelines-"));
      try {
        const cold = await load(server, profile);
        const warm = await load(server, profile);
        rows.push({ cold, warm });
        console.error(`pipelines: run ${r + 1}: ${JSON.stringify({ cold, warm })}`);
      } finally {
        await rm(profile, { recursive: true, force: true });
      }
    }
  } finally {
    server.close();
  }
  const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  const f = (x) => x.toFixed(0);
  const { ua, adapter, shaders, render, compute, renderAsync, computeAsync } = rows[0].cold;
  console.log(`\n${ua}; ${adapter}; medians of ${runs} ${android ? "runs" : "fresh profiles"}\n`);
  console.log(`${shaders} shader modules, ${render} render and ${compute} compute pipelines ` +
    `(${renderAsync} and ${computeAsync} compiled asynchronously first)\n`);
  console.log("| | first draw / last pipeline / idle ms | GPU queue wait during load: total / largest / first submission ms | pipeline calls on the page ms |");
  console.log("|---|---|---|---|");
  for (const key of ["cold", "warm"]) {
    const m = (k) => median(rows.map((r) => r[key][k]));
    console.log(`| ${key} | ${f(m("firstDrawMs"))} / ${f(m("compiledMs"))} / ${f(m("idleMs"))} | ${f(m("waitTotalMs"))} / ${f(m("waitMaxMs"))} / ${f(m("firstWaitMs"))} | ${m("createMs").toFixed(1)} |`);
  }
}
