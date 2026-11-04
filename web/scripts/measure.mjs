// Plan §7.2's measurements in Chrome: input-to-photon latency for pan,
// pinch and rotate, DOM overlay skew, main-thread busy time and frames when
// idle, for Orbis in worker and main-thread mode and the pinned GL JS, on
// web/dist/measure.html. In headless Chromium, or in Chrome on an Android
// device over adb (keep its screen on).
//
//   pnpm measure [--android[=serial]] [--runs=3] [--configs=orbis-worker,orbis-main,gljs]
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
import { inflateSync } from "node:zlib";
import { launchChromium, openAndroid, serveDemo } from "./serve.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const androidArg = args.find((a) => a === "--android" || a.startsWith("--android="));
const android = androidArg ? { serial: androidArg.split("=")[1] ?? null } : null;
const runs = Number(option("runs", "3"));
const configs = option("configs", "orbis-worker,orbis-main,gljs").split(",");
const GESTURES = ["pan", "pinch", "rotate"];

/** Decodes an 8-bit RGB or RGBA, non-interlaced PNG. */
function decodePng(buffer) {
  let offset = 8;
  let width, height, channels;
  const data = [];
  while (offset < buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("latin1", offset + 4, offset + 8);
    const body = buffer.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") {
      width = body.readUInt32BE(0);
      height = body.readUInt32BE(4);
      if (body[8] !== 8 || body[12] !== 0) throw new Error("unsupported PNG");
      channels = { 2: 3, 6: 4 }[body[9]];
      if (!channels) throw new Error(`unsupported PNG colour type ${body[9]}`);
    } else if (type === "IDAT") data.push(body);
    offset += 12 + length;
  }
  const raw = inflateSync(Buffer.concat(data));
  const stride = width * channels;
  const pixels = Buffer.alloc(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? pixels[y * stride + x - channels] : 0;
      const b = y > 0 ? pixels[(y - 1) * stride + x] : 0;
      const c = x >= channels && y > 0 ? pixels[(y - 1) * stride + x - channels] : 0;
      const p = a + b - c;
      const predictor = [0, a, b, (a + b) >> 1,
        Math.abs(p - a) <= Math.abs(p - b) && Math.abs(p - a) <= Math.abs(p - c) ? a : Math.abs(p - b) <= Math.abs(p - c) ? b : c][filter];
      pixels[y * stride + x] = (line[x] + predictor) & 255;
    }
  }
  return { width, height, channels, pixels };
}

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
      const d = 60 + 60 * t;
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

async function run(page, cdp, config, viewport) {
  const [engine, thread] = config.split("-");
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await cdp.send("Performance.enable");
  const busy = async () => (await cdp.send("Performance.getMetrics")).metrics
    .find((m) => m.name === "TaskDuration").value * 1000;
  const query = new URLSearchParams({ engine, ...(thread ? { thread } : {}) });
  await page.goto("about:blank");
  const before = await busy();
  await page.goto(`${server.url}measure.html?${query}`);
  const info = await page.evaluate(async () => {
    const m = await window.measure;
    return { engine: m.engine, thread: m.thread, loadMs: m.loadMs, timeOrigin: m.timeOrigin };
  });
  const loadBusy = (await busy()) - before;
  const idleQuiet = await page.evaluate(() => window.measure.then((m) => m.quiet(2000)));
  const center = [viewport.width / 2, viewport.height / 2];
  const result = { config, thread: info.thread, loadMs: info.loadMs, loadBusyMs: loadBusy, idle: idleQuiet, gestures: {} };
  for (const kind of GESTURES) {
    await page.evaluate(() => window.measure.then(async (m) => { await m.home(); await new Promise((r) => setTimeout(r, 300)); m.reset(); }));
    const frames = [];
    const onFrame = async ({ data, metadata, sessionId }) => {
      frames.push({ data, metadata });
      await cdp.send("Page.screencastFrameAck", { sessionId }).catch(() => {});
    };
    cdp.on("Page.screencastFrame", onFrame);
    await cdp.send("Page.startScreencast", { format: "png", everyNthFrame: 1, maxWidth: Math.min(viewport.width, 800) });
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
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await sleep(700);
    const gestureBusy = (await busy()) - busyStart;
    await cdp.send("Page.stopScreencast");
    cdp.off("Page.screencastFrame", onFrame);
    const log = (await page.evaluate(() => window.measure.then((m) => m.log())))
      .map((e) => ({ t: e.t, p: parameter(kind, [e.x, e.y], center) }));
    const latencies = [];
    const skews = [];
    for (const { data, metadata } of frames) {
      const image = decodePng(Buffer.from(data, "base64"));
      const scale = metadata.deviceWidth / image.width;
      const toCss = (p) => p && [p[0] * scale, p[1] * scale - (metadata.offsetTop ?? 0)];
      const blue = toCss(centroid(image, isBlue));
      const cyan = toCss(centroid(image, isCyan));
      if (!blue) continue;
      const shown = inputTime(log, parameter(kind, blue, center));
      if (!Number.isFinite(shown)) continue;
      latencies.push(metadata.timestamp * 1000 - (info.timeOrigin + shown));
      if (cyan) {
        const annotation = inputTime(log, parameter(kind, cyan, center));
        if (Number.isFinite(annotation)) skews.push(annotation - shown);
      }
    }
    result.gestures[kind] = {
      frames: frames.length, measured: latencies.length,
      latencyP50: percentile(latencies, 0.5), latencyP95: percentile(latencies, 0.95),
      skewP50: percentile(skews.map(Math.abs), 0.5), skewP95: percentile(skews.map(Math.abs), 0.95),
      skewMean: skews.reduce((a, b) => a + b, 0) / Math.max(skews.length, 1),
      busyMs: gestureBusy,
    };
    await page.evaluate(() => window.measure.then((m) => m.idle()));
  }
  result.afterQuiet = await page.evaluate(() => window.measure.then((m) => m.quiet(3000)));
  result.errors = errors;
  return result;
}

const server = await serveDemo();
let browser, page, close;
if (android) {
  ({ browser, page, close } = await openAndroid(server, android.serial));
} else {
  browser = await launchChromium({ unsafe: false });
  page = await browser.newPage({ viewport: { width: 800, height: 600 }, hasTouch: true });
  close = () => browser.close();
}
try {
  const cdp = await page.context().newCDPSession(page);
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
      const result = await run(page, cdp, config, viewport);
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
      thread: mine[0].thread,
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
    console.log(`| ${r.config} (${r.thread}) | ${fmt(r.panP50)} / ${fmt(r.panP95)} | ${fmt(r.pinchP50)} / ${fmt(r.pinchP95)} | ` +
      `${fmt(r.rotateP50)} / ${fmt(r.rotateP95)} | ${fmt(r.panSkew)} / ${fmt(r.pinchSkew)} / ${fmt(r.rotateSkew)} | ` +
      `${fmt(r.loadBusy)} / ${fmt(r.gestureBusy)} | ${r.idle} |`);
  }
  console.log(`\nframes measured per gesture: ${rows.map((r) => `${r.config} ${GESTURES.map((k) => r[`${k}Frames`]).join("/")}`).join("; ")}`);
  console.log(`\n${JSON.stringify(results)}`);
} finally {
  await close();
  server.close();
}
