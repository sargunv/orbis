// Flies the demo's Alps flight (`data/alps-flight.json`, as `?style=alps&flight`
// flies it) once in headless Chromium and reports its frame pacing: the
// intervals between frames reaching the page and between frames the GPU
// process presents, the render CPU of each, and the slowest frames' counters.
//
//   pnpm flight [--engine=orbis|gljs] [--pkg=dir] [--trace=out.json]
//     [--size=WxH] [--ratio=r] [--legs=n]
//
// `--engine=gljs` flies the pinned MapLibre GL JS over the same style and
// legs instead. `--pkg` serves another build of `core.wasm` (a copy of
// web/pkg) for A/B runs. `--trace` keeps the Chrome trace the present
// intervals come from.
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { launchChromium, serveDemo } from "./serve.mjs";

const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.match(/^--([a-z]+)=(.+)$/);
  if (!m) throw new Error(`unknown argument ${a}`);
  return [m[1], m[2]];
}));
const engine = args.engine ?? "orbis";
const [width, height] = (args.size ?? "1600x1000").split("x").map(Number);
const ratio = Number(args.ratio ?? 2);
const data = (name) => fileURLToPath(new URL(`../demo/data/${name}`, import.meta.url));
const legs = JSON.parse(readFileSync(data("alps-flight.json"), "utf8")).legs.slice(0, Number(args.legs ?? 4));

const pkg = fileURLToPath(new URL("../pkg/", import.meta.url));
let restore;
if (args.pkg) {
  const saved = mkdtempSync(join(tmpdir(), "orbis-pkg-"));
  cpSync(pkg, saved, { recursive: true });
  cpSync(args.pkg, pkg, { recursive: true });
  restore = () => { cpSync(saved, pkg, { recursive: true }); rmSync(saved, { recursive: true }); };
}
let server;
try {
  server = await serveDemo({
    mounts: { "gl/": fileURLToPath(new URL("../node_modules/maplibre-gl/dist/", import.meta.url)) },
  });
} finally {
  restore?.();
}
const browser = await launchChromium({ unsafe: false });
const dir = mkdtempSync(join(tmpdir(), "orbis-flight-"));
const tracePath = args.trace ?? join(dir, "trace.json");
try {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: ratio });
  page.on("pageerror", (e) => console.error(`page: ${e.message}`));
  await page.addInitScript(() => {
    globalThis.__frames = [];
    addEventListener("orbis-frame", (e) => {
      const s = e.detail;
      globalThis.__frames.push({
        t: performance.now(), ms: s.ms, jobs: s.jobsDone, meshes: s.meshesUploaded,
        pending: s.pipelinesPending, placement: s.times.placement, loading: s.tilesLoading,
        bundle: s.bundleRecorded,
      });
    });
  });
  // Resolves when the map is idle; flies `legs` and resolves to the
  // flight's start and end on the page's clock.
  let fly;
  if (engine === "gljs") {
    const style = readFileSync(data("alps.json"), "utf8");
    // A page of the server's origin, for the module and its worker.
    await page.goto(`${server.url}gl/maplibre-gl.css`);
    await page.setContent(`<!doctype html><link rel="stylesheet" href="${server.url}gl/maplibre-gl.css">
      <style>html,body,#map{margin:0;width:100%;height:100%}</style><div id="map"></div>
      <script type="module">
        import * as maplibregl from "${server.url}gl/maplibre-gl.mjs";
        maplibregl.setWorkerUrl("${server.url}gl/maplibre-gl-worker.mjs");
        globalThis.maplibregl = maplibregl;
      </script>`);
    await page.waitForFunction(() => globalThis.maplibregl);
    await page.evaluate((style) => new Promise((resolve) => {
      const s = JSON.parse(style);
      const map = new maplibregl.Map({
        container: "map", style: s, center: s.center, zoom: s.zoom, bearing: s.bearing, pitch: s.pitch,
        maxPitch: 85, attributionControl: false,
      });
      globalThis.gl = map;
      map.on("render", () => globalThis.__frames.push({ t: performance.now() }));
      map.once("idle", resolve);
    }), style);
    fly = (legs) => page.evaluate(async (legs) => {
      const map = globalThis.gl;
      const started = await new Promise((r) => requestAnimationFrame(r));
      for (const leg of legs) {
        await new Promise((r) => { map.once("moveend", r); map.flyTo(leg); });
      }
      return { started, ended: performance.now() };
    }, legs);
  } else {
    await page.goto(`${server.url}?style=alps`);
    await page.waitForFunction(() => globalThis.orbis?.map, null, { timeout: 120_000 });
    await page.evaluate(() => new Promise((r) => globalThis.orbis.map.once("idle").then(r)));
    fly = (legs) => page.evaluate(async (legs) => {
      const map = globalThis.orbis.map;
      const started = await new Promise((r) => requestAnimationFrame(r));
      for (const leg of legs) await map.camera.flyTo(leg);
      return { started, ended: performance.now() };
    }, legs);
  }
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Tracing.start", {
    categories: "devtools.timeline,v8.gc,disabled-by-default-devtools.timeline,toplevel,gpu",
    transferMode: "ReturnAsStream",
  });
  const { started, ended } = await fly(legs);
  const traced = new Promise((r) => cdp.once("Tracing.tracingComplete", r));
  await cdp.send("Tracing.end");
  const { stream } = await traced;
  const chunks = [];
  for (;;) {
    const { data, eof } = await cdp.send("IO.read", { handle: stream });
    chunks.push(data);
    if (eof) break;
  }
  writeFileSync(tracePath, chunks.join(""));
  const frames = await page.evaluate(() => globalThis.__frames);
  const moving = frames.filter((f) => f.t > started && f.t <= ended);
  const gaps = moving.slice(1).map((f, i) => ({ gap: f.t - moving[i].t, frame: f }));
  const report = {
    engine, size: `${width}x${height}@${ratio}`, seconds: +((ended - started) / 1000).toFixed(1),
    page: { frames: moving.length, ...pacing(gaps.map((g) => g.gap)) },
    presents: presents(tracePath),
  };
  if (engine === "orbis") {
    report.renderMs = pacing(moving.map((f) => f.ms));
    report.slowest = gaps.sort((a, b) => b.gap - a.gap).slice(0, 8)
      .map(({ gap, frame }) => ({ ...frame, gap: +gap.toFixed(1), t: +(frame.t - started).toFixed(0) }));
  }
  console.log(JSON.stringify(report));
} finally {
  await browser.close();
  server.close();
  rmSync(dir, { recursive: true });
}

/** Percentiles of `xs`, and the counts over a 60 Hz frame, over one and a
 * half (a refresh missed whatever the jitter) and over two. */
function pacing(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const at = (p) => +s[Math.min(s.length - 1, Math.round((s.length - 1) * p))].toFixed(1);
  return {
    p50: at(0.5), p95: at(0.95), p99: at(0.99), max: at(1),
    over16: s.filter((x) => x > 16.7).length, over25: s.filter((x) => x > 25).length,
    over33: s.filter((x) => x > 33.4).length,
  };
}

/** The intervals between the GPU process's presents in `trace`, within the
 * flight (the trace spans it). */
function presents(trace) {
  const d = JSON.parse(readFileSync(trace, "utf8"));
  const events = Array.isArray(d) ? d : d.traceEvents;
  const named = (name) => events.filter((e) => e.name === name).map((e) => e.ts).sort((a, b) => a - b);
  const mac = named("ImageTransportSurfaceOverlayMac::Present");
  const ts = mac.length ? mac : named("Display::DrawAndSwap");
  const gaps = ts.slice(1).map((t, i) => (t - ts[i]) / 1000);
  return { count: ts.length, ...pacing(gaps.length ? gaps : [0]) };
}
