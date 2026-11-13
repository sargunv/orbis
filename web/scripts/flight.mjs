// Flies the demo's Alps flight (`data/alps-flight.json`, as `?style=alps&flight`
// flies it) once in headless Chromium and reports its frame pacing: the
// intervals between frames reaching the page, between the page's animation
// frames and between frames the GPU process presents, the render CPU of
// each, and the slowest frames' counters.
//
//   pnpm flight [--engine=orbis|gljs] [--pkg=dir] [--trace=out.json]
//     [--size=WxH] [--ratio=r] [--legs=n] [--gpu=1|out.json]
//     [--browser=chromium|firefox|webkit] [--query=backend=webgl2]
//     [--prefs=name=value,…] [--record=1] [--busy=ms[/period]]
//
// `--busy=30/100` has the page run 30 ms of JavaScript every 100 ms
// through the flight, as an app busy beside the map.
// `--browser=firefox` flies it in the installed Firefox, headed, with
// WebGPU, over Marionette (firefox.mjs), `--prefs` setting its prefs. In
// place of a trace's presents, `--record=1` records what Firefox's
// compositor composites, a second in every four, and counts the
// composites that showed a new map frame, the last one again, or an older
// one (`stale`: Firefox bug 2091844). Firefox writes each recording's
// PNGs between windows, which slows the page: take frame pacing from runs
// without it. `webkit` flies it in Playwright's WebKit, with neither.
// `--query` adds to the demo page's query (`backend=webgl2` draws with
// WebGL 2).
// `--engine=gljs` flies the pinned MapLibre GL JS over the same style and
// legs instead. `--pkg` serves another build of `core.wasm` (a copy of
// web/pkg) for A/B runs. `--trace` keeps the Chrome trace the present
// intervals come from. `--gpu` times Orbis's passes with timestamp
// queries (`setGpuTiming`) and reports the GPU time of each pass label and
// the slowest submissions' passes; `--gpu=out.json` also keeps every
// submission's.
import {
  cpSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { frameReport, openFirefox, parsePrefs } from "./firefox.mjs";
import { launchChromium, launchWebkit, serveDemo } from "./serve.mjs";

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.match(/^--([a-z]+)=(.+)$/s);
    if (!m) throw new Error(`unknown argument ${a}`);
    return [m[1], m[2]];
  }),
);
const engine = args.engine ?? "orbis";
/** Firefox's composition recordings: this long, once this often. */
const RECORD_MS = 1000;
const RECORD_EVERY_MS = 4000;
const [width, height] = (args.size ?? "1600x1000").split("x").map(Number);
const ratio = Number(args.ratio ?? 2);
const data = (name) =>
  fileURLToPath(new URL(`../demo/data/${name}`, import.meta.url));
const legs = JSON.parse(
  readFileSync(data("alps-flight.json"), "utf8"),
).legs.slice(0, Number(args.legs ?? 4));

const pkg = fileURLToPath(new URL("../pkg/", import.meta.url));
let restore;
if (args.pkg) {
  const saved = mkdtempSync(join(tmpdir(), "orbis-pkg-"));
  cpSync(pkg, saved, { recursive: true });
  cpSync(args.pkg, pkg, { recursive: true });
  restore = () => {
    cpSync(saved, pkg, { recursive: true });
    rmSync(saved, { recursive: true });
  };
}
const dir = mkdtempSync(join(tmpdir(), "orbis-flight-"));
// GL JS's page, served beside the demo for its module and worker.
writeFileSync(
  join(dir, "gl.html"),
  `<!doctype html><link rel="stylesheet" href="../gl/maplibre-gl.css">
  <style>html,body,#map{margin:0;width:100%;height:100%}</style><div id="map"></div>
  <script type="module">
    import * as maplibregl from "../gl/maplibre-gl.mjs";
    maplibregl.setWorkerUrl(new URL("../gl/maplibre-gl-worker.mjs", location.href).href);
    globalThis.maplibregl = maplibregl;
  </script>`,
);
let server;
try {
  server = await serveDemo({
    mounts: {
      "flight/": dir,
      "gl/": fileURLToPath(
        new URL("../node_modules/maplibre-gl/dist/", import.meta.url),
      ),
    },
  });
} finally {
  restore?.();
}
const browserName = args.browser ?? "chromium";
const prefs = parsePrefs(args.prefs);
// Firefox over Marionette (firefox.mjs), for its composition recorder.
const firefox =
  browserName === "firefox"
    ? await openFirefox({ prefs, width, height })
    : null;
const browser = firefox
  ? null
  : browserName === "webkit"
    ? await launchWebkit()
    : await launchChromium({ unsafe: false });
const tracePath = args.trace ?? join(dir, "trace.json");
try {
  const page =
    firefox ??
    (await browser.newPage({
      viewport: { width, height },
      deviceScaleFactor: ratio,
    }));
  page.on?.("pageerror", (e) => console.error(`page: ${e.message}`));
  // Firefox's page runs it once the page has loaded.
  const recordFrames = () => {
    globalThis.__raf = [];
    const tick = (t) => {
      globalThis.__raf.push(t);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    globalThis.__frames = [];
    addEventListener("orbis-frame", (e) => {
      const s = e.detail;
      globalThis.__frames.push({
        t: performance.now(),
        ms: s.ms,
        jobs: s.jobsDone,
        meshes: s.meshesUploaded,
        pending: s.pipelinesPending,
        placement: s.times.placement,
        loading: s.tilesLoading,
        bundle: s.bundleRecorded,
      });
    });
  };
  if (!firefox) await page.addInitScript(recordFrames);
  // Resolves when the map is idle; `fly` is the page function that flies
  // `legs` and resolves to the flight's start and end on the page's clock,
  // and its argument.
  let fly;
  if (engine === "gljs") {
    const style = readFileSync(data("alps.json"), "utf8");
    await page.goto(`${server.url}flight/gl.html`);
    await page.waitForFunction(() => globalThis.maplibregl, null, {
      timeout: 60_000,
    });
    if (firefox) await page.evaluate(recordFrames);
    await page.evaluate(
      (style) =>
        new Promise((resolve) => {
          const s = JSON.parse(style);
          const map = new maplibregl.Map({
            container: "map",
            style: s,
            center: s.center,
            zoom: s.zoom,
            bearing: s.bearing,
            pitch: s.pitch,
            maxPitch: 85,
            attributionControl: false,
          });
          globalThis.gl = map;
          map.on("render", () =>
            globalThis.__frames.push({ t: performance.now() }),
          );
          map.once("idle", resolve);
        }),
      style,
    );
    fly = [
      async (legs) => {
        const map = globalThis.gl;
        const started = await new Promise((r) => requestAnimationFrame(r));
        for (const leg of legs) {
          await new Promise((r) => {
            map.once("moveend", r);
            map.flyTo(leg);
          });
        }
        return { started, ended: performance.now() };
      },
      legs,
    ];
  } else {
    await page.goto(
      `${server.url}?style=alps${args.query ? `&${args.query}` : ""}`,
    );
    await page.waitForFunction(() => globalThis.orbis?.map, null, {
      timeout: 120_000,
    });
    await page.evaluate(
      () => new Promise((r) => globalThis.orbis.map.once("idle").then(r)),
    );
    if (firefox) await page.evaluate(recordFrames);
    fly = [
      async ([legs, gpu]) => {
        const map = globalThis.orbis.map;
        const submissions = [];
        let collecting = Promise.resolve();
        const collect = () =>
          (collecting = collecting.then(async () => {
            const at = performance.now();
            for (const s of await map.gpuTimings())
              submissions.push({ at, ...s });
          }));
        let timer;
        if (gpu) {
          map.setGpuTiming(true);
          // Collected each quarter second, which places each
          // submission in the flight (`at`) to within that.
          timer = setInterval(collect, 250);
        }
        const started = await new Promise((r) => requestAnimationFrame(r));
        for (const leg of legs) await map.camera.flyTo(leg);
        const ended = performance.now();
        if (gpu) {
          clearInterval(timer);
          await collect();
          map.setGpuTiming(false);
        }
        return { started, ended, submissions };
      },
      [legs, !!args.gpu],
    ];
  }
  const cdp =
    browserName === "chromium"
      ? await page.context().newCDPSession(page)
      : null;
  await cdp?.send("Tracing.start", {
    categories:
      "devtools.timeline,v8.gc,disabled-by-default-devtools.timeline,toplevel,gpu",
    transferMode: "ReturnAsStream",
  });
  if (args.busy) {
    const [task, period = 100] = args.busy.split("/").map(Number);
    await page.evaluate(
      ([task, period]) => {
        globalThis.__busy = setInterval(() => {
          const end = performance.now() + task;
          while (performance.now() < end);
        }, period);
      },
      [task, period],
    );
  }
  const { started, ended, submissions, composites } =
    firefox && args.record
      ? await flyRecording(firefox, ...fly)
      : await page.evaluate(...fly);
  if (args.busy) await page.evaluate(() => clearInterval(globalThis.__busy));
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
    writeFileSync(tracePath, chunks.join(""));
  }
  const frames = await page.evaluate(() => globalThis.__frames);
  const raf = (await page.evaluate(() => globalThis.__raf)).filter(
    (t) => t > started && t <= ended,
  );
  const where =
    engine === "orbis"
      ? await page.evaluate(() => ({
          backend: globalThis.orbis.map.backend,
        }))
      : {};
  const moving = frames.filter((f) => f.t > started && f.t <= ended);
  const gaps = moving
    .slice(1)
    .map((f, i) => ({ gap: f.t - moving[i].t, frame: f }));
  const report = {
    engine,
    browser: browserName,
    ...where,
    ...(args.query && { query: args.query }),
    ...(args.prefs && { prefs: args.prefs }),
    size: `${width}x${height}@${ratio}`,
    seconds: +((ended - started) / 1000).toFixed(1),
    page: { frames: moving.length, ...pacing(gaps.map((g) => g.gap)) },
    animationFrames: {
      frames: raf.length,
      ...pacing(raf.slice(1).map((t, i) => t - raf[i])),
    },
    ...(cdp && { presents: presents(tracePath) }),
  };
  if (engine === "orbis") {
    report.renderMs = pacing(moving.map((f) => f.ms));
    report.slowest = gaps
      .sort((a, b) => b.gap - a.gap)
      .slice(0, 8)
      .map(({ gap, frame }) => ({
        ...frame,
        gap: +gap.toFixed(1),
        t: +(frame.t - started).toFixed(0),
      }));
  }
  if (submissions?.length) {
    report.gpu = gpuReport(submissions);
    if (args.gpu.endsWith(".json")) {
      writeFileSync(
        args.gpu,
        JSON.stringify(submissions.map((s) => ({ ...s, at: s.at - started }))),
      );
    }
  }
  if (composites) report.composited = composites;
  console.log(JSON.stringify(report));
} finally {
  await (firefox ?? browser).close();
  server.close();
  rmSync(dir, { recursive: true });
}

/**
 * Flies `fly(arg)` in Firefox's page (firefox.mjs) and records what its
 * compositor composites meanwhile: windows of `RECORD_MS` every
 * `RECORD_EVERY_MS` (the recorder keeps each composite whole until it
 * stops, so a long recording would take gigabytes). Resolves to the
 * flight's result with `composites`, frameReport's counts over every
 * window, the map canvas found in the window's pixels.
 */
async function flyRecording(page, fly, arg) {
  await page.evaluate(
    `(arg) => { globalThis.__flown = undefined;
      (${fly})(arg).then((r) => (globalThis.__flown = r),
        (e) => (globalThis.__flown = { thrown: String(e?.stack ?? e) })); }`,
    arg,
  );
  const recordings = [];
  while (!(await page.evaluate(() => globalThis.__flown !== undefined))) {
    const started = Date.now();
    const recording = await page.record();
    await new Promise((r) => setTimeout(r, RECORD_MS));
    recordings.push(await recording.stop());
    await new Promise((r) =>
      setTimeout(r, Math.max(0, RECORD_EVERY_MS - (Date.now() - started))),
    );
  }
  const flown = await page.evaluate(() => globalThis.__flown);
  if (flown.thrown) throw new Error(`the flight failed: ${flown.thrown}`);
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
  const reports = recordings.map((r) => frameReport(r, rect));
  const sum = (key) => reports.reduce((n, r) => n + r[key], 0);
  return {
    ...flown,
    composites: {
      windows: reports.length,
      composites: sum("composites"),
      fresh: sum("fresh"),
      repeated: sum("repeated"),
      stale: sum("stale"),
      newFrameIntervalP95: Math.max(
        ...reports.map((r) => r.newFrameInterval.p95),
      ),
      over25: reports.reduce((n, r) => n + r.newFrameInterval.over25, 0),
    },
  };
}

/** Percentiles of `xs`, and the counts over a 60 Hz frame, over one and a
 * half (a refresh missed whatever the jitter) and over two. */
function pacing(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const at = (p) =>
    +s[Math.min(s.length - 1, Math.round((s.length - 1) * p))].toFixed(1);
  return {
    p50: at(0.5),
    p95: at(0.95),
    p99: at(0.99),
    max: at(1),
    over16: s.filter((x) => x > 16.7).length,
    over25: s.filter((x) => x > 25).length,
    over33: s.filter((x) => x > 33.4).length,
  };
}

/** Per-pass-label GPU times over `submissions` (`gpuTimings`'s), the
 * distribution of their spans (GPU time), and the slowest submissions. */
function gpuReport(submissions) {
  const labels = {};
  for (const s of submissions) {
    for (const [label, ms] of s.passes) {
      const l = (labels[label] ??= { n: 0, ms: 0, max: 0 });
      l.n++;
      l.ms += ms;
      l.max = Math.max(l.max, ms);
    }
  }
  const passes = Object.entries(labels)
    .sort((a, b) => b[1].ms - a[1].ms)
    .map(([label, l]) => ({
      label,
      n: l.n,
      totalMs: +l.ms.toFixed(0),
      meanMs: +(l.ms / l.n).toFixed(3),
      maxMs: +l.max.toFixed(2),
    }));
  const slowest = [...submissions]
    .sort((a, b) => b.span - a.span)
    .slice(0, 6)
    .map((s) => {
      const byLabel = {};
      for (const [label, ms] of s.passes)
        byLabel[label] = +((byLabel[label] ?? 0) + ms).toFixed(2);
      return { span: +s.span.toFixed(2), passes: byLabel };
    });
  return {
    submissions: submissions.length,
    span: pacing(submissions.map((s) => s.span)),
    passes,
    slowest,
  };
}

/** The intervals between the GPU process's presents in `trace`, within the
 * flight (the trace spans it), and how long the GPU process waits per
 * present for the page's GPU work before compositing it (on a Mac): about
 * a WebGPU frame's GPU time, since the compositor reads the canvas after
 * it. A frame whose GPU time runs past the refresh misses it. */
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
  const waited =
    events
      .filter(
        (e) => e.name === "IOSurfaceImageBacking::WaitForCommandsToBeScheduled",
      )
      .reduce((sum, e) => sum + (e.dur ?? 0), 0) / 1000;
  return {
    count: ts.length,
    ...pacing(gaps.length ? gaps : [0]),
    waitMs: +(waited / Math.max(1, ts.length)).toFixed(2),
  };
}
