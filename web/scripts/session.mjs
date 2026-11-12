// A user's session on a live style, as a user has it: a fresh browser (an
// empty HTTP cache and the browser's shader cache empty), the page opened,
// and real mouse input (wheel notches, drags, right-button tilts and turns)
// from soon after the map's first frame, while its tiles and pipelines
// still load, through the same moves in every engine. In headless Chromium
// or the installed Firefox (headed, on WebGPU: serve.mjs's
// `launchFirefox`), in Orbis and the pinned GL JS on demo/load.html.
//
//   pnpm session [--session=manhattan|paris|alps|globe] [--browser=chromium|firefox]
//     [--engines=orbis,gljs] [--runs=n] [--style=key|url] [--size=WxH]
//     [--ratio=r] [--start=ms] [--warm] [--driver] [--page] [--headed] [--worst=n]
//     [--backend=webgpu|webgl2] [--pkgs=a,b] [--deep]
//     [--dump=prefix] [--profile=firefox-profile.json]
//
// Sessions: `manhattan` opens Liberty over Midtown at z13 and wheel-zooms
// in to about z16 as the buildings load, tilts to 70, turns and drags
// across the island; `paris` tilts Liberty over central Paris from pitch
// 60 to the limit (85) and pans and turns there; `alps` drags, turns and
// zooms the Alps terrain style at pitch 72; `globe` wheel-zooms the demo's
// Globe entry (terrain on) over New York from z2 to about z8.5, tilting
// to the limit on the way. The moves end where the hand stops (no fling),
// so every engine and browser ends at the same camera (`camera` in the
// report says where). `--style` gives a session's moves another catalogue
// key or style URL. `--start` is how long after the first frame the input
// starts (500 ms). `--warm` makes the session once untimed first and
// reloads, so tiles come from the HTTP cache and pipelines from the
// browser's. `--headed` runs Chromium in a window, at the display's
// refresh. `--dump` writes each engine's frames, input windows and
// presents to `<prefix>-<engine>.json`. `--backend` draws Orbis with that
// backend. `--pkgs` runs Orbis on each of these builds of web/pkg (copies
// of it) in turn, run by run, for A/B runs in one session; GL JS runs once
// a run. `--profile` (Firefox, one engine) runs Gecko's profiler on every
// thread of every process from startup and writes its profile there when
// Firefox quits, for profiler.firefox.com; the report's `profile` then
// says, for the worst intervals and the pipelines made mid-move, which
// threads of which processes were busy (CPU ms), so whether a compile
// runs on the thread that presents shows without symbols.
// In Chromium the report has the GPU process's main thread over the
// moves (`gpuProcess`: busy time, tasks over 50 ms, the longest), and with
// `--deep` (GPU service tracing) what its slow tasks did, its GL calls
// (ANGLE's waits on WebGL 2, #140) and its work in the worst present gaps.
//
// For each run it reports when the first frame came, the map's frame
// intervals (as the page hears of them: Orbis's `render` events from its
// worker, GL JS's on the page), the page's animation-frame intervals and,
// in Chromium, the GPU process's presents (what reached the screen) while
// input moves the camera, and how long the map took to settle after
// the last input. For Orbis it names what each of the `--worst` longest
// intervals spent: the frame's render CPU by stage, the work `prepare` did
// between frames, pipelines still compiling, tiles arrived, bytes uploaded
// and GPU memory grown, and what none of that accounts for (`untimedMs`:
// the browser's GPU thread, the compositor, GC). Chromium also reports the
// page's long tasks.
// Firefox on macOS draws Orbis on the page's thread (map.ts's
// `staleWorkerFrames`), so there the page's own frames show the map's
// work, and `--page` draws it there in Chromium too. A map on the page
// has its WebGPU calls timed (each slow frame's, and the session's, with
// the bytes each buffer took), its message handlers too, its calls over
// 5 ms listed by when (`slowCalls`), and the pipelines it made during the
// moves (`pipelinesInMoves`: each a compile mid-gesture). Its cold runs
// bust the driver's shader cache (`?nonce=`, cold.js), which a map in a
// worker can't, unless `--driver` keeps what the driver's cache holds from
// earlier runs (a user's later launches).
import {
  cpSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

import { launchChromium, launchFirefox, serveDemo } from "./serve.mjs";

const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const m = a.match(/^--([a-z]+)(?:=(.+))?$/s);
    if (!m) throw new Error(`unknown argument ${a}`);
    return [m[1], m[2] ?? "1"];
  }),
);
const sessionName = args.session ?? "manhattan";
const browserName = args.browser ?? "chromium";
const engines = (args.engines ?? "orbis,gljs").split(",");
const runs = Number(args.runs ?? 1);
const [width, height] = (args.size ?? "1440x900").split("x").map(Number);
const ratio = Number(args.ratio ?? 2);
const startAfter = Number(args.start ?? 500);
const worstCount = Number(args.worst ?? 6);
const warm = !!args.warm;
const wheelScale = browserName === "firefox" ? ratio : 1;
const dump = args.dump;
const deep = !!args.deep;
const pkgs = args.pkgs ? args.pkgs.split(",").map((p) => resolve(p)) : [null];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const alps = JSON.parse(
  readFileSync(
    fileURLToPath(new URL("../demo/data/alps.json", import.meta.url)),
    "utf8",
  ),
);

/** Drags with `button` from (x0, y0) by (dx, dy) over `ms`, a move a
 * display frame, placed by the time gone: a browser slow to take moves
 * gets fewer, not a longer drag. */
async function drag(page, [x0, y0], [dx, dy], ms, button = "left") {
  await page.mouse.move(x0, y0);
  await page.mouse.down({ button });
  const start = performance.now();
  for (let f = 0; f < 1;) {
    await sleep(16);
    f = Math.min(1, (performance.now() - start) / ms);
    await page.mouse.move(x0 + dx * f, y0 + dy * f);
  }
  // Held still before the release, so no engine's momentum carries the
  // camera on and every engine ends where the hand stopped.
  await sleep(150);
  await page.mouse.up({ button });
}

/** `n` wheel notches at (x, y), `ms` apart: negative `delta` zooms in.
 * Firefox gets Playwright's wheel deltas in device pixels, so its notches
 * are scaled to come out as Chromium's. */
async function wheel(page, [x, y], delta, n, ms = 50) {
  await page.mouse.move(x, y);
  const start = performance.now();
  for (let i = 0; i < n; i++) {
    await page.mouse.wheel(0, delta / wheelScale);
    await sleep(Math.max(0, start + (i + 1) * ms - performance.now()));
  }
}

const c = [width / 2, height / 2];
const low = [width / 2, height * 0.8];
/** Each session: its style, its first camera and its moves, each a
 * function of the page; `moves` are timed, the pauses between them not. */
const SESSIONS = {
  manhattan: {
    style: "liberty",
    at: [-73.9855, 40.758, 13, 0, 0],
    moves: [
      ...Array.from({ length: 6 }, () => (p) => wheel(p, c, -100, 6)),
      (p) => drag(p, c, [0, -140], 800, "right"),
      (p) => drag(p, c, [160, 0], 800, "right"),
      (p) => drag(p, low, [0, -200], 800),
      (p) => drag(p, low, [-300, -100], 800),
      (p) => drag(p, low, [0, -200], 800),
      (p) => wheel(p, c, 100, 8),
      (p) => wheel(p, c, -100, 8),
    ],
  },
  paris: {
    style: "liberty",
    at: [2.3417, 48.8559, 15, 30, 60],
    moves: [
      (p) => drag(p, c, [0, -200], 600, "right"),
      (p) => drag(p, low, [0, -150], 800),
      (p) => drag(p, c, [200, 0], 1000, "right"),
      (p) => drag(p, low, [250, -100], 800),
      (p) => drag(p, low, [0, -150], 800),
      (p) => drag(p, c, [-300, 0], 1200, "right"),
      (p) => drag(p, low, [-250, -100], 800),
      (p) => wheel(p, c, -100, 8),
      (p) => drag(p, low, [0, -150], 800),
    ],
  },
  alps: {
    style: "alps",
    at: [...alps.center, alps.zoom, alps.bearing, alps.pitch],
    moves: [
      (p) => drag(p, low, [0, -400], 800),
      (p) => drag(p, c, [250, 0], 1200, "right"),
      (p) => drag(p, low, [200, -300], 800),
      (p) => wheel(p, c, -100, 8),
      (p) => drag(p, c, [-300, 0], 1200, "right"),
      (p) => drag(p, low, [0, -400], 800),
      (p) => wheel(p, c, 100, 12),
      (p) => drag(p, low, [-300, -200], 800),
    ],
  },
  globe: {
    style: "globe",
    at: [-73.9855, 40.758, 2, 0, 0],
    moves: [
      ...Array.from({ length: 8 }, () => (p) => wheel(p, c, -100, 6)),
      (p) => drag(p, c, [0, -200], 800, "right"),
      ...Array.from({ length: 6 }, () => (p) => wheel(p, c, -100, 6)),
      (p) => drag(p, low, [0, -400], 800),
      (p) => drag(p, c, [200, 0], 1000, "right"),
    ],
  },
};
const session = SESSIONS[sessionName];
if (!session) throw new Error(`unknown session ${sessionName}`);
const style = args.style ?? session.style;

/** Percentiles of `xs`, and how many exceed 25 and 50 ms. */
function pacing(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const at = (p) =>
    s.length
      ? +s[Math.min(s.length - 1, Math.round((s.length - 1) * p))].toFixed(1)
      : 0;
  return {
    n: s.length,
    p50: at(0.5),
    p95: at(0.95),
    p99: at(0.99),
    max: at(1),
    over25: s.filter((x) => x > 25).length,
    over50: s.filter((x) => x > 50).length,
  };
}

/** Intervals between successive `times` that begin and end inside one of
 * `windows`: not the wait for input before a move's first frame. */
function intervals(times, windows) {
  const out = [];
  for (let i = 1; i < times.length; i++) {
    const [s, t] = [times[i - 1], times[i]];
    if (windows.some(([a, b]) => s >= a && t <= b)) out.push({ i, gap: t - s });
  }
  return out;
}

/** The GPU process's presents in a Chrome trace, in ms on the page's
 * clock: each one a new picture on screen. */
function presents(traceEvents, origin) {
  const named = (name) =>
    traceEvents
      .filter((e) => e.name === name)
      .map((e) => e.ts / 1000)
      .sort((a, b) => a - b);
  const mac = named("ImageTransportSurfaceOverlayMac::Present");
  return (mac.length ? mac : named("Display::DrawAndSwap")).map(
    (t) => t - origin,
  );
}

const round = (x, d = 1) => +x.toFixed(d);
const mib = (b) => round(b / (1 << 20));

/** What Orbis's device did over `frames` (each frame's counts since the
 * one before): pipelines asked for, buffers and textures made and their
 * bytes, bytes copied on the GPU, uploads and their calls, texture-write
 * submissions, drape slots redrawn. */
function deviceTotals(frames) {
  const sum = (f) => frames.reduce((s, x) => s + (f(x) ?? 0), 0);
  return {
    frames: frames.length,
    pipelines: sum((f) => f.device.pipelinesRequested),
    created: sum((f) => f.device.buffersCreated + f.device.texturesCreated),
    allocatedMiB: mib(sum((f) => f.device.bytesAllocated)),
    copiedMiB: mib(sum((f) => f.device.bytesCopied)),
    uploadMiB: mib(sum((f) => f.device.uploadBytes)),
    uploadCalls: sum((f) => f.device.uploadCalls),
    textureWriteSubmissions: sum((f) => f.device.textureWriteSubmissions),
    drapeSlots: sum((f) => f.drapeSlots),
  };
}

/** Time to centre: when the first frame came whose tiles under the
 * view's centre had all arrived (from navigation), and for each move how
 * long after it ended that took (0 when they were in already). */
function centre(frames, windows) {
  const first = frames.find((f) => f.centre);
  const afterMoves = windows.map(([, end]) => {
    // The frame showing the move's end: an engine draws none after it
    // when nothing changes.
    if (frames.findLast((f) => f.t <= end)?.centre) return 0;
    const f = frames.find((f) => f.t >= end && f.centre);
    return f ? Math.round(f.t - end) : null;
  });
  const known = afterMoves.filter((x) => x !== null).sort((a, b) => a - b);
  return {
    centreMs: first ? Math.round(first.t) : null,
    centreAfterMoves: {
      p50: known[Math.floor((known.length - 1) / 2)] ?? null,
      max: known.at(-1) ?? null,
      never: afterMoves.length - known.length,
    },
  };
}

/** The WebGPU calls made between two tallies that took 1 ms or more or
 * wrote 1 MiB or more (`writeBuffer:<label>`: its bytes by buffer). */
function callsBetween(a = {}, b = {}) {
  return Object.entries(b)
    .map(([name, e]) => {
      const was = a[name] ?? { n: 0, ms: 0, bytes: 0 };
      return [name, e.n - was.n, e.ms - was.ms, e.bytes - was.bytes];
    })
    .filter(([, , ms, bytes]) => ms >= 1 || bytes >= 1 << 20)
    .sort((x, y) => y[2] - x[2] || y[3] - x[3])
    .map(
      ([name, n, ms, bytes]) =>
        `${name} ${n}× ${round(ms)} ms${bytes ? ` ${mib(bytes)} MiB` : ""}`,
    )
    .join(", ");
}

/** What an Orbis frame interval `gap` ending at `frames[i]` spent. */
function attribute(frames, i, gap, raf, longTasks) {
  const f = frames[i];
  const prev = frames[i - 1];
  const top = (times) =>
    Object.entries(times)
      .filter(([k, v]) => k !== "total" && v >= 1)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([k, v]) => `${k} ${round(v)}`)
      .join(", ");
  const between = f.prepared.total;
  const [from, to] = [prev.t, f.t];
  const pageGap = Math.max(
    0,
    ...raf
      .slice(1)
      .flatMap((t, k) => (t > from && raf[k] < to ? [t - raf[k]] : [])),
  );
  return {
    at: Math.round(f.t),
    gap: round(gap),
    renderMs: round(f.ms),
    render: top(f.times),
    betweenMs: round(between),
    between: top(f.prepared),
    untimedMs: round(Math.max(0, gap - f.ms - between)),
    compiling: f.pending,
    layersWaiting: f.waiting,
    arrived: f.arrived,
    meshes: f.meshes - prev.meshes,
    jobs: f.jobs - prev.jobs,
    uploadedMiB: mib(f.uploaded - prev.uploaded),
    gpuGrewMiB: mib(f.gpuBytes - prev.gpuBytes),
    ...(f.device && {
      pipelines: f.device.pipelinesRequested,
      allocatedMiB: mib(f.device.bytesAllocated),
      uploadCalls: f.device.uploadCalls,
      textureWriteSubmissions: f.device.textureWriteSubmissions,
      drapeSlots: f.drapeSlots,
    }),
    candidatesRebuilt: f.rebuilt,
    ...(f.gpu && { webgpu: callsBetween(prev.gpu, f.gpu) }),
    longestPageFrame: round(pageGap),
    ...(longTasks && {
      longTasks: longTasks
        .filter((l) => l.start < to && l.start + l.duration > from)
        .map((l) => Math.round(l.duration)),
    }),
  };
}

async function launch() {
  if (browserName === "firefox") {
    // Gecko's profiler from startup, written out when Firefox quits: every
    // thread sampled over the whole session, for profiler.firefox.com.
    if (args.profile) {
      Object.assign(process.env, {
        MOZ_PROFILER_STARTUP: "1",
        MOZ_PROFILER_STARTUP_ENTRIES: String(1 << 26),
        MOZ_PROFILER_STARTUP_INTERVAL: "1",
        MOZ_PROFILER_STARTUP_FEATURES: "js,stackwalk,cpu,ipcmessages",
        // Every thread: Firefox's WebGPU server runs on threads of its own.
        MOZ_PROFILER_STARTUP_FILTERS: "*",
        MOZ_PROFILER_SHUTDOWN: args.profile,
      });
    }
    // Timers to the microsecond, for the WebGPU tally: Firefox rounds
    // them to the millisecond otherwise.
    const browser = await launchFirefox({
      prefs: { "privacy.reduceTimerPrecision": false },
    });
    if (!browser.webgpu) console.error("no installed Firefox: WebGL 2");
    return browser;
  }
  if (browserName === "chromium") {
    if (!deep) return launchChromium({ unsafe: false, headed: !!args.headed });
    const { chromium } = await import("playwright");
    const { chromiumPlatform } = await import("./serve.mjs");
    const { executablePath, args: flags } = chromiumPlatform();
    return chromium.launch({
      headless: !args.headed,
      executablePath,
      args: [...flags, "--enable-gpu-service-tracing"],
    });
  }
  throw new Error(`unknown browser ${browserName}`);
}

/** One session in `engine` on a fresh browser: the run's report. */
async function run(server, engine) {
  const browser = await launch();
  try {
    const context = await browser.newContext({
      viewport: { width, height },
      deviceScaleFactor: ratio,
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(m.text());
    });
    const onPage = browserName === "firefox" || !!args.page;
    const query = () =>
      new URLSearchParams({
        engine,
        style,
        at: session.at.join(","),
        frames: "",
        ...(engine === "orbis" && args.backend && { backend: args.backend }),
        // Firefox on macOS draws Orbis on the page anyway; there its
        // shaders can be made unique to the run, past the driver's cache.
        ...(engine === "orbis" &&
          onPage && {
            worker: "0",
            ...(!warm &&
              !args.driver && { nonce: 1 + Math.floor(Math.random() * 9999) }),
          }),
      });
    // WebGPU's calls on the page timed and tallied, for a map drawn there:
    // what each frame spent in the browser's WebGPU, by call.
    await page.addInitScript(() => {
      // Calls on the page over 5 ms, by when: what a long frame waited on.
      const slow = (globalThis.__slow = []);
      const timed = (what, f) => {
        const t = performance.now();
        try {
          return f();
        } finally {
          const ms = performance.now() - t;
          if (ms > 5)
            slow.push({ what, at: Math.round(t), ms: Math.round(ms) });
        }
      };
      // A placeholder's synchronous load of core-deferred.wasm: its
      // request and its compile.
      const Module = WebAssembly.Module;
      WebAssembly.Module = function (bytes) {
        return timed("new WebAssembly.Module", () => new Module(bytes));
      };
      WebAssembly.Module.prototype = Module.prototype;
      for (const name of ["customSections", "exports", "imports"])
        WebAssembly.Module[name] = Module[name];
      const Instance = WebAssembly.Instance;
      WebAssembly.Instance = function (module, imports) {
        return timed(
          "new WebAssembly.Instance",
          () => new Instance(module, imports),
        );
      };
      WebAssembly.Instance.prototype = Instance.prototype;
      const send = XMLHttpRequest.prototype.send;
      XMLHttpRequest.prototype.send = function (body) {
        return timed("XMLHttpRequest.send", () => send.call(this, body));
      };
      globalThis.__timed = timed;
    });
    await page.addInitScript(() => {
      if (typeof GPUDevice === "undefined") return;
      const timed = globalThis.__timed;
      const tally = {};
      globalThis.sessionGpuCalls = () => structuredClone(tally);
      const wrap = (proto, name, bytes) => {
        const original = proto[name];
        if (!original) return;
        proto[name] = function (...a) {
          const t = performance.now();
          const r = timed(name, () => original.apply(this, a));
          const e = (tally[name] ??= { n: 0, ms: 0, bytes: 0 });
          e.n++;
          e.ms += performance.now() - t;
          if (bytes) e.bytes += bytes(a);
          return r;
        };
      };
      const size = (data, offset = 0, len) =>
        len ?? (data.byteLength ?? 0) - offset * (data.BYTES_PER_ELEMENT ?? 1);
      wrap(GPUQueue.prototype, "writeBuffer", (a) => {
        const bytes = size(a[2], a[3], a[4]);
        // And by the buffer's label: which of the map's buffers take the bytes.
        const e = (tally[`writeBuffer:${a[0].label || "?"}`] ??= {
          n: 0,
          ms: 0,
          bytes: 0,
        });
        e.n++;
        e.bytes += bytes;
        return bytes;
      });
      wrap(GPUQueue.prototype, "writeTexture", (a) => a[1].byteLength ?? 0);
      wrap(GPUQueue.prototype, "submit");
      wrap(GPUQueue.prototype, "copyExternalImageToTexture");
      for (const name of [
        "createBuffer",
        "createTexture",
        "createBindGroup",
        "createShaderModule",
        "createRenderPipeline",
        "createComputePipeline",
        "createRenderPipelineAsync",
        "createComputePipelineAsync",
        "createCommandEncoder",
        "createRenderBundleEncoder",
      ])
        wrap(GPUDevice.prototype, name);
      // Each pipeline made, by when and its label: one made while the map
      // moves compiles then, and a browser that compiles on its GPU
      // thread (Firefox) holds the frame back meanwhile, async or not.
      const made = (globalThis.__pipelines = []);
      for (const name of [
        "createRenderPipeline",
        "createComputePipeline",
        "createRenderPipelineAsync",
        "createComputePipelineAsync",
      ]) {
        const original = GPUDevice.prototype[name];
        GPUDevice.prototype[name] = function (desc) {
          made.push({
            at: Math.round(performance.now()),
            label: desc?.label ?? "?",
            async: name.endsWith("Async"),
          });
          return original.call(this, desc);
        };
      }
      wrap(GPUCommandEncoder.prototype, "finish");
      wrap(GPURenderBundleEncoder.prototype, "finish");
      wrap(GPUBuffer.prototype, "destroy");
      wrap(GPUTexture.prototype, "destroy");
      wrap(GPUCanvasContext.prototype, "getCurrentTexture");
      // Message handlers on the page (tile jobs' results, the helper's
      // replies), timed as `onmessage`.
      for (const proto of [MessagePort.prototype, Worker.prototype]) {
        const d = Object.getOwnPropertyDescriptor(proto, "onmessage");
        Object.defineProperty(proto, "onmessage", {
          ...d,
          set(handler) {
            d.set.call(
              this,
              handler &&
                function (e) {
                  const t = performance.now();
                  try {
                    return timed("onmessage", () => handler.call(this, e));
                  } finally {
                    // By the message's keys: what kind of reply took the time.
                    const d = e.data;
                    const kind =
                      d && typeof d === "object" && !ArrayBuffer.isView(d)
                        ? Object.keys(d).sort().join("+")
                        : typeof d;
                    for (const key of ["onmessage", `onmessage:${kind}`]) {
                      const s = (tally[key] ??= { n: 0, ms: 0, bytes: 0 });
                      s.n++;
                      s.ms += performance.now() - t;
                    }
                  }
                },
            );
          },
        });
      }
    });
    await page.addInitScript(() => {
      addEventListener(
        "wheel",
        (e) => (globalThis.__wheel ??= [e.deltaY, e.deltaMode]),
        { capture: true, passive: true },
      );
      globalThis.__raf = [];
      const tick = (t) => {
        globalThis.__raf.push(t);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
      globalThis.__long = [];
      try {
        new PerformanceObserver((list) => {
          for (const e of list.getEntries())
            globalThis.__long.push({
              start: e.startTime,
              duration: e.duration,
            });
        }).observe({ type: "longtask", buffered: true });
      } catch {
        globalThis.__long = null;
      }
    });
    // When the map settled: no frame for two seconds (an idle event fires
    // only after a frame, so a map idle already would never send one), or
    // null after a minute.
    const settle = () =>
      page.evaluate(
        () =>
          new Promise((resolve) => {
            const until = performance.now() + 60_000;
            const poll = () => {
              const last = globalThis.mapFrames.at(-1).t;
              if (performance.now() - last > 2000) resolve(last);
              else if (performance.now() > until) resolve(null);
              else setTimeout(poll, 100);
            };
            poll();
          }),
      );
    const open = async () => {
      await page.goto(`${server.url}load.html?${query()}`, {
        waitUntil: "commit",
      });
      await page.waitForFunction(
        () =>
          globalThis.mapFrames?.length > 0 ||
          document.title.startsWith("error"),
        null,
        { timeout: 120_000, polling: 16 },
      );
      const title = await page.title();
      if (title.startsWith("error")) throw new Error(title);
    };
    const moves = async () => {
      const windows = [];
      for (const move of session.moves) {
        const a = await page.evaluate(() => performance.now());
        await move(page);
        const b = await page.evaluate(() => performance.now());
        windows.push([a, b]);
        await sleep(400);
      }
      return windows;
    };
    await open();
    if (warm) {
      await sleep(startAfter);
      await moves();
      await settle();
      await open();
    }
    const firstFrame = await page.evaluate(() => globalThis.mapFrames[0].t);
    await sleep(startAfter);
    // Chrome's presents, on the trace clock: a `clock` mark ties it to the page's.
    const cdp =
      browserName === "chromium" ? await context.newCDPSession(page) : null;
    await cdp?.send("Tracing.start", {
      categories: [
        "gpu,viz,toplevel,blink.user_timing",
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
    const mark = await page.evaluate(() => {
      performance.mark("session-clock");
      return performance.now();
    });
    const windows = await moves();
    let screen, gpuProcess;
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
      const trace = JSON.parse(chunks.join(""));
      const events = Array.isArray(trace) ? trace : trace.traceEvents;
      const at = events.find((e) => e.name === "session-clock");
      if (at) {
        const origin = at.ts / 1000 - mark;
        screen = presents(events, origin);
        gpuProcess = gpuMain(events, origin, [
          windows[0][0],
          windows.at(-1)[1],
        ]);
      }
    }
    const lastInput = windows.at(-1)[1];
    const settled = await settle();
    const {
      frames,
      raf,
      longTasks,
      where,
      camera,
      wheel: firstWheel,
      wasm,
      slow,
      pipelines,
    } = await page.evaluate(() => ({
      camera: globalThis.mapCamera?.(),
      wheel: globalThis.__wheel,
      wasm: performance
        .getEntriesByType("resource")
        .filter((e) => new URL(e.name).pathname.endsWith(".wasm"))
        .map((e) => ({
          file: new URL(e.name).pathname.split("/").pop(),
          start: Math.round(e.startTime),
          end: Math.round(e.responseEnd),
        })),
      frames: globalThis.mapFrames,
      slow: globalThis.__slow,
      pipelines: globalThis.__pipelines,
      raf: globalThis.__raf,
      longTasks: globalThis.__long,
      where: globalThis.loadDetails?.(),
    }));
    const times = frames.map((f) => f.t);
    const mapGaps = intervals(times, windows);
    const pageGaps = intervals(raf, windows);
    const report = {
      session: sessionName,
      style,
      browser: browserName,
      engine,
      ...(where && { backend: where.backend }),
      warm,
      size: `${width}x${height}@${ratio}`,
      firstFrameMs: Math.round(firstFrame),
      inputSeconds: round(windows.reduce((s, [a, b]) => s + b - a, 0) / 1000),
      mapFrames: pacing(mapGaps.map((g) => g.gap)),
      ...(screen && {
        presents: pacing(intervals(screen, windows).map((g) => g.gap)),
      }),
      ...(gpuProcess && { gpuProcess }),
      pageFrames: pacing(pageGaps.map((g) => g.gap)),
      settleMs: settled === null ? null : Math.round(settled - lastInput),
      ...(frames.some((f) => "centre" in f) && centre(frames, windows)),
      // Where the moves left the camera: the same in every engine and
      // browser when they take input alike.
      firstWheel,
      // The page's calls over 5 ms (WebGPU's, message handlers, wasm
      // compiles and instantiations, synchronous requests), by when.
      slowCalls: slow,
      // Pipelines made once the moves began: each a compile mid-gesture.
      ...(pipelines && {
        pipelinesInMoves: pipelines
          .filter((p) => p.at >= windows[0][0])
          .map((p) => `${p.at} ${p.label}${p.async ? " (async)" : ""}`),
      }),
      // When the page fetched each wasm file (core-deferred.wasm late, in
      // the moves, is a placeholder's synchronous load).
      wasm,
      ...(camera && {
        camera: [
          ...camera.center.map((x) => round(x, 4)),
          round(camera.zoom, 2),
          round(camera.bearing),
          round(camera.pitch),
        ],
      }),
      ...(longTasks && {
        longTasks: pacing(
          longTasks
            .filter((l) =>
              windows.some(([a, b]) => l.start < b && l.start + l.duration > a),
            )
            .map((l) => l.duration),
        ),
      }),
    };
    if (engine === "orbis" && frames.length) {
      const last = frames.at(-1);
      report.totals = {
        frames: frames.length,
        meshes: last.meshes,
        jobs: last.jobs,
        uploadedMiB: mib(last.uploaded),
        gpuMiB: mib(last.gpuBytes),
        framesCompiling: frames.filter((f) => f.pending > 0).length,
        ...(last.gpu && { webgpu: callsBetween({}, last.gpu) }),
        lastCompilingAtMs: Math.round(
          frames.findLast((f) => f.pending > 0)?.t ?? 0,
        ),
      };
      if (last.device) {
        // The gates' counts: what the device did while input held the
        // camera (a pipeline made then is a compile mid-gesture) and in
        // frames that moved it (memory made then moves while it moves).
        const inMoves = frames.filter((f) =>
          windows.some(([a, b]) => f.t >= a && f.t <= b),
        );
        report.device = {
          session: deviceTotals(frames),
          input: deviceTotals(frames.filter((f) => f.input)),
          moving: deviceTotals(frames.filter((f) => f.moved || f.input)),
          inMoves: deviceTotals(inMoves),
        };
      }
      report.worst = mapGaps
        .sort((a, b) => b.gap - a.gap)
        .slice(0, worstCount)
        .map(({ i, gap }) => attribute(frames, i, gap, raf, longTasks));
    }
    if (errors.length)
      report.errors = errors.slice(0, 5).map((e) => e.split("\n")[0]);
    if (browserName === "firefox" && args.profile)
      profiled = {
        report,
        origin: await page.evaluate(() => performance.timeOrigin),
        gaps: [...mapGaps]
          .sort((a, b) => b.gap - a.gap)
          .slice(0, worstCount)
          .map(({ i, gap }) => [frames[i].t - gap, frames[i].t]),
        pipelines: (pipelines ?? []).filter((p) => p.at >= windows[0][0]),
      };
    if (dump)
      writeFileSync(
        `${dump}-${engine}.json`,
        JSON.stringify({ windows, frames, raf, screen }),
      );
    return report;
  } finally {
    await browser.close();
  }
}

/** The GPU process's main thread in a Chrome trace over `span` (page
 * ms): its busy time and tasks over 50 ms, and with `--deep` what its
 * slow tasks did, its GL calls and its work in the worst present gaps. */
function gpuMain(events, origin, [from, to]) {
  const meta = (name) => events.filter((e) => e.ph === "M" && e.name === name);
  const pid = meta("process_name").find(
    (e) => e.args.name === "GPU Process",
  )?.pid;
  const tid = meta("thread_name").find(
    (e) => e.pid === pid && e.args.name === "CrGpuMain",
  )?.tid;
  if (pid === undefined) return undefined;
  const inSpan = (e) =>
    e.ts / 1000 - origin >= from && e.ts / 1000 - origin <= to;
  const onMain = events.filter(
    (e) => e.pid === pid && e.tid === tid && e.ph === "X" && inSpan(e),
  );
  const tasks = onMain.filter(
    (e) => e.name === "ThreadControllerImpl::RunTask",
  );
  const out = {
    busyMs: Math.round(tasks.reduce((s, e) => s + e.dur, 0) / 1000),
    over50: tasks.filter((e) => e.dur > 50_000).length,
    longestMs: Math.round(Math.max(0, ...tasks.map((e) => e.dur)) / 1000),
  };
  if (!deep) return out;
  const tally = (list) => {
    const by = {};
    for (const e of list) {
      const b = (by[e.name] ??= { n: 0, ms: 0, max: 0 });
      b.n++;
      b.ms += e.dur / 1000;
      b.max = Math.max(b.max, e.dur / 1000);
    }
    return Object.entries(by)
      .sort((a, b) => b[1].ms - a[1].ms)
      .slice(0, 12)
      .map(
        ([name, b]) =>
          `${name}: ${b.n}× ${round(b.ms, 0)} ms (max ${round(b.max)})`,
      );
  };
  const slow = tasks.filter((e) => e.dur > 16_000);
  out.inSlowTasks = tally(
    onMain.filter(
      (e) =>
        e.name !== "ThreadControllerImpl::RunTask" &&
        slow.some((t) => e.ts >= t.ts && e.ts < t.ts + t.dur),
    ),
  );
  out.gl = tally(
    onMain
      .filter((e) => e.name.startsWith("TraceGLAPI::"))
      .map((e) => ({ ...e, name: e.name.slice(12) })),
  );
  const shown = presents(events, 0)
    .map((t) => t * 1000)
    .filter((ts) => inSpan({ ts }));
  out.worstGaps = shown
    .slice(1)
    .map((t, i) => ({ g: (t - shown[i]) / 1000, from: shown[i], to: t }))
    .sort((a, b) => b.g - a.g)
    .slice(0, 4)
    .map(({ g, from, to }) => {
      const inside = tasks.filter((t) => t.ts < to && t.ts + t.dur > from);
      const busy = inside.reduce((s, t) => s + t.dur, 0) / 1000;
      return `${round(g, 0)} ms: GPU main busy ${round(busy, 0)} ms`;
    });
  return out;
}

/** Which threads of which Firefox processes were busy, by CPU ms, in
 * each of `windows` (page ms; `origin` the page's `timeOrigin`), from the
 * Gecko profile at `path`: thread names need no symbols. */
function profileThreads(path, origin, windows) {
  let raw = readFileSync(path);
  if (raw[0] === 0x1f) raw = gunzipSync(raw);
  const profile = JSON.parse(raw.toString("utf8"));
  const threads = [];
  const walk = (p) => {
    const unit = p.meta?.sampleUnits?.threadCPUDelta ?? "µs";
    const scale = unit === "ns" ? 1e-6 : unit === "µs" ? 1e-3 : 0;
    for (const t of p.threads ?? []) {
      const s = t.samples;
      if (!s?.data?.length) continue;
      const time = s.schema.time;
      const cpu = s.schema.threadCPUDelta;
      threads.push({
        name: `${t.processName ?? t.processType ?? "?"}:${t.name}`,
        start: p.meta.startTime,
        samples: s.data.map((d) => [
          d[time],
          cpu === undefined ? null : d[cpu],
        ]),
        scale,
        interval: p.meta.interval ?? 1,
      });
    }
    for (const sub of p.processes ?? []) walk(sub);
  };
  walk(profile);
  return windows.map(([a, b]) => {
    const [from, to] = [origin + a, origin + b];
    const busy = threads
      .map((t) => {
        let ms = 0;
        for (const [at, delta] of t.samples) {
          const epoch = t.start + at;
          if (epoch < from || epoch > to) continue;
          // Without CPU deltas, a sample stands for its interval.
          ms += delta === null || !t.scale ? t.interval : delta * t.scale;
        }
        return [t.name, ms];
      })
      .filter(([, ms]) => ms >= 2)
      .sort((x, y) => y[1] - x[1])
      .slice(0, 6)
      .map(([name, ms]) => `${name} ${round(ms, 0)}`);
    return `${Math.round(a)}–${Math.round(b)} ms: ${busy.join(", ") || "nothing busy"}`;
  });
}

/** What the last Orbis run in Firefox saw, for the profile's report. */
let profiled;

/** Serves `pkg` (a copy of web/pkg) or web/pkg itself as the demo. */
async function serve(pkg) {
  const live = fileURLToPath(new URL("../pkg/", import.meta.url));
  if (!pkg) return serveDemo();
  const saved = mkdtempSync(join(tmpdir(), "orbis-pkg-"));
  cpSync(live, saved, { recursive: true });
  try {
    cpSync(pkg, live, { recursive: true });
    return await serveDemo();
  } finally {
    cpSync(saved, live, { recursive: true });
    rmSync(saved, { recursive: true });
  }
}

for (let r = 0; r < runs; r++) {
  for (const [k, pkg] of pkgs.entries()) {
    const server = await serve(pkg);
    try {
      // The engines' order turns each run, so neither always goes first;
      // GL JS runs once a run.
      const these = (r % 2 ? [...engines].reverse() : engines).filter(
        (e) => e === "orbis" || k === 0,
      );
      for (const engine of these) {
        const report = await run(server, engine);
        if (profiled && existsSync(args.profile)) {
          const { origin, gaps, pipelines } = profiled;
          report.profile = {
            worst: profileThreads(args.profile, origin, gaps),
            // 300 ms from each pipeline made mid-move: its compile.
            pipelines: profileThreads(
              args.profile,
              origin,
              pipelines.map((p) => [p.at, p.at + 300]),
            ).map((line, i) => `${pipelines[i].label}: ${line}`),
          };
          profiled = undefined;
        }
        console.log(JSON.stringify({ run: r, ...(pkg && { pkg }), ...report }));
      }
    } finally {
      server.close();
    }
  }
}
