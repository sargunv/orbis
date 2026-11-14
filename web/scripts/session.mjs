// A user's session on a live style, as a user has it: a fresh browser (an
// empty HTTP cache and the browser's shader cache empty), the page opened,
// and real mouse input (wheel notches, drags, right-button tilts and turns)
// from soon after the map's first frame, while its tiles and pipelines
// still load, through the same moves in every engine. In headless Chromium,
// the installed Firefox (headed, on WebGPU: serve.mjs's `launchFirefox`) or
// Playwright's WebKit, in Orbis and the pinned GL JS on demo/load.html.
//
//   pnpm session [--session=manhattan|paris|alps|rotate|globe|nepal] [--browser=chromium|firefox|webkit]
//     [--engines=orbis,gljs] [--runs=n] [--style=key|url] [--size=WxH]
//     [--ratio=r] [--start=ms] [--input=ms] [--warm] [--driver] [--headed]
//     [--worst=n] [--backend=webgpu|webgl2] [--pkgs=a,b] [--fetches=a,b]
//     [--latency=ms] [--deep] [--terrain=0|1] [--gpu] [--dump=prefix]
//
// Sessions: `manhattan` opens Liberty over Midtown at z13 and wheel-zooms
// in to about z16 as the buildings load, tilts to 70, turns and drags
// across the island; `paris` tilts Liberty over central Paris from pitch
// 60 to the limit (85) and pans and turns there; `alps` drags, turns and
// zooms the Alps terrain style at pitch 72; `rotate` turns it slowly at
// pitch 75 to 85; `globe` wheel-zooms the demo's
// Globe entry (terrain on) over New York from z2 to about z8.5, tilting
// to the limit on the way; `nepal` flings the Globe entry in and out
// between z1 and z7 over the Himalaya at a spun wheel's and a trackpad's
// rates, its moves given in zoom levels and each engine's deltas sized
// to zoom them (Orbis's notch zooms twice GL JS's). The moves end where the hand stops (no fling),
// so every engine and browser ends at the same camera (`camera` in the
// report says where). `--style` gives a session's moves another catalogue
// key or style URL; `--terrain=0` turns a catalogue entry's terrain off
// (on by default, as the demo opens it). `--start` is how long after the first frame the input
// starts (500 ms); `--input` starts it that many ms after the page's
// navigation instead, whether the map has drawn or not, so engines and
// builds whose first frames come at different times meet the same hand
// (a first frame that waits shows in `moveToFrame`). `--warm` makes the session once untimed first and
// reloads, so tiles come from the HTTP cache and pipelines from the
// browser's. `--headed` runs Chromium in a window, at the display's
// refresh. `--dump` writes each engine's frames, input windows,
// presents, pipelines and the page thread's stalls (`beats`) to
// `<prefix>-<engine>.json`, each frame with its camera (`at`: lon, lat,
// zoom). `--backend` draws Orbis with that
// backend. `--pkgs` runs Orbis on each of these builds of web/pkg (whole
// copies of it, each in its place alone) in turn, run by run, for A/B
// runs in one session; GL JS runs once a run. `--fetches` runs Orbis with
// each of these `tileFetches` in turn, run by run, the same way (one
// build, so a cap is measured alone). `--latency` (Chromium) adds that
// many ms to every request's answer through DevTools' network
// conditions, the job workers' tile fetches too: a phone's network.
// In Chromium the report has the GPU process's main thread over the
// moves (`gpuProcess`: busy time, tasks over 50 ms, the longest), and with
// `--deep` (GPU service tracing) what its tasks and its slow ones did, its
// GL calls (ANGLE's waits on WebGL 2, #140) and its work in the worst
// present gaps. `--gpu` times Orbis's GPU work over the moves by its
// timestamp queries (`gpuTimes`: each frame's span, how many run past a
// 120 Hz and a 60 Hz frame, each pass's sum), its submissions in
// `--dump`'s file, and GL JS's by a WebGL 2 timer query around each frame
// it renders (one pass, `frame`), where the browser has them.
//
// For each run it reports when the first frame came, the map's frame
// intervals (as the page hears of them: Orbis's `render` events, GL JS's
// own), how long from each move's start the map's
// first frame came (`moveToFrame`: a move no frame answered shows only
// there), the page's animation-frame intervals that overlap a move (a
// stall that swallows a whole move counts) and, in Chromium, the GPU
// process's presents (what reached the screen) while input moves the
// camera, and how long the map took to settle after the last input. The
// pipelines' compiles are timed on the page: those over 100 ms of page
// frames that ran while one compiled (`compileStalls`), and when the last
// settled (`lastCompileMs`). For Orbis it names what each of the `--worst` longest
// intervals spent: the frame's render CPU by stage, pipelines still
// compiling, tiles arrived, bytes uploaded and GPU memory grown, and what
// none of that accounts for (`untimedMs`: the browser's GPU thread, the
// compositor, GC). Chromium also reports the page's long tasks.
// Orbis draws on the page's thread, as GL JS does, so the page's own frames
// show the map's work. It has its WebGPU calls timed (each slow frame's, and the session's, with
// the bytes each buffer took), its message handlers too, its calls over
// 5 ms listed by when (`slowCalls`), the pipelines it made during the
// moves (`pipelinesInMoves`: each a compile mid-gesture), and the async
// ones that finished compiling during them (`pipelinesSettledInMoves`:
// each a compile that ran into the moves). Its cold runs
// bust the driver's shader cache (`?nonce=`, cold.js), unless `--driver` keeps
// what the driver's cache holds from earlier runs (a user's later
// launches).
import {
  cpSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  launchChromium,
  launchFirefox,
  launchWebkit,
  serveDemo,
} from "./serve.mjs";

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
const inputAt = args.input == null ? null : Number(args.input);
const worstCount = Number(args.worst ?? 6);
const warm = !!args.warm;
const wheelScale = browserName === "firefox" ? ratio : 1;
const dump = args.dump;
const deep = !!args.deep;
const pkgs = args.pkgs ? args.pkgs.split(",").map((p) => resolve(p)) : [null];
const fetchCaps = args.fetches ? args.fetches.split(",").map(Number) : [null];
const latency = Number(args.latency ?? 0);

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

/** The engine whose session runs, for moves given in zoom levels. */
let engineNow = "orbis";
/** How much of a wheel's or a trackpad's pixels the page hears: headless
 * Chromium halves them (its report's `firstWheel`). */
const heard = browserName === "chromium" && !args.headed ? 0.5 : 1;

/** Zooms `levels` in (out, negative) with wheel notches `ms` apart, each a
 * notch Orbis zooms a third of a level by (100 px heard, at mln-rs's
 * 1/300 a pixel); GL JS, whose notches zoom less (its logistic 1/450 a
 * pixel), gets them 2.2 times as large, so that both cameras take the
 * same path. */
function fling(page, levels, ms) {
  const n = Math.round(Math.abs(levels) * 3);
  const px = (100 / heard) * (engineNow === "gljs" ? 2.2 : 1);
  return wheel(page, c, -Math.sign(levels) * px, n, ms);
}

/** Zooms `levels` with a trackpad's scroll: 8 px every 8 ms (Orbis's
 * 1/300 a pixel); GL JS reads it as a trackpad, at twice that rate, and
 * gets half. Dispatched on the page from a timer there: Playwright's
 * wheel events carry a whole detent's legacy `wheelDeltaY`, which reads
 * as a mouse wheel's notch however small the delta. */
function scroll(page, levels) {
  const n = Math.round((Math.abs(levels) * 300) / 8);
  const px = 8 * (engineNow === "gljs" ? 0.5 : 1) * -Math.sign(levels);
  return page.evaluate(
    ([x, y, n, px]) =>
      new Promise((resolve) => {
        const start = performance.now();
        let i = 0;
        const send = () => {
          const init = { deltaY: px, deltaMode: 0, clientX: x, clientY: y };
          const at = document.elementFromPoint(x, y);
          at.dispatchEvent(
            new WheelEvent("wheel", {
              ...init,
              bubbles: true,
              cancelable: true,
            }),
          );
          if (++i >= n) return resolve();
          setTimeout(send, Math.max(0, start + i * 8 - performance.now()));
        };
        send();
      }),
    [...c, n, px],
  );
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
  // Turned slowly at pitch 75 to 85 over the Alps terrain style: what a
  // turn alone changes in the terrain's cover and its drape (the native
  // session also reports how still the cover held).
  rotate: {
    style: "alps",
    at: [...alps.center, 12, alps.bearing, 78],
    moves: [
      (p) => drag(p, c, [300, 0], 4000, "right"),
      (p) => drag(p, c, [0, -60], 600, "right"),
      (p) => drag(p, c, [300, 0], 4000, "right"),
      (p) => drag(p, c, [0, 20], 600, "right"),
      (p) => drag(p, c, [-300, 0], 4000, "right"),
    ],
  },
  // Flung in and out over the Himalaya on the Globe entry (terrain on),
  // between the whole globe (z1) and regional zooms (z7), twice: a spun
  // mouse wheel (a notch every 16-25 ms) and a trackpad's scroll (small
  // deltas every 8 ms), barely pausing between flings.
  nepal: {
    style: "globe",
    at: [84, 28, 1, 0, 0],
    pause: 150,
    moves: [1, 2].flatMap(() => [
      (p) => fling(p, 6, 20),
      (p) => fling(p, -6, 20),
      (p) => fling(p, 6, 16),
      (p) => fling(p, -3, 16),
      (p) => fling(p, 3, 25),
      (p) => fling(p, -6, 16),
      (p) => scroll(p, 6),
      (p) => scroll(p, -6),
    ]),
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

/** Intervals between successive `times` that overlap one of `windows`:
 * for the page's animation frames, which come on every display frame,
 * so a stall that swallows a whole move (no frame inside it at all) is
 * one interval too. */
function overlapping(times, windows) {
  const out = [];
  for (let i = 1; i < times.length; i++) {
    const [s, t] = [times[i - 1], times[i]];
    if (windows.some(([a, b]) => s < b && t > a))
      out.push({ i, s, gap: t - s });
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

/** Orbis's tile fetches over `frames` (each frame's since the one before):
 * started, aborted once started and completed, and those started after
 * `lastInput`. */
function fetchTotals(frames, lastInput) {
  const sum = (i, from = -Infinity) =>
    frames
      .filter((f) => f.t > from)
      .reduce((s, f) => s + (f.fetches?.[i] ?? 0), 0);
  return {
    started: sum(0),
    aborted: sum(1),
    completed: sum(2),
    startedAfterInput: sum(0, lastInput),
  };
}

/** What Orbis's device did over `frames` (each frame's counts since the
 * one before): pipelines asked for, buffers and textures made and their
 * bytes, bytes copied on the GPU, uploads and their calls, texture-write
 * submissions, render commands encoded, drape slots redrawn. */
function deviceTotals(frames) {
  const sum = (f) => frames.reduce((s, x) => s + (f(x) ?? 0), 0);
  return {
    frames: frames.length,
    // Pipelines asked for, and made: a compile finished (on the web the
    // browser's async compile and wgpu's own creation after it).
    pipelinesAsked: sum((f) => f.device.pipelinesRequested),
    pipelines: sum((f) => f.device.pipelinesCompiled),
    created: sum((f) => f.device.buffersCreated + f.device.texturesCreated),
    allocatedMiB: mib(sum((f) => f.device.bytesAllocated)),
    copiedMiB: mib(sum((f) => f.device.bytesCopied)),
    // Of those, what pools, tables, tile pages and the drape atlas made
    // and copied moving into memory of another size: none while moving.
    poolMoves: sum((f) => f.device.poolMoves),
    poolAllocatedMiB: mib(sum((f) => f.device.poolBytesAllocated)),
    poolCopiedMiB: mib(sum((f) => f.device.poolBytesCopied)),
    uploadMiB: mib(sum((f) => f.device.uploadBytes)),
    uploadCalls: sum((f) => f.device.uploadCalls),
    textureWriteSubmissions: sum((f) => f.device.textureWriteSubmissions),
    renderCommands: sum((f) => f.device.renderCommands),
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
    untimedMs: round(Math.max(0, gap - f.ms)),
    compiling: f.pending,
    layersWaiting: f.waiting,
    arrived: f.arrived,
    meshes: f.meshes - prev.meshes,
    jobs: f.jobs - prev.jobs,
    uploadedMiB: mib(f.uploaded - prev.uploaded),
    gpuGrewMiB: mib(f.gpuBytes - prev.gpuBytes),
    ...(f.device && {
      pipelines: f.device.pipelinesCompiled,
      allocatedMiB: mib(f.device.bytesAllocated),
      uploadCalls: f.device.uploadCalls,
      textureWriteSubmissions: f.device.textureWriteSubmissions,
      renderCommands: f.device.renderCommands,
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
    // Timers to the microsecond, for the WebGPU tally: Firefox rounds
    // them to the millisecond otherwise.
    const browser = await launchFirefox({
      prefs: { "privacy.reduceTimerPrecision": false },
    });
    if (!browser.webgpu) console.error("no installed Firefox: WebGL 2");
    return browser;
  }
  if (browserName === "chromium") {
    // `--gpu`'s timestamps unquantized need WebGPU's unsafe APIs.
    if (!deep)
      return launchChromium({ unsafe: !!args.gpu, headed: !!args.headed });
    const { chromium } = await import("playwright");
    const { chromiumPlatform } = await import("./serve.mjs");
    const { executablePath, args: flags } = chromiumPlatform();
    return chromium.launch({
      headless: !args.headed,
      executablePath,
      args: [...flags, "--enable-gpu-service-tracing"],
    });
  }
  if (browserName === "webkit") return launchWebkit({ headed: !!args.headed });
  throw new Error(`unknown browser ${browserName}`);
}

/** One session in `engine` on a fresh browser: the run's report. */
async function run(server, engine, fetches) {
  engineNow = engine;
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
    // A cold run's shaders, unique to it past the browser's and the
    // driver's caches (cold.js).
    const nonce =
      engine === "orbis" && !warm && !args.driver
        ? 1 + Math.floor(Math.random() * 9999)
        : 0;
    const query = () =>
      new URLSearchParams({
        engine,
        style,
        at: session.at.join(","),
        frames: "",
        ...(args.terrain && { terrain: args.terrain }),
        ...(engine === "orbis" && args.backend && { backend: args.backend }),
        ...(nonce && { nonce }),
        ...(engine === "orbis" && fetches && { fetches }),
      });
    if (latency && browserName === "chromium") {
      const network = await context.newCDPSession(page);
      await network.send("Network.enable");
      await network.send("Network.emulateNetworkConditions", {
        offline: false,
        latency,
        downloadThroughput: -1,
        uploadThroughput: -1,
      });
    }
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
      // Modules made on the calling thread: a feature module's compile
      // or instantiation (builds.ts).
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
          const pipeline = {
            at: Math.round(performance.now()),
            label: desc?.label ?? "?",
            // Its stages' override constants and targets: which variant.
            what: JSON.stringify([
              desc?.vertex?.constants ?? desc?.compute?.constants,
              desc?.fragment?.constants,
              desc?.fragment?.targets?.map((t) => t.format),
              desc?.multisample?.count,
            ]),
            async: name.endsWith("Async"),
          };
          made.push(pipeline);
          const result = original.call(this, desc);
          // When an async compile finished: one that runs into the moves
          // may hold frames back where the browser compiles on the thread
          // that presents.
          if (pipeline.async)
            result.then(
              () => (pipeline.settled = Math.round(performance.now())),
              () => {},
            );
          return result;
        };
      }
      wrap(GPUCommandEncoder.prototype, "finish");
      // GPU-side fills: no bytes cross the queue, but each is a call.
      wrap(
        GPUCommandEncoder.prototype,
        "clearBuffer",
        (a) => a[2] ?? a[0].size - (a[1] ?? 0),
      );
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
        // deltaMode first: Firefox reports pixels to a page that reads a
        // delta before it, the map's listener included.
        (e) => {
          const mode = e.deltaMode;
          globalThis.__wheel ??= [e.deltaY, mode];
        },
        { capture: true, passive: true },
      );
      globalThis.__raf = [];
      const tick = (t) => {
        globalThis.__raf.push(t);
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
      // The page's thread, beside its animation frames: a timer every
      // 10 ms, and each gap over 50 ms between them (the thread blocked,
      // where a gap in the animation frames alone is the browser holding
      // them back).
      globalThis.__beats = [];
      let beat = performance.now();
      setInterval(() => {
        const t = performance.now();
        if (t - beat > 50) globalThis.__beats.push([beat, t - beat]);
        beat = t;
      }, 10);
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
      // With `--input`, until the input's time instead of the map's
      // first frame.
      await page.waitForFunction(
        (at) =>
          (at == null
            ? globalThis.mapFrames?.length > 0
            : performance.now() >= at) || document.title.startsWith("error"),
        inputAt,
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
        await sleep(session.pause ?? 400);
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
    if (inputAt == null) await sleep(startAfter);
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
    // GL JS's by WebGL 2 timer queries, where the browser has them.
    const timing =
      !!args.gpu && (await page.evaluate(() => !!globalThis.mapGpu));
    if (timing) await page.evaluate(() => globalThis.mapGpu.start());
    const windows = await moves();
    const submissions =
      timing && (await page.evaluate(() => globalThis.mapGpu.take()));
    const gpuTimes = timing && gpuPasses(submissions);
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
      beats,
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
      beats: globalThis.__beats,
      longTasks: globalThis.__long,
      where: globalThis.loadDetails?.(),
    }));
    const times = frames.map((f) => f.t);
    const mapGaps = intervals(times, windows);
    const pageGaps = overlapping(raf, windows);
    const report = {
      session: sessionName,
      style,
      browser: browserName,
      engine,
      ...(where && { backend: where.backend }),
      warm,
      size: `${width}x${height}@${ratio}`,
      firstFrameMs: Math.round(times[0] ?? -1),
      inputSeconds: round(windows.reduce((s, [a, b]) => s + b - a, 0) / 1000),
      mapFrames: pacing(mapGaps.map((g) => g.gap)),
      // From each move's start to the map's first frame after it: one
      // over a frame or two is a move the map didn't answer, which
      // `mapFrames` (intervals inside the moves) can't show when no frame
      // came in the move at all.
      moveToFrame: pacing(
        windows.map(([a]) => (times.find((t) => t >= a) ?? a) - a),
      ),
      ...(screen && {
        presents: pacing(intervals(screen, windows).map((g) => g.gap)),
      }),
      ...(gpuProcess && { gpuProcess }),
      ...(gpuTimes && { gpuTimes }),
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
        pipelinesSettledInMoves: pipelines
          .filter((p) => p.settled >= windows[0][0])
          .map((p) => `${p.settled} ${p.label} (asked at ${p.at})`),
        // The page's frames over 100 ms in the moves while an async
        // compile ran (asked before the frame's end, settled after its
        // start): where the browser compiles on the thread that presents
        // (Firefox), each holds the frame back.
        compileStalls: pageGaps
          .filter((g) => g.gap > 100)
          .map((g) => {
            const end = g.s + g.gap;
            const running = pipelines.filter(
              (p) => p.async && p.at < end && (p.settled ?? end) > g.s,
            );
            return running.length
              ? `${Math.round(g.s)}+${Math.round(g.gap)}: ${running.map((p) => p.label).join(", ")}`
              : null;
          })
          .filter(Boolean),
        // When the last compile settled: past the first move's start,
        // the browser compiled into the moves.
        lastCompileMs: Math.max(0, ...pipelines.map((p) => p.settled ?? 0)),
      }),
      // When the page fetched each wasm file (a feature module's, as the
      // style declared it).
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
        // Tile fetches started, aborted once started (the view left
        // their tiles) and completed, and those started after the last input.
        fetches: fetchTotals(frames, lastInput),
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
      // The first errors and the last, with their stacks: what broke, and
      // what that left the map doing.
      report.errors = [
        ...errors.slice(0, 3),
        ...errors.slice(Math.max(3, errors.length - 2)),
      ].map((e) => e.split("\n").slice(0, 12).join("\n"));
    if (errors.length > 5) report.errorCount = errors.length;
    if (dump)
      writeFileSync(
        `${dump}-${engine}.json`,
        JSON.stringify({
          windows,
          frames,
          raf,
          beats,
          screen,
          pipelines,
          submissions,
        }),
      );
    return report;
  } finally {
    await browser.close();
  }
}

/** Orbis's GPU over the moves (`--gpu`), from its submissions'
 * timestamps: each drawing submission's span (its first pass's start, or
 * the end of the pass before if later, to its last one's end), how many
 * spans run past a 120 Hz and a 60 Hz frame, and each pass label's GPU
 * milliseconds summed and at most. */
function gpuPasses(submissions) {
  const drawn = submissions.filter((s) => s.passes.length);
  const by = {};
  for (const s of drawn)
    for (const [label, ms] of s.passes) {
      const b = (by[label] ??= { n: 0, ms: 0, max: 0 });
      b.n++;
      b.ms += ms;
      b.max = Math.max(b.max, ms);
    }
  return {
    spans: pacing(drawn.map((s) => s.span)),
    over8: drawn.filter((s) => s.span > 8.3).length,
    over16: drawn.filter((s) => s.span > 16.7).length,
    totalMs: Math.round(drawn.reduce((t, s) => t + s.span, 0)),
    passes: Object.entries(by)
      .sort((a, b) => b[1].ms - a[1].ms)
      .slice(0, 14)
      .map(
        ([label, b]) =>
          `${label}: ${b.n}× ${round(b.ms, 0)} ms (max ${round(b.max)})`,
      ),
  };
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
  // Everything its tasks did, by event (nested events count in each).
  out.inTasks = tally(
    onMain.filter((e) => e.name !== "ThreadControllerImpl::RunTask"),
  );
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

/** Serves `pkg` (a copy of web/pkg) or web/pkg itself as the demo. */
async function serve(pkg) {
  const live = fileURLToPath(new URL("../pkg/", import.meta.url));
  if (!pkg) return serveDemo();
  const saved = mkdtempSync(join(tmpdir(), "orbis-pkg-"));
  cpSync(live, saved, { recursive: true });
  // The build is `pkg` whole: copied over web/pkg, a file it lacks would
  // come from the build there, a feature module from another build among
  // them.
  const replace = (from) => {
    rmSync(live, { recursive: true, force: true });
    cpSync(from, live, { recursive: true });
  };
  try {
    replace(pkg);
    return await serveDemo();
  } finally {
    replace(saved);
    rmSync(saved, { recursive: true });
  }
}

const variants = pkgs.flatMap((pkg) =>
  fetchCaps.map((fetches) => [pkg, fetches]),
);
for (let r = 0; r < runs; r++) {
  for (const [k, [pkg, fetches]] of variants.entries()) {
    const server = await serve(pkg);
    try {
      // The engines' order turns each run, so neither always goes first;
      // GL JS runs once a run.
      const these = (r % 2 ? [...engines].reverse() : engines).filter(
        (e) => e === "orbis" || k === 0,
      );
      for (const engine of these) {
        const report = await run(server, engine, fetches);
        const variant = engine === "orbis" && {
          ...(pkg && { pkg }),
          ...(fetches && { fetches }),
        };
        console.log(JSON.stringify({ run: r, ...variant, ...report }));
      }
    } finally {
      server.close();
    }
  }
}
