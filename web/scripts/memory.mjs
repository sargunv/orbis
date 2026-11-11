// Browser memory in headless Chromium: Orbis, the pinned GL JS and MapLibre
// Native's browser build (maplibre-native-ffi's WebGL 2 backend, as the
// demo's comparison loads it) on demo/load.html, once each view is idle and
// again after a gesture tour (load.js's `tour`: 6 s of pan, zoom, turn and
// tilt, a camera a display frame, until idle).
//
//   pnpm memory [--cache=<cache-style.db> | --style=<url>] [--engines=orbis,gljs,native]
//     [--views=a,b] [--runs=1] [--backend=webgl2]
//
// By default it loads MapLibre Native's five real-world render tests at
// 512×512, served offline from the render-test cache in the mln-rs
// checkout `cargo xtask h2h` builds (load.mjs's `--cache`); `--style` loads
// mln-rs's Liberty views of that style at 1024×768 instead.
//
// Each sample is a fresh browser, so its GPU process holds one engine's
// map. After a garbage collection in every context, it reads:
// - JS heap: V8's used heap, summed over the page and its workers;
// - buffers: their ArrayBuffers' backing stores, wasm memories aside;
// - wasm: every wasm memory the page and its workers made or instantiated,
//   one shared by threads (Native's) once: their sizes, which count pages
//   never touched (Native's memory starts at 512 MB), where the renderer's
//   footprint counts only what's resident;
// - renderer: the renderer processes' physical footprint (macOS
//   `footprint`), page, workers and wasm together;
// - GPU: the GPU process's graphics memory (footprint's IOAccelerator,
//   IOSurface and graphics categories) and whole footprint, less what it
//   held at launch, once it holds still (it falls by tens of MB in the
//   second or two after the map settles): an estimate of the engine's GPU
//   memory, its WebGPU or WebGL device's own included.
// Orbis also reports what it allocated itself (buffers and textures) and,
// on WebGL 2 (`--backend=webgl2`, `createMap`'s `backend`), what its CPU
// driver keeps in wasm memory.
import { execFileSync } from "node:child_process";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { launchChromium, serveDemo } from "./serve.mjs";
import {
  extractCache,
  median,
  realWorldCache,
  VIEWS as CAMERAS,
} from "./views.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ??
  fallback;
const style = option("style");
const cache = style ? undefined : (option("cache") ?? realWorldCache());
if (!style && !cache)
  throw new Error(
    "no render-test cache: build `cargo xtask h2h size` once, or give --cache=<cache-style.db> or --style=<url>",
  );
const engines = option("engines", "orbis,gljs,native").split(",");
const VIEWS = CAMERAS[cache ? "realWorld" : "liberty"];
const views = option("views", Object.keys(VIEWS).join(",")).split(",");
const runs = Number(option("runs", "1"));
const backend = option("backend");
const viewport = cache
  ? { width: 512, height: 512 }
  : { width: 1024, height: 768 };
const MB = 2 ** 20;

/** A process's physical footprint and its graphics memory, in bytes, from macOS `footprint`. */
function footprint(pid) {
  if (process.platform !== "darwin") return { total: NaN, graphics: NaN };
  const text = execFileSync("footprint", ["-f", "bytes", "-p", String(pid)], {
    encoding: "utf8",
  });
  let graphics = 0;
  for (const line of text.split("\n")) {
    const row = line.match(/^\s*(\d+) B\s+\d+ B\s+\d+ B\s+\d+\s+(.*)$/);
    if (row && /graphics|IOAccelerator|IOSurface/.test(row[2]))
      graphics += Number(row[1]);
  }
  return {
    total: Number(text.match(/phys_footprint: (\d+) B/)?.[1] ?? NaN),
    graphics,
  };
}

/** The browser's processes' footprints: the renderers' summed, and the GPU process's. */
async function processes(browser) {
  const session = await browser.newBrowserCDPSession();
  const { processInfo } = await session.send("SystemInfo.getProcessInfo");
  await session.detach();
  const renderer = processInfo
    .filter((p) => p.type === "renderer")
    .reduce((sum, p) => sum + footprint(p.id).total, 0);
  const gpu = processInfo.find((p) => p.type === "GPU");
  return {
    renderer,
    gpu: gpu ? footprint(gpu.id) : { total: NaN, graphics: NaN },
  };
}

/**
 * Keeps every wasm memory the context makes or instantiates in
 * `__wasmMemories`, where `WASM_BYTES` finds them: wasm memory isn't in
 * V8's heap figures, and a module's own memory is nowhere a page can reach.
 */
const WASM_HOOK = `(() => {
  if (globalThis.__wasmMemories) return;
  const memories = globalThis.__wasmMemories = new Set();
  const W = WebAssembly, { Instance, Memory } = W;
  const keep = (instance) => { for (const v of Object.values(instance.exports)) if (v instanceof Memory) memories.add(v); return instance; };
  W.Memory = new Proxy(Memory, { construct: (t, a) => { const m = Reflect.construct(t, a); memories.add(m); return m; } });
  W.Instance = new Proxy(Instance, { construct: (t, a) => keep(Reflect.construct(t, a)) });
  for (const name of ["instantiate", "instantiateStreaming"]) {
    const original = W[name];
    W[name] = function (...a) { return original.apply(this, a).then((r) => { keep(r instanceof Instance ? r : r.instance); return r; }); };
  }
})()`;
/** The context's wasm memories' bytes: its own, and those shared between threads. */
const WASM_BYTES = `[...(globalThis.__wasmMemories ?? [])].map((m) => [m.buffer.byteLength, m.buffer instanceof SharedArrayBuffer])`;

/**
 * CDP on the page and, through it, on each of its workers, each with
 * `WASM_HOOK` in place before its scripts run. Resolves to a function that
 * sends a method with its params to every context and resolves to their
 * results.
 */
async function sessions(context, page) {
  await page.addInitScript(WASM_HOOK);
  const cdp = await context.newCDPSession(page);
  const workers = new Set();
  const pending = new Map();
  let id = 0;
  // A worker that ends with a message in flight never answers it.
  const send = (sessionId, method, params = {}) =>
    new Promise((resolve) => {
      pending.set(++id, resolve);
      setTimeout(() => resolve({}), 10_000);
      cdp
        .send("Target.sendMessageToTarget", {
          sessionId,
          message: JSON.stringify({ id, method, params }),
        })
        .catch(() => resolve({}));
    });
  cdp.on("Target.attachedToTarget", async ({ sessionId }) => {
    workers.add(sessionId);
    await send(sessionId, "Runtime.evaluate", { expression: WASM_HOOK });
    await send(sessionId, "Runtime.runIfWaitingForDebugger");
  });
  cdp.on("Target.detachedFromTarget", (e) => workers.delete(e.sessionId));
  cdp.on("Target.receivedMessageFromTarget", (e) => {
    const message = JSON.parse(e.message);
    pending.get(message.id)?.(message.result ?? {});
    pending.delete(message.id);
  });
  await cdp.send("Target.setAutoAttach", {
    autoAttach: true,
    waitForDebuggerOnStart: true,
    flatten: false,
  });
  return (method, params) =>
    Promise.all([
      cdp.send(method, params),
      ...[...workers].map((s) => send(s, method, params)),
    ]);
}

async function sample(server, engine, view) {
  const browser = await launchChromium({ unsafe: false });
  try {
    const launched = await processes(browser);
    const context = await browser.newContext({
      viewport,
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    const all = await sessions(context, page);
    const styleUrl = cache
      ? `${server.url}real-world/styles/${view}.json`
      : style;
    await page.goto(
      `${server.url}load.html?${new URLSearchParams({ engine, style: styleUrl, at: VIEWS[view], ...(backend && engine === "orbis" && { backend }) })}`,
    );
    await page.waitForFunction(
      () => /^(done|error)/.test(document.title),
      null,
      { timeout: 120_000 },
    );
    const title = await page.title();
    if (!title.startsWith("done"))
      throw new Error(`${engine} ${view}: ${title}`);
    const read = async () => {
      await page.waitForTimeout(1000);
      await all("HeapProfiler.collectGarbage");
      const heaps = await all("Runtime.getHeapUsage");
      const memories = (
        await all("Runtime.evaluate", {
          expression: WASM_BYTES,
          returnByValue: true,
        })
      ).flatMap((r) => r.result?.value ?? []);
      // The GPU process lets go of tens of MB a second or two after a load
      // or a tour settles: read the footprints until they hold still.
      let now = await processes(browser);
      for (let i = 0; i < 5; i++) {
        await page.waitForTimeout(1000);
        const next = await processes(browser);
        const still =
          Math.abs(next.gpu.graphics - now.gpu.graphics) < MB &&
          Math.abs(next.renderer - now.renderer) < MB;
        now = next;
        if (still) break;
      }
      return {
        heap: heaps.reduce((s, h) => s + (h.usedSize ?? 0), 0),
        buffers: heaps.reduce((s, h) => s + (h.backingStorageSize ?? 0), 0),
        // Threads that share a memory each see it: it counts once.
        wasm:
          memories
            .filter(([, shared]) => !shared)
            .reduce((s, [bytes]) => s + bytes, 0) +
          Math.max(
            0,
            ...memories.filter(([, shared]) => shared).map(([bytes]) => bytes),
          ),
        contexts: heaps.length,
        renderer: now.renderer,
        gpu: now.gpu.total - launched.gpu.total,
        graphics: now.gpu.graphics - launched.gpu.graphics,
        own: await page.evaluate(() => window.gpuBytes?.()),
        cpu: await page.evaluate(() => window.cpuCopies?.()),
      };
    };
    console.error(`memory: ${engine} ${view}: loaded`);
    const loaded = await read();
    console.error(`memory: ${engine} ${view}: touring`);
    await page.evaluate(() =>
      Promise.race([
        window.tour(),
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error("not idle 120 s after the tour")),
            120_000,
          ),
        ),
      ]),
    );
    const toured = await read();
    if (errors.length)
      console.error(
        `memory: ${engine} ${view}: ${errors.slice(0, 3).join("; ")}`,
      );
    return { loaded, toured };
  } finally {
    await browser.close();
  }
}

const out = fileURLToPath(new URL("../../target/h2h/web/", import.meta.url));
mkdirSync(out, { recursive: true });
const extracted = cache ? join(out, "real-world/") : undefined;
const server = await serveDemo({
  isolated: true,
  mounts: extracted ? { "real-world/": extracted } : {},
});
if (extracted) extractCache(cache, extracted, `${server.url}real-world/`);
const rows = [];
try {
  for (let run = 0; run < runs; run++) {
    for (const view of views) {
      for (const engine of engines) {
        const row = { engine, view, run };
        try {
          Object.assign(row, await sample(server, engine, view));
        } catch (e) {
          row.error = String(e?.message ?? e).slice(0, 300);
        }
        console.error(`memory: ${JSON.stringify(row)}`);
        rows.push(row);
      }
    }
  }
} finally {
  server.close();
}

const mb = (x) => (Number.isFinite(x) ? (x / MB).toFixed(1) : "—");
const cell = (engine, view, phase, key) =>
  median(
    rows
      .filter((r) => r.engine === engine && r.view === view && r[phase])
      .map((r) => r[phase][key])
      .filter((x) => x !== undefined && x !== null),
  );
const KEYS = [
  ["heap", "JS heap"],
  ["buffers", "buffers"],
  ["wasm", "wasm"],
  ["renderer", "renderer"],
  ["graphics", "GPU graphics Δ"],
  ["gpu", "GPU process Δ"],
  ["own", "Orbis's own GPU"],
  ["cpu", "Orbis's CPU copies"],
];
const lines = [
  `### Browser memory, MB (Chromium, ${viewport.width}×${viewport.height} @1×${backend === "webgl2" ? ", Orbis on WebGL 2" : ""}; ${runs > 1 ? `medians of ${runs} runs` : "1 run"})`,
  "",
  `| engine | phase | view | ${KEYS.map(([, name]) => name).join(" | ")} |`,
  `|---|---|---|${KEYS.map(() => "---|").join("")}`,
];
for (const engine of engines) {
  for (const [phase, label] of [
    ["loaded", "loaded"],
    ["toured", "after tour"],
  ]) {
    for (const view of views) {
      const failed = rows.find(
        (r) => r.engine === engine && r.view === view && r.error,
      );
      if (
        failed &&
        !rows.some((r) => r.engine === engine && r.view === view && r[phase])
      ) {
        lines.push(
          `| ${engine} | ${label} | ${view} | failed: ${failed.error} |`,
        );
        continue;
      }
      lines.push(
        `| ${engine} | ${label} | ${view} | ${KEYS.map(([key]) => mb(cell(engine, view, phase, key))).join(" | ")} |`,
      );
    }
    if (views.length > 1) {
      const mean = (key) =>
        views.reduce((s, view) => s + cell(engine, view, phase, key), 0) /
        views.length;
      lines.push(
        `| **${engine}** | **${label}** | **mean** | ${KEYS.map(([key]) => `**${mb(mean(key))}**`).join(" | ")} |`,
      );
    }
  }
}
lines.push(
  "",
  `JS heap, buffers and wasm memory: summed over the page and its workers, after a garbage collection; renderer: the renderer processes' footprint; ` +
    `GPU: the GPU process's graphics memory and footprint less its own at launch, an estimate of the engine's GPU memory. ` +
    `Style: ${cache ? `the real-world render tests, from ${cache}` : style}.`,
);
console.log(lines.join("\n"));
