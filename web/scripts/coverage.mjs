// Which of core.wasm's functions a map's first frame runs, by scene, and
// what a module split along that line would weigh: the measure for
// whether lazily loaded wasm modules would earn their complexity.
//
//   cargo xtask web --names
//   pnpm coverage [--scenes=liberty,paris,globe,terrain,raster,tokyo,
//     tokyo-mlt,protomaps,versatiles] [--tour] [--out=<dir>] [--reuse]
//
// `--names` leaves `target/web/core-names.wasm`: core.wasm as `cargo xtask
// web` optimizes it, with its function names. binaryen's `wasm-split
// --instrument` adds a first-call stamp to each function of it, and each
// scene loads demo/load.html in headless Chromium with that module in
// place of core.wasm, once to its first idle (so the map's style, tiles,
// layout, placement and first frames ran, in the page and every worker);
// with `--tour`, through load.js's gesture tour after that too (pan, zoom,
// turn and tilt, until idle again). Every instance then writes its
// profile, and the scene's is their union. `--reuse` reports on the
// profiles already in `--out` instead.
//
// It prints, per scene and for the union of them all:
// - the share of functions and code bytes that ran, and the brotli sizes
//   of `wasm-split`'s two modules for that profile: what ran, which a
//   first frame needs, and the rest, which could load later;
// - the code only that scene ran, and that none ran, split out of the
//   rest: what a module of that feature would take off every other load;
// - code by crate and by module of ours, with the share each scene ran (a
//   function's crate is the first `orb_` crate in its name, else its first).
// Profiles and modules stay in `--out` (by default target/coverage), for
// `wasm-split --split --profile=…` runs of your own.
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliCompressSync, constants } from "node:zlib";
import { launchChromium, serveDemo } from "./serve.mjs";
import { VIEWS } from "./views.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const root = fileURLToPath(new URL("../../", import.meta.url));
const named = join(root, "target/web/core-names.wasm");
if (!existsSync(named)) throw new Error("no target/web/core-names.wasm: run `cargo xtask web --names` (a build without it removes it)");
const out = option("out", join(root, "target/coverage"));
mkdirSync(out, { recursive: true });

/** Each scene's load.html query: a style and a camera. */
const SCENES = {
  liberty: { style: "liberty", at: VIEWS.liberty.ams13 },
  paris: { style: "liberty", at: VIEWS.liberty.paris14p60 },
  globe: { style: "globe", at: VIEWS.globe.europe3p40 },
  terrain: { style: "alps", at: VIEWS.terrain.alps11p72 },
  raster: { style: "osm", at: "-0.12,51.51,12,0,0" },
  tokyo: { style: "tokyo-mvt", at: "139.74,35.687,15,0,0" },
  "tokyo-mlt": { style: "tokyo-mlt", at: "139.74,35.687,15,0,0" },
  // Other schemas and styles than OpenFreeMap's.
  protomaps: { style: "protomaps-light", at: VIEWS.liberty.ams13 },
  versatiles: { style: "versatiles-colorful", at: VIEWS.liberty.ams13 },
};
const tour = args.includes("--tour");
const scenes = option("scenes", Object.keys(SCENES).join(",")).split(",");
for (const s of scenes) if (!SCENES[s]) throw new Error(`--scenes: no ${s}; ${Object.keys(SCENES).join(", ")}`);

// The features core.wasm is built with (xtask/src/web.rs).
const FEATURES = ["simd", "bulk-memory", "sign-ext", "nontrapping-float-to-int", "multivalue", "reference-types"]
  .map((f) => `--enable-${f}`);
const split = (...a) => execFileSync("wasm-split", [...FEATURES, ...a], { stdio: ["ignore", "pipe", "inherit"], maxBuffer: 1 << 28 });
const brotli = (bytes) => brotliCompressSync(bytes, {
  params: { [constants.BROTLI_PARAM_QUALITY]: 11, [constants.BROTLI_PARAM_LGWIN]: 22 },
}).length;
const kib = (bytes) => `${(bytes / 1024).toFixed(0)} KiB`;
const pct = (part, whole) => `${((part * 100) / whole).toFixed(1)}%`;

/** The module's defined functions, in index order, as { name, size } (code bytes). */
function functions(wasm) {
  let at = 8;
  const u32 = () => { let v = 0, shift = 0, b; do { b = wasm[at++]; v |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80); return v >>> 0; };
  const name = () => { const n = u32(); const s = new TextDecoder().decode(wasm.subarray(at, at + n)); at += n; return s; };
  let imported = 0;
  const sizes = [], names = new Map();
  while (at < wasm.length) {
    const id = wasm[at++], size = u32(), end = at + size;
    if (id === 2) {
      for (let n = u32(); n--;) {
        name(); name();
        const kind = wasm[at++];
        if (kind === 0) { u32(); imported++; }
        else if (kind === 1) { at++; const flags = u32(); u32(); if (flags & 1) u32(); }
        else if (kind === 2) { const flags = u32(); u32(); if (flags & 1) u32(); }
        else if (kind === 3) { at += 2; }
        else { at++; u32(); }
      }
    } else if (id === 10) {
      for (let n = u32(); n--;) { const body = u32(); sizes.push(body); at += body; }
    } else if (id === 0 && name() === "name") {
      while (at < end) {
        const sub = wasm[at++], length = u32(), next = at + length;
        if (sub === 1) for (let n = u32(); n--;) { const index = u32(); names.set(index, name()); }
        at = next;
      }
    }
    at = end;
  }
  // binaryen calls a function without a name by its index among those defined.
  return sizes.map((size, i) => ({ name: names.get(imported + i) ?? String(i), size }));
}

const funcs = functions(readFileSync(named));

/** A profile's first-call stamps, one per defined function (wasm-split's format: an 8-byte module hash, then u32s). */
const stamps = (profile) => new Uint32Array(profile.buffer.slice(profile.byteOffset + 8, profile.byteOffset + profile.length));

/** The union of profiles of one module: a function ran if it ran in any. */
function union(profiles) {
  const result = Buffer.from(profiles[0]);
  const all = stamps(result);
  for (const p of profiles.slice(1)) stamps(p).forEach((s, i) => { if (s && !all[i]) all[i] = s; });
  Buffer.from(all.buffer).copy(result, 8);
  return result;
}

/** Keeps every wasm instance a context makes in `__wasmInstances`. */
const HOOK = `(() => {
  if (globalThis.__wasmInstances) return;
  const instances = globalThis.__wasmInstances = [];
  const W = WebAssembly, { Instance } = W;
  const keep = (instance) => { instances.push(instance); return instance; };
  W.Instance = new Proxy(Instance, { construct: (t, a) => keep(Reflect.construct(t, a)) });
  for (const name of ["instantiate", "instantiateStreaming"]) {
    const original = W[name];
    W[name] = function (...a) { return original.apply(this, a).then((r) => { keep(r instanceof Instance ? r : r.instance); return r; }); };
  }
})()`;

/** Each instrumented instance's profile, base64, written into its own memory. */
const DUMP = `(globalThis.__wasmInstances ?? []).filter((i) => i.exports.__write_profile).map(({ exports: e }) => {
  const size = ${8 + 4 * funcs.length};
  const at = e.__wbindgen_malloc(size, 8);
  const bytes = new Uint8Array(e.memory.buffer, at, e.__write_profile(at, size));
  let text = "";
  for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(text);
})`;

/**
 * CDP on the page, its workers and theirs (the map worker starts the job
 * workers), with `HOOK` in each before its scripts run; resolves to an
 * `evaluate` in all of them.
 */
async function sessions(context, page) {
  await page.addInitScript(HOOK);
  const cdp = await context.newCDPSession(page);
  const targets = new Map();
  // Follows the targets that `send`'s target attaches, whose events `listen` hands on.
  const follow = (send, listen) => {
    const children = new Map();
    listen((method, params) => {
      if (method === "Target.attachedToTarget") {
        const child = { pending: new Map(), listeners: [], id: 0 };
        children.set(params.sessionId, child);
        // A worker that ends with a message in flight never answers it.
        const sendChild = (m, p = {}) => new Promise((resolve) => {
          const id = ++child.id;
          child.pending.set(id, resolve);
          setTimeout(() => resolve({}), 10_000);
          const message = JSON.stringify({ id, method: m, params: p });
          send("Target.sendMessageToTarget", { sessionId: params.sessionId, message }).catch(() => resolve({}));
        });
        targets.set(child, sendChild);
        follow(sendChild, (listener) => child.listeners.push(listener));
        (async () => {
          await sendChild("Runtime.evaluate", { expression: HOOK });
          await sendChild("Target.setAutoAttach", { autoAttach: true, waitForDebuggerOnStart: true, flatten: false });
          await sendChild("Runtime.runIfWaitingForDebugger");
        })();
      } else if (method === "Target.receivedMessageFromTarget") {
        const child = children.get(params.sessionId);
        const message = JSON.parse(params.message);
        if (!child) return;
        if (message.id) {
          child.pending.get(message.id)?.(message.result ?? {});
          child.pending.delete(message.id);
        } else {
          for (const listener of child.listeners) listener(message.method, message.params);
        }
      } else if (method === "Target.detachedFromTarget") {
        targets.delete(children.get(params.sessionId));
        children.delete(params.sessionId);
      }
    });
  };
  follow((m, p) => cdp.send(m, p), (listener) => {
    for (const event of ["Target.attachedToTarget", "Target.receivedMessageFromTarget", "Target.detachedFromTarget"]) {
      cdp.on(event, (params) => listener(event, params));
    }
  });
  await cdp.send("Target.setAutoAttach", { autoAttach: true, waitForDebuggerOnStart: true, flatten: false });
  return async (expression) => {
    const params = { expression, returnByValue: true };
    const results = await Promise.all([cdp.send("Runtime.evaluate", params), ...[...targets.values()].map((send) => send("Runtime.evaluate", params))]);
    return results.flatMap((r) => r.result?.value ?? []);
  };
}

/** Loads each scene with the instrumented module and saves its profile in `out`. */
async function run() {
  const work = mkdtempSync(join(tmpdir(), "orbis-coverage-"));
  try {
    const instrumented = join(work, "core.wasm");
    split("--instrument", named, "-o", instrumented);
    // The demo as built, with the instrumented module in core.wasm's place.
    const assets = join(work, "assets");
    const server = await serveDemo({ mounts: { "assets/": assets } });
    const dist = fileURLToPath(new URL("../dist/assets/", import.meta.url));
    mkdirSync(assets);
    for (const file of readdirSync(dist)) {
      if (/^core_bg-.*\.wasm$/.test(file)) copyFileSync(instrumented, join(assets, file));
      else symlinkSync(join(dist, file), join(assets, file));
    }
    const browser = await launchChromium();
    try {
      for (const scene of scenes) {
        const context = await browser.newContext({ viewport: { width: 1024, height: 768 }, deviceScaleFactor: 1 });
        const page = await context.newPage();
        const evaluate = await sessions(context, page);
        await page.goto(`${server.url}load.html?${new URLSearchParams({ engine: "orbis", ...SCENES[scene] })}`);
        await page.waitForFunction(() => /^(done|error)/.test(document.title), null, { timeout: 120_000 });
        const title = await page.title();
        if (!title.startsWith("done")) throw new Error(`${scene}: ${title}`);
        if (tour) await page.evaluate(() => window.tour());
        await page.screenshot({ path: join(out, `${scene}.png`) });
        const written = (await evaluate(DUMP)).map((b) => Buffer.from(b, "base64"));
        if (!written.length) throw new Error(`${scene}: no instance wrote a profile`);
        writeFileSync(join(out, `${scene}.profile`), union(written));
        console.error(`${scene}: profiles from ${written.length} instances`);
        await context.close();
      }
    } finally {
      await browser.close();
      server.close();
    }
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

if (!args.includes("--reuse")) await run();
const profiles = Object.fromEntries(scenes.map((s) => [s, readFileSync(join(out, `${s}.profile`))]));
if (scenes.length > 1) profiles.union = union(Object.values(profiles));
writeFileSync(join(out, "union.profile"), profiles.union ?? profiles[scenes[0]]);

const ran = Object.fromEntries(Object.entries(profiles).map(([s, p]) => [s, stamps(p)]));
const columns = Object.keys(profiles);
const bytes = (filter) => funcs.reduce((sum, f, i) => sum + (filter(i) ? f.size : 0), 0);
const code = bytes(() => true);
const shipped = readFileSync(join(root, "web/pkg/core_bg.wasm"));
const lines = [
  `core.wasm: ${funcs.length} functions, ${kib(code)} of code; ${kib(shipped.length)} raw, ${kib(brotli(shipped))} brotli as shipped (web/pkg)`, "",
  `| profile | functions run | code run | code beyond ${scenes[0]} | first-frame module | deferred module |`, "|---|---|---|---|---|---|",
];
for (const scene of columns) {
  const primary = join(out, `${scene}.primary.wasm`), secondary = join(out, `${scene}.deferred.wasm`);
  split("--split", "--strip-debug", `--profile=${join(out, `${scene}.profile`)}`, named, "-o1", primary, "-o2", secondary);
  const [a, b] = [primary, secondary].map((f) => readFileSync(f));
  const count = ran[scene].filter(Boolean).length, run = bytes((i) => ran[scene][i]);
  lines.push(`| ${scene} | ${count} (${pct(count, funcs.length)}) | ${kib(run)} (${pct(run, code)}) | ${kib(bytes((i) => ran[scene][i] && !ran[scenes[0]][i]))} ` +
    `| ${kib(brotli(a))} brotli (${kib(a.length)} raw) | ${kib(brotli(b))} brotli (${kib(b.length)} raw) |`);
}

// What each scene alone ran, and what none did, split out of the rest:
// what a module of that feature would take off every other scene's load.
if (scenes.length > 1) {
  const stripped = join(out, "core.wasm");
  execFileSync("wasm-opt", [...FEATURES, "--strip-debug", named, "-o", stripped]);
  const whole = brotli(readFileSync(stripped));
  lines.push("", `| code that only it ran | functions | code | the rest's module | its module |`, "|---|---|---|---|---|");
  for (const scene of [...scenes, "no scene"]) {
    const only = funcs.filter((_, i) => scene === "no scene"
      ? scenes.every((s) => !ran[s][i]) : ran[scene][i] && scenes.every((s) => s === scene || !ran[s][i]));
    if (!only.length) {
      lines.push(`| ${scene} | 0 | | | |`);
      continue;
    }
    const list = join(out, `${scene.replace(" ", "-")}.only.txt`);
    writeFileSync(list, only.map((f) => f.name).join("\n"));
    const rest = join(out, "rest.wasm"), part = join(out, "part.wasm");
    split("--split", "--strip-debug", `--split-funcs=@${list}`, named, "-o1", rest, "-o2", part);
    const [a, b] = [rest, part].map((f) => brotli(readFileSync(f)));
    rmSync(rest);
    rmSync(part);
    lines.push(`| ${scene} | ${only.length} | ${kib(only.reduce((s, f) => s + f.size, 0))} | ${kib(a)} brotli (${kib(a - whole)}) | ${kib(b)} brotli |`);
  }
  lines.push("", `The rest's module against the whole, ${kib(whole)} brotli with its names stripped (${stripped}).`);
}

/** A table of code by `group` (of a function's name), with the share of it each profile ran. */
function table(title, group, limit) {
  const groups = new Map();
  funcs.forEach((f, i) => {
    const key = group(f.name);
    const g = groups.get(key) ?? groups.set(key, { code: 0, ...Object.fromEntries(columns.map((s) => [s, 0])) }).get(key);
    g.code += f.size;
    for (const s of columns) if (ran[s][i]) g[s] += f.size;
  });
  lines.push("", `| ${title} | code | ${columns.map((s) => `${s} ran`).join(" | ")} |`, `|---|---|${columns.map(() => "---|").join("")}`);
  for (const [key, g] of [...groups].sort((a, b) => b[1].code - a[1].code).slice(0, limit)) {
    lines.push(`| ${key} | ${kib(g.code)} | ${columns.map((s) => pct(g[s], g.code)).join(" | ")} |`);
  }
}
// A function's crate is the first of ours its name mentions, else its first crate.
const HASHED = /([A-Za-z_][A-Za-z0-9_]*)\[[0-9a-f]{16}\]/g;
const crateOf = (name) => {
  const crates = [...name.matchAll(HASHED)].map((m) => m[1]);
  return crates.find((c) => c.startsWith("orb_")) ?? crates[0] ?? "(no crate in name: JS exports, shims)";
};
table("crate", crateOf, 25);
table("module of ours", (name) => name.match(/(orb_\w+)\[[0-9a-f]{16}\]::(\w+)/)?.slice(1).join("::") ?? "(other crates)", 50);
lines.push("", `Code is function bodies before brotli; the modules are wasm-split's (names stripped). Profiles, modules and screenshots: ${out}`);
console.log(lines.join("\n"));
