// Load times in headless Chromium with WebGPU: Orbis, the pinned GL JS and
// mln-rs's web build (@bartlwrs/mln-rs, fetched by version and checked
// against its npm integrity into the h2h cache) on demo/load.html, from
// the page's script start to the engine's first `idle`, 1024×768 at 1×.
// Each sample opens a fresh browser context, so the HTTP cache is empty;
// the browser (and its shader cache) is shared, after one untimed pass of
// each engine and view. Pages are cross-origin isolated, which mln-rs's
// threads need, for every engine alike.
//
//   pnpm load --style=<url> [--runs=5] [--engines=orbis,gljs,mlnrs]
//     [--views=ams13,paris14p60,world2,tokyo16] [--out=<dir>]
//
// `cargo xtask h2h web` runs it against its local mirror of OpenFreeMap
// (`--style=<mirror>/styles/liberty`), after `cargo xtask web`.
//
// `--cache=<cache-style.db>` loads MapLibre Native's five real-world render
// tests instead (chicago, nepal, norway, sanfrancisco, uruguay), 512×512 at
// their cameras, served offline from that render-test cache (mln-rs keeps
// it at tools/render-test/testdata/metrics/); what the cache lacks is a 404
// for every engine alike. `--settings=normal,cpu4x,cores2` repeats each
// load with the page's thread throttled 4× over CDP (`cpu4x`) and with the
// page told it has 2 cores (`cores2`: `navigator.hardwareConcurrency`,
// which sizes each engine's worker pool; the machine's cores stay).
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";
import { inflateSync } from "node:zlib";
import { launchChromium, serveDemo } from "./serve.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const cache = option("cache");
const style = option("style");
if (!style && !cache) {
  throw new Error("--style=<url> is required, e.g. a `cargo xtask h2h mirror`'s /styles/liberty, or --cache=<cache-style.db>");
}
const runs = Number(option("runs", "5"));
const engines = option("engines", "orbis,gljs,mlnrs").split(",");
// Other builds of web/dist, as `--builds=before=<dir>`, run as engine
// `orbis@before` beside this tree's `orbis`.
const builds = Object.fromEntries(option("builds", "").split(",").filter(Boolean).map((b) => b.split("=")));
// mln-rs's liberty-compare cameras as lon,lat,zoom,bearing,pitch, or the
// real-world render tests' (their `setCenter` and `setZoom`).
const VIEWS = cache ? {
  chicago: "-87.6942445,41.8703965,13,0,0", nepal: "85.48805250000001,28.115547,13,0,0",
  norway: "10.8181,64.863174,12,0,0", sanfrancisco: "-122.448635,37.7669995,15,0,0",
  uruguay: "-56.509552,-32.865788,9,0,0",
} : {
  ams13: "4.89,52.37,13,0,0", paris14p60: "2.35,48.857,14,90,60",
  world2: "10,30,1.5,0,0", tokyo16: "139.70,35.69,16,0,0",
};
const views = option("views", Object.keys(VIEWS).join(",")).split(",");
const SETTINGS = {
  normal: async () => {},
  cpu4x: (cdp) => cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 }),
  cores2: (cdp) => cdp.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 2 }),
};
const settings = option("settings", "normal").split(",");
for (const s of settings) if (!SETTINGS[s]) throw new Error(`--settings: no ${s}; ${Object.keys(SETTINGS).join(", ")}`);
const viewport = cache ? { width: 512, height: 512 } : { width: 1024, height: 768 };
const out = option("out", fileURLToPath(new URL("../../target/h2h/web/", import.meta.url)));
const MLNRS = {
  name: "@bartlwrs/mln-rs@0.1.0",
  tarball: "https://registry.npmjs.org/@bartlwrs/mln-rs/-/mln-rs-0.1.0.tgz",
  integrity: "sha512-6rLjFbfTOpqYJz7HC4r4Ci3IrF34WuRjFZxHRW4+2vyLatA6VHMGcVYw3bV8SffvfH6+zplemkOCudP380ydAA==",
};

/** The h2h cache, as `cargo xtask h2h` finds it. */
function cacheDir() {
  if (process.env.ORB_H2H_CACHE) return process.env.ORB_H2H_CACHE;
  if (process.platform === "darwin") return join(homedir(), "Library/Caches/orbis-h2h");
  return join(process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache"), "orbis-h2h");
}

/** mln-rs's web build, unpacked into the cache: its `dist/` directory. */
async function mlnrs() {
  const dir = join(cacheDir(), "mln-rs-web-0.1.0");
  const dist = join(dir, "package/dist/");
  if (existsSync(join(dist, "index.js"))) return dist;
  const bytes = Buffer.from(await (await fetch(MLNRS.tarball)).arrayBuffer());
  const integrity = "sha512-" + createHash("sha512").update(bytes).digest("base64");
  if (integrity !== MLNRS.integrity) throw new Error(`${MLNRS.tarball}: integrity ${integrity}, not ${MLNRS.integrity}`);
  const partial = `${dir}.partial-${process.pid}`;
  rmSync(partial, { recursive: true, force: true });
  mkdirSync(partial, { recursive: true });
  writeFileSync(join(partial, "package.tgz"), bytes);
  execFileSync("tar", ["-xzf", join(partial, "package.tgz"), "-C", partial]);
  rmSync(dir, { recursive: true, force: true });
  execFileSync("mv", [partial, dir]);
  return dist;
}

const median = (v) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[Math.floor((s.length - 1) / 2)] : NaN; };

/**
 * The cache's real-world styles, their tiles, sprites and glyphs as files
 * under `dir`, the styles' `local://` and `mapbox://` URLs pointing at `base`.
 */
function extractCache(db, dir, base) {
  const cached = new DatabaseSync(db, { readOnly: true });
  const bytes = (data, compressed) => (compressed ? inflateSync(data) : Buffer.from(data));
  const write = (path, data) => {
    mkdirSync(dirname(join(dir, path)), { recursive: true });
    writeFileSync(join(dir, path), data);
  };
  for (const { url, data, compressed } of cached.prepare("SELECT url, data, compressed FROM resources").all()) {
    const style = url.match(/^http:\/\/styles\/(.*\.json)$/)?.[1];
    const asset = url.match(/^mapbox:\/\/((?:sprites|fonts)\/.*)$/)?.[1];
    if (style) {
      const doc = bytes(data, compressed).toString().replaceAll("local://", base).replaceAll("mapbox://", base);
      write(`styles/${style}`, doc);
    } else if (asset) {
      write(decodeURIComponent(asset), bytes(data, compressed));
    }
  }
  const tiles = cached.prepare("SELECT url_template, z, x, y, data, compressed FROM tiles WHERE url_template LIKE 'http://mvt-fixtures/%'");
  for (const { url_template, z, x, y, data, compressed } of tiles.all()) {
    const path = url_template.slice("http://".length).replace("{z}", z).replace("{x}", x).replace("{y}", y);
    write(path, bytes(data, compressed));
  }
  cached.close();
}

mkdirSync(out, { recursive: true });
const extracted = cache ? join(out, "real-world/") : undefined;
const mounts = {
  ...(engines.includes("mlnrs") ? { "mlnrs/": await mlnrs() } : {}),
  ...(extracted ? { "real-world/": extracted } : {}),
};
const server = await serveDemo({ isolated: true, mounts, builds });
if (extracted) {
  rmSync(extracted, { recursive: true, force: true });
  extractCache(cache, extracted, `${server.url}real-world/`);
}
const styleOf = (view) => (cache ? `${server.url}real-world/styles/${view}.json` : style);
const browser = await launchChromium();
const results = [];
try {
  for (let run = -1; run < runs; run++) {
    for (const setting of settings) for (const view of views) {
      for (const engine of engines) {
        const context = await browser.newContext({ viewport, deviceScaleFactor: 1 });
        const page = await context.newPage();
        await SETTINGS[setting](await context.newCDPSession(page));
        const errors = [];
        page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
        page.on("pageerror", (e) => errors.push(e.message));
        const query = new URLSearchParams({
          engine: engine.split("@")[0], style: styleOf(view), at: VIEWS[view], mlnrs: `${new URL(server.url).pathname}mlnrs/index.js`,
        });
        const row = { engine, view, setting, run };
        // When the style's fetch starts, on the epoch clock: from whichever
        // thread the engine fetches it.
        let styleRequest;
        context.on("request", (r) => { if (r.url() === styleOf(view) && !styleRequest) styleRequest = r; });
        try {
          const build = engine.split("@")[1];
          await page.goto(`${server.url}${build ? `@${build}/` : ""}load.html?${query}`);
          await page.waitForFunction(() => /^(done|error)/.test(document.title), null, { timeout: 90_000 });
          const title = await page.title();
          if (title.startsWith("done ")) {
            const done = JSON.parse(title.slice(5));
            Object.assign(row, done);
            const started = styleRequest?.timing().startTime;
            if (started > 0) row.style = started - done.origin;
          } else {
            row.error = title;
          }
          if (run === 0) await page.screenshot({ path: join(out, `${engine}-${view}-${setting}.png`) });
        } catch (e) {
          row.error = String(e).slice(0, 300);
        }
        if (errors.length) row.consoleErrors = errors.slice(0, 3);
        await context.close();
        console.error(JSON.stringify(row));
        if (run >= 0) results.push(row);
      }
    }
  }
} finally {
  await browser.close();
  server.close();
}
writeFileSync(join(out, "load.json"), JSON.stringify(results, null, 1));

const { width, height } = viewport;
const tagged = settings.length > 1;
const gljs = JSON.parse(readFileSync(new URL("../node_modules/maplibre-gl/package.json", import.meta.url))).version;
const lines = [];
// From the page's script, and from the style's fetch, which leaves out
// what each engine loads before its map starts.
for (const [what, from] of [["the page's script", "script"], ["the style's fetch", "style"]]) {
  lines.push(`### Web load: ms from ${what} to first idle (Chromium, WebGPU, ${width}×${height} @1×; ${runs} runs)`, "",
    `| view | ${tagged ? "setting | " : ""}${engines.join(" | ")} |`,
    `|---|${tagged ? "---|" : ""}${engines.map(() => "---|").join("")}`);
  const idles = (engine, view, setting) => results
    .filter((r) => r.engine === engine && r.view === view && r.setting === setting && r[from] !== undefined)
    .map((r) => r.idle - r[from]);
  for (const setting of settings) {
    for (const view of views) {
      const cells = engines.map((engine) => {
        const idle = idles(engine, view, setting);
        if (!idle.length) {
          const failed = results.find((r) => r.engine === engine && r.view === view && r.setting === setting);
          return `failed: ${failed?.error ?? "no run"}`;
        }
        return `${median(idle).toFixed(0)} [${Math.min(...idle).toFixed(0)}–${Math.max(...idle).toFixed(0)}]`;
      });
      lines.push(`| ${view} | ${tagged ? `${setting} | ` : ""}${cells.join(" | ")} |`);
    }
    // Idle comes on a display frame, so medians move in frames: the views' mean
    // shows a change that moves some of them by one.
    const overall = engines.map((engine) => {
      const medians = views.map((view) => median(idles(engine, view, setting)));
      return (medians.reduce((a, b) => a + b, 0) / medians.length).toFixed(0);
    });
    if (views.length > 1) lines.push(`| mean of views | ${tagged ? `${setting} | ` : ""}${overall.join(" | ")} |`);
  }
  lines.push("");
}
lines.push(`Medians [min–max]. Style ${cache ? `the real-world render tests, from ${cache}` : style}; mln-rs ${MLNRS.name}; GL JS ${gljs}. "Idle" is each engine's own event; GL JS's may not wait for symbol fades. Rows and screenshots: ${out}`);
console.log(lines.join("\n"));
