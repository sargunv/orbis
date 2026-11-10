// Load times in headless Chromium with WebGPU: Orbis, the pinned GL JS and
// mln-rs's browser build (its examples/web at the commit `cargo xtask h2h`
// pins, built once into the h2h cache: `views.mjs`'s `mlnrs`) on
// demo/load.html, from the page's script start to the engine's first
// `idle` (mln-rs's: fully loaded), 1024×768 at 1×. Each sample opens a
// fresh browser context, so the HTTP cache is empty; the browser (and its
// shader cache) is shared, after one untimed pass of each engine and view.
// Pages are cross-origin isolated, which MapLibre Native's threads need,
// for every engine alike.
//
//   pnpm load --style=<url> [--runs=5] [--engines=orbis,gljs,mlnrs]
//     [--views=ams13,paris14p60,world2,tokyo16] [--throttle=<Mbit/s>] [--out=<dir>]
//
// `--throttle` sends the engines' own files compressed over a link of that
// bandwidth (`serveDemo`'s `throttle`), as a first visit downloads them;
// the style's tiles and fonts come as fast as their server sends them.
//
// `cargo xtask h2h web` runs it against its local mirror of OpenFreeMap
// (`--style=<mirror>/styles/liberty`), after `cargo xtask web`.
//
// `--cache=<cache-style.db>` loads MapLibre Native's five real-world render
// tests instead (chicago, nepal, norway, sanfrancisco, uruguay), 512×512 at
// their cameras, served offline from that render-test cache (mln-rs keeps
// it at tools/render-test/testdata/metrics/; `--cache=default` finds it in
// the checkout `cargo xtask h2h` builds); what the cache lacks is a 404
// for every engine alike. `--settings=normal,cpu4x,cores2` repeats each
// load with the page's thread throttled 4× over CDP (`cpu4x`) and with the
// page told it has 2 cores (`cores2`: `navigator.hardwareConcurrency`,
// which sizes each engine's worker pool; the machine's cores stay).
//
// `--scene=terrain` loads the Alps terrain style (live OpenFreeMap and
// Mapterhorn tiles) and `--scene=globe` Liberty on the globe under the
// demo's sky (`--style`, by default the live Liberty), each at three views,
// in Orbis and GL JS by default. `--engines` takes `native` too: MapLibre
// Native's browser build, as the demo's comparison loads it (not on the
// globe, which it lacks).
//
// `--android [--physical]` loads in Chrome on this worktree's Android
// device instead (`openAndroid`; `--physical` a phone the caller assigned),
// a tab per sample with the HTTP cache disabled, after `chrome://gpuclean`
// empties the GPU process's shader cache. The phone's driver keeps
// compiled shaders of its own, so Orbis runs on the page's thread with
// shaders unique to the sample (`demo/cold.js`): cold, as a first visit
// compiles them; `--warm` keeps both caches and the shaders as they are,
// as a later visit finds them. `--page` runs Orbis on the page's thread on
// this machine too, and `--cold` with it gives it shaders unique to the
// sample. `--backend=webgl2` draws Orbis with WebGL 2 (`createMap`'s
// `backend`). `--style=tokyo` loads the committed extract, which GL JS
// can't (it has no `pmtiles://`).
import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { launchChromium, openAndroid, serveDemo } from "./serve.mjs";
import {
  extractCache,
  median,
  MLNRS,
  mlnrs,
  realWorldCache,
  VIEWS as CAMERAS,
} from "./views.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) =>
  args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ??
  fallback;
const scene = option("scene");
if (scene && !["terrain", "globe"].includes(scene))
  throw new Error("--scene takes terrain or globe");
const cache =
  option("cache") === "default" ? realWorldCache() : option("cache");
let style = option(
  "style",
  scene === "globe"
    ? "https://tiles.openfreemap.org/styles/liberty"
    : undefined,
);
if (!style && !cache && !scene) {
  throw new Error(
    "--style=<url> is required, e.g. a `cargo xtask h2h mirror`'s /styles/liberty, or --cache=<cache-style.db>, or --scene",
  );
}
const runs = Number(option("runs", "5"));
const engines = option(
  "engines",
  scene ? "orbis,gljs" : "orbis,gljs,mlnrs",
).split(",");
// Native draws the style flat in Mercator: another scene, timed as if the same.
if (scene === "globe" && engines.includes("native"))
  throw new Error("--scene=globe: MapLibre Native has no globe projection");
// Other builds of web/dist, as `--builds=before=<dir>`, run as engine
// `orbis@before` beside this tree's `orbis`.
const builds = Object.fromEntries(
  option("builds", "")
    .split(",")
    .filter(Boolean)
    .map((b) => b.split("=")),
);
const VIEWS =
  CAMERAS[
    cache ? "realWorld" : (scene ?? (style === "tokyo" ? "tokyo" : "liberty"))
  ];
const views = option("views", Object.keys(VIEWS).join(",")).split(",");
const SETTINGS = {
  normal: async () => {},
  cpu4x: (cdp) => cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 }),
  cores2: (cdp) =>
    cdp.send("Emulation.setHardwareConcurrencyOverride", {
      hardwareConcurrency: 2,
    }),
};
const settings = option("settings", "normal").split(",");
for (const s of settings)
  if (!SETTINGS[s])
    throw new Error(`--settings: no ${s}; ${Object.keys(SETTINGS).join(", ")}`);
const viewport = cache
  ? { width: 512, height: 512 }
  : { width: 1024, height: 768 };
const throttle = option("throttle") && (Number(option("throttle")) * 1e6) / 8;
const physical = args.includes("--physical");
const warm = args.includes("--warm");
const onPage = args.includes("--page");
const cold = args.includes("--cold");
if (cold && !onPage && !android) throw new Error("--cold runs with --page");
const backend = option("backend");
const android = physical || args.includes("--android");
if (android && settings.some((s) => s !== "normal"))
  throw new Error("--android takes no --settings");
const out = option(
  "out",
  fileURLToPath(new URL("../../target/h2h/web/", import.meta.url)),
);

mkdirSync(out, { recursive: true });
const extracted = cache ? join(out, "real-world/") : undefined;
const mounts = {
  ...(engines.includes("mlnrs") ? { "mlnrs/": mlnrs() } : {}),
  ...(extracted ? { "real-world/": extracted } : {}),
};
const server = await serveDemo({
  isolated: true,
  mounts,
  builds,
  throttle,
  unthrottled: ["real-world/"],
});
if (extracted) extractCache(cache, extracted, `${server.url}real-world/`);
// The Alps terrain style, by its name in the build.
if (scene === "terrain") {
  const dist = fileURLToPath(new URL("../dist/assets/", import.meta.url));
  style = `${server.url}assets/${readdirSync(dist).find((f) => /^alps-.*\.json$/.test(f))}`;
}
const styleOf = (view) =>
  cache ? `${server.url}real-world/styles/${view}.json` : style;
const browser = android ? undefined : await launchChromium();
// A sample's page: a fresh context in headless Chromium, or a tab in the
// device's Chrome after its GPU cache is emptied.
async function samplePage() {
  if (!android) {
    const context = await browser.newContext({
      viewport,
      deviceScaleFactor: 1,
    });
    return {
      context,
      page: await context.newPage(),
      close: () => context.close(),
    };
  }
  if (!warm) {
    const clean = await openAndroid(server, { physical });
    await clean.page.goto("chrome://gpuclean").catch(() => {});
    await clean.close();
  }
  const { page, close } = await openAndroid(server, { physical });
  await (
    await page.context().newCDPSession(page)
  ).send("Network.setCacheDisabled", { cacheDisabled: true });
  return { context: page.context(), page, close };
}
const results = [];
try {
  for (let run = -1; run < runs; run++) {
    for (const setting of settings)
      for (const view of views) {
        // The engines' order turns each run, so that neither always follows the other.
        for (const engine of run % 2 ? [...engines].reverse() : engines) {
          const { context, page, close } = await samplePage();
          await SETTINGS[setting](await context.newCDPSession(page));
          const errors = [];
          page.on("console", (m) => {
            if (m.type() === "error") errors.push(m.text());
          });
          page.on("pageerror", (e) => errors.push(e.message));
          const query = new URLSearchParams({
            engine: engine.split("@")[0],
            style: styleOf(view),
            at: VIEWS[view],
            mlnrs: `${new URL(server.url).pathname}mlnrs/index.js`,
            ...(scene === "globe" && { globe: "" }),
            // Orbis's shaders unique to the sample, on the page's thread.
            ...((android || onPage) &&
              engine.startsWith("orbis") && {
                worker: "0",
                ...((android ? !warm : cold) && {
                  nonce: 1 + Math.floor(Math.random() * 9999),
                }),
              }),
            ...(backend && engine.startsWith("orbis") && { backend }),
          });
          const row = { engine, view, setting, run };
          // When the style's fetch starts, on the epoch clock: from whichever
          // thread the engine fetches it.
          let styleRequest;
          const onRequest = (r) => {
            if (r.url() === styleOf(view) && !styleRequest) styleRequest = r;
          };
          context.on("request", onRequest);
          try {
            const build = engine.split("@")[1];
            await page.goto(
              `${server.url}${build ? `@${build}/` : ""}load.html?${query}`,
            );
            await page.waitForFunction(
              () => /^(done|error)/.test(document.title),
              null,
              { timeout: 90_000 },
            );
            const title = await page.title();
            if (title.startsWith("done ")) {
              const done = JSON.parse(title.slice(5));
              Object.assign(row, done);
              // A build from before the page reported its time origin is
              // asked for it.
              const origin =
                done.origin ??
                (await page.evaluate(() => performance.timeOrigin));
              if (engine.startsWith("orbis"))
                Object.assign(
                  row,
                  await page.evaluate(() => window.loadDetails?.()),
                );
              const started = styleRequest?.timing().startTime;
              if (started > 0) row.style = started - origin;
            } else {
              row.error = title;
            }
            if (run === 0)
              await page.screenshot({
                path: join(out, `${engine}-${view}-${setting}.png`),
              });
          } catch (e) {
            row.error = String(e).slice(0, 300);
          }
          if (errors.length) row.consoleErrors = errors.slice(0, 3);
          context.off("request", onRequest);
          await close();
          console.error(JSON.stringify(row));
          if (run >= 0) results.push(row);
        }
      }
  }
} finally {
  await browser?.close();
  server.close();
}
writeFileSync(join(out, "load.json"), JSON.stringify(results, null, 1));

const { width, height } = android
  ? { width: "phone", height: "screen" }
  : viewport;
const tagged = settings.length > 1;
const gljs = JSON.parse(
  readFileSync(
    new URL("../node_modules/maplibre-gl/package.json", import.meta.url),
  ),
).version;
const lines = [];
// From the page's script, and from the style's fetch, which leaves out
// what each engine loads before its map starts.
for (const [what, from] of [
  ["the page's script", "script"],
  ["the style's fetch", "style"],
]) {
  lines.push(
    `### Web load: ms from ${what} to first idle (${android ? `Chrome on Android, ${warm ? "warm" : "cold"} GPU cache` : `Chromium${onPage ? `, Orbis on the page${cold ? " with cold shaders" : ""}` : ""}`}, ${backend === "webgl2" ? "Orbis on WebGL 2" : "WebGPU"}, ${width}×${height}${android ? "" : " @1×"}; ${runs} runs)`,
    "",
    `| view | ${tagged ? "setting | " : ""}${engines.join(" | ")} |`,
    `|---|${tagged ? "---|" : ""}${engines.map(() => "---|").join("")}`,
  );
  const idles = (engine, view, setting) =>
    results
      .filter(
        (r) =>
          r.engine === engine &&
          r.view === view &&
          r.setting === setting &&
          r[from] !== undefined,
      )
      .map((r) => r.idle - r[from]);
  for (const setting of settings) {
    for (const view of views) {
      const cells = engines.map((engine) => {
        const idle = idles(engine, view, setting);
        if (!idle.length) {
          const failed = results.find(
            (r) =>
              r.engine === engine && r.view === view && r.setting === setting,
          );
          return `failed: ${failed?.error ?? "no run"}`;
        }
        return `${median(idle).toFixed(0)} [${Math.min(...idle).toFixed(0)}–${Math.max(...idle).toFixed(0)}]`;
      });
      lines.push(
        `| ${view} | ${tagged ? `${setting} | ` : ""}${cells.join(" | ")} |`,
      );
    }
    // Idle comes on a display frame, so medians move in frames: the views' mean
    // shows a change that moves some of them by one.
    const overall = engines.map((engine) => {
      const medians = views.map((view) => median(idles(engine, view, setting)));
      return (medians.reduce((a, b) => a + b, 0) / medians.length).toFixed(0);
    });
    if (views.length > 1)
      lines.push(
        `| mean of views | ${tagged ? `${setting} | ` : ""}${overall.join(" | ")} |`,
      );
  }
  lines.push("");
}
const about = cache
  ? `the real-world render tests, from ${cache}`
  : `${style}${scene === "globe" ? " on the globe" : ""}`;
lines.push(
  `Medians [min–max]. ${throttle ? `Engine files compressed over a ${option("throttle")} Mbit/s link. ` : ""}Style ${about}; mln-rs ${MLNRS.name}; GL JS ${gljs}. "Idle" is each engine's own event (MapLibre Native's: its snapshot fully loaded, polled each frame; mln-rs's: its map fully loaded); GL JS's may not wait for symbol fades, and mln-rs's doesn't. Rows and screenshots: ${out}`,
);
console.log(lines.join("\n"));
if (android) process.exit(0);
