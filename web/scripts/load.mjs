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
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { launchChromium, serveDemo } from "./serve.mjs";

const args = process.argv.slice(2);
const option = (name, fallback) => args.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const style = option("style");
if (!style) throw new Error("--style=<url> is required, e.g. a `cargo xtask h2h mirror`'s /styles/liberty");
const runs = Number(option("runs", "5"));
const engines = option("engines", "orbis,gljs,mlnrs").split(",");
// mln-rs's liberty-compare cameras, as lon,lat,zoom,bearing,pitch.
const VIEWS = {
  ams13: "4.89,52.37,13,0,0", paris14p60: "2.35,48.857,14,90,60",
  world2: "10,30,1.5,0,0", tokyo16: "139.70,35.69,16,0,0",
};
const views = option("views", Object.keys(VIEWS).join(",")).split(",");
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

mkdirSync(out, { recursive: true });
const mounts = engines.includes("mlnrs") ? { "mlnrs/": await mlnrs() } : {};
const server = await serveDemo({ isolated: true, mounts });
const browser = await launchChromium();
const results = [];
try {
  for (let run = -1; run < runs; run++) {
    for (const view of views) {
      for (const engine of engines) {
        const context = await browser.newContext({ viewport: { width: 1024, height: 768 }, deviceScaleFactor: 1 });
        const page = await context.newPage();
        const errors = [];
        page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
        page.on("pageerror", (e) => errors.push(e.message));
        const query = new URLSearchParams({ engine, style, at: VIEWS[view], mlnrs: `${new URL(server.url).pathname}mlnrs/index.js` });
        const row = { engine, view, run };
        try {
          await page.goto(`${server.url}load.html?${query}`);
          await page.waitForFunction(() => /^(done|error)/.test(document.title), null, { timeout: 90_000 });
          const title = await page.title();
          if (title.startsWith("done ")) Object.assign(row, JSON.parse(title.slice(5)));
          else row.error = title;
          if (run === 0) await page.screenshot({ path: join(out, `${engine}-${view}.png`) });
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

const lines = [`### Web load: ms from the page's script to first idle (Chromium, WebGPU, 1024×768 @1×; ${runs} runs)`, "",
  `| view | ${engines.join(" | ")} |`, `|---|${engines.map(() => "---|").join("")}`];
for (const view of views) {
  const cells = engines.map((engine) => {
    const rows = results.filter((r) => r.engine === engine && r.view === view);
    const idle = rows.filter((r) => !r.error).map((r) => r.idle - r.script);
    if (!idle.length) return `failed: ${rows[0]?.error ?? "no run"}`;
    return `${median(idle).toFixed(0)} [${Math.min(...idle).toFixed(0)}–${Math.max(...idle).toFixed(0)}]`;
  });
  lines.push(`| ${view} | ${cells.join(" | ")} |`);
}
lines.push("", `Medians [min–max]. Style ${style}; mln-rs ${MLNRS.name}; GL JS ${JSON.parse(readFileSync(new URL("../node_modules/maplibre-gl/package.json", import.meta.url))).version}. "Idle" is each engine's own event; GL JS's may not wait for symbol fades. Rows and screenshots: ${out}`);
console.log(lines.join("\n"));
