// Builds and serves web/dist, opens it in headless Chromium with WebGPU, waits for the
// map to go idle and writes a screenshot.
//
//   pnpm snapshot <out.png> [#zoom/lat/lon[/bearing/pitch]]
//     [--format=mvt|mlt] [--style=<key>|URL]
//     [--size=WxH] [--compare=reference-map.png] [--lose=<ms>]
//     [--reopen[=offline]] [--still] [--pair=<left>,<right>]
//
// `--pair` shows the demo's comparison of two engines (`orbis`, `gljs`,
// `native`) side by side and waits for both sides to load.
//
// `--reopen` loads the page again once it is idle, in the same browser
// profile, and times the reopen from the browser's cache;
// `=offline` fails every request but the demo server's (the browser logs
// the stale entries' failed revalidations; they aren't counted as errors).
//
// `--lose` loses the GPU that long after the map loads and waits for it to
// recover.
//
// The map app draws the Tokyo extract (MVT by default) unless `--style`
// names another.
// Besides <out.png>, it writes <out>-map.png without the app's chrome, and
// with `--still` <out>-still.png: the map's own snapshot of the view.
import { readFile, writeFile } from "node:fs/promises";
import { launchChromium, serveDemo } from "./serve.mjs";

const args = process.argv.slice(2);
const format = args.find((a) => a.startsWith("--format="))?.slice("--format=".length);
const compare = args.find((a) => a.startsWith("--compare="))?.slice("--compare=".length);
const style = args.find((a) => a.startsWith("--style="))?.slice("--style=".length);
const lose = args.find((a) => a.startsWith("--lose="))?.slice("--lose=".length);
const reopen = args.find((a) => a.startsWith("--reopen"));
const pair = args.find((a) => a.startsWith("--pair="))?.slice("--pair=".length);
let offline = false;
const [width, height] = (args.find((a) => a.startsWith("--size="))?.slice(7) ?? "1024x768").split("x").map(Number);
const [out = "snapshot.png", hash = ""] = args.filter((a) => !a.startsWith("--"));

// MapLibre Native needs a cross-origin isolated page; serving it so spares the reload.
const server = await serveDemo({ isolated: Boolean(pair?.includes("native")) });
const params = new URLSearchParams();
if (lose) params.set("lose", lose);
if (pair) params.set("compare", pair);
// Snapshots draw the committed extract unless told otherwise.
if (style) params.set("style", style);
else params.set("style", `tokyo-${format ?? "mvt"}`);
const query = `?${params}`;
const url = `${server.url}${query}${hash}`;

const browser = await launchChromium();
try {
  const page = await browser.newPage({ viewport: { width, height } });
  let warnings = 0;
  page.on("console", (m) => {
    if (offline && m.text().includes("ERR_INTERNET_DISCONNECTED")) return;
    if (["error", "warning"].includes(m.type())) { warnings++; console.error(`page: ${m.text()}`); }
    else if (m.type() === "info") console.log(`page: ${m.text()}`);
  });
  page.on("pageerror", (e) => console.error(`page: ${e.message}`));
  const idle = () => page.waitForFunction(() => {
    const app = document.getElementById("app")?.dataset;
    return app?.map === "error" || (app?.map === "idle"
      && (!new URLSearchParams(location.search).has("lose") || app.recovered === "true"));
  }, null, { timeout: 60_000 });
  let started = Date.now();
  await page.goto(url);
  await idle();
  if (pair) {
    await page.waitForFunction(() => {
      const sides = globalThis.app.compare.sides;
      return sides?.length === 2 && sides.every((side) => side.loaded);
    }, null, { timeout: 120_000 });
  }
  console.log(`opened: ${Date.now() - started} ms to idle`);
  if (reopen) {
    if (reopen === "--reopen=offline") {
      offline = true;
      await page.route((u) => !u.href.startsWith(server.url), (route) => route.abort("internetdisconnected"));
    }
    started = Date.now();
    await page.reload();
    await idle();
    console.log(`reopened${reopen === "--reopen=offline" ? " offline" : ""}: ${Date.now() - started} ms to idle`);
  }
  await page.waitForTimeout(250);
  const stats = await page.evaluate(() => {
    const s = globalThis.app.map?.stats;
    return s ? `z${s.zoom.toFixed(2)}, ${s.ms.toFixed(1)} ms CPU, ${s.draws} draws, ` +
      `${s.tilesDrawn} tiles` : document.getElementById("toast").textContent;
  });
  console.log(`${stats} (${Date.now() - started} ms)`);
  await page.screenshot({ path: out });
  await page.locator(".corner, .attribution, .toast").evaluateAll((elements) => {
    for (const element of elements) element.style.visibility = "hidden";
  });
  const pixels = await page.locator("#map").screenshot({ path: out.replace(/\.png$/, "-map.png") });
  if (args.includes("--still")) {
    const started = Date.now();
    const png = await page.evaluate(async () => {
      const image = await globalThis.orbis.map.snapshot();
      const canvas = new OffscreenCanvas(image.width, image.height);
      canvas.getContext("2d").putImageData(image, 0, 0);
      const bytes = new Uint8Array(await (await canvas.convertToBlob({ type: "image/png" })).arrayBuffer());
      return btoa(Array.from(bytes, (b) => String.fromCharCode(b)).join(""));
    });
    await writeFile(out.replace(/\.png$/, "-still.png"), Buffer.from(png, "base64"));
    console.log(`still: ${Date.now() - started} ms`);
  }
  if (compare) {
    const reference = await readFile(compare);
    const differing = await page.evaluate(async ([a, b]) => {
      const decode = async (base64) => {
        const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const context = canvas.getContext("2d");
        context.drawImage(bitmap, 0, 0);
        return context.getImageData(0, 0, bitmap.width, bitmap.height).data;
      };
      const [x, y] = await Promise.all([decode(a), decode(b)]);
      if (x.length !== y.length) throw new Error("image dimensions differ");
      let different = 0;
      for (let i = 0; i < x.length; i += 4) {
        if (x[i] !== y[i] || x[i+1] !== y[i+1] || x[i+2] !== y[i+2] || x[i+3] !== y[i+3]) different++;
      }
      return different;
    }, [pixels.toString("base64"), reference.toString("base64")]);
    console.log(`${differing} differing pixels vs ${compare}`);
    if (differing) throw new Error(`${differing} differing pixels`);
  }
  if (warnings) throw new Error(`${warnings} page warnings/errors`);
} finally {
  await browser.close();
  server.close();
}
