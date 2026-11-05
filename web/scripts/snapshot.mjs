// Builds and serves web/dist, opens it in headless Chromium with WebGPU, waits for the
// map to go idle and writes a screenshot.
//
//   pnpm snapshot <out.png> [#zoom/lat/lon[/bearing/pitch]] [--thread=main|worker]
//     [--format=mvt|mlt] [--style=<key>|URL]
//     [--size=WxH] [--compare=reference-map.png] [--renderer=webgpu|webgl2] [--lose=<ms> [--switch]]
//
// `--lose` loses the GPU that long after the map loads and waits for it to
// recover (plan §8.8); with `--switch`, onto WebGL2.
//
// The map app draws the Tokyo extract (MVT by default) unless `--style`
// names another.
// Besides <out.png>, it writes <out>-map.png without the app's chrome.
import { readFile } from "node:fs/promises";
import { launchChromium, serveDemo } from "./serve.mjs";

const args = process.argv.slice(2);
const thread = args.find((a) => a.startsWith("--thread="))?.slice("--thread=".length);
const format = args.find((a) => a.startsWith("--format="))?.slice("--format=".length);
const compare = args.find((a) => a.startsWith("--compare="))?.slice("--compare=".length);
const style = args.find((a) => a.startsWith("--style="))?.slice("--style=".length);
const renderer = args.find((a) => a.startsWith("--renderer="))?.slice("--renderer=".length);
const lose = args.find((a) => a.startsWith("--lose="))?.slice("--lose=".length);
const switchTier = args.includes("--switch");
const [width, height] = (args.find((a) => a.startsWith("--size="))?.slice(7) ?? "1024x768").split("x").map(Number);
const [out = "snapshot.png", hash = ""] = args.filter((a) => !a.startsWith("--"));

const server = await serveDemo();
const params = new URLSearchParams();
if (thread) params.set("thread", thread);
if (renderer) params.set("renderer", renderer);
if (lose) params.set("lose", lose);
if (switchTier) params.set("switch", "");
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
    if (["error", "warning"].includes(m.type())) { warnings++; console.error(`page: ${m.text()}`); }
    else if (m.type() === "info") console.log(`page: ${m.text()}`);
  });
  page.on("pageerror", (e) => console.error(`page: ${e.message}`));
  await page.goto(url);
  const started = Date.now();
  await page.waitForFunction(() => {
    const app = document.getElementById("app")?.dataset;
    return app?.map === "error" || (app?.map === "idle"
      && (!new URLSearchParams(location.search).has("lose") || app.recovered === "true"));
  }, null, { timeout: 60_000 });
  await page.waitForTimeout(250);
  const stats = await page.evaluate(() => {
    const s = globalThis.app.map?.stats;
    return s ? `${app.map.thread} thread, ${document.getElementById("app").dataset.renderer}, z${s.zoom.toFixed(2)}, ${s.ms.toFixed(1)} ms CPU, ${s.draws} draws, ` +
      `${s.tilesDrawn} tiles` : document.getElementById("toast").textContent;
  });
  console.log(`${stats} (${Date.now() - started} ms)`);
  await page.screenshot({ path: out });
  await page.locator(".corner, .attribution, .toast, .demo-annotation").evaluateAll((elements) => {
    for (const element of elements) element.style.visibility = "hidden";
  });
  const pixels = await page.locator("#map").screenshot({ path: out.replace(/\.png$/, "-map.png") });
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
