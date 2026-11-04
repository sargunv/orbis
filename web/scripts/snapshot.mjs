// Builds and serves web/dist, opens it in headless Chromium with WebGPU, waits for the
// map to go idle and writes a screenshot.
//
//   pnpm snapshot <out.png> [#zoom/lat/lon[/bearing/pitch]] [--thread=main|worker]
//     [--format=mvt|mlt] [--style=URL|scripts] [--compare=reference-map.png]
//
// The map viewer draws the Tokyo extract (MVT by default) unless `--style`
// names another.
import { readFile } from "node:fs/promises";
import { launchChromium, serveDemo } from "./serve.mjs";

const args = process.argv.slice(2);
const entry = args.find((a) => a.startsWith("--page="))?.slice(7) ?? "";
if (!["", "playground.html", "cities.html"].includes(entry)) throw new Error("unsupported page");
const thread = args.find((a) => a.startsWith("--thread="))?.slice("--thread=".length);
const format = args.find((a) => a.startsWith("--format="))?.slice("--format=".length);
const compare = args.find((a) => a.startsWith("--compare="))?.slice("--compare=".length);
const style = args.find((a) => a.startsWith("--style="))?.slice("--style=".length);
const [out = "snapshot.png", hash = ""] = args.filter((a) => !a.startsWith("--"));

const server = await serveDemo();
const params = new URLSearchParams();
if (thread) params.set("thread", thread);
// Snapshots draw the committed extract unless told otherwise.
if (style) params.set("style", style);
else if (entry === "") params.set("style", `tokyo-${format ?? "mvt"}`);
const query = `?${params}`;
const url = `${server.url}${entry}${query}${hash}`;

const browser = await launchChromium();
try {
  const page = await browser.newPage({ viewport: { width: 1024, height: 768 } });
  let warnings = 0;
  page.on("console", (m) => {
    if (["error", "warning"].includes(m.type())) { warnings++; console.error(`page: ${m.text()}`); }
    else if (m.type() === "info") console.log(`page: ${m.text()}`);
  });
  page.on("pageerror", (e) => console.error(`page: ${e.message}`));
  await page.goto(url);
  const started = Date.now();
  await page.waitForFunction(() => {
    const text = document.getElementById("hud").textContent;
    return text.endsWith(" · idle") || text.startsWith("WebGPU") || text.includes("Error");
  }, null, { timeout: 60_000 });
  await page.waitForTimeout(250);
  console.log(`${await page.locator("#hud").textContent()} (${Date.now() - started} ms)`);
  await page.screenshot({ path: out });
  await page.locator("#hud, #about, .demo-annotation").evaluateAll((elements) => {
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
