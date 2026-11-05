// The Chrome transport for orb-fixtures. Rust sends one JSON request per line;
// a render's response names raw premultiplied RGBA bytes written by this
// process, a query's (`query` set) carries its features' JSON.
// Fixture selection, metadata semantics and pixel comparison belong to Rust.
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { resolve, sep, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { chromium } from "playwright";

const root = fileURLToPath(new URL("../../", import.meta.url));
// The fixture driver is in `harness.wasm`, which no demo page loads. Serve the
// generated wasm-bindgen modules directly; only the public demo uses dist/.
const mounts = { "/pkg/": resolve(root, "web/pkg"), "/assets/": resolve(root, "fixtures/gl-js/assets") };
const types = { ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/") {
      res.writeHead(200, { "content-type": "text/html" });
      res.end('<!doctype html><script type="module">import init, { FixtureRenderer } from "/pkg/harness.js"; await init(); window.renderer = await FixtureRenderer.create(new URL("/assets/", location).href, new URLSearchParams(location.search).get("renderer") ?? "webgpu");</script>');
      return;
    }
    const mount = Object.keys(mounts).find((prefix) => url.pathname.startsWith(prefix));
    if (!mount) { res.writeHead(404).end(); return; }
    const path = resolve(mounts[mount], decodeURIComponent(url.pathname.slice(mount.length)));
    if (!path.startsWith(mounts[mount] + sep)) { res.writeHead(403).end(); return; }
    const body = await readFile(path);
    const range = req.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
    if (range) {
      const start = Number(range[1]), end = Number(range[2]);
      if (start > end || end >= body.length) { res.writeHead(416).end(); return; }
      res.writeHead(206, { "content-range": `bytes ${start}-${end}/${body.length}` });
      res.end(body.subarray(start, end + 1));
    } else {
      res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" });
      res.end(body);
    }
  } catch { res.writeHead(404).end(); }
});
let browser;
const reply = (value) => process.stdout.write(JSON.stringify(value) + "\n");
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  browser = await chromium.launch({ args: ["--enable-unsafe-webgpu", ...(process.platform === "darwin" ? ["--use-angle=metal"] : [])] });
  const page = await browser.newPage();
  page.on("console", (message) => { if (["error", "warning"].includes(message.type())) console.error(message.text()); });
  page.on("pageerror", (error) => console.error(error));
  // `--renderer=webgl2` drives the Compat tier (compat code in harness.wasm).
  const renderer = process.argv.find((a) => a.startsWith("--renderer="))?.slice("--renderer=".length) ?? "webgpu";
  await page.goto(`http://127.0.0.1:${server.address().port}/?renderer=${renderer}`);
  await page.waitForFunction(() => window.renderer, null, { timeout: 60_000 });
  reply({ ready: true, browser: browser.version(), max_size: await page.evaluate(() => window.renderer.maxTextureDimension2D), text_matrix: await page.evaluate(() => window.renderer.textMatrixDigest()) });
  for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    const request = JSON.parse(line);
    if (request.query) {
      // A query fixture: settle, then query; features come back as JSON.
      try {
        const r = request;
        const features = await page.evaluate((r) => window.renderer.query(r.style, r.width, r.height, r.logical_width, r.logical_height, r.pixel_ratio, r.timeout_ms, r.fade_duration, r.cross_source_collisions, r.local_ideographs, r.max_pitch, r.operations, r.query), r);
        reply({ features });
      } catch (error) {
        await page.reload();
        await page.waitForFunction(() => window.renderer, null, { timeout: 60_000 });
        reply({ error: String(error) });
      }
      continue;
    }
    try {
      // Playwright's test timeout is not in use here. A watchdog also covers
      // device loss/readback that never completes after the render deadline.
      const result = await page.evaluate(async (r) => {
        let timer;
        try {
          return await Promise.race([
            (async () => {
              const pixels = await window.renderer.render(r.style, r.width, r.height, r.logical_width, r.logical_height, r.pixel_ratio, r.timeout_ms, r.fade_duration, r.cross_source_collisions, r.local_ideographs, r.max_pitch, r.operations);
              try {
                // Chunk the conversion to avoid the JS argument-count limit.
                const bytes = pixels.pixels;
                let binary = "";
                for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
                return { rgba: btoa(binary), warnings: pixels.warnings, frames: pixels.frames, tiles_drawn: pixels.tiles_drawn, timed_out: pixels.timed_out };
              } finally { pixels.free(); }
            })(),
            new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("render/readback timeout")), r.timeout_ms + 10_000); }),
          ]);
        } finally { clearTimeout(timer); }
      }, request);
      await writeFile(request.output, Buffer.from(result.rgba, "base64"));
      delete result.rgba;
      reply(result);
    } catch (error) {
      // A rejected render may still own a pending wasm borrow. A new page
      // discards that device and map before the next request.
      await page.reload();
      await page.waitForFunction(() => window.renderer, null, { timeout: 60_000 });
      reply({ error: String(error) });
    }
  }
} catch (error) {
  reply({ error: String(error) });
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  server.close();
}
