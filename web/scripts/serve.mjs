// Builds and serves web/dist over HTTP with Range support, and launches headless
// Chromium with WebGPU, for the snapshot and bench scripts.
import { createServer } from "node:http";
import { open, readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { build } from "vite";

const root = fileURLToPath(new URL("../dist/", import.meta.url));
const types = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".css": "text/css" };

// Resolves to { url, close } once listening on a free local port.
// `isolated` serves the pages cross-origin isolated, which refines
// `performance.now()` from 100 µs to 5 µs in Chrome.
export async function serveDemo({ isolated = false } = {}) {
  await build({ configFile: fileURLToPath(new URL("../vite.config.js", import.meta.url)), logLevel: "warn" });
  const isolation = isolated
    ? { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp" }
    : {};
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    if (!url.pathname.startsWith("/orbis/")) { res.writeHead(404).end(); return; }
    let pathname = decodeURIComponent(url.pathname.slice("/orbis/".length));
    if (!pathname || pathname.endsWith("/")) pathname += "index.html";
    const path = normalize(join(root, pathname));
    try {
      if (!path.startsWith(root)) { res.writeHead(403).end(); return; }
      const size = (await stat(path)).size;
      const match = req.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
      if (match) {
        const start = Number(match[1]), end = Number(match[2]);
        if (start > end || end >= size) { res.writeHead(416).end(); return; }
        const file = await open(path);
        try {
          const body = Buffer.alloc(end - start + 1);
          await file.read(body, 0, body.length, start);
          res.writeHead(206, { "content-range": `bytes ${start}-${end}/${size}`,
            "content-length": body.length, "accept-ranges": "bytes" });
          res.end(body);
        } finally { await file.close(); }
        return;
      }
      const body = await readFile(path);
      res.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream",
        ...isolation });
      res.end(body);
    } catch {
      res.writeHead(404).end();
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return { url: `http://127.0.0.1:${server.address().port}/orbis/`, close: () => server.close() };
}

// `unsafe` allows WebGPU's unsafe APIs, which also stops Chrome quantizing
// timestamps; a user's Chrome runs without it.
export function launchChromium({ unsafe = true } = {}) {
  return chromium.launch({
    args: [...(unsafe ? ["--enable-unsafe-webgpu"] : []), "--enable-features=Vulkan,WebGPU",
      "--use-angle=metal"],
  });
}
