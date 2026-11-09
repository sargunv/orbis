// Builds and serves web/dist over HTTP with Range support, and launches headless
// Chromium with WebGPU, for the snapshot and bench scripts.
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { open, readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { build } from "vite";

const root = fileURLToPath(new URL("../dist/", import.meta.url));
const androidSerial = fileURLToPath(new URL("../../scripts/android-serial", import.meta.url));
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".css": "text/css" };

// Resolves to { url, close } once listening on a free local port.
// `isolated` serves the pages cross-origin isolated, which refines
// `performance.now()` from 100 µs to 5 µs in Chrome. `assets` serves that
// directory at `fixtures/` too (the GL JS fixtures'), its JSON's `local://`
// URLs pointing there; `offline` points its `https://` and `mapbox://` URLs
// at its `https/` and `mapbox/` directories too (MapLibre Native's cache).
// `mounts` serves more directories, by path prefix (`{ "mlnrs/": dir }`).
// `builds` serves other builds of web/dist beside this one, by label, for
// A/B runs in one session: `@<label>/load.html` is that build's page, and
// its content-hashed assets are found under `assets/` with this build's.
export async function serveDemo({ isolated = false, assets, offline = false, mounts = {}, builds = {} } = {}) {
  await build({ configFile: fileURLToPath(new URL("../vite.config.js", import.meta.url)), logLevel: "warn" });
  const isolation = isolated
    ? { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp" }
    : {};
  const server = createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    if (!url.pathname.startsWith("/orbis/")) { res.writeHead(404).end(); return; }
    let pathname = decodeURIComponent(url.pathname.slice("/orbis/".length));
    if (!pathname || pathname.endsWith("/")) pathname += "index.html";
    const mount = Object.keys(mounts).find((prefix) => pathname.startsWith(prefix));
    const page = pathname.match(/^@([^/]+)\/(.*)$/);
    let [dir, file] = mount ? [mounts[mount], pathname.slice(mount.length)]
      : assets && pathname.startsWith("fixtures/") ? [assets, pathname.slice(9)]
        : page && builds[page[1]] ? [builds[page[1]], page[2]] : [root, pathname];
    if (dir === root && file.startsWith("assets/") && !existsSync(join(root, file))) {
      dir = Object.values(builds).find((other) => existsSync(join(other, file))) ?? dir;
    }
    const path = normalize(join(dir, file));
    try {
      if (!path.startsWith(dir)) { res.writeHead(403).end(); return; }
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
      let body = await readFile(path);
      if (dir === assets && extname(path) === ".json") {
        const here = `http://127.0.0.1:${server.address().port}/orbis/fixtures/`;
        body = body.toString().replaceAll("local://", here).replaceAll("http://localhost:2900/", here);
        if (offline) body = body.replaceAll("https://", `${here}https/`).replaceAll("mapbox://", `${here}mapbox/`);
      }
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
// timestamps; a user's Chrome runs without it. `headed` opens a window on
// the display rather than drawing offscreen; `unpaced` runs animation frames
// as fast as they draw rather than at the display's rate.
export function launchChromium({ unsafe = true, headed = false, unpaced = false } = {}) {
  return chromium.launch({
    headless: !headed,
    args: [...(unsafe ? ["--enable-unsafe-webgpu"] : []), "--enable-features=Vulkan,WebGPU",
      "--use-angle=metal", ...(unpaced ? ["--disable-gpu-vsync", "--disable-frame-rate-limit"] : [])],
  });
}

// Opens a tab in Chrome on the Android device this worktree tests on: its
// emulator from devices-cli, or the one `ANDROID_SERIAL` names
// (`scripts/android-serial`), serving `server` to it over `adb reverse` and
// driving it through DevTools. `physical` measures a phone that the caller
// assigned to this worktree with `devices-cli assign`, and fails rather than
// fall back to the emulator. Resolves to { browser, page, close }.
export async function openAndroid(server, { physical = false } = {}) {
  const serial = execFileSync(androidSerial, physical ? ["--physical"] : [], { encoding: "utf8" }).trim();
  const adb = (...a) => execFileSync("adb", ["-s", serial, ...a], { encoding: "utf8" });
  const port = new URL(server.url).port;
  adb("reverse", `tcp:${port}`, `tcp:${port}`);
  const devtools = adb("forward", "tcp:0", "localabstract:chrome_devtools_remote").trim();
  // A worktree's emulator instance starts with no Chrome profile, and Chrome's
  // first-run screen never opens the DevTools socket, so its Chrome is told to
  // skip that screen; Chrome reads the command line on a debuggable build,
  // which the emulator image is. A phone keeps the Chrome its owner set up.
  // The emulator's GPU is SwiftShader Vulkan, a software adapter that WebGPU
  // refuses unless Chrome allows it, as headless Chromium does
  // (`launchChromium`), so the emulator's Chrome allows it too.
  const emulator = adb("shell", "getprop", "ro.build.characteristics").includes("emulator");
  if (emulator) {
    const flags = ["chrome", "--disable-fre", "--no-default-browser-check", "--enable-unsafe-webgpu"];
    adb("shell", `echo '${flags.join(" ")}' > /data/local/tmp/chrome-command-line`);
    adb("shell", "am", "force-stop", "com.android.chrome");
  }
  // A phone's Chrome comes to the front without a new tab, so runs leave
  // none behind (each run's own tab closes with it); the emulator's, just
  // restarted, opens on a blank one.
  adb("shell", "am", "start", "-a", "android.intent.action.VIEW", "-n",
    "com.android.chrome/com.google.android.apps.chrome.Main", ...(emulator ? ["-d", "about:blank"] : []));
  let browser;
  for (let i = 0; ; i++) {
    try { browser = await chromium.connectOverCDP(`http://127.0.0.1:${devtools}`); break; }
    catch (e) { if (i > 60) throw e; await new Promise((r) => setTimeout(r, 500)); }
  }
  const page = await browser.contexts()[0].newPage();
  return {
    browser, page,
    close: async () => {
      await page.close().catch(() => {});
      await browser.close();
      adb("forward", "--remove", `tcp:${devtools}`);
      adb("reverse", "--remove", `tcp:${port}`);
    },
  };
}
