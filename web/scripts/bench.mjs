// Builds and runs web/dist/bench.html and prints its Markdown report: in
// headless Chromium, or in Chrome on an Android device over adb.
//
//   pnpm bench [--precise] [--android[=serial]] [--name=value…]
//
// `--name=value` pairs become the page's query (format, style, frames,
// warmup, rounds, size, ratio, at, timing, experiment, labels). Chrome runs as a
// user's would, quantizing timestamps (to 65.5 µs on a Mac) and
// `performance.now()` (to 100 µs). `--precise` allows WebGPU's unsafe APIs,
// which stops the first, and serves the page cross-origin isolated, which
// refines the second to 5 µs: to check that the page's means agree with
// exact medians. `--android` reverses the server's port onto the device,
// opens the page in its Chrome and drives it through DevTools; keep the
// screen on and unlocked.
import { execFileSync } from "node:child_process";
import { chromium } from "playwright";
import { launchChromium, serveDemo } from "./serve.mjs";

const args = process.argv.slice(2);
const precise = args.includes("--precise");
let android = null;
const params = new URLSearchParams();
for (const arg of args) {
  const match = arg.match(/^--([a-z]+)(?:=(.*))?$/);
  if (match?.[1] === "precise" && match[2] === undefined) continue;
  if (match?.[1] === "android") { android = { serial: match[2] ?? null }; continue; }
  if (match && match[2] !== undefined) params.set(match[1], match[2]);
  else throw new Error(`unknown argument ${arg}`);
}
if (android && precise) throw new Error("--precise needs headless Chromium");

const server = await serveDemo({ isolated: precise });
let browser, page, cleanup = () => {};
if (android) {
  const adb = (...a) => execFileSync("adb", [...(android.serial ? ["-s", android.serial] : []), ...a],
    { encoding: "utf8" });
  const port = new URL(server.url).port;
  adb("reverse", `tcp:${port}`, `tcp:${port}`);
  const devtools = adb("forward", "tcp:0", "localabstract:chrome_devtools_remote").trim();
  cleanup = () => {
    adb("forward", "--remove", `tcp:${devtools}`);
    adb("reverse", "--remove", `tcp:${port}`);
  };
  adb("shell", "am", "start", "-a", "android.intent.action.VIEW", "-n",
    "com.android.chrome/com.google.android.apps.chrome.Main", "-d", "about:blank");
  for (let i = 0; ; i++) {
    try { browser = await chromium.connectOverCDP(`http://127.0.0.1:${devtools}`); break; }
    catch (e) { if (i > 60) throw e; await new Promise((r) => setTimeout(r, 500)); }
  }
  page = await browser.contexts()[0].newPage();
} else {
  browser = await launchChromium({ unsafe: precise });
  page = await browser.newPage();
}
try {
  let errors = 0;
  page.on("console", (m) => {
    // Mobile Chrome asks for a favicon the demo has none of.
    if (m.location().url?.endsWith("/favicon.ico")) return;
    if (["error", "warning"].includes(m.type())) { errors++; console.error(`page: ${m.text()}`); }
  });
  page.on("pageerror", (e) => { errors++; console.error(`page: ${e.message}`); });
  await page.goto(`${server.url}bench.html?${params}`);
  let last = "";
  for (;;) {
    const { state, text } = await page.locator("#status")
      .evaluate((e) => ({ state: e.dataset.state, text: e.textContent }), null, { timeout: 600_000 });
    if (text !== last) { console.error(`bench: ${text}`); last = text; }
    if (state === "error") throw new Error(text);
    if (state === "done") break;
    await page.waitForTimeout(500);
  }
  process.stdout.write(await page.locator("#markdown").inputValue());
  if (errors) throw new Error(`${errors} page warnings/errors`);
} finally {
  if (android) await page.close().catch(() => {});
  await browser.close();
  server.close();
  cleanup();
}
