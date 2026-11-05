// Builds and runs web/dist/bench.html and prints its Markdown report: in
// headless Chromium, or in Chrome on Android.
//
//   pnpm bench [--precise] [--android [--physical]] [--name=value…]
//
// `--name=value` pairs become the page's query (format, style, frames,
// warmup, rounds, size, ratio, at, experiment; workers
// (job workers) and paths for `--experiment=gestures`: city|alps, alps with `--style=alps`). Chrome runs as a
// user's would, quantizing timestamps (to 65.5 µs on a Mac) and
// `performance.now()` (to 100 µs). `--precise` allows WebGPU's unsafe APIs,
// which stops the first, and serves the page cross-origin isolated, which
// refines the second to 5 µs: to check that the page's means agree with
// exact medians. `--android` reverses the server's port onto this worktree's
// Android emulator (`scripts/android-serial`; `ANDROID_SERIAL` overrides) and
// opens the page in its Chrome; keep the screen on and unlocked. `--physical`
// measures a phone instead of the emulator: assign one to this worktree with
// `devices-cli assign <serial>` first and run `devices-cli release` after.
import { launchChromium, openAndroid, serveDemo } from "./serve.mjs";

const args = process.argv.slice(2);
const precise = args.includes("--precise");
const physical = args.includes("--physical");
const android = physical || args.includes("--android");
const params = new URLSearchParams();
for (const arg of args) {
  if (arg === "--precise" || arg === "--android" || arg === "--physical") continue;
  const match = arg.match(/^--([a-z]+)=(.+)$/);
  if (!match) throw new Error(`unknown argument ${arg}`);
  params.set(match[1], match[2]);
}
if (android && precise) throw new Error("--precise needs headless Chromium");

const server = await serveDemo({ isolated: precise });
let browser, page, cleanup = () => {};
if (android) {
  ({ browser, page, close: cleanup } = await openAndroid(server, { physical }));
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
  if (android) await cleanup();
  else await browser.close();
  server.close();
}
