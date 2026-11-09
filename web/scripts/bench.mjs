// Builds and runs web/dist/bench.html and prints its Markdown report: in
// headless Chromium, or in Chrome on Android.
//
//   pnpm bench [--precise] [--android [--physical]] [--globe] [--name=value…]
//
// `--name=value` pairs become the page's query (format, style, frames,
// warmup, rounds, size, ratio, at, experiment; workers (job workers),
// paths and engine for `--experiment=gestures`; filter, legs and engine
// for `--experiment=stutter`). Gesture paths are city, dense (pan, zoom
// and turn at pitch 60 over Paris, on live Liberty unless `--style` gives
// another, such as a `cargo xtask h2h mirror`'s Liberty), alps (the
// default with `--style=terrain`) or globe (the default with
// `--style=globe` or `--globe`, live Liberty or `--style` on the globe).
// `--engine=both` runs the gestures or stutter experiment in Orbis and GL
// JS, interleaved path by path on one page (stutter after an untimed pass
// of each), and prints both reports; for gestures `orbis,gl` does the
// same, for stutter it loads the page once an engine.
// Chrome runs as a
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

const FLAGS = ["--precise", "--android", "--physical", "--globe"];
const args = process.argv.slice(2);
const precise = args.includes("--precise");
const physical = args.includes("--physical");
const android = physical || args.includes("--android");
const params = new URLSearchParams();
if (args.includes("--globe")) params.set("globe", "");
for (const arg of args) {
  if (FLAGS.includes(arg)) continue;
  const match = arg.match(/^--([a-z]+)=(.+)$/);
  if (!match) throw new Error(`unknown argument ${arg}`);
  params.set(match[1], match[2]);
}
if (android && precise) throw new Error("--precise needs headless Chromium");
// The gestures page interleaves engines itself, path by path; for the
// rest, one page load an engine.
const engines = params.get("experiment") === "gestures" ? [params.get("engine") ?? "orbis"]
  : (params.get("engine") ?? "orbis").split(",");

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
    const from = m.location().url ?? "";
    // Mobile Chrome asks for a favicon the demo has none of.
    if (from.endsWith("/favicon.ico")) return;
    if (!["error", "warning"].includes(m.type())) return;
    console.error(`page: ${m.text()}`);
    // GL JS's console warnings are about the style, not the run.
    if (!from.includes("/maplibre-gl")) errors++;
  });
  page.on("pageerror", (e) => { errors++; console.error(`page: ${e.message}`); });
  for (const engine of engines) {
    params.set("engine", engine);
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
    process.stdout.write(`${await page.locator("#markdown").inputValue()}\n`);
  }
  if (errors) throw new Error(`${errors} page warnings/errors`);
} finally {
  if (android) await cleanup();
  else await browser.close();
  server.close();
}
