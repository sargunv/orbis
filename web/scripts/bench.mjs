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
// default with `--style=alps`) or globe (the default with
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
const engines =
  params.get("experiment") === "gestures"
    ? [params.get("engine") ?? "orbis"]
    : (params.get("engine") ?? "orbis").split(",");

const server = await serveDemo({ isolated: precise });
let browser,
  page,
  cleanup = () => {};
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
    // Warnings are shown, not failures: Chrome warns that it ignores
    // `powerPreference` on Windows, GL JS about the style. GL JS's errors
    // are about the style too, not the run.
    if (m.type() === "error" && !from.includes("/maplibre-gl")) errors++;
  });
  page.on("pageerror", (e) => {
    errors++;
    console.error(`page: ${e.message}`);
  });
  // The gestures page's own intervals are between animation frames, which
  // keep their pace while the GPU runs behind: the GPU process's presents
  // show what reached the screen.
  const tracing =
    params.get("experiment") === "gestures" && !android
      ? await startTrace(await page.context().newCDPSession(page))
      : null;
  for (const engine of engines) {
    params.set("engine", engine);
    await page.goto(`${server.url}bench.html?${params}`);
    let last = "";
    for (;;) {
      const { state, text } = await page
        .locator("#status")
        .evaluate(
          (e) => ({ state: e.dataset.state, text: e.textContent }),
          null,
          { timeout: 600_000 },
        );
      if (text !== last) {
        console.error(`bench: ${text}`);
        last = text;
      }
      if (state === "error") throw new Error(text);
      if (state === "done") break;
      await page.waitForTimeout(500);
    }
    process.stdout.write(`${await page.locator("#markdown").inputValue()}\n`);
  }
  if (tracing) process.stdout.write(presentsReport(await tracing.stop()));
  if (errors) throw new Error(`${errors} page errors`);
} finally {
  if (android) await cleanup();
  else await browser.close();
  server.close();
}

/** Traces the page's user timing marks and the GPU process's presents; `stop()` resolves to the events. */
async function startTrace(cdp) {
  await cdp.send("Tracing.start", {
    categories: "blink.user_timing,gpu,viz",
    transferMode: "ReturnAsStream",
  });
  return {
    async stop() {
      const done = new Promise((resolve) =>
        cdp.once("Tracing.tracingComplete", resolve),
      );
      await cdp.send("Tracing.end");
      const { stream } = await done;
      const chunks = [];
      for (let eof = false; !eof;) {
        const read = await cdp.send("IO.read", { handle: stream });
        chunks.push(read.data);
        eof = read.eof;
      }
      const trace = JSON.parse(chunks.join(""));
      return Array.isArray(trace) ? trace : trace.traceEvents;
    },
  };
}

/**
 * The GPU process's presents between each path's `gestures:<engine>:<path>:start`
 * and `:end` marks (gestures.js), as a Markdown table: their intervals and
 * how many are late, as the page counts its animation frames'.
 */
function presentsReport(events) {
  const times = (name) =>
    events
      .filter((e) => e.name === name)
      .map((e) => e.ts / 1000)
      .sort((a, b) => a - b);
  const mac = times("ImageTransportSurfaceOverlayMac::Present");
  const presents = mac.length ? mac : times("Display::DrawAndSwap");
  const marks = new Map(
    events
      .filter((e) => e.name.startsWith("gestures:"))
      .map((e) => [e.name, e.ts / 1000]),
  );
  const late = [16.7, 25];
  const at = (s, p) =>
    s[Math.min(s.length - 1, Math.round((s.length - 1) * p))]?.toFixed(1) ??
    "—";
  const rows = [];
  for (const [name, start] of marks) {
    if (!name.endsWith(":start")) continue;
    const [, engine, path] = name.split(":");
    const end = marks.get(name.replace(/:start$/, ":end"));
    const shown = presents.filter((t) => t > start && t <= end);
    const gaps = shown
      .slice(1)
      .map((t, i) => t - shown[i])
      .sort((a, b) => a - b);
    rows.push(
      `| ${engine === "gl" ? "GL JS" : "Orbis"} | ${path} | ${shown.length} | ${[0.5, 0.95, 0.99, 1].map((p) => at(gaps, p)).join(" | ")} | ` +
        `${late.map((l) => gaps.filter((g) => g > l + 1).length).join(" | ")} |`,
    );
  }
  if (!rows.length) return "";
  return [
    "### GPU process presents while the camera moved",
    "",
    `| engine | path | presents | interval p50 | p95 | p99 | max | ${late.map((l) => `gaps > ${l} ms`).join(" | ")} |`,
    "|---|---|---|---|---|---|---|---|---|",
    ...rows,
    "",
    `From a Chrome trace (${mac.length ? "ImageTransportSurfaceOverlayMac::Present" : "Display::DrawAndSwap"}): every frame the browser ` +
      "showed, whichever engine drew it, in the order they ran; gaps with 1 ms of slack. A GPU running behind shows here and not " +
      "in the page's intervals. Headless Chrome's presents wander a couple of ms around the display frame, so gaps over 16.7 ms " +
      "count that too; over 25 ms, a frame missed.",
    "",
  ].join("\n");
}
