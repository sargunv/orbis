// Builds and runs web/dist/bench.html and prints its Markdown report: in
// headless Chromium, or in Chrome on Android.
//
//   pnpm bench [--precise] [--android [--physical]] [--firefox [--prefs=name=value,…]] [--webkit] [--unpaced] [--globe] [--name=value…]
//
// `--name=value` pairs become the page's query (format, style, frames,
// warmup, rounds, size, ratio, at, experiment, backend (`webgl2` draws
// Orbis with WebGL 2), views (`0,2`: the first and third alone), drop
// (`symbol,line`: the style without those layers) and engine (`orbis,gl`:
// GL JS too, the engines taking turns view by view); workers (job workers),
// paths, engine and gpu (`--gpu=1`: GPU passes on frames that laid the
// cull's candidates out again and on the rest) and pace (`--pace=wall`:
// each path takes 6 s however slowly frames come, as a hand's gesture
// does, where by default it takes 360 display frames) for
// `--experiment=gestures`; filter, legs, engine and rest (ms at rest
// before each path moves) for `--experiment=stutter`). Gesture paths are city, dense (pan, zoom
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
// `performance.now()` (to 100 µs). `--firefox` runs the page in Firefox
// instead: the installed one, headed, on WebGPU (serve.mjs's
// `launchFirefox`; `--prefs` sets its prefs), else Playwright's headless,
// which draws Orbis with WebGL 2. `--webkit` runs it in Playwright's
// WebKit, on WebGPU, Orbis on the page's thread as in Safari. Neither
// traces presents (flight.mjs records Firefox's composites). `--unpaced` runs animation frames as fast as the
// browser draws them (Chromium's flags, Firefox's `layout.frame_rate`
// 0; in Firefox with Orbis on the page, whose workers stay paced), so a
// still view's repaint interval is what a frame costs rather than the
// display's. `--precise` allows WebGPU's unsafe APIs,
// which stops the first, and serves the page cross-origin isolated, which
// refines the second to 5 µs: to check that the page's means agree with
// exact medians. `--android` reverses the server's port onto this worktree's
// Android emulator (`scripts/android-serial`; `ANDROID_SERIAL` overrides) and
// opens the page in its Chrome; keep the screen on and unlocked. `--physical`
// measures a phone instead of the emulator: assign one to this worktree with
// `devices-cli assign <serial>` first and run `devices-cli release` after.
// A device's frame bench is checked against this machine: the same views
// drawn here, in headless Chromium on the same backend, their snapshots
// compared cell by cell, and the run fails where a view's picture differs
// (a phone's WebGL 2 drew no fills or lines for days of benches, #140).
import { parsePrefs } from "./firefox.mjs";
import {
  launchChromium,
  launchFirefox,
  launchWebkit,
  openAndroid,
  serveDemo,
} from "./serve.mjs";

const FLAGS = [
  "--precise",
  "--android",
  "--physical",
  "--globe",
  "--firefox",
  "--webkit",
  "--unpaced",
];
// The picture check (`checkPictures`).
/** A cell whose channels differ by more than this differs. */
const CELL_TOLERANCE = 24;
/** A view whose cells differ beyond this share drew wrong. */
const WRONG_SHARE = 0.1;

const args = process.argv.slice(2);
const precise = args.includes("--precise");
const inFirefox = args.includes("--firefox");
const inWebkit = args.includes("--webkit");
const prefs = parsePrefs(
  args.find((a) => a.startsWith("--prefs="))?.slice("--prefs=".length),
);
const unpaced = args.includes("--unpaced");
const physical = args.includes("--physical");
const android = physical || args.includes("--android");
const params = new URLSearchParams();
if (args.includes("--globe")) params.set("globe", "");
for (const arg of args) {
  if (FLAGS.includes(arg) || arg.startsWith("--prefs=")) continue;
  const match = arg.match(/^--([a-z]+)=(.+)$/);
  if (!match) throw new Error(`unknown argument ${arg}`);
  params.set(match[1], match[2]);
}
if ((android || inFirefox || inWebkit) && precise)
  throw new Error("--precise needs headless Chromium");
if (android && (inFirefox || inWebkit || unpaced))
  throw new Error("--firefox, --webkit and --unpaced run on this machine");
if (inWebkit && unpaced) throw new Error("--unpaced needs Chromium or Firefox");
// The gestures and frames pages interleave engines themselves, path by
// path or view by view; for the rest, one page load an engine.
const engines = ["gestures", "frames"].includes(
  params.get("experiment") ?? "frames",
)
  ? [params.get("engine") ?? "orbis"]
  : (params.get("engine") ?? "orbis").split(",");

const server = await serveDemo({ isolated: precise });
let browser,
  page,
  cleanup = () => {};
if (android) {
  ({ browser, page, close: cleanup } = await openAndroid(server, { physical }));
} else {
  browser = inFirefox
    ? await launchFirefox({
        prefs: {
          // As Talos runs Firefox unpaced: tasks still run between frames.
          ...(unpaced && {
            "layout.frame_rate": 0,
            "docshell.event_starvation_delay_hint": 1,
          }),
          ...prefs,
        },
      })
    : inWebkit
      ? await launchWebkit()
      : await launchChromium({ unsafe: precise, unpaced });
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
    params.get("experiment") === "gestures" &&
    !android &&
    !inFirefox &&
    !inWebkit
      ? await startTrace(await page.context().newCDPSession(page))
      : null;
  const checked =
    android && (params.get("experiment") ?? "frames") === "frames";
  for (const engine of engines) {
    params.set("engine", engine);
    await runPage(page, `${server.url}bench.html?${params}`);
    process.stdout.write(`${await page.locator("#markdown").inputValue()}\n`);
    if (checked) {
      const pictures = await page.evaluate(() => globalThis.benchPictures);
      const wrong = await checkPictures(server, params, pictures);
      if (wrong) {
        process.exitCode = 1;
        console.error(
          `bench: ${wrong} views drew pictures unlike this machine's`,
        );
      }
    }
  }
  if (tracing) process.stdout.write(presentsReport(await tracing.stop()));
  if (errors) throw new Error(`${errors} page errors`);
} finally {
  if (android) await cleanup();
  else await browser.close();
  server.close();
}

/** Runs the bench page at `url` on `page` until it's done. */
async function runPage(page, url) {
  await page.goto(url);
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
    if (state === "done") return;
    await page.waitForTimeout(500);
  }
}

/**
 * Draws `params`' views in headless Chromium on this machine, a round of
 * two frames each, and prints how the device's `pictures`
 * (`frames.js`'s `benchPictures`) differ from them; resolves to how many
 * views differ past `WRONG_SHARE`.
 */
async function checkPictures(server, params, pictures) {
  const here = new URLSearchParams(params);
  for (const [key, value] of [
    ["frames", "2"],
    ["warmup", "1"],
    ["rounds", "1"],
  ])
    here.set(key, value);
  const browser = await launchChromium();
  let reference;
  try {
    const page = await browser.newPage();
    await runPage(page, `${server.url}bench.html?${here}`);
    reference = await page.evaluate(() => globalThis.benchPictures);
  } finally {
    await browser.close();
  }
  const lines = ["", "#### Pictures against this machine's", ""];
  let wrong = 0;
  for (const [engine, views] of Object.entries(pictures)) {
    for (const [view, cells] of Object.entries(views)) {
      const theirs = reference[engine]?.[view];
      if (!theirs || theirs.length !== cells.length) {
        lines.push(`- ${engine}, ${view}: **no picture here to compare**`);
        wrong++;
        continue;
      }
      let differ = 0;
      for (let i = 0; i < cells.length; i += 3) {
        const d = Math.max(
          ...[0, 1, 2].map((c) => Math.abs(cells[i + c] - theirs[i + c])),
        );
        if (d > CELL_TOLERANCE) differ++;
      }
      const share = differ / (cells.length / 3);
      const bad = share > WRONG_SHARE;
      if (bad) wrong++;
      lines.push(
        `- ${engine}, ${view}: ${(share * 100).toFixed(1)}% of cells differ${bad ? " — **drawn wrong**" : ""}`,
      );
    }
  }
  process.stdout.write(`${lines.join("\n")}\n`);
  return wrong;
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
