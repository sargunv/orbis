// The installed Firefox driven over Marionette, for what Playwright can't
// reach: the frames Firefox's compositor composited, through the chrome
// window's composition recorder (`windowUtils.startCompositionRecording`,
// as browsertime's visual metrics use it). Playwright's WebDriver BiDi
// session and a Marionette session can't share one Firefox, so this
// launches its own, headed (headless Firefox blocklists WebGPU), and gives
// a page with the few calls the measurement scripts make.
//
// `frameReport` reads a recording back: which map frame each composite
// showed, by a hash of the canvas's pixels, so it counts composites that
// showed an older map frame than the one before (stale frames, as Firefox
// bug 2091844 shows a WebGPU canvas drawn in a worker while the page's DOM
// changes) and composites that showed no new one.
import { spawn } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createConnection } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { decodePng } from "./png.mjs";

/** The installed Firefox, where there is one: `ORB_FIREFOX`, else the
 * platform's usual install. */
export const installedFirefox = () =>
  [
    process.env.ORB_FIREFOX,
    {
      darwin: "/Applications/Firefox.app/Contents/MacOS/firefox",
      linux: "/usr/bin/firefox",
    }[process.platform],
  ].find((path) => path && existsSync(path));

/** `name=value,…` as Firefox prefs: booleans, numbers or strings. */
export const parsePrefs = (text) =>
  Object.fromEntries(
    (text?.split(",") ?? []).map((pref) => {
      const [name, value] = pref.split("=");
      return [
        name,
        value === "true"
          ? true
          : value === "false"
            ? false
            : Number.isNaN(Number(value))
              ? value
              : Number(value),
      ];
    }),
  );

/**
 * Launches the installed Firefox on a fresh profile with `prefs`, its
 * window `width`×`height` CSS pixels of content, and resolves to a page:
 * `goto`, `evaluate(fn, arg)` (a function source run in the page, awaited),
 * `waitForFunction`, `record()` (starts the composition recorder; its
 * `stop()` resolves to the composites' PNG paths, by time) and `close`.
 */
export async function openFirefox({
  prefs = {},
  width = 1600,
  height = 1000,
} = {}) {
  const firefox = installedFirefox();
  if (!firefox) throw new Error("no installed Firefox (ORB_FIREFOX)");
  const profile = mkdtempSync(join(tmpdir(), "orbis-firefox-"));
  const recordings = join(profile, "recordings");
  mkdirSync(recordings);
  const port = 20000 + Math.floor(Math.random() * 20000);
  const all = {
    "marionette.port": port,
    "layers.windowrecording.path": `${recordings}/`,
    "browser.shell.checkDefaultBrowser": false,
    "browser.startup.homepage_override.mstone": "ignore",
    "browser.startup.page": 0,
    "browser.aboutwelcome.enabled": false,
    "datareporting.policy.dataSubmissionEnabled": false,
    "toolkit.telemetry.reportingpolicy.firstRun": false,
    "app.update.disabledForTesting": true,
    ...prefs,
  };
  writeFileSync(
    join(profile, "user.js"),
    Object.entries(all)
      .map(([k, v]) => `user_pref(${JSON.stringify(k)}, ${JSON.stringify(v)});`)
      .join("\n"),
  );
  const child = spawn(
    firefox,
    ["-marionette", "-no-remote", "-profile", profile, "about:blank"],
    {
      stdio: "ignore",
      // The composition recorder is on the chrome window.
      env: { ...process.env, MOZ_REMOTE_ALLOW_SYSTEM_ACCESS: "1" },
    },
  );
  const cleanup = () => {
    child.kill();
    rmSync(profile, { recursive: true, force: true });
  };
  // A run stopped by a signal (an alarm's timeout) leaves no Firefox.
  const stop = (signal) => {
    cleanup();
    process.exit(128 + ({ SIGINT: 2, SIGTERM: 15, SIGALRM: 14 }[signal] ?? 1));
  };
  for (const signal of ["SIGINT", "SIGTERM", "SIGALRM"])
    process.once(signal, stop);
  process.once("exit", () => child.kill());
  let marionette;
  try {
    marionette = await connect(port);
    await marionette.send("WebDriver:NewSession", {});
    // Scripts as long as a flight.
    await marionette.send("WebDriver:SetTimeouts", { script: 600_000 });
  } catch (e) {
    cleanup();
    throw e;
  }
  const run = async (context, script, args) => {
    await marionette.send("Marionette:SetContext", { value: context });
    const { value } = await marionette.send("WebDriver:ExecuteAsyncScript", {
      script,
      args,
    });
    if (value?.thrown) throw new Error(value.thrown);
    return value;
  };
  // Marionette runs content scripts in a sandbox over the page, where
  // globals they set never reach the page's own; so the function runs in a
  // script element in the page's realm and posts its result back as JSON on
  // an event.
  const evaluate = async (fn, arg) => {
    const reply = await run(
      "content",
      `const [source, done] = arguments;
       const id = "orbis-marionette-" + Math.random().toString(36).slice(2);
       document.addEventListener(id, (e) => done(e.detail), { once: true });
       const script = document.createElement("script");
       script.textContent = source.replace("@ID@", JSON.stringify(id));
       document.documentElement.append(script);
       script.remove();`,
      [
        `Promise.resolve()
          .then(() => (${fn})(${JSON.stringify(arg ?? null)}))
          .then((value) => ({ value }), (e) => ({ thrown: String(e?.stack ?? e) }))
          .then((r) => {
            let detail;
            try {
              detail = JSON.stringify(r);
            } catch {
              // As Playwright gives what it can't serialize.
              detail = "{}";
            }
            document.dispatchEvent(new CustomEvent(@ID@, { detail }));
          });`,
      ],
    );
    const { value, thrown } = JSON.parse(reply);
    if (thrown) throw new Error(thrown);
    return value;
  };
  const chrome = (body) =>
    run(
      "chrome",
      `const done = arguments[0];
       const w = Services.wm.getMostRecentWindow("navigator:browser");
       (async () => { ${body} })().then(done, (e) => done({ thrown: String(e) }));`,
      [],
    );
  // The content area's size: the window's, less the toolbars.
  const chromeHeight = await chrome(
    "return w.outerHeight - w.gBrowser.selectedBrowser.getBoundingClientRect().height;",
  );
  await marionette.send("WebDriver:SetWindowRect", {
    x: 0,
    y: 0,
    width,
    height: height + chromeHeight,
  });
  return {
    async goto(url) {
      await marionette.send("Marionette:SetContext", { value: "content" });
      await marionette.send("WebDriver:Navigate", { url });
    },
    evaluate,
    async waitForFunction(fn, arg, { timeout = 30_000 } = {}) {
      const until = Date.now() + timeout;
      while (!(await evaluate(`(arg) => !!(${fn})(arg)`, arg))) {
        if (Date.now() > until) throw new Error(`timed out waiting for ${fn}`);
        await new Promise((r) => setTimeout(r, 100));
      }
    },
    /** The content area in the recording's pixels: its offset in the window, and the device pixel ratio. */
    content: () =>
      chrome(`const r = w.gBrowser.selectedBrowser.getBoundingClientRect();
        return { x: r.x, y: r.y, ratio: w.devicePixelRatio };`),
    async record() {
      const before = new Set(
        existsSync(recordings) ? readdirSync(recordings) : [],
      );
      await chrome("w.windowUtils.startCompositionRecording();");
      return {
        async stop() {
          await chrome("await w.windowUtils.stopCompositionRecording(true);");
          // Firefox writes the PNGs after the call resolves.
          let dir;
          for (let i = 0; i < 600 && !dir; i++) {
            dir =
              existsSync(recordings) &&
              readdirSync(recordings).find((d) => !before.has(d));
            if (!dir) await new Promise((r) => setTimeout(r, 100));
          }
          if (!dir) throw new Error("the composition recorder wrote nothing");
          dir = join(recordings, dir);
          // Complete once the count stops changing.
          let count = -1;
          for (;;) {
            const n = readdirSync(dir).length;
            if (n === count) break;
            count = n;
            await new Promise((r) => setTimeout(r, 1000));
          }
          return readdirSync(dir)
            .map((name) => {
              const [, index, ms] =
                name.match(/^frame-(\d+)-(\d+)\.png$/) ?? [];
              return index && { path: join(dir, name), index: +index, ms: +ms };
            })
            .filter(Boolean)
            .sort((a, b) => a.index - b.index);
        },
      };
    },
    async close() {
      await marionette
        .send("Marionette:Quit", { flags: ["eForceQuit"] })
        .catch(() => {});
      marionette.close();
      await new Promise((r) => {
        child.once("exit", r);
        setTimeout(r, 5000);
      });
      cleanup();
    },
  };
}

/** A Marionette connection on `port`: `send(command, params)` resolves to its reply. */
async function connect(port) {
  let socket;
  for (let i = 0; !socket; i++) {
    try {
      socket = await new Promise((resolve, reject) => {
        const s = createConnection(port, "127.0.0.1", () => resolve(s));
        s.once("error", reject);
      });
    } catch (e) {
      if (i > 150)
        throw new Error(`Firefox's Marionette never listened: ${e.message}`);
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  const waiting = new Map();
  let buffer = Buffer.alloc(0);
  let id = 0;
  let greeted;
  const greeting = new Promise((r) => (greeted = r));
  // Each packet is `<length>:<JSON>`; the first is a greeting, the rest
  // `[1, id, error, result]`.
  socket.on("data", (data) => {
    buffer = Buffer.concat([buffer, data]);
    for (;;) {
      const colon = buffer.indexOf(58);
      if (colon < 0) return;
      const length = Number(buffer.subarray(0, colon).toString());
      if (buffer.length < colon + 1 + length) return;
      const packet = JSON.parse(
        buffer.subarray(colon + 1, colon + 1 + length).toString(),
      );
      buffer = buffer.subarray(colon + 1 + length);
      if (!Array.isArray(packet)) {
        greeted();
        continue;
      }
      const [, at, error, result] = packet;
      const w = waiting.get(at);
      waiting.delete(at);
      if (error) w.reject(new Error(`${error.error}: ${error.message}`));
      else w.resolve(result);
    }
  });
  await greeting;
  return {
    send: (command, params = {}) =>
      new Promise((resolve, reject) => {
        const at = ++id;
        waiting.set(at, { resolve, reject });
        const body = Buffer.from(JSON.stringify([0, at, command, params]));
        socket.write(`${body.length}:`);
        socket.write(body);
      }),
    close: () => socket.destroy(),
  };
}

/**
 * What a recording's composites (`record().stop()`'s) showed of the map
 * canvas at `rect` (`{ x, y, width, height }` in the recording's pixels):
 * each composite's map frame by a hash of a grid of its pixels, then how
 * many composites showed a new map frame, the same one again, or one the
 * screen had shown before the last (stale), and the intervals between
 * composites that showed a new one. The map's frames during a flight are
 * all distinct, so a frame seen again after another is an older one shown.
 */
export function frameReport(composites, rect) {
  const seen = new Map();
  let previous;
  let fresh = 0,
    repeated = 0,
    stale = 0;
  const shown = [];
  for (const { path, ms } of composites) {
    const image = decodePng(readFileSync(path));
    let hash = 2166136261;
    for (let gy = 0; gy < 48; gy++) {
      const y = Math.min(
        image.height - 1,
        Math.round(rect.y + ((gy + 0.5) * rect.height) / 48),
      );
      for (let gx = 0; gx < 64; gx++) {
        const x = Math.min(
          image.width - 1,
          Math.round(rect.x + ((gx + 0.5) * rect.width) / 64),
        );
        const at = (y * image.width + x) * image.channels;
        for (let c = 0; c < 3; c++)
          hash = Math.imul(hash ^ image.pixels[at + c], 16777619);
      }
    }
    if (hash === previous) repeated++;
    else if (seen.has(hash)) stale++;
    else {
      fresh++;
      shown.push(ms);
    }
    seen.set(hash, ms);
    previous = hash;
    rmSync(path);
  }
  const gaps = shown
    .slice(1)
    .map((t, i) => t - shown[i])
    .sort((a, b) => a - b);
  const at = (p) =>
    gaps[Math.min(gaps.length - 1, Math.round((gaps.length - 1) * p))] ?? 0;
  return {
    composites: composites.length,
    fresh,
    repeated,
    stale,
    newFrameInterval: {
      p50: at(0.5),
      p95: at(0.95),
      max: at(1),
      over25: gaps.filter((g) => g > 25).length,
    },
  };
}
