// The panel over the map when it stops or hangs: what happened, a Reload
// button, and the details to report it (the message, the page, the
// browser, the last frame's counters and recent console errors), copied
// or read in place. A map that stopped says why; one that hasn't drawn or
// gone idle in a while says what it waits on, and can be dismissed, since
// a slow network isn't an error.
import { h } from "./dom.js";

/** How many console errors and warnings the details keep. */
const KEPT = 20;

/**
 * Starts keeping the page's recent console errors and warnings, uncaught
 * errors and unhandled rejections, and returns the panel. `context()`
 * gives the details' page state: `{ scene, backend, stats }`. `onFatal`
 * hears an uncaught error or rejection that stopped the map's wasm.
 */
export function failure(stage, { context, onFatal }) {
  const recent = [];
  const keep = (kind, args) => {
    const text = args
      .map((a) => (a instanceof Error ? a.stack || a.message : String(a)))
      .join(" ");
    recent.push(`${(performance.now() / 1000).toFixed(1)} s ${kind}: ${text}`);
    if (recent.length > KEPT) recent.shift();
  };
  for (const kind of ["error", "warn"]) {
    const original = console[kind].bind(console);
    console[kind] = (...args) => {
      keep(kind, args);
      original(...args);
    };
  }
  // A trap in the map's wasm that reached the page uncaught, or a call
  // into a map that stopped: the map is done.
  const fatal = (reason) =>
    reason instanceof WebAssembly.RuntimeError ||
    String(reason?.message ?? reason).startsWith("the map stopped");
  addEventListener("error", (e) => {
    keep("uncaught", [e.error ?? e.message]);
    if (fatal(e.error)) onFatal(e.error);
  });
  addEventListener("unhandledrejection", (e) => {
    keep("unhandled rejection", [e.reason]);
    if (fatal(e.reason)) onFatal(e.reason);
  });

  const title = h("h2", { id: "failure-title" });
  const message = h("p", { class: "failure-message" });
  const waits = h("ul", { class: "failure-waits" });
  const details = h("pre", { class: "failure-details" });
  const copy = h(
    "button",
    { type: "button", class: "secondary", onclick: () => void copyDetails() },
    "Copy details",
  );
  const dismiss = h(
    "button",
    { type: "button", class: "secondary", onclick: () => hide() },
    "Keep waiting",
  );
  const panel = h(
    "section",
    {
      class: "failure",
      id: "failure",
      role: "alertdialog",
      "aria-labelledby": "failure-title",
      hidden: true,
    },
    title,
    message,
    waits,
    h(
      "div",
      { class: "failure-actions" },
      h(
        "button",
        { type: "button", class: "primary", onclick: () => location.reload() },
        "Reload",
      ),
      copy,
      dismiss,
    ),
    h("details", {}, h("summary", {}, "Details"), details),
  );
  stage.append(panel);

  /** What shows: `stopped`, `hang`, or nothing. */
  let mode = null;
  let headline = "";
  /** The message as the map said it, for the details. */
  let said = "";

  const text = () => {
    const { scene, backend, stats } = context();
    const commit = __ORBIS_COMMIT__;
    return [
      `Orbis demo: ${headline}`,
      said || message.textContent,
      ...[...waits.children].map((li) => `- ${li.textContent}`),
      "",
      `Scene: ${scene ?? "none"}`,
      `URL: ${location.href}`,
      `Build: ${commit.sha}${commit.dirty ? " (dirty)" : ""}`,
      `Backend: ${backend ?? "none yet"}`,
      `WebGPU: ${"gpu" in navigator ? "exposed" : "absent"}`,
      `User agent: ${navigator.userAgent}`,
      `Device pixel ratio: ${devicePixelRatio}, viewport ${innerWidth}x${innerHeight}`,
      `Since load: ${(performance.now() / 1000).toFixed(1)} s`,
      "",
      "Last frame's counters:",
      stats ? JSON.stringify(stats, null, 1) : "no frame yet",
      "",
      "Recent console errors and warnings:",
      ...(recent.length ? recent : ["none"]),
    ].join("\n");
  };

  async function copyDetails() {
    const report = text();
    details.textContent = report;
    try {
      await navigator.clipboard.writeText(report);
      copy.textContent = "Copied";
    } catch {
      // No clipboard here (an insecure origin, a denied permission): the
      // details open to copy by hand.
      panel.querySelector("details").open = true;
      copy.textContent = "Copy failed: select the details";
    }
    setTimeout(() => (copy.textContent = "Copy details"), 3000);
  }

  function show(next, heading, body, reasons = [], raw = "") {
    mode = next;
    headline = heading;
    said = raw;
    title.textContent = heading;
    message.textContent = body;
    waits.replaceChildren(...reasons.map((r) => h("li", {}, r)));
    waits.hidden = reasons.length === 0;
    dismiss.hidden = next !== "hang";
    details.textContent = text();
    panel.dataset.mode = next;
    panel.hidden = false;
  }

  function hide() {
    mode = null;
    panel.hidden = true;
  }

  return {
    /** The map stopped, or couldn't start (`heading`), saying why. */
    stopped(why, heading = "The map stopped") {
      // The heading says it stopped; the message says why, as a sentence.
      const reason = why.replace(/^the map stopped: /, "");
      const sentence = reason.charAt(0).toUpperCase() + reason.slice(1);
      show("stopped", heading, sentence, [], why);
    },
    /** The map hasn't finished loading in `seconds`: what it waits on. */
    hang(seconds, reasons) {
      if (mode === "stopped") return;
      show(
        "hang",
        "Still loading",
        `The map hasn't finished loading after ${seconds} s. It's waiting on:`,
        reasons,
      );
    },
    /** What it waits on now, while the still-loading panel shows. */
    update(reasons) {
      if (mode !== "hang") return;
      waits.replaceChildren(...reasons.map((r) => h("li", {}, r)));
    },
    /** The map loaded after all: the still-loading panel goes. */
    loaded() {
      if (mode === "hang") hide();
    },
    get mode() {
      return mode;
    },
  };
}
