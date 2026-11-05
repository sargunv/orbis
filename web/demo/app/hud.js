// The debug overlay: the last frame's counters, the thread mode switch and
// the build. Off by default; `?debug` or the menu turns it on.
import { h } from "./dom.js";

const kib = (bytes) => `${(bytes / 1024).toFixed(1)} KiB`;
const COMMIT = __ORBIS_COMMIT__;

export function hud({ onThread }) {
  const root = document.getElementById("hud");
  const list = h("dl");
  const threads = h("span", { class: "segmented", role: "group", "aria-label": "Where the map draws" },
    ["main", "worker"].map((thread) => h("button", {
      type: "button", "data-thread": thread, "aria-pressed": "false", onclick: () => onThread(thread),
    }, thread === "main" ? "Main thread" : "Worker")));
  const commit = h("a", { href: `https://github.com/sargunv/orbis/commit/${COMMIT.sha}` },
    `${COMMIT.short}${COMMIT.dirty ? "-dirty" : ""}`);
  root.replaceChildren(list, h("div", { class: "hud-foot" }, threads, h("span", {}, "Build ", commit)));
  let latest;
  let hovered = "";
  let tier = "";
  let recovery = "";
  let queued = false;

  const draw = () => {
    queued = false;
    const ack = latest;
    if (!ack || root.hidden) return;
    const rows = [
      ...(tier ? [["renderer", tier]] : []),
      ["zoom", ack.zoom.toFixed(2)],
      ["CPU", `${ack.ms.toFixed(1)} ms`],
      ["draws", ack.draws],
      ["dispatches", ack.dispatches],
      ["candidates", ack.candidates],
      ["tiles", ack.tilesLoading ? `${ack.tilesDrawn}, ${ack.tilesLoading} loading` : ack.tilesDrawn],
      ["bundle", ack.bundleRecorded ? "recorded" : "replayed"],
      ["placements", ack.placementsWritten],
      ["paints", ack.paintsWritten],
      ["from JS", kib(ack.uploadedFromJs)],
      ["from wasm", kib(ack.uploadedFromWasm)],
      ...(ack.texelsFromJs || ack.texelsFromWasm
        ? [["texels JS", kib(ack.texelsFromJs)], ["texels wasm", kib(ack.texelsFromWasm)]] : []),
      ...(ack.jobsDone ? [["jobs", ack.jobsDone], ["headers", kib(ack.headerBytes)]] : []),
      ["generation", ack.generation],
      ["state", ack.idle ? "idle" : "drawing"],
      ...(recovery ? [["recovered", recovery]] : []),
      ...(hovered ? [["hover", hovered]] : []),
    ];
    list.replaceChildren(...rows.map(([key, value]) => h("div", {}, h("dt", {}, key), h("dd", {}, String(value)))));
  };
  const request = () => { if (!queued) { queued = true; requestAnimationFrame(draw); } };

  return {
    set visible(on) { root.hidden = !on; request(); },
    thread(thread) {
      for (const button of threads.children) button.setAttribute("aria-pressed", String(button.dataset.thread === thread));
    },
    frame(ack) { latest = ack; request(); },
    hover(text) { hovered = text; request(); },
    renderer(text) { tier = text; request(); },
    recovered(text) { recovery = text; request(); },
  };
}
