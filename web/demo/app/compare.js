// The comparison: another engine's map of the same style over the app's,
// clipped to the right of a draggable divider, so each side shows one map
// and takes its own input. The map that last took input leads and the
// other jumps to its camera, so the two don't echo each other's moves.
// The other map takes Orbis's style as it stands (projection, sky and the
// entry's calls applied) and follows the style editor's edits.
//
// An engine loads only when a comparison opens. Each is a module with a
// `label` and `open(container, { style, camera, constraints })`, which
// returns the map as `{ camera, jumpTo, onMove, stop, setStyle, loaded,
// remove }`; `unsupported(scene)` says why it can't show an entry.
import { h } from "./dom.js";
import { HIGHLIGHT_PREFIX } from "./highlight.js";

const ENGINES = {
  gljs: { load: () => import("./gljs.js"), unsupported: (scene) => scene.notInGlJs },
};

// Where the divider sits, as a fraction of the stage's width; it stays for
// the next comparison.
let split = 0.5;

/** The toggle in the app's menu, and the comparison while it's open. */
export function comparison({ onToggle }) {
  const engine = ENGINES.gljs;
  const toggle = document.getElementById("compare-toggle");
  const note = document.getElementById("compare-note");
  let wanted = false;
  let map;
  let scene;
  let shown;
  let attempt = 0;

  async function update() {
    const id = ++attempt;
    shown?.close();
    shown = undefined;
    const reason = scene && engine.unsupported(scene);
    toggle.disabled = !map || Boolean(reason);
    toggle.setAttribute("aria-checked", String(wanted && !reason));
    note.textContent = reason ?? "";
    note.hidden = !reason;
    if (!wanted || !map || reason) return;
    const module = await engine.load();
    if (id === attempt) shown = show(module, map, scene);
  }
  const run = () => update().catch((e) => console.error(e));

  toggle.addEventListener("click", () => {
    wanted = toggle.getAttribute("aria-checked") !== "true";
    onToggle(wanted);
    run();
  });

  return {
    set wanted(on) { wanted = on; run(); },
    /** Compares `next`'s map of `entry`, once its style is in; with no map, closes. */
    attach(next, entry) { map = next; scene = entry; run(); },
    /** The other map's camera and whether it's settled, for the headless checks. */
    get other() { return shown && { camera: shown.other.camera(), loaded: shown.other.loaded() }; },
  };
}

function show(engine, map, scene) {
  const stage = map.container.parentElement;
  const container = h("div", { class: "compare-map" });
  const divider = h("div", {
    class: "compare-divider", role: "slider", tabindex: "0", "aria-label": "Divider between the maps",
    "aria-valuemin": "0", "aria-valuemax": "100",
  });
  const labels = [h("span", { class: "compare-label own" }, "Orbis"),
    h("span", { class: "compare-label other" }, engine.label)];
  map.container.after(container, divider, ...labels);

  const style = () => {
    const now = map.getStyle();
    return { ...now, layers: now.layers.filter((l) => !l.id.startsWith(HIGHLIGHT_PREFIX)) };
  };
  const other = engine.open(container, { style: style(), camera: map.camera.current, constraints: scene.constraints });

  // Input on a map makes it lead and stops the other's motion, such as a
  // drag's momentum. The capture phase sees it before either map does.
  let leader = "orbis";
  const listening = new AbortController();
  const { signal } = listening;
  for (const type of ["pointerdown", "wheel", "keydown"]) {
    stage.addEventListener(type, (e) => {
      if (e.target === divider) return;
      const next = container.contains(e.target) ? "other" : "orbis";
      if (next === leader) return;
      leader = next;
      if (leader === "other") map.camera.stop(); else other.stop();
    }, { capture: true, passive: true, signal });
  }
  const offs = [
    map.on("move", () => { if (leader === "orbis") other.jumpTo(map.camera.current); }),
    other.onMove(() => { if (leader === "other") map.camera.jumpTo(other.camera()); }),
  ];
  // Style edits, at most once a frame.
  let restyle = 0;
  offs.push(map.on("styledata", () => {
    restyle ||= requestAnimationFrame(() => { restyle = 0; other.setStyle(style()); });
  }));

  // The divider: a drag or the arrow keys move it.
  const place = (fraction) => {
    split = Math.min(1, Math.max(0, fraction));
    stage.style.setProperty("--split", `${split * 100}%`);
    divider.setAttribute("aria-valuenow", String(Math.round(split * 100)));
  };
  place(split);
  divider.addEventListener("pointerdown", (e) => divider.setPointerCapture(e.pointerId), { signal });
  divider.addEventListener("pointermove", (e) => {
    if (!divider.hasPointerCapture(e.pointerId)) return;
    const rect = stage.getBoundingClientRect();
    place((e.clientX - rect.left) / rect.width);
  }, { signal });
  divider.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 0.1 : 0.02;
    const next = { ArrowLeft: split - step, ArrowRight: split + step, Home: 0, End: 1 }[e.key];
    if (next === undefined) return;
    e.preventDefault();
    place(next);
  }, { signal });

  return {
    other,
    close() {
      listening.abort();
      for (const off of offs) off();
      cancelAnimationFrame(restyle);
      other.remove();
      for (const element of [container, divider, ...labels]) element.remove();
    },
  };
}
