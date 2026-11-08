// The map app: Orbis drawing a picked style full-bleed through the
// `orbis` package, with a style editor, feature popups and highlights
// through feature state, and a debug overlay.
//
// URL: `?style=<key|url>` (styles.js), `&flight` (fly an entry's looping
// flight, as picking it does; without it the map stays put and settles),
// `&debug`, `&edit`, `&lose=<ms>` (lose the GPU that long after loading,
// to show recovery),
// and the camera as `#zoom/lat/lon[/bearing/pitch]`.
import { createMap } from "orbis";
import { GROUPS, LIVE_CAMERA, sceneFor, sceneUrl } from "./styles.js";
import { expose, hashCamera } from "./expose.js";
import { attribution } from "./app/attribution.js";
import { editor as createEditor } from "./app/editor.js";
import { Highlighter, targetOf } from "./app/highlight.js";
import { hud as createHud } from "./app/hud.js";
import { picker as createPicker } from "./app/picker.js";
import { offline as createOffline } from "./app/offline.js";
import { featurePopup } from "./app/popup.js";

const app = document.getElementById("app");
const container = document.getElementById("map");
const toast = document.getElementById("toast");
const params = () => new URLSearchParams(location.search);

let map;
let current;
let popup;
let highlighter;

// Chrome tone: dark maps and a dark system preference turn it dark.
const prefersDark = matchMedia("(prefers-color-scheme: dark)");
function tone() {
  document.documentElement.dataset.tone = current?.scene.dark || prefersDark.matches ? "dark" : "light";
}
prefersDark.addEventListener("change", tone);

function say(text, { error = false } = {}) {
  toast.textContent = text;
  toast.classList.toggle("error", error);
}

// URL state.
function styleParam({ scene, format }) {
  if (scene.formats) return `${scene.key}-${format}`;
  return scene.key === "liberty" ? null : scene.key;
}
const searchOf = (query) => {
  const search = query.toString().replace(/=(?=&|$)/g, "");
  return search ? `?${search}` : "";
};
/** The app's URL for `next` with `camera` as its hash. */
function urlFor(next, camera) {
  const query = params();
  query.delete("format");
  query.delete("flight");
  const style = styleParam(next);
  if (style) query.set("style", style); else query.delete("style");
  if (next.scene.flight) query.set("flight", "");
  return `${location.pathname}${searchOf(query)}${camera ? hashOf(camera) : ""}`;
}
function hashOf({ center: [lon, lat], zoom, bearing = 0, pitch = 0 }) {
  const parts = [zoom.toFixed(2), lat.toFixed(5), lon.toFixed(5)];
  if (Math.abs(bearing) > 0.05 || pitch > 0.05) parts.push(bearing.toFixed(1), pitch.toFixed(0));
  return `#${parts.join("/")}`;
}
function writeHash() {
  if (map) history.replaceState(history.state, "", `${location.pathname}${location.search}${hashOf(map.camera.current)}`);
}
function setFlag(name, on) {
  const query = params();
  if (on) query.set(name, ""); else query.delete(name);
  history.replaceState(history.state, "", `${location.pathname}${searchOf(query)}${location.hash}`);
}

// The chrome.
const attributionControl = attribution();
const picker = createPicker(GROUPS, { onPick: (scene, format) => go({ scene, format }) });
const hudControl = createHud();
const editor = createEditor({
  onClose: () => setEditing(false),
  onReplaced: () => { highlighter?.reset(); clearPopup(); },
});

function setEditing(on) {
  app.classList.toggle("editing", on);
  editor.show(on);
  document.getElementById("edit-toggle").setAttribute("aria-pressed", String(on));
  setFlag("edit", on);
  if (!on) {
    void highlighter?.set("hover", null);
    editor.inspect(null);
    document.getElementById("edit-toggle").focus({ preventScroll: true });
  }
}
document.getElementById("edit-toggle").addEventListener("click", () => setEditing(!editor.open));

const debugToggle = document.getElementById("debug-toggle");
function setDebug(on) {
  hudControl.visible = on;
  debugToggle.setAttribute("aria-checked", String(on));
  setFlag("debug", on);
}
debugToggle.addEventListener("click", () => setDebug(debugToggle.getAttribute("aria-checked") !== "true"));

const menu = document.getElementById("menu");
menu.addEventListener("beforetoggle", (e) => {
  if (e.newState !== "open") return;
  const rect = document.getElementById("menu-button").getBoundingClientRect();
  menu.style.setProperty("--menu-top", `${rect.bottom + 8}px`);
  menu.style.setProperty("--menu-right", `${innerWidth - rect.right}px`);
});
const offlineAreas = createOffline({
  view() {
    if (!map) return null;
    const { width, height } = map.canvas.getBoundingClientRect();
    const corners = [[0, 0], [width, 0], [width, height], [0, height]].map((p) => map.unproject(p));
    if (corners.some((c) => !c)) return null;
    const lons = corners.map((c) => c[0]);
    const lats = corners.map((c) => c[1]);
    const zoom = Math.floor(map.camera.current.zoom);
    return {
      name: `${current.scene.name} near ${map.camera.current.center.map((v) => v.toFixed(3)).reverse().join(", ")}`,
      style: sceneUrl(current.scene, current.format),
      bounds: [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)],
      zooms: [zoom, Math.min(zoom + 3, 16)],
    };
  },
});
document.getElementById("offline-button").addEventListener("click", () => { menu.hidePopover(); offlineAreas.open(); });
// A still of the view, once everything in it has loaded, as a PNG.
document.getElementById("picture-button").addEventListener("click", async () => {
  menu.hidePopover();
  if (!map) return;
  say("Drawing the picture…");
  try {
    const image = await map.snapshot();
    const canvas = new OffscreenCanvas(image.width, image.height);
    canvas.getContext("2d").putImageData(image, 0, 0);
    const url = URL.createObjectURL(await canvas.convertToBlob({ type: "image/png" }));
    Object.assign(document.createElement("a"), { href: url, download: `${current.scene.key}.png` }).click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    say("");
  } catch (e) {
    say(`The picture failed: ${e instanceof Error ? e.message : e}`, { error: true });
  }
});
const about = document.getElementById("about");
document.getElementById("about-button").addEventListener("click", () => { menu.hidePopover(); about.showModal(); });
{
  const commit = __ORBIS_COMMIT__;
  const link = document.createElement("a");
  link.href = `https://github.com/sargunv/orbis/commit/${commit.sha}`;
  link.textContent = `${commit.short}${commit.dirty ? "-dirty" : ""}`;
  document.getElementById("build").append("This build: ", link);
}

// The map's focus ring shows for keyboard users only: the canvas takes
// focus on every pointer press too.
addEventListener("keydown", (e) => { if (e.key === "Tab") app.classList.add("keyboard"); }, true);
addEventListener("pointerdown", () => app.classList.remove("keyboard"), true);

// Camera controls.
const compass = document.getElementById("compass");
document.getElementById("zoom-in").addEventListener("click", () => {
  map?.camera.easeTo({ zoom: map.camera.current.zoom + 1, duration: reducedMotion() ? 0 : 300 });
});
document.getElementById("zoom-out").addEventListener("click", () => {
  map?.camera.easeTo({ zoom: map.camera.current.zoom - 1, duration: reducedMotion() ? 0 : 300 });
});
compass.addEventListener("click", () => map?.camera.easeTo({ bearing: 0, pitch: 0, duration: reducedMotion() ? 0 : 400 }));
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
function showCompass() {
  const { bearing, pitch } = map.camera.current;
  compass.style.setProperty("--bearing", `${-bearing}deg`);
  compass.style.setProperty("--pitch", `${pitch * 0.7}deg`);
}

function clearPopup() {
  popup?.remove();
  popup = undefined;
  void highlighter?.set("selected", null);
}

/**
 * Moves to `next`: flies there when only the camera differs, else loads
 * its style in a new page. (A page draws one map: on Android Chrome a
 * second map in the same page doesn't present yet.) Basemaps keep the
 * camera between them.
 */
function go(next) {
  const keep = current.scene.basemap && next.scene.basemap && map;
  const camera = keep ? map.camera.current : next.scene.camera ?? (next.scene.basemap ? LIVE_CAMERA : undefined);
  const sameCalls = JSON.stringify(current.scene.calls) === JSON.stringify(next.scene.calls);
  if (map && sceneUrl(current.scene, current.format) === sceneUrl(next.scene, next.format) && sameCalls
    && !next.scene.flight) {
    // The same style on another surface switches in place: no tile loads.
    const projection = (scene) => scene.projection ?? { type: "mercator" };
    if (JSON.stringify(projection(current.scene)) !== JSON.stringify(projection(next.scene))) {
      map.edit((tx) => tx.setProjection(projection(next.scene)));
    }
    current = next;
    picker.show(next.scene, next.format);
    history.replaceState(history.state, "", urlFor(next, camera ?? map.camera.current));
    if (camera) map.camera.flyTo({ bearing: 0, pitch: 0, ...camera, duration: reducedMotion() ? 0 : undefined });
    return;
  }
  location.assign(urlFor(next, camera));
}

async function load(next, { camera }) {
  current = next;
  tone();
  picker.show(next.scene, next.format);
  app.dataset.map = "loading";
  say(`Loading ${next.scene.name}…`);
  let created;
  try {
    created = await createMap({
      container,
      style: sceneUrl(next.scene, next.format),
      camera,
      constraints: next.scene.constraints,
    });
  } catch (e) {
    app.dataset.map = "error";
    say(`The map couldn't start: ${e instanceof Error ? e.message : e}`, { error: true });
    console.error(e);
    return;
  }
  map = created;
  expose(map);
  highlighter = new Highlighter(map, () => editor.style);
  map.on("error", ({ message }) => {
    console.error(message);
    if (app.dataset.map === "loading") say(message, { error: true });
  });
  map.on("styleload", async () => {
    if (next.scene.projection) await map.edit((tx) => tx.setProjection(next.scene.projection));
    if (next.scene.calls) await map.edit((tx) => { for (const [name, ...args] of next.scene.calls) tx[name](...args); });
    await editor.attach(map);
    attributionControl.update(editor.style);
  });
  map.on("load", () => { if (!toast.classList.contains("error")) say(""); });
  map.on("warning", ({ message }) => console.warn(message));
  map.on("render", (ack) => {
    app.dataset.map = ack.idle ? "idle" : "drawing";
    hudControl.frame(ack);
    editor.frame(ack);
    showCompass();
  });
  map.on("moveend", writeHash);
  // How long a device loss took to recover: to the first frame on the new
  // device, and to the next complete one.
  let lostAt;
  map.on("restored", ({ recoveredMs }) => {
    const idle = map.on("idle", () => {
      idle();
      const recovered = `first frame ${Math.round(recoveredMs)} ms` +
        (lostAt === undefined ? "" : `, complete ${Math.round(performance.now() - lostAt)} ms`);
      hudControl.recovered(recovered);
      console.info(`orbis: recovered from device loss: ${recovered}`);
      app.dataset.recovered = "true";
    });
  });
  const query = params();
  if (query.has("lose")) {
    const loaded = map.on("load", () => {
      loaded();
      setTimeout(() => { lostAt = performance.now(); map.loseDevice(); },
        Number(query.get("lose")) || 0);
    });
  }
  wireFeatures(map);

  if (next.scene.flight && params().has("flight")) {
    // From the style's own camera, which it adopts as it loads.
    (async () => {
      await created.once("styleload");
      for (let leg = 0; ; leg = (leg + 1) % next.scene.flight.length) {
        if (!await created.camera.flyTo(next.scene.flight[leg])) return;
      }
    })();
  }
  if (next.scene.basemap || next.scene.key === "tokyo" || next.scene.key === "playground") {
    const pin = document.createElement("div");
    pin.className = "pin demo-annotation";
    pin.title = "Tokyo Station";
    map.annotations.add({ element: pin, lngLat: [139.7671, 35.6812] });
  }
}

/** Popups on click, hover state while editing, a pointer cursor over features. */
function wireFeatures(target) {
  target.on("click", async ({ point, lngLat }) => {
    clearPopup();
    const [feature] = await target.queryRenderedFeatures(point);
    if (!feature || !lngLat) return;
    const element = featurePopup(feature, {
      onClose: clearPopup,
      onEdit: (layer) => { setEditing(true); editor.reveal(layer); },
    });
    popup = target.annotations.add({ element, lngLat, anchor: "bottom", offset: [0, -14] });
    const selected = feature.id !== undefined && targetOf({ ...feature, sourceLayer: feature.sourceLayer ?? null }, editor.style);
    await highlighter?.set("selected", selected || null);
  });
  let pointer;
  let busy = false;
  const hover = async () => {
    if (busy) return;
    busy = true;
    try {
      while (pointer !== undefined) {
        const point = pointer;
        pointer = undefined;
        const hit = point && target.hitTest(point);
        target.canvas.style.cursor = hit ? "pointer" : "";
        hudControl.hover(hit ? `${hit.layer}${hit.id !== null ? ` #${hit.id}` : ""}` : "");
        if (!editor.open) continue;
        editor.inspect(hit);
        await highlighter.set("hover", targetOf(hit, editor.style));
      }
    } finally { busy = false; }
  };
  target.canvas.addEventListener("pointermove", (e) => {
    if (e.buttons || e.pointerType === "touch") return;
    pointer = [e.offsetX, e.offsetY];
    void hover();
  });
  target.canvas.addEventListener("pointerleave", () => { pointer = null; void hover(); });
}

// Hooks for the headless checks and the console.
window.app = { get map() { return map; }, editor, get scene() { return current?.scene.key; } };

// Start.
const start = params();
setDebug(start.has("debug"));
if (start.has("edit")) setEditing(true);
{
  const camera = hashCamera();
  let first;
  try { first = sceneFor(start, camera); } catch (e) { say(String(e.message), { error: true }); throw e; }
  history.replaceState(history.state, "", urlFor(first, camera));
  await load(first, { camera: camera ?? first.scene.camera ?? (first.scene.basemap ? LIVE_CAMERA : undefined) });
}

