// The map app: Orbis drawing a picked style full-bleed through the
// `orbis` package, with a style editor (and, while it's open, feature
// popups and highlights through feature state), and a debug overlay. One
// map shows every entry: picking one sets its style on the map, which
// diffs from the one before, so shared sources keep their tiles and paint
// eases by the style's transition.
//
// URL: `?style=<key|url>` (styles.js), `&flight` (fly an entry's looping
// flight; without it the map stays put and settles),
// `&debug`, `&edit`, `&compare[=<left>,<right>]` (two engines' maps of
// the style side by side, Orbis and GL JS by default, app/compare.js),
// `&light=map|viewport` (where an entry with the light switch anchors its
// light, when not where its style does, app/light.js), `&terrain=1|0`
// (whether an entry with the terrain switch draws terrain, when not its
// default, app/terrain.js),
// `&lose=<ms>` (lose the GPU that long after
// loading, to show recovery), and the camera as
// `#zoom/lat/lon[/bearing/pitch]`.
import { createMap } from "orbis";
import { DEFAULT_SCENE, GROUPS, LIVE_CAMERA, sceneFor, sceneStyle, sceneUrl, terrainByDefault } from "./styles.js";
import { expose, hashCamera } from "./expose.js";
import { attribution } from "./app/attribution.js";
import { comparison, formatPair, parsePair } from "./app/compare.js";
import { editor as createEditor } from "./app/editor.js";
import { Highlighter, targetOf } from "./app/highlight.js";
import { hud as createHud } from "./app/hud.js";
import { anchorOf, lightSwitch, setAnchor } from "./app/light.js";
import { picker as createPicker } from "./app/picker.js";
import { setTerrain, terrainOf, terrainSwitch } from "./app/terrain.js";
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
/** Bumped on every switch: a flight or camera move for an older entry stops. */
let switches = 0;
/** A switched-to style is on its way: the map reads as loading, not idle, until it arrives. */
let styleLoading = false;
/** The loaded style's own sky, which an entry's `sky` stands in for. */
let styleSky = null;
/** The loaded style's own illumination anchor, which `&light` overrides. */
let styleAnchor;
/** The terrain the entry's switch turns on: its own, or the loaded style's. */
let entryTerrain = null;

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
  return scene.key === DEFAULT_SCENE.key ? null : scene.key;
}
const searchOf = (query) => {
  const search = query.toString().replace(/=(?=&|$)/g, "");
  return search ? `?${search}` : "";
};
/** The app's URL for `next` with `camera` as its hash. `&flight`, `&light` and `&terrain` stay only for an entry with one. */
function urlFor(next, camera) {
  const query = params();
  query.delete("format");
  if (!next.scene.flight) query.delete("flight");
  if (!next.scene.light) query.delete("light");
  if (!next.scene.terrain) query.delete("terrain");
  const style = styleParam(next);
  if (style) query.set("style", style); else query.delete("style");
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
/** Sets the URL's `name` to `value` (`true` for a bare flag), or removes it for a falsy one. */
function setParam(name, value) {
  const query = params();
  if (value === true) query.set(name, "");
  else if (value) query.set(name, value);
  else query.delete(name);
  history.replaceState(history.state, "", `${location.pathname}${searchOf(query)}${location.hash}`);
}
const setFlag = (name, on) => setParam(name, on);

// The chrome.
const attributionControl = attribution();
const picker = createPicker(GROUPS, { onPick: (scene, format) => go({ scene, format }) });
const hudControl = createHud();
const compare = comparison({
  onChange(pair) {
    if (menu.matches(":popover-open")) menu.hidePopover();
    setParam("compare", pair && (formatPair(pair) || true));
  },
  beforeReload: () => writeHash(),
});
const editor = createEditor({
  onClose: () => setEditing(false),
  onReplaced: () => { highlighter?.reset(); clearPopup(); void showLight(); void showTerrain(); },
});
const light = lightSwitch({
  async onPick(anchor) {
    if (!map) return;
    light.show(anchor);
    if (anchor !== anchorOf(await map.getStyle())) await setAnchor(map, anchor);
    await showLight();
    await editor.refresh();
  },
});
const terrain = terrainSwitch({
  async onPick(on) {
    if (!map) return;
    terrain.show(on);
    if (on !== terrainOf(await map.getStyle())) await setTerrain(map, on ? entryTerrain : null);
    await showTerrain();
    await editor.refresh();
  },
});
/** Whether `&terrain` asks for terrain: true, false, or null for the entry's default. */
function wantedTerrain() {
  const value = params().get("terrain");
  return value === "1" ? true : value === "0" ? false : null;
}
/** Shows whether the style draws terrain in its switch and the URL, for an entry with the terrain switch. */
async function showTerrain() {
  if (!map || styleLoading || !current.scene.terrain) {
    terrain.show(null);
    return;
  }
  const on = terrainOf(await map.getStyle());
  terrain.show(on);
  setParam("terrain", on !== terrainByDefault(current.scene) && (on ? "1" : "0"));
}

/** Shows the style's illumination anchor in its switch and the URL, for an entry with the light switch. */
async function showLight() {
  if (!map || styleLoading || !current.scene.light) {
    light.show(null);
    return;
  }
  const anchor = anchorOf(await map.getStyle());
  light.show(anchor);
  setParam("light", anchor !== styleAnchor && anchor);
}

/** Opens or closes the style editor; feature popups and highlights live only while it's open. */
function setEditing(on) {
  app.classList.toggle("editing", on);
  editor.show(on);
  document.getElementById("edit-toggle").setAttribute("aria-pressed", String(on));
  setFlag("edit", on);
  if (!on) {
    clearPopup();
    void highlighter?.set("hover", null);
    editor.inspect(null);
    if (map) map.canvas.style.cursor = "";
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
  async view() {
    if (!map) return null;
    const { width, height } = map.canvas.getBoundingClientRect();
    const corners = await Promise.all([[0, 0], [width, 0], [width, height], [0, height]].map((p) => map.unproject(p)));
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

const projectionOf = (scene) => scene.projection ?? { type: "mercator" };

/**
 * Moves to `next`, keeping the camera: switching entries never moves it,
 * though the entry's constraints clamp it where they must (the globe no
 * smaller than at zoom 0, a bounded dataset where it has data). Back and
 * forward pass the history entry's `camera`. The same style on another
 * surface switches its projection in place; another style is set on the
 * map as the diff from this one. `history` pushes the move as a history
 * entry, or replaces the current one.
 */
function go(next, { camera, history: mode = "push" } = {}) {
  if (!map) {
    // No map started (no WebGPU, say): the new page tries again.
    location.assign(urlFor(next, camera));
    return;
  }
  const switched = ++switches;
  const url = urlFor(next, camera ?? map.camera.current);
  if (mode === "push") history.pushState(null, "", url); else history.replaceState(history.state, "", url);
  const previous = current;
  current = next;
  tone();
  picker.show(next.scene, next.format);
  map.camera.stop();
  map.camera.setConstraints(next.scene.constraints ?? {});
  if (camera) map.camera.jumpTo(camera);
  if (sceneUrl(previous.scene, previous.format) === sceneUrl(next.scene, next.format)
    && !previous.scene.restyle && !next.scene.restyle) {
    // The same style on another surface or under another sky: one edit,
    // and nothing loads again.
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const projection = !same(projectionOf(previous.scene), projectionOf(next.scene));
    const sky = !same(previous.scene.sky, next.scene.sky);
    if (projection || sky) {
      void map.edit((tx) => {
        if (projection) tx.setProjection(projectionOf(next.scene));
        if (sky) tx.setSky(next.scene.sky ?? styleSky);
      });
      // The comparison reopens with the projection, if it can draw it.
      compare.attach(map, next.scene);
    }
    startFlight(switched);
    return;
  }
  compare.attach(undefined, next.scene);
  clearPopup();
  say(`Loading ${next.scene.name}…`);
  styleLoading = true;
  void showLight();
  void showTerrain();
  app.dataset.map = "loading";
  // The style's `styleload` applies the entry's projection and sky, unless
  // it restyles, which puts them in the style. The style's own camera is a
  // fresh page's (load), never a switch's.
  sceneStyle(next.scene, next.format, { terrain: wantedTerrain() === true }).then((style) => { if (switched === switches) map.setStyle(style); }, (e) => {
    if (switched !== switches) return;
    styleLoading = false;
    app.dataset.map = "error";
    say(e instanceof Error ? e.message : String(e), { error: true });
  });
  const loaded = map.on("styleload", () => {
    loaded();
    startFlight(switched);
    if (!toast.classList.contains("error")) say("");
  });
}

/** Flies the entry's looping flight, if it has one and the URL asks, until the user or another entry takes the camera. */
async function startFlight(switched) {
  const { flight } = current.scene;
  if (!flight || !params().has("flight")) return;
  for (let leg = 0; switched === switches; leg = (leg + 1) % flight.length) {
    if (!await map.camera.flyTo(flight[leg])) return;
  }
}

// Back and forward move between entries as picking them did.
addEventListener("popstate", () => {
  let next;
  try { next = sceneFor(params()); } catch { return; }
  go(next, { camera: hashCamera(), history: "none" });
});

async function load(next, { camera }) {
  current = next;
  tone();
  picker.show(next.scene, next.format);
  compare.attach(undefined, next.scene);
  app.dataset.map = "loading";
  say(`Loading ${next.scene.name}…`);
  let created;
  try {
    created = await createMap({
      container,
      style: await sceneStyle(next.scene, next.format, { terrain: wantedTerrain() === true }),
      camera,
      constraints: next.scene.constraints,
      // The demo shows what Orbis can do beyond the spec too.
      previews: "all",
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
    if (app.dataset.map === "loading" || toast.textContent.startsWith("Loading")) say(message, { error: true });
  });
  // Each style, the first and those switched to: the entry's projection and
  // sky (unless it restyles, which put them in the style), then the editor,
  // attribution and comparison read it.
  map.on("styleload", async () => {
    const { scene } = current;
    styleLoading = false;
    highlighter.reset();
    const loaded = await map.getStyle();
    styleSky = loaded.sky ?? null;
    if (!scene.restyle && (scene.projection || scene.sky)) {
      await map.edit((tx) => {
        if (scene.projection) tx.setProjection(scene.projection);
        if (scene.sky) tx.setSky(scene.sky);
      });
    }
    if (scene.light) {
      styleAnchor = anchorOf(loaded);
      const wanted = params().get("light");
      if ((wanted === "map" || wanted === "viewport") && wanted !== styleAnchor) await setAnchor(map, wanted);
    }
    await showLight();
    if (scene.terrain) {
      // The entry's own terrain, or the style's as it loaded; `&terrain=0`
      // takes the style's away (`&terrain=1` put the entry's in the style).
      entryTerrain = scene.terrain === true ? loaded.terrain ?? null : scene.terrain;
      if (wantedTerrain() === false && terrainOf(loaded)) await setTerrain(map, null);
    }
    await showTerrain();
    await editor.attach(map);
    attributionControl.update(editor.style);
    compare.attach(map, scene);
  });
  map.on("load", () => { if (!toast.classList.contains("error")) say(""); });
  map.on("warning", ({ message }) => console.warn(message));
  map.on("render", (ack) => {
    app.dataset.map = styleLoading ? "loading" : ack.idle ? "idle" : "drawing";
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
  // From the style's own camera, which it adopts as it loads.
  const switched = switches;
  const loaded = map.on("styleload", () => { loaded(); startFlight(switched); });
}

/**
 * While the style editor is open: popups and a highlight on click, hover
 * state, a pointer cursor over features. Otherwise a click or tap is only a
 * gesture; the debug overlay still names the feature under the pointer.
 */
function wireFeatures(target) {
  target.on("click", async ({ point, lngLat }) => {
    clearPopup();
    if (!editor.open) return;
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
        if (!editor.open && !hudControl.visible) continue;
        const hit = point && await target.hitTest(point);
        hudControl.hover(hit ? `${hit.layer}${hit.id !== null ? ` #${hit.id}` : ""}` : "");
        if (!editor.open) continue;
        target.canvas.style.cursor = hit ? "pointer" : "";
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
window.app = { get map() { return map; }, editor, compare, get scene() { return current?.scene.key; },
  get popup() { return popup; },
  /** Switches to `style`, a catalogue key or a style URL, as picking an entry does. */
  show(style) { go(sceneFor(new URLSearchParams({ style }))); } };

// Start.
const start = params();
setDebug(start.has("debug"));
compare.want(parsePair(start.get("compare")));
if (start.has("edit")) setEditing(true);
{
  const camera = hashCamera();
  let first;
  try { first = sceneFor(start); } catch (e) { say(String(e.message), { error: true }); throw e; }
  history.replaceState(history.state, "", urlFor(first, camera));
  await load(first, { camera: camera ?? first.scene.camera ?? (first.scene.basemap ? LIVE_CAMERA : undefined) });
}

