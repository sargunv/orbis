// The map app: Orbis drawing a picked style full-bleed through the
// `orbis` package, with the map options panel (the style's switches,
// app/options.js) and a debug overlay. One map shows every entry: picking
// one sets its style on the map, which diffs from the one before, so
// shared sources keep their tiles and paint eases by the style's
// transition.
//
// URL: `?style=<key|url>` (styles.js), `&flight` (fly an entry's looping
// flight; without it the map stays put and settles),
// `&debug`, `&compare[=<left>,<right>]` (two engines' maps of
// the style side by side, Orbis and GL JS by default, app/compare.js),
// the map options that differ from the entry's own (`&terrain=1`,
// `&light=map`, `&shadows=1`, `&projection=mercator`, …, app/options.js),
// `&backend=webgpu|webgl2` (the renderer, Auto by default, app/options.js),
// `&lose=<ms>` (lose the GPU that long after
// loading, to show recovery), and the camera as
// `#zoom/lat/lon[/bearing/pitch]`.
import { createMap } from "orbis";

import { attribution } from "./app/attribution.js";
import { comparison, formatPair, parsePair } from "./app/compare.js";
import { hud as createHud } from "./app/hud.js";
import { offline as createOffline } from "./app/offline.js";
import {
  mapOptions,
  opening,
  OPTION_PARAMS,
  wantedBackend,
} from "./app/options.js";
import { picker as createPicker } from "./app/picker.js";
import { expose, hashCamera } from "./expose.js";
import {
  DEFAULT_SCENE,
  GROUPS,
  LIVE_CAMERA,
  sceneFor,
  sceneStyle,
  sceneUrl,
} from "./styles.js";

const app = document.getElementById("app");
const container = document.getElementById("map");
const toast = document.getElementById("toast");
const params = () => new URLSearchParams(location.search);

let map;

/** What each gap does to the map, said briefly. */
const GAP_NAMES = {
  "depth-precision": "coarser 3D depth",
  "heatmap-precision": "8-bit heatmaps",
};

/** Notes what the map's GPU draws otherwise than WebGPU, under the scene
 * button; nothing when it draws everything. */
function showGaps(gaps) {
  const badge = document.getElementById("degraded");
  badge.hidden = gaps.length === 0;
  const backend = map?.backend === "webgl2" ? "WebGL 2" : "This GPU";
  badge.textContent = `${backend}: ${gaps.map((g) => GAP_NAMES[g.gap] ?? g.gap).join(", ")}`;
  badge.title = gaps
    .map(({ gap, layers }) =>
      layers.length
        ? `${GAP_NAMES[gap] ?? gap}: ${layers.join(", ")}`
        : `${GAP_NAMES[gap] ?? gap}: the whole map`,
    )
    .join("\n");
}

let current;
/** Bumped on every switch: a flight or camera move for an older entry stops. */
let switches = 0;
/** A switched-to style is on its way: the map reads as loading, not idle, until it arrives. */
let styleLoading = false;
/** The loaded style's own sky, which an entry's `sky` stands in for. */
let styleSky = null;
/** The loaded style as its entry designs it, and the terrain its switch turns on: the options' baseline. */
let designed;
/** The last style the demo built for an entry (`styleFor`), as designed: `{ scene, style }`. */
let design;

// Chrome tone: dark maps and a dark system preference turn it dark.
const prefersDark = matchMedia("(prefers-color-scheme: dark)");
function tone() {
  document.documentElement.dataset.tone =
    current?.scene.dark || prefersDark.matches ? "dark" : "light";
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
/**
 * The app's URL for `next` with `camera` as its hash. `&flight` stays only
 * for an entry with one, and the map options only for the entry they were
 * chosen on (or with `keepOptions`, for a page or a history entry that
 * names them).
 */
function urlFor(
  next,
  camera,
  { keepOptions = next.scene === current?.scene } = {},
) {
  const query = params();
  query.delete("format");
  if (!next.scene.flight) query.delete("flight");
  if (!keepOptions) for (const name of OPTION_PARAMS) query.delete(name);
  const style = styleParam(next);
  if (style) query.set("style", style);
  else query.delete("style");
  return `${location.pathname}${searchOf(query)}${camera ? hashOf(camera) : ""}`;
}
function hashOf({ center: [lon, lat], zoom, bearing = 0, pitch = 0 }) {
  const parts = [zoom.toFixed(2), lat.toFixed(5), lon.toFixed(5)];
  if (Math.abs(bearing) > 0.05 || pitch > 0.05)
    parts.push(bearing.toFixed(1), pitch.toFixed(0));
  return `#${parts.join("/")}`;
}
let hashTimer;
function writeHash() {
  clearTimeout(hashTimer);
  hashTimer = undefined;
  if (map)
    history.replaceState(
      history.state,
      "",
      `${location.pathname}${location.search}${hashOf(map.camera.current)}`,
    );
}
/** Writes the camera once moves pause: a camera set every frame (the
 * comparison's other map, a script's tour) ends a move every frame, and
 * Safari throws past 100 `replaceState` calls in 10 s. */
function writeHashSoon() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(writeHash, 250);
}
/** Sets the URL's `name` to `value` (`true` for a bare flag), or removes it for a falsy one. */
function setParam(name, value) {
  const query = params();
  if (value === true) query.set(name, "");
  else if (value) query.set(name, value);
  else query.delete(name);
  history.replaceState(
    history.state,
    "",
    `${location.pathname}${searchOf(query)}${location.hash}`,
  );
}

// The chrome.
const attributionControl = attribution();
const picker = createPicker(GROUPS, {
  onPick: (scene, format) => go({ scene, format }),
});
const hudControl = createHud();
const compare = comparison({
  onChange(pair) {
    if (menu.matches(":popover-open")) menu.hidePopover();
    setParam("compare", pair && (formatPair(pair) || true));
  },
  beforeReload: () => writeHash(),
});
// Another renderer starts the page again, at this camera.
const options = mapOptions({
  params,
  setParam,
  onBackend(backend) {
    writeHash();
    setParam("backend", backend === "auto" ? null : backend);
    location.reload();
  },
});
/**
 * What the map loads for `next`: its URL, or for an entry the demo builds
 * (a restyle), that style with the URL's map options and the entry's
 * defaults in it, which then keeps the style as designed in `design` for
 * the options to read.
 */
async function styleFor(next) {
  const style = await sceneStyle(next.scene, next.format);
  if (typeof style === "string") return style;
  const { scene } = next;
  const terrain = scene.terrain === true ? style.terrain : scene.terrain;
  design = { scene, style };
  return opening(style, { scene, terrain: terrain ?? null }, params());
}

const debugToggle = document.getElementById("debug-toggle");
function setDebug(on) {
  hudControl.visible = on;
  debugToggle.setAttribute("aria-checked", String(on));
  setParam("debug", on);
}
debugToggle.addEventListener("click", () =>
  setDebug(debugToggle.getAttribute("aria-checked") !== "true"),
);

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
    const corners = await Promise.all(
      [
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
      ].map((p) => map.unproject(p)),
    );
    if (corners.some((c) => !c)) return null;
    const lons = corners.map((c) => c[0]);
    const lats = corners.map((c) => c[1]);
    const zoom = Math.floor(map.camera.current.zoom);
    return {
      name: `${current.scene.name} near ${map.camera.current.center
        .map((v) => v.toFixed(3))
        .reverse()
        .join(", ")}`,
      style: sceneUrl(current.scene, current.format),
      bounds: [
        Math.min(...lons),
        Math.min(...lats),
        Math.max(...lons),
        Math.max(...lats),
      ],
      zooms: [zoom, Math.min(zoom + 3, 16)],
    };
  },
});
document.getElementById("offline-button").addEventListener("click", () => {
  menu.hidePopover();
  offlineAreas.open();
});
// A still of the view, once everything in it has loaded, as a PNG.
document
  .getElementById("picture-button")
  .addEventListener("click", async () => {
    menu.hidePopover();
    if (!map) return;
    say("Drawing the picture…");
    try {
      const image = await map.snapshot();
      const canvas = new OffscreenCanvas(image.width, image.height);
      canvas.getContext("2d").putImageData(image, 0, 0);
      const url = URL.createObjectURL(
        await canvas.convertToBlob({ type: "image/png" }),
      );
      Object.assign(document.createElement("a"), {
        href: url,
        download: `${current.scene.key}.png`,
      }).click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      say("");
    } catch (e) {
      say(`The picture failed: ${e instanceof Error ? e.message : e}`, {
        error: true,
      });
    }
  });
const about = document.getElementById("about");
document.getElementById("about-button").addEventListener("click", () => {
  menu.hidePopover();
  about.showModal();
});
{
  const commit = __ORBIS_COMMIT__;
  const link = document.createElement("a");
  link.href = `https://github.com/sargunv/orbis/commit/${commit.sha}`;
  link.textContent = `${commit.short}${commit.dirty ? "-dirty" : ""}`;
  document.getElementById("build").append("This build: ", link);
}

// The map's focus ring shows for keyboard users only: the canvas takes
// focus on every pointer press too.
addEventListener(
  "keydown",
  (e) => {
    if (e.key === "Tab") app.classList.add("keyboard");
  },
  true,
);
addEventListener("pointerdown", () => app.classList.remove("keyboard"), true);

// Camera controls.
const compass = document.getElementById("compass");
document.getElementById("zoom-in").addEventListener("click", () => {
  map?.camera.easeTo({
    zoom: map.camera.current.zoom + 1,
    duration: reducedMotion() ? 0 : 300,
  });
});
document.getElementById("zoom-out").addEventListener("click", () => {
  map?.camera.easeTo({
    zoom: map.camera.current.zoom - 1,
    duration: reducedMotion() ? 0 : 300,
  });
});
compass.addEventListener("click", () =>
  map?.camera.easeTo({
    bearing: 0,
    pitch: 0,
    duration: reducedMotion() ? 0 : 400,
  }),
);
const reducedMotion = () =>
  matchMedia("(prefers-reduced-motion: reduce)").matches;
function showCompass() {
  const { bearing, pitch } = map.camera.current;
  compass.style.setProperty("--bearing", `${-bearing}deg`);
  compass.style.setProperty("--pitch", `${pitch * 0.7}deg`);
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
  clearTimeout(hashTimer);
  const url = urlFor(next, camera ?? map.camera.current, {
    keepOptions: mode !== "push" || next.scene === current.scene,
  });
  if (mode === "push") history.pushState(null, "", url);
  else history.replaceState(history.state, "", url);
  const previous = current;
  current = next;
  tone();
  picker.show(next.scene, next.format);
  map.camera.stop();
  map.camera.setConstraints(next.scene.constraints ?? {});
  if (camera) map.camera.jumpTo(camera);
  if (
    sceneUrl(previous.scene, previous.format) ===
      sceneUrl(next.scene, next.format) &&
    !previous.scene.restyle &&
    !next.scene.restyle
  ) {
    // The same style on another surface or under another sky: one edit,
    // and nothing loads again.
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    const projection = !same(
      projectionOf(previous.scene),
      projectionOf(next.scene),
    );
    const sky = !same(previous.scene.sky, next.scene.sky);
    if (projection || sky) {
      options.loading();
      void map
        .edit((tx) => {
          if (projection) tx.setProjection(projectionOf(next.scene));
          if (sky) tx.setSky(next.scene.sky ?? styleSky);
        })
        .then(() => {
          // The other entry's options, from the style as it designs it.
          if (switched === switches)
            void options.attach(map, { ...designed, scene: next.scene });
        });
      // The comparison reopens with the projection, if it can draw it.
      compare.attach(map, next.scene);
    }
    startFlight(switched);
    return;
  }
  compare.attach(undefined, next.scene);
  say(`Loading ${next.scene.name}…`);
  styleLoading = true;
  options.loading();
  app.dataset.map = "loading";
  // The style's `styleload` applies the entry's projection and sky, unless
  // it restyles, which puts them in the style. The style's own camera is a
  // fresh page's (load), never a switch's.
  styleFor(next).then(
    (style) => {
      if (switched === switches) map.setStyle(style);
    },
    (e) => {
      if (switched !== switches) return;
      styleLoading = false;
      app.dataset.map = "error";
      say(e instanceof Error ? e.message : String(e), { error: true });
    },
  );
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
    if (!(await map.camera.flyTo(flight[leg]))) return;
  }
}

// Back and forward move between entries as picking them did.
addEventListener("popstate", () => {
  let next;
  try {
    next = sceneFor(params());
  } catch {
    return;
  }
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
      style: await styleFor(next),
      camera,
      constraints: next.scene.constraints,
      // The demo shows what Orbis can do beyond the spec too.
      previews: "all",
      backend: wantedBackend(params()),
    });
  } catch (e) {
    options.drawing(null);
    app.dataset.map = "error";
    say(`The map couldn't start: ${e instanceof Error ? e.message : e}`, {
      error: true,
    });
    console.error(e);
    return;
  }
  map = created;
  options.drawing(map.backend);
  showGaps([]);
  map.on("gaps", ({ gaps }) => showGaps(gaps));
  expose(map);
  map.on("error", ({ message }) => {
    console.error(message);
    if (
      app.dataset.map === "loading" ||
      toast.textContent.startsWith("Loading")
    )
      say(message, { error: true });
  });
  // Each style, the first and those switched to: the entry's projection and
  // sky (unless it restyles, which put them in the style), then the options,
  // attribution and comparison read it.
  map.on("styleload", async () => {
    const { scene } = current;
    styleLoading = false;
    const loaded = await map.getStyle();
    styleSky = loaded.sky ?? null;
    if (!scene.restyle && (scene.projection || scene.sky)) {
      await map.edit((tx) => {
        if (scene.projection) tx.setProjection(scene.projection);
        if (scene.sky) tx.setSky(scene.sky);
      });
    }
    // The options read the style as the entry designs it, then apply the
    // URL's: the terrain the switch turns on is the entry's own, or the
    // style's as it loaded. A style the demo built is designed as built,
    // before the options' edits it loaded with.
    const own = design?.scene === scene ? design.style : undefined;
    designed = {
      designed: own ?? {
        ...loaded,
        ...(scene.projection && { projection: scene.projection }),
        ...(scene.sky && { sky: scene.sky }),
      },
      terrain:
        scene.terrain === true
          ? ((own ?? loaded).terrain ?? null)
          : (scene.terrain ?? null),
    };
    await options.attach(map, { ...designed, scene });
    attributionControl.update(loaded);
    compare.attach(map, scene);
  });
  map.on("load", () => {
    if (!toast.classList.contains("error")) say("");
  });
  map.on("warning", ({ message }) => console.warn(message));
  map.on("render", (ack) => {
    app.dataset.map = styleLoading ? "loading" : ack.idle ? "idle" : "drawing";
    hudControl.frame(ack);
    showCompass();
  });
  map.on("moveend", writeHashSoon);
  // How long a device loss took to recover: to the first frame on the new
  // device, and to the next complete one.
  let lostAt;
  map.on("restored", ({ recoveredMs }) => {
    const idle = map.on("idle", () => {
      idle();
      const recovered =
        `first frame ${Math.round(recoveredMs)} ms` +
        (lostAt === undefined
          ? ""
          : `, complete ${Math.round(performance.now() - lostAt)} ms`);
      hudControl.recovered(recovered);
      console.info(`orbis: recovered from device loss: ${recovered}`);
      app.dataset.recovered = "true";
    });
  });
  const query = params();
  if (query.has("lose")) {
    const loaded = map.on("load", () => {
      loaded();
      setTimeout(
        () => {
          lostAt = performance.now();
          map.loseDevice();
        },
        Number(query.get("lose")) || 0,
      );
    });
  }
  hoverNames(map);
  // From the style's own camera, which it adopts as it loads.
  const switched = switches;
  const loaded = map.on("styleload", () => {
    loaded();
    startFlight(switched);
  });
}

/** While the debug overlay shows, it names the feature under the pointer. */
function hoverNames(target) {
  let pointer;
  let busy = false;
  const hover = async () => {
    if (busy) return;
    busy = true;
    try {
      while (pointer !== undefined) {
        const point = pointer;
        pointer = undefined;
        if (!hudControl.visible) continue;
        const hit = point && (await target.hitTest(point));
        hudControl.hover(
          hit ? `${hit.layer}${hit.id !== null ? ` #${hit.id}` : ""}` : "",
        );
      }
    } finally {
      busy = false;
    }
  };
  target.canvas.addEventListener("pointermove", (e) => {
    if (e.buttons || e.pointerType === "touch") return;
    pointer = [e.offsetX, e.offsetY];
    void hover();
  });
  target.canvas.addEventListener("pointerleave", () => {
    pointer = null;
    void hover();
  });
}

// Hooks for the headless checks and the console.
window.app = {
  get map() {
    return map;
  },
  compare,
  options,
  get scene() {
    return current?.scene.key;
  },
  /** Switches to `style`, a catalogue key or a style URL, as picking an entry does. */
  show(style) {
    go(sceneFor(new URLSearchParams({ style })));
  },
};

// Start.
const start = params();
setDebug(start.has("debug"));
compare.want(parsePair(start.get("compare")));
{
  const camera = hashCamera();
  let first;
  try {
    first = sceneFor(start);
  } catch (e) {
    say(String(e.message), { error: true });
    throw e;
  }
  history.replaceState(
    history.state,
    "",
    urlFor(first, camera, { keepOptions: true }),
  );
  await load(first, {
    camera:
      camera ??
      first.scene.camera ??
      (first.scene.basemap ? LIVE_CAMERA : undefined),
  });
}
