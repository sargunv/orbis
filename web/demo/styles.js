import { opening } from "./app/options.js";
import alpsFlight from "./data/alps-flight.json";
import alps from "./data/alps.json?url";
import arctic from "./data/crs/arctic.json?url";
import earthquake from "./data/earthquake.json?url";
import GLOBE_SKY from "./data/globe-sky.json";
import globe from "./data/globe.json?url";
import neon from "./data/neon.json?url";
import osmCarto from "./data/osm-carto.json?url";
import papercraft from "./data/papercraft.json?url";
import catalogue from "./data/scenes.json";
import seaChart from "./data/sea-chart.json?url";
import shimmer from "./data/shimmer.json?url";
import mlt from "./data/tokyo-mlt.json?url";
// The demo's styles: the catalogue the map app's picker shows (the
// basemaps by provider, and a few entries for features: the globe with
// terrain, a projection, a CRS), and `demoStyle` for the bench and
// measurement pages.
import mvt from "./data/tokyo-mvt.json?url";
import watercolor from "./data/watercolor.json?url";

/** OpenFreeMap's live Liberty style over the planet: the benches' and measurements' live style. */
export const LIVE = "https://tiles.openfreemap.org/styles/liberty";
/** Where a fresh page of a basemap opens without a camera in its hash: central Tokyo, around Tokyo Station. */
export const LIVE_CAMERA = catalogue.basemapCamera;

/** The union of the zoom-`z` tiles that touch `bounds`: where an extract cut that way has data at every zoom. */
function tileBounds([west, south, east, north], z) {
  const n = 2 ** z;
  const x = (lon) => Math.floor(((lon + 180) / 360) * n);
  const y = (lat) =>
    Math.floor(
      ((1 - Math.asinh(Math.tan((lat * Math.PI) / 180)) / Math.PI) / 2) * n,
    );
  const lon = (x) => (x / n) * 360 - 180;
  const lat = (y) =>
    (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n))) * 180) / Math.PI;
  return [lon(x(west)), lat(y(south) + 1), lon(x(east) + 1), lat(y(north))];
}

/** The committed Tokyo extract's bounds (data/README.md), cut at zoom 14. */
const TOKYO_BOUNDS = tileBounds([139.725, 35.677, 139.755, 35.697], 14);

/**
 * A picker entry. Switching entries keeps the camera; `camera` is only
 * where a fresh page without a camera in its hash opens, by default
 * LIVE_CAMERA for a `basemap` and the style's own for the rest. `flight` is
 * a looping flight that `&flight` flies. `swatch` is [land, water, line]
 * for the picker's chip; `dark` styles turn the app's chrome dark.
 * `constraints` keep the camera where a bounded dataset has data, or a
 * view in range (the globe no smaller than at zoom 0), on both maps of a
 * comparison, and clamp it on a switch. `projection` is a v8 projection the
 * map switches to once the style loads, and `sky` a v8 sky it puts in
 * place of the style's then too. `own` marks the demo's own style, a file
 * in data/ whose URLs are all absolute (`fetchStyle` throws on one that
 * isn't): the page fetches it and the map loads it with the projection,
 * the sky and the map options in it, whole (`sceneStyle`). `light` offers the map options' choice
 * between a map- and a viewport-anchored light, and `terrain` their
 * switch for the style's own terrain, on by default (app/options.js,
 * which offers the rest by what the style draws); `off` names those of
 * its switches that
 * start off though the style designs them (every Orbis-only one does).
 * `orbisOnly` marks an entry that exists to show an Orbis extension, a
 * projection or CRS beyond the spec, which it draws by default; the
 * picker tags it. `unsupported` says, by engine, why that engine can't
 * show the entry in the comparison (app/compare.js). `quake` makes the
 * entry's host-fed DEM source ripple from `quake.epicentre` in the wave
 * `quake.wave` describes (app/quake.js, and orb-demo's quake.rs).
 * `marker.view` is where a click on the landmark the style marks flies
 * (app/marker.js).
 *
 * @typedef {{ key: string, name: string, detail?: string, url: string, provider: string,
 *   swatch: [string, string, string], dark?: boolean, orbisOnly?: boolean, basemap?: boolean, camera?: object,
 *   constraints?: object, flight?: object[], formats?: Record<string, string>,
 *   light?: boolean, terrain?: boolean, off?: string[], projection?: object, sky?: object,
 *   quake?: { epicentre: [number, number], wave: Record<string, number> }, marker?: { view: object }, own?: boolean,
 *   unsupported?: { gljs?: string, native?: string } }} Scene
 */

/** The catalogue's local files, as the build emits them. */
const LOCAL = {
  "alps.json": alps,
  "alps-flight.json": alpsFlight,
  "crs/arctic.json": arctic,
  "earthquake.json": earthquake,
  "globe.json": globe,
  "neon.json": neon,
  "osm-carto.json": osmCarto,
  "papercraft.json": papercraft,
  "sea-chart.json": seaChart,
  "shimmer.json": shimmer,
  "watercolor.json": watercolor,
};

/** Whether a catalogue URL names its scheme, rather than a file in data/. */
const isAbsolute = (url) => /^[a-z][a-z\d+.-]*:/i.test(url);

/**
 * A catalogue URL as the page fetches it: a local file as the build emits
 * it, which it emits only where LOCAL imports it, so one missing from
 * LOCAL throws here rather than going unfetched on the deployed site.
 */
function local(url) {
  if (isAbsolute(url)) return url;
  const emitted = LOCAL[url];
  if (!emitted)
    throw new Error(`data/scenes.json: add ${url} to LOCAL in styles.js`);
  return emitted;
}

/**
 * The picker's catalogue, from data/scenes.json, which orb-demo shows
 * too: local URLs and flights resolved, the local styles marked `own`.
 *
 * @type {{ name: string, scenes: Scene[] }[]}
 */
export const GROUPS = catalogue.groups.map((group) => ({
  name: group.name,
  scenes: group.scenes.map((scene) => ({
    ...scene,
    url: local(scene.url),
    own: !isAbsolute(scene.url),
    ...(scene.flight && { flight: LOCAL[scene.flight].legs }),
  })),
}));
for (const group of GROUPS)
  for (const scene of group.scenes) scene.provider = group.name;

const SCENES = Object.fromEntries(
  GROUPS.flatMap((g) => g.scenes).map((s) => [s.key, s]),
);
/** Old keys that links may still use: the Terrain entry's became the Globe's. */
const ALIASES = { terrain: "globe" };
/** The committed Tokyo extract, by URL only: the benchmarks', checks' and offline work's style. */
SCENES.tokyo = {
  key: "tokyo",
  name: "Tokyo, offline",
  detail: "Bundled extract, no network",
  url: mvt,
  provider: "Orbis",
  formats: { mvt, mlt },
  constraints: { bounds: TOKYO_BOUNDS, maxPitch: 0 },
  swatch: ["#f8f4f0", "#a0c8f0", "#e9ac77"],
};

/**
 * The Alps terrain style and its flight, by key only: the stable style the
 * flight, the sessions and orb-bench measure terrain on.
 */
SCENES.alps = {
  key: "alps",
  name: "Alps",
  detail: "Terrain over OpenFreeMap tiles, for the benches",
  url: alps,
  provider: "Orbis",
  flight: alpsFlight.legs,
  light: true,
  terrain: true,
  swatch: ["#eef0e4", "#9fb7c9", "#7d8a74"],
};

/** The default scene. */
export const DEFAULT_SCENE = SCENES.globe;

/**
 * The scene `params` name (and, for Tokyo, its format): a catalogue key,
 * `tokyo[-mvt|-mlt]`, a style URL matching an entry, or any other style
 * URL as a scene of its own.
 */
export function sceneFor(params) {
  const style = params.get("style");
  const format = params.get("format");
  if (format && !["mvt", "mlt"].includes(format))
    throw new Error("format must be mvt or mlt");
  if (!style)
    return format ? { scene: SCENES.tokyo, format } : { scene: DEFAULT_SCENE };
  const tokyo = style.match(/^tokyo(?:-(mvt|mlt))?$/);
  if (tokyo)
    return { scene: SCENES.tokyo, format: tokyo[1] ?? format ?? "mvt" };
  if (SCENES[ALIASES[style] ?? style])
    return { scene: SCENES[ALIASES[style] ?? style] };
  const url = new URL(style, location.href).href;
  const known = Object.values(SCENES).find(
    (s) =>
      !s.projection && !s.sky && new URL(s.url, location.href).href === url,
  );
  if (known) return { scene: known };
  return {
    scene: {
      key: style,
      name: new URL(url).hostname,
      detail: url,
      url,
      provider: "Custom style",
      basemap: true,
      swatch: ["#e8e8e8", "#c8c8c8", "#9a9a9a"],
    },
  };
}

/** The style URL a scene draws in `format`. */
export function sceneUrl(scene, format) {
  const url = scene.formats?.[format] ?? scene.url;
  return new URL(url, location.href).href;
}

/** `style` as `scene` draws it, with its projection and sky. */
function withScene(scene, style) {
  return {
    ...style,
    ...(scene.projection && { projection: scene.projection }),
    ...(scene.sky && { sky: scene.sky }),
  };
}

/**
 * A scene's style: its URL, or for the demo's own (`own`), the style
 * fetched with the entry's projection and sky, which the map options turn
 * into what the map loads (app/options.js's `opening`), so that the map
 * builds its tiles once rather than again for an edit.
 */
export async function sceneStyle(scene, format) {
  const url = sceneUrl(scene, format);
  if (!scene.own) return url;
  return withScene(scene, await fetchStyle(url));
}

/**
 * The style at `url`, to load as an object, whose URLs then resolve
 * against the page rather than the file: a style naming one (glyphs,
 * sprite, a source's TileJSON, tiles or data) that the two resolve
 * differently throws rather than loading the wrong thing.
 */
async function fetchStyle(url) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`The style failed to load: HTTP ${response.status}`);
  const style = await response.json();
  const sprites = Array.isArray(style.sprite)
    ? style.sprite.map((s) => s.url)
    : [style.sprite];
  const sources = Object.values(style.sources ?? {}).flatMap((s) => [
    s.url,
    ...(s.tiles ?? []),
    ...(s.urls ?? []),
    s.data,
  ]);
  const file = new URL(url, location.href);
  const relative = [style.glyphs, ...sprites, ...sources].find(
    (u) =>
      typeof u === "string" &&
      new URL(u, file).href !== new URL(u, location.href).href,
  );
  if (relative !== undefined)
    throw new Error(
      `${url} names ${relative}, which resolves against the page here, not the file: make it absolute`,
    );
  return style;
}

/**
 * The style a bench or measurement page shows: as the map app picks it
 * from `params`, or without a style, `fallback`: "live" (Liberty) or
 * "tokyo" (the committed extract, for benchmarks and offline work).
 */
export function demoStyle(params, fallback = "live") {
  if (!params.get("style") && !params.get("format"))
    return fallback === "tokyo" ? sceneUrl(SCENES.tokyo, "mvt") : LIVE;
  const { scene, format } = sceneFor(params);
  return sceneUrl(scene, format);
}

/**
 * As `demoStyle`, with the scene's projection and sky put in the style
 * itself, fetched, so that every engine loads them with it, and the demo's
 * own style with its map options as the app opens them: its defaults, which draw
 * nothing GL JS and Native can't, or those `params` names (`&blend=1`,
 * `&terrain=0`); `globe` in `params` puts
 * Liberty-on-the-globe's projection and sky in any style: a blue sky
 * around the globe, a subtle rim at low zoom that thins out as it hands
 * over to Mercator (data/globe-sky.json, which orb-bench's sessions put
 * on too).
 * Otherwise the style's URL.
 */
export async function demoStyleWithScene(params, fallback = "live") {
  const url = demoStyle(params, fallback);
  const scene = params.has("globe")
    ? { projection: { type: "globe" }, sky: GLOBE_SKY }
    : params.get("style")
      ? sceneFor(params).scene
      : {};
  if (!scene.projection && !scene.sky && !scene.own) return url;
  const style = withScene(scene, await fetchStyle(url));
  if (!scene.own) return style;
  return opening(style, { scene, terrain: style.terrain ?? null }, params);
}
