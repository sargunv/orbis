import { opening } from "./app/options.js";
import { quakeStyle } from "./app/quake.js";
import alpsFlight from "./data/alps-flight.json";
import alps from "./data/alps.json?url";
import arctic from "./data/crs/arctic.json?url";
import globe from "./data/globe.json";
import osmCarto from "./data/osm-carto.json?url";
import catalogue from "./data/scenes.json";
import mlt from "./data/tokyo-mlt.json?url";
// The demo's styles: the catalogue the map app's picker shows (the
// basemaps by provider, and a few entries for features: the globe with
// terrain, a projection, a CRS), and `demoStyle` for the bench and
// measurement pages.
import mvt from "./data/tokyo-mvt.json?url";

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
 * place of the style's then too. `restyle` turns the fetched style into the
 * entry's (sources, layers, terrain, light), so a live style keeps tracking
 * its upstream; the map then loads that style, with the projection and sky
 * in it, whole (`sceneStyle`). `light` offers the map options' choice
 * between a map- and a viewport-anchored light, and `terrain` their
 * terrain switch, with the terrain it turns on, off by default, or `true`
 * for the style's own, on by default (app/options.js, which offers the
 * rest by what the style draws); `off` names those of its switches that
 * start off though the style designs them (every Orbis-only one does).
 * `orbisOnly` marks an entry that exists to show an Orbis extension, a
 * projection or CRS beyond the spec, which it draws by default; the
 * picker tags it. `unsupported` says, by engine, why that engine can't
 * show the entry in the comparison (app/compare.js). `quake` makes the
 * entry's host-fed DEM source ripple from `quake.epicentre` (app/quake.js).
 *
 * @typedef {{ key: string, name: string, detail?: string, url: string, provider: string,
 *   swatch: [string, string, string], dark?: boolean, orbisOnly?: boolean, basemap?: boolean, camera?: object,
 *   constraints?: object, flight?: object[], formats?: Record<string, string>,
 *   light?: boolean, terrain?: object | true, off?: string[], projection?: object, sky?: object,
 *   quake?: { epicentre: [number, number] },
 *   restyle?: (style: object) => object,
 *   unsupported?: { gljs?: string, native?: string } }} Scene
 */

/**
 * A blue sky around the globe, a subtle rim at low zoom that thins out as
 * it hands over to Mercator: what `?globe` puts on any style for the
 * benches and measurements.
 */
const GLOBE_SKY = {
  "sky-color": "#4f8fd9",
  "horizon-color": "#d6e8fb",
  "fog-color": "#e4eef8",
  "sky-horizon-blend": 0.7,
  "horizon-fog-blend": 0.6,
  "fog-ground-blend": 0.7,
  "atmosphere-blend": [
    "interpolate",
    ["linear"],
    ["zoom"],
    0,
    0.4,
    2,
    0.32,
    4,
    0.18,
    8,
    0,
  ],
};

/** Whether a filter is in v8's legacy syntax rather than an expression. */
function isLegacyFilter(filter) {
  if (!Array.isArray(filter)) return false;
  const [op, ...args] = filter;
  if (["all", "any", "none"].includes(op)) return args.every(isLegacyFilter);
  return (
    ["==", "!=", "<", ">", "<=", ">=", "in", "!in", "has", "!has"].includes(
      op,
    ) && typeof args[0] === "string"
  );
}

/**
 * Protomaps' layers and tiles as the Globe draws them (data/globe.json,
 * which orb-demo applies too): its own palette set over each layer's paint
 * by id, keeping Protomaps' widths, filters and layout, so the road
 * hierarchy and label placement follow upstream; the sea's depth from AWS
 * Terrain Tiles (ETOPO1 and GEBCO offshore), drawn two ways: below zoom 6
 * over the background, the ocean once the water fill leaves out Protomaps'
 * ocean polygons, and under the land, with the seafloor's relief, so
 * neither shows on land however low it lies; from zoom 6, where Protomaps'
 * landuse fades in, over Protomaps' ocean polygons drawn over the landuse,
 * so that parks and military areas reaching out to sea stay under it, with
 * Mapterhorn's land below sea level and the Caspian Sea put back over it;
 * Mapterhorn's DEM for the entry's terrain (which keeps the sea at sea
 * level) and a hillshade over the land and lakes, under the roads and
 * labels; the buildings extruded under the labels, lit by a sun 60° from
 * overhead. One preview is designed in, for the map options to switch on,
 * off by default (app/options.js) so that the Globe opens as GL JS and
 * MapLibre Native could draw it: the land's hillshade multiplied into the
 * fills beneath it (`dev.sargunv.orbis:hillshade-layer-blend`), in colours
 * made for it, which its metadata holds the unblended ones of. Each
 * addition is placed by the layers it finds, so a style with other ids
 * still draws.
 */
function globeStyle(style) {
  const layers = style.layers.map((layer) => {
    const paint = globe.paint[layer.id];
    const recoloured = paint
      ? { ...layer, paint: { ...layer.paint, ...paint } }
      : layer;
    if (layer.id !== "water") return recoloured;
    // In the filter's own syntax: Protomaps' is legacy.
    const legacy = !layer.filter || isLegacyFilter(layer.filter);
    const ocean = ["!=", legacy ? "kind" : ["get", "kind"], "ocean"];
    return {
      ...recoloured,
      filter: layer.filter ? ["all", layer.filter, ocean] : ocean,
    };
  });
  const labels = () => {
    const i = layers.findIndex((l) => l.type === "symbol");
    return i < 0 ? layers.length : i;
  };
  const earth = layers.findIndex((l) => l.id === "earth");
  layers.splice(earth < 0 ? 1 : earth, 0, ...globe.sea);
  // The ocean polygons, from z6 over the landuse, and the sources the
  // Globe derives from the water layer's: those with no `type` of their own.
  const water = layers.findIndex((l) => l.id === "water");
  const vector = layers[water]?.source;
  const sources = { ...style.sources };
  for (const [id, source] of Object.entries(globe.sources)) {
    if (source.type) sources[id] = source;
    else if (vector) sources[id] = { ...style.sources[vector], ...source };
  }
  if (vector) {
    const ocean = globe.ocean.map((l) =>
      l.source ? l : { ...l, source: vector },
    );
    layers.splice(water, 0, ...ocean);
  }
  const lakes = layers.findIndex((l) => l.id === "water");
  layers.splice(lakes < 0 ? labels() : lakes + 1, 0, globe.hillshade);
  const at = layers.findIndex(
    (l) => l["source-layer"] === "buildings" && l.type === "fill",
  );
  const buildings = layers[at];
  if (buildings) {
    // Protomaps carries `height` on only some footprints (about half in
    // Tokyo and Manhattan, a tenth in central Paris) and no level counts:
    // the rest stand at OpenMapTiles' default `render_height`, 5 m, as
    // Liberty's do where OpenStreetMap has neither.
    layers.splice(labels(), 0, {
      ...globe.buildings,
      source: buildings.source,
    });
  }
  return {
    ...style,
    sources,
    light: globe.light,
    layers,
  };
}

/** What a catalogue entry's `restyle` names. */
const RESTYLES = { globe: globeStyle, quake: quakeStyle };
/** The catalogue's local files, as the build emits them. */
const LOCAL = {
  "alps.json": alps,
  "alps-flight.json": alpsFlight,
  "crs/arctic.json": arctic,
  "osm-carto.json": osmCarto,
};

/**
 * The picker's catalogue, from data/scenes.json, which orb-demo shows
 * too: local URLs and flights resolved, restyles named by function.
 *
 * @type {{ name: string, scenes: Scene[] }[]}
 */
export const GROUPS = catalogue.groups.map((group) => ({
  name: group.name,
  scenes: group.scenes.map((scene) => ({
    ...scene,
    url: LOCAL[scene.url] ?? scene.url,
    ...(scene.flight && { flight: LOCAL[scene.flight].legs }),
    ...(scene.restyle && { restyle: RESTYLES[scene.restyle] }),
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
 * flight, the stutter bench and orb-bench measure terrain on.
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
      !s.projection &&
      !s.sky &&
      !s.restyle &&
      new URL(s.url, location.href).href === url,
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

/**
 * `style` as `scene` draws it: restyled, with its projection and sky, and
 * with the entry's terrain when `terrain` asks for it.
 */
function withScene(scene, style, { terrain = false } = {}) {
  const restyled = scene.restyle ? scene.restyle(style) : style;
  return {
    ...restyled,
    ...(scene.projection && { projection: scene.projection }),
    ...(scene.sky && { sky: scene.sky }),
    ...(terrain &&
      typeof scene.terrain === "object" && { terrain: scene.terrain }),
  };
}

/**
 * A scene's style: its URL, or for a scene that restyles, the style
 * fetched and restyled as the entry designs it, which the map options turn
 * into what the map loads (app/options.js's `opening`), so that the map
 * builds its tiles once rather than again for an edit.
 */
export async function sceneStyle(scene, format) {
  const url = sceneUrl(scene, format);
  if (!scene.restyle) return url;
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`The style failed to load: HTTP ${response.status}`);
  return withScene(scene, await response.json());
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
 * As `demoStyle`, with the scene's projection, sky, restyle and terrain
 * (on, whatever the app's default) put in the style itself, fetched, so
 * that every engine loads them with it, and a restyle's map options as the
 * app opens them: its defaults, which draw nothing GL JS and Native can't,
 * or those `params` names (`&blend=1`); `globe` in `params` puts
 * Liberty-on-the-globe's projection and sky (GLOBE_SKY) in any style.
 * Otherwise the style's URL.
 */
export async function demoStyleWithScene(params, fallback = "live") {
  const url = demoStyle(params, fallback);
  const scene = params.has("globe")
    ? { projection: { type: "globe" }, sky: GLOBE_SKY }
    : params.get("style")
      ? sceneFor(params).scene
      : {};
  if (!scene.projection && !scene.sky && !scene.restyle) return url;
  const style = withScene(scene, await (await fetch(url)).json(), {
    terrain: true,
  });
  if (!scene.restyle) return style;
  const query = new URLSearchParams(params);
  if (!query.has("terrain")) query.set("terrain", "1");
  return opening(style, { scene, terrain: style.terrain ?? null }, query);
}
