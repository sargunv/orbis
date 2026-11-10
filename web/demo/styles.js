import alpsFlight from "./data/alps-flight.json";
import alps from "./data/alps.json?url";
import arctic from "./data/crs/arctic.json?url";
import osmCarto from "./data/osm-carto.json?url";
import mlt from "./data/tokyo-mlt.json?url";
// The demo's styles: the catalogue the map app's picker shows (the
// basemaps by provider, and a few entries for features: the globe with
// terrain, a projection, a CRS), and `demoStyle` for the bench and
// measurement pages.
import mvt from "./data/tokyo-mvt.json?url";

/** OpenFreeMap's live Liberty style over the planet: the benches' and measurements' live style. */
export const LIVE = "https://tiles.openfreemap.org/styles/liberty";
/** Where a fresh page of a basemap opens without a camera in its hash: central Tokyo, around Tokyo Station. */
export const LIVE_CAMERA = { center: [139.767, 35.681], zoom: 13 };

// The demo key for api.protomaps.com that maplibre-compose's demo app
// publishes; it allows any origin.
const PROTOMAPS_KEY = "73c45a97eddd43fb";
const protomaps = (name) =>
  `https://api.protomaps.com/styles/v5/${name}/en.json?key=${PROTOMAPS_KEY}`;
const versatiles = (name) =>
  `https://tiles.versatiles.org/assets/styles/${name}/style.json`;

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
 * start off though the style designs them. `unsupported` says, by
 * engine, why that engine can't show the entry in the comparison
 * (app/compare.js).
 *
 * @typedef {{ key: string, name: string, detail?: string, url: string, provider: string,
 *   swatch: [string, string, string], dark?: boolean, basemap?: boolean, camera?: object,
 *   constraints?: object, flight?: object[], formats?: Record<string, string>,
 *   light?: boolean, terrain?: object | true, off?: string[], projection?: object, sky?: object,
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

/** Mapterhorn's global elevation: Terrarium-encoded WebP (data/README.md). */
const MAPTERHORN = {
  type: "raster-dem",
  tiles: ["https://tiles.mapterhorn.com/{z}/{x}/{y}.webp"],
  encoding: "terrarium",
  tileSize: 512,
  maxzoom: 17,
};

/**
 * The Globe entry's terrain, off until its switch turns it on: draped lines
 * still jump in width at zoom steps (#151).
 */
const PROTOMAPS_TERRAIN = { source: "terrain", exaggeration: 1.3 };

/**
 * The Globe entry's sky, from Protomaps Light's palette: a pale cyan after
 * its water (#80deea) fading to a horizon and fog of its land (#e2dfda),
 * so far terrain dissolves into the map's own ground, and a full
 * atmosphere at low zoom that thins out as it hands over to Mercator: in
 * the entry's linear light it tints the sunlit face and glows at the limb
 * (GL JS's model washes the face out at this blend). Behind it,
 * style-spec#1889's night: a deep blue-black backdrop with stars that fade
 * out by zoom 6, by when the sky covers them anyway.
 */
const PROTOMAPS_SKY = {
  "sky-color": "#9fd3e0",
  "horizon-color": "#eef1ef",
  "fog-color": "#e6e4df",
  "sky-horizon-blend": 0.6,
  "horizon-fog-blend": 0.6,
  "fog-ground-blend": 0.6,
  "atmosphere-blend": [
    "interpolate",
    ["linear"],
    ["zoom"],
    0,
    1,
    2,
    0.9,
    4,
    0.5,
    8,
    0,
  ],
  "backdrop-color": "#060a17",
  "star-opacity": ["interpolate", ["linear"], ["zoom"], 2, 1, 6, 0],
};

/**
 * Protomaps Light (v5) with what it lacks for a 3D map: Mapterhorn's DEM
 * for PROTOMAPS_TERRAIN, a hillshade from a DEM source of its own over the
 * land and water fills and under the roads and labels, a viewport-anchored
 * light (the spec's default, which reads best on the globe in both engines),
 * the buildings extruded under the labels with their footprints rounded
 * (with shadows and ambient occlusion), and colours mixed in linear
 * light (`dev.sargunv.orbis:color-space`). Each
 * addition is placed by the layers it finds, so a style with other ids
 * still draws.
 */
function protomaps3d(style) {
  const layers = [...style.layers];
  const labels = () => {
    const i = layers.findIndex((l) => l.type === "symbol");
    return i < 0 ? layers.length : i;
  };
  // Over the water's fill, before the waterways, piers and roads drawn after it.
  const water = layers.findIndex((l) => l.id === "water");
  layers.splice(water < 0 ? labels() : water + 1, 0, {
    id: "hillshade",
    type: "hillshade",
    source: "hillshade",
    paint: {
      "hillshade-illumination-anchor": "viewport",
      "hillshade-exaggeration": [
        "interpolate",
        ["linear"],
        ["zoom"],
        3,
        0.6,
        12,
        0.35,
      ],
      "hillshade-shadow-color": "#6f6a63",
      "hillshade-highlight-color": "#fbfaf7",
      "hillshade-accent-color": "#8d877e",
    },
  });
  const at = layers.findIndex(
    (l) => l["source-layer"] === "buildings" && l.type === "fill",
  );
  const buildings = layers[at];
  if (buildings) {
    // The flat footprints round off as the extruded ones do
    // (style-spec#1047's preview), so they match as the buildings rise.
    layers[at] = {
      ...buildings,
      layout: { ...buildings.layout, "fill-rounded-corner-distance": 2 },
    };
    // Protomaps carries `height` on only some footprints (about half in
    // Tokyo and Manhattan, a tenth in central Paris) and no level counts:
    // the rest stand at OpenMapTiles' default `render_height`, 5 m, as
    // Liberty's do where OpenStreetMap has neither.
    const height = ["coalesce", ["get", "height"], 5];
    layers.splice(labels(), 0, {
      id: "buildings_3d",
      type: "fill-extrusion",
      source: buildings.source,
      "source-layer": "buildings",
      minzoom: 14,
      filter: ["==", ["geometry-type"], "Polygon"],
      // Softly rounded corners (spec 26.2.0; clamped to a fifth of each edge).
      layout: { "fill-extrusion-rounded-corner-distance": 2 },
      paint: {
        // Protomaps Light's buildings fill (#cccccc at 0.5) over its land (#e2dfda).
        "fill-extrusion-color": "#d7d6d3",
        // Rising out of the ground as the zoom reaches 15.
        "fill-extrusion-height": [
          "interpolate",
          ["linear"],
          ["zoom"],
          14,
          0,
          15,
          height,
        ],
        "fill-extrusion-base": [
          "interpolate",
          ["linear"],
          ["zoom"],
          14,
          0,
          15,
          ["min", ["coalesce", ["get", "min_height"], 0], height],
        ],
        "fill-extrusion-opacity": 0.9,
        // Grounded by ambient occlusion (style-spec#1796's preview): walls
        // darken toward their feet and the ground along them.
        "fill-extrusion-ambient-occlusion-wall-intensity": 0.6,
        "fill-extrusion-ambient-occlusion-ground-intensity": 0.5,
      },
    });
  }
  return {
    ...style,
    sources: {
      ...style.sources,
      terrain: {
        ...MAPTERHORN,
        attribution:
          '<a href="https://mapterhorn.com/attribution">© Mapterhorn</a>',
      },
      hillshade: MAPTERHORN,
    },
    // The buildings cast shadows on the ground from it (style-spec#1797's
    // preview), turning with the view or fixed to the ground as the
    // entry's light switch anchors it.
    light: {
      anchor: "viewport",
      position: [1.15, 210, 30],
      intensity: 0.35,
      "shadow-intensity": 0.5,
    },
    // Colours mix in linear light: the scattering tints the sunlit face
    // rather than washing it out, fog and the hillshade blend as light
    // does, and lines and labels antialias evenly. GL JS and Native ignore
    // metadata.
    metadata: { ...style.metadata, "dev.sargunv.orbis:color-space": "linear" },
    layers,
  };
}

/** @type {{ name: string, scenes: Scene[] }[]} */
export const GROUPS = [
  {
    name: "Features",
    scenes: [
      {
        key: "globe",
        name: "Globe",
        detail:
          "Protomaps Light on the globe in 3D: hillshade, buildings, a night sky, and Mapterhorn terrain",
        url: protomaps("light"),
        projection: { type: "globe" },
        sky: PROTOMAPS_SKY,
        restyle: protomaps3d,
        // Shadows and ambient occlusion start off: zooming fast from space
        // into a city stutters as the buildings arrive with either on.
        terrain: PROTOMAPS_TERRAIN,
        off: ["shadows", "ao"],
        camera: { center: [10, 30], zoom: 1.5 },
        constraints: { minZoom: 0 },
        flight: alpsFlight.legs,
        light: true,
        dark: true,
        swatch: ["#e2dfda", "#80deea", "#060a17"],
      },
      {
        key: "equal-earth",
        name: "Equal Earth",
        detail:
          "The whole world equal-area, handing over to Web Mercator by zoom 7",
        url: LIVE,
        projection: { type: "equal-earth" },
        camera: { center: [0, 10], zoom: 0.5 },
        swatch: ["#f8f4f0", "#a0c8f0", "#e9ac77"],
        unsupported: {
          gljs: "GL JS has no Equal Earth projection",
          native: "MapLibre Native has no Equal Earth projection",
        },
      },
      {
        key: "arctic",
        name: "Arctic",
        detail: "A polar stereographic CRS, EPSG:3413",
        url: arctic,
        dark: true,
        swatch: ["#24476b", "#e9eef2", "#9aa6ad"],
        unsupported: {
          gljs: "GL JS can't draw tile matrix sets",
          native: "MapLibre Native can't draw tile matrix sets",
        },
      },
    ],
  },
  {
    name: "OpenFreeMap",
    scenes: [
      {
        key: "liberty",
        name: "Liberty",
        url: LIVE,
        basemap: true,
        swatch: ["#f8f4f0", "#a0c8f0", "#e9ac77"],
      },
      {
        key: "bright",
        name: "Bright",
        url: "https://tiles.openfreemap.org/styles/bright",
        basemap: true,
        swatch: ["#f8f4f0", "#a0c8f0", "#f2934a"],
      },
      {
        key: "positron",
        name: "Positron",
        url: "https://tiles.openfreemap.org/styles/positron",
        basemap: true,
        swatch: ["#f2f3f0", "#c2c8ca", "#ffffff"],
      },
    ],
  },
  {
    name: "Protomaps",
    scenes: [
      {
        key: "protomaps-light",
        name: "Light",
        url: protomaps("light"),
        basemap: true,
        swatch: ["#e2dfda", "#80deea", "#ffffff"],
      },
      {
        key: "protomaps-dark",
        name: "Dark",
        url: protomaps("dark"),
        basemap: true,
        dark: true,
        swatch: ["#34373d", "#22252b", "#4a4f57"],
      },
      {
        key: "protomaps-white",
        name: "White",
        url: protomaps("white"),
        basemap: true,
        swatch: ["#ffffff", "#dcdcdc", "#c4c4c4"],
      },
      {
        key: "protomaps-grayscale",
        name: "Grayscale",
        url: protomaps("grayscale"),
        basemap: true,
        swatch: ["#e0e0e0", "#a3a3a3", "#fafafa"],
      },
      {
        key: "protomaps-black",
        name: "Black",
        url: protomaps("black"),
        basemap: true,
        dark: true,
        swatch: ["#2b2b2b", "#0f0f0f", "#444444"],
      },
    ],
  },
  {
    name: "VersaTiles",
    scenes: [
      {
        key: "versatiles-colorful",
        name: "Colorful",
        url: versatiles("colorful"),
        basemap: true,
        swatch: ["#f9f4ee", "#beddf3", "#ffcc88"],
      },
      {
        key: "versatiles-eclipse",
        name: "Eclipse",
        url: versatiles("colorful-dark"),
        basemap: true,
        dark: true,
        swatch: ["#171b20", "#0b2033", "#4a4436"],
      },
      {
        key: "versatiles-graybeard",
        name: "Graybeard",
        url: versatiles("gray"),
        basemap: true,
        swatch: ["#f2f2f2", "#cfcfcf", "#ffffff"],
      },
    ],
  },
  {
    name: "OpenStreetMap",
    scenes: [
      {
        key: "osm",
        name: "OpenStreetMap Carto",
        detail: "Raster tiles from openstreetmap.org",
        url: osmCarto,
        basemap: true,
        swatch: ["#f2efe9", "#aad3df", "#e892a2"],
      },
      {
        key: "americana",
        name: "Americana",
        detail: "Route shields come from its own page, not shown",
        url: "https://americanamap.org/style.json",
        basemap: true,
        swatch: ["#f3efe6", "#a6cfe6", "#c8553d"],
      },
    ],
  },
];
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
 * What the map loads for a scene: its URL, or for a scene that restyles,
 * the style fetched and restyled, so that the map builds its tiles once
 * rather than again for an edit; `terrain` puts in the entry's terrain.
 */
export async function sceneStyle(scene, format, { terrain = false } = {}) {
  const url = sceneUrl(scene, format);
  if (!scene.restyle) return url;
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`The style failed to load: HTTP ${response.status}`);
  return withScene(scene, await response.json(), { terrain });
}

/**
 * The style a bench or measurement page shows: as the map app picks it
 * from `params`, or without a style, `fallback`: "live" (Liberty) or
 * "tokyo" (the committed extract, for benchmarks and offline work).
 */
export function demoStyle(params, fallback = "live") {
  if (!params.get("style") && !params.get("format") && fallback === "tokyo")
    return sceneUrl(SCENES.tokyo, "mvt");
  const { scene, format } = sceneFor(params);
  return sceneUrl(scene, format);
}

/**
 * As `demoStyle`, with the scene's projection, sky, restyle and terrain
 * (on, whatever the app's default) put in the style itself, fetched, so
 * that every engine loads them with it; `globe`
 * in `params` puts Liberty-on-the-globe's projection and sky (GLOBE_SKY)
 * in any style. Otherwise the style's URL.
 */
export async function demoStyleWithScene(params, fallback = "live") {
  const url = demoStyle(params, fallback);
  const scene = params.has("globe")
    ? { projection: { type: "globe" }, sky: GLOBE_SKY }
    : params.get("style")
      ? sceneFor(params).scene
      : {};
  if (!scene.projection && !scene.sky && !scene.restyle) return url;
  return withScene(scene, await (await fetch(url)).json(), { terrain: true });
}
