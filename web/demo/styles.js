// The demo's styles: the catalogue the map app's picker shows, grouped by
// provider, and `demoStyle` for the bench and measurement pages.
import mvt from "./data/tokyo-mvt.json?url";
import mlt from "./data/tokyo-mlt.json?url";

import playground from "./data/playground.json?url";
import scripts from "./data/scripts.json?url";
import arctic from "./data/crs/arctic.json?url";
import greenland from "./data/crs/greenland.json?url";
import nz from "./data/crs/nz.json?url";
import equalEarth from "./data/crs/equal-earth.json?url";
import blueMarble from "./data/crs/blue-marble.json?url";
import alps from "./data/alps.json?url";
import alpsSunlitScene from "./data/alps-sunlit.json";
import osmCarto from "./data/osm-carto.json?url";
import alpsFlight from "./data/alps-flight.json";
import globe from "./data/globe.json?url";
import manhattan from "./data/manhattan.json?url";
import video from "./data/video.json?url";
import landmarksScene from "./data/tokyo-landmarks.json";
import litMorning from "./data/lit/tokyo-morning.json?url";
import litNoon from "./data/lit/tokyo-noon.json?url";
import litDusk from "./data/lit/tokyo-dusk.json?url";
import litNight from "./data/lit/tokyo-night.json?url";
import tower from "./data/models/tower.glb?url";
import torii from "./data/models/torii.glb?url";
import pagoda from "./data/models/pagoda.glb?url";

// The sunlit Alps scene names its style relatively, which the bundle's
// hashed asset names break: point it at the bundled style.
const alpsSunlit = `data:application/json,${encodeURIComponent(JSON.stringify({
  ...alpsSunlitScene, style: new URL(alps, location.href).href,
}))}`;

// The landmarks scene names its models relatively, which hashed asset
// names break too.
const absolute = (url) => new URL(url, location.href).href;
const landmarks = `data:application/json,${encodeURIComponent(JSON.stringify({
  ...landmarksScene, models: { tower: absolute(tower), torii: absolute(torii), pagoda: absolute(pagoda) },
}))}`;

/** OpenFreeMap's live Liberty style over the planet: the demo's default. */
export const LIVE = "https://tiles.openfreemap.org/styles/liberty";
/** Where the live basemaps open: central Tokyo, around Tokyo Station. */
export const LIVE_CAMERA = { center: [139.767, 35.681], zoom: 13 };

// The demo key for api.protomaps.com that maplibre-compose's demo app
// publishes; it allows any origin.
const PROTOMAPS_KEY = "73c45a97eddd43fb";
const protomaps = (name) => `https://api.protomaps.com/styles/v5/${name}/en.json?key=${PROTOMAPS_KEY}`;
const versatiles = (name) => `https://tiles.versatiles.org/assets/styles/${name}/style.json`;

/** The union of the zoom-`z` tiles that touch `bounds`: where an extract cut that way has data at every zoom. */
function tileBounds([west, south, east, north], z) {
  const n = 2 ** z;
  const x = (lon) => Math.floor(((lon + 180) / 360) * n);
  const y = (lat) => Math.floor(((1 - Math.asinh(Math.tan((lat * Math.PI) / 180)) / Math.PI) / 2) * n);
  const lon = (x) => (x / n) * 360 - 180;
  const lat = (y) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n))) * 180) / Math.PI;
  return [lon(x(west)), lat(y(south) + 1), lon(x(east) + 1), lat(y(north))];
}

/** The committed Tokyo extract's bounds (data/README.md), cut at zoom 14. */
const TOKYO_BOUNDS = tileBounds([139.725, 35.677, 139.755, 35.697], 14);

const CITIES = {
  cairo: { center: [31.2357, 30.0444], zoom: 13 },
  delhi: { center: [77.209, 28.6139], zoom: 13 },
  bangkok: { center: [100.5018, 13.7563], zoom: 13 },
};

/**
 * A picker entry. `basemap` entries keep the camera when switched to; the
 * rest open at their own. `swatch` is [land, water, line] for the picker's
 * chip; `dark` styles turn the app's chrome dark. `constraints` keep the
 * camera where a bounded dataset has data. `projection` is a v8 projection
 * the map switches to once the style loads; `calls` are GL JS map calls
 * (`[name, ...args]`) applied then too.
 *
 * @typedef {{ key: string, name: string, detail?: string, url: string, provider: string,
 *   swatch: [string, string, string], dark?: boolean, basemap?: boolean, camera?: object,
 *   constraints?: object, flight?: object[], formats?: Record<string, string>,
 *   projection?: object, calls?: unknown[][] }} Scene
 */

/** The world, for layers that cover everything. */
const WORLD = { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "Polygon",
  coordinates: [[[-180, -85.06], [180, -85.06], [180, 85.06], [-180, 85.06], [-180, -85.06]]] } } };

/**
 * Liberty at dusk, by layer blends (style-spec 26): a blue tint multiplies
 * everything beneath the boundaries; major roads glow, added on top in
 * `plus-lighter` at a layer opacity; water screens a cool sheen.
 */
const DUSK = [
  ["addSource", "dusk", WORLD],
  ["addLayer", { id: "dusk-tint", type: "fill", source: "dusk",
    paint: { "fill-color": "#4d5894", "fill-layer-blend": "multiply" } }, "boundary_3"],
  ["addLayer", { id: "dusk-glow", type: "line", source: "openmaptiles", "source-layer": "transportation",
    filter: ["match", ["get", "class"], ["motorway", "trunk", "primary", "secondary"], true, false],
    layout: { "line-cap": "round", "line-join": "round" },
    paint: { "line-color": "#ff9d4d", "line-blur": ["interpolate", ["linear"], ["zoom"], 8, 2, 16, 10],
      "line-width": ["interpolate", ["exponential", 1.5], ["zoom"], 8, 2, 16, 18],
      "line-layer-blend": "plus-lighter", "line-layer-opacity": 0.55 } }, "boundary_3"],
  ["setPaintProperty", "water", "fill-layer-blend", "screen"],
];

/**
 * GL JS's image source example: a weather radar frame over the northeast
 * of the United States, on Liberty.
 */
const RADAR = [
  ["addSource", "radar", { type: "image", url: "https://maplibre.org/maplibre-gl-js/docs/assets/radar.gif",
    coordinates: [[-80.425, 46.437], [-71.516, 46.437], [-71.516, 37.936], [-80.425, 37.936]] }],
  ["addLayer", { id: "radar", type: "raster", source: "radar",
    paint: { "raster-fade-duration": 0, "raster-opacity": 0.85 } }],
];

/** The landmarks scene's glTF models, at points of a GeoJSON source, under Liberty's labels. */
const point = (model, coordinates) => ({ type: "Feature", properties: { model },
  geometry: { type: "Point", coordinates } });
const LANDMARKS = [
  ["addSource", "landmarks", { type: "geojson", data: { type: "FeatureCollection", features: [
    point("tower", [139.74543, 35.65858]),
    point("torii", [139.7497, 35.6572]),
    point("pagoda", [139.7462, 35.6553]),
  ] } }],
  ["addLayer", { id: "landmarks", type: "model", source: "landmarks",
    layout: { "model-id": ["get", "model"] } }, "road_one_way_arrow"],
];

/** @type {{ name: string, scenes: Scene[] }[]} */
export const GROUPS = [
  {
    name: "Orbis showcases",
    scenes: [
      { key: "globe", name: "Globe", detail: "Liberty on the v8 globe in space, Web Mercator from zoom 12",
        url: globe, projection: { type: "globe" }, camera: { center: [10, 30], zoom: 1.5 }, dark: true,
        swatch: ["#f8f4f0", "#a0c8f0", "#02040b"] },
      { key: "manhattan", name: "Manhattan in 3D", detail: "Low afternoon sun: shadows, and labels hidden behind buildings",
        url: manhattan, camera: { center: [-73.985, 40.752], zoom: 15.6, bearing: 29, pitch: 62 },
        swatch: ["#d8d4cf", "#a0c8f0", "#e9ac77"] },
      { key: "dusk", name: "Liberty at dusk", detail: "Layer blends: a multiplied tint, glowing roads",
        url: LIVE, calls: DUSK, camera: { center: [139.767, 35.681], zoom: 12.5 }, dark: true,
        swatch: ["#4c4f6e", "#6f86b8", "#ff9d4d"] },
      { key: "blue-marble", name: "Blue Marble globe", detail: "NASA imagery in EPSG:4326 tiles, to the poles",
        url: blueMarble, dark: true, swatch: ["#2f4a2a", "#0b2a5c", "#d8d0b8"] },
      { key: "alps", name: "Alps terrain flight", detail: "3D terrain, sky and a looping camera flight", url: alps,
        flight: alpsFlight.legs, swatch: ["#eef0e4", "#9fb7c9", "#7d8a74"] },
      { key: "alps-sunlit", name: "Alps in morning sun", detail: "The same flight, lit by the sun at a time and place", url: alpsSunlit,
        flight: alpsFlight.legs, swatch: ["#c9cdb8", "#8fa9bd", "#6b7660"] },
      { key: "landmarks", name: "Tokyo landmarks", detail: "glTF models among sunlit 3D buildings",
        url: landmarks, calls: LANDMARKS, camera: { center: [139.7471, 35.6566], zoom: 16.2, bearing: -25, pitch: 60 },
        swatch: ["#e9e4dc", "#a0c8f0", "#e8542a"] },
      { key: "arctic", name: "Arctic", detail: "Polar stereographic, EPSG:3413", url: arctic, dark: true,
        swatch: ["#24476b", "#e9eef2", "#9aa6ad"] },
      { key: "equal-earth", name: "Equal Earth", detail: "The whole world equal-area, handing over to Web Mercator by zoom 7",
        url: equalEarth, camera: { center: [0, 10], zoom: 0.5 }, swatch: ["#f8f4f0", "#a0c8f0", "#e9ac77"] },
      { key: "nz", name: "New Zealand", detail: "Transverse Mercator, NZTM2000", url: nz,
        swatch: ["#f8f4f0", "#a0c8f0", "#bfd6a8"] },
      { key: "greenland", name: "Greenland ice flow", detail: "Polar imagery warped onto Web Mercator", url: greenland,
        constraints: { bounds: [-76, 58, -8, 84.5], maxZoom: 8 }, swatch: ["#f4f7fa", "#a0c8f0", "#7b4fb0"] },
      { key: "radar", name: "Weather radar", detail: "An image source: a radar frame over the northeast US",
        url: LIVE, calls: RADAR, camera: { center: [-75.789, 41.874], zoom: 5 },
        swatch: ["#f8f4f0", "#a0c8f0", "#44b85c"] },
      { key: "video", name: "Video on a map", detail: "A video source: drone footage over the coast",
        url: video, constraints: { minZoom: 14 }, dark: true, swatch: ["#3b4a35", "#1f3a52", "#c9b98f"] },
      { key: "cairo", name: "Cairo", detail: "Arabic labels, shaped with HarfRust", url: scripts,
        camera: CITIES.cairo, swatch: ["#f6f3ee", "#a9cbe8", "#d9a066"] },
      { key: "delhi", name: "Delhi", detail: "Devanagari labels", url: scripts,
        camera: CITIES.delhi, swatch: ["#f6f3ee", "#a9cbe8", "#d9a066"] },
      { key: "bangkok", name: "Bangkok", detail: "Thai labels", url: scripts,
        camera: CITIES.bangkok, swatch: ["#f6f3ee", "#a9cbe8", "#d9a066"] },
      { key: "tokyo", name: "Tokyo, offline", detail: "Bundled extract, no network", url: mvt,
        formats: { mvt, mlt }, constraints: { bounds: TOKYO_BOUNDS, maxPitch: 0 },
        swatch: ["#f8f4f0", "#a0c8f0", "#e9ac77"] },
    ],
  },
  {
    name: "Lit Tokyo",
    scenes: [
      ["lit-morning", "Morning", "Liberty lit as one surface by a low eastern sun", litMorning,
        ["#e8dcc8", "#8fb4d8", "#f0a868"]],
      ["lit-noon", "Noon", "The whole map under the midday sun, its buildings' shadows on the ground", litNoon,
        ["#f2efe8", "#9cc4ec", "#f6d27a"]],
      ["lit-dusk", "Dusk", "A low orange sun, glossy water", litDusk, ["#c9a990", "#6a7fb8", "#ff9d4d"]],
      ["lit-night", "Night", "Streets that light themselves under a dark sky", litNight,
        ["#20253a", "#0f1a38", "#f6d27a"], true],
    ].map(([key, name, detail, url, swatch, dark]) => ({ key, name, detail, url, swatch, dark,
      camera: { center: [139.7655, 35.6795], zoom: 15.6, bearing: -20, pitch: 60 } })),
  },
  {
    name: "OpenFreeMap",
    scenes: [
      { key: "liberty", name: "Liberty", url: LIVE, basemap: true, swatch: ["#f8f4f0", "#a0c8f0", "#e9ac77"] },
      { key: "bright", name: "Bright", url: "https://tiles.openfreemap.org/styles/bright", basemap: true,
        swatch: ["#f8f4f0", "#a0c8f0", "#f2934a"] },
      { key: "positron", name: "Positron", url: "https://tiles.openfreemap.org/styles/positron", basemap: true,
        swatch: ["#f2f3f0", "#c2c8ca", "#ffffff"] },
    ],
  },
  {
    name: "Protomaps",
    scenes: [
      { key: "protomaps-light", name: "Light", url: protomaps("light"), basemap: true,
        swatch: ["#e2dfda", "#80deea", "#ffffff"] },
      { key: "protomaps-dark", name: "Dark", url: protomaps("dark"), basemap: true, dark: true,
        swatch: ["#34373d", "#22252b", "#4a4f57"] },
      { key: "protomaps-white", name: "White", url: protomaps("white"), basemap: true,
        swatch: ["#ffffff", "#dcdcdc", "#c4c4c4"] },
      { key: "protomaps-grayscale", name: "Grayscale", url: protomaps("grayscale"), basemap: true,
        swatch: ["#e0e0e0", "#a3a3a3", "#fafafa"] },
      { key: "protomaps-black", name: "Black", url: protomaps("black"), basemap: true, dark: true,
        swatch: ["#2b2b2b", "#0f0f0f", "#444444"] },
    ],
  },
  {
    name: "VersaTiles",
    scenes: [
      { key: "versatiles-colorful", name: "Colorful", url: versatiles("colorful"), basemap: true,
        swatch: ["#f9f4ee", "#beddf3", "#ffcc88"] },
      { key: "versatiles-eclipse", name: "Eclipse", url: versatiles("colorful-dark"), basemap: true, dark: true,
        swatch: ["#171b20", "#0b2033", "#4a4436"] },
      { key: "versatiles-graybeard", name: "Graybeard", url: versatiles("gray"), basemap: true,
        swatch: ["#f2f2f2", "#cfcfcf", "#ffffff"] },
    ],
  },
  {
    name: "OpenStreetMap",
    scenes: [
      { key: "osm", name: "OpenStreetMap Carto", detail: "Raster tiles from openstreetmap.org", url: osmCarto,
        basemap: true, swatch: ["#f2efe9", "#aad3df", "#e892a2"] },
      { key: "americana", name: "Americana", detail: "Route shields come from its own page, not shown",
        url: "https://americanamap.org/style.json", basemap: true,
        camera: { center: [-77.036, 38.897], zoom: 13 }, swatch: ["#f3efe6", "#a6cfe6", "#c8553d"] },
    ],
  },
];
for (const group of GROUPS) for (const scene of group.scenes) scene.provider = group.name;

const SCENES = Object.fromEntries(GROUPS.flatMap((g) => g.scenes).map((s) => [s.key, s]));
/** The feature-state sample the style editor's tests draw; by URL only. */
SCENES.playground = { key: "playground", name: "Editor sample", detail: "A Tokyo district with hover state",
  url: playground, provider: "Orbis", swatch: ["#edf2f5", "#bfd5c7", "#567b99"] };

/** The default scene. */
export const DEFAULT_SCENE = SCENES.liberty;

const near = (camera, center) => camera && Math.hypot(camera.center[0] - center[0], camera.center[1] - center[1]) < 5;

/**
 * The scene `params` name (and, for Tokyo, its format): a catalogue key,
 * `tokyo-mvt|tokyo-mlt`, `scripts` (the city nearest `camera`), a style
 * URL matching an entry, or any other style URL as a scene of its own.
 */
export function sceneFor(params, camera) {
  const style = params.get("style");
  const format = params.get("format");
  if (format && !["mvt", "mlt"].includes(format)) throw new Error("format must be mvt or mlt");
  if (!style) return format ? { scene: SCENES.tokyo, format } : { scene: DEFAULT_SCENE };
  const tokyo = style.match(/^tokyo(?:-(mvt|mlt))?$/);
  if (tokyo) return { scene: SCENES.tokyo, format: tokyo[1] ?? format ?? "mvt" };
  if (style === "scripts") {
    const city = Object.keys(CITIES).find((key) => near(camera, CITIES[key].center)) ?? "cairo";
    return { scene: SCENES[city] };
  }
  if (SCENES[style]) return { scene: SCENES[style] };
  const url = new URL(style, location.href).href;
  const known = Object.values(SCENES).find((s) => new URL(s.url, location.href).href === url);
  if (known) return { scene: known };
  return { scene: { key: style, name: new URL(url).hostname, detail: url, url, provider: "Custom style",
    basemap: true, swatch: ["#e8e8e8", "#c8c8c8", "#9a9a9a"] } };
}

/** The style URL a scene draws in `format`. */
export function sceneUrl(scene, format) {
  const url = scene.formats?.[format] ?? scene.url;
  return new URL(url, location.href).href;
}

/**
 * The style a bench or measurement page shows: as the map app picks it
 * from `params`, or without a style, `fallback`: "live" (Liberty) or
 * "tokyo" (the committed extract, for benchmarks and offline work).
 */
export function demoStyle(params, fallback = "live") {
  if (!params.get("style") && !params.get("format") && fallback === "tokyo") return sceneUrl(SCENES.tokyo, "mvt");
  const { scene, format } = sceneFor(params);
  return sceneUrl(scene, format);
}
