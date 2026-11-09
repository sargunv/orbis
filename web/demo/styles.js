// The demo's styles: the catalogue the map app's picker shows (the
// basemaps by provider, and a few entries for features: a projection, a
// CRS, the globe, terrain), and `demoStyle` for the bench and measurement
// pages.
import mvt from "./data/tokyo-mvt.json?url";
import mlt from "./data/tokyo-mlt.json?url";

import arctic from "./data/crs/arctic.json?url";
import alps from "./data/alps.json?url";
import osmCarto from "./data/osm-carto.json?url";
import alpsFlight from "./data/alps-flight.json";

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

/**
 * A picker entry. `basemap` entries keep the camera when switched to; the
 * rest open at their own. `swatch` is [land, water, line] for the picker's
 * chip; `dark` styles turn the app's chrome dark. `constraints` keep the
 * camera where a bounded dataset has data, or a view in range (the globe
 * no smaller than at zoom 0), on both maps of a comparison. `projection` is a v8 projection
 * the map switches to once the style loads, and `sky` a v8 sky it puts
 * in place of the style's then too. `notInGlJs` says why the
 * comparison with GL JS (app/compare.js) can't show the entry.
 *
 * @typedef {{ key: string, name: string, detail?: string, url: string, provider: string,
 *   swatch: [string, string, string], dark?: boolean, basemap?: boolean, camera?: object,
 *   constraints?: object, flight?: object[], formats?: Record<string, string>,
 *   projection?: object, sky?: object, notInGlJs?: string }} Scene
 */

/** A blue sky around the globe, a subtle rim at low zoom that thins out as it hands over to Mercator. */
const GLOBE_SKY = { "sky-color": "#4f8fd9", "horizon-color": "#d6e8fb", "fog-color": "#e4eef8",
  "sky-horizon-blend": 0.7, "horizon-fog-blend": 0.6, "fog-ground-blend": 0.7,
  "atmosphere-blend": ["interpolate", ["linear"], ["zoom"], 0, 0.4, 2, 0.32, 4, 0.18, 8, 0] };

/** @type {{ name: string, scenes: Scene[] }[]} */
export const GROUPS = [
  {
    name: "Features",
    scenes: [
      { key: "globe", name: "Globe", detail: "Liberty on the v8 globe under a v8 sky, Web Mercator from zoom 12",
        url: LIVE, projection: { type: "globe" }, sky: GLOBE_SKY, camera: { center: [10, 30], zoom: 1.5 },
        constraints: { minZoom: 0 }, dark: true, swatch: ["#f8f4f0", "#a0c8f0", "#02040b"] },
      { key: "equal-earth", name: "Equal Earth", detail: "The whole world equal-area, handing over to Web Mercator by zoom 7",
        url: LIVE, projection: { type: "equal-earth" }, camera: { center: [0, 10], zoom: 0.5 },
        swatch: ["#f8f4f0", "#a0c8f0", "#e9ac77"],
        notInGlJs: "GL JS has no Equal Earth projection" },
      { key: "arctic", name: "Arctic", detail: "A polar stereographic CRS, EPSG:3413", url: arctic, dark: true,
        swatch: ["#24476b", "#e9eef2", "#9aa6ad"],
        notInGlJs: "GL JS can't draw tile matrix sets" },
      { key: "alps", name: "Alps terrain", detail: "3D terrain, sky and a looping camera flight", url: alps,
        flight: alpsFlight.legs, swatch: ["#eef0e4", "#9fb7c9", "#7d8a74"] },
    ],
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
/** The committed Tokyo extract, by URL only: the benchmarks', checks' and offline work's style. */
SCENES.tokyo = { key: "tokyo", name: "Tokyo, offline", detail: "Bundled extract, no network", url: mvt,
  provider: "Orbis", formats: { mvt, mlt }, constraints: { bounds: TOKYO_BOUNDS, maxPitch: 0 },
  swatch: ["#f8f4f0", "#a0c8f0", "#e9ac77"] };

/** The default scene. */
export const DEFAULT_SCENE = SCENES.liberty;

/**
 * The scene `params` name (and, for Tokyo, its format): a catalogue key,
 * `tokyo[-mvt|-mlt]`, a style URL matching an entry, or any other style
 * URL as a scene of its own.
 */
export function sceneFor(params) {
  const style = params.get("style");
  const format = params.get("format");
  if (format && !["mvt", "mlt"].includes(format)) throw new Error("format must be mvt or mlt");
  if (!style) return format ? { scene: SCENES.tokyo, format } : { scene: DEFAULT_SCENE };
  const tokyo = style.match(/^tokyo(?:-(mvt|mlt))?$/);
  if (tokyo) return { scene: SCENES.tokyo, format: tokyo[1] ?? format ?? "mvt" };
  if (SCENES[style]) return { scene: SCENES[style] };
  const url = new URL(style, location.href).href;
  const known = Object.values(SCENES)
    .find((s) => !s.projection && !s.sky && new URL(s.url, location.href).href === url);
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
