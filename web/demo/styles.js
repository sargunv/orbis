import mvt from "./data/tokyo-mvt.json?url";
import mlt from "./data/tokyo-mlt.json?url";

import playground from "./data/playground.json?url";
import scripts from "./data/scripts.json?url";
import arctic from "./data/crs/arctic.json?url";
import greenland from "./data/crs/greenland.json?url";

/** OpenFreeMap's live Liberty style over the planet: the demo's default. */
export const LIVE = "https://tiles.openfreemap.org/styles/liberty";
/** Where the live map opens: central Tokyo, around Tokyo Station. */
export const LIVE_CAMERA = { center: [139.767, 35.681], zoom: 13 };

/**
 * The style a demo page shows. `style=tokyo-mvt` or `tokyo-mlt` (or
 * `format=mvt|mlt`) is the committed Tokyo extract, for benchmarks,
 * snapshots and offline work; `style=scripts` the Cairo, Delhi and Bangkok
 * page's; `style=arctic` the Arctic on EPSG:3413 and `style=greenland`
 * Greenland's ice on Web Mercator (scene documents, plan §11.4); any other
 * `style` a URL. Without one, the playground has its own
 * and every other page `fallback`: "live" (OpenFreeMap) or "tokyo".
 */
export function demoStyle(params, fallback = "live") {
  const style = params.get("style");
  const format = params.get("format");
  if (format && !["mvt", "mlt"].includes(format)) throw new Error("format must be mvt or mlt");
  const tokyo = { "tokyo-mvt": mvt, "tokyo-mlt": mlt }[style] ?? (format ? { mvt, mlt }[format] : undefined);
  const named = { scripts, arctic, greenland }[style];
  const url = named
    ?? tokyo ?? style
    ?? (location.pathname.endsWith("playground.html") ? playground : fallback === "tokyo" ? mvt : LIVE);
  return new URL(url, location.href).href;
}

/** Whether `params` show the live default map. */
export function isLive(params) {
  return !params.get("style") && !params.get("format");
}
