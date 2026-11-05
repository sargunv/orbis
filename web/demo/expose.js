// The demo pages' console and test hooks: `globalThis.orbis` (the map's
// edit and query API) and an `orbis-frame` window event per frame.
import { setRequestTransform } from "orbis";

export function expose(map) {
  globalThis.orbis = {
    map,
    edit: (build) => map.edit(build),
    setFeatureState: (feature, state) => map.setFeatureState(feature, state),
    removeFeatureState: (feature, key) => map.removeFeatureState(feature, key),
    setGlobalState: (name, value) => map.setGlobalState(name, value),
    hitTest: (x, y) => map.hitTest([x, y]),
    getStyle: async () => JSON.stringify(await map.getStyle()),
    semanticSnapshot: () => map.semanticSnapshot(),
    loseDevice: () => map.loseDevice(),
    trapPlacementWorker: () => map.trapPlacementWorker(),
    setRequestTransform,
  };
  map.on("render", (ack) => window.dispatchEvent(new CustomEvent("orbis-frame", { detail: ack })));
}

/** `#zoom/lat/lon[/bearing/pitch]`, or undefined. */
export function hashCamera() {
  const hash = location.hash.slice(1).split("/").map(Number);
  if (hash.length < 3 || !hash.every(Number.isFinite)) return undefined;
  const [zoom, lat, lon, bearing = 0, pitch = 0] = hash;
  return { center: [lon, lat], zoom, bearing, pitch };
}

/** The gesture feel for this device. */
export function feel() {
  if (!matchMedia("(pointer: coarse)").matches) return "desktop";
  return /iPhone|iPad|iPod|Macintosh/.test(navigator.userAgent) ? "ios" : "android";
}
