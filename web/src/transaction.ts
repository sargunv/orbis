// Style edits as transactions of GL JS map calls: one style
// generation per transaction.

/** The calls a style edit can make, as GL JS names them. */
export interface Transaction {
  setPaintProperty(layer: string, name: string, value: unknown): Transaction;
  setLayoutProperty(layer: string, name: string, value: unknown): Transaction;
  setFilter(layer: string, filter: unknown): Transaction;
  setLayerZoomRange(layer: string, minzoom: number, maxzoom: number): Transaction;
  addLayer(layer: object, before?: string): Transaction;
  removeLayer(layer: string): Transaction;
  moveLayer(layer: string, before?: string): Transaction;
  addSource(id: string, source: object): Transaction;
  removeSource(id: string): Transaction;
  /** A GeoJSON source's new data (GL JS's `setData`): its drawn tiles stay
   * until the new data's replace them. */
  setSourceData(id: string, data: object): Transaction;
  /** A GeoJSON source's data changed by a diff (GL JS's `updateData`):
   * only the diff goes to the map and its workers. */
  updateSourceData(id: string, diff: GeoJSONSourceDiff): Transaction;
  setGlobalStateProperty(name: string, value: unknown): Transaction;
  setFeatureState(feature: FeatureTarget, state: Record<string, unknown>): Transaction;
  removeFeatureState(feature: Partial<FeatureTarget>, key?: string): Transaction;
  setStyle(style: object): Transaction;
  /** v8 `projection`: `{type: "globe"}`, `"vertical-perspective"`, `"mercator"`, `"equal-earth"`, `"natural-earth"`, `"equirectangular"` or a coordinate reference system such as `"EPSG:3413"`, with an optional `transition: [start, end] | false`. Another CRS plane reloads every tile. */
  setProjection(projection: object): Transaction;
  /** v8 `sky`, or `null` for none. */
  setSky(sky: object | null): Transaction;
  /** v8 `terrain`, or `null` for none. */
  setTerrain(terrain: object | null): Transaction;
  /** v8 `light`. */
  setLight(light: object): Transaction;
  /** v8 root `transition`. */
  setTransition(transition: object): Transaction;
  /** v8 `sprite` (a URL or `[{id, url}]`), or `null` for none: its images load in place of the old ones. */
  setSprite(sprite: string | object[] | null): Transaction;
}

/** GL JS's `GeoJSONSourceDiff`: features by id (or the source's `promoteId`). */
export interface GeoJSONSourceDiff {
  removeAll?: boolean;
  remove?: (string | number)[];
  add?: object[];
  update?: {
    id: string | number;
    newGeometry?: object;
    removeAllProperties?: boolean;
    removeProperties?: string[];
    addOrUpdateProperties?: { key: string; value: unknown }[];
  }[];
}

export interface FeatureTarget {
  source: string;
  sourceLayer?: string;
  id: string | number;
}

const CALLS = [
  "setPaintProperty", "setLayoutProperty", "setFilter", "setLayerZoomRange",
  "addLayer", "removeLayer", "moveLayer", "addSource", "removeSource", "setSourceData", "updateSourceData",
  "setGlobalStateProperty", "setFeatureState", "removeFeatureState", "setStyle",
  "setProjection", "setSky", "setTerrain", "setLight", "setTransition", "setSprite",
] as const;

/** A transaction's calls as the JSON array the core applies. */
export function transaction(build: (tx: Transaction) => void): string {
  const commands: unknown[][] = [];
  const tx = Object.fromEntries(CALLS.map((name) =>
    [name, (...args: unknown[]) => { commands.push([name, ...args]); return tx; }])) as unknown as Transaction;
  build(tx);
  return JSON.stringify(commands);
}

