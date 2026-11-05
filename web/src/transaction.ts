// Style edits as transactions of GL JS map calls (plan §11.2): one style
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
  setGlobalStateProperty(name: string, value: unknown): Transaction;
  setFeatureState(feature: FeatureTarget, state: Record<string, unknown>): Transaction;
  removeFeatureState(feature: Partial<FeatureTarget>, key?: string): Transaction;
  setStyle(style: object): Transaction;
  /** v8 `projection`: `{type: "globe"}`, `"vertical-perspective"`, `"mercator"`, `"equal-earth"`, `"natural-earth"`, `"equirectangular"`, with an optional `transition: [start, end] | false`. */
  setProjection(projection: object): Transaction;
}

export interface FeatureTarget {
  source: string;
  sourceLayer?: string;
  id: string | number;
}

const CALLS = [
  "setPaintProperty", "setLayoutProperty", "setFilter", "setLayerZoomRange",
  "addLayer", "removeLayer", "moveLayer", "addSource", "removeSource", "setSourceData",
  "setGlobalStateProperty", "setFeatureState", "removeFeatureState", "setStyle",
  "setProjection",
] as const;

/** A transaction's calls as the JSON array the core applies. */
export function transaction(build: (tx: Transaction) => void): string {
  const commands: unknown[][] = [];
  const tx = Object.fromEntries(CALLS.map((name) =>
    [name, (...args: unknown[]) => { commands.push([name, ...args]); return tx; }])) as unknown as Transaction;
  build(tx);
  return JSON.stringify(commands);
}

