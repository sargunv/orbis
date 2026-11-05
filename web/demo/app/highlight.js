// Hover and selection highlights through feature state (decision 0016):
// for each source layer a feature is picked from, three layers drawn in
// the accent where `hover` or `selected` state is set, added before the
// style's first symbol layer. Picking a feature is then one feature-state
// write, which redraws without touching any mesh.

export const HIGHLIGHT_PREFIX = "orbis-highlight:";
const ACCENT = "#e5007e";

const state = (key) => ["boolean", ["feature-state", key], false];
const byState = (selected, hover, otherwise) => ["case", state("selected"), selected, state("hover"), hover, otherwise];
const isType = (...types) => ["in", ["geometry-type"], ["literal", types]];

function layers(source, sourceLayer) {
  const id = `${HIGHLIGHT_PREFIX}${source}:${sourceLayer ?? ""}:`;
  const base = { source, ...(sourceLayer ? { "source-layer": sourceLayer } : {}) };
  return [
    { ...base, id: `${id}fill`, type: "fill", filter: isType("Polygon", "MultiPolygon"),
      paint: { "fill-color": ACCENT, "fill-opacity": byState(0.3, 0.16, 0) } },
    { ...base, id: `${id}line`, type: "line", filter: ["!", isType("Point", "MultiPoint")],
      layout: { "line-join": "round", "line-cap": "round" },
      paint: { "line-color": ACCENT, "line-width": byState(3, 2, 0), "line-opacity": byState(1, 0.85, 0) } },
    { ...base, id: `${id}circle`, type: "circle", filter: isType("Point", "MultiPoint"),
      paint: { "circle-radius": byState(9, 8, 0), "circle-opacity": 0, "circle-stroke-color": ACCENT,
        "circle-stroke-width": byState(3, 2.5, 0) } },
  ];
}

/** A feature target for a hit, or null where state can't address it. */
export function targetOf(hit, style) {
  if (!hit || hit.id === null || hit.id === undefined) return null;
  const type = style?.sources?.[hit.source]?.type;
  if (type !== "vector" && type !== "geojson") return null;
  return { source: hit.source, ...(hit.sourceLayer ? { sourceLayer: hit.sourceLayer } : {}), id: hit.id };
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export class Highlighter {
  /** `style()` is the style as the app last read it. */
  constructor(map, style) {
    this.map = map;
    this.style = style;
    this.added = new Set();
    this.targets = { hover: null, selected: null };
    this.chain = Promise.resolve();
  }

  /** The style was replaced: its highlight layers and state are gone. */
  reset() {
    this.added.clear();
    this.targets = { hover: null, selected: null };
  }

  /** Moves `key` ("hover" or "selected") to `target`; resolves once applied. */
  set(key, target) {
    this.chain = this.chain.catch(() => {}).then(() => this.#apply(key, target));
    return this.chain;
  }

  async #apply(key, target) {
    const previous = this.targets[key];
    if (same(previous, target)) return;
    const style = this.style();
    const firstSymbol = style?.layers.find((l) => l.type === "symbol" && !l.id.startsWith(HIGHLIGHT_PREFIX))?.id;
    const group = target && `${target.source}:${target.sourceLayer ?? ""}`;
    const add = target && !this.added.has(group);
    await this.map.edit((tx) => {
      if (previous && style?.sources[previous.source]) tx.removeFeatureState(previous, key);
      if (!target) return;
      if (add) for (const layer of layers(target.source, target.sourceLayer)) tx.addLayer(layer, firstSymbol);
      tx.setFeatureState(target, { [key]: true });
    });
    if (add) this.added.add(group);
    this.targets[key] = target;
  }
}
