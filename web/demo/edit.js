// The style edit API (plan §11.2, §14.3): `edit(tx => …)` collects GL JS
// map calls into one transaction, applied as one style generation;
// `setFeatureState` and `setGlobalState` are one-call transactions. Each
// resolves to the generation whose frames show the edit in full (frame acks
// carry `generation`), or rejects with the transaction's error.
//
// `apply(commands)` sends a JSON array of calls to the map, on this thread
// or in the render worker, and resolves to its generation.

// The calls a transaction can make, as GL JS names them.
const CALLS = [
  "setPaintProperty", "setLayoutProperty", "setFilter", "setLayerZoomRange",
  "addLayer", "removeLayer", "moveLayer", "addSource", "removeSource",
  "setGlobalStateProperty", "setFeatureState", "removeFeatureState", "setStyle",
];

export function editApi(apply) {
  const edit = (build) => {
    const commands = [];
    const tx = Object.fromEntries(CALLS.map((name) =>
      [name, (...args) => { commands.push([name, ...args]); return tx; }]));
    build(tx);
    return apply(JSON.stringify(commands));
  };
  return {
    edit,
    setFeatureState: (feature, state) => edit((tx) => tx.setFeatureState(feature, state)),
    removeFeatureState: (feature, key) => edit((tx) => tx.removeFeatureState(feature, key)),
    setGlobalState: (name, value) => edit((tx) => tx.setGlobalStateProperty(name, value)),
  };
}
