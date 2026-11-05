// Orbis on the web: `createMap(options)` → a map with a camera, style edits,
// events, queries and annotations, drawing on the page's thread (plan §14.3).
export { createMap, OrbisMap, CameraControl } from "./map.ts";
export { transaction } from "./transaction.ts";
export type { Transaction, FeatureTarget } from "./transaction.ts";
export { Annotation, Annotations } from "./annotations.ts";
export { downloadRegion, deleteRegion, regions } from "./regions.ts";
export type { Region, RegionDefinition, RegionEvent } from "./regions.ts";
export type { Anchor, AnnotationOptions } from "./annotations.ts";
export type * from "./types.ts";
