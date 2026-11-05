// The package's public types. Coordinates are [longitude, latitude] in
// degrees; points are [x, y] in CSS pixels from the map's top-left corner.

import type { FrameMemory, FrameStats } from "../pkg/core.js";

export type LngLat = [number, number];
export type Point = [number, number];
/** [west, south, east, north] in degrees. */
export type Bounds = [number, number, number, number];

/** A camera: where it looks and how (GL JS `CameraOptions`). */
export interface Camera {
  center: LngLat;
  zoom: number;
  /** Degrees clockwise from north. */
  bearing: number;
  /** Degrees from looking straight down. */
  pitch: number;
  /** Degrees the view turns about its axis, clockwise. */
  roll: number;
  /** Metres the center stands above sea level: the terrain's, unless
   * the center isn't clamped to the ground. */
  elevation: number;
  /** The vertical field of view, degrees. */
  fov: number;
  /** CSS pixels the map keeps clear at its edges (GL JS `padding`): the
   * center and the vanishing point sit in the middle of the rest. */
  padding: Padding;
}

/** A camera given by its eye (GL JS `FreeCameraOptions`). */
export interface FreeCamera {
  /** The point under the eye. */
  position: LngLat;
  /** Metres above sea level. */
  altitude: number;
  bearing: number;
  pitch: number;
  roll: number;
}

/** Where a camera move ends; what is left out stays. */
export interface CameraOptions {
  center?: LngLat;
  zoom?: number;
  bearing?: number;
  pitch?: number;
  roll?: number;
  /** Metres the center stands above sea level (GL JS `elevation`); the
   * terrain takes it while the center is clamped to the ground. */
  elevation?: number;
  /** Without `center`: the point a zoom or rotation keeps in place. */
  around?: LngLat;
  /** Pixels the center ends up from the padded viewport's middle. */
  offset?: Point;
  /** The vertical field of view, degrees. */
  fov?: number;
  /** The camera's padding from here on; sides left out stay. */
  padding?: number | Partial<Padding>;
}

/** A cubic Bézier's control points, or "linear". GL JS's ease by default. */
export type Easing = [number, number, number, number] | "linear";

export interface EaseOptions extends CameraOptions {
  /** Milliseconds; 500 by default, 0 jumps. */
  duration?: number;
  easing?: Easing;
}

export interface FlyOptions extends CameraOptions {
  /** How far the flight zooms out (ρ); 1.42 by default. */
  curve?: number;
  /** Average speed in ρ-screenfuls per second; 1.2 by default. */
  speed?: number;
  screenSpeed?: number;
  duration?: number;
  /** A flight longer than this jumps instead. */
  maxDuration?: number;
  minZoom?: number;
  easing?: Easing;
}

export interface Padding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface FitOptions {
  padding?: number | Partial<Padding>;
  offset?: Point;
  maxZoom?: number;
  bearing?: number;
  pitch?: number;
}

export interface Constraints {
  minZoom?: number;
  maxZoom?: number;
  minPitch?: number;
  maxPitch?: number;
  /** Keep the viewport inside this box. */
  bounds?: Bounds | null;
}

export interface MapOptions {
  /** The element the map fills; its canvas and overlays go inside. */
  container: HTMLElement;
  /** A v8 style URL, or a style object. */
  style: string | object;
  /**
   * What draws the map: "webgpu" (the Modern tier), "webgl2" (Compat, plan
   * §8.7, from the lazily loaded compat.wasm), or "auto" (the default):
   * WebGPU where it works, else WebGL2. A lost device is
   * replaced on the same tier where possible, else on WebGL2.
   */
  renderer?: "auto" | "webgpu" | "webgl2";
  /** The starting camera; the style's by default. */
  camera?: CameraOptions;
  constraints?: Constraints;
  /** Gesture feel; "desktop" (GL JS's) by default. */
  feel?: "desktop" | "ios" | "android";
  /** Whether input moves the map; true by default. */
  interactive?: boolean;
  /** Device pixels per CSS pixel to draw with; the display's
   * (`devicePixelRatio`) by default. */
  pixelRatio?: number;
  /** How long labels fade in and out, ms; 300 by default. */
  fadeDuration?: number;
  /** Whether labels of different sources collide; true by default. */
  crossSourceCollisions?: boolean;
  /** The CSS font family ideographs (CJK and the like) of the default
   * font stack draw with instead of the style's glyphs; `sans-serif` by
   * default, `false` for the glyphs. */
  localIdeographFontFamily?: string | false;
  /** Multisampling for 3D surfaces; true by default. */
  antialias?: boolean;
  /** Whether terrain tiles hang skirts hiding the cracks between tiles of
   * different detail; translucent terrain shows them. True by default. */
  terrainSkirts?: boolean;
  /** How many job workers lay out tiles; by default
   * `clamp(hardwareConcurrency − 2, 1, 4)` (plan §7.1). */
  jobWorkers?: number;
}

/** The topmost feature under a point, from the last frame. */
export interface Hit {
  layer: string;
  source: string;
  sourceLayer: string | null;
  id: string | null;
  generation: number;
}

/** A GeoJSON geometry in longitude and latitude. */
export interface Geometry {
  type: "Point" | "MultiPoint" | "LineString" | "MultiLineString" | "Polygon" | "MultiPolygon";
  coordinates: unknown;
}

/** A queried feature, GL JS's `MapGeoJSONFeature` as reference. */
export interface Feature {
  type: "Feature";
  id?: string | number;
  geometry: Geometry;
  properties: Record<string, unknown>;
  source: string;
  sourceLayer?: string;
  /** The style layer it was drawn by; absent from `querySourceFeatures`. */
  layer?: { id: string; type: string; source: string; "source-layer"?: string };
  state: Record<string, unknown>;
}

/** A point, or a box between two corners. */
export type QueryGeometry = Point | [Point, Point];

export interface QueryOptions {
  /** Only these style layers. */
  layers?: string[];
}

export interface SourceQueryOptions {
  sourceLayer?: string;
  /** A v8 filter expression. */
  filter?: unknown;
}

/** A frame's counters: orb-web's `FrameStats`, `memory` its `FrameMemory`. */
export type { FrameStats, FrameMemory };

/** A GPU submission's times ([`OrbisMap.gpuTimings`]), milliseconds. */
export interface GpuSubmission {
  /** Each pass's label and GPU time; empty without timestamp queries. */
  passes: [string, number][];
  /** Encoding and submitting the passes. */
  cpu: number;
}

/** An image for the style (`addImage`): `ImageData`, or straight
 * (not premultiplied) RGBA8 rows, top first. */
export interface StyleImageData {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
}

/** How icons and patterns read an added image, as sprite metadata spells it. */
export interface StyleImageOptions {
  pixelRatio?: number;
  /** A signed distance field, tinted by `icon-color`. */
  sdf?: boolean;
  stretchX?: [number, number][];
  stretchY?: [number, number][];
  content?: [number, number, number, number];
  textFitWidth?: "stretchOrShrink" | "stretchOnly" | "proportional";
  textFitHeight?: "stretchOrShrink" | "stretchOnly" | "proportional";
}


export interface CameraEventData { user: boolean }
export interface PointerEventData { point: Point; lngLat: LngLat | null; button?: number }

/** Every event, by name, with what its listeners receive. */
export interface MapEvents {
  movestart: CameraEventData; move: CameraEventData; moveend: CameraEventData;
  zoomstart: CameraEventData; zoom: CameraEventData; zoomend: CameraEventData;
  rotatestart: CameraEventData; rotate: CameraEventData; rotateend: CameraEventData;
  pitchstart: CameraEventData; pitch: CameraEventData; pitchend: CameraEventData;
  click: PointerEventData;
  dblclick: PointerEventData;
  /** The style is in (GL JS `style.load`). */
  styleload: Record<string, never>;
  /** The style changed; frames show it in full from `generation`. */
  styledata: { generation: number };
  sourcedata: { source: string; loaded: boolean };
  /** The first frame with the style and every visible tile in. */
  load: Record<string, never>;
  /** No loads, fades, placement or camera motion: no more frames until something changes. */
  idle: Record<string, never>;
  error: { message: string };
  /** Something in the style or its data the map can't draw as written. */
  warning: { message: string };
  /** A tile's layers name an image the style doesn't have (GL JS
   * `styleimagemissing`); once per name until it's added with `addImage`. */
  styleimagemissing: { id: string };
  /**
   * The tier the map draws at: on its first frame, and after a device loss
   * on its first frame on the new device, with the milliseconds since the
   * loss.
   */
  capability: { tier: "modern" | "compat"; recoveredMs?: number };
  /** The GPU was lost (GL JS `webglcontextlost`): the map draws nothing
   * until it has another, then reports `capability`. Failing that, `error`. */
  lost: Record<string, never>;
  /** A frame was drawn. */
  render: FrameStats;
}

/** A marker the map draws: an image at a point (`map.addMarker`). */
export interface MarkerOptions {
  lngLat: LngLat;
  /** An image of the style (`addImage`, or its sprite's); a built-in pin
   * by default. */
  image?: string;
  /** Which part of the image sits on the point (`icon-anchor`): the pin's
   * tip, or the centre of other images, by default. */
  anchor?: "center" | "top" | "bottom" | "left" | "right" | "top-left" | "top-right" | "bottom-left" | "bottom-right";
}

/** A location fix for `map.setLocation`. */
export interface LocationFix {
  lngLat: LngLat;
  /** Horizontal accuracy, metres: the circle drawn around the dot. */
  accuracy?: number;
  /** Degrees clockwise from north the device faces. */
  heading?: number;
}
