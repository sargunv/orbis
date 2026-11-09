// The package's public types. Coordinates are [longitude, latitude] in
// degrees; points are [x, y] in CSS pixels from the map's top-left corner.

import type { FrameMemory as CoreMemory, FrameStats as CoreStats } from "../pkg/core.js";

/** A wasm-bindgen object's fields, as a plain object, its objects' too. */
type Fields<T> = {
  -readonly [K in keyof T as T[K] extends (...args: never[]) => unknown ? never : K]: T[K] extends object ? Fields<T[K]> : T[K]
};

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

/** How one part of a momentum slows from its release speed, in the
 * part's unit (pixels, zoom levels, degrees): maplibre-native-ffi's
 * `mln_momentum_decay`. Left out, each setting takes the part's default:
 * friction with a 140 ms half-life. */
export interface MomentumDecay {
  /** "friction" halves the speed every `halfLife` ms until `stopSpeed`;
   * "ease" moves `distance` over `duration` ms along `easing`. */
  model?: "friction" | "ease";
  halfLife?: number;
  stopSpeed?: number;
  distance?: number;
  duration?: number;
  easing?: Easing;
  /** A slower release doesn't move. */
  minSpeed?: number;
  /** A faster friction release counts as this. */
  maxSpeed?: number;
  /** The farthest the part moves. */
  maxDistance?: number;
}

/** Where a part settles once it stops: the nearest of the values every
 * `spacing` from `offset`, within `tolerance`, eased over `duration` ms. */
export interface MomentumSnap {
  spacing?: number;
  offset?: number;
  tolerance?: number;
  duration?: number;
}

/** A zoom, bearing or pitch that keeps changing after release. */
export interface AnchoredMomentum {
  /** Zoom levels, or degrees, per second at release; without it the part
   * only settles. */
  velocity?: number;
  /** The point the change keeps in place; the center without one. */
  anchor?: Point;
  decay?: MomentumDecay;
  /** `true` for the part's default: the nearest whole zoom level within
   * 0.5, north within 7°, flat within 7°. */
  snap?: boolean | MomentumSnap;
}

/** How a gesture the page recognized itself keeps moving the camera after
 * release: maplibre-native-ffi's `mln_gesture_momentum`. */
export interface GestureMomentum {
  /** Pixels per second the content moved on screen at release. */
  pan?: { velocity: Point; decay?: MomentumDecay };
  zoom?: AnchoredMomentum;
  bearing?: AnchoredMomentum;
  pitch?: AnchoredMomentum;
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
  /** The starting camera; the style's by default. */
  camera?: CameraOptions;
  constraints?: Constraints;
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
  /** The previews the style may use: style inputs beyond the approved v8
   * spec, by name (`"style-spec#1865"`, `"orbis:crs"`), or `"all"`. A
   * preview that is off warns and its inputs fall back to their defaults.
   * None by default. */
  previews?: readonly string[] | "all";
  /** Whether labels keep their place while the camera moves, as MapLibre
   * Native keeps them: a label tries its previous variable anchor first (in
   * tilted views, only that anchor), and in tilted views labels shown before
   * are placed before new and hidden ones. What labels show then depends on
   * how the camera got there; off, on the camera alone. True by default. */
  stableLabels?: boolean;
  /** How many job workers lay out tiles; by default
   * `clamp(hardwareConcurrency − 2, 1, 4)`. */
  jobWorkers?: number;
  /** Whether the map may run in a worker, off the page's thread: where
   * the browser can draw WebGPU into an OffscreenCanvas there and tells
   * the page of pointer moves as they come (Chromium-based browsers).
   * True by default; false keeps it on the page. */
  worker?: boolean;
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

/** GPU memory by pool: orb-web's `FrameMemory`. */
export type FrameMemory = Fields<CoreMemory>;

/** A frame's counters: orb-web's `FrameStats`. */
export type FrameStats = Fields<CoreStats> & {
  /** The `timeStamp` of the latest input the frame shows, if any, on the
   * page's clock: input latency is the frame's display time after it. */
  inputTime?: number;
};

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
  /** The map needs a tile of a source the app feeds (`"tiles":
   * ["orbis-host:"]`): answer with `setHostTile`. Once per tile until answered. */
  tilerequest: { source: string; z: number; x: number; y: number };
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
  /** The GPU was lost (GL JS `webglcontextlost`): the map draws nothing
   * until it has another, then reports `restored`. Failing that, `error`. */
  lost: Record<string, never>;
  /** The map draws on a new GPU after a loss (GL JS
   * `webglcontextrestored`): its first frame there, the milliseconds since
   * the loss. */
  restored: { recoveredMs: number };
  /** A frame was drawn. */
  render: FrameStats;
}

