// The package's public types. Coordinates are [longitude, latitude] in
// degrees; points are [x, y] in CSS pixels from the map's top-left corner.

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
}

/** Where a camera move ends; what is left out stays. */
export interface CameraOptions {
  center?: LngLat;
  zoom?: number;
  bearing?: number;
  pitch?: number;
  /** Without `center`: the point a zoom or rotation keeps in place. */
  around?: LngLat;
  /** Pixels the center ends up from the viewport's middle. */
  offset?: Point;
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
   * Where the map draws: all on this thread ("main", the default by plan
   * §7.2's rule, decision 0021), or in a render worker with the camera on
   * this thread ("worker"). "auto" takes the worker where OffscreenCanvas
   * and WebGPU in workers work, else this thread.
   */
  thread?: "auto" | "worker" | "main";
  /** The starting camera; the style's by default. */
  camera?: CameraOptions;
  constraints?: Constraints;
  /** Gesture feel; "desktop" (GL JS's) by default. */
  feel?: "desktop" | "ios" | "android";
  /** Whether input moves the map; true by default. */
  interactive?: boolean;
}

/** The topmost feature under a point, from the published picking index. */
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

/** A frame's counters. */
export interface FrameStats {
  /** CPU milliseconds the frame took where it rendered. */
  ms: number;
  zoom: number;
  generation: number;
  draws: number;
  dispatches: number;
  candidates: number;
  tilesDrawn: number;
  tilesLoading: number;
  bundleRecorded: boolean;
  placementsWritten: number;
  paintsWritten: number;
  idle: boolean;
  meshesUploaded: number;
  rowsUploaded: number;
  uploadedFromWasm: number;
  uploadedFromJs: number;
  texelsFromWasm: number;
  texelsFromJs: number;
  headerBytes: number;
  jobsDone: number;
  warnings: string[];
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
  /** A frame was drawn. */
  render: FrameStats;
}
