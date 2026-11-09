// The map: one wasm map on the page's thread owns the camera,
// its gestures, drawing and synchronous queries; tile jobs
// and labels run in workers.

import { Annotations } from "./annotations.ts";
import { Emitter, type Listener } from "./events.ts";
import { frameLoop } from "./frames.ts";
import { startMain, type Core, type Host } from "./hosts.ts";
import { Videos, type VideoEventObject } from "./videos.ts";
import { attachInput } from "./input.ts";
import { preloadStyle } from "./preload.ts";
import { defaultJobWorkers } from "./workers.ts";
import { transaction, type FeatureTarget, type Transaction } from "./transaction.ts";
import type {
  Bounds, Camera, CameraOptions, Constraints, EaseOptions, Feature, FitOptions, FlyOptions, FreeCamera, Padding,
  FrameStats, GestureMomentum, GpuSubmission, Hit, LngLat, MapEvents, MapOptions, Point, QueryGeometry, QueryOptions,
  SourceQueryOptions, StyleImageData, StyleImageOptions,
} from "./types.ts";

/** An image's RGBA bytes, as the wasm map takes them. */
const rgba = (image: StyleImageData) =>
  new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength);

/** The wasm map's events (`OrbisMap::take_events` in orb-web). */
type CoreEvent =
  | { type: "styleload" | "load" | "idle" }
  | { type: "styledata"; generation: number }
  | { type: "sourcedata"; source: string; loaded: boolean }
  | { type: "error" | "warning"; message: string }
  | { type: "styleimagemissing"; id: string }
  | { type: "restored" }
  | { type: "camera"; change: "move" | "zoom" | "rotate" | "pitch"; phase: "start" | "progress" | "end"; user: boolean }
  | { type: "animationend"; id: number; interrupted: boolean }
  | { type: "click"; x: number; y: number; button: number }
  | { type: "dblclick"; x: number; y: number }
  | { type: "tilerequest"; source: string; z: number; x: number; y: number }
  | VideoEventObject;

/** The camera: its state, moves and limits. */
export class CameraControl {
  #core: () => Core;
  #changed: () => void;
  #animations: Map<number, (arrived: boolean) => void>;

  /** @internal */
  constructor(core: () => Core, changed: () => void, animations: Map<number, (arrived: boolean) => void>) {
    this.#core = core;
    this.#changed = changed;
    this.#animations = animations;
  }

  /** The camera now. */
  get current(): Camera {
    const c = this.#core().camera();
    const camera: Camera = {
      center: [c.lon, c.lat], zoom: c.zoom, bearing: c.bearing, pitch: c.pitch, roll: c.roll,
      elevation: c.elevation, fov: c.fov, padding: c.padding as Padding,
    };
    c.free();
    return camera;
  }

  /** The eye now (GL JS `getFreeCameraOptions`). */
  get free(): FreeCamera { return this.#core().freeCamera() as FreeCamera; }

  /** Puts the eye there at once (GL JS `setFreeCameraOptions`), keeping the
   * field of view and padding; what `camera` leaves out stays. */
  jumpToFree(camera: Partial<FreeCamera>): void {
    this.#core().jumpToFree(camera);
    this.#changed();
  }

  /** Whether the center stands on the terrain at rest (GL JS
   * `centerClampedToGround`, true by default). Off, it keeps the elevation
   * it is given, and the camera orbits a point in the air. */
  get centerClampedToGround(): boolean { return this.#core().centerClampedToGround(); }
  set centerClampedToGround(clamped: boolean) {
    this.#core().setCenterClampedToGround(clamped);
    this.#changed();
  }

  /** Puts the center this many metres above sea level at once (GL JS
   * `setCenterElevation`); while clamped to the ground, the terrain takes it back. */
  setCenterElevation(elevation: number): void {
    this.#core().setCenterElevation(elevation);
    this.#changed();
  }

  /** Whether an animation or a gesture is moving the camera. */
  get moving(): boolean { return this.#core().isMoving(); }

  jumpTo(options: CameraOptions): void {
    this.#core().jumpTo(options);
    this.#changed();
  }

  /** Starts at the next frame; resolves true on arrival, false when interrupted. */
  easeTo(options: EaseOptions): Promise<boolean> {
    return this.#track(this.#core().easeTo(options));
  }

  /** Flies along GL JS's zoom-out curve from the next frame; resolves
   * true on arrival, false when interrupted. */
  flyTo(options: FlyOptions): Promise<boolean> {
    return this.#track(this.#core().flyTo(options));
  }

  /** The camera that shows `bounds`, or null when the padding leaves no room. */
  cameraForBounds(bounds: Bounds, options: FitOptions = {}): CameraOptions | null {
    return this.#core().cameraForBounds(new Float64Array(bounds), options) as CameraOptions | null;
  }

  /** Flies (or, with `linear`, eases) to show `bounds`; resolves false when it doesn't fit or is interrupted. */
  fitBounds(bounds: Bounds, options: FitOptions & FlyOptions & { linear?: boolean } = {}): Promise<boolean> {
    const target = this.cameraForBounds(bounds, options);
    if (!target) return Promise.resolve(false);
    const move = { ...options, ...target, offset: undefined };
    return options.linear ? this.easeTo(move) : this.flyTo(move);
  }

  /** Continues a gesture the page recognized itself: each part glides on
   * from its release velocity and slows by its decay, then settles on its
   * snap. Input or a camera move stops it. Resolves true once it ends on
   * its own, false when stopped. The map's own gestures end this way. */
  startMomentum(momentum: GestureMomentum): Promise<boolean> {
    const id = this.#core().startMomentum(momentum);
    if (id < 0) {
      this.#changed();
      return Promise.resolve(true);
    }
    return this.#track(id);
  }

  /** Stops the running animation; its promise resolves false. */
  stop(): void {
    this.#core().stop();
    this.#changed();
  }

  setConstraints(constraints: Constraints): void {
    this.#core().setConstraints(constraints);
    this.#changed();
  }

  #track(id: number): Promise<boolean> {
    this.#changed();
    return new Promise((resolve) => this.#animations.set(id, resolve));
  }
}

/** A map in a container element. Create one with {@link createMap}. */
export class OrbisMap {
  readonly container: HTMLElement;
  readonly camera: CameraControl;
  readonly annotations: Annotations;
  #host: Host;
  /** Video sources' players, on the page's thread. */
  #videos: Videos;
  #events = new Emitter<MapEvents>();
  #loop: ReturnType<typeof frameLoop>;
  #snapshotting = false;
  #snapshots: Promise<unknown> = Promise.resolve();
  #stats: FrameStats | undefined;
  #animations = new Map<number, (arrived: boolean) => void>();
  #detach: (() => void)[] = [];
  #canvas: HTMLCanvasElement;
  #frame: HTMLElement;
  #interactive: boolean;
  /** The device pixels per CSS pixel the map draws with; the display's
   * when undefined. */
  #pixelRatio: number | undefined;
  #detachInput: (() => void) | undefined;
  #observer: ResizeObserver;
  /** A device loss's start, while the host recovers. */
  #lostAt: number | undefined;
  /** The last device loss's start, until the new device draws. */
  #recovering: number | undefined;
  #removed = false;
  /** Whether a prepare is queued. */
  #preparing = false;

  /** @internal */
  constructor(container: HTMLElement, frame: HTMLElement, canvas: HTMLCanvasElement, overlay: HTMLElement,
    host: Host, interactive: boolean, pixelRatio?: number) {
    this.container = container;
    this.#frame = frame;
    this.#canvas = canvas;
    this.#host = host;
    this.#videos = new Videos(() => (this.#removed ? undefined : this.#host.core), () => this.#request());
    this.#interactive = interactive;
    this.#pixelRatio = pixelRatio;
    this.#loop = frameLoop((now) => this.#tick(now), (e) => this.#error(e));
    this.camera = new CameraControl(() => this.#host.core, () => this.#request(), this.#animations);
    this.annotations = new Annotations(overlay, () => this.#placeAnnotations());
    this.#observer = new ResizeObserver(() => this.#resize());
    this.#attach();
    this.#detach.push(() => { this.#detachInput?.(); this.#observer.disconnect(); });
    // Tiles that failed while offline load again when the network returns.
    const online = () => { this.#host.core.retryFailedTiles(); this.#request(); };
    addEventListener("online", online);
    this.#detach.push(() => removeEventListener("online", online));
    this.#request();
  }

  /** The canvas the map draws into and takes input from. */
  get canvas(): HTMLCanvasElement { return this.#canvas; }

  #attach(): void {
    const core = () => (this.#lostAt === undefined && !this.#removed ? this.#host.core : undefined);
    if (this.#interactive) this.#detachInput = attachInput(this.#canvas, core, () => this.#request());
    this.#observer.observe(this.#canvas);
  }

  /**
   * The camera's view, `width × height` CSS pixels (the canvas's by
   * default) at `scale` device pixels each, drawn offscreen once everything
   * in it has loaded and nothing fades: the same style, data and camera
   * always draw the same pixels. The map draws no frames of its own
   * meanwhile. Rejects after a minute.
   */
  snapshot(width = this.#canvas.clientWidth, height = this.#canvas.clientHeight,
    scale = this.#pixelRatio ?? devicePixelRatio): Promise<ImageData> {
    const run = async () => {
      const core = this.#host.core;
      this.#snapshotting = true;
      core.beginSnapshot(width, height, scale);
      try {
        const deadline = performance.now() + 60_000;
        while (!core.snapshotFrame()) {
          if (performance.now() > deadline) throw new Error("the map didn't settle in a minute");
          await new Promise((resolve) => setTimeout(resolve, 16));
        }
        const pixels: Uint8Array = await core.endSnapshot();
        const px = (v: number) => Math.max(1, Math.round(v * scale));
        return new ImageData(new Uint8ClampedArray(pixels.buffer as ArrayBuffer, pixels.byteOffset, pixels.length),
          px(width), px(height));
      } catch (e) {
        core.cancelSnapshot();
        throw e;
      } finally {
        this.#snapshotting = false;
        this.#loop.request();
      }
    };
    // One at a time.
    const next = this.#snapshots.then(run, run);
    this.#snapshots = next.catch(() => {});
    return next;
  }

  /** Loses the GPU on purpose, to exercise recovery. */
  loseDevice(): void { this.#host.loseDevice(); }
  /** Traps the placement worker's wasm, as a bug would, to exercise its
   * replacement: labels place again within a frame or two. */
  trapPlacementWorker(): void { this.#host.trapPlacementWorker(); }
  /** Traps every job worker's wasm, as a bug would, to exercise their
   * replacement: tiles load again on the new workers. */
  trapJobWorkers(): void {
    for (let i = 0; i < this.#host.jobWorkers; i++) this.#host.trapJobWorker(i);
  }

  /** The last frame's counters. */
  get stats(): FrameStats | undefined { return this.#stats; }

  /** The next frame the map draws, as the canvas shows it: for
   * screenshots and tests. Asks for that frame. */
  async readPixels(): Promise<ImageData> {
    const pixels: Promise<Uint8Array> = this.#host.core.readNextFrame();
    this.#request();
    const data = await pixels;
    return new ImageData(new Uint8ClampedArray(data.buffer as ArrayBuffer, data.byteOffset, data.length),
      this.canvas.width, this.canvas.height);
  }

  /** Draws a frame even if nothing changed (GL JS `triggerRepaint`). */
  triggerRepaint(): void { this.#request(); }

  /** Times the GPU's work from now on, for measurements: `gpuTimings`
   * collects it. WebGPU only. */
  setGpuTiming(on: boolean): void { this.#host.core.setGpuTiming(on); }

  /** Each GPU submission's times since the last call, once the GPU has
   * finished them, in order: its passes' GPU milliseconds by label (where
   * the device has timestamp queries) and the CPU's to submit them. */
  gpuTimings(): Promise<GpuSubmission[]> { return this.#host.core.gpuTimings(); }

  on<K extends keyof MapEvents>(type: K, listener: Listener<MapEvents[K]>): () => void {
    return this.#events.on(type, listener);
  }
  off<K extends keyof MapEvents>(type: K, listener: Listener<MapEvents[K]>): void {
    this.#events.off(type, listener);
  }
  once<K extends keyof MapEvents>(type: K): Promise<MapEvents[K]> { return this.#events.once(type); }
  /** `for await (const e of map.events("idle"))`. */
  events<K extends keyof MapEvents>(type: K): AsyncGenerator<MapEvents[K]> { return this.#events.events(type); }

  /** The pixel showing a point, on the terrain, or null behind the eye: now,
   * or after a jump to `camera` (without making it). */
  project(lngLat: LngLat, camera?: CameraOptions): Point | null {
    const p = this.#host.core.project(lngLat[0], lngLat[1], camera);
    return p ? [p[0], p[1]] : null;
  }

  /** The ground point under a pixel, on the terrain, or null above the
   * horizon: now, or after a jump to `camera`. */
  unproject(point: Point, camera?: CameraOptions): LngLat | null {
    const g = this.#host.core.unproject(point[0], point[1], camera);
    return g ? [g[0], g[1]] : null;
  }

  /** The topmost feature under a pixel, synchronously, from the last frame. */
  hitTest(point: Point): Hit | null {
    return this.#host.core.hitTest(point[0], point[1]) as Hit | null;
  }

  /** Every feature drawn under a point or box (all on screen without one), topmost first. */
  async queryRenderedFeatures(geometry?: QueryGeometry, options: QueryOptions = {}): Promise<Feature[]> {
    const box = geometry === undefined
      ? [0, 0, this.canvas.clientWidth, this.canvas.clientHeight]
      : typeof geometry[0] === "number" ? [geometry[0], geometry[1], geometry[0], geometry[1]] as number[]
        : [...(geometry as [Point, Point])[0], ...(geometry as [Point, Point])[1]];
    const json = await this.#host.core.queryRendered(new Float64Array(box),
      options.layers ? JSON.stringify(options.layers) : undefined) as string;
    return JSON.parse(json);
  }

  /** Every feature of a source in its loaded tiles that matches a filter. */
  async querySourceFeatures(source: string, options: SourceQueryOptions = {}): Promise<Feature[]> {
    const json = await this.#host.core.querySource(source, options.sourceLayer,
      options.filter === undefined ? undefined : JSON.stringify(options.filter)) as string;
    return JSON.parse(json);
  }

  /** A cluster's members one zoom in: clusters and points (GL JS `getClusterChildren`). */
  async getClusterChildren(source: string, clusterId: number): Promise<Feature[]> {
    return JSON.parse(await this.#host.core.queryCluster(source, "children", clusterId, 0, 0) as string);
  }

  /** Up to `limit` of a cluster's points, past the first `offset` (GL JS `getClusterLeaves`). */
  async getClusterLeaves(source: string, clusterId: number, limit = 10, offset = 0): Promise<Feature[]> {
    return JSON.parse(await this.#host.core.queryCluster(source, "leaves", clusterId, limit, offset) as string);
  }

  /** The zoom at which a cluster breaks apart (GL JS `getClusterExpansionZoom`). */
  async getClusterExpansionZoom(source: string, clusterId: number): Promise<number> {
    return JSON.parse(await this.#host.core.queryCluster(source, "expansionZoom", clusterId, 0, 0) as string);
  }

  /** Shows another style (a URL, or a style object) as the diff from this
   * one, like GL JS's `setStyle(style, {diff: true})`: sources defined alike
   * keep their tiles, layers alike keep their meshes, and paint eases by the
   * new style's `transition`. The camera stays. `styleload` fires once the
   * new style is in; a style with another tile grid, or other glyphs, loads
   * every tile afresh. */
  setStyle(style: string | object): void {
    const url = typeof style === "string"
      ? new URL(style, location.href).href
      : `data:application/json,${encodeURIComponent(JSON.stringify(style))}`;
    this.#host.core.setStyleUrl(url);
    this.#request();
  }

  /** Applies GL JS map calls as one transaction, or throws and applies
   * none; returns the style generation whose frames show it in full
   * (`render` events carry it). */
  edit(build: (tx: Transaction) => void): number {
    const generation = this.#host.core.edit(transaction(build));
    this.#request();
    return generation;
  }
  setFeatureState(feature: FeatureTarget, state: Record<string, unknown>): number {
    return this.edit((tx) => tx.setFeatureState(feature, state));
  }
  removeFeatureState(feature: Partial<FeatureTarget>, key?: string): number {
    return this.edit((tx) => tx.removeFeatureState(feature, key));
  }
  setGlobalState(name: string, value: unknown): number {
    return this.edit((tx) => tx.setGlobalStateProperty(name, value));
  }


  /** The app's answer for a `tilerequest` of a source it feeds
   * (`{"type": "vector", "tiles": ["orbis-host:"]}`): the tile in the
   * source's encoding, or its source layers as GeoJSON (`{"roads": {...}}`,
   * which the map tiles), or null for an empty tile. */
  setHostTile(source: string, tile: { z: number; x: number; y: number },
    data: Uint8Array | ArrayBuffer | Record<string, object> | null): void {
    const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data instanceof Uint8Array ? data : undefined;
    const geojson = data && !bytes ? JSON.stringify(data) : undefined;
    this.#host.core.hostTile(source, tile.z, tile.x, tile.y, bytes, geojson);
    this.#request();
  }

  /** Asks for every tile of the app's source again: its data changed. Drawn
   * tiles stay until the new ones replace them. */
  reloadHostTiles(source: string): void {
    this.#host.core.reloadHostTiles(source);
    this.#request();
  }

  /** The picture a `canvas` source shows now ({"type": "canvas", "canvas":
   * name, "coordinates": …}): the app draws it, and hands each new one in. */
  setSourceFrame(source: string, image: StyleImageData): void {
    this.#host.core.sourceFrame(source, image.width, image.height, rgba(image));
    this.#request();
  }

  /** Adds an image to the style (GL JS `addImage`) for icons, text and
   * patterns; throws when it has one by that name. */
  addImage(name: string, image: StyleImageData, options: StyleImageOptions = {}): void {
    this.#host.core.addImage(name, image.width, image.height, rgba(image), JSON.stringify(options));
    this.#request();
  }
  /** Replaces an image's pixels, of the same size (GL JS `updateImage`). */
  updateImage(name: string, image: StyleImageData): void {
    this.#host.core.updateImage(name, image.width, image.height, rgba(image));
    this.#request();
  }
  /** Removes an image from the style (GL JS `removeImage`). */
  removeImage(name: string): void {
    this.#host.core.removeImage(name);
    this.#request();
  }
  /** Plays, pauses or seeks (to `seconds`) a video source's video (GL JS
   * `VideoSource.play`, `pause`, `seek`); throws for a source that isn't one. */
  playVideo(source: string): void { this.#controlVideo(source, "play", 0); }
  pauseVideo(source: string): void { this.#controlVideo(source, "pause", 0); }
  seekVideo(source: string, seconds: number): void { this.#controlVideo(source, "seek", seconds); }
  #controlVideo(source: string, request: string, seconds: number): void {
    this.#host.core.controlVideo(source, request, seconds);
    // The request reaches the player with the next frame's events.
    this.#request();
  }
  /** Whether the style has an image by this name, from its sprites or added. */
  hasImage(name: string): boolean { return this.#host.core.hasImage(name); }

  /** The style as edits have left it. */
  getStyle(): object { return JSON.parse(this.#host.core.getStyle()); }
  /** Layer `id` as the style has it, its paint and layout properties included. */
  getLayer(id: string): Record<string, unknown> | undefined {
    const json = this.#host.core.getLayer(id);
    return json === undefined ? undefined : JSON.parse(json);
  }
  getSource(id: string): Record<string, unknown> | undefined {
    const json = this.#host.core.getSource(id);
    return json === undefined ? undefined : JSON.parse(json);
  }
  /** The layers' ids in drawing order. */
  getLayersOrder(): string[] { return this.#host.core.layerIds(); }
  getSourceIds(): string[] { return this.#host.core.sourceIds(); }

  /** Stops drawing and releases the map's workers and DOM. Animations
   * still running resolve false. Calling it again does nothing. */
  remove(): void {
    if (this.#removed) return;
    this.#removed = true;
    this.#loop.stop();
    for (const detach of this.#detach) detach();
    for (const resolve of this.#animations.values()) resolve(false);
    this.#animations.clear();
    this.#videos.destroy();
    this.#host.destroy();
    this.#frame.remove();
  }

  #request(): void { this.#loop.request(); }

  #resize(): void {
    if (this.#removed || this.#lostAt !== undefined) return;
    const ratio = this.#pixelRatio ?? devicePixelRatio;
    const [w, h] = [Math.round(this.canvas.clientWidth * ratio), Math.round(this.canvas.clientHeight * ratio)];
    this.canvas.width = w;
    this.canvas.height = h;
    this.#host.resize(w, h, ratio);
    this.#request();
  }

  /** One frame, shown at `now`; returns whether another is needed. */
  #tick(now: number): boolean {
    // A snapshot draws its own frames, and asks for one when it ends.
    if (this.#removed || this.#lostAt !== undefined || this.#snapshotting) return false;
    const stats = this.#host.core.render(now);
    if (stats.lost) {
      this.#recover();
      return false;
    }
    this.#stats = stats;
    this.#drain();
    if (this.#events.has("render")) this.#events.emit("render", stats);
    this.#placeAnnotations();
    return stats.needsFrame;
  }

  /** Gets a new GPU after a loss and draws on. */
  #recover(): void {
    if (this.#lostAt !== undefined) return;
    this.#lostAt = performance.now();
    this.#events.emit("lost", {});
    this.#host.recover().then(() => {
      this.#recovering = this.#lostAt;
      this.#lostAt = undefined;
      this.#request();
    }, (e) => this.#error(`the GPU was lost and no other could be made: ${e}`));
  }

  /** @internal A failure where the map draws. */
  fail(e: unknown): void { this.#error(e); }

  /** @internal The map has new data to draw. Its next requests go out
   * in a task of their own, without waiting for the frame. */
  wake(): void {
    this.#request();
    if (this.#preparing) return;
    this.#preparing = true;
    afterTask(() => {
      this.#preparing = false;
      if (this.#removed || this.#lostAt !== undefined || this.#snapshotting) return;
      try {
        this.#host.core.prepare();
      } catch (e) {
        this.#error(e);
      }
    });
  }

  /** The wasm map's events since the last frame, to listeners. */
  #drain(): void {
    for (const event of this.#host.core.takeEvents() as CoreEvent[]) {
      switch (event.type) {
        case "styleload": case "load": case "idle":
          this.#events.emit(event.type, {});
          break;
        case "styledata": this.#events.emit("styledata", { generation: event.generation }); break;
        case "sourcedata": this.#events.emit("sourcedata", { source: event.source, loaded: event.loaded }); break;
        case "error": this.#error(event.message); break;
        case "warning": this.#events.emit("warning", { message: event.message }); break;
        case "video": this.#videos.request(event); break;
        case "styleimagemissing": this.#events.emit("styleimagemissing", { id: event.id }); break;
        case "restored": {
          if (this.#recovering === undefined) break;
          const recoveredMs = performance.now() - this.#recovering;
          this.#recovering = undefined;
          this.#events.emit("restored", { recoveredMs });
          break;
        }
        case "camera": {
          const phase = event.phase === "progress" ? "" : event.phase;
          this.#events.emit(`${event.change}${phase}` as "move", { user: event.user });
          break;
        }
        case "animationend":
          this.#animations.get(event.id)?.(!event.interrupted);
          this.#animations.delete(event.id);
          break;
        case "tilerequest":
          this.#events.emit("tilerequest", { source: event.source, z: event.z, x: event.x, y: event.y });
          break;
        case "click": case "dblclick": {
          if (!this.#events.has(event.type)) break;
          const point: Point = [event.x, event.y];
          const button = event.type === "click" ? event.button : undefined;
          this.#events.emit(event.type, { point, lngLat: this.unproject(point), button });
          break;
        }
      }
    }
  }

  #placeAnnotations(): void {
    if (!this.#stats || !this.annotations.size) return;
    this.annotations.place((points) => this.#host.core.projectPoints(points));
  }

  #error(e: unknown): void {
    const message = String(e instanceof Error ? e.message : e);
    if (this.#events.has("error")) this.#events.emit("error", { message });
    else console.error(`orbis: ${message}`);
  }
}

const CORE = new URL("../pkg/core_bg.wasm", import.meta.url);

/** Runs `f` in a task of its own, soon: a message, not a timer, which
 * browsers delay. */
const afterTask = (() => {
  const channel = new MessageChannel();
  const queued: (() => void)[] = [];
  channel.port1.onmessage = () => queued.shift()?.();
  return (f: () => void) => {
    queued.push(f);
    channel.port2.postMessage(null);
  };
})();

/** Why WebGPU gives no adapter here, if it doesn't. */
async function noWebgpu(): Promise<string | undefined> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return "this browser has none";
  try {
    if (await gpu.requestAdapter()) return undefined;
    return "it found no GPU here";
  } catch (e) {
    return `it failed: ${e instanceof Error ? e.message : e}`;
  }
}

/** Creates a map filling `options.container`. Resolves once it can take calls; its style loads on.
 * Rejects, saying why, where the browser has no WebGPU. */
export async function createMap(options: MapOptions): Promise<OrbisMap> {
  // The style's first fetches start, and core.wasm downloads and compiles,
  // while WebGPU is checked.
  preloadStyle(options.style, options.pixelRatio ?? devicePixelRatio);
  const module = WebAssembly.compileStreaming(fetch(CORE));
  module.catch(() => {});
  const missing = await noWebgpu();
  if (missing) throw new Error(`Orbis needs WebGPU, and ${missing}`);
  const frame = document.createElement("div");
  Object.assign(frame.style, { position: "relative", width: "100%", height: "100%", overflow: "hidden" });
  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, { display: "block", width: "100%", height: "100%" });
  const overlay = document.createElement("div");
  Object.assign(overlay.style, { position: "absolute", inset: "0", pointerEvents: "none" });
  frame.append(canvas, overlay);
  options.container.append(frame);
  const ratio = options.pixelRatio ?? devicePixelRatio;
  canvas.width = Math.round(canvas.clientWidth * ratio);
  canvas.height = Math.round(canvas.clientHeight * ratio);
  const style = typeof options.style === "string"
    ? options.style
    : `data:application/json,${encodeURIComponent(JSON.stringify(options.style))}`;
  // The map exists once its host does: until then its callbacks wait.
  let map: OrbisMap | undefined;
  const core = {
    fadeDuration: options.fadeDuration, crossSourceCollisions: options.crossSourceCollisions,
    localIdeographFontFamily: options.localIdeographFontFamily, antialias: options.antialias,
    terrainSkirts: options.terrainSkirts,
  };
  const host = await startMain(module, canvas, style, core, options.jobWorkers ?? defaultJobWorkers(), {
    wake: () => map?.wake(),
    error: (e) => (map ? map.fail(e) : console.error(e)),
    setup: (core) => {
      if (options.constraints) core.setConstraints(options.constraints);
      if (options.camera) core.jumpTo(options.camera);
    },
  });
  host.resize(canvas.width, canvas.height, ratio);
  map = new OrbisMap(options.container, frame, canvas, overlay, host, options.interactive ?? true,
    options.pixelRatio);
  return map;
}
