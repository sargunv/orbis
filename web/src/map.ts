// The map: one API over both thread modes (plan §14.3). The camera, its
// gestures and synchronous queries live on this thread in the ui role;
// drawing happens here or in a render worker.

import init, { UiRole } from "../pkg/core.js";
import { Annotations } from "./annotations.ts";
import { Emitter, type Listener } from "./events.ts";
import { ackCamera, ackGeneration, ackIdle, ackLost, frameLoop, frameStats, type FrameAck } from "./frames.ts";
import { probeWorker, startMain, startWorker, type Host } from "./hosts.ts";
import { attachInput } from "./input.ts";
import { defaultJobWorkers } from "./workers.ts";
import { transaction, type FeatureTarget, type Transaction } from "./transaction.ts";
import type {
  Bounds, Camera, CameraOptions, Constraints, EaseOptions, Easing, Feature, FitOptions, FlyOptions,
  FrameStats, Hit, LngLat, MapEvents, MapOptions, Point, QueryGeometry, QueryOptions,
  SourceQueryOptions, StyleImageData, StyleImageOptions,
} from "./types.ts";

const NAN2 = [NaN, NaN];
const CHANGES = ["move", "zoom", "rotate", "pitch"];
const PHASES = ["start", "", "end"];
function targetArray(o: CameraOptions): Float64Array {
  return new Float64Array([
    ...(o.center ?? NAN2), o.zoom ?? NaN, o.bearing ?? NaN, o.pitch ?? NaN,
    ...(o.around ?? NAN2), ...(o.offset ?? [0, 0]), o.roll ?? NaN, o.elevation ?? NaN,
  ]);
}
function targetFrom(a: ArrayLike<number>): CameraOptions {
  const known = (v: number) => (Number.isNaN(v) ? undefined : v);
  const target: CameraOptions = {
    center: [a[0], a[1]], zoom: known(a[2]), bearing: known(a[3]), pitch: known(a[4]),
    roll: known(a[9]), elevation: known(a[10]),
  };
  for (const key of Object.keys(target) as (keyof CameraOptions)[]) if (target[key] === undefined) delete target[key];
  return target;
}
/** An image's RGBA bytes, as the wasm map takes them. */
const rgba = (image: StyleImageData) =>
  new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength);
const easing = (e?: Easing) => (e === "linear" ? new Float64Array() : e ? new Float64Array(e) : undefined);

/** The camera: its state, moves and limits. */
export class CameraControl {
  #ui: UiRole;
  #changed: () => void;
  #animations: Map<number, (arrived: boolean) => void>;

  /** @internal */
  constructor(ui: UiRole, changed: () => void, animations: Map<number, (arrived: boolean) => void>) {
    this.#ui = ui;
    this.#changed = changed;
    this.#animations = animations;
  }

  /** The camera now: synchronous, on this thread, wherever the map draws. */
  get current(): Camera {
    const [lon, lat, , bearing, pitch, , zoom, elevation, roll] = this.#ui.camera();
    return { center: [lon, lat], zoom, bearing, pitch, roll, elevation };
  }

  /** Whether the center stands on the terrain at rest (GL JS
   * `centerClampedToGround`, true by default). Off, it keeps the elevation
   * it is given, and the camera orbits a point in the air. */
  get centerClampedToGround(): boolean { return this.#ui.centerClampedToGround(); }
  set centerClampedToGround(clamped: boolean) {
    this.#ui.setCenterClampedToGround(clamped);
    this.#changed();
  }

  /** Puts the center this many metres above sea level at once (GL JS
   * `setCenterElevation`); while clamped to the ground, the terrain takes it back. */
  setCenterElevation(elevation: number): void {
    this.#ui.setCenterElevation(elevation);
    this.#changed();
  }

  /** Whether an animation or a gesture is moving the camera. */
  get moving(): boolean { return this.#ui.isMoving(); }

  jumpTo(options: CameraOptions): void {
    this.#ui.jumpTo(targetArray(options));
    this.#changed();
  }

  /** Resolves true on arrival, false when interrupted. */
  easeTo(options: EaseOptions): Promise<boolean> {
    const id = this.#ui.easeTo(targetArray(options), options.duration ?? 500, easing(options.easing), performance.now());
    return this.#track(id);
  }

  /** Flies along GL JS's zoom-out curve; resolves true on arrival, false when interrupted. */
  flyTo(options: FlyOptions): Promise<boolean> {
    const o = options;
    const fly = new Float64Array([o.curve ?? NaN, o.speed ?? NaN, o.screenSpeed ?? NaN, o.duration ?? NaN,
      o.maxDuration ?? NaN, o.minZoom ?? NaN]);
    const id = this.#ui.flyTo(targetArray(options), fly, easing(options.easing), performance.now());
    return this.#track(id);
  }

  /** The camera that shows `bounds`, or null when the padding leaves no room. */
  cameraForBounds(bounds: Bounds, options: FitOptions = {}): CameraOptions | null {
    const p = typeof options.padding === "number"
      ? { top: options.padding, right: options.padding, bottom: options.padding, left: options.padding }
      : options.padding ?? {};
    const target = this.#ui.cameraForBounds(new Float64Array(bounds), new Float64Array([
      p.top ?? 0, p.right ?? 0, p.bottom ?? 0, p.left ?? 0, ...(options.offset ?? [0, 0]),
      options.maxZoom ?? NaN, options.bearing ?? NaN, options.pitch ?? NaN,
    ]));
    return target ? targetFrom(target) : null;
  }

  /** Flies (or, with `linear`, eases) to show `bounds`; resolves false when it doesn't fit or is interrupted. */
  fitBounds(bounds: Bounds, options: FitOptions & FlyOptions & { linear?: boolean } = {}): Promise<boolean> {
    const target = this.cameraForBounds(bounds, options);
    if (!target) return Promise.resolve(false);
    const move = { ...options, ...target, offset: undefined };
    return options.linear ? this.easeTo(move) : this.flyTo(move);
  }

  /** Stops the running animation; its promise resolves false. */
  stop(): void {
    this.#ui.stop();
    this.#changed();
  }

  setConstraints(c: Constraints): void {
    this.#ui.setConstraints(c.minZoom ?? -2, c.maxZoom ?? 22, c.minPitch ?? 0, c.maxPitch ?? 85,
      c.bounds ? new Float64Array(c.bounds) : undefined);
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
  #ui: UiRole;
  #host!: Host;
  #events = new Emitter<MapEvents>(() => this.#wantFrames());
  #loop: ReturnType<typeof frameLoop>;
  #pushed = -1;
  #dirty = true;
  #latest: FrameAck | undefined;
  #latestStats: FrameStats | undefined;
  /** Whether the page wants every frame acknowledged: annotations or `render` listeners. */
  #everyFrame = false;
  #idle = false;
  #animations = new Map<number, (arrived: boolean) => void>();
  #detach: (() => void)[] = [];
  #canvas: HTMLCanvasElement;
  #interactive = true;
  #detachInput: (() => void) | undefined;
  #observer: ResizeObserver | undefined;
  /** A device loss's start, while the host recovers. */
  #lostAt: number | undefined;
  /** The last device loss's start, until the new device draws. */
  #recovering: number | undefined;

  /** @internal */
  constructor(container: HTMLElement, canvas: HTMLCanvasElement, overlay: HTMLElement, ui: UiRole) {
    this.container = container;
    this.#canvas = canvas;
    this.#ui = ui;
    this.#loop = frameLoop((now) => this.#tick(now), (e) => this.#error(e));
    const request = () => this.#loop.request();
    this.camera = new CameraControl(ui, request, this.#animations);
    this.annotations = new Annotations(overlay, () => {
      this.#wantFrames();
      this.#placeAnnotations();
    });
  }

  /** @internal */
  start(host: Host, interactive: boolean): void {
    this.#host = host;
    if (this.#everyFrame) host.acknowledge?.(true);
    this.#interactive = interactive;
    this.#observer = new ResizeObserver(() => this.#resize());
    this.#attach();
    this.#detach.push(() => { this.#detachInput?.(); this.#observer?.disconnect(); });
    this.#loop.request();
  }

  /** The canvas the map draws into and takes input from. */
  get canvas(): HTMLCanvasElement { return this.#canvas; }

  #attach(): void {
    if (this.#interactive) this.#detachInput = attachInput(this.#canvas, this.#ui, () => this.#loop.request());
    this.#observer?.observe(this.#canvas);
  }

  /** Loses the GPU on purpose, to exercise recovery (main-thread maps);
   * with `switchTier`, a WebGPU map recovers on WebGL2. */
  loseDevice(switchTier = false): void { this.#host.loseDevice?.(switchTier); }

  /** Where the map draws: "worker" (a render worker) or "main". */
  get thread(): "main" | "worker" { return this.#host.thread; }

  /** The last acknowledged frame's counters. A render worker acknowledges
   * frames the page needs (with annotations or `render` listeners, every one). */
  get stats(): FrameStats | undefined {
    if (this.#latest && !this.#latestStats) this.#latestStats = frameStats(this.#latest);
    return this.#latestStats;
  }

  /** The wasm map in main-thread mode, for benches and tests; unstable. */
  get core() { return this.#host.core; }

  on<K extends keyof MapEvents>(type: K, listener: Listener<MapEvents[K]>): () => void {
    return this.#events.on(type, listener);
  }
  off<K extends keyof MapEvents>(type: K, listener: Listener<MapEvents[K]>): void {
    this.#events.off(type, listener);
  }
  once<K extends keyof MapEvents>(type: K): Promise<MapEvents[K]> { return this.#events.once(type); }
  /** `for await (const e of map.events("idle"))`. */
  events<K extends keyof MapEvents>(type: K): AsyncGenerator<MapEvents[K]> { return this.#events.events(type); }

  /** The pixel showing a point now, or null behind the eye. Synchronous in both thread modes. */
  project(lngLat: LngLat): Point | null {
    const p = this.#ui.project(lngLat[0], lngLat[1]);
    return p ? [p[0], p[1]] : null;
  }

  /** The ground point under a pixel now, or null above the horizon. */
  unproject(point: Point): LngLat | null {
    const g = this.#ui.unproject(point[0], point[1]);
    return g ? [g[0], g[1]] : null;
  }

  /** The topmost feature under a pixel, synchronously, from the last published frame. */
  hitTest(point: Point): Hit | null {
    return (this.#host.hitTest ?? ((x, y) => this.#ui.hitTest(x, y) as Hit | null))(point[0], point[1]);
  }

  /** Every feature drawn under a point or box (all on screen without one), topmost first. */
  async queryRenderedFeatures(geometry?: QueryGeometry, options: QueryOptions = {}): Promise<Feature[]> {
    const box = geometry === undefined
      ? [0, 0, this.canvas.clientWidth, this.canvas.clientHeight]
      : typeof geometry[0] === "number" ? [geometry[0], geometry[1], geometry[0], geometry[1]] as number[]
        : [...(geometry as [Point, Point])[0], ...(geometry as [Point, Point])[1]];
    const json = await this.#host.request("queryRendered", new Float64Array(box),
      options.layers ? JSON.stringify(options.layers) : undefined) as string;
    return JSON.parse(json);
  }

  /** Every feature of a source in its loaded tiles that matches a filter. */
  async querySourceFeatures(source: string, options: SourceQueryOptions = {}): Promise<Feature[]> {
    const json = await this.#host.request("querySource", source, options.sourceLayer,
      options.filter === undefined ? undefined : JSON.stringify(options.filter)) as string;
    return JSON.parse(json);
  }

  /** Applies GL JS map calls as one transaction; resolves to the style
   * generation whose frames show it in full (`render` events carry it). */
  async edit(build: (tx: Transaction) => void): Promise<number> {
    const generation = await this.#host.edit(transaction(build));
    this.#dirty = true;
    this.#idle = false;
    this.#loop.request();
    return generation;
  }
  setFeatureState(feature: FeatureTarget, state: Record<string, unknown>): Promise<number> {
    return this.edit((tx) => tx.setFeatureState(feature, state));
  }
  removeFeatureState(feature: Partial<FeatureTarget>, key?: string): Promise<number> {
    return this.edit((tx) => tx.removeFeatureState(feature, key));
  }
  setGlobalState(name: string, value: unknown): Promise<number> {
    return this.edit((tx) => tx.setGlobalStateProperty(name, value));
  }

  /** Adds an image to the style (GL JS `addImage`) for icons, text and
   * patterns; rejects when it has one by that name. */
  async addImage(name: string, image: StyleImageData, options: StyleImageOptions = {}): Promise<void> {
    await this.#host.request("addImage", name, image.width, image.height, rgba(image), JSON.stringify(options));
    this.#changed();
  }
  /** Replaces an image's pixels, of the same size (GL JS `updateImage`). */
  async updateImage(name: string, image: StyleImageData): Promise<void> {
    await this.#host.request("updateImage", name, image.width, image.height, rgba(image));
    this.#changed();
  }
  /** Removes an image from the style (GL JS `removeImage`). */
  async removeImage(name: string): Promise<void> {
    await this.#host.request("removeImage", name);
    this.#changed();
  }
  /** Whether the style has an image by this name, from its sprites or added. */
  async hasImage(name: string): Promise<boolean> {
    return await this.#host.request("hasImage", name) as boolean;
  }
  #changed() {
    this.#dirty = true;
    this.#idle = false;
    this.#loop.request();
  }

  /** The style as edits have left it. */
  async getStyle(): Promise<object> {
    return JSON.parse(await this.#host.request("getStyle") as string);
  }

  /** What the labels on screen say, for assistive technology (plan §13); null before labels place. */
  async semanticSnapshot(): Promise<object | null> {
    const json = await this.#host.request("semanticSnapshot") as string | undefined;
    return json ? JSON.parse(json) : null;
  }

  /** Stops drawing and releases the map's workers and DOM. */
  remove(): void {
    for (const detach of this.#detach) detach();
    this.#host.destroy();
    this.canvas.parentElement?.remove();
  }

  #resize(): void {
    const ratio = devicePixelRatio;
    const [w, h] = [this.canvas.clientWidth, this.canvas.clientHeight];
    this.#ui.resize(w, h);
    if (this.#host.thread === "main") {
      this.canvas.width = Math.round(w * ratio);
      this.canvas.height = Math.round(h * ratio);
    }
    this.#host.resize(Math.round(w * ratio), Math.round(h * ratio), ratio);
    this.#dirty = true;
    this.#loop.request();
  }

  #tick(now: number): boolean {
    // Moves before the host starts reach it with the first frame.
    if (!this.#host) return false;
    const moving = this.#ui.frame(now);
    this.#drainUi();
    // Until the app or the user moves it, the map's camera follows its style.
    const revision = this.#ui.revision();
    if (this.#ui.modified() && revision !== this.#pushed) {
      this.#pushed = revision;
      this.#host.setCamera(new Float64Array(this.#ui.camera()));
      this.#dirty = true;
      this.#idle = false;
    }
    let more = moving;
    if (this.#host.render && this.#dirty && this.#lostAt === undefined) {
      const ack = this.#host.render();
      if (ackLost(ack)) {
        this.#recover();
        return false;
      }
      this.#dirty = !ackIdle(ack);
      this.#frame(ack);
      more ||= this.#dirty;
    }
    if (moving) this.#idle = false;
    else this.#maybeIdle();
    return more;
  }

  /** Gets a new GPU after a loss (plan §8.8) and draws on. */
  #recover(): void {
    if (this.#lostAt !== undefined || !this.#host.recover) return;
    this.#lostAt = performance.now();
    const replace = (canvas: HTMLCanvasElement) => {
      this.#detachInput?.();
      this.#observer?.unobserve(this.#canvas);
      this.#canvas.replaceWith(canvas);
      this.#canvas = canvas;
      this.#attach();
    };
    this.#host.recover(replace, new Float64Array(this.#ui.camera())).then(() => {
      this.#recovering = this.#lostAt;
      this.#lostAt = undefined;
      this.#pushed = -1;
      this.#dirty = true;
      this.#loop.request();
    }, (e) => this.#error(`the GPU was lost and no other could be made: ${e}`));
  }

  /** @internal A frame drawn here or acknowledged by the render worker. */
  frame(ack: FrameAck): void { this.#frame(ack); this.#maybeIdle(); }

  /** @internal A failure where the map draws. */
  fail(e: unknown): void { this.#error(e); }

  /** @internal The core has new data to draw (main-thread mode). */
  wake(): void {
    this.#dirty = true;
    this.#loop.request();
  }

  #frame(ack: FrameAck): void {
    this.#latest = ack;
    this.#latestStats = undefined;
    if (ack.hits) this.#ui.updateHits(ack.hits);
    if (ack.elevation) {
      // The camera stands on the new terrain: send it with the next tick.
      this.#ui.updateElevation(ack.elevation);
      this.#loop.request();
    }
    this.#ui.advanceHits(ackGeneration(ack));
    if (!ackIdle(ack)) this.#idle = false;
    // Loads and projection edits move the camera's surface, before a
    // style's camera is adopted on it.
    const surface = ack.events?.findLast((e) => e.type === "styledata");
    if (surface) this.#ui.setSurface(surface.surface);
    for (const event of ack.events ?? []) {
      switch (event.type) {
        case "styleload":
          if (event.camera && this.#ui.adoptStyleCamera(event.camera)) this.#loop.request();
          this.#events.emit("styleload", {});
          break;
        case "styledata": this.#events.emit("styledata", { generation: event.generation }); break;
        case "sourcedata": this.#events.emit("sourcedata", { source: event.source, loaded: event.loaded }); break;
        case "load": this.#events.emit("load", {}); break;
        case "capability": {
          const recoveredMs = this.#recovering === undefined ? undefined : performance.now() - this.#recovering;
          this.#recovering = undefined;
          this.#events.emit("capability", { tier: event.tier, recoveredMs });
          break;
        }
        case "error": this.#error(event.message); break;
        case "styleimagemissing": this.#events.emit("styleimagemissing", { id: event.id }); break;
        // Idle needs the camera at rest too; see #maybeIdle.
        case "idle": break;
      }
    }
    if (this.#events.has("render")) this.#events.emit("render", this.stats!);
    this.#placeAnnotations();
  }

  /** Tells a render worker whether to acknowledge every frame. */
  #wantFrames(): void {
    const every = (this.annotations?.size ?? 0) > 0 || this.#events.has("render");
    if (every === this.#everyFrame) return;
    this.#everyFrame = every;
    this.#host?.acknowledge?.(every);
  }

  /** GL JS's `idle`: the last frame drawn is idle and the camera is at rest. */
  #maybeIdle(): void {
    const latest = this.#latest;
    if (this.#idle || !latest || !ackIdle(latest) || this.#ui.isMoving() || (this.#host.render && this.#dirty)) return;
    // The frame must show the camera as it is now: a worker's frame for an
    // older camera can arrive after this one was sent.
    if (this.#ui.modified()) {
      const camera = this.#ui.camera();
      const shown = ackCamera(latest);
      for (let i = 0; i < 6; i++) if (camera[i] !== shown[i]) return;
    }
    this.#idle = true;
    this.#events.emit("idle", {});
  }

  #placeAnnotations(): void {
    if (!this.#latest || !this.annotations.size) return;
    const camera = ackCamera(this.#latest);
    this.annotations.place((points) => this.#ui.projectWith(camera, points));
  }

  /** The ui role's events, packed four numbers each (`UiRole::take_events`). */
  #drainUi(): void {
    const events = this.#ui.takeEvents();
    for (let i = 0; i < events.length; i += 4) {
      const [kind, a, b, c] = [events[i], events[i + 1], events[i + 2], events[i + 3]];
      if (kind === 12) {
        this.#animations.get(a)?.(b === 0);
        this.#animations.delete(a);
      } else if (kind === 13 || kind === 14) {
        const type = kind === 13 ? "click" : "dblclick";
        if (!this.#events.has(type)) continue;
        const point: Point = [a, b];
        this.#events.emit(type, { point, lngLat: this.unproject(point), button: kind === 13 ? c : undefined });
      } else {
        const type = `${CHANGES[Math.floor(kind / 3)]}${PHASES[kind % 3]}` as "move";
        if (this.#events.has(type)) this.#events.emit(type, { user: a !== 0 });
      }
    }
  }

  #error(e: unknown): void {
    const message = String(e instanceof Error ? e.message : e);
    if (this.#events.has("error")) this.#events.emit("error", { message });
    else console.error(`orbis: ${message}`);
  }
}

const CORE = new URL("../pkg/core_bg.wasm", import.meta.url);

/** Whether WebGPU gives an adapter here. */
async function webgpuAdapter(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  try {
    return !!gpu && !!await gpu.requestAdapter();
  } catch {
    return false;
  }
}

/** Empties the HTTP cache maps with `httpCache` keep (`MapOptions.httpCache`). */
export async function clearHttpCache(): Promise<void> {
  await caches.delete("orbis-http");
}

/** Creates a map filling `options.container`. Resolves once it can take calls; its style loads on. */
export async function createMap(options: MapOptions): Promise<OrbisMap> {
  const renderer = options.renderer ?? "auto";
  const tier = renderer === "webgl2" || (renderer === "auto" && !await webgpuAdapter()) ? "compat" : "modern";
  const module = WebAssembly.compileStreaming(fetch(CORE));
  const frame = document.createElement("div");
  Object.assign(frame.style, { position: "relative", width: "100%", height: "100%", overflow: "hidden" });
  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, { display: "block", width: "100%", height: "100%" });
  const overlay = document.createElement("div");
  Object.assign(overlay.style, { position: "absolute", inset: "0", pointerEvents: "none" });
  frame.append(canvas, overlay);
  options.container.append(frame);
  const ratio = devicePixelRatio;
  canvas.width = Math.round(canvas.clientWidth * ratio);
  canvas.height = Math.round(canvas.clientHeight * ratio);

  // WebGL2 draws on this thread (its module is the renderer role's alone).
  let thread = tier === "compat" ? "main" : options.thread ?? "main";
  let probed: Awaited<ReturnType<typeof probeWorker>> | undefined;
  const offscreen = typeof OffscreenCanvas !== "undefined" && "transferControlToOffscreen" in canvas;
  if (thread !== "main") {
    probed = offscreen ? await probeWorker() : undefined;
    const works = offscreen && probed?.ok;
    if (thread === "worker" && !works) {
      throw new Error(`worker mode needs OffscreenCanvas and WebGPU in workers (${probed?.reason ?? "no OffscreenCanvas"})`);
    }
    thread = works ? "worker" : "main";
    if (thread === "main") probed?.worker.terminate();
  }
  const compiled = await module;
  await init({ module_or_path: compiled });
  const ui = new UiRole(canvas.clientWidth, canvas.clientHeight);
  ui.setFeel(options.feel ?? "desktop");
  const map = new OrbisMap(options.container, canvas, overlay, ui);
  if (options.constraints) map.camera.setConstraints(options.constraints);
  if (options.camera) map.camera.jumpTo(options.camera);
  const style = typeof options.style === "string"
    ? options.style
    : `data:application/json,${encodeURIComponent(JSON.stringify(options.style))}`;
  const callbacks = {
    frame: (ack: FrameAck) => map.frame(ack),
    wake: () => map.wake(),
    error: (e: unknown) => map.fail(e),
  };
  const jobWorkers = options.jobWorkers ?? defaultJobWorkers();
  const httpCache = options.httpCache?.maxBytes ?? 0;
  const host = thread === "worker"
    ? startWorker(compiled, probed!.worker, canvas, style, jobWorkers, httpCache, callbacks)
    : await startMain(compiled, canvas, style, jobWorkers, httpCache, callbacks, tier);
  map.start(host, options.interactive ?? true);
  return map;
}
