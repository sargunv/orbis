// The map: one API over both thread modes (plan §14.3). The camera, its
// gestures and synchronous queries live on this thread in the ui role;
// drawing happens here or in a render worker.

import init, { UiRole } from "../pkg/core.js";
import { Annotations } from "./annotations.ts";
import { Emitter, type Listener } from "./events.ts";
import { frameLoop, type FrameAck } from "./frames.ts";
import { probeWorker, startMain, startWorker, type Host } from "./hosts.ts";
import { attachInput } from "./input.ts";
import { transaction, type FeatureTarget, type Transaction } from "./transaction.ts";
import type {
  Bounds, Camera, CameraOptions, Constraints, EaseOptions, Easing, Feature, FitOptions, FlyOptions,
  FrameStats, Hit, LngLat, MapEvents, MapOptions, Point, QueryGeometry, QueryOptions,
  SourceQueryOptions,
} from "./types.ts";

const NAN2 = [NaN, NaN];
function targetArray(o: CameraOptions): Float64Array {
  return new Float64Array([
    ...(o.center ?? NAN2), o.zoom ?? NaN, o.bearing ?? NaN, o.pitch ?? NaN,
    ...(o.around ?? NAN2), ...(o.offset ?? [0, 0]),
  ]);
}
function targetFrom(a: ArrayLike<number>): CameraOptions {
  const known = (v: number) => (Number.isNaN(v) ? undefined : v);
  const target: CameraOptions = { center: [a[0], a[1]], zoom: known(a[2]), bearing: known(a[3]), pitch: known(a[4]) };
  for (const key of Object.keys(target) as (keyof CameraOptions)[]) if (target[key] === undefined) delete target[key];
  return target;
}
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
    const [lon, lat, , bearing, pitch, , zoom] = this.#ui.camera();
    return { center: [lon, lat], zoom, bearing, pitch };
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
  readonly canvas: HTMLCanvasElement;
  readonly camera: CameraControl;
  readonly annotations: Annotations;
  #ui: UiRole;
  #host!: Host;
  #events = new Emitter<MapEvents>();
  #loop: ReturnType<typeof frameLoop>;
  #pushed = -1;
  #dirty = true;
  #latest: FrameAck | undefined;
  #idle = false;
  #animations = new Map<number, (arrived: boolean) => void>();
  #detach: (() => void)[] = [];

  /** @internal */
  constructor(container: HTMLElement, canvas: HTMLCanvasElement, overlay: HTMLElement, ui: UiRole) {
    this.container = container;
    this.canvas = canvas;
    this.#ui = ui;
    this.#loop = frameLoop((now) => this.#tick(now), (e) => this.#error(e));
    const request = () => this.#loop.request();
    this.camera = new CameraControl(ui, request, this.#animations);
    this.annotations = new Annotations(overlay, () => this.#placeAnnotations());
  }

  /** @internal */
  start(host: Host, interactive: boolean): void {
    this.#host = host;
    if (interactive) this.#detach.push(attachInput(this.canvas, this.#ui, () => this.#loop.request()));
    const observer = new ResizeObserver(() => this.#resize());
    observer.observe(this.canvas);
    this.#detach.push(() => observer.disconnect());
    this.#loop.request();
  }

  /** Where the map draws: "worker" (a render worker) or "main". */
  get thread(): "main" | "worker" { return this.#host.thread; }

  /** The last frame's counters. */
  get stats(): FrameStats | undefined { return this.#latest; }

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
    if (this.#host.render && this.#dirty) {
      const ack = this.#host.render();
      this.#dirty = !ack.idle;
      this.#frame(ack);
      more ||= this.#dirty;
    }
    if (moving) this.#idle = false;
    else this.#maybeIdle();
    return more;
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
    if (ack.hitSnapshot) this.#ui.updateHits(ack.hitSnapshot);
    this.#ui.advanceHits(ack.generation);
    if (!ack.idle) this.#idle = false;
    for (const event of ack.events) {
      switch (event.type) {
        case "styleload":
          this.#ui.setProjection(event.projection);
          if (event.camera && this.#ui.adoptStyleCamera(event.camera)) this.#loop.request();
          this.#events.emit("styleload", {});
          break;
        case "styledata": this.#events.emit("styledata", { generation: event.generation }); break;
        case "sourcedata": this.#events.emit("sourcedata", { source: event.source, loaded: event.loaded }); break;
        case "load": this.#events.emit("load", {}); break;
        case "error": this.#error(event.message); break;
        // Idle needs the camera at rest too; see #maybeIdle.
        case "idle": break;
      }
    }
    this.#events.emit("render", ack);
    this.#placeAnnotations();
  }

  /** GL JS's `idle`: the last frame drawn is idle and the camera is at rest. */
  #maybeIdle(): void {
    const latest = this.#latest;
    if (this.#idle || !latest?.idle || this.#ui.isMoving() || (this.#host.render && this.#dirty)) return;
    // The frame must show the camera as it is now: a worker's frame for an
    // older camera can arrive after this one was sent.
    if (this.#ui.modified()) {
      const camera = this.#ui.camera();
      for (let i = 0; i < 6; i++) if (camera[i] !== latest.camera[i]) return;
    }
    this.#idle = true;
    this.#events.emit("idle", {});
  }

  #placeAnnotations(): void {
    const camera = this.#latest?.camera;
    if (!camera) return;
    this.annotations.place((points) => new Float64Array(this.#ui.projectWith(camera, points)));
  }

  #drainUi(): void {
    for (const event of this.#ui.takeEvents() as { type: string; [key: string]: unknown }[]) {
      if (event.type === "animationend") {
        this.#animations.get(event.id as number)?.(!event.interrupted);
        this.#animations.delete(event.id as number);
      } else if (event.type === "click" || event.type === "dblclick") {
        const point: Point = [event.x as number, event.y as number];
        this.#events.emit(event.type, { point, lngLat: this.unproject(point), button: event.button as number | undefined });
      } else {
        this.#events.emit(event.type as "move", { user: event.user as boolean });
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

/** Creates a map filling `options.container`. Resolves once it can take calls; its style loads on. */
export async function createMap(options: MapOptions): Promise<OrbisMap> {
  const gpu = (navigator as Navigator & { gpu?: unknown }).gpu;
  if (!gpu) throw new Error("WebGPU is not available in this browser");
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

  let thread = options.thread ?? "main";
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
  const host = thread === "worker"
    ? startWorker(compiled, probed!.worker, canvas, style, callbacks)
    : await startMain(compiled, canvas, style, callbacks);
  map.start(host, options.interactive ?? true);
  return map;
}
