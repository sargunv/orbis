// The map itself: the wasm map with its frames, input, queries,
// snapshots and recovery from a lost GPU, drawing into the page's canvas
// on the page's thread. The page's side (map.ts) calls it and hears from
// it by reports: the map's state after each frame, with its events.

import {
  isModuleName,
  MODULE_NAMES,
  type Backend,
  type Build,
  type ModuleName,
} from "./builds.ts";
import { frameLoop } from "./frames.ts";
import { startHost, type Core, type CoreOptions, type Host } from "./hosts.ts";
import type {
  Camera,
  CameraOptions,
  Constraints,
  Degradation,
  EaseOptions,
  ElevationPortMessage,
  ElevationPortReply,
  ElevationPortShown,
  FitOptions,
  FlyOptions,
  FrameStats,
  FreeCamera,
  Gap,
  GestureMomentum,
  GestureOptions,
  GpuSubmission,
  Hit,
  HostElevationsAnswer,
  MapEvents,
  Padding,
  PointerEventData,
  TileLod,
} from "./types.ts";
import type { VideoEventObject } from "./videos.ts";
import type { Workers } from "./workers.ts";

/** What the page reads of the map without asking. */
export interface MapState {
  camera: Camera;
  free: FreeCamera;
  moving: boolean;
  clamped: boolean;
}

/** An event for the page: one of the map's events, or for the page itself
 * an animation's end (its promise settles) or a video request (videos.ts). */
export type EngineEvent =
  | { [K in keyof MapEvents]: { type: K; data: MapEvents[K] } }[keyof MapEvents]
  | { type: "animationend"; data: { id: number; arrived: boolean } }
  | { type: "video"; data: VideoEventObject };

/** What the map tells the page after a frame, or when its annotations or
 * box zoom change between frames. */
export interface Report {
  state: MapState;
  /** The frame's counters, after a frame. */
  stats?: FrameStats;
  events: EngineEvent[];
  /** The annotations' pixels, [x, y, …] in the order the page gave them. */
  points?: Float64Array;
  /** The box a Shift-drag zooms to, as corners, or null once it's gone;
   * absent when unchanged. */
  box?: number[] | null;
  /** Why the map stopped, once its wasm trapped or its GPU was lost for
   * good: the message its calls fail with from now on, which its `error`
   * event said. */
  stopped?: string;
  /** WebGPU, picked by `auto`, is lost for good (`why`, since `lostAt`):
   * the page starts the map again on WebGL 2, or stops this one. */
  fallback?: { why: string; lostAt: number };
}

/** Pixels the map read back: straight RGBA, or the canvas's bytes. */
export interface Pixels {
  pixels: Uint8Array;
  width: number;
  height: number;
}

/** The wasm map's events (`OrbisMap::take_events` in orb-web). */
type CoreEvent =
  | { type: "styleload" | "load" | "idle" }
  | { type: "styledata"; generation: number }
  | { type: "sourcedata"; source: string; loaded: boolean }
  | { type: "error" | "warning"; message: string }
  | { type: "styleimagemissing"; id: string }
  | { type: "restored" }
  | { type: "degraded"; gaps: Degradation[] }
  | {
      type: "camera";
      change: "move" | "zoom" | "rotate" | "pitch";
      phase: "start" | "progress" | "end";
      user: boolean;
    }
  | { type: "animationend"; id: number; interrupted: boolean }
  | { type: "click" | "dblclick" | "contextmenu"; x: number; y: number }
  | { type: "tilerequest"; source: string; z: number; x: number; y: number }
  | { type: "elevationsshown"; source: string; batch: number }
  | VideoEventObject;

/** The input the page forwards: the wasm map's input methods. */
export type InputKind =
  | "pointerDown"
  | "pointerMove"
  | "pointerUp"
  | "pointerCancel"
  | "wheel"
  | "pinch"
  | "key"
  | "blur";

/** How the map starts. */
export interface EngineSetup {
  canvas: HTMLCanvasElement;
  style: string;
  options: CoreOptions;
  camera?: CameraOptions;
  constraints?: Constraints;
  /** Physical width and height and device pixels per CSS pixel. */
  size: [number, number, number];
}

/** A wasm-bindgen object's getters as a plain object, which outlives
 * it. */
function plain<T extends object>(object: T): T {
  const out: Record<string, unknown> = {};
  for (const [key, d] of Object.entries(
    Object.getOwnPropertyDescriptors(Object.getPrototypeOf(object)),
  )) {
    if (!d.get) continue;
    const value = (object as Record<string, unknown>)[key];
    out[key] =
      value !== null && typeof value === "object" && "free" in value
        ? plain(value)
        : value;
  }
  return out as T;
}

/** A device that lasted this long before its loss ends a run of losses. */
const STABLE_MS = 30_000;
/** How long after each loss of a run a new device is made. */
const BACKOFF_MS = [0, 250, 1000];
/** How long a map that hasn't been idle waits to fetch its API module. */
const API_AT_MOST = 5000;
/** How long a feature module that failed to load waits to be asked for
 * again. */
const MODULE_RETRY = 2000;

/** The calls that run the API's module (queries, edits, the style's
 * document, images, snapshots, fitting bounds, host-fed heights, video
 * controls, GPU timings), which waits for it if it isn't in yet. */
const API_CALLS = new Set<string>([
  "hitTest",
  "queryRendered",
  "querySource",
  "queryCluster",
  "getStyle",
  "getLayer",
  "getSource",
  "layerIds",
  "sourceIds",
  "hasImage",
  "setStyleUrl",
  "edit",
  "addImage",
  "updateImage",
  "removeImage",
  "snapshot",
  "readPixels",
  "cameraForBounds",
  "hostElevations",
  "reloadHostTiles",
  "retryFailedTiles",
  "controlVideo",
  "gpuTimings",
]);

export class Engine {
  #host: Host;
  #canvas: HTMLCanvasElement;
  #report: (report: Report) => void;
  #loop: ReturnType<typeof frameLoop>;
  #events: EngineEvent[] = [];
  /** Ports apps' workers feed heights through. */
  #ports: MessagePort[] = [];
  #stats: FrameStats | undefined;
  #snapshotting = false;
  #snapshots: Promise<unknown> = Promise.resolve();
  /** A device loss's start, while the host recovers. */
  #lostAt: number | undefined;
  /** The last device loss's start, until the new device draws. */
  #recovering: number | undefined;
  /** The gaps warned of on the console: each once per map. */
  #warnedGaps = new Set<Gap>();
  /** What the map draws otherwise than on WebGPU now. */
  #gaps: Degradation[] = [];
  /** When the device was made, and the losses in the current run. */
  #deviceAt = performance.now();
  #losses = 0;
  #removed = false;
  /** Why the map stopped: every call fails with it. */
  #stopped: string | undefined;
  /** The annotations' points, [lon, lat, …]. */
  #annotations: Float64Array = new Float64Array(0);
  /** The box zoom's corners last reported, as text. */
  #box = "null";
  /** The latest input's time, on the page's clock. */
  #inputTime: number | undefined;
  /** The state last read. */
  #state: MapState | undefined;
  /** The last frame's time: a frame never goes back before it. */
  #shown = -Infinity;
  /** The time a test runner's clock gives frames. */
  #frameClock: number | undefined;
  /** The feature modules asked for (builds.ts). */
  #asked = new Set<ModuleName>();
  /** Those whose last load failed, reported once until one loads. */
  #failed = new Set<ModuleName>();
  /** Calls waiting for the API's module, in order: later calls wait
   * behind them, so the map takes calls in the order they came. */
  #waiting: Promise<unknown> | undefined;

  private constructor(
    host: Host,
    setup: EngineSetup,
    report: (report: Report) => void,
  ) {
    this.#host = host;
    this.#canvas = setup.canvas;
    this.#report = report;
    // The map's one beat: all its work (what arrived, the camera, loads,
    // labels, the draw) runs in a frame on the display's refresh, within
    // the frame's budget (orb-runtime's `Budget`). Input, calls and
    // wakes between frames only queue and ask for the next.
    this.#loop = frameLoop(
      (now) => this.#tick(now),
      (e) => this.#fail(e),
    );
  }

  /** The map in `setup.canvas` once `compiling` has compiled its build,
   * with `workers` (spawned on the same build); `report` hears from it. */
  static async start(
    compiling: Promise<Build>,
    workers: Workers,
    setup: EngineSetup,
    report: (report: Report) => void,
  ): Promise<Engine> {
    // The map exists once its host does: until then its callbacks wait.
    let engine: Engine | undefined;
    const host = await startHost(
      compiling,
      workers,
      setup.canvas,
      setup.style,
      setup.options,
      {
        wake: () => engine?.wake(),
        error: (e) => (engine ? engine.#fail(e) : console.error(e)),
        // Before the map exists, the trapping call's error fails its start.
        trapped: (e) => {
          if (engine) engine.#trapped(e);
        },
        setup: (core) => {
          if (setup.constraints) core.setConstraints(setup.constraints);
          if (setup.camera) core.jumpTo(setup.camera);
        },
      },
    );
    host.resize(...setup.size);
    engine = new Engine(host, setup, report);
    // WebGPU expected to draw didn't: the page hears why.
    if (host.unexpected)
      engine.#events.push({
        type: "warning",
        data: {
          message: `the map draws with WebGL 2. WebGPU: ${host.unexpected}`,
        },
      });
    // The first frame takes the style in and asks for its tiles.
    engine.wake();
    // The API's module comes once the map is idle, out of the first
    // frames' way, or after a while for a map that doesn't settle.
    const started = engine;
    setTimeout(() => started.#want(["api"]).catch(() => {}), API_AT_MOST);
    return engine;
  }

  get #core(): Core {
    return this.#host.core;
  }

  /** Calls `method`: at once, or once the API's module is in, for a call
   * that needs it, and any call made while one waits. */
  call(method: string, args: unknown[]): Promise<unknown> {
    if (this.#stopped !== undefined)
      return Promise.reject(new Error(this.#stopped));
    const run = () =>
      (this[method as keyof Engine] as (...a: unknown[]) => unknown)(...args);
    const needs = API_CALLS.has(method) && !this.#host.modules().has("api");
    if (!needs && !this.#waiting) {
      try {
        return Promise.resolve(run());
      } catch (e) {
        return Promise.reject(e);
      }
    }
    const result = (this.#waiting ?? this.#want(["api"])).then(run);
    const waiting = result.catch(() => {});
    this.#waiting = waiting;
    void waiting.then(() => {
      if (this.#waiting === waiting) this.#waiting = undefined;
    });
    return result;
  }

  /** Loads feature modules in every thread of the map, then tells the
   * map which it has; a module that fails to load is an `error`. */
  #want(names: readonly string[]): Promise<void> {
    const loads = names.filter(isModuleName).map((name) => {
      this.#asked.add(name);
      return this.#host.loadModules([name]).then(
        () => {
          this.#failed.delete(name);
          if (this.#removed) return;
          const loaded = this.#host.modules();
          let bits = 0;
          MODULE_NAMES.forEach((name, i) => {
            if (loaded.has(name)) bits |= 1 << i;
          });
          this.#core.setModules(bits);
        },
        (e) => {
          // Asked for again by the frame after a while, if it still
          // wants it, not by every frame; the first failure of a run is
          // the error.
          setTimeout(() => {
            this.#asked.delete(name);
            if (!this.#removed) this.#request();
          }, MODULE_RETRY);
          if (!this.#failed.has(name))
            this.#fail(`core-${name}.wasm didn't load: ${e}`);
          this.#failed.add(name);
          throw e;
        },
      );
    });
    return Promise.all(loads).then(() => {});
  }

  /** The GPU API the map draws with. */
  get backend(): Backend {
    return this.#host.backend;
  }

  /** The feature modules the map has. */
  modules(): ReadonlySet<ModuleName> {
    return this.#host.modules();
  }

  /** What the page reads without asking. */
  state(): MapState {
    // A snapshot's end and a recovery hold the wasm map while they wait:
    // the state stays as it was, as it does once the map is gone.
    if (
      this.#state &&
      (this.#removed || this.#snapshotting || this.#lostAt !== undefined)
    )
      return this.#state;
    const core = this.#core;
    const c = core.camera();
    const camera: Camera = {
      center: [c.lon, c.lat],
      zoom: c.zoom,
      bearing: c.bearing,
      pitch: c.pitch,
      roll: c.roll,
      elevation: c.elevation,
      fov: c.fov,
      padding: c.padding as Padding,
    };
    c.free();
    this.#state = {
      camera,
      free: core.freeCamera() as FreeCamera,
      moving: core.isMoving(),
      clamped: core.centerClampedToGround(),
    };
    return this.#state;
  }

  // Input, at the page's event times.

  input(kind: InputKind, args: (number | string)[], time: number): void {
    if (this.#removed || this.#lostAt !== undefined || this.#snapshotting)
      return;
    this.#inputTime = time;
    const core = this.#core as unknown as Record<
      InputKind,
      (...a: unknown[]) => boolean | undefined
    >;
    // Input applies as it arrives and shows from the next frame: drawn
    // now, off the display's beat, a frame puts the frames after it out of
    // step with the display, which drops one of them.
    const used = core[kind](...args, time);
    // A release may end a click, which the next frame reports.
    if (used || kind === "blur" || kind === "pointerUp") this.#request();
    // The box shows and goes between frames too.
    const corners = this.#core.boxZoom();
    const box = corners ? Array.from(corners) : null;
    if (String(box) !== this.#box) {
      this.#box = String(box);
      this.#send({ box });
    }
  }

  // The camera.

  jumpTo(options: CameraOptions): void {
    this.#core.jumpTo(options);
    this.#request();
  }
  jumpToFree(camera: Partial<FreeCamera>): void {
    this.#core.jumpToFree(camera);
    this.#request();
  }
  setCenterClampedToGround(clamped: boolean): void {
    this.#core.setCenterClampedToGround(clamped);
    this.#request();
  }
  setGestures(options: GestureOptions): void {
    this.#core.setGestures(options);
    this.#request();
  }
  setCenterElevation(elevation: number): void {
    this.#core.setCenterElevation(elevation);
    this.#request();
  }
  /** The animation's id. */
  easeTo(options: EaseOptions): number {
    this.#request();
    return this.#core.easeTo(options);
  }
  flyTo(options: FlyOptions): number {
    this.#request();
    return this.#core.flyTo(options);
  }
  /** The momentum's id, or −1 when it has nothing to do. */
  startMomentum(momentum: GestureMomentum): number {
    this.#request();
    return this.#core.startMomentum(momentum);
  }
  stop(): void {
    this.#core.stop();
    this.#request();
  }
  setConstraints(constraints: Constraints): void {
    this.#core.setConstraints(constraints);
    this.#request();
  }
  cameraForBounds(
    bounds: Float64Array,
    options: FitOptions,
  ): CameraOptions | null {
    return this.#core.cameraForBounds(bounds, options) as CameraOptions | null;
  }

  // Queries.

  project(
    lon: number,
    lat: number,
    camera?: CameraOptions,
  ): Float64Array | undefined {
    return this.#core.project(lon, lat, camera);
  }
  unproject(
    x: number,
    y: number,
    camera?: CameraOptions,
  ): Float64Array | undefined {
    return this.#core.unproject(x, y, camera);
  }
  hitTest(x: number, y: number): Hit | null {
    return this.#core.hitTest(x, y) as Hit | null;
  }
  async queryRendered(box: Float64Array, layers?: string): Promise<string> {
    return this.#core.queryRendered(box, layers);
  }
  async querySource(
    source: string,
    sourceLayer?: string,
    filter?: string,
  ): Promise<string> {
    return this.#core.querySource(source, sourceLayer, filter);
  }
  async queryCluster(
    source: string,
    kind: string,
    id: number,
    limit: number,
    offset: number,
  ): Promise<string> {
    return this.#core.queryCluster(source, kind, id, limit, offset);
  }
  getStyle(): string {
    return this.#core.getStyle();
  }
  getLayer(id: string): string | undefined {
    return this.#core.getLayer(id);
  }
  getSource(id: string): string | undefined {
    return this.#core.getSource(id);
  }
  layerIds(): string[] {
    return this.#core.layerIds();
  }
  sourceIds(): string[] {
    return this.#core.sourceIds();
  }
  hasImage(name: string): boolean {
    return this.#core.hasImage(name);
  }

  // The style and its data.

  setStyleUrl(url: string): void {
    this.#core.setStyleUrl(url);
    this.#request();
  }
  /** The generation whose frames show the edit. */
  edit(commands: string): number {
    const generation = this.#core.edit(commands);
    this.#request();
    return generation;
  }
  addImage(
    name: string,
    width: number,
    height: number,
    pixels: Uint8Array,
    options: string,
  ): void {
    this.#core.addImage(name, width, height, pixels, options);
    this.#request();
  }
  updateImage(
    name: string,
    width: number,
    height: number,
    pixels: Uint8Array,
  ): void {
    this.#core.updateImage(name, width, height, pixels);
    this.#request();
  }
  removeImage(name: string): void {
    this.#core.removeImage(name);
    this.#request();
  }
  hostTile(
    source: string,
    z: number,
    x: number,
    y: number,
    bytes?: Uint8Array,
    geojson?: string,
  ): void {
    this.#core.hostTile(source, z, x, y, bytes, geojson);
    this.#request();
  }
  /** One step of a host-fed `raster-dem` source's heights: tile `i` is
   * `zxy[3i..3i + 3]`, its `side`² heights from `side² · i`. */
  hostElevations(
    source: string,
    side: number,
    zxy: Uint32Array,
    heights: Float32Array,
  ): HostElevationsAnswer {
    const out = this.#core.hostElevations(source, side, zxy, heights);
    this.#request();
    const dropped = [];
    for (let i = 1; i + 2 < out.length; i += 3)
      dropped.push({ z: out[i]!, x: out[i + 1]!, y: out[i + 2]! });
    return { batch: out[0]!, dropped };
  }
  /** Takes heights from an app's worker over `port` (`ElevationPortMessage`),
   * answering each, and tells it each batch that shows. */
  elevationPort(port: MessagePort): void {
    this.#ports.push(port);
    port.onmessage = ({ data }: MessageEvent<ElevationPortMessage>) => {
      const { id, source, side, zxy, heights } = data;
      void this.call("hostElevations", [source, side, zxy, heights])
        .then(
          (answer) => ({ id, ...(answer as HostElevationsAnswer) }),
          (e: unknown) => ({
            id,
            error: String(e instanceof Error ? e.message : e),
          }),
        )
        .then((reply: ElevationPortReply) => port.postMessage(reply));
    };
  }
  reloadHostTiles(source: string): void {
    this.#core.reloadHostTiles(source);
    this.#request();
  }
  sourceFrame(
    source: string,
    width: number,
    height: number,
    pixels: Uint8Array,
  ): void {
    this.#core.sourceFrame(source, width, height, pixels);
    this.#request();
  }
  /** A video source's new frame, in its element. A source the style
   * dropped is ignored: its close is on the way. */
  videoFrame(source: string, video: HTMLVideoElement): void {
    if (this.#removed || this.#lostAt !== undefined || this.#snapshotting)
      return;
    try {
      this.#core.videoFrame(source, video);
      this.#request();
    } catch {
      // The source is gone.
    }
  }
  videoFailed(source: string, message: string): void {
    this.#core.videoFailed(source, message);
    this.#request();
  }
  /** The request reaches the player with the next frame's events. */
  controlVideo(source: string, request: string, seconds: number): void {
    this.#core.controlVideo(source, request, seconds);
    this.#request();
  }
  /** Tiles that failed while offline load again. */
  retryFailedTiles(): void {
    this.#core.retryFailedTiles();
    this.#request();
  }

  // The canvas, frames and annotations.

  resize(width: number, height: number, ratio: number): void {
    if (this.#removed || this.#lostAt !== undefined) return;
    this.#host.resize(width, height, ratio);
    this.#request();
  }
  triggerRepaint(): void {
    this.#request();
  }
  /** The annotations' points, [lon, lat, …]; returns their pixels once
   * the map has drawn. */
  setAnnotations(points: Float64Array): Float64Array | undefined {
    this.#annotations = points;
    return this.#stats ? this.#project() : undefined;
  }
  setGpuTiming(on: boolean): void {
    this.#core.setGpuTiming(on);
  }
  setKeepAllCull(on: boolean): void {
    this.#core.setKeepAllCull(on);
    this.#request();
  }
  setTileLod(lod: TileLod): void {
    this.#core.setTileLod(lod);
    this.#request();
  }
  async gpuTimings(): Promise<GpuSubmission[]> {
    return this.#core.gpuTimings();
  }

  /** The next frame's pixels, as the canvas shows it. */
  async readPixels(premultiplied: boolean): Promise<Pixels> {
    const pixels: Promise<Uint8Array> = this.#core.readNextFrame(premultiplied);
    this.#request();
    const { width, height } = this.#canvas;
    return { pixels: await pixels, width, height };
  }

  /** The view `width × height` CSS pixels at `scale`, drawn offscreen once
   * everything in it has loaded; frames wait meanwhile. One at a time. */
  snapshot(width: number, height: number, scale: number): Promise<Pixels> {
    const run = async () => {
      const core = this.#core;
      this.#snapshotting = true;
      core.beginSnapshot(width, height, scale);
      try {
        const deadline = performance.now() + 60_000;
        while (!core.snapshotFrame()) {
          if (performance.now() > deadline)
            throw new Error("the map didn't settle in a minute");
          await new Promise((resolve) => setTimeout(resolve, 16));
        }
        const pixels: Uint8Array = await core.endSnapshot();
        const px = (v: number) => Math.max(1, Math.round(v * scale));
        return { pixels, width: px(width), height: px(height) };
      } catch (e) {
        core.cancelSnapshot();
        throw e;
      } finally {
        this.#snapshotting = false;
        this.#loop.request();
      }
    };
    const next = this.#snapshots.then(run, run);
    this.#snapshots = next.catch(() => {});
    return next;
  }

  // Tests.

  loseDevice(): void {
    this.#host.loseDevice();
  }
  trapPlacementWorker(): void {
    this.#host.trapPlacementWorker();
  }
  trapJobWorkers(): void {
    for (let i = 0; i < this.#host.jobWorkers; i++) this.#host.trapJobWorker(i);
  }
  trap(): void {
    this.#core.trap();
  }
  /** Frames show `time` until another call. */
  setFrameClock(time?: number): void {
    this.#frameClock = time;
  }

  /** Ports apps' workers feed heights through, which the map no longer
   * answers: for another map to take. */
  takePorts(): MessagePort[] {
    return this.#ports.splice(0);
  }

  /** Stops the map for good, saying why with one `error`: every call
   * fails with `message` from now on. */
  end(message: string): void {
    if (this.#removed) return;
    this.#halt(message);
    this.#host.destroy();
  }

  /** Stops drawing and releases the map's workers. */
  remove(): void {
    if (this.#removed) return;
    this.#removed = true;
    for (const port of this.#ports.splice(0)) port.close();
    this.#loop.stop();
    this.#host.destroy();
  }

  #request(): void {
    this.#loop.request();
  }

  /** The map has news (data arrived, a compile ended): the next frame
   * takes it in. */
  wake(): void {
    this.#request();
  }

  /** One frame, shown at `now`; returns whether another is needed. */
  #tick(now: number): boolean {
    // A snapshot draws its own frames, and asks for one when it ends.
    if (this.#removed || this.#lostAt !== undefined || this.#snapshotting)
      return false;
    // A test's frame clock may have run ahead of the display's: animations
    // and fades must not step back once it stops.
    now = this.#shown = Math.max(this.#frameClock ?? now, this.#shown);
    const frame = this.#core.render(now);
    if (frame.lost) {
      frame.free();
      this.#recover();
      return false;
    }
    const stats = plain(frame) as unknown as FrameStats;
    frame.free();
    // A style waits for the modules its features need.
    const bits = this.#core.modulesWanted();
    if (bits) {
      const wanted = MODULE_NAMES.filter((_, i) => bits & (1 << i));
      if (wanted.some((name) => !this.#asked.has(name)))
        void this.#want(wanted).catch(() => {});
    }
    stats.inputTime = this.#inputTime;
    this.#stats = stats;
    this.#drain();
    // A frame held back while compiles run drew nothing: no `render`.
    this.#send(stats.drew === false ? {} : { stats });
    return stats.needsFrame;
  }

  /** Sends the page a report with the events since the last one. */
  #send(report: Omit<Report, "state" | "events">): void {
    if (this.#removed) return;
    const points =
      this.#annotations.length && this.#stats ? this.#project() : undefined;
    const events = this.#events;
    this.#events = [];
    this.#report({
      state: this.state(),
      events,
      ...(points ? { points } : {}),
      ...report,
    });
  }

  #project(): Float64Array {
    return this.#annotations.length
      ? this.#core.projectPoints(this.#annotations)
      : new Float64Array(0);
  }

  /** Gets a new GPU after a loss and draws on: at once after the first
   * loss of a run, later after each one after (a device that lasted
   * `STABLE_MS` ends a run), and past `BACKOFF_MS`'s last not at all, with
   * an `error` event, so a driver that loses every device doesn't reset
   * the map forever. Native runtimes answer losses the same way. */
  #recover(): void {
    if (this.#lostAt !== undefined) return;
    const now = (this.#lostAt = performance.now());
    this.#losses = now - this.#deviceAt < STABLE_MS ? this.#losses + 1 : 1;
    this.#events.push({ type: "lost", data: {} });
    const wait = BACKOFF_MS[this.#losses - 1];
    if (wait === undefined) {
      this.#gone(
        `the GPU was lost ${this.#losses} times in a row, each device within ` +
          `${STABLE_MS / 1000} s of the last, so no other is made`,
      );
      return;
    }
    this.#send({});
    setTimeout(() => {
      if (this.#removed) return;
      this.#host.recover().then(
        () => {
          this.#recovering = this.#lostAt;
          this.#lostAt = undefined;
          this.#deviceAt = performance.now();
          this.#request();
        },
        (e) =>
          this.#gone(
            `the GPU was lost and no other could be made: ${e instanceof Error ? e.message : e}`,
          ),
      );
    }, wait);
  }

  /** The GPU is lost for good: WebGPU picked by `auto` hands the map to
   * the page to start again on WebGL 2; anything else stops it. */
  #gone(why: string): void {
    if (this.#removed) return;
    if (this.backend === "webgpu" && !this.#host.forced)
      this.#send({ fallback: { why, lostAt: this.#lostAt ?? 0 } });
    else this.end(`the map stopped: ${why}`);
  }

  /** The wasm map trapped: it draws no more, and says why once. Its
   * calls throw `e` from now on. */
  #trapped(e: Error): void {
    if (this.#removed) return;
    this.#halt(e.message);
  }

  /** Draws no more, and says why once: calls fail with `message`. */
  #halt(message: string): void {
    // The state as the last frame left it, or as the wasm map says if it
    // still can, before any frame.
    let state = this.#state;
    try {
      state ??= this.state();
    } catch {
      // It trapped.
    }
    this.#removed = true;
    this.#stopped = message;
    this.#loop.stop();
    const events = this.#events;
    this.#events = [];
    events.push({ type: "error", data: { message } });
    if (state) this.#report({ state, events, stopped: message });
  }

  /** A failure, as an `error` event. */
  #fail(e: unknown): void {
    // Once the map is gone, what fails is a call into it.
    if (this.#removed) return;
    const message = String(e instanceof Error ? e.message : e);
    this.#events.push({ type: "error", data: { message } });
    this.#send({});
  }

  /** The wasm map's events since the last frame, for the page. */
  #drain(): void {
    const out = this.#events;
    for (const event of this.#core.takeEvents() as CoreEvent[]) {
      switch (event.type) {
        case "idle":
          void this.#want(["api"]).catch(() => {});
          out.push({ type: event.type, data: {} });
          break;
        case "styleload":
        case "load":
          out.push({ type: event.type, data: {} });
          break;
        case "styledata":
          out.push({
            type: "styledata",
            data: { generation: event.generation },
          });
          break;
        case "sourcedata":
          out.push({
            type: "sourcedata",
            data: { source: event.source, loaded: event.loaded },
          });
          break;
        case "error":
        case "warning":
          out.push({ type: event.type, data: { message: event.message } });
          break;
        case "video":
          out.push({ type: "video", data: event });
          break;
        case "styleimagemissing":
          out.push({ type: "styleimagemissing", data: { id: event.id } });
          break;
        case "degraded": {
          // Each gap that arrived, or reaches other layers, then the set.
          const was = (d: Degradation) =>
            this.#gaps.some(
              (g) => g.gap === d.gap && g.layers.join() === d.layers.join(),
            );
          for (const { gap, layers } of event.gaps.filter((d) => !was(d))) {
            if (!this.#warnedGaps.has(gap)) {
              this.#warnedGaps.add(gap);
              const named = layers.length ? ` (${layers.join(", ")})` : "";
              console.warn(
                `orbis: on ${this.backend === "webgl2" ? "WebGL 2" : "this GPU"} the map draws ${gap} otherwise than on WebGPU${named}`,
              );
            }
            out.push({ type: "degraded", data: { gap, layers } });
          }
          this.#gaps = event.gaps;
          out.push({ type: "gaps", data: { gaps: event.gaps } });
          break;
        }
        case "restored": {
          if (this.#recovering === undefined) break;
          const recoveredMs = performance.now() - this.#recovering;
          this.#recovering = undefined;
          out.push({
            type: "restored",
            data: { recoveredMs, backend: this.backend },
          });
          break;
        }
        case "camera": {
          const phase = event.phase === "progress" ? "" : event.phase;
          out.push({
            type: `${event.change}${phase}` as "move",
            data: { user: event.user },
          });
          break;
        }
        case "animationend":
          out.push({
            type: "animationend",
            data: { id: event.id, arrived: !event.interrupted },
          });
          break;
        case "elevationsshown": {
          const shown: ElevationPortShown = {
            shown: event.batch,
            source: event.source,
          };
          for (const port of this.#ports) port.postMessage(shown);
          out.push({ type: "elevationsshown", data: shown });
          break;
        }
        case "tilerequest":
          out.push({
            type: "tilerequest",
            data: { source: event.source, z: event.z, x: event.x, y: event.y },
          });
          break;
        case "click":
        case "dblclick":
        case "contextmenu": {
          const g = this.#core.unproject(event.x, event.y, undefined);
          const data: PointerEventData = {
            point: [event.x, event.y],
            lngLat: g ? [g[0]!, g[1]!] : null,
          };
          out.push({ type: event.type, data } as EngineEvent);
          break;
        }
      }
    }
  }
}
