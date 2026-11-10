// The map itself: the wasm map with its frames, input, queries,
// snapshots and recovery from a lost GPU, wherever it runs. In the map
// worker (map-worker.ts) it draws into an OffscreenCanvas; where the
// browser can't do that, on the page into the canvas. The page's side
// (map.ts) reaches it through a link (link.ts) and hears from it by
// reports: the map's state after each frame, with its events.

import type { Backend, Build } from "./builds.ts";
import { frameLoop } from "./frames.ts";
import { startHost, type Core, type CoreOptions, type Host } from "./hosts.ts";
import type {
  Camera,
  CameraOptions,
  Constraints,
  EaseOptions,
  FitOptions,
  FlyOptions,
  FrameStats,
  FreeCamera,
  GestureMomentum,
  GestureOptions,
  GpuSubmission,
  Hit,
  MapEvents,
  Padding,
  PointerEventData,
  TileLod,
} from "./types.ts";
import type { VideoEventObject } from "./videos.ts";
import type { Workers } from "./workers.ts";

/** What the page reads of the map without asking: as of its last report. */
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
  | {
      type: "camera";
      change: "move" | "zoom" | "rotate" | "pitch";
      phase: "start" | "progress" | "end";
      user: boolean;
    }
  | { type: "animationend"; id: number; interrupted: boolean }
  | { type: "click" | "dblclick" | "contextmenu"; x: number; y: number }
  | { type: "tilerequest"; source: string; z: number; x: number; y: number }
  | VideoEventObject;

/** The input the page forwards: the wasm map's input methods. */
export type InputKind =
  | "pointerDown"
  | "pointerMove"
  | "pointerUp"
  | "pointerCancel"
  | "wheel"
  | "key"
  | "blur";

/** How the map starts. */
export interface EngineSetup {
  canvas: HTMLCanvasElement | OffscreenCanvas;
  style: string;
  options: CoreOptions;
  camera?: CameraOptions;
  constraints?: Constraints;
  /** Physical width and height and device pixels per CSS pixel. */
  size: [number, number, number];
  /** Milliseconds to add to the page's event times to put them on this
   * thread's clock: 0 on the page. */
  clock: number;
  /** Whether input that moves the map draws a frame as it arrives, rather
   * than at the next animation frame: in a worker, whose input comes as
   * soon as the page has it (`pointerrawupdate`), just after an animation
   * frame has begun, and whose frames the compositor shows a frame after
   * the page's. Drawn at the next animation frame, input would show a
   * frame later than on the page's own thread, whose moves wait for its
   * next frame anyway. */
  drawOnInput?: boolean;
}

/** A wasm-bindgen object's getters as a plain object, which crosses to
 * the page by message. */
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

export class Engine {
  #host: Host;
  #canvas: HTMLCanvasElement | OffscreenCanvas;
  #clock: number;
  #report: (report: Report) => void;
  #loop: ReturnType<typeof frameLoop>;
  #events: EngineEvent[] = [];
  #stats: FrameStats | undefined;
  #snapshotting = false;
  #snapshots: Promise<unknown> = Promise.resolve();
  /** A device loss's start, while the host recovers. */
  #lostAt: number | undefined;
  /** The last device loss's start, until the new device draws. */
  #recovering: number | undefined;
  /** When the device was made, and the losses in the current run. */
  #deviceAt = performance.now();
  #losses = 0;
  #removed = false;
  /** Whether a prepare is queued. */
  #preparing = false;
  /** The annotations' points, [lon, lat, …]. */
  #annotations: Float64Array = new Float64Array(0);
  /** The box zoom's corners last reported, as text. */
  #box = "null";
  /** The latest input's time, on the page's clock. */
  #inputTime: number | undefined;
  /** The state last read. */
  #state: MapState | undefined;
  #drawOnInput: boolean;
  /** Whether input drew a frame since the last animation frame. */
  #drewOnInput = false;
  /** Whether something asked for a frame since the last one drew. */
  #changed = false;
  /** The frame drawn if the input expected after an animation frame
   * doesn't come. */
  #fallback: ReturnType<typeof setTimeout> | undefined;
  /** The last frame's time: a frame never goes back before it. */
  #shown = -Infinity;
  /** The last animation frame's time, and the display's frame interval. */
  #vsync = 0;
  #interval = 1000 / 60;

  private constructor(
    host: Host,
    setup: EngineSetup,
    report: (report: Report) => void,
  ) {
    this.#host = host;
    this.#canvas = setup.canvas;
    this.#clock = setup.clock;
    this.#drawOnInput = setup.drawOnInput ?? false;
    this.#report = report;
    this.#loop = frameLoop(
      (now) => {
        const since = now - this.#vsync;
        if (since > 4 && since < 50)
          this.#interval += (since - this.#interval) / 8;
        this.#vsync = now;
        const drewOnInput = this.#drewOnInput;
        this.#drewOnInput = false;
        if (!drewOnInput || !this.#core.isMoving()) return this.#tick(now);
        // Input drew the last frame of a gesture, whose next input is likely
        // on its way: the next frame waits for it, or for half a frame.
        this.#fallback ??= setTimeout(() => {
          this.#fallback = undefined;
          if (this.#changed || this.#stats?.needsFrame)
            this.#draw(performance.now());
        }, this.#interval / 2);
        return false;
      },
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
        setup: (core) => {
          if (setup.constraints) core.setConstraints(setup.constraints);
          if (setup.camera) core.jumpTo(setup.camera);
        },
      },
    );
    host.resize(...setup.size);
    engine = new Engine(host, setup, report);
    // The style goes in and its tiles are asked for as soon as the map is
    // made, without waiting for the first frame.
    engine.wake();
    return engine;
  }

  get #core(): Core {
    return this.#host.core;
  }

  /** The GPU API the map draws with. */
  get backend(): Backend {
    return this.#host.backend;
  }

  /** What the page reads without asking. */
  state(): MapState {
    // A snapshot's end and a recovery hold the wasm map while they wait:
    // the state stays as it was.
    if (this.#state && (this.#snapshotting || this.#lostAt !== undefined))
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
      (...a: unknown[]) => boolean
    >;
    // A release may end a click, which the next frame reports.
    if (
      core[kind](...args, time + this.#clock) ||
      kind === "blur" ||
      kind === "pointerUp"
    ) {
      if (this.#drawOnInput && !this.#drewOnInput) {
        this.#drewOnInput = true;
        this.#draw(performance.now());
        // The next animation frame lets the next input draw.
        this.#loop.request();
      } else {
        this.#request();
      }
    }
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
  /** A video source's new frame: its element on the page, or an
   * ImageBitmap of it. A source the style dropped is ignored: its close
   * is on the way. */
  videoFrame(source: string, frame: HTMLVideoElement | ImageBitmap): void {
    // A dropped bitmap is closed rather than left to the collector, which
    // holds its pixels meanwhile; one the map took, it closes once copied.
    const drop = () => {
      if (frame instanceof ImageBitmap) frame.close();
    };
    if (this.#removed || this.#lostAt !== undefined || this.#snapshotting)
      return drop();
    try {
      this.#core.videoFrame(source, frame);
      this.#request();
    } catch {
      drop();
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

  /** Stops drawing and releases the map's workers. */
  remove(): void {
    if (this.#removed) return;
    this.#removed = true;
    this.#loop.stop();
    clearTimeout(this.#fallback);
    this.#host.destroy();
  }

  #request(): void {
    this.#changed = true;
    this.#loop.request();
  }

  /** A frame now, outside an animation frame, and the next animation
   * frame if it needs one. */
  #draw(now: number): void {
    try {
      if (this.#tick(now)) this.#loop.request();
    } catch (e) {
      this.#fail(e);
    }
  }

  /** The map has new data to draw. Its next requests go out in a task of
   * their own, without waiting for the frame. */
  wake(): void {
    this.#request();
    if (this.#preparing) return;
    this.#preparing = true;
    afterTask(() => {
      this.#preparing = false;
      if (this.#removed || this.#lostAt !== undefined || this.#snapshotting)
        return;
      try {
        this.#core.prepare();
      } catch (e) {
        this.#fail(e);
      }
    });
  }

  /** One frame, shown at `now`; returns whether another is needed. */
  #tick(now: number): boolean {
    // A snapshot draws its own frames, and asks for one when it ends.
    if (this.#removed || this.#lostAt !== undefined || this.#snapshotting)
      return false;
    this.#changed = false;
    clearTimeout(this.#fallback);
    this.#fallback = undefined;
    // A frame input drew can come after an animation frame began and
    // before its callback ran, with a later time than that callback's:
    // animations and fades must not step back.
    now = this.#shown = Math.max(now, this.#shown);
    const frame = this.#core.render(now);
    if (frame.lost) {
      frame.free();
      this.#recover();
      return false;
    }
    const stats = plain(frame) as unknown as FrameStats;
    frame.free();
    stats.inputTime = this.#inputTime;
    this.#stats = stats;
    this.#drain();
    this.#send({ stats });
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
      this.#fail(
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
        (e) => this.#fail(`the GPU was lost and no other could be made: ${e}`),
      );
    }, wait);
  }

  /** A failure, as an `error` event. */
  #fail(e: unknown): void {
    const message = String(e instanceof Error ? e.message : e);
    this.#events.push({ type: "error", data: { message } });
    this.#send({});
  }

  /** The wasm map's events since the last frame, for the page. */
  #drain(): void {
    const out = this.#events;
    for (const event of this.#core.takeEvents() as CoreEvent[]) {
      switch (event.type) {
        case "styleload":
        case "load":
        case "idle":
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
        case "restored": {
          if (this.#recovering === undefined) break;
          const recoveredMs = performance.now() - this.#recovering;
          this.#recovering = undefined;
          out.push({ type: "restored", data: { recoveredMs } });
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
