// The page's side of a map: its DOM, input, events, annotations and video
// players, and the API. The map itself (engine.ts: the wasm map, its
// camera, gestures, drawing and queries) runs on the page's thread, as GL
// JS does, and draws on its animation frames; tile jobs and labels run in
// workers of their own. Calls into the map answer with promises (a call
// may wait for the feature module it needs), and the map reports after
// its frames.

import { Annotations } from "./annotations.ts";
import {
  fallbackBuild,
  loadModule,
  pickBuild,
  type Build,
  type ModuleName,
} from "./builds.ts";
import {
  Engine,
  type EngineEvent,
  type EngineSetup,
  type MapState,
  type Pixels,
  type Report,
} from "./engine.ts";
import { Emitter, type Listener } from "./events.ts";
import type { CoreOptions } from "./hosts.ts";
import { allowed, attachInput, type Allowed } from "./input.ts";
import { preloadStyle, styleModules } from "./preload.ts";
import {
  transaction,
  type FeatureTarget,
  type Transaction,
} from "./transaction.ts";
import type {
  Bounds,
  Camera,
  CameraOptions,
  Constraints,
  EaseOptions,
  Feature,
  FitOptions,
  FlyOptions,
  FreeCamera,
  FrameStats,
  GestureMomentum,
  GestureOptions,
  GpuSubmission,
  Hit,
  HostElevationsAnswer,
  HostElevationTile,
  LngLat,
  MapEvents,
  MapOptions,
  Point,
  QueryGeometry,
  QueryOptions,
  SourceQueryOptions,
  StyleImageData,
  StyleImageOptions,
  TileLod,
} from "./types.ts";
import { Videos } from "./videos.ts";
import { defaultJobWorkers, spawnWorkers } from "./workers.ts";

/** The map's methods the page calls by name. */
type Calls = Omit<
  { [K in keyof Engine]: Engine[K] },
  | "state"
  | "input"
  | "fallback"
  | "wake"
  | "backend"
  | "modules"
  | "call"
  | "remove"
  | "end"
  | "takePorts"
>;
type Method = keyof Calls;
/** Calls the map's `method`: a promise of its answer, rejected once the
 * map is removed. */
type Call = <K extends Method>(
  method: K,
  ...args: Parameters<Calls[K]>
) => Promise<Awaited<ReturnType<Calls[K]>>>;

/** An image's RGBA bytes, as the wasm map takes them. */
const rgba = (image: StyleImageData) =>
  new Uint8Array(
    image.data.buffer,
    image.data.byteOffset,
    image.data.byteLength,
  );

const imageData = ({ pixels, width, height }: Pixels) =>
  new ImageData(
    new Uint8ClampedArray(
      pixels.buffer as ArrayBuffer,
      pixels.byteOffset,
      pixels.length,
    ),
    width,
    height,
  );

/** The camera: its state, moves and limits. The state is the map's now:
 * a move shows in it as soon as the call that made it returns. */
export class CameraControl {
  #call: Call;
  #state: () => MapState;
  #track: (id: Promise<number>) => Promise<boolean>;
  #fail: (e: unknown) => void;

  /** @internal */
  constructor(
    call: Call,
    state: () => MapState,
    track: (id: Promise<number>) => Promise<boolean>,
    fail: (e: unknown) => void,
  ) {
    this.#call = call;
    this.#state = state;
    this.#track = track;
    this.#fail = fail;
  }
  #send(call: Promise<unknown>): void {
    call.catch(this.#fail);
  }

  /** The camera now. */
  get current(): Camera {
    return structuredClone(this.#state().camera);
  }

  /** The eye now (GL JS `getFreeCameraOptions`). */
  get free(): FreeCamera {
    return structuredClone(this.#state().free);
  }

  /** Puts the eye there at once (GL JS `setFreeCameraOptions`), keeping the
   * field of view and padding; what `camera` leaves out stays. */
  jumpToFree(camera: Partial<FreeCamera>): void {
    this.#send(this.#call("jumpToFree", camera));
  }

  /** Whether the center stands on the terrain at rest (GL JS
   * `centerClampedToGround`, true by default). Off, it keeps the elevation
   * it is given, and the camera orbits a point in the air. */
  get centerClampedToGround(): boolean {
    return this.#state().clamped;
  }
  set centerClampedToGround(clamped: boolean) {
    this.#send(this.#call("setCenterClampedToGround", clamped));
  }

  /** Puts the center this many metres above sea level at once (GL JS
   * `setCenterElevation`); while clamped to the ground, the terrain takes it back. */
  setCenterElevation(elevation: number): void {
    this.#send(this.#call("setCenterElevation", elevation));
  }

  /** Whether an animation or a gesture is moving the camera. */
  get moving(): boolean {
    return this.#state().moving;
  }

  /** Moves at once. Each move takes only the properties it sets: animations
   * of the others go on, as in MapLibre Native, and one left with nothing to
   * move is interrupted. {@link stop} ends them all. */
  jumpTo(options: CameraOptions): void {
    this.#send(this.#call("jumpTo", options));
  }

  /** Starts at the next frame; resolves true on arrival, false when interrupted. */
  easeTo(options: EaseOptions): Promise<boolean> {
    return this.#track(this.#call("easeTo", options));
  }

  /** Flies along GL JS's zoom-out curve from the next frame; resolves
   * true on arrival, false when interrupted. */
  flyTo(options: FlyOptions): Promise<boolean> {
    return this.#track(this.#call("flyTo", options));
  }

  /** The camera that shows `bounds`, or null when the padding leaves no room. */
  cameraForBounds(
    bounds: Bounds,
    options: FitOptions = {},
  ): Promise<CameraOptions | null> {
    return this.#call("cameraForBounds", new Float64Array(bounds), options);
  }

  /** Flies (or, with `linear`, eases) to show `bounds`; resolves false when it doesn't fit or is interrupted. */
  async fitBounds(
    bounds: Bounds,
    options: FitOptions & FlyOptions & { linear?: boolean } = {},
  ): Promise<boolean> {
    const target = await this.cameraForBounds(bounds, options);
    if (!target) return false;
    const move = { ...options, ...target, offset: undefined };
    return options.linear ? this.easeTo(move) : this.flyTo(move);
  }

  /** Continues a gesture the page recognized itself: each part glides on
   * from its release velocity and slows by its decay, then settles on its
   * snap. Input or a camera move stops it. Resolves true once it ends on
   * its own, false when stopped. The map's own gestures end this way. */
  startMomentum(momentum: GestureMomentum): Promise<boolean> {
    return this.#track(this.#call("startMomentum", momentum));
  }

  /** Stops every running animation; their promises resolve false. */
  stop(): void {
    this.#send(this.#call("stop"));
  }

  setConstraints(constraints: Constraints): void {
    this.#send(this.#call("setConstraints", constraints));
  }
}

/** What a map was made with, and has been given since, to start it again
 * on WebGL 2 after WebGPU is lost for good. */
interface Restart {
  core: CoreOptions;
  jobWorkers: number;
  interactive: boolean;
  /** The style last set, as a URL: what starts again where the style as
   * edits left it can't be read. */
  style: string;
  constraints?: Constraints;
  gestures?: GestureOptions;
  /** Images added, by name: their size, RGBA bytes and options. */
  images: Map<string, [number, number, Uint8Array, string]>;
}

/** A map in a container element. Create one with {@link createMap}. */
export class OrbisMap {
  readonly container: HTMLElement;
  readonly camera: CameraControl;
  readonly annotations: Annotations;
  #engine: Engine;
  #restart: Restart;
  /** The input and resize observer on the canvas, to move with it. */
  #canvasDetach: (() => void)[] = [];
  /** `load` fired: a map started again on WebGL 2 doesn't fire it again. */
  #loaded = false;
  #fallingBack = false;
  /** The loss a map started again on WebGL 2 recovers from, until it
   * draws; then its added images, until its style is in. */
  #recoveringFrom: number | undefined;
  #readdImages = false;
  /** Video sources' players, on the page's thread. */
  #videos: Videos;
  #events = new Emitter<MapEvents>();
  #stats: FrameStats | undefined;
  /** Animations the page started, by id. */
  #animations = new Map<number, (arrived: boolean) => void>();
  #detach: (() => void)[] = [];
  #canvas: HTMLCanvasElement;
  #frame: HTMLElement;
  /** The device pixels per CSS pixel the map draws with; the display's
   * when undefined. */
  #pixelRatio: number | undefined;
  #showBox: ((corners: number[] | null) => void) | undefined;
  #allow: ((allowed: Allowed) => void) | undefined;
  #removed = false;
  /** Why the map stopped, if its wasm trapped. */
  #stopped: string | undefined;

  /** @internal */
  constructor(
    container: HTMLElement,
    frame: HTMLElement,
    canvas: HTMLCanvasElement,
    overlay: HTMLElement,
    engine: Engine,
    reports: Reports,
    restart: Restart,
    pixelRatio?: number,
  ) {
    this.container = container;
    this.#frame = frame;
    this.#canvas = canvas;
    this.#engine = engine;
    this.#restart = restart;
    this.#pixelRatio = pixelRatio;
    const fail = (e: unknown) => this.#error(e);
    this.#videos = new Videos(
      () =>
        this.#removed
          ? undefined
          : {
              videoFrame: (source, video) => {
                this.#call("videoFrame", source, video).catch(() => {});
              },
              videoFailed: (source, message) => {
                this.#call("videoFailed", source, message).catch(fail);
              },
            },
      () => {},
    );
    this.camera = new CameraControl(
      this.#call,
      () => this.#engine.state(),
      (id) => this.#track(id),
      fail,
    );
    this.annotations = new Annotations(overlay, (points) => {
      this.#call("setAnnotations", points).then((pixels) => {
        if (pixels) this.annotations.place(pixels);
      }, fail);
    });
    this.#attachCanvas();
    // Tiles that failed while offline load again when the network returns.
    const online = () => {
      this.#call("retryFailedTiles").catch(fail);
    };
    addEventListener("online", online);
    this.#detach.push(() => removeEventListener("online", online));
    reports.listen((report) => this.#receive(report));
  }

  /** Takes the canvas's input and follows its size. */
  #attachCanvas(): void {
    const canvas = this.#canvas;
    if (this.#restart.interactive) {
      const input = attachInput(canvas, (kind, args, time) => {
        if (!this.#removed) this.#engine.input(kind, args, time);
      });
      this.#showBox = input.showBox;
      const allow = input.allow;
      this.#allow = allow;
      if (this.#restart.gestures) allow(allowed(this.#restart.gestures));
      this.#canvasDetach.push(input.detach);
    }
    const observer = new ResizeObserver(() => this.#resize());
    observer.observe(canvas);
    this.#canvasDetach.push(() => observer.disconnect());
  }

  #call: Call = (method, ...args) => {
    if (this.#removed) return Promise.reject(new Error("the map was removed"));
    const call = this.#engine.call(method, args) as Promise<never>;
    void call.then(
      () => this.#remember(method, args),
      () => {},
    );
    return call;
  };

  /** Keeps what a call gave the map that its style doesn't hold, to give
   * a map started again on WebGL 2. */
  #remember(method: Method, args: unknown[]): void {
    const restart = this.#restart;
    switch (method) {
      case "setStyleUrl":
        restart.style = args[0] as string;
        break;
      case "setConstraints":
        restart.constraints = args[0] as Constraints;
        break;
      case "setGestures":
        restart.gestures = args[0] as GestureOptions;
        break;
      case "setTileLod":
        restart.core.tileLod = {
          ...restart.core.tileLod,
          ...(args[0] as TileLod),
        };
        break;
      case "addImage": {
        const [name, width, height, pixels, options] = args as [
          string,
          number,
          number,
          Uint8Array,
          string,
        ];
        restart.images.set(name, [width, height, pixels.slice(), options]);
        break;
      }
      case "updateImage": {
        const [name, , , pixels] = args as [string, number, number, Uint8Array];
        const image = restart.images.get(name);
        if (image) image[2] = pixels.slice();
        break;
      }
      case "removeImage":
        restart.images.delete(args[0] as string);
        break;
    }
  }

  /** WebGPU is lost for good (`why`, since `lostAt`): the map starts
   * again on WebGL 2 in a new canvas (a canvas that had a WebGPU context
   * takes no other), with its style as edits left it, camera, constraints,
   * gestures, added images and annotations, and the ports apps feed heights
   * through. `restored` says so once it draws; failing that, the map stops
   * with an `error` saying why. */
  async #fallBack({ why, lostAt }: { why: string; lostAt: number }) {
    if (this.#fallingBack || this.#removed) return;
    this.#fallingBack = true;
    const old = this.#engine;
    const restart = this.#restart;
    const style = await old.call("getStyle", []).then(
      (json) => `data:application/json,${encodeURIComponent(json as string)}`,
      () => restart.style,
    );
    if (this.#removed) return;
    const { camera, clamped } = old.state();
    const canvas = newCanvas();
    this.#canvas.replaceWith(canvas);
    for (const detach of this.#canvasDetach.splice(0)) detach();
    this.#canvas = canvas;
    const held = reports();
    let engine: Engine;
    try {
      engine = await launch(
        fallbackBuild(why),
        canvas,
        {
          style,
          options: restart.core,
          camera: {
            center: camera.center,
            zoom: camera.zoom,
            bearing: camera.bearing,
            pitch: camera.pitch,
            roll: camera.roll,
            fov: camera.fov,
            padding: camera.padding,
          },
          constraints: restart.constraints,
        },
        this.#pixelRatio ?? devicePixelRatio,
        restart.jobWorkers,
        held.deliver,
        [...old.modules()],
      );
    } catch (e) {
      const message = String(e instanceof Error ? e.message : e);
      old.end(
        message.startsWith("the map stopped")
          ? message
          : `the map stopped: ${message}`,
      );
      return;
    }
    if (this.#removed) {
      engine.remove();
      return;
    }
    for (const port of old.takePorts()) engine.elevationPort(port);
    old.remove();
    this.#engine = engine;
    this.#fallingBack = false;
    this.#attachCanvas();
    if (!clamped) {
      engine.setCenterClampedToGround(false);
      engine.jumpTo({ elevation: camera.elevation });
    }
    if (restart.gestures) engine.setGestures(restart.gestures);
    for (const resolve of this.#animations.values()) resolve(false);
    this.#animations.clear();
    this.#recoveringFrom = lostAt;
    this.#readdImages = restart.images.size > 0;
    if (this.annotations.size) this.annotations.changed();
    held.listen((report) => this.#receive(report));
  }

  /** Sets what the map's gestures may move, how releases continue and
   * when presses are clicks (MapLibre Compose's `MapInteractions`); what
   * `options` leaves out takes its default. A wheel or key whose movement
   * is off scrolls the page as it would without the map. A change cancels
   * a gesture under way. */
  setGestures(options: GestureOptions): void {
    this.#allow?.(allowed(options));
    this.#call("setGestures", options).catch((e) => this.#error(e));
  }

  /** Whether the map draws: false once it stopped (an `error` starting
   * "the map stopped" said why) or was removed. */
  get running(): boolean {
    return !this.#removed && this.#stopped === undefined;
  }

  /** The canvas the map draws into and takes input from. */
  get canvas(): HTMLCanvasElement {
    return this.#canvas;
  }

  /** The GPU API the map draws with: WebGPU, or WebGL 2 where the browser
   * gives WebGPU no adapter, its WebGPU stopped working (a `restored`
   * event says so), or the map asked for it ({@link MapOptions.backend}).
   * What WebGL 2 draws otherwise than WebGPU, the map says with `degraded`
   * events (#140). */
  get backend(): "webgpu" | "webgl2" {
    return this.#engine.backend;
  }

  /** Why the map draws with WebGL 2 where it was left to pick and WebGPU
   * was expected to work: WebGPU failed or didn't answer for an adapter,
   * made no device, or was lost for good. An app may remember it for the
   * browser, whose WebGPU then needn't be tried (and perhaps crash its GPU
   * process) again. */
  get fallback(): string | undefined {
    return this.#engine.fallback;
  }

  /**
   * The camera's view, `width × height` CSS pixels (the canvas's by
   * default) at `scale` device pixels each, drawn offscreen once everything
   * in it has loaded and nothing fades: the same style, data and camera
   * always draw the same pixels. The map draws no frames of its own
   * meanwhile. Rejects after a minute.
   */
  async snapshot(
    width = this.#canvas.clientWidth,
    height = this.#canvas.clientHeight,
    scale = this.#pixelRatio ?? devicePixelRatio,
  ): Promise<ImageData> {
    return imageData(await this.#call("snapshot", width, height, scale));
  }

  /** Loses the GPU on purpose, to exercise recovery. */
  loseDevice(): void {
    this.#send(this.#call("loseDevice"));
  }
  /** Traps the placement worker's wasm, as a bug would, to exercise its
   * replacement: labels place again within a frame or two. */
  trapPlacementWorker(): void {
    this.#send(this.#call("trapPlacementWorker"));
  }
  /** Traps every job worker's wasm, as a bug would, to exercise their
   * replacement: tiles load again on the new workers. */
  trapJobWorkers(): void {
    this.#send(this.#call("trapJobWorkers"));
  }
  /** Traps the map's own wasm, as a bug would, to exercise how it fails:
   * one `error` event says the map stopped, and every call after rejects
   * with the same message. */
  trap(): void {
    this.#send(this.#call("trap"));
  }
  /** For test runners: frames show `time` (milliseconds on the page's
   * clock, as `performance.now()` gives it) instead of their animation
   * frame's, never earlier than the last frame, as under fake timers:
   * transitions and fades move only when the clock does. Without `time`,
   * frames follow the animation frames again. Resolves once the map has
   * it, so every frame it reports after that shows `time`. */
  setFrameClock(time?: number): Promise<void> {
    return this.#call("setFrameClock", time);
  }

  /** The last frame's counters. */
  get stats(): FrameStats | undefined {
    return this.#stats;
  }

  /** The next frame the map draws, as the canvas shows it: for
   * screenshots and tests. Asks for that frame. Its pixels are straight
   * RGBA, or with `premultiplied` the canvas's own bytes. Light added over
   * transparency (the globe's atmosphere in space) has more colour than
   * alpha there; straight, its alpha rises to its brightest channel, so
   * it shows as the canvas does over black. */
  async readPixels({
    premultiplied = false,
  }: { premultiplied?: boolean } = {}): Promise<ImageData> {
    return imageData(await this.#call("readPixels", premultiplied));
  }

  /** Draws a frame even if nothing changed (GL JS `triggerRepaint`). */
  triggerRepaint(): void {
    this.#send(this.#call("triggerRepaint"));
  }

  /** Times the GPU's work from now on, for measurements: `gpuTimings`
   * collects it. WebGPU only. */
  setGpuTiming(on: boolean): void {
    this.#send(this.#call("setGpuTiming", on));
  }
  /** Has the cull keep every candidate, for measurements of what it saves:
   * the map draws the same, only slower. */
  setKeepAllCull(on: boolean): void {
    this.#send(this.#call("setKeepAllCull", on));
  }
  /** Sets how tiles coarsen with distance in pitched views; fields left out stay as they were. */
  setTileLod(lod: TileLod): void {
    this.#send(this.#call("setTileLod", lod));
  }

  /** Each GPU submission's times since the last call, once the GPU has
   * finished them, in order: its passes' GPU milliseconds by label (where
   * the device has timestamp queries) and the CPU's to submit them. */
  gpuTimings(): Promise<GpuSubmission[]> {
    return this.#call("gpuTimings");
  }

  on<K extends keyof MapEvents>(
    type: K,
    listener: Listener<MapEvents[K]>,
  ): () => void {
    return this.#events.on(type, listener);
  }
  off<K extends keyof MapEvents>(
    type: K,
    listener: Listener<MapEvents[K]>,
  ): void {
    this.#events.off(type, listener);
  }
  once<K extends keyof MapEvents>(type: K): Promise<MapEvents[K]> {
    return this.#events.once(type);
  }
  /** `for await (const e of map.events("idle"))`. */
  events<K extends keyof MapEvents>(type: K): AsyncGenerator<MapEvents[K]> {
    return this.#events.events(type);
  }

  /** The pixel showing a point, on the terrain, or null behind the eye: now,
   * or after a jump to `camera` (without making it). */
  async project(lngLat: LngLat, camera?: CameraOptions): Promise<Point | null> {
    const p = await this.#call("project", lngLat[0], lngLat[1], camera);
    return pair(p);
  }

  /** The ground point under a pixel, on the terrain, or null above the
   * horizon: now, or after a jump to `camera`. */
  async unproject(
    point: Point,
    camera?: CameraOptions,
  ): Promise<LngLat | null> {
    const g = await this.#call("unproject", point[0], point[1], camera);
    return pair(g);
  }

  /** The topmost feature under a pixel, from the last frame. */
  hitTest(point: Point): Promise<Hit | null> {
    return this.#call("hitTest", point[0], point[1]);
  }

  /** Every feature drawn under a point or box (all on screen without one), topmost first. */
  async queryRenderedFeatures(
    geometry?: QueryGeometry,
    options: QueryOptions = {},
  ): Promise<Feature[]> {
    const box =
      geometry === undefined
        ? [0, 0, this.canvas.clientWidth, this.canvas.clientHeight]
        : typeof geometry[0] === "number"
          ? ([geometry[0], geometry[1], geometry[0], geometry[1]] as number[])
          : [
              ...(geometry as [Point, Point])[0],
              ...(geometry as [Point, Point])[1],
            ];
    return JSON.parse(
      await this.#call(
        "queryRendered",
        new Float64Array(box),
        options.layers ? JSON.stringify(options.layers) : undefined,
      ),
    );
  }

  /** Every feature of a source in its loaded tiles that matches a filter. */
  async querySourceFeatures(
    source: string,
    options: SourceQueryOptions = {},
  ): Promise<Feature[]> {
    return JSON.parse(
      await this.#call(
        "querySource",
        source,
        options.sourceLayer,
        options.filter === undefined
          ? undefined
          : JSON.stringify(options.filter),
      ),
    );
  }

  /** A cluster's members one zoom in: clusters and points (GL JS `getClusterChildren`). */
  async getClusterChildren(
    source: string,
    clusterId: number,
  ): Promise<Feature[]> {
    return JSON.parse(
      await this.#call("queryCluster", source, "children", clusterId, 0, 0),
    );
  }

  /** Up to `limit` of a cluster's points, past the first `offset` (GL JS `getClusterLeaves`). */
  async getClusterLeaves(
    source: string,
    clusterId: number,
    limit = 10,
    offset = 0,
  ): Promise<Feature[]> {
    return JSON.parse(
      await this.#call(
        "queryCluster",
        source,
        "leaves",
        clusterId,
        limit,
        offset,
      ),
    );
  }

  /** The zoom at which a cluster breaks apart (GL JS `getClusterExpansionZoom`). */
  async getClusterExpansionZoom(
    source: string,
    clusterId: number,
  ): Promise<number> {
    return JSON.parse(
      await this.#call(
        "queryCluster",
        source,
        "expansionZoom",
        clusterId,
        0,
        0,
      ),
    );
  }

  /** Shows another style (a URL, or a style object) as the diff from this
   * one, like GL JS's `setStyle(style, {diff: true})`: sources defined alike
   * keep their tiles, layers alike keep their meshes, and paint eases by the
   * new style's `transition`. The camera stays. `styleload` fires once the
   * new style is in; a style with another tile grid, or other glyphs, loads
   * every tile afresh. */
  setStyle(style: string | object): void {
    this.#send(this.#call("setStyleUrl", styleUrl(style)));
  }

  /** Applies GL JS map calls as one transaction, or rejects and applies
   * none; resolves to the style generation whose frames show it in full
   * (`render` events carry it). */
  edit(build: (tx: Transaction) => void): Promise<number> {
    let commands: string;
    try {
      commands = transaction(build);
    } catch (e) {
      return Promise.reject(e);
    }
    return this.#call("edit", commands);
  }
  setFeatureState(
    feature: FeatureTarget,
    state: Record<string, unknown>,
  ): Promise<number> {
    return this.edit((tx) => tx.setFeatureState(feature, state));
  }
  removeFeatureState(
    feature: Partial<FeatureTarget>,
    key?: string,
  ): Promise<number> {
    return this.edit((tx) => tx.removeFeatureState(feature, key));
  }
  setGlobalState(name: string, value: unknown): Promise<number> {
    return this.edit((tx) => tx.setGlobalStateProperty(name, value));
  }

  /** The app's answer for a `tilerequest` of a source it feeds
   * (`{"type": "vector", "tiles": ["orbis-host:"]}`): the tile in the
   * source's encoding, or its source layers as GeoJSON (`{"roads": {...}}`,
   * which the map tiles), or null for an empty tile. */
  setHostTile(
    source: string,
    tile: { z: number; x: number; y: number },
    data: Uint8Array | ArrayBuffer | Record<string, object> | null,
  ): Promise<void> {
    const bytes =
      data instanceof ArrayBuffer
        ? new Uint8Array(data)
        : data instanceof Uint8Array
          ? data
          : undefined;
    const geojson = data && !bytes ? JSON.stringify(data) : undefined;
    return this.#call(
      "hostTile",
      source,
      tile.z,
      tile.x,
      tile.y,
      bytes,
      geojson,
    );
  }

  /** Heights in metres for tiles of a `raster-dem` source the app feeds
   * (`"tiles": ["orbis-host:"]`), `side`² a tile: one step of the app's
   * ground. Tiles a `tilerequest` asked for take them as their answer;
   * tiles drawn keep their place and memory and are written over, and the
   * batch shows whole in one frame (`elevationsshown`), terrain, hillshade
   * and the elevations the camera and labels read together. A batch given
   * before the last one shows replaces it: pace by `elevationsshown`. For
   * heights made in a worker, `elevationPort()` takes them packed. */
  setHostElevations(
    source: string,
    side: number,
    tiles: HostElevationTile[],
  ): Promise<HostElevationsAnswer> {
    const zxy = new Uint32Array(tiles.length * 3);
    const heights = new Float32Array(tiles.length * side * side);
    tiles.forEach((t, i) => {
      zxy.set([t.z, t.x, t.y], i * 3);
      heights.set(t.heights, i * side * side);
    });
    return this.#call("hostElevations", source, side, zxy, heights);
  }

  /** A port that takes `setHostElevations`'s batches straight to the map,
   * for heights an app makes in a worker of its own: post it there, send
   * `ElevationPortMessage`s on it (transferring their buffers), and hear
   * each batch's `ElevationPortReply` and, once it shows,
   * `ElevationPortShown`. The batches reach the map as the worker packed
   * them, without passing through the app's code on the page. */
  elevationPort(): MessagePort {
    const { port1, port2 } = new MessageChannel();
    this.#call("elevationPort", port2).catch(() => port1.close());
    return port1;
  }

  /** Asks for every tile of the app's source again: its data changed. Drawn
   * tiles stay until the new ones replace them. */
  reloadHostTiles(source: string): Promise<void> {
    return this.#call("reloadHostTiles", source);
  }

  /** The picture a `canvas` source shows now ({"type": "canvas", "canvas":
   * name, "coordinates": …}): the app draws it, and hands each new one in. */
  setSourceFrame(source: string, image: StyleImageData): Promise<void> {
    return this.#call(
      "sourceFrame",
      source,
      image.width,
      image.height,
      rgba(image),
    );
  }

  /** Adds an image to the style (GL JS `addImage`) for icons, text and
   * patterns; rejects when it has one by that name. */
  addImage(
    name: string,
    image: StyleImageData,
    options: StyleImageOptions = {},
  ): Promise<void> {
    return this.#call(
      "addImage",
      name,
      image.width,
      image.height,
      rgba(image),
      JSON.stringify(options),
    );
  }
  /** Replaces an image's pixels, of the same size (GL JS `updateImage`). */
  updateImage(name: string, image: StyleImageData): Promise<void> {
    return this.#call(
      "updateImage",
      name,
      image.width,
      image.height,
      rgba(image),
    );
  }
  /** Removes an image from the style (GL JS `removeImage`). */
  removeImage(name: string): Promise<void> {
    return this.#call("removeImage", name);
  }
  /** Plays, pauses or seeks (to `seconds`) a video source's video (GL JS
   * `VideoSource.play`, `pause`, `seek`); rejects for a source that isn't one. */
  playVideo(source: string): Promise<void> {
    return this.#call("controlVideo", source, "play", 0);
  }
  pauseVideo(source: string): Promise<void> {
    return this.#call("controlVideo", source, "pause", 0);
  }
  seekVideo(source: string, seconds: number): Promise<void> {
    return this.#call("controlVideo", source, "seek", seconds);
  }
  /** Whether the style has an image by this name, from its sprites or added. */
  hasImage(name: string): Promise<boolean> {
    return this.#call("hasImage", name);
  }

  /** The style as edits have left it. */
  async getStyle(): Promise<object> {
    return JSON.parse(await this.#call("getStyle"));
  }
  /** Layer `id` as the style has it, its paint and layout properties included. */
  async getLayer(id: string): Promise<Record<string, unknown> | undefined> {
    const json = await this.#call("getLayer", id);
    return json === undefined ? undefined : JSON.parse(json);
  }
  async getSource(id: string): Promise<Record<string, unknown> | undefined> {
    const json = await this.#call("getSource", id);
    return json === undefined ? undefined : JSON.parse(json);
  }
  /** The layers' ids in drawing order. */
  getLayersOrder(): Promise<string[]> {
    return this.#call("layerIds");
  }
  getSourceIds(): Promise<string[]> {
    return this.#call("sourceIds");
  }

  /** Stops drawing and releases the map's workers and DOM. Animations
   * still running resolve false. Calling it again does nothing. */
  remove(): void {
    if (this.#removed) return;
    this.#removed = true;
    for (const detach of [...this.#detach, ...this.#canvasDetach]) detach();
    for (const resolve of this.#animations.values()) resolve(false);
    this.#animations.clear();
    this.#videos.destroy();
    this.#engine.remove();
    this.#frame.remove();
  }

  #send(call: Promise<unknown>): void {
    call.catch((e) => this.#error(e));
  }

  /** Resolves with animation `id`'s end. The id arrives before any
   * frame's report can end it: frames wait for an animation frame. */
  #track(id: Promise<number>): Promise<boolean> {
    return id.then((id) => {
      if (id < 0 || this.#removed) return id < 0;
      return new Promise<boolean>((resolve) =>
        this.#animations.set(id, resolve),
      );
    });
  }

  #resize(): void {
    if (this.#removed) return;
    const ratio = this.#pixelRatio ?? devicePixelRatio;
    const [w, h] = [
      Math.round(this.canvas.clientWidth * ratio),
      Math.round(this.canvas.clientHeight * ratio),
    ];
    this.canvas.width = w;
    this.canvas.height = h;
    this.#send(this.#call("resize", w, h, ratio));
  }

  /** What the map reports: its frame's counters, events and annotations. */
  #receive(report: Report): void {
    if (this.#removed) return;
    if (report.box !== undefined) this.#showBox?.(report.box);
    if (report.points) this.annotations.place(report.points);
    if (report.stats) this.#stats = report.stats;
    for (const event of report.events) this.#emit(event);
    this.#stopped ??= report.stopped;
    if (report.stats && this.#recoveringFrom !== undefined) {
      const recoveredMs = performance.now() - this.#recoveringFrom;
      this.#recoveringFrom = undefined;
      this.#events.emit("restored", { recoveredMs, backend: this.backend });
    }
    if (report.stats && this.#events.has("render"))
      this.#events.emit("render", report.stats);
    if (report.fallback) void this.#fallBack(report.fallback);
  }

  #emit(event: EngineEvent): void {
    switch (event.type) {
      case "animationend": {
        // The map's own animations (a gesture's glide, a key's ease) end
        // here too, with ids the page never had.
        this.#animations.get(event.data.id)?.(event.data.arrived);
        this.#animations.delete(event.data.id);
        break;
      }
      case "video":
        this.#videos.request(event.data);
        break;
      case "error":
        this.#error(event.data.message);
        break;
      case "load":
        if (this.#loaded) break;
        this.#loaded = true;
        this.#events.emit("load", event.data);
        break;
      case "styleload":
        // A map started again on WebGL 2 takes the images added to the
        // last once its style is in.
        if (this.#readdImages) {
          this.#readdImages = false;
          for (const [name, image] of this.#restart.images)
            this.#call("addImage", name, ...image).catch(() => {});
        }
        this.#events.emit("styleload", event.data);
        break;
      default:
        this.#events.emit(event.type, event.data as never);
    }
  }

  /** A failure, as the map's `error` event or on the console; none once
   * removed, when calls still on their way fail for that alone, nor for
   * calls after the map stopped, which its `error` event said once. */
  #error(e: unknown): void {
    if (this.#removed) return;
    const message = String(e instanceof Error ? e.message : e);
    if (message === this.#stopped) return;
    if (this.#events.has("error")) this.#events.emit("error", { message });
    else console.error(`orbis: ${message}`);
  }
}

/** Reports, held until the page's side of the map listens. */
interface Reports {
  deliver(report: Report): void;
  listen(onreport: (report: Report) => void): void;
}

function reports(): Reports {
  const held: Report[] = [];
  let target: ((report: Report) => void) | undefined;
  return {
    deliver: (report) => {
      if (target) target(report);
      else held.push(report);
    },
    listen: (onreport) => {
      target = onreport;
      for (const report of held.splice(0)) onreport(report);
    },
  };
}

/** Creates a map filling `options.container`. Resolves once it can take calls; its style loads on.
 * It draws with WebGPU, or WebGL 2 where WebGPU gives no adapter ({@link OrbisMap.backend}),
 * unless `options.backend` picks one; rejects, saying why, where the browser has
 * neither or lacks the one picked. */
export async function createMap(options: MapOptions): Promise<OrbisMap> {
  const ratio = options.pixelRatio ?? devicePixelRatio;
  const style = styleUrl(options.style);
  const core = {
    fadeDuration: options.fadeDuration,
    crossSourceCollisions: options.crossSourceCollisions,
    localIdeographFontFamily: options.localIdeographFontFamily,
    antialias: options.antialias,
    terrainSkirts: options.terrainSkirts,
    previews: options.previews,
    stableLabels: options.stableLabels,
    tileLod: options.tileLod,
    missingTiles: options.missingTiles,
    tileFetches: options.tileFetches,
  };
  const frame = document.createElement("div");
  Object.assign(frame.style, {
    position: "relative",
    width: "100%",
    height: "100%",
    overflow: "hidden",
  });
  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    position: "absolute",
    inset: "0",
    pointerEvents: "none",
  });
  let canvas = newCanvas();
  frame.append(canvas, overlay);
  options.container.append(frame);
  // The style's first fetches start, the build is picked, downloads and
  // compiles, and the workers load.
  preloadStyle(
    typeof options.style === "string" ? style : options.style,
    ratio,
    options.camera?.zoom,
  );
  const build = pickBuild(undefined, options.backend);
  build.catch(() => {});
  // The modules the style declares download beside its TileJSON.
  void Promise.all([build, styleModules()]).then(
    ([{ backend }, names]) =>
      names.forEach((name) => loadModule(backend, name).catch(() => {})),
    () => {},
  );
  const jobWorkers = options.jobWorkers ?? defaultJobWorkers();
  const setup = {
    style,
    options: core,
    camera: options.camera,
    constraints: options.constraints,
  };
  let held = reports();
  let engine: Engine;
  try {
    engine = await launch(
      build,
      canvas,
      setup,
      ratio,
      jobWorkers,
      held.deliver,
    );
  } catch (e) {
    // WebGPU gave an adapter, yet its map didn't start on it (no device,
    // say): WebGL 2 in a new canvas, as a canvas that had a WebGPU context
    // takes no other. A trap is a bug, which WebGL 2 would hit too.
    const picked = await build.catch(() => undefined);
    const trapped =
      e instanceof Error && e.cause instanceof WebAssembly.RuntimeError;
    if (picked?.backend !== "webgpu" || picked.forced || trapped) {
      frame.remove();
      throw e;
    }
    const fresh = newCanvas();
    canvas.replaceWith(fresh);
    canvas = fresh;
    held = reports();
    const why = `its map didn't start: ${e instanceof Error ? e.message : e}`;
    try {
      engine = await launch(
        fallbackBuild(why),
        canvas,
        setup,
        ratio,
        jobWorkers,
        held.deliver,
      );
    } catch (e) {
      frame.remove();
      throw e;
    }
  }
  const map = new OrbisMap(
    options.container,
    frame,
    canvas,
    overlay,
    engine,
    held,
    {
      core,
      jobWorkers,
      interactive: options.interactive ?? true,
      style,
      constraints: options.constraints,
      images: new Map(),
    },
    options.pixelRatio,
  );
  if (options.gestures) map.setGestures(options.gestures);
  return map;
}

/** A canvas filling the map's frame. */
function newCanvas(): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  Object.assign(canvas.style, {
    display: "block",
    width: "100%",
    height: "100%",
  });
  return canvas;
}

/** The map drawing into `canvas`, in the page, at `ratio` device pixels a
 * CSS pixel, once `build` has compiled, with `jobWorkers` job workers
 * and `modules` fetched at once; `deliver` hears its reports. */
function launch(
  build: Promise<Build>,
  canvas: HTMLCanvasElement,
  setup: Omit<EngineSetup, "canvas" | "size">,
  ratio: number,
  jobWorkers: number,
  deliver: (report: Report) => void,
  modules: ModuleName[] = [],
): Promise<Engine> {
  const size: [number, number, number] = [
    Math.round(canvas.clientWidth * ratio),
    Math.round(canvas.clientHeight * ratio),
    ratio,
  ];
  [canvas.width, canvas.height] = size;
  build.catch(() => {});
  void build.then(
    ({ backend }) =>
      modules.forEach((name) => loadModule(backend, name).catch(() => {})),
    () => {},
  );
  const workers = spawnWorkers(build, jobWorkers);
  return Engine.start(build, workers, { ...setup, canvas, size }, deliver);
}

/** A style as the map loads it: a URL resolved against the page's, or a
 * style object as a `data:` URL. */
function styleUrl(style: string | object): string {
  return typeof style === "string"
    ? new URL(style, location.href).href
    : `data:application/json,${encodeURIComponent(JSON.stringify(style))}`;
}

/** The two numbers a query answered, or null when it had no answer. */
function pair(xy: Float64Array | undefined): [number, number] | null {
  const [x, y] = xy ?? [];
  return x === undefined || y === undefined ? null : [x, y];
}
