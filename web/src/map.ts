// The page's side of a map: its DOM, input, events, annotations and video
// players, and the API. The map itself (engine.ts: the wasm map, its
// camera, gestures, drawing and queries) runs in the map worker where the
// browser can draw into an OffscreenCanvas there and deliver
// pointer moves as they come (`pointerrawupdate`): Chromium-based
// browsers. Elsewhere it runs on the page. Either way tile jobs and labels
// run in workers of their own, and the page reaches the map through a
// link (link.ts): calls that answer with promises, and reports after
// frames.

import { Annotations } from "./annotations.ts";
import { pickBuild, type BackendChoice } from "./builds.ts";
import type { EngineEvent, MapState, Pixels, Report } from "./engine.ts";
import { Emitter, type Listener } from "./events.ts";
import { allowed, attachInput, type Allowed } from "./input.ts";
import { directLink, workerLink, type Link } from "./link.ts";
import { preloadStyle } from "./preload.ts";
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
import {
  defaultJobWorkers,
  spawnWorkers,
  terminateWorkers,
} from "./workers.ts";

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

/** The camera: its state, moves and limits. The state is the map's as of
 * its last frame or answer: a move shows in it once the map has it. */
export class CameraControl {
  #link: Link;
  #track: (id: Promise<number>) => Promise<boolean>;
  #fail: (e: unknown) => void;

  /** @internal */
  constructor(
    link: Link,
    track: (id: Promise<number>) => Promise<boolean>,
    fail: (e: unknown) => void,
  ) {
    this.#link = link;
    this.#track = track;
    this.#fail = fail;
  }

  #state(): MapState {
    return this.#link.state();
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
    this.#send(this.#link.call("jumpToFree", camera));
  }

  /** Whether the center stands on the terrain at rest (GL JS
   * `centerClampedToGround`, true by default). Off, it keeps the elevation
   * it is given, and the camera orbits a point in the air. */
  get centerClampedToGround(): boolean {
    return this.#state().clamped;
  }
  set centerClampedToGround(clamped: boolean) {
    this.#send(this.#link.call("setCenterClampedToGround", clamped));
  }

  /** Puts the center this many metres above sea level at once (GL JS
   * `setCenterElevation`); while clamped to the ground, the terrain takes it back. */
  setCenterElevation(elevation: number): void {
    this.#send(this.#link.call("setCenterElevation", elevation));
  }

  /** Whether an animation or a gesture is moving the camera. */
  get moving(): boolean {
    return this.#state().moving;
  }

  /** Moves at once. Each move takes only the properties it sets: animations
   * of the others go on, as in MapLibre Native, and one left with nothing to
   * move is interrupted. {@link stop} ends them all. */
  jumpTo(options: CameraOptions): void {
    this.#send(this.#link.call("jumpTo", options));
  }

  /** Starts at the next frame; resolves true on arrival, false when interrupted. */
  easeTo(options: EaseOptions): Promise<boolean> {
    return this.#track(this.#link.call("easeTo", options));
  }

  /** Flies along GL JS's zoom-out curve from the next frame; resolves
   * true on arrival, false when interrupted. */
  flyTo(options: FlyOptions): Promise<boolean> {
    return this.#track(this.#link.call("flyTo", options));
  }

  /** The camera that shows `bounds`, or null when the padding leaves no room. */
  cameraForBounds(
    bounds: Bounds,
    options: FitOptions = {},
  ): Promise<CameraOptions | null> {
    return this.#link.call(
      "cameraForBounds",
      new Float64Array(bounds),
      options,
    );
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
    return this.#track(this.#link.call("startMomentum", momentum));
  }

  /** Stops every running animation; their promises resolve false. */
  stop(): void {
    this.#send(this.#link.call("stop"));
  }

  setConstraints(constraints: Constraints): void {
    this.#send(this.#link.call("setConstraints", constraints));
  }
}

/** A map in a container element. Create one with {@link createMap}. */
export class OrbisMap {
  readonly container: HTMLElement;
  readonly camera: CameraControl;
  readonly annotations: Annotations;
  #link: Link;
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
    link: Link,
    interactive: boolean,
    pixelRatio?: number,
  ) {
    this.container = container;
    this.#frame = frame;
    this.#canvas = canvas;
    this.#link = link;
    this.#pixelRatio = pixelRatio;
    const fail = (e: unknown) => this.#error(e);
    this.#videos = new Videos(
      () =>
        this.#removed
          ? undefined
          : {
              videoFrame: (source, video) => {
                // A worker takes the frame as a bitmap, closed if it never gets there.
                if (!link.worker) {
                  link.call("videoFrame", source, video).catch(() => {});
                  return;
                }
                createImageBitmap(video).then(
                  (bitmap) =>
                    link
                      .call("videoFrame", source, bitmap)
                      .catch(() => bitmap.close()),
                  () => {},
                );
              },
              videoFailed: (source, message) => {
                link.call("videoFailed", source, message).catch(fail);
              },
            },
      () => {},
    );
    this.camera = new CameraControl(link, (id) => this.#track(id), fail);
    this.annotations = new Annotations(overlay, (points) => {
      link.call("setAnnotations", points).then((pixels) => {
        if (pixels) this.annotations.place(pixels);
      }, fail);
    });
    if (interactive) {
      const input = attachInput(
        canvas,
        (kind, args, time) => {
          if (!this.#removed) link.input(kind, args, time);
        },
        link.worker,
      );
      this.#showBox = input.showBox;
      this.#allow = input.allow;
      this.#detach.push(input.detach);
    }
    const observer = new ResizeObserver(() => this.#resize());
    observer.observe(canvas);
    this.#detach.push(() => observer.disconnect());
    // Tiles that failed while offline load again when the network returns.
    const online = () => {
      link.call("retryFailedTiles").catch(fail);
    };
    addEventListener("online", online);
    this.#detach.push(() => removeEventListener("online", online));
    link.listen((report) => this.#receive(report));
  }

  /** Sets what the map's gestures may move, how releases continue and
   * when presses are clicks (MapLibre Compose's `MapInteractions`); what
   * `options` leaves out takes its default. A wheel or key whose movement
   * is off scrolls the page as it would without the map. A change cancels
   * a gesture under way. */
  setGestures(options: GestureOptions): void {
    this.#allow?.(allowed(options));
    this.#link.call("setGestures", options).catch((e) => this.#error(e));
  }

  /** The canvas the map draws into and takes input from. */
  get canvas(): HTMLCanvasElement {
    return this.#canvas;
  }

  /** Whether the map runs in a worker (the default where the browser
   * allows it, {@link MapOptions.worker}), not on the page's thread. */
  get inWorker(): boolean {
    return this.#link.worker;
  }

  /** The GPU API the map draws with: WebGPU, or WebGL 2 where the browser
   * gives WebGPU no adapter or the map asked for it ({@link
   * MapOptions.backend}). What WebGL 2 draws otherwise than WebGPU, the
   * map says with `degraded` events (#140). */
  get backend(): "webgpu" | "webgl2" {
    return this.#link.backend;
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
    return imageData(await this.#link.call("snapshot", width, height, scale));
  }

  /** Loses the GPU on purpose, to exercise recovery. */
  loseDevice(): void {
    this.#send(this.#link.call("loseDevice"));
  }
  /** Traps the placement worker's wasm, as a bug would, to exercise its
   * replacement: labels place again within a frame or two. */
  trapPlacementWorker(): void {
    this.#send(this.#link.call("trapPlacementWorker"));
  }
  /** Traps every job worker's wasm, as a bug would, to exercise their
   * replacement: tiles load again on the new workers. */
  trapJobWorkers(): void {
    this.#send(this.#link.call("trapJobWorkers"));
  }
  /** Traps the map's own wasm, as a bug would, to exercise how it fails:
   * one `error` event says the map stopped, and every call after rejects
   * with the same message. */
  trap(): void {
    this.#send(this.#link.call("trap"));
  }
  /** For test runners: frames show `time` (milliseconds on the page's
   * clock, as `performance.now()` gives it) instead of their animation
   * frame's, never earlier than the last frame, as under fake timers:
   * transitions and fades move only when the clock does. Without `time`,
   * frames follow the animation frames again. Resolves once the map has
   * it, so every frame it reports after that shows `time`. */
  setFrameClock(time?: number): Promise<void> {
    return this.#link.call("setFrameClock", time);
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
    return imageData(await this.#link.call("readPixels", premultiplied));
  }

  /** Draws a frame even if nothing changed (GL JS `triggerRepaint`). */
  triggerRepaint(): void {
    this.#send(this.#link.call("triggerRepaint"));
  }

  /** Times the GPU's work from now on, for measurements: `gpuTimings`
   * collects it. WebGPU only. */
  setGpuTiming(on: boolean): void {
    this.#send(this.#link.call("setGpuTiming", on));
  }
  /** Has the cull keep every candidate, for measurements of what it saves:
   * the map draws the same, only slower. */
  setKeepAllCull(on: boolean): void {
    this.#send(this.#link.call("setKeepAllCull", on));
  }
  /** Sets how tiles coarsen with distance in pitched views; fields left out stay as they were. */
  setTileLod(lod: TileLod): void {
    this.#send(this.#link.call("setTileLod", lod));
  }

  /** Each GPU submission's times since the last call, once the GPU has
   * finished them, in order: its passes' GPU milliseconds by label (where
   * the device has timestamp queries) and the CPU's to submit them. */
  gpuTimings(): Promise<GpuSubmission[]> {
    return this.#link.call("gpuTimings");
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
    const p = await this.#link.call("project", lngLat[0], lngLat[1], camera);
    return pair(p);
  }

  /** The ground point under a pixel, on the terrain, or null above the
   * horizon: now, or after a jump to `camera`. */
  async unproject(
    point: Point,
    camera?: CameraOptions,
  ): Promise<LngLat | null> {
    const g = await this.#link.call("unproject", point[0], point[1], camera);
    return pair(g);
  }

  /** The topmost feature under a pixel, from the last frame. */
  hitTest(point: Point): Promise<Hit | null> {
    return this.#link.call("hitTest", point[0], point[1]);
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
      await this.#link.call(
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
      await this.#link.call(
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
      await this.#link.call(
        "queryCluster",
        source,
        "children",
        clusterId,
        0,
        0,
      ),
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
      await this.#link.call(
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
      await this.#link.call(
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
    this.#send(this.#link.call("setStyleUrl", styleUrl(style)));
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
    return this.#link.call("edit", commands);
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
    return this.#link.call(
      "hostTile",
      source,
      tile.z,
      tile.x,
      tile.y,
      bytes,
      geojson,
    );
  }

  /** Asks for every tile of the app's source again: its data changed. Drawn
   * tiles stay until the new ones replace them. */
  reloadHostTiles(source: string): Promise<void> {
    return this.#link.call("reloadHostTiles", source);
  }

  /** The picture a `canvas` source shows now ({"type": "canvas", "canvas":
   * name, "coordinates": …}): the app draws it, and hands each new one in. */
  setSourceFrame(source: string, image: StyleImageData): Promise<void> {
    return this.#link.call(
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
    return this.#link.call(
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
    return this.#link.call(
      "updateImage",
      name,
      image.width,
      image.height,
      rgba(image),
    );
  }
  /** Removes an image from the style (GL JS `removeImage`). */
  removeImage(name: string): Promise<void> {
    return this.#link.call("removeImage", name);
  }
  /** Plays, pauses or seeks (to `seconds`) a video source's video (GL JS
   * `VideoSource.play`, `pause`, `seek`); rejects for a source that isn't one. */
  playVideo(source: string): Promise<void> {
    return this.#link.call("controlVideo", source, "play", 0);
  }
  pauseVideo(source: string): Promise<void> {
    return this.#link.call("controlVideo", source, "pause", 0);
  }
  seekVideo(source: string, seconds: number): Promise<void> {
    return this.#link.call("controlVideo", source, "seek", seconds);
  }
  /** Whether the style has an image by this name, from its sprites or added. */
  hasImage(name: string): Promise<boolean> {
    return this.#link.call("hasImage", name);
  }

  /** The style as edits have left it. */
  async getStyle(): Promise<object> {
    return JSON.parse(await this.#link.call("getStyle"));
  }
  /** Layer `id` as the style has it, its paint and layout properties included. */
  async getLayer(id: string): Promise<Record<string, unknown> | undefined> {
    const json = await this.#link.call("getLayer", id);
    return json === undefined ? undefined : JSON.parse(json);
  }
  async getSource(id: string): Promise<Record<string, unknown> | undefined> {
    const json = await this.#link.call("getSource", id);
    return json === undefined ? undefined : JSON.parse(json);
  }
  /** The layers' ids in drawing order. */
  getLayersOrder(): Promise<string[]> {
    return this.#link.call("layerIds");
  }
  getSourceIds(): Promise<string[]> {
    return this.#link.call("sourceIds");
  }

  /** Stops drawing and releases the map's workers and DOM. Animations
   * still running resolve false. Calling it again does nothing. */
  remove(): void {
    if (this.#removed) return;
    this.#removed = true;
    for (const detach of this.#detach) detach();
    for (const resolve of this.#animations.values()) resolve(false);
    this.#animations.clear();
    this.#videos.destroy();
    this.#link.destroy();
    this.#frame.remove();
  }

  #send(call: Promise<unknown>): void {
    call.catch((e) => this.#error(e));
  }

  /** Resolves with animation `id`'s end. The id arrives before any
   * frame's report can end it: in a worker, the call's reply is posted in
   * the task that started the animation, before its frames, and this runs
   * in a microtask of the reply's message; on the page, frames wait for
   * an animation frame. */
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
    // A worker sizes its offscreen canvas itself.
    if (!this.#link.worker) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.#send(this.#link.call("resize", w, h, ratio));
  }

  /** What the map reports: its frame's counters, events and annotations. */
  #receive(report: Report): void {
    if (this.#removed) return;
    if (report.box !== undefined) this.#showBox?.(report.box);
    if (report.points) this.annotations.place(report.points);
    if (report.stats) this.#stats = report.stats;
    for (const event of report.events) this.#emit(event);
    if (report.stats && this.#events.has("render"))
      this.#events.emit("render", report.stats);
    this.#stopped ??= report.stopped;
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

/** Whether this browser runs the map in a worker: it can hand a canvas to
 * one, and tells the page of pointer moves as they come. Without that
 * (Safari) the page hears of a move as its own frame begins, when the
 * worker's has too: forwarded then, input shows two frames later than on
 * the page (in Chromium, pointer moves so forwarded showed a pan 72 ms
 * after input, against 39 ms on the page). Chromium-based browsers and
 * Firefox can; whether the worker has WebGPU, it says. */
const workerCapable = () =>
  typeof HTMLCanvasElement.prototype.transferControlToOffscreen ===
    "function" && "onpointerrawupdate" in HTMLElement.prototype;

/** Whether this is Firefox on macOS with a WebGPU adapter, where a map
 * asking for anything but WebGL 2 draws on the page's thread: Firefox shows
 * older frames of a WebGPU canvas a worker draws whenever the page's DOM
 * changes, as a map's compass and markers change it each frame, and
 * slows until it freezes (https://bugzilla.mozilla.org/show_bug.cgi?id=2091844;
 * its WebGL 2 is unaffected). Drawn on the page, every frame shows in
 * order. Firefox 159 still shows some: three 60 s Alps flights in a worker
 * (`pnpm flight --browser=firefox --record=1`) composited one stale frame
 * in 739, and none in 911 on the page. Remove once Firefox fixes it. */
async function staleWorkerFrames(backend?: BackendChoice): Promise<boolean> {
  const agent = navigator.userAgent;
  if (
    backend === "webgl2" ||
    !/ Firefox\//.test(agent) ||
    !/Mac OS X/.test(agent)
  )
    return false;
  return !!(await navigator.gpu?.requestAdapter().catch(() => null));
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
  const canvas = () => {
    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, {
      display: "block",
      width: "100%",
      height: "100%",
    });
    frame.replaceChildren(canvas, overlay);
    return canvas;
  };
  let element = canvas();
  options.container.append(frame);
  const size = (): [number, number, number] => [
    Math.round(element.clientWidth * ratio),
    Math.round(element.clientHeight * ratio),
    ratio,
  ];
  const setup = {
    style,
    options: core,
    camera: options.camera,
    constraints: options.constraints,
  };
  const preload = typeof options.style === "string" ? style : options.style;
  let link: Link | string | undefined;
  try {
    if (
      (options.worker ?? true) &&
      workerCapable() &&
      !(await staleWorkerFrames(options.backend))
    ) {
      // The worker's canvas starts at the element's size.
      [element.width, element.height] = size();
      link = await workerLink(element, {
        // The page's count of cores sizes the job workers, as on the page.
        ...setup,
        preload,
        size: size(),
        jobWorkers: options.jobWorkers ?? defaultJobWorkers(),
        origin: performance.timeOrigin,
        pageWebgpu: "gpu" in navigator,
        backend: options.backend ?? "auto",
      });
      // A worker that can't draw leaves the page to run the map, on a
      // canvas of its own.
      if (typeof link === "string") element = canvas();
    }
    if (typeof link !== "object")
      link = await startOnPage(
        element,
        setup,
        preload,
        size(),
        options.jobWorkers,
        options.backend,
      );
  } catch (e) {
    frame.remove();
    throw e;
  }
  const map = new OrbisMap(
    options.container,
    frame,
    element,
    overlay,
    link,
    options.interactive ?? true,
    options.pixelRatio,
  );
  if (options.gestures) map.setGestures(options.gestures);
  return map;
}

/** The map on the page's thread. */
async function startOnPage(
  canvas: HTMLCanvasElement,
  setup: {
    style: string;
    options: object;
    camera?: CameraOptions;
    constraints?: Constraints;
  },
  preload: string | object,
  size: [number, number, number],
  jobWorkers?: number,
  backend?: BackendChoice,
): Promise<Link> {
  // The style's first fetches start, the build is picked, downloads and
  // compiles, and the workers load.
  preloadStyle(preload, size[2], setup.camera?.zoom);
  const build = pickBuild(undefined, backend);
  build.catch(() => {});
  // A page whose map runs in a worker never loads the bindings.
  const engine = import("./engine.ts");
  const workers = spawnWorkers(build, jobWorkers ?? defaultJobWorkers());
  [canvas.width, canvas.height] = size;
  const { Engine } = await engine.catch((e) => {
    terminateWorkers(workers);
    throw e;
  });
  return directLink((report) =>
    Engine.start(build, workers, { ...setup, canvas, size, clock: 0 }, report),
  );
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
