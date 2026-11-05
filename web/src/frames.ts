// Frame scheduling (plan §7.5): frames run on the display's refresh
// (requestAnimationFrame, or a timer in workers without it) and only while
// something asks for them: no frames when idle.

import type { FrameStats } from "./types.ts";

const nextFrame: (draw: (now: number) => void) => unknown =
  globalThis.requestAnimationFrame?.bind(globalThis) ??
  ((draw) => setTimeout(() => draw(performance.now()), 16));

/**
 * Runs `frame` on the next display frame after each `request()`, and again
 * while it returns true.
 */
export function frameLoop(frame: (now: number) => boolean, onError: (e: unknown) => void) {
  let scheduled = false;
  const request = () => {
    if (scheduled) return;
    scheduled = true;
    nextFrame(() => {
      scheduled = false;
      try {
        if (frame(performance.now())) request();
      } catch (e) {
        onError(e);
      }
    });
  };
  return { request, get scheduled() { return scheduled; } };
}

/** `FrameInfo::stats`' counters, in order (orb-web); the camera follows. */
const STATS = [
  "zoom", "generation", "idle", "meshesUploaded", "rowsUploaded", "draws", "dispatches",
  "candidates", "tilesDrawn", "tilesLoading", "bundleRecorded", "placementsWritten",
  "paintsWritten", "uploadedFromWasm", "uploadedFromJs", "texelsFromWasm", "texelsFromJs",
  "headerBytes", "jobsDone", "lost",
] as const;
const GENERATION = 1;
const IDLE = 2;
const LOST = STATS.indexOf("lost");
const CAMERA = STATS.length;

/** What crosses from the renderer to the page after a frame: small, its
 * counters packed in one transferable array. */
export interface FrameAck {
  /** CPU milliseconds the frame took where it rendered. */
  ms: number;
  /** Counters, then the frame's camera ([`ackCamera`]). */
  stats: Float64Array;
  /** Picking's changes for the ui role (render worker). */
  hits?: Uint8Array;
  /** The terrain's elevation pyramid when it changed; empty when it's gone. */
  elevation?: Uint8Array;
  /** The map's events since the last frame, if any. */
  events?: MapEventObject[];
  warnings?: string[];
}

export const ackIdle = (ack: FrameAck) => ack.stats[IDLE] !== 0;
/** The GPU was lost and nothing was drawn; the host recovers. */
export const ackLost = (ack: FrameAck) => ack.stats[LOST] !== 0;
export const ackGeneration = (ack: FrameAck) => ack.stats[GENERATION];
/** The frame's camera: [lon, lat, resolution, bearing, pitch, fov, zoom, elevation]. */
export const ackCamera = (ack: FrameAck) => ack.stats.subarray(CAMERA, CAMERA + 8);

/** A frame's counters as the package reports them. */
export function frameStats(ack: FrameAck): FrameStats {
  const stats = { ms: ack.ms, warnings: ack.warnings ?? [] } as Record<string, unknown>;
  STATS.forEach((name, i) => { stats[name] = ack.stats[i]; });
  stats.idle = ack.stats[IDLE] !== 0;
  stats.bundleRecorded = stats.bundleRecorded !== 0;
  delete stats.lost;
  return stats as unknown as FrameStats;
}

export type MapEventObject =
  | { type: "styleload"; camera: Float64Array | null }
  | { type: "styledata"; generation: number; surface: string }
  | { type: "sourcedata"; source: string; loaded: boolean }
  | { type: "load" } | { type: "idle" }
  | { type: "error"; message: string }
  | { type: "capability"; tier: "modern" | "compat" };

/** The wasm map's per-frame surface. */
export interface Renderer {
  render(): RenderInfo;
  takeEvents(): MapEventObject[];
}

/** The fields of core.wasm's `FrameInfo`. */
export interface RenderInfo {
  stats: Float64Array; hits?: Uint8Array; elevation?: Uint8Array; warnings: string[];
  free(): void;
}

/** Draws one frame and reports it as plain data that can cross to the page. */
export function renderFrame(map: Renderer): FrameAck {
  const started = performance.now();
  const info = map.render();
  const ms = performance.now() - started;
  const ack: FrameAck = { ms, stats: info.stats };
  if (info.hits) ack.hits = info.hits;
  if (info.elevation) ack.elevation = info.elevation;
  const events = map.takeEvents();
  if (events.length) ack.events = events;
  const warnings = info.warnings;
  if (warnings.length) ack.warnings = warnings;
  info.free();
  return ack;
}
