// Frame scheduling (plan §7.5): frames run on the display's refresh
// (requestAnimationFrame) and only while something asks for them: no
// frames when idle.

import type { FrameStats } from "./types.ts";
import type { VideoEventObject } from "./videos.ts";

/**
 * Runs `frame` on the next display frame after each `request()`, and again
 * while it returns true.
 */
export function frameLoop(frame: (now: number) => boolean, onError: (e: unknown) => void) {
  let scheduled = false;
  const request = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
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

/** `FrameInfo::stats`' counters, in order (orb-web); the memory follows. */
const STATS = [
  "zoom", "generation", "idle", "meshesUploaded", "rowsUploaded", "draws", "dispatches",
  "candidates", "tilesDrawn", "tilesLoading", "bundleRecorded", "placementsWritten",
  "paintsWritten", "uploadedFromWasm", "uploadedFromJs", "texelsFromWasm", "texelsFromJs",
  "headerBytes", "jobsDone", "pipelinesPending", "layersWaiting", "lost",
] as const;
/** `FrameInfo::stats`' memory words after the counters (orb-web's
 * `memory_stats`), in bytes but `cachedTiles`. */
const MEMORY = [
  "budget", "allocated", "verticesUsed", "verticesAllocated", "verticesCeiling",
  "verticesLargestFree", "indicesUsed", "indicesAllocated", "indicesCeiling", "compactedUsed",
  "compactedAllocated", "propsUsed", "propsAllocated", "textures", "cachedTiles", "cached",
  "cacheBudget",
] as const;
const GENERATION = 1;
const IDLE = 2;
const LOST = STATS.indexOf("lost");

/** What a frame reports: its counters packed in one array. */
export interface FrameAck {
  /** CPU milliseconds the frame took. */
  ms: number;
  stats: Float64Array;
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

/** A frame's counters as the package reports them. */
export function frameStats(ack: FrameAck): FrameStats {
  const stats = { ms: ack.ms, warnings: ack.warnings ?? [] } as Record<string, unknown>;
  STATS.forEach((name, i) => { stats[name] = ack.stats[i]; });
  const memory = {} as Record<string, number>;
  MEMORY.forEach((name, i) => { memory[name] = ack.stats[STATS.length + i] ?? 0; });
  stats.memory = memory;
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
  | { type: "styleimagemissing"; id: string }
  | { type: "capability"; tier: "modern" | "compat" }
  | VideoEventObject;

/** The wasm map's per-frame surface. */
export interface Renderer {
  render(): RenderInfo;
  takeEvents(): MapEventObject[];
}

/** The fields of core.wasm's `FrameInfo`. */
export interface RenderInfo {
  stats: Float64Array; elevation?: Uint8Array; warnings: string[];
  free(): void;
}

/** Draws one frame and reports it as plain data. */
export function renderFrame(map: Renderer): FrameAck {
  const started = performance.now();
  const info = map.render();
  const ms = performance.now() - started;
  const ack: FrameAck = { ms, stats: info.stats };
  if (info.elevation) ack.elevation = info.elevation;
  const events = map.takeEvents();
  if (events.length) ack.events = events;
  const warnings = info.warnings;
  if (warnings.length) ack.warnings = warnings;
  info.free();
  return ack;
}
