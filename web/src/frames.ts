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

/** What crosses from the renderer to the page after each frame. */
export interface FrameAck extends FrameStats {
  /** The frame's camera: [lon, lat, resolution, bearing, pitch, fov, zoom, elevation]. */
  camera: Float64Array;
  hitSnapshot?: Uint8Array;
  /** The terrain's elevation pyramid when it changed; empty when it's gone. */
  elevation?: Uint8Array;
  /** The map's events since the last frame. */
  events: MapEventObject[];
}

export type MapEventObject =
  | { type: "styleload"; camera: Float64Array | null; projection: string }
  | { type: "styledata"; generation: number }
  | { type: "sourcedata"; source: string; loaded: boolean }
  | { type: "load" } | { type: "idle" }
  | { type: "error"; message: string };

/** The wasm map's per-frame surface. */
export interface Renderer {
  render(): RenderInfo;
  takeEvents(): MapEventObject[];
}

/** The fields of core.wasm's `FrameInfo`. */
export interface RenderInfo {
  zoom: number; camera: Float64Array; hit_snapshot?: Uint8Array; elevation?: Uint8Array;
  generation: number;
  meshes_uploaded: number; rows_uploaded: number; draws: number; dispatches: number;
  candidates: number; tiles_drawn: number; tiles_loading: number; bundle_recorded: boolean;
  placements_written: number; paints_written: number; idle: boolean; warnings: string[];
  uploaded_from_wasm: number; uploaded_from_js: number; texels_from_wasm: number;
  texels_from_js: number; header_bytes: number; jobs_done: number;
  free(): void;
}

/** Draws one frame and reports it as plain data that can cross to the page. */
export function renderFrame(map: Renderer): FrameAck {
  const started = performance.now();
  const info = map.render();
  const ms = performance.now() - started;
  const ack: FrameAck = {
    ms,
    camera: info.camera,
    hitSnapshot: info.hit_snapshot,
    elevation: info.elevation,
    events: map.takeEvents(),
    zoom: info.zoom,
    generation: info.generation,
    meshesUploaded: info.meshes_uploaded,
    rowsUploaded: info.rows_uploaded,
    draws: info.draws,
    dispatches: info.dispatches,
    candidates: info.candidates,
    tilesDrawn: info.tiles_drawn,
    tilesLoading: info.tiles_loading,
    bundleRecorded: info.bundle_recorded,
    placementsWritten: info.placements_written,
    paintsWritten: info.paints_written,
    idle: info.idle,
    warnings: info.warnings,
    uploadedFromWasm: info.uploaded_from_wasm,
    uploadedFromJs: info.uploaded_from_js,
    texelsFromWasm: info.texels_from_wasm,
    texelsFromJs: info.texels_from_js,
    headerBytes: info.header_bytes,
    jobsDone: info.jobs_done,
  };
  info.free();
  return ack;
}
