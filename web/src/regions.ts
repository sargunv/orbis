// Offline regions (plan D24): a style's map of an area between two zooms,
// downloaded into the origin private file system. Every map on the origin
// draws from complete regions before the network, so a downloaded area
// opens offline. Each JS context reads the regions once; one that changes
// them tells the others on the `orbis-regions` broadcast channel.

import init, { deleteRegion as deleteCore, downloadRegion as downloadCore, listRegions } from "../pkg/core.js";

const CORE = new URL("../pkg/core_bg.wasm", import.meta.url);
const CHANNEL = "orbis-regions";

export interface RegionDefinition {
  name: string;
  /** A v8 style URL. */
  style: string;
  /** West, south, east, north, in degrees. */
  bounds: [number, number, number, number];
  /** The map zooms the region covers, inclusive. */
  zooms: [number, number];
}

export interface Region extends RegionDefinition {
  id: number;
  /** Whether its download finished; only complete regions are drawn from. */
  complete: boolean;
  bytes: number;
}

export type RegionEvent =
  | { type: "progress"; id: number; done: number; total: number; bytes: number }
  | { type: "complete"; id: number; bytes: number; failed: number }
  | { type: "failed"; id: number; message: string };

/** Calls `reload` whenever another context changes the regions. */
export function onRegionsChanged(reload: () => void): () => void {
  const channel = new BroadcastChannel(CHANNEL);
  channel.onmessage = () => reload();
  return () => channel.close();
}

function changed(): void {
  const channel = new BroadcastChannel(CHANNEL);
  channel.postMessage(null);
  channel.close();
}

/** Downloads a region; `onEvent` follows its progress. Resolves to its id
 * once it starts. */
export async function downloadRegion(definition: RegionDefinition,
  onEvent?: (event: RegionEvent) => void): Promise<number> {
  await init({ module_or_path: CORE });
  return downloadCore(definition.name, definition.style, new Float64Array(definition.bounds),
    new Float64Array(definition.zooms), (event: RegionEvent) => {
      if (event.type === "complete") changed();
      onEvent?.(event);
    });
}

/** Deletes a region, stopping its download. */
export async function deleteRegion(id: number): Promise<void> {
  await init({ module_or_path: CORE });
  await deleteCore(id);
  changed();
}

/** The regions on this origin. */
export async function regions(): Promise<Region[]> {
  await init({ module_or_path: CORE });
  return JSON.parse(await listRegions());
}
