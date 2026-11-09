// Offline regions: a style's map of an area between two zooms,
// downloaded into the origin private file system. Every map on the origin
// draws from complete regions before the network, so a downloaded area
// opens offline. Each JS context reads the regions once; one that changes
// them tells the others on the `orbis-regions` broadcast channel.

/** core.wasm on this thread, loaded on first use: a page whose map runs
 * in a worker loads it only for regions. */
const core = async () => {
  const module = await import("../pkg/core.js");
  await module.default({ module_or_path: CORE });
  return module;
};

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

/** Downloads starting: a region's id is the next free one once the one
 * before has written its manifest. */
let starting: Promise<unknown> = Promise.resolve();

/** Downloads a region; `onEvent` follows its progress. Resolves to its id
 * once it starts. */
export async function downloadRegion(definition: RegionDefinition,
  onEvent?: (event: RegionEvent) => void): Promise<number> {
  const { downloadRegion: downloadCore } = await core();
  const start = starting.then(() => downloadCore(definition.name, definition.style,
    new Float64Array(definition.bounds), new Float64Array(definition.zooms), (event: RegionEvent) => {
      if (event.type === "complete") changed();
      onEvent?.(event);
    }));
  starting = start.catch(() => {});
  return start;
}

/** Deletes a region, stopping its download. */
export async function deleteRegion(id: number): Promise<void> {
  const { deleteRegion: deleteCore } = await core();
  await deleteCore(id);
  changed();
}

/** The regions on this origin. */
export async function regions(): Promise<Region[]> {
  const { listRegions } = await core();
  return JSON.parse(await listRegions());
}
