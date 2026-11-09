// What a new map fetches first, fetched while core.wasm downloads and
// compiles: its style, its sources' TileJSON and its sprite. The map's own
// fetches of those URLs take these responses (once each, through the
// page's request transform) rather than fetch again; one not taken in a
// minute is dropped.

import { transformed } from "./requests.ts";

const preloads = new Map<string, Promise<Response>>();

/** The response fetched ahead for `url`, once, if there is one. */
export function takePreload(url: string): Promise<Response> | undefined {
  const response = preloads.get(url);
  preloads.delete(url);
  return response;
}

function preload(url: string): void {
  if (preloads.has(url) || !/^https?:/.test(url)) return;
  const response = transformed(url).then((request) => {
    if (!request) throw new Error(`${url}: refused`);
    return fetch(request.url, { headers: request.headers });
  });
  // Unclaimed, it fails quietly: the map fetches again.
  response.catch(() => {});
  preloads.set(url, response);
  setTimeout(() => { if (preloads.get(url) === response) preloads.delete(url); }, 60_000);
}

/** Starts fetching what a map of `style` (a URL or a document) at
 * `pixelRatio` will fetch first. */
export function preloadStyle(style: string | object, pixelRatio: number): void {
  if (typeof style === "object") return preloadParts(style as StyleParts, undefined, pixelRatio);
  if (!/^https?:/.test(style)) return;
  preload(style);
  // A clone to read: the map takes the original.
  void preloads.get(style)?.then((r) => r.ok ? r.clone().json() : undefined)
    .then((doc) => { if (doc) preloadParts(doc, style, pixelRatio); }, () => {});
}

interface StyleParts {
  sources?: Record<string, { url?: unknown }>;
  sprite?: unknown;
}

/** The style's TileJSON and sprite, at the URLs the map asks for. */
function preloadParts(doc: StyleParts, base: string | undefined, pixelRatio: number): void {
  const resolve = (url: string) => {
    try {
      return base === undefined || /^https?:/.test(url) ? url : new URL(url, base).href;
    } catch {
      return undefined;
    }
  };
  for (const source of Object.values(doc.sources ?? {})) {
    const url = typeof source?.url === "string" ? resolve(source.url) : undefined;
    if (url) preload(url);
  }
  const sprites = typeof doc.sprite === "string" ? [doc.sprite]
    : Array.isArray(doc.sprite) ? doc.sprite.map((s) => (s as { url?: unknown })?.url) : [];
  for (const sprite of sprites) {
    const url = typeof sprite === "string" ? resolve(sprite) : undefined;
    if (!url) continue;
    // orb-runtime's `sprite::resource_url`.
    const split = url.search(/[?#]/) < 0 ? url.length : url.search(/[?#]/);
    const scale = pixelRatio > 1 ? "@2x" : "";
    for (const extension of [".json", ".png"]) preload(`${url.slice(0, split)}${scale}${extension}${url.slice(split)}`);
  }
}
