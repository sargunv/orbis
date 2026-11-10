// What a new map fetches first, fetched while core.wasm downloads and
// compiles: its style, its sources' TileJSON, its sprite and its labels'
// first glyphs. The map's own
// fetches of those URLs take these responses (once each, through the
// page's request transform) rather than fetch again; one not taken in a
// minute is dropped.

import { transformed } from "./requests.ts";

const preloads = new Map<
  string,
  { response: Promise<Response>; expiry: ReturnType<typeof setTimeout> }
>();

/** The response fetched ahead for `url`, once, if there is one. */
export function takePreload(url: string): Promise<Response> | undefined {
  const preloaded = preloads.get(url);
  if (!preloaded) return undefined;
  preloads.delete(url);
  clearTimeout(preloaded.expiry);
  return preloaded.response;
}

function preload(url: string): void {
  if (preloads.has(url) || !/^https?:/.test(url)) return;
  const response = transformed(url).then((request) => {
    if (!request) throw new Error(`${url}: refused`);
    return fetch(request.url, { headers: request.headers });
  });
  // Unclaimed, it fails quietly: the map fetches again.
  response.catch(() => {});
  const expiry = setTimeout(() => {
    if (preloads.get(url)?.response === response) preloads.delete(url);
  }, 60_000);
  preloads.set(url, { response, expiry });
}

/** Starts fetching what a map of `style` (a URL or a document) at
 * `pixelRatio` will fetch first; with the camera's `zoom`, the first glyph
 * ranges of its labels at that zoom too (the style's zoom without one). */
export function preloadStyle(
  style: string | object,
  pixelRatio: number,
  zoom?: number,
): void {
  if (typeof style === "object")
    return preloadParts(style as StyleParts, undefined, pixelRatio, zoom);
  if (!/^https?:/.test(style)) return;
  preload(style);
  // A clone to read: the map takes the original.
  void preloads
    .get(style)
    ?.response.then((r) => (r.ok ? r.clone().json() : undefined))
    .then(
      (doc) => {
        if (doc) preloadParts(doc, style, pixelRatio, zoom);
      },
      () => {},
    );
}

interface StyleParts {
  sources?: Record<string, { url?: unknown }>;
  sprite?: unknown;
  glyphs?: unknown;
  zoom?: unknown;
  layers?: {
    type?: unknown;
    minzoom?: unknown;
    maxzoom?: unknown;
    layout?: {
      visibility?: unknown;
      "text-field"?: unknown;
      "text-font"?: unknown;
    };
  }[];
}

/** The style spec's `text-font` default. */
const DEFAULT_FONT = ["Open Sans Regular", "Arial Unicode MS Regular"];

/** The style's TileJSON, sprite and first glyphs, at the URLs the map asks for. */
function preloadParts(
  doc: StyleParts,
  base: string | undefined,
  pixelRatio: number,
  zoom?: number,
): void {
  const resolve = (url: string) => {
    try {
      return base === undefined || /^https?:/.test(url)
        ? url
        : new URL(url, base).href;
    } catch {
      return undefined;
    }
  };
  for (const source of Object.values(doc.sources ?? {})) {
    const url =
      typeof source?.url === "string" ? resolve(source.url) : undefined;
    if (url) preload(url);
  }
  const sprites =
    typeof doc.sprite === "string"
      ? [doc.sprite]
      : Array.isArray(doc.sprite)
        ? doc.sprite.map((s) => (s as { url?: unknown })?.url)
        : [];
  for (const sprite of sprites) {
    const url = typeof sprite === "string" ? resolve(sprite) : undefined;
    if (!url) continue;
    // orb-runtime's `sprite::resource_url`.
    const split = url.search(/[?#]/) < 0 ? url.length : url.search(/[?#]/);
    const scale = pixelRatio > 1 ? "@2x" : "";
    for (const extension of [".json", ".png"])
      preload(`${url.slice(0, split)}${scale}${extension}${url.slice(split)}`);
  }
  // The template's braces stay as the map has them.
  const template =
    typeof doc.glyphs === "string"
      ? resolve(doc.glyphs)?.replace(/%7B/gi, "{").replace(/%7D/gi, "}")
      : undefined;
  const at = zoom ?? (typeof doc.zoom === "number" ? doc.zoom : 0);
  for (const stack of template ? firstFonts(doc, at) : []) {
    // orb-text's `glyphs_url`: the first range, 0–255.
    preload(
      template!.replace("{fontstack}", stack).replace("{range}", "0-255"),
    );
  }
}

/** The font stacks, joined, of the labels shown at `zoom` whose
 * `text-font` is a constant: those whose first range a map asks for
 * before its tiles do (orb-runtime's `TileWorker::first_glyphs`). */
function firstFonts(doc: StyleParts, zoom: number): Set<string> {
  const stacks = new Set<string>();
  for (const layer of Array.isArray(doc.layers) ? doc.layers : []) {
    const layout = layer?.layout ?? {};
    const shown =
      layer?.type === "symbol" &&
      layout.visibility !== "none" &&
      (typeof layer.minzoom !== "number" || zoom >= layer.minzoom) &&
      (typeof layer.maxzoom !== "number" || zoom < layer.maxzoom);
    if (
      !shown ||
      layout["text-field"] === undefined ||
      layout["text-field"] === ""
    )
      continue;
    const font = layout["text-font"];
    const fonts =
      font === undefined
        ? DEFAULT_FONT
        : Array.isArray(font) && font[0] === "literal"
          ? font[1]
          : font;
    if (
      Array.isArray(fonts) &&
      fonts.length &&
      fonts.every((f) => typeof f === "string")
    )
      stacks.add(fonts.join(","));
  }
  return stacks;
}
