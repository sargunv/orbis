// Loads shaper.wasm: `font-faces` files drawn with HarfRust
// and skrifa, as natively, outside core.wasm. The map calls this once per
// thread, the first time a style declares a font file.
export async function loadShaper(): Promise<unknown> {
  const shaper = await import("../pkg/shaper.js");
  await shaper.default();
  return new shaper.Shaper();
}
