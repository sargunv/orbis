import assert from "node:assert/strict";
import { test } from "node:test";

const { preloadStyle, takePreload } = await import("./preload.ts");
const { setRequestTransform } = await import("./requests.ts");

const fetched: string[] = [];
globalThis.fetch = (async (url: string) => {
  fetched.push(url);
  const style = { sources: { a: { url: "tiles.json" }, b: { url: "https://b.test/tj" }, c: { type: "geojson" } }, sprite: "sprites/s?k=1" };
  return new Response(url === "https://a.test/style.json" ? JSON.stringify(style) : "{}");
}) as typeof fetch;

test("a style's URL, TileJSON and sprite are fetched ahead, as the map asks for them, once each", async () => {
  preloadStyle("https://a.test/style.json", 2);
  assert.ok(await takePreload("https://a.test/style.json"));
  assert.equal(takePreload("https://a.test/style.json"), undefined, "taken once");
  await new Promise((resolve) => setTimeout(resolve, 10));
  for (const url of ["https://a.test/tiles.json", "https://b.test/tj", "https://a.test/sprites/s@2x.json?k=1", "https://a.test/sprites/s@2x.png?k=1"]) {
    assert.ok(takePreload(url), url);
  }
});

test("the page's request transform applies, and a refused request isn't fetched", async () => {
  fetched.length = 0;
  setRequestTransform((url) => (url.includes("refused") ? null : { url: `${url}?signed` }));
  preloadStyle({ version: 8, sources: { a: { url: "https://c.test/refused" }, b: { url: "https://c.test/ok" } }, sprite: [] }, 1);
  await assert.rejects(takePreload("https://c.test/refused")!);
  assert.ok(await takePreload("https://c.test/ok"));
  assert.deepEqual(fetched, ["https://c.test/ok?signed"]);
  setRequestTransform(undefined);
});
