import assert from "node:assert/strict";
import { test } from "node:test";

// map.ts imports core.wasm's bindings; the transaction builder doesn't need them.
const { transaction } = await import("./transaction.ts");

test("a transaction collects GL JS calls in order as one JSON array", () => {
  const json = transaction((tx) =>
    tx
      .setPaintProperty("water", "fill-color", "blue")
      .setFeatureState({ source: "s", id: 1 }, { hover: true }),
  );
  assert.deepEqual(JSON.parse(json), [
    ["setPaintProperty", "water", "fill-color", "blue"],
    ["setFeatureState", { source: "s", id: 1 }, { hover: true }],
  ]);
});
