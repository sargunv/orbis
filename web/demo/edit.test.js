import assert from "node:assert/strict";
import { test } from "node:test";
import { editApi } from "./edit.js";

test("a transaction's calls reach the map as one JSON array", async () => {
  const sent = [];
  const api = editApi(async (commands) => { sent.push(JSON.parse(commands)); return sent.length; });
  const generation = await api.edit((tx) => tx
    .setPaintProperty("water", "fill-color", "blue")
    .setFilter("roads", ["==", ["get", "class"], "primary"]));
  assert.equal(generation, 1);
  assert.deepEqual(sent[0], [
    ["setPaintProperty", "water", "fill-color", "blue"],
    ["setFilter", "roads", ["==", ["get", "class"], "primary"]],
  ]);
  await api.setFeatureState({ source: "s", id: 7 }, { hover: true });
  await api.setGlobalState("accent", "red");
  assert.deepEqual(sent.slice(1), [
    [["setFeatureState", { source: "s", id: 7 }, { hover: true }]],
    [["setGlobalStateProperty", "accent", "red"]],
  ]);
});
