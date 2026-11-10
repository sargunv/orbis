// Cold pipeline compiles in the measurement pages: `?nonce=<n>` makes
// every shader module's code unique to `n`, so neither the browser's
// caches nor the driver's hold its pipelines, whatever earlier loads left
// there (`chrome://gpuclean` doesn't reach the driver's own cache on an
// Adreno phone). It moves the discarded vertex's z by `n`, which every
// vertex stage that culls a vertex reads; the vertex still falls outside
// the view volume. Only on the page's thread (`createMap`'s `worker:
// false`), where the device's methods can be wrapped.
export function bustShaderCaches(params) {
  const nonce = Number(params.get("nonce"));
  if (!nonce || typeof GPUDevice === "undefined") return false;
  const z = (2 + (nonce % 10000) * 1e-4).toFixed(4);
  const original = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function (desc) {
    const code = desc.code.replace(
      /vec4<f32>\(2\.0,\s*2\.0,\s*2\.0,\s*1\.0\)/g,
      `vec4<f32>(2.0, 2.0, ${z}, 1.0)`,
    );
    return original.call(this, { ...desc, code });
  };
  return true;
}
