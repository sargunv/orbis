// Cold pipeline compiles in the measurement pages: `?nonce=<n>` makes
// every shader module's code unique to `n`, so neither the browser's
// caches nor the driver's hold its pipelines, whatever earlier loads left
// there (`chrome://gpuclean` doesn't reach the driver's own cache on an
// Adreno phone). It moves the discarded vertex's z by `n`, which every
// vertex stage that culls a vertex reads; the vertex still falls outside
// the view volume. WebGL 2's shaders (the GLSL naga writes from the same
// WGSL) are changed the same way. Only on the page's thread (`createMap`'s
// `worker: false`), where the device's methods can be wrapped. Returns
// how many shaders it changed so far, of how many.
export function bustShaderCaches(params) {
  const nonce = Number(params.get("nonce"));
  let changed = 0;
  let total = 0;
  if (!nonce) return () => ({ changed, total });
  const z = (2 + (nonce % 10000) * 1e-4).toFixed(4);
  if (typeof GPUDevice !== "undefined") {
    const original = GPUDevice.prototype.createShaderModule;
    GPUDevice.prototype.createShaderModule = function (desc) {
      const code = desc.code.replace(
        /vec4<f32>\(2\.0,\s*2\.0,\s*2\.0,\s*1\.0\)/g,
        `vec4<f32>(2.0, 2.0, ${z}, 1.0)`,
      );
      total++;
      if (code !== desc.code) changed++;
      return original.call(this, { ...desc, code });
    };
  }
  if (typeof WebGL2RenderingContext !== "undefined") {
    const original = WebGL2RenderingContext.prototype.shaderSource;
    WebGL2RenderingContext.prototype.shaderSource = function (shader, code) {
      // A define unique to the sample misses the browser's program cache,
      // which keys on the source, where the shader has no such vertex.
      const busted = code
        .replace(
          /vec4\(2\.0,\s*2\.0,\s*2\.0,\s*1\.0\)/g,
          `vec4(2.0, 2.0, ${z}, 1.0)`,
        )
        .replace(/^(#version[^\n]*\n)/, `$1#define ORB_NONCE_${nonce}\n`);
      total++;
      if (busted !== code) changed++;
      return original.call(this, shader, busted);
    };
  }
  return () => ({ changed, total });
}
