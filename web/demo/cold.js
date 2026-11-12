// Cold pipeline compiles in the measurement pages: `?nonce=<n>` makes
// every shader module's code unique to `n`, so neither the browser's
// caches nor the driver's hold its pipelines, whatever earlier loads left
// there (`chrome://gpuclean` doesn't reach the driver's own cache on an
// Adreno phone). Every WGSL entry point opens with a local holding `n`,
// as orb-hal-wgpu's `set_shader_nonce` does natively: compaction to the
// entry point keeps it and the back end writes it, where a change to a
// function the stage doesn't call would go before the driver saw it.
// WebGL 2's shaders (the GLSL naga writes from the same WGSL) get a
// define unique to `n`, and their discarded vertex moved by it. On the page's thread (`createMap`'s
// `worker: false`), where the device's methods can be wrapped, or in the
// map worker, called ahead of its script (session.mjs). Returns
// how many shaders it changed so far, of how many.
export function bustShaderCaches(params) {
  const nonce = Number(params.get("nonce"));
  let changed = 0;
  let total = 0;
  if (!nonce) return () => ({ changed, total });
  if (typeof GPUDevice !== "undefined") {
    const original = GPUDevice.prototype.createShaderModule;
    GPUDevice.prototype.createShaderModule = function (desc) {
      const code = desc.code.replace(
        /(@(?:vertex|fragment|compute)\b[^{]*\{)/g,
        `$1 var orb_nonce: u32 = ${nonce >>> 0}u;`,
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
          `vec4(2.0, 2.0, ${(2 + (nonce % 10000) * 1e-4).toFixed(4)}, 1.0)`,
        )
        .replace(/^(#version[^\n]*\n)/, `$1#define ORB_NONCE_${nonce}\n`);
      total++;
      if (busted !== code) changed++;
      return original.call(this, shader, busted);
    };
  }
  return () => ({ changed, total });
}
