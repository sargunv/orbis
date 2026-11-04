export async function adapterInfo(gpu) {
  if (!gpu) return { gpu: false };
  let adapter = null;
  try { adapter = await gpu.requestAdapter(); } catch (e) { return { gpu: true, error: String(e) }; }
  if (!adapter) return { gpu: true, adapter: false };
  const info = adapter.info ?? {};
  let device = false;
  try { device = !!(await adapter.requestDevice()); } catch {}
  return {
    gpu: true, adapter: true, device,
    vendor: info.vendor, architecture: info.architecture,
    features: [...adapter.features].sort(),
    maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
  };
}

