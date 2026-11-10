// The drape clear's `fetch` (as `fetch.wgsl` is the draw pipelines'): it
// reads its regions only through this, so a device without storage
// buffers in the vertex stage can read them from a texture by swapping
// this module alone.

@group(0) @binding(1) var<storage, read> regions: array<vec4<u32>>;

fn region_at(i: u32) -> vec4<u32> { return regions[i]; }
