// The downsampler's `fetch` (as `fetch.wgsl` is the draw pipelines'): it
// reads its regions only through this, so a device without storage
// buffers can read them from a texture by swapping this module alone.

// Per region at level 0: its layer, corner and side, in texels (or for
// `extend`, its layer and its tile's area).
@group(0) @binding(2) var<storage, read> regions: array<vec4<u32>>;

fn region_at(i: u32) -> vec4<u32> { return regions[i]; }
