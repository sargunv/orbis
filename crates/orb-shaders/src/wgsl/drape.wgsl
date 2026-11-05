// Drape atlas upkeep: clearing the slots a frame redraws, one instanced
// quad per region of `regions` (the downsampler's: layer, corner and side
// in texels), before their layers draw into them.

// The atlas's size in texels.
@group(0) @binding(0) var<uniform> atlas_size: vec4<f32>;
@group(0) @binding(1) var<storage, read> regions: array<vec4<u32>>;

@vertex
fn vs_slot(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> @builtin(position) vec4<f32> {
    let region = regions[instance];
    let corner = vec2<f32>(f32(vertex & 1u), f32((vertex >> 1u) & 1u));
    let uv = (vec2<f32>(region.yz) + corner * f32(region.w)) / atlas_size.xy;
    return vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

@fragment
fn fs_clear() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
