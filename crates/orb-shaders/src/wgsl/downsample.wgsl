// Mip levels: each a 2×2 box filter of the level
// above, as WebGL's `generateMipmap` makes GL JS's, over square regions of
// an array texture's layers: the drape slots a frame redrew, or a raster
// page's new layers. Texels are premultiplied, so the average is correct;
// odd edges clamp to the level above's last texel.

@group(0) @binding(0) var source: texture_2d_array<f32>;
@group(0) @binding(1) var destination: texture_storage_2d_array<rgba8unorm, write>;
// Per region at level 0: its layer, corner and side, in texels.
@group(0) @binding(2) var<storage, read> regions: array<vec4<u32>>;
// The level being written.
@group(0) @binding(3) var<uniform> level: vec4<u32>;

// One invocation per texel; `z` picks the region.
@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let region = regions[id.z];
    if any(id.xy >= vec2<u32>(max(region.w >> level.x, 1u))) {
        return;
    }
    let p = (region.yz >> vec2<u32>(level.x)) + id.xy;
    let last = vec2<i32>(textureDimensions(source)) - 1;
    let s = vec2<i32>(p) * 2;
    let layer = region.x;
    textureStore(destination, p, layer, 0.25 * (
        textureLoad(source, min(s, last), layer, 0)
        + textureLoad(source, min(s + vec2<i32>(1, 0), last), layer, 0)
        + textureLoad(source, min(s + vec2<i32>(0, 1), last), layer, 0)
        + textureLoad(source, min(s + vec2<i32>(1, 1), last), layer, 0)));
}
