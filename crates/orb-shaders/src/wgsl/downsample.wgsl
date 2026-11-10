// Mip levels: each a 2×2 box filter of the level
// above, as WebGL's `generateMipmap` makes GL JS's, over square regions of
// an array texture's layers: the drape slots a frame redrew, or a raster
// page's new layers. Texels are premultiplied, so the average is correct;
// odd edges clamp to the level above's last texel.

@group(0) @binding(0) var source: texture_2d_array<f32>;
@group(0) @binding(1) var destination: texture_storage_2d_array<rgba8unorm, write>;
// Per region at level 0: its layer, corner and side, in texels.
@group(0) @binding(2) var<storage, read> regions: array<vec4<u32>>;
// The level being written, and 1 in y when texels are linear light
// sRGB-encoded (a linear-light drape), which average decoded.
@group(0) @binding(3) var<uniform> level: vec4<u32>;

fn downsample_load(p: vec2<i32>, layer: u32) -> vec4<f32> {
    let texel = textureLoad(source, p, layer, 0);
    if level.y == 0u {
        return texel;
    }
    return vec4<f32>(srgb_to_linear(texel.rgb), texel.a);
}

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
    var mean = 0.25 * (
        downsample_load(min(s, last), layer)
        + downsample_load(min(s + vec2<i32>(1, 0), last), layer)
        + downsample_load(min(s + vec2<i32>(0, 1), last), layer)
        + downsample_load(min(s + vec2<i32>(1, 1), last), layer));
    if level.y != 0u {
        mean = vec4<f32>(linear_to_srgb(mean.rgb), mean.a);
    }
    textureStore(destination, p, layer, mean);
}

// Edges: for each region (`regions`: its layer, then the corners of the
// texels inside its tile's area, `[min, max)` packed as x | y << 16), the
// layer's level 0 with every texel past that area taken from the nearest
// inside, as a texture's edge clamps: into `destination`'s layer `z`, which
// is copied back before the levels are made. A tile of a partial tile
// matrix that reaches past its set's extent holds fill there, which
// filtering and the levels would read into its edge.
@compute @workgroup_size(8, 8)
fn extend(@builtin(global_invocation_id) id: vec3<u32>) {
    let region = regions[id.z];
    if any(id.xy >= textureDimensions(source)) {
        return;
    }
    let lo = vec2<u32>(region.y & 0xffffu, region.y >> 16u);
    let hi = vec2<u32>(region.z & 0xffffu, region.z >> 16u);
    let p = clamp(id.xy, lo, hi - 1u);
    textureStore(destination, id.xy, id.z, textureLoad(source, p, region.x, 0));
}
