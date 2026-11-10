// Mip levels: each a 2×2 box filter of the level
// above, as WebGL's `generateMipmap` makes GL JS's, over square regions of
// an array texture's layers: the drape slots a frame redrew, or a raster
// page's new layers. Texels are premultiplied, so the average is correct;
// odd edges clamp to the level above's last texel.
//
// Edges: for each region (its layer, then the corners of the texels
// inside its tile's area, `[min, max)` packed as x | y << 16), the layer's
// level 0 with every texel past that area taken from the nearest inside,
// as a texture's edge clamps. A tile of a partial tile matrix that reaches
// past its set's extent holds fill there, which filtering and the levels
// would read into its edge.
//
// `mip_texel` and `extend_texel` make each texel, written by the compute
// entry points (`downsample_compute.wgsl`) or drawn by the fragment ones
// (`downsample_draw.wgsl`), one body for either.

@group(0) @binding(0) var source: texture_2d_array<f32>;
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

// Texel `p` of the level being written, in `layer`: the mean of the four
// beneath it in `source`, the level above.
fn mip_texel(p: vec2<u32>, layer: u32) -> vec4<f32> {
    let last = vec2<i32>(textureDimensions(source)) - 1;
    let s = vec2<i32>(p) * 2;
    var mean = 0.25 * (
        downsample_load(min(s, last), layer)
        + downsample_load(min(s + vec2<i32>(1, 0), last), layer)
        + downsample_load(min(s + vec2<i32>(0, 1), last), layer)
        + downsample_load(min(s + vec2<i32>(1, 1), last), layer));
    if level.y != 0u {
        mean = vec4<f32>(linear_to_srgb(mean.rgb), mean.a);
    }
    return mean;
}

// Texel `p` of an extended region's level 0: its own inside the tile's
// area, the nearest inside past it.
fn extend_texel(p: vec2<u32>, region: vec4<u32>) -> vec4<f32> {
    let lo = vec2<u32>(region.y & 0xffffu, region.y >> 16u);
    let hi = vec2<u32>(region.z & 0xffffu, region.z >> 16u);
    return textureLoad(source, clamp(p, lo, hi - 1u), region.x, 0);
}
