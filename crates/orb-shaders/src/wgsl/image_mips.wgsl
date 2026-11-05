// A raster page's mip levels (#3): each level a 2×2 box filter of the one
// above, for the layers just uploaded, as WebGL's `generateMipmap` makes
// GL JS's (`useMipmap`). Texels are premultiplied, so the average is
// correct. Odd edges clamp to the level above's last texel.

@group(0) @binding(0) var source: texture_2d_array<f32>;
@group(0) @binding(1) var destination: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> layers: array<u32>;

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(destination)) {
        return;
    }
    let layer = layers[id.z];
    let last = vec2<i32>(textureDimensions(source)) - 1;
    let p = vec2<i32>(id.xy) * 2;
    textureStore(destination, id.xy, layer, 0.25 * (
        textureLoad(source, min(p, last), layer, 0)
        + textureLoad(source, min(p + vec2<i32>(1, 0), last), layer, 0)
        + textureLoad(source, min(p + vec2<i32>(0, 1), last), layer, 0)
        + textureLoad(source, min(p + vec2<i32>(1, 1), last), layer, 0)));
}
