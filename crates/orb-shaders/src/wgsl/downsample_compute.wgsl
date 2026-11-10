// The downsampler's compute entry points: an invocation per texel, `z`
// picking the region.

@group(0) @binding(1) var destination: texture_storage_2d_array<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let region = region_at(id.z);
    if any(id.xy >= vec2<u32>(max(region.w >> level.x, 1u))) {
        return;
    }
    let p = (region.yz >> vec2<u32>(level.x)) + id.xy;
    textureStore(destination, p, region.x, mip_texel(p, region.x));
}

// Into `destination`'s layer `z`, which is copied back before the levels
// are made.
@compute @workgroup_size(8, 8)
fn extend(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(source)) {
        return;
    }
    textureStore(destination, id.xy, id.z, extend_texel(id.xy, region_at(id.z)));
}
