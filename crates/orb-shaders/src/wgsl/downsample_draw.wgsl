// The downsampler's fragment entry points, for devices without compute
// (WebGL 2): one instanced quad per region, drawn into a view of the
// level's layer, or of the extended layer, as the compute entry points
// write them.

struct RegionOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) region: vec4<u32>,
};

// Region `instance`'s square at the level being written, over a target of
// `size` texels.
fn region_quad(vertex: u32, instance: u32, size: vec2<f32>) -> RegionOut {
    let region = region_at(instance);
    let corner = vec2<f32>(f32(vertex & 1u), f32((vertex >> 1u) & 1u));
    let side = f32(max(region.w >> level.x, 1u));
    let at = vec2<f32>(region.yz >> vec2<u32>(level.x)) + corner * side;
    let uv = at / size;
    var out: RegionOut;
    out.position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.region = region;
    return out;
}

@vertex
fn vs_mip(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> RegionOut {
    let size = max(textureDimensions(source) / 2u, vec2<u32>(1u));
    return region_quad(vertex, instance, vec2<f32>(size));
}

@fragment
fn fs_mip(in: RegionOut) -> @location(0) vec4<f32> {
    return mip_texel(vec2<u32>(in.position.xy), in.region.x);
}

// The whole extended layer: an extended region covers its texture.
@vertex
fn vs_extend(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> RegionOut {
    let corner = vec2<f32>(f32(vertex & 1u), f32((vertex >> 1u) & 1u));
    var out: RegionOut;
    out.position = vec4<f32>(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.region = region_at(instance);
    return out;
}

@fragment
fn fs_extend(in: RegionOut) -> @location(0) vec4<f32> {
    return extend_texel(vec2<u32>(in.position.xy), in.region);
}
