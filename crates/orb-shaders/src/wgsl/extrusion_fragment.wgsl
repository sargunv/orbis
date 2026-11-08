// Extrusions' fragment stages, apart from `extrusion.wgsl` for the
// vertex-only clip-plane shaders, which leave them out.
@fragment
fn fs_extrusion(in: ExtrusionOut) -> @location(0) vec4<f32> {
    return extrusion_shade(in);
}
@fragment
fn fs_extrusion_cut(in: ExtrusionOut) -> @location(0) vec4<f32> {
    if !extrusion_inside(tile_at(in.tile), in.tile_pos) { discard; }
    return extrusion_shade(in);
}
