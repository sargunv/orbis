// Extrusions' fragment stages, apart from `extrusion.wgsl` for the
// vertex-only clip-plane shaders, which leave them out.
@fragment
fn fs_extrusion(in: ExtrusionOut) -> @location(0) vec4<f32> {
    return extrusion_shade(in);
}
@fragment
fn fs_extrusion_cut(in: ExtrusionOut) -> @location(0) vec4<f32> {
    if sphere_whole() {
        // The point on the building, not on the chord between its
        // vertices: a low-zoom roof's chords sag far under the sphere.
        let p = normalize(in.sphere.xyz) * in.sphere.w - frame.globe_target;
        if sphere_hides(p, true) || (in.whole == 0u && !extrusion_inside(tile_at(in.tile), in.tile_pos)) {
            discard;
        }
    } else if !extrusion_inside(tile_at(in.tile), in.tile_pos) {
        discard;
    }
    return extrusion_shade(in);
}
