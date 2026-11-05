// Casters' fragment stages, apart for the vertex-only clip-plane caster.
@fragment
fn fs_caster(in: CasterOut) {}
@fragment
fn fs_caster_cut(in: CasterOut) {
    if !extrusion_inside(tile_at(in.tile), in.tile_pos) { discard; }
}
