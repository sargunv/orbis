// Extrusions into one shadow cascade (decision 0046): depth only, from the
// light, the placements that draw a building in the view drawing it here.
// Whole and cut buildings draw apart, as in the view (`vs_extrusion`).
struct ShadowPass { cascade: u32, };
@group(2) @binding(0) var<uniform> shadow_pass: ShadowPass;
struct CasterOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(perspective, centroid) tile_pos: vec2<f32>,
    @location(1) @interpolate(flat) tile: u32,
};
@vertex
fn vs_caster(@builtin(vertex_index) index: u32) -> CasterOut {
    return caster_part(index, EXTRUSION_WHOLE);
}
@vertex
fn vs_caster_cut(@builtin(vertex_index) index: u32) -> CasterOut {
    return caster_part(index, EXTRUSION_CUT);
}
fn caster_part(index: u32, part: u32) -> CasterOut {
    var out: CasterOut;
    var v: Pulled;
    if !pull_indexed(node.index, index, 4u, &v) || extrusion_drawn(v) != part {
        out.position = discarded_vertex();
        return out;
    }
    let e = extrusion_vertex(v);
    out.position = shadow.cascades.view_proj[shadow_pass.cascade] * vec4<f32>(e.world, 1.0);
    out.tile_pos = e.tile_pos;
    out.tile = e.tile;
    return out;
}
@fragment
fn fs_caster(in: CasterOut) {}
@fragment
fn fs_caster_cut(in: CasterOut) {
    if !extrusion_inside(tile_at(in.tile), in.tile_pos) { discard; }
}
