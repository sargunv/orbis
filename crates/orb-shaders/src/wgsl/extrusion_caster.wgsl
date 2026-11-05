// Extrusions into one shadow cascade (decision 0046): depth only, from the
// light, the placements that draw a building in the view drawing it here.
// Whole and cut buildings draw apart, as in the view (`vs_extrusion`).
struct ShadowPass { cascade: u32, };
@group(2) @binding(0) var<uniform> shadow_pass: ShadowPass;
struct CasterOut {
    @builtin(position) position: vec4<f32>,
    //planes @builtin(clip_distances) clip: array<f32, 4>,
    @location(0) @interpolate(perspective, centroid) tile_pos: vec2<f32>,
    @location(1) @interpolate(flat) tile: u32,
};
@vertex
fn vs_caster(@builtin(vertex_index) index: u32) -> CasterOut {
    return caster_part(index, 1u << EXTRUSION_WHOLE);
}
@vertex
fn vs_caster_cut(@builtin(vertex_index) index: u32) -> CasterOut {
    return caster_part(index, (1u << EXTRUSION_CUT) | (1u << EXTRUSION_CUT_SPLIT));
}
@vertex
fn vs_caster_cut_planes(@builtin(vertex_index) index: u32) -> CasterOut {
    return caster_part(index, 1u << EXTRUSION_CUT);
}
@vertex
fn vs_caster_cut_split(@builtin(vertex_index) index: u32) -> CasterOut {
    return caster_part(index, 1u << EXTRUSION_CUT_SPLIT);
}
fn caster_part(index: u32, parts: u32) -> CasterOut {
    var out: CasterOut;
    var v: Pulled;
    if !pull_indexed(node.index, index, 4u, &v) || (parts & (1u << extrusion_drawn(v))) == 0u {
        out.position = discarded_vertex();
        return out;
    }
    let e = extrusion_vertex(v);
    out.position = shadow.cascades.view_proj[shadow_pass.cascade] * vec4<f32>(e.world, 1.0);
    out.tile_pos = e.tile_pos;
    out.tile = e.tile;
    //planes out.clip = extrusion_planes(tile_at(v.tile), e.tile_pos);
    return out;
}
