// Extrusions into one shadow cascade (decision 0046): depth only, from the
// light, the placements that draw a building in the view drawing it here.
struct ShadowPass { cascade: u32, };
@group(2) @binding(0) var<uniform> shadow_pass: ShadowPass;
struct CasterOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(perspective, centroid) tile_pos: vec2<f32>,
    @location(1) @interpolate(flat) tile: u32,
    @location(2) @interpolate(flat) cut: u32,
};
@vertex
fn vs_caster(@builtin(vertex_index) index: u32) -> CasterOut {
    var out: CasterOut;
    var v: Pulled;
    if !pull_indexed(node.index, index, 4u, &v) || !extrusion_drawn(v) {
        out.position = discarded_vertex();
        return out;
    }
    let e = extrusion_vertex(v);
    out.position = shadow.cascades.view_proj[shadow_pass.cascade] * vec4<f32>(e.world, 1.0);
    out.tile_pos = e.tile_pos;
    out.tile = e.tile;
    out.cut = e.cut;
    return out;
}
@fragment
fn fs_caster(in: CasterOut) {
    let tile = tile_at(in.tile);
    var inside = in.cut == 0u;
    for (var i = 0u; i < 4u && !inside; i++) {
        let lo = unpack_i16x2(tile.clips[i].x);
        let hi = unpack_i16x2(tile.clips[i].y);
        inside = all(in.tile_pos >= lo) && all(in.tile_pos <= hi);
    }
    if !inside { discard; }
}
