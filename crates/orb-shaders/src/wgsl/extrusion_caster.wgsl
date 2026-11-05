// Extrusions into one shadow cascade (decision 0046): depth only, from the
// light, the placements that draw a building in the view drawing it here.
// A cut building isn't clipped: where two tiles' parts of it overlap they
// write the same depths, so one pipeline draws every building, without the
// discard or the second and third passes over the indices the view's need.
struct ShadowPass { cascade: u32, };
@group(2) @binding(0) var<uniform> shadow_pass: ShadowPass;
struct CasterOut {
    @builtin(position) position: vec4<f32>,
};
@vertex
fn vs_caster(@builtin(vertex_index) index: u32) -> CasterOut {
    var out: CasterOut;
    var v: Pulled;
    if !pull_indexed(node.index, index, 4u, &v) || extrusion_drawn(v) == 0u {
        out.position = discarded_vertex();
        return out;
    }
    let e = extrusion_vertex(v);
    out.position = shadow.cascades.view_proj[shadow_pass.cascade] * vec4<f32>(e.world, 1.0);
    return out;
}
@fragment
fn fs_caster(in: CasterOut) {}
