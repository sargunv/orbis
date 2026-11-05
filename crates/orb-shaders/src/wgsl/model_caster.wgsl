// Models into one shadow cascade: depth only, from the
// light, their opaque and masked materials, in two pipelines so only the
// masked one discards. Masked texels below their cutoff cast nothing.
struct ShadowPass { cascade: u32, };
@group(3) @binding(0) var<uniform> shadow_pass: ShadowPass;
struct ModelCasterOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) material: u32,
};
@vertex
fn vs_model_caster(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> ModelCasterOut {
    var out: ModelCasterOut;
    out.position = discarded_vertex();
    var p: ModelPlaced;
    if !model_placed(vertex, instance, &p) { return out; }
    out.position = shadow.cascades.view_proj[shadow_pass.cascade] * vec4<f32>(model_world(p), 1.0);
    out.uv = vec2<f32>(model_word_f(p.w + 4u), model_word_f(p.w + 5u));
    out.color = unpack4x8unorm(model_words[p.w + 6u]);
    out.material = model_words[p.w + 7u];
    return out;
}
// Opaque materials: depth alone, nothing discarded.
@fragment
fn fs_model_caster(in: ModelCasterOut) {}
// Masked materials: texels under the cutoff cast nothing.
@fragment
fn fs_model_caster_masked(in: ModelCasterOut) {
    let uv = ModelUv(in.uv, dpdx(in.uv), dpdy(in.uv));
    let m = model_draw.material_base + in.material * MODEL_MATERIAL_WORDS;
    if model_base(m, uv, in.color).a < model_word_f(m + 7u) {
        discard;
    }
}
