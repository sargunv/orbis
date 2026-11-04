@group(1) @binding(1) var density: texture_2d<f32>;
@group(1) @binding(2) var ramp: texture_2d<f32>;
@group(1) @binding(3) var linear_sampler: sampler;
struct HeatmapColorOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_heatmap_color(@builtin(vertex_index) index: u32) -> HeatmapColorOut {
    let positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: HeatmapColorOut;
    if node_cull_at(node.index).candidate_count == 0u {
        out.position = discarded_vertex();
        return out;
    }
    let p = positions[index];
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = p * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    return out;
}
@fragment
fn fs_heatmap_color(in: HeatmapColorOut) -> @location(0) vec4<f32> {
    let t = textureSample(density, linear_sampler, in.uv).r;
    return textureSample(ramp, linear_sampler, vec2<f32>(t, 0.5)) * node.opacity;
}
