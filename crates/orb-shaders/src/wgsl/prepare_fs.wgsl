// The prepare steps drawn into one layer each (Compat, which has no
// compute; plan §8.5's render-to-texture `Prepare`): a triangle over the
// layer, one fragment per texel, the job from a uniform.

struct PrepareJob {
    unpack: vec4<f32>,
    neighbours: array<vec4<u32>, 2>,
    layer: u32,
    zoom: f32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(2) var<uniform> prepare_job: PrepareJob;

fn uniform_job() -> DemJob {
    var job: DemJob;
    job.unpack = prepare_job.unpack;
    job.layer = prepare_job.layer;
    job.zoom = prepare_job.zoom;
    for (var i = 0u; i < 8u; i++) {
        job.neighbours[i] = prepare_job.neighbours[i / 4u][i % 4u];
    }
    return job;
}

@vertex
fn vs_prepare(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
