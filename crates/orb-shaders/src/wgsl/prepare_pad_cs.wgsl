// The pad step as a compute dispatch (Modern): one invocation per texel,
// `z` picks the job.

@group(0) @binding(1) var padded: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> jobs: array<DemJob>;

@compute @workgroup_size(8, 8)
fn pad(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raw).x + 4u;
    if any(id.xy >= vec2<u32>(size)) {
        return;
    }
    let job = jobs[id.z];
    textureStore(padded, id.xy, job.layer, pad_texel(job, id.xy));
}
