@group(0) @binding(0) var raw: texture_2d_array<f32>;
@group(0) @binding(1) var padded: texture_storage_2d_array<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn pad(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = i32(textureDimensions(raw).x);
    if any(id.xy >= vec2<u32>(u32(dim + 4))) {
        return;
    }
    let job = jobs[id.z];
    let p = vec2<i32>(id.xy) - 2;
    let d = vec2<i32>(select(0, -1, p.x < 0) + select(0, 1, p.x >= dim),
        select(0, -1, p.y < 0) + select(0, 1, p.y >= dim));
    var layer = job.layer;
    var q = clamp(p, vec2<i32>(0), vec2<i32>(dim - 1));
    if any(d != vec2<i32>(0)) {
        let k = u32((d.y + 1) * 3 + d.x + 1);
        let neighbour = job.neighbours[select(k, k - 1u, k > 4u)];
        if neighbour != NO_TILE {
            layer = neighbour;
            q = p - d * dim;
        }
    }
    textureStore(padded, id.xy, job.layer, textureLoad(raw, q, layer, 0));
}
