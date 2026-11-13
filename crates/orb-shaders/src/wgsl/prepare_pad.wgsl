// Texel `id` of the job's padded layer, (dim + 4)² texels.
fn pad_texel(id: vec2<u32>) -> vec4<f32> {
    let dim = i32(source_side());
    let p = vec2<i32>(id) - 2;
    let d = vec2<i32>(select(0, -1, p.x < 0) + select(0, 1, p.x >= dim),
        select(0, -1, p.y < 0) + select(0, 1, p.y >= dim));
    var layer = job.layer;
    var q = clamp(p, vec2<i32>(0), vec2<i32>(dim - 1));
    if any(d != vec2<i32>(0)) {
        let k = u32((d.y + 1) * 3 + d.x + 1);
        let n = select(k, k - 1u, k > 4u);
        let neighbour = job.neighbours[n / 4u][n % 4u];
        if neighbour != NO_TILE {
            layer = neighbour;
            q = p - d * dim;
        }
    }
    return source_texel(q, layer);
}

@fragment
fn fs_pad(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return pad_texel(vec2<u32>(position.xy));
}
