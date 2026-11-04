// Point quads share one position and feature record per instance. Each
// visible chunk reserves 256 instances; padding exits before shading.
const CHUNK_POINTS: u32 = 256u;
fn pull_point(node_index: u32, instance: u32, out: ptr<function, Pulled>) -> bool {
    let entry = visible[node_cull[node_index].visible_base + instance / CHUNK_POINTS];
    let chunk = chunks[entry.chunk];
    let point = instance % CHUNK_POINTS;
    if point >= chunk.index_count { return false; }
    (*out).tile = entry.tile;
    (*out).vertex = chunk.vertex_base + indices[chunk.first_index + point] * 2u;
    (*out).prop_base = chunk.prop_base;
    (*out).state_base = chunk.state_base;
    (*out).row_words = chunk.row_words;
    return true;
}
fn point_corner(index: u32) -> vec2<f32> {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0));
    return corners[index % 6u];
}

// Clip the anchor, never the footprint. Half-open rectangles give an
// anchor on a boundary exactly one owner, including fallback partitions.
fn point_owned(tile: Tile, p: vec2<f32>) -> bool {
    for (var i = 0u; i < 4u; i++) {
        if all(p >= unpack_i16x2(tile.clips[i].x)) && all(p < unpack_i16x2(tile.clips[i].y)) {
            return true;
        }
    }
    return false;
}
