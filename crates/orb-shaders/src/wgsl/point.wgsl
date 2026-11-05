// Point quads share one position and feature record per instance. Each
// visible chunk reserves 256 instances; padding exits before shading.
const CHUNK_POINTS: u32 = 256u;
fn pull_point(node_index: u32, instance: u32, out: ptr<function, Pulled>) -> bool {
    let entry = visible_at(node_cull_at(node_index).visible_base + instance / CHUNK_POINTS);
    let chunk = chunk_at(entry.chunk);
    let point = instance % CHUNK_POINTS;
    if point >= chunk.index_count { return false; }
    (*out).tile = entry.tile;
    (*out).vertex = chunk.vertex_base + index_at(chunk.first_index + point) * 2u;
    (*out).prop_base = chunk.prop_base;
    (*out).state_base = chunk.state_base;
    (*out).row_words = chunk.row_words;
    return true;
}
// An instanced node's point `instance`: a binary search of its visible
// list for the last chunk with no more instances before it (the list's
// tail holds MAX_INSTANCES), then the point in that chunk.
fn pull_instance(node_index: u32, instance: u32, out: ptr<function, Pulled>) -> bool {
    let node = node_cull_at(node_index);
    var lo = 0u;
    var hi = node.candidate_count;
    while hi - lo > 1u {
        let mid = (lo + hi) / 2u;
        if (visible_at(node.visible_base + mid).tile >> 16u) <= instance {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let entry = visible_at(node.visible_base + lo);
    let before = entry.tile >> 16u;
    if node.candidate_count == 0u || before > instance {
        return false;
    }
    let chunk = chunk_at(entry.chunk);
    let point = instance - before;
    if point >= chunk.index_count { return false; }
    (*out).tile = entry.tile & 0xffffu;
    (*out).vertex = chunk.vertex_base + index_at(chunk.first_index + point) * 2u;
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
