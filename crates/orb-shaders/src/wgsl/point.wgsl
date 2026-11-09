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

// Vertex `index` of a point quad split into `cells` × `cells` (six
// vertices a cell), in [-1, 1].
fn point_corner(index: u32, cells: u32) -> vec2<f32> {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0));
    let cell = index / 6u;
    let at = vec2<f32>(f32(cell % cells), f32(cell / cells)) + corners[index % 6u];
    return at * (2.0 / f32(cells)) - 1.0;
}

// The cells per side of a node's point quads (`point_cells`).
fn node_point_cells(surface: bool) -> u32 {
    return point_cells(surface, frame.globe_amount, frame.globe_plane);
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
