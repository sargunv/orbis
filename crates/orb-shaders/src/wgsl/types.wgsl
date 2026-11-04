// Scene tables shared by every pipeline. Layouts must match orb-scene's
// `#[repr(C)]` rows.

struct Frame {
    // Camera-relative world → clip.
    view_proj: mat4x4<f32>,
    // Logical pixels.
    viewport: vec2<f32>,
    // World units per logical pixel at the camera target.
    world_per_px: f32,
    // Physical pixels per logical pixel.
    pixel_ratio: f32,
    // Style zoom of the camera.
    zoom: f32,
    // Clockwise map rotation, in radians.
    bearing: f32,
    // A camera-relative world offset that puts image tiles on the pixel
    // grid (GL JS's aligned projection matrix).
    align: vec2<f32>,
    extrusion_scale: f32,
    ground_horizon: f32,
    sky_horizon: f32,
    // How far symbol fades have run since the last placement commit.
    symbol_fade_change: f32,
    // Camera pitch, radians.
    pitch: f32,
    _pad3: u32,
    // The zoom's f64 bits, which the GPU interpreter branches on.
    zoom_bits: vec2<u32>,
};

// One placement of a tile's data: data units → camera-relative world is
// `origin + vec2(x, -y) * scale`. The tile draws only inside its clip
// rectangles.
struct Tile {
    origin: vec2<f32>,
    scale: f32,
    // The tile's zoom level, for zoom-and-feature curves.
    zoom: f32,
    // Up to four rectangles in data units as packed i16 pairs (min, max);
    // empty when min > max.
    clips: array<vec2<u32>, 4>,
    pattern_origin: vec2<u32>,
    extent: f32,
    // The world copy it draws.
    wrap: i32,
};

// A run of at most 256 triangles of one mesh.
struct Chunk {
    first_index: u32,
    index_count: u32,
    // First word of the mesh's vertices in the vertex arena.
    vertex_base: u32,
    // First word of the mesh's property rows in the property arena.
    prop_base: u32,
    // Bounds in tile data units, as packed i16 pairs.
    bounds_min: u32,
    bounds_max: u32,
    // First word of the mesh's feature-state remap in the property arena:
    // one state index per feature slot, or NO_STATE.
    state_base: u32,
    // Words per feature of the mesh's property rows.
    row_words: u32,
};

const NO_STATE: u32 = 0xffffffffu;

// A chunk to draw, and the tile placement to draw it with. An indexed
// node's slice of the visible list holds its runs instead: a chunk of the
// run's mesh, for the mesh's bases, and the placement.
struct Visible {
    chunk: u32,
    tile: u32,
};

// How a node draws (orb-scene's `NodeDraw`).
const DRAW_CHUNKS: u32 = 0u;
// Points clip anchors, then expand the whole footprint.
const DRAW_POINTS: u32 = 1u;
const DRAW_INDEXED: u32 = 2u;

// A render node's slice of the candidate and visible lists.
struct NodeCull {
    // The node's first slot in the candidate and visible lists.
    visible_base: u32,
    // Extra margin around chunk bounds, in pixels (line extrusion).
    margin_px: f32,
    candidate_count: u32,
    draw: u32,
    elevation: f32,
    // Indexed nodes: their slots in the compacted index buffer, and the
    // bits of a compacted index that address the run's vertex.
    index_base: u32,
    index_count: u32,
    run_shift: u32,
};

// A compacted index that draws nothing; no run reaches it.
const HIDDEN_INDEX: u32 = 0xfffffffeu;

// Every chunk is drawn as this many vertices; the excess is degenerate.
const CHUNK_VERTICES: u32 = 768u;

fn unpack_i16x2(w: u32) -> vec2<f32> {
    return vec2<f32>(vec2<i32>(bitcast<i32>(w << 16u) >> 16u, bitcast<i32>(w) >> 16u));
}

fn tile_to_world(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    return tile.origin + vec2<f32>(p.x, -p.y) * tile.scale;
}

// A clip position outside the view volume, for padding vertices.
fn discarded_vertex() -> vec4<f32> {
    return vec4<f32>(2.0, 2.0, 2.0, 1.0);
}

// Whether a data-unit point lies in one of the tile's clip rectangles.
// The maximum edge is excluded so adjacent regions never double-blend.
// Fragments outside belong to a neighbour or a finer tile.
fn inside_clips(tile: Tile, p: vec2<f32>) -> bool {
    for (var i = 0u; i < 4u; i++) {
        let lo = unpack_i16x2(tile.clips[i].x);
        let hi = unpack_i16x2(tile.clips[i].y);
        if all(p >= lo) && all(p < hi) {
            return true;
        }
    }
    return false;
}
