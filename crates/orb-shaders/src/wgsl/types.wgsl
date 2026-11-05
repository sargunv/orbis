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
    _pad2: f32,
    // How far symbol fades have run since the last placement commit.
    symbol_fade_change: f32,
    // Camera pitch and roll, radians.
    pitch: f32,
    roll: f32,
    // The zoom's f64 bits, which the GPU interpreter branches on.
    zoom_bits: vec2<u32>,
    // The terrain's DEM unpack vector (metres = r·x + g·y + b·z − w),
    // its exaggeration (0 without terrain) and DEM pixels per side.
    dem_unpack: vec4<f32>,
    terrain_exaggeration: f32,
    dem_dim: f32,
    _pad4: vec2<f32>,
    // The style's lights (decision 0026; see lighting.wgsl): the ambient
    // and directional colours with their levels in w, and the vector
    // toward the directional light in tile axes, w = 1 when terrain is lit.
    light_ambient: vec4<f32>,
    light_color: vec4<f32>,
    light_position: vec4<f32>,
    // The globe (orb-scene's `FrameGlobe`): camera-relative ECEF metres →
    // clip, the horizon plane (`dot(xyz, p) + w`: 1 under the eye, 0 at the
    // horizon), the target's ECEF position, how much of a globe the view
    // is (0 the plane), and metres per plane world unit at the target.
    globe_view_proj: mat4x4<f32>,
    globe_horizon: vec4<f32>,
    globe_target: vec3<f32>,
    globe_amount: f32,
    globe_metres: f32,
    _pad5: f32,
    _pad6: vec2<f32>,
};

// One placement of a tile's data: data units → camera-relative world is
// `origin + vec2(x, -y) * scale`, or `origin` plus the tile's warp (see
// project.wgsl). The tile draws only inside its clip rectangles.
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
    // World units per logical camera pixel at this placement's scale.
    world_per_px: f32,
    // The DEM elevated content stands on: page layer (low 16 bits, all
    // ones for none) and depth below the tile (bits 16–23), then the
    // tile's offset in the DEM tile as two u16s.
    dem: u32,
    dem_offset: u32,
    // Lowest and highest terrain elevation under it as two i16s, in
    // units of TERRAIN_RANGE_UNIT metres.
    terrain_range: u32,
    // A warped tile's quadratic patches on the plane: first word in
    // `props` and patches per side; 0 patches for an affine tile.
    warp_base: u32,
    warp_cells: u32,
    // On a globe, its patches in space (27 words each) around
    // `globe_origin`, and flags: bit 0, no position on the plane.
    globe_origin: array<f32, 3>,
    globe_base: u32,
    globe_cells: u32,
    globe_flags: u32,
};

// Words per tile row (orb-scene's `TILE_WORDS`).
const TILE_WORDS: u32 = 28u;

const TERRAIN_RANGE_UNIT: f32 = 4.0;

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

// ImageAtlas table: rect in physical texels, array page, source pixel ratio,
// SDF flag. Missing rows are zero, and sample transparent without division.
struct ImageRect {
    origin: vec2<u32>,
    size: vec2<u32>,
    layer: u32,
    pixel_ratio: f32,
    sdf: u32,
    _pad: u32,
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
const DRAW_IMAGES: u32 = 0u;
// Points clip anchors, then expand the whole footprint.
const DRAW_POINTS: u32 = 1u;
const DRAW_INDEXED: u32 = 2u;
// On a node's draw word: its candidates are drape placements, kept by
// their clip rectangles alone.
const DRAPED: u32 = 256u;

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

// An affine tile's data units → camera-relative world (project.wgsl
// handles warped tiles too).
fn tile_affine(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    return tile.origin + vec2<f32>(p.x, -p.y) * tile.scale;
}

fn unpack_i16x2(w: u32) -> vec2<f32> {
    return vec2<f32>(vec2<i32>(bitcast<i32>(w << 16u) >> 16u, bitcast<i32>(w) >> 16u));
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
