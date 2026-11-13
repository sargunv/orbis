// Scene tables shared by every pipeline. Layouts must match orb-render's
// `#[repr(C)]` rows.

// Property rows, one per feature (`prop_row`): two words per paint
// property in the plan's order (`table_number`), then each family's
// extra words. Rust reads these through orb-shaders' `layout`. Pattern and
// dash ids are three, for layout zoom -1, 0 and +1.
const FILL_ROW_PATTERN: u32 = 4u;
const FILL_ROW_WORDS: u32 = 7u;
// Then, in an outline node's rows where `fill-antialias` is per feature
// (`FILL_ANTIALIAS_ROW_MASK`), a word: 1 where the feature has an outline.
const FILL_ROW_ANTIALIAS: u32 = 7u;
const FILL_ANTIALIAS_ROW_MASK: u32 = 128u;
const LINE_ROW_PATTERN: u32 = 12u;
const LINE_ROW_DASH: u32 = 15u;
const LINE_ROW_FLOOR_WIDTH: u32 = 18u;
const LINE_ROW_GRADIENT: u32 = 19u;
const LINE_ROW_WORDS: u32 = 20u;
const EXTRUSION_ROW_PATTERN: u32 = 6u;
// The feature's centroid (two i16s), and 1 when the building is whole in
// the tile's data.
const EXTRUSION_ROW_CENTROID: u32 = 9u;
// Flags: bit 0 the whole building, bit 1 its vertical gradient, bits 8-15
// and 16-23 its opacity at the tile's zoom and the next, when per feature.
const EXTRUSION_ROW_WHOLE: u32 = 10u;
const EXTRUSION_ROW_WORDS: u32 = 11u;
// Then, for a colour over the elevation, two words for each of its stops
// after the first that come from the row.
const CIRCLE_ROW_WORDS: u32 = 14u;
// Then, where `circle-pitch-scale` or `-alignment` is per feature
// (`CIRCLE_PITCH_ROW_MASK`), a word of the feature's flags: bit 1 map
// pitch scale, bit 2 map alignment.
const CIRCLE_ROW_PITCH: u32 = 14u;
const CIRCLE_PITCH_ROW_MASK: u32 = 128u;
// A fill, line, extrusion or circle node's per-feature `*-translate`
// (`table_mask` bit `TRANSLATE_ROW_MASK`) ends each row: its pixels at the
// tile's zoom and the next, each an i16 pair in `TRANSLATE_ROW_UNITS` per
// pixel.
const TRANSLATE_ROW_MASK: u32 = 32768u;
const TRANSLATE_ROW_WORDS: u32 = 2u;
const TRANSLATE_ROW_UNITS: u32 = 16u;
const HEATMAP_ROW_WORDS: u32 = 4u;
const SYMBOL_ROW_WORDS: u32 = 10u;

// The most light sources a hillshade node holds.
const MAX_LIGHTS: u32 = 4u;
// The most stops and input operators of an extrusion's colour over the
// elevation (`extrusion.wgsl`'s `ElevationColor`).
const MAX_ELEVATION_STOPS: u32 = 16u;
const MAX_ELEVATION_OPS: u32 = 4u;

struct Frame {
    // Camera-relative world → clip.
    view_proj: mat4x4<f32>,
    // The row of clip w less clip z: reversed-Z depth (`ground_clip_reversed`).
    view_depth: vec4<f32>,
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
    // Milliseconds since the map began, for image fades.
    time: f32,
    // The same, for feature-state fades: drape slots, which don't fade
    // images, redraw while a state fades and take the frame's.
    state_time: f32,
    // How far symbol fades have run since the last placement commit.
    symbol_fade_change: f32,
    // Camera pitch and roll, radians.
    pitch: f32,
    roll: f32,
    // The terrain's DEM unpack vector (metres = r·x + g·y + b·z − w),
    // its exaggeration (0 without terrain) and DEM pixels per side.
    dem_unpack: vec4<f32>,
    terrain_exaggeration: f32,
    dem_dim: f32,
    // The lowest ground the frame's terrain tiles can show, metres with
    // exaggeration, 0 when none is lower: what a solid point on the
    // globe must be behind to be hidden (`sphere_hides`).
    terrain_lowest: f32,
    // v8's `light` (see lighting.wgsl): its premultiplied colour with its
    // intensity in w; its position in GL JS's extrusion axes is below.
    light_color: vec4<f32>,
    // The target's clip w (orb-camera's `View::camera_to_center`).
    camera_to_center: f32,
    // The low-zoom surface (orb-render's `FrameGlobe`), the sphere or a plane
    // as `globe_plane` says: camera-relative ECEF metres →
    // clip, the horizon plane (`dot(xyz, p) + w`: 1 under the eye, 0 at the
    // horizon), the target's ECEF position (below), how much of a globe the
    // view is (0 the plane), and metres per plane world unit at the target.
    globe_view_proj: mat4x4<f32>,
    globe_horizon: vec4<f32>,
    globe_amount: f32,
    globe_metres: f32,
    // 1 when the low-zoom surface is a plane (its world units, z up, no
    // horizon) rather than the sphere.
    globe_plane: u32,
    // The quanta of `camera_whole` and `globe_target_whole` (orb-render's
    // `quantum`).
    quantum: f32,
    // The camera's world position as whole quanta and a remainder: rows'
    // origins less it are camera-relative (`tile_origin`).
    camera_whole: vec2<i32>,
    camera_rest: vec2<f32>,
    globe_quantum: f32,
    // The cells per side of a surface node's point quads (`point_cells`):
    // orb-render's `surface_cells` of the view, so the cull, the point
    // shaders and a driver that draws without the cull count them alike.
    surface_cells: u32,
    // The cull rows of a scene block: its render nodes' and their cut
    // streams' (orb-render's `Scene::row`). A block's rows start at
    // `block_row`, its cull workgroups at `block_group` and its symbol
    // placement workgroups at `block_quad_group`: the block's frame block
    // says where (the scene's own, unbound to a block, says 0).
    placed_nodes: u32,
    block_row: u32,
    block_group: u32,
    block_quad_group: u32,
    // The vec3s last, one after another: ANGLE's Metal back end gives a
    // vec3 16 bytes where std140 packs a scalar into its fourth, and a
    // block laid out otherwise than std140 is converted on the CPU at a
    // draw, waiting for the GPU's copy into it (#140).
    light_position: vec3<f32>,
    globe_target: vec3<f32>,
    // The eye, camera-relative ECEF metres.
    globe_eye: vec3<f32>,
    // `globe_target` split as the camera is, at `globe_quantum`.
    globe_target_whole: vec3<i32>,
    globe_target_rest: vec3<f32>,
};

// One placement of a tile's data: data units → world is
// `origin + vec2(x, -y) * scale`, or `origin` plus the tile's warp (see
// project.wgsl). The tile draws only inside its clip rectangles. Nothing
// in it depends on the camera: its origins are absolute (`tile_origin`,
// `tile_globe_origin`).
struct Tile {
    // In world units, as whole quanta and the remainder in `origin_rest`.
    origin: vec2<i32>,
    scale: f32,
    // The tile's zoom level, for zoom-and-feature curves.
    zoom: f32,
    // Up to four rectangles in data units as packed i16 pairs (min, max);
    // empty when min > max.
    clips: array<vec2<u32>, 4>,
    // Where its content may land, packed as a clip rectangle: a drape
    // placement's slot, everywhere for a map placement. Content translated
    // into a slot is owned where it was and bounded where it lands.
    bound: vec2<u32>,
    pattern_origin: vec2<u32>,
    extent: f32,
    // The world copy it draws.
    wrap: i32,
    // World units per logical camera pixel at this placement's scale, as a
    // multiple of the frame's (`tile_world_per_px`).
    px_scale: f32,
    // The DEM elevated content stands on: page layer (low 16 bits, all
    // ones for none) and depth below the tile (bits 16–23), then the
    // tile's offset in the DEM tile as two u16s.
    dem: u32,
    dem_offset: u32,
    // Lowest and highest terrain elevation under it as two i16s, in
    // units of TERRAIN_RANGE_UNIT metres.
    terrain_range: u32,
    // A warped tile's quadratic patches on the plane (`project.wgsl`):
    // first word in `props` and patches per side, x | y << 16; 0 for an
    // affine tile.
    warp_base: u32,
    warp_cells: u32,
    // On a globe, its patches in space (27 words each) around
    // `globe_origin`, and flags: bit 0, no position on the plane.
    globe_origin: array<i32, 3>,
    globe_base: u32,
    globe_cells: u32,
    globe_flags: u32,
    // The remainders of `origin` and `globe_origin`.
    origin_rest: vec2<f32>,
    globe_origin_rest: array<f32, 3>,
    // The tiles around it at its zoom that draw whole the polygons they
    // own, bit 3 (dy + 1) + dx + 1 (`TileRow::neighbours`).
    neighbours: u32,
    // Its clip as fragments test it where its content lands where it was
    // (`fragment_clip`), which the CPU works out once.
    clip: vec2<u32>,
};


const TERRAIN_RANGE_UNIT: f32 = 4.0;

// A run of at most 256 triangles of one mesh.
struct Chunk {
    // First index in the index arena's u16 indices (`index_at`); a symbol
    // chunk's first entry in its layer's quad order instead.
    first_index: u32,
    index_count: u32,
    // First word of the mesh's vertices in the vertex arena.
    vertex_base: u32,
    // The mesh's vertex that the chunk's indices count from. A symbol
    // chunk has no indices: its world copies placement shows any of its
    // quads in, bit `wrap & 31` (orb-render's symbol meshes).
    first_vertex: u32,
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

// How a node draws (orb-render's `NodeDraw`).
const DRAW_IMAGES: u32 = 0u;
// Points clip anchors, then expand the whole footprint.
const DRAW_POINTS: u32 = 1u;
const DRAW_INDEXED: u32 = 2u;
// On a node's draw word: its candidates are drape placements, kept by
// their clip rectangles alone.
const DRAPED: u32 = 256u;
// A node's draw word flag: hidden, it draws nothing and keeps its
// candidates.
const HIDDEN: u32 = 512u;
// A point node's draw word flag: its quads lie on the surface (heatmap
// kernels, map-aligned circles), so on a globe each is a grid of cells
// whose vertices go onto the sphere one by one.
const SURFACE: u32 = 1024u;
// A node's draw word flag: its meshes draw past their placements' clip
// rectangles (whole extrusions), so the cull keeps chunks by their bounds.
const UNCLIPPED: u32 = 2048u;
// A node's draw word flag: it draws symbols, which the cull keeps only in
// the world copies placement shows them in (`Chunk`'s `first_vertex`).
const SYMBOLS: u32 = 4096u;
// A point quad's cells per side: one, or the frame's for a surface node's.
fn point_cells(surface: bool) -> u32 {
    return select(1u, frame.surface_cells, surface);
}

// A render node's slice of the candidate and visible lists.
// Words of a `NodeCull` row.
const NODE_CULL_WORDS: u32 = 10u;

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
    // A symbol node's quads' first texel in the placement target
    // (`symbol.wgsl`'s), 64 for each of its candidate rows; any other
    // node's is where the next symbol node's would be.
    quad_base: u32,
    // The node's first workgroup of the cull dispatch (`cull.wgsl`): the
    // groups of the nodes before it, so the firsts rise with the node.
    first_group: u32,
};

// A compacted index that draws nothing: zero, below every run's indices
// (a run's are tagged with its number plus one), so a cleared range is
// hidden. The CPU driver hides with the primitive restart index instead,
// which WebGL 2 skips (it drops a whole draw holding an index from 2^31
// on); where it isn't a restart, a vertex takes it as hidden too. No run
// reaches it either: the all-ones tag is no run's.
const HIDDEN_INDEX: u32 = 0u;
const RESTART_INDEX: u32 = 0xffffffffu;

// Every chunk is drawn as this many vertices; the excess is degenerate.
const CHUNK_VERTICES: u32 = 768u;

// The tile's origin, camera-relative: whole quanta subtract as integers,
// exactly however the compiler reorders float arithmetic, then the
// remainders, which keeps it to the camera's f64 position at any zoom.
fn tile_origin(tile: Tile) -> vec2<f32> {
    return vec2<f32>(tile.origin - frame.camera_whole) * frame.quantum
        + (tile.origin_rest - frame.camera_rest);
}

// The anchor of the tile's patches in space, camera-relative.
fn tile_globe_origin(tile: Tile) -> vec3<f32> {
    let whole = vec3<i32>(tile.globe_origin[0], tile.globe_origin[1], tile.globe_origin[2]);
    let rest = vec3<f32>(tile.globe_origin_rest[0], tile.globe_origin_rest[1], tile.globe_origin_rest[2]);
    return vec3<f32>(whole - frame.globe_target_whole) * frame.globe_quantum
        + (rest - frame.globe_target_rest);
}

// World units per logical pixel at the placement's scale.
fn tile_world_per_px(tile: Tile) -> f32 {
    return frame.world_per_px * tile.px_scale;
}

// An affine tile's data units → camera-relative world (project.wgsl
// handles warped tiles too).
fn tile_affine(tile: Tile, p: vec2<f32>) -> vec2<f32> {
    return tile_origin(tile) + vec2<f32>(p.x, -p.y) * tile.scale;
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
    return owned(tile, p) && landed(tile, p);
}

// Whether the placement draws data at `p`, in its clip rectangles.
fn owned(tile: Tile, p: vec2<f32>) -> bool {
    for (var i = 0u; i < 4u; i++) {
        let lo = unpack_i16x2(tile.clips[i].x);
        let hi = unpack_i16x2(tile.clips[i].y);
        if all(p >= lo) && all(p < hi) {
            return true;
        }
    }
    return false;
}

// The box around the placement's clip rectangles within its bound, as
// (min, max); empty (min past max) when it draws nothing.
fn clips_box(tile: Tile) -> vec4<f32> {
    var lo = vec2<f32>(1e9);
    var hi = vec2<f32>(-1e9);
    for (var i = 0u; i < 4u; i++) {
        let a = unpack_i16x2(tile.clips[i].x);
        let b = unpack_i16x2(tile.clips[i].y);
        if all(a < b) {
            lo = min(lo, a);
            hi = max(hi, b);
        }
    }
    return vec4<f32>(max(lo, unpack_i16x2(tile.bound.x)), min(hi, unpack_i16x2(tile.bound.y)));
}

// A pipeline's plane variant (orb-render's plane pipelines): every tile is
// affine on the plane, with no globe and no terrain, so the projections
// leave out the warps', the sphere's and the terrain's terms. Adreno's
// compilers inline every call, so those terms cost each projection's
// compile whether a frame runs them or not; without them a cold load
// compiles its labels in half the time (#69). Off by default: the full
// variant draws every view.
override plane: bool = false;

// Whether a placement draws through a planar warp's patches.
fn warped(tile: Tile) -> bool {
    return !plane && tile.warp_cells != 0u;
}

// A placement's clip as its fragments test it (`clip_drawn`): the one
// rectangle it owns, within its bound, where its tile is affine and its
// content lands where it was (`moved` false, or a map placement, which
// owns where content lands), so the test reads no tile; otherwise
// CLIP_FETCHED, and fragments test the tile's rectangles and bound. On a
// pitched city view, reading the tile in every line and fill fragment
// cost a fifth of the frame; redrawing drape slots over the Alps, a
// quarter of the drape pass, image fragments included. The row holds it
// packed for content that lands where it was (orb-render's
// `TileRow::fragment_clip`): working it out in every vertex cost lines a
// sixth of their vertex time on the moto g (#180). The vertex unpacks it
// into (min, max): the Adreno 619's GLES compiler miscompiled the
// fragment stages' unpacking of it, and WebGL 2 there drew no fill or
// line, every fragment outside its rectangle.
const ROW_CLIP_FETCHED: vec2<u32> = vec2<u32>(0x80008000u, 0x80008000u);
// ROW_CLIP_FETCHED unpacked.
const CLIP_FETCHED: vec4<f32> = vec4<f32>(-32768.0);
fn fragment_clip(tile: Tile, moved: bool) -> vec4<f32> {
    if moved && draped_placement(tile) {
        return CLIP_FETCHED;
    }
    return row_clip(tile);
}

// The row's packed clip as (min, max).
fn row_clip(tile: Tile) -> vec4<f32> {
    return vec4<f32>(unpack_i16x2(tile.clip.x), unpack_i16x2(tile.clip.y));
}

// Whether `p` lies in a `fragment_clip` rectangle.
fn in_clip(clip: vec4<f32>, p: vec2<f32>) -> bool {
    return all(p >= clip.xy) && all(p < clip.zw);
}

// Whether `p` lies where the placement's content may land (`bound`).
fn landed(tile: Tile, p: vec2<f32>) -> bool {
    return all(p >= unpack_i16x2(tile.bound.x)) && all(p < unpack_i16x2(tile.bound.y));
}

// Whether the placement draws into a drape slot, bounded by it: never in
// a plane variant, which draws no terrain.
fn draped_placement(tile: Tile) -> bool {
    return !plane && tile.bound.x != BOUND_EVERYWHERE_MIN;
}
// `ClipRect::EVERYWHERE`'s packed minimum.
const BOUND_EVERYWHERE_MIN: u32 = 0x80008000u;
