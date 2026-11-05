//! The GPU scene (plan §8.2): persistent tables a frame reads, the arenas
//! that hold tile streams, and the bookkeeping that keeps them consistent.
//!
//! A camera-only frame writes the frame block and the tile rows; everything
//! else changes only when meshes arrive or leave. Row layouts match the
//! structs in `orb-shaders`' `types.wgsl` and `cull.wgsl`.

#![forbid(unsafe_code)]

mod arena;
mod glyph;
mod image;
mod line;
mod pages;
mod shelf;
mod symbol;
pub use glyph::{GLYPH_PADDING, GLYPH_PAGE, GLYPH_ROW_WORDS, GlyphAtlas};
pub use image::{ImageAtlas, ImageBindings, ImageKey, ImageMetadata, ImageUpload};

use std::ops::Range;

use arena::{Arena, Slots};
use orb_hal::{BufferDesc, BufferId, BufferUsages, Gpu, HostBytesId};
use orb_layout::Mesh;
pub use orb_layout::{Chunk, TileBox};
pub use pages::{PageTexture, Pixels, TilePage};
pub use symbol::{
    PLACED_WORDS, QUAD_WORDS, SYMBOL_HEADER_WORDS, SYMBOL_STATE_WORDS, SymbolMeshData,
    pack_image_rect, symbol_header,
};

fn state_base(range: &Range<u32>) -> u32 {
    if range.is_empty() {
        NO_STATE
    } else {
        range.start
    }
}

fn write_chunk_word(
    gpu: &mut dyn Gpu,
    chunks: BufferId,
    mirror: Option<&mut CullTables>,
    chunk: u32,
    word: u64,
    value: u32,
) {
    let at = u64::from(chunk) * CHUNK_WORDS as u64 + word;
    write_words(
        gpu,
        chunks,
        mirror.map(|m| &mut m.chunks),
        at as u32,
        &[value],
    );
}

/// Writes words into a table, and into its CPU copy on Compat.
fn write_words(
    gpu: &mut dyn Gpu,
    buffer: BufferId,
    mirror: Option<&mut Vec<u32>>,
    word: u32,
    words: &[u32],
) {
    gpu.write_buffer(buffer, u64::from(word) * 4, bytemuck::cast_slice(words));
    if let Some(mirror) = mirror {
        write_words_cpu(mirror, word, words);
    }
}

/// CPU copies of the tables the cull reads, kept on Compat for its CPU
/// cull (plan §8.3): rows as the GPU tables hold them.
#[derive(Debug, Default)]
pub struct CullTables {
    /// Tile placement rows by slot.
    pub tiles: Vec<u32>,
    /// Chunk rows by slot.
    pub chunks: Vec<u32>,
    /// `(chunk, placement, slot, run, drawn)` rows, grouped by node; the
    /// cull keeps `drawn` current.
    pub candidates: Vec<u32>,
    /// Per node: its slice of the candidate and visible lists, margin,
    /// draw word, elevation and compacted slots (`cull.wgsl`'s `NodeCull`).
    pub node_cull: Vec<u32>,
    /// The index arena, which indexed nodes' slots copy.
    pub indices: Vec<u32>,
}

/// Words of a [`CullTables::tiles`] row.
pub const TILE_ROW_WORDS: usize = TILE_WORDS;
/// Words of a [`CullTables::chunks`] row.
pub const CHUNK_ROW_WORDS: usize = CHUNK_WORDS;
/// Words of a [`CullTables::candidates`] row.
pub const CANDIDATE_ROW_WORDS: usize = CANDIDATE_WORDS;
/// Words of a [`CullTables::node_cull`] row.
pub const NODE_CULL_ROW_WORDS: usize = NODE_CULL_WORDS;
/// Words of a visible-list entry: chunk and placement.
pub const VISIBLE_ROW_WORDS: usize = VISIBLE_WORDS;

/// Per-frame values every pipeline reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameBlock {
    /// Camera-relative world → clip, column-major.
    pub view_proj: [[f32; 4]; 4],
    /// Logical pixels.
    pub viewport: [f32; 2],
    /// World units per logical pixel at the camera target.
    pub world_per_px: f32,
    /// Physical pixels per logical pixel.
    pub pixel_ratio: f32,
    /// Style zoom of the camera, exactly as the CPU evaluates paint at it:
    /// the GPU interpreter branches on its f64 bits.
    pub zoom: f64,
    /// Clockwise map rotation, in radians.
    pub bearing: f32,
    /// A camera-relative world offset that puts image tiles on the pixel
    /// grid, as GL JS's aligned projection matrix does.
    pub align: [f32; 2],
    /// Projected world units per v8 extrusion metre at the camera target.
    pub extrusion_scale: f32,
    /// Top logical row where the finite view frustum intersects ground.
    pub ground_horizon: f32,
    /// How far symbol fades have run since the last placement commit (GL
    /// JS `symbolFadeChange`).
    pub symbol_fade_change: f32,
    /// Camera pitch, radians.
    pub pitch: f32,
    /// Camera roll about the view axis, radians.
    pub roll: f32,
    /// Milliseconds since the map began, which image fades measure from
    /// (their rows hold when each tile arrived).
    pub time: f32,
    /// The terrain elevated content stands on, if any.
    pub terrain: Option<FrameTerrain>,
    /// The style's lights (decision 0026); without them lit surfaces are
    /// black.
    pub lights: Option<FrameLights>,
    /// The view's globe, when it is one at its zoom.
    pub globe: Option<FrameGlobe>,
}

/// The style's lights as shaders take them (decision 0026): the ambient
/// and directional terms of GL JS's flat response, each a premultiplied
/// colour and a level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameLights {
    pub ambient: [f32; 3],
    pub ambient_level: f32,
    pub directional: [f32; 3],
    pub directional_level: f32,
    /// The directional light's position as GL JS's `sphericalToCartesian`
    /// gives it: tile axes (y south) with the horizontal parts negated, as
    /// long as v8's radial position (1 for scene lights). Viewport
    /// anchoring is already applied.
    pub position: [f32; 3],
    /// Whether the terrain surface is lit too.
    pub terrain: bool,
}

/// A globe view's terms (`project.wgsl`): positions on the sphere are
/// camera-relative ECEF metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameGlobe {
    /// Camera-relative globe space → clip, column-major.
    pub view_proj: [[f32; 4]; 4],
    /// The target's ECEF position, for directions on the sphere.
    pub target: [f32; 3],
    /// How much of a globe the view is: clip positions mix the plane's
    /// (0) and the globe's (1).
    pub amount: f32,
    /// The horizon plane as `(n, w)`: a camera-relative position `p` on
    /// the sphere faces the eye where `dot(n, p) + w >= 0`, rising to 1
    /// under the eye. Depth comes from it, as in GL JS.
    pub horizon: [f32; 4],
    /// Metres on the globe per plane world unit at the target: offsets and
    /// heights in world units scale by it.
    pub metres_per_unit: f32,
    /// The eye, camera-relative: 3D content behind the sphere hides.
    pub eye: [f32; 3],
}

/// What elevated content reads of the terrain's DEM page.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameTerrain {
    /// GL JS's unpack vector.
    pub unpack: [f32; 4],
    pub exaggeration: f32,
    /// DEM pixels per side, without borders.
    pub dem_dim: f32,
}

const FRAME_WORDS: usize = 84;

/// Bytes of a frame block's uniform buffer.
pub const FRAME_BYTES: u64 = FRAME_WORDS as u64 * 4;

/// Writes a frame block into a uniform buffer of [`FRAME_BYTES`]: the
/// scene's own, or another view's of the same scene (the terrain drape's).
pub fn write_frame(gpu: &mut dyn Gpu, buffer: BufferId, frame: &FrameBlock) {
    let mut words = [0u32; FRAME_WORDS];
    for (i, v) in frame.view_proj.iter().flatten().enumerate() {
        words[i] = v.to_bits();
    }
    words[16] = frame.viewport[0].to_bits();
    words[17] = frame.viewport[1].to_bits();
    words[18] = frame.world_per_px.to_bits();
    words[19] = frame.pixel_ratio.to_bits();
    words[20] = (frame.zoom as f32).to_bits();
    words[21] = frame.bearing.to_bits();
    words[22] = frame.align[0].to_bits();
    words[23] = frame.align[1].to_bits();
    words[24] = frame.extrusion_scale.to_bits();
    words[25] = frame.ground_horizon.to_bits();
    words[26] = frame.time.to_bits();
    words[27] = frame.symbol_fade_change.to_bits();
    words[28] = frame.pitch.to_bits();
    words[29] = frame.roll.to_bits();
    words[30] = frame.zoom.to_bits() as u32;
    words[31] = (frame.zoom.to_bits() >> 32) as u32;
    if let Some(t) = frame.terrain {
        for (word, v) in words[32..36].iter_mut().zip(t.unpack) {
            *word = v.to_bits();
        }
        words[36] = t.exaggeration.to_bits();
        words[37] = t.dem_dim.to_bits();
    }
    if let Some(l) = frame.lights {
        let mut put = |at: usize, v: [f32; 3], w: f32| {
            for (word, v) in words[at..at + 4].iter_mut().zip([v[0], v[1], v[2], w]) {
                *word = v.to_bits();
            }
        };
        put(40, l.ambient, l.ambient_level);
        put(44, l.directional, l.directional_level);
        put(48, l.position, if l.terrain { 1.0 } else { 0.0 });
    }
    if let Some(g) = frame.globe {
        for (i, v) in g.view_proj.iter().flatten().enumerate() {
            words[52 + i] = v.to_bits();
        }
        for (word, v) in words[68..72].iter_mut().zip(g.horizon) {
            *word = v.to_bits();
        }
        for (word, v) in words[72..75].iter_mut().zip(g.target) {
            *word = v.to_bits();
        }
        words[75] = g.amount.to_bits();
        words[76] = g.metres_per_unit.to_bits();
        for (word, v) in words[80..83].iter_mut().zip(g.eye) {
            *word = v.to_bits();
        }
    }
    gpu.write_buffer(buffer, 0, bytemuck::cast_slice(&words));
}

/// One placement of a tile's data: its transform into camera-relative world
/// space, `origin + (x, -y) · scale` (or, with a planar warp, `origin` plus
/// the warp's quadratic patches; `scale` is then its local scale at the
/// centre), and where it may draw. On a globe, its position on the sphere
/// is `globe_origin` plus its warp's patches in space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileRow {
    pub origin: [f32; 2],
    pub scale: f32,
    /// The tile's zoom level.
    pub zoom: f32,
    /// Rectangles in data units the placement draws in; unused entries are
    /// [`ClipRect::EMPTY`].
    pub clips: [ClipRect; 4],
    /// Global pixel origin at the integer camera zoom, modulo 2^32.
    pub pattern_origin: [u32; 2],
    pub extent: f32,
    /// The world copy it draws; symbol meshes keep state words per copy.
    pub wrap: i32,
    /// World units per logical pixel of the camera, at this placement's
    /// scale: pixel sizes (line widths, translations) convert with it. A
    /// map placement has the camera's; a drape placement scales it with
    /// its slot.
    pub world_per_px: f32,
    /// The DEM tile elevated content of this placement stands on.
    pub dem: DemRef,
    /// The lowest and highest terrain elevation under the placement, in
    /// metres with exaggeration, for culling.
    pub terrain_range: [f32; 2],
    /// Which render nodes draw it.
    pub target: PlacementTarget,
    /// Its planar patches (`project.wgsl`), or [`TileWarp::NONE`].
    pub warp: TileWarp,
    /// On a globe view, where it is on the sphere.
    pub globe: Option<RowGlobe>,
}

/// A placement on a globe: its patches in space, their anchor less the
/// camera target in metres, and whether it also has a position on the
/// plane (a tile the plane can't hold, such as one at a pole, draws only
/// on the globe).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowGlobe {
    pub warp: TileWarp,
    pub origin: [f32; 3],
    pub on_plane: bool,
}

/// A tile's biquadratic patches in the property arena (plan §9.2's
/// `GpuPerVertex`): `cells`² of them over the tile, row by row, with
/// [`WARP_PATCH_WORDS`] f32 each on a plane and [`GLOBE_PATCH_WORDS`] on a
/// globe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileWarp {
    base: u32,
    cells: u32,
    words: u32,
}

impl Default for TileWarp {
    fn default() -> Self {
        Self::NONE
    }
}

impl TileWarp {
    /// No warp: an affine tile.
    pub const NONE: Self = Self {
        base: 0,
        cells: 0,
        words: 0,
    };

    /// Its first word in the property arena.
    pub fn base(self) -> u32 {
        self.base
    }

    /// Its patches per tile side; 0 for none.
    pub fn cells(self) -> u32 {
        self.cells
    }
}

/// Where a placement draws: on the map, or into the terrain's drape
/// textures for the render nodes of one drape stack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PlacementTarget {
    #[default]
    Map,
    Drape(u8),
}

/// A placement's DEM: tile data unit `p` reads the DEM page's `layer` at
/// `(p / extent + offset) / 2^depth` of the DEM tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DemRef {
    pub layer: u16,
    pub depth: u8,
    pub offset: [u16; 2],
}

impl DemRef {
    /// No DEM: the ground at sea level.
    pub const NONE: Self = Self {
        layer: u16::MAX,
        depth: 0,
        offset: [0; 2],
    };

    fn words(self) -> [u32; 2] {
        [
            u32::from(self.layer) | u32::from(self.depth) << 16,
            u32::from(self.offset[0]) | u32::from(self.offset[1]) << 16,
        ]
    }
}

/// A rectangle in tile data units, minimum inclusive and maximum exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipRect {
    pub min: [i16; 2],
    pub max: [i16; 2],
}

impl ClipRect {
    pub const EMPTY: Self = Self {
        min: [i16::MAX; 2],
        max: [i16::MIN; 2],
    };

    pub const fn new(min: [i16; 2], max: [i16; 2]) -> Self {
        Self { min, max }
    }
}

/// Words of a tile row (`types.wgsl`'s `Tile`, `TILE_WORDS` there).
const TILE_WORDS: usize = 28;
/// Words per biquadratic patch of a planar warp: nine nodes' positions
/// (`project.wgsl`).
pub const WARP_PATCH_WORDS: usize = 18;
/// Words per biquadratic patch on a globe: nine nodes' positions in space.
pub const GLOBE_PATCH_WORDS: usize = 27;
const CHUNK_WORDS: usize = 8;
const CANDIDATE_WORDS: usize = 5;
const VISIBLE_WORDS: usize = 2;
const NODE_CULL_WORDS: usize = 8;
/// Words per node of [`SceneBuffers::args`].
pub const ARGS_WORDS: usize = 8;
/// Candidates per cull workgroup of an indexed node (`cull.wgsl`).
const INDEXED_BATCH: u32 = 64;
/// A candidate's `drawn` word before its first cull: neither drawn (1) nor
/// hidden (0), so the cull pass writes its slot (`cull.wgsl`).
const UNKNOWN: u32 = u32::MAX;

/// How many of each thing the scene can hold. Buffers are allocated once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneCapacity {
    /// Tile placements: a tile drawn in two world copies takes two.
    pub placements: u32,
    pub chunks: u32,
    /// `(chunk, placement)` pairs the cull pass considers, over all nodes.
    pub candidates: u32,
    pub nodes: u32,
    pub vertex_words: u32,
    pub index_words: u32,
    /// Slots of indexed nodes' candidates in the compacted index buffer:
    /// each candidate's chunk indices once per placement.
    pub compacted_words: u32,
    pub prop_words: u32,
    /// Rows of the glyph atlas's table, which follows the property arena.
    pub glyph_rows: u32,
}

impl Default for SceneCapacity {
    fn default() -> Self {
        Self {
            placements: 1024,
            chunks: 1 << 17,
            candidates: 1 << 18,
            // A render node per drawn layer and symbol pass: Americana
            // needs about 400.
            nodes: 1024,
            vertex_words: 16 << 20,
            index_words: 16 << 20,
            compacted_words: 16 << 20,
            prop_words: 4 << 20,
            glyph_rows: 1 << 16,
        }
    }
}

/// The scene's GPU buffers, for binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneBuffers {
    pub frame: BufferId,
    /// Tile placements.
    pub tiles: BufferId,
    pub chunks: BufferId,
    /// `(chunk, placement)` pairs for the cull pass, grouped by node, and
    /// what it last wrote for indexed nodes' candidates.
    pub candidates: BufferId,
    pub node_cull: BufferId,
    /// Per node, `(chunk, placement)` pairs: the visible list the cull pass
    /// writes for pulled and point nodes, the runs for indexed nodes.
    pub visible: BufferId,
    /// Per node, [`ARGS_WORDS`] words the cull pass writes: indirect draw
    /// arguments (four for `drawIndirect`, five for
    /// `drawIndexedIndirect`), then the counters of `cull.wgsl`.
    pub args: BufferId,
    pub vertices: BufferId,
    /// Mesh indices: what pulled chunks read, and what the cull pass
    /// copies into indexed nodes' slots.
    pub indices: BufferId,
    /// Indexed nodes' index buffer: their candidates' slots, each the
    /// chunk's indices tagged with its run, or hidden.
    pub compacted: BufferId,
    pub props: BufferId,
}

/// A stream of u32 words to upload: in core memory, or held by the host
/// (plan §10.4: payloads reach the GPU without entering the render
/// worker's wasm memory).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Words<'a> {
    Core(&'a [u32]),
    Host {
        bytes: HostBytesId,
        /// Byte offset of the first word in `bytes`.
        offset: u64,
        /// Number of words.
        len: u32,
    },
}

/// Copies words into a CPU table at word `at`, growing it.
fn write_words_cpu(table: &mut Vec<u32>, at: u32, words: &[u32]) {
    let end = at as usize + words.len();
    if table.len() < end {
        table.resize(end, 0);
    }
    table[at as usize..end].copy_from_slice(words);
}

impl Words<'_> {
    /// The words in core memory: copied from the host if it holds them.
    fn read(&self, gpu: &dyn Gpu) -> std::borrow::Cow<'_, [u32]> {
        match *self {
            Self::Core(words) => std::borrow::Cow::Borrowed(words),
            Self::Host { bytes, offset, len } => {
                let bytes = gpu.read_host_bytes(bytes, offset..offset + u64::from(len) * 4);
                std::borrow::Cow::Owned(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|w| u32::from_le_bytes(*w))
                        .collect(),
                )
            }
        }
    }

    pub fn len(&self) -> u32 {
        match self {
            Self::Core(words) => words.len() as u32,
            Self::Host { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn write(&self, gpu: &mut dyn Gpu, buffer: BufferId, word_offset: u32) {
        let offset = u64::from(word_offset) * 4;
        match *self {
            Self::Core(words) => gpu.write_buffer(buffer, offset, bytemuck::cast_slice(words)),
            Self::Host {
                bytes,
                offset: start,
                len,
            } => {
                if len > 0 {
                    gpu.write_buffer_from_host(
                        buffer,
                        offset,
                        bytes,
                        start..start + u64::from(len) * 4,
                    );
                }
            }
        }
    }
}

/// A mesh's streams and chunks, ready to upload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshData<'a> {
    pub vertices: Words<'a>,
    pub indices: Words<'a>,
    /// Property rows.
    pub props: Words<'a>,
    /// Words per feature of the rows: the family's, then any input
    /// columns of interpreted paint. Shaders that read rows take their
    /// stride from the mesh, so rows and stride always agree.
    pub row_words: u32,
    pub chunks: &'a [Chunk],
}

/// The words and chunks of a mesh to insert, for [`Scene::fits`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshSize {
    pub vertices: u32,
    pub indices: u32,
    pub props: u32,
    pub chunks: u32,
}

impl<'a> MeshData<'a> {
    /// A mesh laid out in core memory, with its property rows.
    pub fn new(mesh: &'a Mesh, props: &'a [u32], row_words: u32) -> Self {
        Self {
            vertices: Words::Core(&mesh.vertices),
            indices: Words::Core(&mesh.indices),
            props: Words::Core(props),
            row_words,
            chunks: &mesh.chunks,
        }
    }
}

/// A tile's data in the scene: its meshes and placements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileId(u32);

/// A mesh resident in the scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshId(u32);

/// Metres per unit of a placement's packed terrain range (`types.wgsl`'s
/// `TERRAIN_RANGE_UNIT`).
const TERRAIN_RANGE_UNIT: f32 = 4.0;

/// A terrain range as two i16s of [`TERRAIN_RANGE_UNIT`], rounded
/// outward so it still covers the range.
fn pack_range([lo, hi]: [f32; 2]) -> u32 {
    let unit = |v: f32| v.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16 as u16;
    u32::from(unit((lo / TERRAIN_RANGE_UNIT).floor()))
        | u32::from(unit((hi / TERRAIN_RANGE_UNIT).ceil())) << 16
}

/// The scene is full; evict something and retry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutOfSpace;

#[derive(Debug, Default)]
struct TileEntry {
    meshes: Vec<MeshId>,
    placements: Vec<Placement>,
}

/// A tile placement: its row in the placement table, and the row's values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub slot: u32,
    pub row: TileRow,
}

#[derive(Debug)]
struct MeshEntry {
    tile: TileId,
    node: u32,
    chunk_slots: Vec<u32>,
    chunks: Vec<Chunk>,
    bounds: TileBox,
    vertices: Range<u32>,
    indices: Range<u32>,
    props: Range<u32>,
    row_words: u32,
    /// Per feature slot, its feature-state index, in the property arena.
    state: Range<u32>,
    /// A symbol mesh's state words, in the property arena.
    symbol_states: Range<u32>,
    /// World copies the state words are kept for.
    symbol_copies: u32,
    /// A symbol mesh's draw-order table: its offset in the mesh's vertex
    /// words, and its quads.
    symbol_order: [u32; 2],
}

/// Chunk word that holds the first word of the mesh's feature-state
/// remap, or [`NO_STATE`].
const CHUNK_STATE_WORD: u64 = 6;
const CHUNK_PROPS_WORD: u64 = 3;
const CHUNK_ROW_WORDS_WORD: u64 = 7;

/// No feature-state remap: every feature has the default state.
pub const NO_STATE: u32 = u32::MAX;

/// A resident mesh as CPU-side culling reads it (the Compat path of plan
/// §8.3, and benchmarks).
#[derive(Clone, Copy, Debug)]
pub struct ResidentMesh<'a> {
    /// The mesh's words in the vertex arena.
    pub vertices: &'a Range<u32>,
    /// The mesh's indices in the index arena.
    pub indices: &'a Range<u32>,
    /// First word of its property rows in the property arena.
    pub prop_base: u32,
    /// First word of its feature-state remap, or [`NO_STATE`].
    pub state_base: u32,
    /// Words per feature of its property rows.
    pub row_words: u32,
    /// Chunks, with indices relative to [`indices`](Self::indices).
    pub chunks: &'a [Chunk],
    /// The union of the chunks' bounds.
    pub bounds: TileBox,
    /// Where the mesh's tile draws.
    pub placements: &'a [Placement],
}

#[derive(Debug)]
struct NodeEntry {
    meshes: Vec<MeshId>,
    cull: NodeCull,
}

/// How the cull pass treats a render node's chunks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeCull {
    /// Extra margin around chunk bounds, in pixels.
    pub margin_px: f32,
    /// Maximum nonnegative elevation in metres (base and roof).
    pub elevation: f32,
    pub draw: NodeDraw,
    /// Hidden nodes contribute no candidates and draw nothing.
    pub visible: bool,
    /// The placements it draws: on the map, or those of its drape stack,
    /// which the cull keeps by their clip rectangles alone.
    pub target: PlacementTarget,
}

/// How a render node draws its visible chunks, which the cull pass
/// prepares (plan §8.3, decision 0002).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeDraw {
    /// Image meshes, by vertex pulling: one instance of 768 vertices per
    /// visible chunk, from the node's visible list. Their fragments clip in
    /// the image's own tile units (a warped tile's mesh is in other units),
    /// so the cull doesn't narrow their bounds to the clip rectangles.
    Images,
    /// Point families: 256 six-vertex instances per visible chunk. Anchors
    /// are clipped, then the full footprints expanded.
    Points,
    /// Compacted indices: each candidate has a slot in the compacted index
    /// buffer, which the cull pass fills with the chunk's indices when it
    /// becomes visible and with hidden indices when it leaves the view;
    /// one `drawIndexedIndirect` covers the node's slots. Indices are
    /// `run << run_shift | vertex`, a run being a mesh in one placement.
    Indexed,
}

impl NodeDraw {
    fn word(self) -> u32 {
        match self {
            Self::Images => 0,
            Self::Points => 1,
            Self::Indexed => 2,
        }
    }
}

/// The cull's flag on a node's draw word: its candidates are drape
/// placements (`cull.wgsl`'s `DRAPED`).
const DRAPED: u32 = 1 << 8;

#[derive(Debug)]
pub struct Scene {
    buffers: SceneBuffers,
    capacity: SceneCapacity,
    placement_slots: Slots,
    chunk_slots: Slots,
    vertex_arena: Arena,
    index_arena: Arena,
    prop_arena: Arena,
    tiles: Vec<Option<TileEntry>>,
    free_tiles: Vec<u32>,
    meshes: Vec<Option<MeshEntry>>,
    free_meshes: Vec<u32>,
    nodes: Vec<NodeEntry>,
    candidates_dirty: bool,
    candidate_count: u32,
    candidates_dropped: u32,
    compacted_words: u32,
    /// Rows of the cull dispatch: the most batches an indexed node has.
    cull_batches: u32,
    /// On Compat, what the CPU cull reads.
    cull: Option<CullTables>,
    revision: u64,
    /// Placement rows written since the last `take_placements_written`.
    placements_written: u32,
}

/// Counts for the stats HUD and memory accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneStats {
    pub meshes: u32,
    pub candidates: u32,
    /// Candidates beyond [`SceneCapacity::candidates`] or
    /// [`SceneCapacity::compacted_words`], or beyond an indexed node's runs
    /// (see [`NodeDraw::Indexed`]), not drawn.
    pub candidates_dropped: u32,
    pub vertex_words: u32,
    pub index_words: u32,
    pub compacted_words: u32,
    pub prop_words: u32,
}

impl Scene {
    pub fn new(gpu: &mut dyn Gpu, capacity: SceneCapacity) -> Self {
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        // Compat culls on the CPU: only its outputs reach the GPU, so the
        // tables only the cull dispatch reads stay in core memory.
        let compat = gpu.caps().tier == orb_hal::Tier::Compat;
        let cull_only = if compat {
            BufferUsages::COPY_DST
        } else {
            storage
        };
        let mut buffer = |label: &str, words: u64, usage: BufferUsages| {
            gpu.create_buffer(&BufferDesc {
                label,
                size: (words * 4).max(16),
                usage,
            })
        };
        let c = capacity;
        let words = |count: u32, per: usize| u64::from(count) * per as u64;
        let buffers = SceneBuffers {
            frame: buffer(
                "frame",
                FRAME_WORDS as u64,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            tiles: buffer("tiles", words(c.placements, TILE_WORDS), storage),
            chunks: buffer("chunks", words(c.chunks, CHUNK_WORDS), storage),
            candidates: buffer(
                "candidates",
                words(c.candidates, CANDIDATE_WORDS),
                cull_only,
            ),
            node_cull: buffer("node cull", words(c.nodes, NODE_CULL_WORDS), storage),
            visible: buffer("visible", words(c.candidates, VISIBLE_WORDS), storage),
            args: buffer(
                "draw args",
                words(c.nodes, ARGS_WORDS),
                cull_only
                    | BufferUsages::INDIRECT
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            ),
            vertices: buffer("vertices", u64::from(c.vertex_words), storage),
            indices: buffer("indices", u64::from(c.index_words), storage),
            compacted: buffer(
                "compacted indices",
                u64::from(c.compacted_words),
                if compat {
                    BufferUsages::INDEX | BufferUsages::COPY_DST
                } else {
                    BufferUsages::STORAGE | BufferUsages::INDEX
                },
            ),
            props: buffer(
                "props",
                u64::from(c.prop_words) + u64::from(c.glyph_rows) * GLYPH_ROW_WORDS as u64,
                storage,
            ),
        };
        Self {
            buffers,
            capacity,
            placement_slots: Slots::new(c.placements),
            chunk_slots: Slots::new(c.chunks),
            vertex_arena: Arena::new(c.vertex_words),
            index_arena: Arena::new(c.index_words),
            prop_arena: Arena::new(c.prop_words),
            tiles: Vec::new(),
            free_tiles: Vec::new(),
            meshes: Vec::new(),
            free_meshes: Vec::new(),
            nodes: Vec::new(),
            candidates_dirty: true,
            candidate_count: 0,
            candidates_dropped: 0,
            compacted_words: 0,
            cull_batches: 1,
            cull: compat.then(CullTables::default),
            revision: 0,
            placements_written: 0,
        }
    }

    pub fn buffers(&self) -> &SceneBuffers {
        &self.buffers
    }

    /// The CPU copies the Compat cull reads, as of the last
    /// [`prepare`](Self::prepare); `None` on Modern.
    pub fn cull_tables(&mut self) -> Option<&mut CullTables> {
        self.cull.as_mut()
    }

    pub fn capacity(&self) -> &SceneCapacity {
        &self.capacity
    }

    /// The first word of the glyph table in the property buffer: its
    /// [`SceneCapacity::glyph_rows`] rows follow the arena.
    pub fn glyph_table(&self) -> u32 {
        self.capacity.prop_words
    }

    /// Counts changes to meshes and placement counts; equal revisions mean
    /// the same meshes in the same slots.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The cull dispatch's grid, as of the last [`prepare`](Self::prepare):
    /// one workgroup per node, by as many rows as the indexed node with the
    /// most candidates has batches of them.
    pub fn cull_grid(&self) -> [u32; 2] {
        [self.nodes.len() as u32, self.cull_batches]
    }

    /// Orders tiles within each render node independently of upload order.
    /// Meshes of the same tile keep their relative order; tiles not listed
    /// follow the listed ones. The order persists until changed again.
    pub fn set_tile_order(&mut self, tiles: &[TileId]) {
        let order: std::collections::HashMap<_, _> = tiles
            .iter()
            .enumerate()
            .map(|(index, &tile)| (tile, index))
            .collect();
        let meshes = &self.meshes;
        for node in &mut self.nodes {
            let key = |id: &MeshId| {
                let tile = meshes[id.0 as usize].as_ref().expect("live mesh").tile;
                order.get(&tile).copied().unwrap_or(usize::MAX)
            };
            if !node.meshes.is_sorted_by_key(key) {
                node.meshes.sort_by_key(key);
                self.candidates_dirty = true;
                self.revision += 1;
            }
        }
    }

    /// The meshes of render `node` in candidate order.
    pub fn node_meshes(&self, node: u32) -> impl Iterator<Item = ResidentMesh<'_>> {
        self.nodes[node as usize].meshes.iter().map(|id| {
            let mesh = self.meshes[id.0 as usize]
                .as_ref()
                .expect("node lists live meshes");
            let tile = self.tiles[mesh.tile.0 as usize]
                .as_ref()
                .expect("meshes have tiles");
            ResidentMesh {
                vertices: &mesh.vertices,
                indices: &mesh.indices,
                prop_base: mesh.props.start,
                state_base: state_base(&mesh.state),
                row_words: mesh.row_words,
                chunks: &mesh.chunks,
                bounds: mesh.bounds,
                placements: &tile.placements,
            }
        })
    }

    pub fn stats(&self) -> SceneStats {
        SceneStats {
            meshes: (self.meshes.len() - self.free_meshes.len()) as u32,
            candidates: self.candidate_count,
            candidates_dropped: self.candidates_dropped,
            vertex_words: self.vertex_arena.used(),
            index_words: self.index_arena.used(),
            compacted_words: self.compacted_words,
            prop_words: self.prop_arena.used(),
        }
    }

    /// Renumbers the render nodes with their meshes: node `i` becomes
    /// `map[i]`, and the scene has `nodes.len()` nodes. Nodes mapped to
    /// `None` must hold no meshes.
    pub fn remap_nodes(&mut self, map: &[Option<u32>], nodes: &[NodeCull]) {
        assert_eq!(map.len(), self.nodes.len(), "one entry per node");
        assert!(
            nodes.len() as u32 <= self.capacity.nodes,
            "too many render nodes"
        );
        let mut entries: Vec<NodeEntry> = nodes
            .iter()
            .map(|&cull| NodeEntry {
                meshes: Vec::new(),
                cull,
            })
            .collect();
        for (old, node) in self.nodes.drain(..).enumerate() {
            match map[old] {
                Some(new) => {
                    for &id in &node.meshes {
                        self.meshes[id.0 as usize].as_mut().expect("live mesh").node = new;
                    }
                    entries[new as usize].meshes = node.meshes;
                }
                None => assert!(node.meshes.is_empty(), "removed node {old} holds meshes"),
            }
        }
        self.nodes = entries;
        self.candidates_dirty = true;
        self.revision += 1;
    }

    /// Sets how each render node is culled. Changing the number of nodes
    /// requires the scene to hold no meshes.
    pub fn set_nodes(&mut self, nodes: &[NodeCull]) {
        if nodes.len() != self.nodes.len() {
            assert!(
                self.nodes.iter().all(|n| n.meshes.is_empty()),
                "nodes changed while meshes are resident"
            );
            assert!(
                nodes.len() as u32 <= self.capacity.nodes,
                "too many render nodes"
            );
            self.nodes = nodes
                .iter()
                .map(|&cull| NodeEntry {
                    meshes: Vec::new(),
                    cull,
                })
                .collect();
        } else {
            for (node, &cull) in self.nodes.iter_mut().zip(nodes) {
                node.cull = cull;
            }
        }
        self.candidates_dirty = true;
    }

    /// A new tile with no meshes and no placements.
    pub fn insert_tile(&mut self) -> TileId {
        match self.free_tiles.pop() {
            Some(id) => {
                self.tiles[id as usize] = Some(TileEntry::default());
                TileId(id)
            }
            None => {
                self.tiles.push(Some(TileEntry::default()));
                TileId(self.tiles.len() as u32 - 1)
            }
        }
    }

    /// Removes a tile with its meshes and placements.
    pub fn remove_tile(&mut self, tile: TileId) {
        let entry = self.tiles[tile.0 as usize]
            .take()
            .expect("tile already removed");
        self.free_tiles.push(tile.0);
        for placement in entry.placements {
            self.placement_slots.free(placement.slot);
        }
        for mesh in entry.meshes {
            self.drop_mesh(mesh);
        }
        self.candidates_dirty = true;
        self.revision += 1;
    }

    /// Uploads a warp's patches (`cells`² of them, row by row) for tile
    /// rows to draw through: [`WARP_PATCH_WORDS`] f32 each on a plane,
    /// [`GLOBE_PATCH_WORDS`] on a globe.
    pub fn add_warp(
        &mut self,
        gpu: &mut dyn Gpu,
        cells: u32,
        patches: &[f32],
    ) -> Result<TileWarp, OutOfSpace> {
        let count = (cells * cells) as usize;
        debug_assert!(
            patches.len() == count * WARP_PATCH_WORDS || patches.len() == count * GLOBE_PATCH_WORDS
        );
        let range = self
            .prop_arena
            .alloc(patches.len() as u32)
            .ok_or(OutOfSpace)?;
        let words: Vec<u32> = patches.iter().map(|v| v.to_bits()).collect();
        Words::Core(&words).write(gpu, self.buffers.props, range.start);
        Ok(TileWarp {
            base: range.start,
            cells,
            words: patches.len() as u32,
        })
    }

    /// Frees a warp no row draws through any more.
    pub fn remove_warp(&mut self, warp: TileWarp) {
        if warp.cells > 0 {
            self.prop_arena.free(warp.base..warp.base + warp.words);
        }
    }

    /// Sets where a tile draws this frame: one row per placement, none to
    /// hide it.
    pub fn place_tile(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: TileId,
        rows: &[TileRow],
    ) -> Result<(), OutOfSpace> {
        let entry = self.tiles[tile.0 as usize].as_mut().expect("unknown tile");
        if entry.placements.len() != rows.len() {
            let needed = rows.len().saturating_sub(entry.placements.len()) as u32;
            if needed > self.placement_slots.available() {
                return Err(OutOfSpace);
            }
            while entry.placements.len() > rows.len() {
                let placement = entry.placements.pop().expect("non-empty");
                self.placement_slots.free(placement.slot);
            }
            while entry.placements.len() < rows.len() {
                entry.placements.push(Placement {
                    slot: self.placement_slots.alloc().expect("checked above"),
                    row: rows[entry.placements.len()],
                });
            }
            self.candidates_dirty = true;
            self.revision += 1;
        }
        self.placements_written += rows.len() as u32;
        for (placement, row) in entry.placements.iter_mut().zip(rows) {
            placement.row = *row;
            let slot = placement.slot;
            let pack = |p: [i16; 2]| u32::from(p[0] as u16) | u32::from(p[1] as u16) << 16;
            let mut words = [0u32; TILE_WORDS];
            words[..4].copy_from_slice(&[
                row.origin[0].to_bits(),
                row.origin[1].to_bits(),
                row.scale.to_bits(),
                row.zoom.to_bits(),
            ]);
            words[12..14].copy_from_slice(&row.pattern_origin);
            words[14] = row.extent.to_bits();
            words[15] = row.wrap as u32;
            words[16] = row.world_per_px.to_bits();
            words[17..19].copy_from_slice(&row.dem.words());
            words[19] = pack_range(row.terrain_range);
            words[20] = row.warp.base;
            words[21] = row.warp.cells;
            if let Some(globe) = row.globe {
                words[22..25].copy_from_slice(&globe.origin.map(f32::to_bits));
                words[25] = globe.warp.base;
                words[26] = globe.warp.cells;
                words[27] = u32::from(!globe.on_plane);
            }
            for (i, clip) in row.clips.iter().enumerate() {
                words[4 + i * 2] = pack(clip.min);
                words[5 + i * 2] = pack(clip.max);
            }
            write_words(
                gpu,
                self.buffers.tiles,
                self.cull.as_mut().map(|c| &mut c.tiles),
                slot * TILE_WORDS as u32,
                &words,
            );
        }
        Ok(())
    }

    /// Placement rows written since the last call.
    pub fn take_placements_written(&mut self) -> u32 {
        std::mem::take(&mut self.placements_written)
    }

    /// Uploads a mesh of `tile` for render `node`, with its property rows.
    /// Host-held words are written straight from the host; releasing them
    /// stays with the caller.
    /// Whether meshes of these sizes would all insert now, in order: the
    /// arena ranges and chunk slots [`insert_mesh`](Self::insert_mesh)
    /// would take, found without writing anything.
    pub fn fits(&self, meshes: impl IntoIterator<Item = MeshSize>) -> bool {
        let mut vertices = self.vertex_arena.clone();
        let mut indices = self.index_arena.clone();
        let mut props = self.prop_arena.clone();
        let mut chunks = 0u64;
        for mesh in meshes {
            if vertices.alloc(mesh.vertices).is_none()
                || indices.alloc(mesh.indices).is_none()
                || props.alloc(mesh.props).is_none()
            {
                return false;
            }
            chunks += u64::from(mesh.chunks);
        }
        chunks <= u64::from(self.chunk_slots.available())
    }

    pub fn insert_mesh(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: TileId,
        node: u32,
        mesh: &MeshData<'_>,
    ) -> Result<MeshId, OutOfSpace> {
        assert!(
            (node as usize) < self.nodes.len(),
            "unknown render node {node}"
        );
        assert!(self.tiles[tile.0 as usize].is_some(), "unknown tile");
        let len = |v: usize| u32::try_from(v).map_err(|_| OutOfSpace);
        if self.chunk_slots.available() < len(mesh.chunks.len())? {
            return Err(OutOfSpace);
        }
        let vertices = self
            .vertex_arena
            .alloc(mesh.vertices.len())
            .ok_or(OutOfSpace)?;
        let Some(indices) = self.index_arena.alloc(mesh.indices.len()) else {
            self.vertex_arena.free(vertices);
            return Err(OutOfSpace);
        };
        let Some(prop_range) = self.prop_arena.alloc(mesh.props.len()) else {
            self.vertex_arena.free(vertices);
            self.index_arena.free(indices);
            return Err(OutOfSpace);
        };

        mesh.vertices
            .write(gpu, self.buffers.vertices, vertices.start);
        mesh.indices.write(gpu, self.buffers.indices, indices.start);
        if let Some(cull) = &mut self.cull {
            let words = mesh.indices.read(gpu);
            write_words_cpu(&mut cull.indices, indices.start, &words);
        }
        mesh.props.write(gpu, self.buffers.props, prop_range.start);

        let mut chunk_ids = Vec::with_capacity(mesh.chunks.len());
        let mut bounds = TileBox {
            min: [i16::MAX; 2],
            max: [i16::MIN; 2],
        };
        for chunk in mesh.chunks {
            bounds.min = [0, 1].map(|i| bounds.min[i].min(chunk.bounds.min[i]));
            bounds.max = [0, 1].map(|i| bounds.max[i].max(chunk.bounds.max[i]));
            let id = self.chunk_slots.alloc().expect("checked above");
            let pack = |p: [i16; 2]| u32::from(p[0] as u16) | u32::from(p[1] as u16) << 16;
            let row: [u32; CHUNK_WORDS] = [
                indices.start + chunk.first_index,
                chunk.index_count,
                vertices.start,
                prop_range.start,
                pack(chunk.bounds.min),
                pack(chunk.bounds.max),
                NO_STATE,
                mesh.row_words,
            ];
            write_words(
                gpu,
                self.buffers.chunks,
                self.cull.as_mut().map(|c| &mut c.chunks),
                id * CHUNK_WORDS as u32,
                &row,
            );
            chunk_ids.push(id);
        }

        let entry = MeshEntry {
            tile,
            node,
            chunk_slots: chunk_ids,
            chunks: mesh.chunks.to_vec(),
            bounds,
            vertices,
            indices,
            props: prop_range,
            row_words: mesh.row_words,
            state: 0..0,
            symbol_states: 0..0,
            symbol_copies: 0,
            symbol_order: [0; 2],
        };
        Ok(self.add_mesh(entry))
    }

    /// Registers an uploaded mesh with its node and tile.
    fn add_mesh(&mut self, entry: MeshEntry) -> MeshId {
        let (node, tile) = (entry.node, entry.tile);
        let id = match self.free_meshes.pop() {
            Some(id) => {
                self.meshes[id as usize] = Some(entry);
                MeshId(id)
            }
            None => {
                self.meshes.push(Some(entry));
                MeshId(self.meshes.len() as u32 - 1)
            }
        };
        self.nodes[node as usize].meshes.push(id);
        self.tiles[tile.0 as usize]
            .as_mut()
            .expect("checked above")
            .meshes
            .push(id);
        self.candidates_dirty = true;
        self.revision += 1;
        id
    }

    /// Patches image references after uploading rows from either core or
    /// host memory. No vertex data or bind groups change when images arrive.
    pub fn set_mesh_image(&mut self, gpu: &mut dyn Gpu, mesh: MeshId, word: u32, image: u32) {
        let entry = self.meshes[mesh.0 as usize].as_ref().expect("live mesh");
        assert!(word < entry.props.len() as u32);
        gpu.write_buffer(
            self.buffers.props,
            u64::from(entry.props.start + word) * 4,
            &image.to_le_bytes(),
        );
    }

    /// Replaces a mesh's property rows, keeping its geometry: a column job's
    /// result, `row_words` per feature. Image references are patched again
    /// by the caller.
    pub fn set_mesh_props(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        props: &Words<'_>,
        row_words: u32,
    ) -> Result<(), OutOfSpace> {
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        if row_words != entry.row_words {
            entry.row_words = row_words;
            for &chunk in &entry.chunk_slots {
                write_chunk_word(
                    gpu,
                    self.buffers.chunks,
                    self.cull.as_mut(),
                    chunk,
                    CHUNK_ROW_WORDS_WORD,
                    row_words,
                );
            }
        }
        if props.len() != entry.props.len() as u32 {
            let range = self.prop_arena.alloc(props.len()).ok_or(OutOfSpace)?;
            self.prop_arena
                .free(std::mem::replace(&mut entry.props, range.clone()));
            for &chunk in &entry.chunk_slots {
                write_chunk_word(
                    gpu,
                    self.buffers.chunks,
                    self.cull.as_mut(),
                    chunk,
                    CHUNK_PROPS_WORD,
                    range.start,
                );
            }
            self.revision += 1;
        }
        props.write(gpu, self.buffers.props, entry.props.start);
        Ok(())
    }

    /// Sets the feature-state index of each of a mesh's feature slots, or
    /// (empty) gives every feature the default state.
    pub fn set_mesh_state(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        indices: &[u32],
    ) -> Result<(), OutOfSpace> {
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        if indices.len() != entry.state.len() {
            let range = self
                .prop_arena
                .alloc(indices.len() as u32)
                .ok_or(OutOfSpace)?;
            self.prop_arena
                .free(std::mem::replace(&mut entry.state, range.clone()));
            for &chunk in &entry.chunk_slots {
                write_chunk_word(
                    gpu,
                    self.buffers.chunks,
                    self.cull.as_mut(),
                    chunk,
                    CHUNK_STATE_WORD,
                    state_base(&range),
                );
            }
            self.revision += 1;
        }
        if !indices.is_empty() {
            Words::Core(indices).write(gpu, self.buffers.props, entry.state.start);
        }
        Ok(())
    }

    /// Words of the property buffer for a table outside any mesh, such as
    /// the feature-state table. The words are not cleared.
    pub fn alloc_table(&mut self, words: u32) -> Result<Range<u32>, OutOfSpace> {
        self.prop_arena.alloc(words).ok_or(OutOfSpace)
    }

    pub fn free_table(&mut self, range: Range<u32>) {
        self.prop_arena.free(range);
    }

    /// Writes words of a table from [`alloc_table`](Self::alloc_table),
    /// starting at word `at` of the property buffer.
    pub fn write_table(&mut self, gpu: &mut dyn Gpu, at: u32, words: &[u32]) {
        Words::Core(words).write(gpu, self.buffers.props, at);
    }

    pub fn remove_mesh(&mut self, id: MeshId) {
        let tile = self.meshes[id.0 as usize]
            .as_ref()
            .expect("mesh already removed")
            .tile;
        self.tiles[tile.0 as usize]
            .as_mut()
            .expect("mesh outlived its tile")
            .meshes
            .retain(|&m| m != id);
        self.drop_mesh(id);
        self.candidates_dirty = true;
        self.revision += 1;
    }

    fn drop_mesh(&mut self, id: MeshId) {
        let entry = self.meshes[id.0 as usize]
            .take()
            .expect("mesh already removed");
        self.free_meshes.push(id.0);
        self.nodes[entry.node as usize].meshes.retain(|&m| m != id);
        for chunk in entry.chunk_slots {
            self.chunk_slots.free(chunk);
        }
        self.vertex_arena.free(entry.vertices);
        self.index_arena.free(entry.indices);
        self.prop_arena.free(entry.props);
        self.prop_arena.free(entry.state);
        self.prop_arena.free(entry.symbol_states);
    }

    /// Writes the frame block, and the candidate lists if meshes or
    /// placements changed. Returns the number of candidates.
    pub fn prepare(&mut self, gpu: &mut dyn Gpu, frame: &FrameBlock) -> u32 {
        if self.candidates_dirty {
            self.candidates_dirty = false;
            self.rebuild_candidates(gpu);
        }

        write_frame(gpu, self.buffers.frame, frame);
        self.candidate_count
    }

    /// Lays out every node's candidates contiguously, in mesh, placement
    /// and chunk order (or by sort key): the order the cull pass keeps and
    /// the draws follow. An indexed node's runs go to its slice of the
    /// visible list, and its candidates get consecutive slots of the
    /// compacted index buffer, to be written by the next cull.
    fn rebuild_candidates(&mut self, gpu: &mut dyn Gpu) {
        let limit = self.capacity.candidates as usize;
        let mut rows: Vec<u32> = Vec::new();
        let mut runs: Vec<(u32, [u32; 2])> = Vec::new();
        let mut node_cull = Vec::with_capacity(self.nodes.len() * NODE_CULL_WORDS);
        let mut dropped = 0u32;
        let mut compacted = 0u32;
        let mut batches = 1u32;
        for node in &self.nodes {
            let base = (rows.len() / CANDIDATE_WORDS) as u32;
            let indexed = node.cull.draw == NodeDraw::Indexed;
            // (sort key, chunk slot, placement slot, index count, run)
            let mut candidates = Vec::new();
            let mut node_runs: Vec<[u32; 2]> = Vec::new();
            let mut vertex_words = 1u32;
            if node.cull.visible {
                for mesh in &node.meshes {
                    let mesh = self.meshes[mesh.0 as usize]
                        .as_ref()
                        .expect("node lists live meshes");
                    let tile = self.tiles[mesh.tile.0 as usize]
                        .as_ref()
                        .expect("meshes have tiles");
                    let Some(&first) = mesh.chunk_slots.first() else {
                        continue;
                    };
                    vertex_words = vertex_words.max(mesh.vertices.len() as u32);
                    for placement in &tile.placements {
                        if placement.row.target != node.cull.target {
                            continue;
                        }
                        let run = node_runs.len() as u32;
                        node_runs.push([first, placement.slot]);
                        for (&chunk, record) in mesh.chunk_slots.iter().zip(&mesh.chunks) {
                            candidates.push((
                                record.sort_key,
                                chunk,
                                placement.slot,
                                record.index_count,
                                run,
                            ));
                        }
                    }
                }
            }
            if candidates.iter().any(|c| c.0.is_some()) {
                candidates.sort_by(|a, b| a.0.unwrap_or(0.0).total_cmp(&b.0.unwrap_or(0.0)));
            }
            // An indexed node's index is `run << run_shift | vertex`; the
            // all-ones run is reserved for hidden indices.
            let run_shift = 32 - (vertex_words - 1).leading_zeros().min(31);
            // WebGL rejects indices from 2^31: Compat's fit in 31 bits.
            let index_bits: u32 = if self.cull.is_some() { 31 } else { 32 };
            let max_runs = (1u64 << index_bits.saturating_sub(run_shift)) - 1;
            let index_base = compacted;
            // Runs in order of first use by a kept candidate: no more than
            // the kept candidates, so they fit the node's visible slice.
            let mut renumbered = vec![u32::MAX; node_runs.len()];
            let mut used = 0u32;
            for (_, chunk, tile, index_count, run) in candidates {
                let fits = (rows.len() / CANDIDATE_WORDS) < limit
                    && (!indexed
                        || ((renumbered[run as usize] != u32::MAX || u64::from(used) < max_runs)
                            && u64::from(compacted) + u64::from(index_count)
                                <= u64::from(self.capacity.compacted_words)));
                if !fits {
                    dropped += 1;
                    continue;
                }
                if indexed {
                    let run = &mut renumbered[run as usize];
                    if *run == u32::MAX {
                        *run = used;
                        used += 1;
                    }
                    rows.extend([chunk, tile, compacted, *run, UNKNOWN]);
                    compacted += index_count;
                } else {
                    rows.extend([chunk, tile, 0, 0, 0]);
                }
            }
            let count = (rows.len() / CANDIDATE_WORDS) as u32 - base;
            if indexed {
                batches = batches.max(count.div_ceil(INDEXED_BATCH));
                let mut order = vec![[0u32; 2]; used as usize];
                for (run, &new) in node_runs.iter().zip(&renumbered) {
                    if new != u32::MAX {
                        order[new as usize] = *run;
                    }
                }
                runs.extend((base..).zip(order));
            }
            node_cull.extend([
                base,
                node.cull.margin_px.to_bits(),
                count,
                node.cull.draw.word()
                    | if node.cull.target == PlacementTarget::Map {
                        0
                    } else {
                        DRAPED
                    },
                node.cull.elevation.to_bits(),
                index_base,
                compacted - index_base,
                if indexed { run_shift } else { 0 },
            ]);
        }
        self.candidate_count = (rows.len() / CANDIDATE_WORDS) as u32;
        self.candidates_dropped = dropped;
        self.compacted_words = compacted;
        self.cull_batches = batches;
        if !rows.is_empty() {
            gpu.write_buffer(self.buffers.candidates, 0, bytemuck::cast_slice(&rows));
        }
        if !node_cull.is_empty() {
            gpu.write_buffer(self.buffers.node_cull, 0, bytemuck::cast_slice(&node_cull));
        }
        if let Some(cull) = &mut self.cull {
            cull.candidates = rows;
            cull.node_cull = node_cull;
        }
        // Each node's runs are contiguous in its slice.
        let mut at = 0;
        while at < runs.len() {
            let start = runs[at].0;
            let mut end = at + 1;
            while end < runs.len() && runs[end].0 == start + (end - at) as u32 {
                end += 1;
            }
            let words: Vec<u32> = runs[at..end].iter().flat_map(|(_, run)| *run).collect();
            gpu.write_buffer(
                self.buffers.visible,
                u64::from(start) * (VISIBLE_WORDS as u64 * 4),
                bytemuck::cast_slice(&words),
            );
            at = end;
        }
    }
}

#[cfg(test)]
mod row_layout {
    /// `types.wgsl` reads tile rows with orb-scene's stride.
    #[test]
    fn shaders_share_the_tile_row_stride() {
        let types = include_str!("../../orb-shaders/src/wgsl/types.wgsl");
        let declared = format!("const TILE_WORDS: u32 = {}u;", super::TILE_WORDS);
        assert!(
            types.contains(&declared),
            "types.wgsl must declare {declared}"
        );
    }
}
