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

fn write_chunk_word(gpu: &mut dyn Gpu, chunks: BufferId, chunk: u32, word: u64, value: u32) {
    gpu.write_buffer(
        chunks,
        (u64::from(chunk) * CHUNK_WORDS as u64 + word) * 4,
        &value.to_le_bytes(),
    );
}

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
    /// Top logical row of the apparent sky horizon.
    pub sky_horizon: f32,
    /// How far symbol fades have run since the last placement commit (GL
    /// JS `symbolFadeChange`).
    pub symbol_fade_change: f32,
    /// Camera pitch, radians.
    pub pitch: f32,
}

const FRAME_WORDS: usize = 32;

/// One placement of a tile's data: its transform into camera-relative world
/// space, `origin + (x, -y) · scale`, and where it may draw.
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

const TILE_WORDS: usize = 16;
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

impl Words<'_> {
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
}

/// How a render node draws its visible chunks, which the cull pass
/// prepares (plan §8.3, decision 0002).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeDraw {
    /// Vertex pulling: one instance of 768 vertices per visible chunk, from
    /// the node's visible list.
    Chunks,
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
            Self::Chunks => 0,
            Self::Points => 1,
            Self::Indexed => 2,
        }
    }
}

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
            candidates: buffer("candidates", words(c.candidates, CANDIDATE_WORDS), storage),
            node_cull: buffer("node cull", words(c.nodes, NODE_CULL_WORDS), storage),
            visible: buffer("visible", words(c.candidates, VISIBLE_WORDS), storage),
            args: buffer(
                "draw args",
                words(c.nodes, ARGS_WORDS),
                BufferUsages::STORAGE
                    | BufferUsages::INDIRECT
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            ),
            vertices: buffer("vertices", u64::from(c.vertex_words), storage),
            indices: buffer("indices", u64::from(c.index_words), storage),
            compacted: buffer(
                "compacted indices",
                u64::from(c.compacted_words),
                BufferUsages::STORAGE | BufferUsages::INDEX,
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
            revision: 0,
            placements_written: 0,
        }
    }

    pub fn buffers(&self) -> &SceneBuffers {
        &self.buffers
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
            for (i, clip) in row.clips.iter().enumerate() {
                words[4 + i * 2] = pack(clip.min);
                words[5 + i * 2] = pack(clip.max);
            }
            gpu.write_buffer(
                self.buffers.tiles,
                u64::from(slot) * (TILE_WORDS as u64 * 4),
                bytemuck::cast_slice(&words),
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
            gpu.write_buffer(
                self.buffers.chunks,
                u64::from(id) * (CHUNK_WORDS as u64 * 4),
                bytemuck::cast_slice(&row),
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
        words[26] = frame.sky_horizon.to_bits();
        words[27] = frame.symbol_fade_change.to_bits();
        words[28] = frame.pitch.to_bits();
        words[30] = frame.zoom.to_bits() as u32;
        words[31] = (frame.zoom.to_bits() >> 32) as u32;
        gpu.write_buffer(self.buffers.frame, 0, bytemuck::cast_slice(&words));
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
            let max_runs = (1u64 << (32 - run_shift)) - 1;
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
                node.cull.draw.word(),
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
