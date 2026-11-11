//! Symbol meshes: one stream of a symbol layer's tile, its
//! tables back to back in the vertex arena behind a header, and the state
//! words placement writes, in the property arena.
//!
//! The layout matches `symbol.wgsl`. Chunks are runs of at most 64 quads;
//! their first index and count are quads, not indices.

use super::Table;
use orb_hal::Gpu;

use crate::scene::{Chunk, MeshEntry, MeshId, NO_STATE, OutOfSpace, Scene, TileId, Words};

/// Words of a symbol mesh's header, ahead of its tables.
pub const SYMBOL_HEADER_WORDS: usize = 20;

/// Header words: offsets are in words from the mesh's first vertex word.
pub mod symbol_header {
    pub const QUADS: usize = 0;
    pub const PLACED: usize = 1;
    pub const GLYPH_OFFSETS: usize = 2;
    pub const LINES: usize = 3;
    /// Per tile-local glyph id, its glyph atlas row.
    pub const GLYPHS: usize = 4;
    /// Per tile-local image index, its rectangle in the image pages
    /// (see `pack_image_rect`), or `u32::MAX` while it has none.
    pub const IMAGES: usize = 5;
    /// Per symbol instance, its feature's property-row slot.
    pub const INSTANCE_ROWS: usize = 6;
    /// Icons only: per placed icon, the packed size of the text it is
    /// fitted to (it moves with that text's variable anchor).
    pub const TEXT_SIZES: usize = 7;
    /// The quads in draw order (placement sorts some layers by viewport y).
    pub const ORDER: usize = 8;
    /// Text only: per quad, the colour its `format` section overrides
    /// `text-color` with, where its flags say so.
    pub const COLORS: usize = 9;
    /// First word of the state words in the property arena: per world
    /// copy, a block of them. A placement draws the block of its wrap
    /// modulo [`COPIES`].
    pub const STATES: usize = 10;
    /// How many blocks of state words there are: a power of two.
    pub const COPIES: usize = 11;
    /// The rest is the runtime's: sizes (see `symbol.wgsl`).
    pub const PARAMS: usize = 12;
}

/// An image's padded rectangle origin and page layer in one word: x and y
/// in 12 bits each, the layer in 8.
pub fn pack_image_rect(x: u32, y: u32, layer: u32) -> u32 {
    assert!(
        x < 4096 && y < 4096 && layer < 256,
        "image rect fits its packing"
    );
    x | y << 12 | layer << 24
}

/// Words per placed symbol in a mesh's state.
pub const SYMBOL_STATE_WORDS: u32 = 3;
/// Words per quad and per placed symbol in the tables.
pub const QUAD_WORDS: u32 = 12;
pub const PLACED_WORDS: u32 = 11;

/// A symbol stream ready to upload. Tables are word streams of
/// `orb-symbol`'s rows.
#[derive(Clone, Copy, Debug)]
pub struct SymbolMeshData<'a> {
    pub quads: Words<'a>,
    pub placed: Words<'a>,
    pub glyph_offsets: Words<'a>,
    pub lines: Words<'a>,
    pub glyphs: &'a [u32],
    pub images: &'a [u32],
    pub instance_rows: Words<'a>,
    pub text_sizes: &'a [u32],
    pub colors: Words<'a>,
    /// Paint property rows by feature slot.
    pub props: Words<'a>,
    /// Header words from [`symbol_header::PARAMS`] on.
    pub params: [u32; SYMBOL_HEADER_WORDS - symbol_header::PARAMS],
    pub chunks: &'a [Chunk],
}

impl Scene {
    /// Uploads a symbol stream of `tile` for render `node`, with state
    /// words for one world copy. They start hidden (zero opacity, not
    /// placed).
    pub(crate) fn insert_symbol_mesh(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: TileId,
        node: u32,
        data: &SymbolMeshData<'_>,
    ) -> Result<MeshId, OutOfSpace> {
        assert!(
            (node as usize) < self.nodes.len(),
            "unknown render node {node}"
        );
        assert!(self.tiles[tile.0 as usize].is_some(), "unknown tile");
        let placed_count = data.placed.len() / PLACED_WORDS;
        let order: Vec<u32> = (0..data.quads.len() / QUAD_WORDS).collect();
        let tables = [
            data.quads,
            data.placed,
            data.glyph_offsets,
            data.lines,
            Words::Core(data.glyphs),
            Words::Core(data.images),
            data.instance_rows,
            Words::Core(data.text_sizes),
            Words::Core(&order),
            data.colors,
        ];
        let words = SYMBOL_HEADER_WORDS as u32 + tables.iter().map(Words::len).sum::<u32>();
        if !self.chunk_slots.reserve(data.chunks.len() as u32) {
            return Err(OutOfSpace);
        }
        let vertices = self.vertex_arena.alloc(words).ok_or(OutOfSpace)?;
        let Some(props) = self.prop_arena.alloc(data.props.len()) else {
            self.vertex_arena.free(vertices);
            return Err(OutOfSpace);
        };
        let Some(states) = self.prop_arena.alloc(placed_count * SYMBOL_STATE_WORDS) else {
            self.vertex_arena.free(vertices);
            self.prop_arena.free(props);
            return Err(OutOfSpace);
        };
        let mut header = [0u32; SYMBOL_HEADER_WORDS];
        let mut at = SYMBOL_HEADER_WORDS as u32;
        for (i, table) in tables.iter().enumerate() {
            header[i] = at;
            self.tables
                .write(gpu, Table::Vertices, vertices.start + at, *table);
            at += table.len();
        }
        header[symbol_header::STATES] = states.start;
        header[symbol_header::COPIES] = 1;
        header[symbol_header::PARAMS..].copy_from_slice(&data.params);
        self.tables
            .write(gpu, Table::Vertices, vertices.start, Words::Core(&header));
        self.tables
            .write(gpu, Table::Props, props.start, data.props);
        let zeros = vec![0; states.len()];
        self.tables
            .write_words(gpu, Table::Props, states.start, &zeros);

        let mut chunk_ids = Vec::with_capacity(data.chunks.len());
        for chunk in data.chunks {
            let id = self.chunk_slots.alloc().expect("checked above");
            let pack = |p: [i16; 2]| u32::from(p[0] as u16) | u32::from(p[1] as u16) << 16;
            let row: [u32; crate::scene::CHUNK_WORDS] = [
                chunk.first_index,
                chunk.index_count,
                vertices.start,
                0,
                props.start,
                pack(chunk.bounds.min),
                pack(chunk.bounds.max),
                NO_STATE,
                0,
            ];
            self.tables.write_words(
                gpu,
                Table::Chunks,
                id * crate::scene::CHUNK_WORDS as u32,
                &row,
            );
            chunk_ids.push(id);
        }
        let entry = MeshEntry {
            tile,
            node,
            chunk_slots: chunk_ids,
            chunks: data.chunks.to_vec(),
            vertices,
            indices: 0..0,
            props,
            // The symbol shader addresses its rows itself.
            row_words: 0,
            state: 0..0,
            symbol_states: states,
            symbol_copies: 1,
            symbol_order: [header[symbol_header::ORDER], order.len() as u32],
        };
        Ok(self.add_mesh(entry))
    }

    /// Sets the order a symbol mesh's quads draw in: every quad once.
    pub(crate) fn write_symbol_order(&mut self, gpu: &mut dyn Gpu, mesh: MeshId, order: &[u32]) {
        let entry = self.meshes[mesh.0 as usize].as_ref().expect("live mesh");
        let [offset, quads] = entry.symbol_order;
        assert_eq!(order.len() as u32, quads, "one entry per quad");
        self.tables.write(
            gpu,
            Table::Vertices,
            entry.vertices.start + offset,
            Words::Core(order),
        );
    }

    /// How many world copies a symbol mesh keeps state words for.
    pub(crate) fn symbol_copies(&self, mesh: MeshId) -> u32 {
        self.meshes[mesh.0 as usize]
            .as_ref()
            .expect("live mesh")
            .symbol_copies
    }

    /// Keeps state words for `copies` world copies (a power of two), all
    /// hidden until written.
    pub(crate) fn set_symbol_copies(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        copies: u32,
    ) -> Result<(), OutOfSpace> {
        assert!(copies.is_power_of_two(), "copies are a power of two");
        let entry = self.meshes[mesh.0 as usize].as_ref().expect("live mesh");
        let block = entry.symbol_states.len() as u32 / entry.symbol_copies;
        let states = self.prop_arena.alloc(block * copies).ok_or(OutOfSpace)?;
        let zeros = vec![0; states.len()];
        self.tables
            .write_words(gpu, Table::Props, states.start, &zeros);
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        let old = std::mem::replace(&mut entry.symbol_states, states.clone());
        entry.symbol_copies = copies;
        let header = entry.vertices.start + symbol_header::STATES as u32;
        self.prop_arena.free(old);
        self.tables.write(
            gpu,
            Table::Vertices,
            header,
            Words::Core(&[states.start, copies]),
        );
        Ok(())
    }

    /// Writes a symbol mesh's state words for world copy `wrap`:
    /// [`SYMBOL_STATE_WORDS`] per placed symbol.
    pub(crate) fn write_symbol_states(
        &mut self,
        gpu: &mut dyn Gpu,
        mesh: MeshId,
        wrap: i32,
        words: &[u32],
    ) {
        let entry = self.meshes[mesh.0 as usize].as_ref().expect("live mesh");
        let block = entry.symbol_states.len() as u32 / entry.symbol_copies;
        assert_eq!(words.len() as u32, block, "one state per placed symbol");
        let copy = wrap.rem_euclid(entry.symbol_copies as i32) as u32;
        let at = entry.symbol_states.start + copy * block;
        self.tables.write(gpu, Table::Props, at, Words::Core(words));
    }
}
