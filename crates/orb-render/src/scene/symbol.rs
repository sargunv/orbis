//! Symbol meshes: one stream of a symbol layer's tile, its
//! tables back to back in the vertex arena behind a header, and the state
//! words placement writes, in the property arena.
//!
//! The layout matches `symbol.wgsl`. Chunks are runs of at most 64 quads;
//! their first index and count are quads, not indices.
//!
//! A chunk's row also says in which world copies placement shows any of
//! its quads, so that the cull drops the rest: in a dense view most labels
//! are hidden, and each of their quads would otherwise run its vertices
//! only to read that it is.

use std::ops::Range;

use super::Table;
use orb_hal::Gpu;

use crate::scene::{
    CHUNK_WORDS, Chunk, MeshEntry, MeshId, NO_STATE, OutOfSpace, Scene, TileId, Words,
    write_chunk_word,
};

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

/// The bits of a placed symbol's first state word that say whether it
/// draws (`symbol.wgsl`'s, as orb-symbol's `state` writes them).
pub mod symbol_state {
    /// Opacity at the commit, 0 to 127.
    pub const OPACITY: u32 = 0x7f;
    /// Fading in rather than out.
    pub const PLACED: u32 = 0x80;
    /// Drawn by nothing.
    pub const HIDDEN: u32 = 0x100;
}

/// Whether a placed symbol whose first state word is `word` may draw: not
/// hidden, and shown or fading.
fn shows(word: u32) -> bool {
    use symbol_state::*;
    word & HIDDEN == 0 && word & (OPACITY | PLACED) != 0
}

/// The chunk row word that holds, for a symbol chunk, the world copies it
/// shows in: bit `wrap & 31` (`types.wgsl`'s `Chunk`).
const SHOWN_WORD: u64 = orb_shaders::layout::chunk::FIRST_VERTEX as u64;

/// Words per quad and per placed symbol in the tables.
pub const QUAD_WORDS: u32 = 12;
pub const PLACED_WORDS: u32 = 11;

/// A symbol mesh's placement state: its state words, and in which world
/// copies its chunks show anything.
#[derive(Debug)]
pub(super) struct SymbolEntry {
    /// Per world copy, a block of state words, in the property arena.
    pub(super) states: Range<u32>,
    /// How many blocks: a power of two.
    copies: u32,
    /// The draw-order table's first word, from the mesh's first vertex
    /// word.
    order_at: u32,
    /// Per quad, its placed symbol (`u32::MAX` for none).
    quad_placed: Vec<u32>,
    /// The quads in draw order, as last written: empty for quad order.
    order: Vec<u32>,
    /// Per block, a bit per placed symbol: whether it [`shows`].
    shown: Vec<u64>,
    /// Each chunk's row word of copies it shows in, as last written.
    masks: Vec<u32>,
}

impl SymbolEntry {
    /// Words of [`shown`](Self::shown) per block.
    fn block_words(&self) -> usize {
        let placed = self.states.len() as u32 / self.copies / SYMBOL_STATE_WORDS;
        (placed as usize).div_ceil(64)
    }

    /// Each chunk's copies, by `wrap & 31` as the cull tests them: those
    /// whose block shows any of the chunk's quads. Past 32 blocks a bit
    /// stands for several, and every chunk keeps them all.
    fn masks(&self, chunks: &[Chunk]) -> Vec<u32> {
        let words = self.block_words();
        let shown = |block: u32, quad: u32| {
            let p = self.quad_placed.get(quad as usize).copied();
            p.filter(|&p| p != u32::MAX).is_some_and(|p| {
                let at = block as usize * words + p as usize / 64;
                self.shown[at] >> (p % 64) & 1 != 0
            })
        };
        let quads = |c: &Chunk| {
            (c.first_index..c.first_index + c.index_count)
                .map(|at| self.order.get(at as usize).copied().unwrap_or(at))
        };
        (chunks.iter())
            .map(|c| {
                if self.copies > 32 {
                    return u32::MAX;
                }
                let blocks = (0..self.copies)
                    .filter(|&block| quads(c).any(|quad| shown(block, quad)))
                    .fold(0u32, |bits, block| bits | 1 << block);
                (0..32)
                    .filter(|&bit| blocks >> (bit & (self.copies - 1)) & 1 != 0)
                    .fold(0, |mask, bit| mask | 1 << bit)
            })
            .collect()
    }
}

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
    /// Per placed symbol, its first quad and how many it has.
    pub placed_quads: &'a [[u32; 2]],
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
        debug_assert_eq!(data.placed_quads.len() as u32, placed_count);
        let quads = data.quads.len() / QUAD_WORDS;
        let order: Vec<u32> = (0..quads).collect();
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
            // Shown in no copy until placement says otherwise.
            let row: [u32; CHUNK_WORDS] = [
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
            self.tables
                .write_words(gpu, Table::Chunks, id * CHUNK_WORDS as u32, &row);
            chunk_ids.push(id);
        }
        let mut quad_placed = vec![u32::MAX; quads as usize];
        for (p, &[first, count]) in data.placed_quads.iter().enumerate() {
            let end = first.saturating_add(count).min(quads);
            quad_placed[first.min(end) as usize..end as usize].fill(p as u32);
        }
        let symbol = SymbolEntry {
            states,
            copies: 1,
            order_at: header[symbol_header::ORDER],
            quad_placed,
            order: Vec::new(),
            shown: vec![0; (placed_count as usize).div_ceil(64)],
            masks: vec![0; data.chunks.len()],
        };
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
            symbol: Some(Box::new(symbol)),
        };
        Ok(self.add_mesh(entry))
    }

    fn symbol(&self, mesh: MeshId) -> &SymbolEntry {
        let entry = self.meshes[mesh.0 as usize].as_ref().expect("live mesh");
        entry.symbol.as_deref().expect("a symbol mesh")
    }

    /// Writes the chunk rows of a symbol mesh whose copies they show in
    /// changed.
    fn write_symbol_masks(&mut self, gpu: &mut dyn Gpu, mesh: MeshId) {
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        let symbol = entry.symbol.as_deref_mut().expect("a symbol mesh");
        let masks = symbol.masks(&entry.chunks);
        for ((&slot, old), new) in (entry.chunk_slots.iter()).zip(&mut symbol.masks).zip(masks) {
            if *old != new {
                *old = new;
                write_chunk_word(gpu, &mut self.tables, slot, SHOWN_WORD, new);
            }
        }
    }

    /// Sets the order a symbol mesh's quads draw in: every quad once.
    pub(crate) fn write_symbol_order(&mut self, gpu: &mut dyn Gpu, mesh: MeshId, order: &[u32]) {
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        let symbol = entry.symbol.as_deref_mut().expect("a symbol mesh");
        assert_eq!(order.len(), symbol.quad_placed.len(), "one entry per quad");
        if symbol.order == order {
            return;
        }
        symbol.order = order.to_vec();
        let at = entry.vertices.start + symbol.order_at;
        self.tables
            .write(gpu, Table::Vertices, at, Words::Core(order));
        self.write_symbol_masks(gpu, mesh);
    }

    /// How many world copies a symbol mesh keeps state words for.
    pub(crate) fn symbol_copies(&self, mesh: MeshId) -> u32 {
        self.symbol(mesh).copies
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
        let symbol = self.symbol(mesh);
        let block = symbol.states.len() as u32 / symbol.copies;
        let states = self.prop_arena.alloc(block * copies).ok_or(OutOfSpace)?;
        let zeros = vec![0; states.len()];
        self.tables
            .write_words(gpu, Table::Props, states.start, &zeros);
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        let symbol = entry.symbol.as_deref_mut().expect("a symbol mesh");
        let old = std::mem::replace(&mut symbol.states, states.clone());
        symbol.copies = copies;
        symbol.shown = vec![0; copies as usize * symbol.block_words()];
        let header = entry.vertices.start + symbol_header::STATES as u32;
        self.prop_arena.free(old);
        self.tables.write(
            gpu,
            Table::Vertices,
            header,
            Words::Core(&[states.start, copies]),
        );
        self.write_symbol_masks(gpu, mesh);
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
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        let symbol = entry.symbol.as_deref_mut().expect("a symbol mesh");
        let block = symbol.states.len() as u32 / symbol.copies;
        assert_eq!(words.len() as u32, block, "one state per placed symbol");
        let copy = wrap.rem_euclid(symbol.copies as i32) as u32;
        let at = symbol.states.start + copy * block;
        let bits = symbol.block_words();
        let shown = &mut symbol.shown[copy as usize * bits..][..bits];
        shown.fill(0);
        for (p, state) in (words
            .as_chunks::<{ SYMBOL_STATE_WORDS as usize }>()
            .0
            .iter())
        .enumerate()
        {
            shown[p / 64] |= u64::from(shows(state[0])) << (p % 64);
        }
        self.tables.write(gpu, Table::Props, at, Words::Core(words));
        self.write_symbol_masks(gpu, mesh);
    }
}
