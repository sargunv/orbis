//! Symbol meshes: one stream of a symbol layer's tile, its
//! tables back to back in the vertex arena behind a header, and the state
//! words placement writes, in the property arena.
//!
//! The layout matches `symbol.wgsl`. Chunks are runs of at most 64 quads;
//! their first index and count are quads, not indices.
//!
//! A chunk's row also says in which world copies placement shows any of
//! its quads, so that the cull drops the rest: in a dense view most labels
//! are hidden, and each of their quads would otherwise be placed and run
//! its vertices only to read that it is. Labels fading out count as shown
//! until their fade has run, when they leave too.

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
    /// The rest is the runtime's: sizes (see `symbol.wgsl`), then flags.
    pub const PARAMS: usize = 12;
    /// The flags word: [`LINEAR_IMAGES`], [`TEXT`].
    pub const FLAGS: usize = 18;
    /// Its images need linear filtering.
    pub const LINEAR_IMAGES: u32 = 1;
    /// It is a layer's text, not its icons: the half of its layer's draw
    /// it reads its paint from.
    pub const TEXT: u32 = 2;
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

/// Whether a placed symbol whose first state word is `word` still draws
/// once the fades since the last placement have run: not hidden, and
/// placed (one fading out has then faded).
fn stays(word: u32) -> bool {
    use symbol_state::*;
    word & HIDDEN == 0 && word & PLACED != 0
}

/// Per placed symbol of state `words`, a bit of `shown`, whether it
/// [`shows`], and of `stayed`, whether it [`stays`].
fn shown_bits(words: &[u32], shown: &mut [u64], stayed: &mut [u64]) {
    let states = words.as_chunks::<{ SYMBOL_STATE_WORDS as usize }>().0;
    for ((shown, stayed), states) in shown.iter_mut().zip(stayed).zip(states.chunks(64)) {
        (*shown, *stayed) = (states.iter().enumerate()).fold((0, 0), |(a, b), (p, state)| {
            let [shows, stays] = [shows(state[0]), stays(state[0])].map(u64::from);
            (a | shows << p, b | stays << p)
        });
    }
}

/// The chunk row word that holds, for a symbol chunk, the world copies it
/// shows in: bit `wrap & 31` (`types.wgsl`'s `Chunk`).
const SHOWN_WORD: u64 = orb_shaders::layout::chunk::FIRST_VERTEX as u64;

/// Words per quad and per placed symbol in the tables.
pub const QUAD_WORDS: u32 = 12;
pub const PLACED_WORDS: u32 = 11;

/// Texels a row of the symbol placement target (`symbol.wgsl`'s
/// `PLACED_WIDTH`): within GLSL ES 3.00's smallest texture side.
pub(crate) const PLACED_WIDTH: u32 = 2048;

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
    /// The placed symbols each chunk draws quads of, as ranges: chunk
    /// `k`'s are `runs[run_ends[k - 1]..run_ends[k]]`. A chunk's quads run
    /// through few placed symbols in turn, so testing a chunk on each
    /// placement reads a few words of [`shown`](Self::shown), not a word
    /// per quad.
    runs: Vec<[u32; 2]>,
    run_ends: Vec<u32>,
    /// Per block, a bit per placed symbol: whether it [`shows`], and
    /// whether it [`stays`].
    shown: Vec<u64>,
    stayed: Vec<u64>,
    /// Each chunk's row word of copies it shows in, as last written.
    masks: Vec<u32>,
}

impl SymbolEntry {
    fn new(
        states: Range<u32>,
        order_at: u32,
        placed_quads: &[[u32; 2]],
        quads: u32,
        chunks: &[Chunk],
    ) -> Self {
        let mut quad_placed = vec![u32::MAX; quads as usize];
        for (p, &[first, count]) in placed_quads.iter().enumerate() {
            let end = first.saturating_add(count).min(quads);
            quad_placed[first.min(end) as usize..end as usize].fill(p as u32);
        }
        let mut entry = Self {
            states,
            copies: 1,
            order_at,
            quad_placed,
            order: Vec::new(),
            runs: Vec::new(),
            run_ends: Vec::new(),
            shown: vec![0; placed_quads.len().div_ceil(64)],
            stayed: vec![0; placed_quads.len().div_ceil(64)],
            masks: vec![0; chunks.len()],
        };
        entry.lay_runs(chunks);
        entry
    }

    /// Words of [`shown`](Self::shown) per block.
    fn block_words(&self) -> usize {
        let placed = self.states.len() as u32 / self.copies / SYMBOL_STATE_WORDS;
        (placed as usize).div_ceil(64)
    }

    /// Sets [`runs`](Self::runs) from the quads' draw order.
    fn lay_runs(&mut self, chunks: &[Chunk]) {
        self.runs.clear();
        self.run_ends.clear();
        for c in chunks {
            let first = self.runs.len();
            for at in c.first_index..c.first_index + c.index_count {
                let quad = self.order.get(at as usize).copied().unwrap_or(at);
                let p = self.quad_placed.get(quad as usize).copied();
                let Some(p) = p.filter(|&p| p != u32::MAX) else {
                    continue;
                };
                match self.runs[first..].last_mut() {
                    Some(run) if (run[0]..=run[1]).contains(&p) => run[1] = run[1].max(p + 1),
                    _ => self.runs.push([p, p + 1]),
                }
            }
            self.run_ends.push(self.runs.len() as u32);
        }
    }

    /// Sets each chunk's copies, by `wrap & 31` as the cull tests them:
    /// those whose block shows any of its quads (that [`stays`] once the
    /// fades have run, `faded`), testing block `block` again and keeping
    /// the others' bits (every block for `None`), and calls `changed` with
    /// each chunk whose mask changed. Past 32 blocks a bit stands for
    /// several, and every chunk keeps them all.
    fn update_masks(
        &mut self,
        block: Option<u32>,
        faded: bool,
        mut changed: impl FnMut(usize, u32),
    ) {
        let (copies, words) = (self.copies, self.block_words());
        let bits = if faded { &self.stayed } else { &self.shown };
        // The bits standing for block `b` are this shifted by `b`: `b`,
        // `b + copies`, and so on.
        let first = (0..32)
            .step_by(copies as usize)
            .fold(0u32, |m, b| m | 1 << b);
        let blocks = block.map_or(0..copies, |b| b..b + 1);
        let mut from = 0;
        for (k, (old, &end)) in self.masks.iter_mut().zip(&self.run_ends).enumerate() {
            let runs = &self.runs[from..end as usize];
            from = end as usize;
            let mask = if copies > 32 {
                u32::MAX
            } else {
                let kept = block.map_or(0, |b| *old & !(first << b));
                (blocks.clone())
                    .filter(|&b| {
                        let shown = &bits[b as usize * words..][..words];
                        runs.iter().any(|&run| any_bit(shown, run))
                    })
                    .fold(kept, |mask, b| mask | first << b)
            };
            if *old != mask {
                *old = mask;
                changed(k, mask);
            }
        }
    }
}

/// Whether any bit of `bits` from `start` to `end` is set.
fn any_bit(bits: &[u64], [start, end]: [u32; 2]) -> bool {
    let mut at = start;
    while at < end {
        let bit = at % 64;
        let n = (64 - bit).min(end - at);
        if bits[at as usize / 64] >> bit & (u64::MAX >> (64 - n)) != 0 {
            return true;
        }
        at += n;
    }
    false
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

/// A symbol stream's tables, in its header's order, the quads' draw
/// order last but one.
fn symbol_tables<'a>(data: &SymbolMeshData<'a>, order: &'a [u32]) -> [Words<'a>; 10] {
    [
        data.quads,
        data.placed,
        data.glyph_offsets,
        data.lines,
        Words::Core(data.glyphs),
        Words::Core(data.images),
        data.instance_rows,
        Words::Core(data.text_sizes),
        Words::Core(order),
        data.colors,
    ]
}

/// The vertex words a symbol stream takes: its header and tables.
fn symbol_words(data: &SymbolMeshData<'_>) -> u32 {
    let quads = data.quads.len() / QUAD_WORDS;
    let tables = symbol_tables(data, &[]);
    SYMBOL_HEADER_WORDS as u32 + quads + tables.iter().map(Words::len).sum::<u32>()
}

impl Scene {
    /// Uploads symbol streams of `tile`, each for its render node, with
    /// state words for one world copy, all in one block: a layer's icons
    /// and text draw in one list, icons first, which another block's
    /// draw would come after. They start hidden (zero opacity, not
    /// placed). All of them or none.
    pub(crate) fn insert_symbol_meshes(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: TileId,
        meshes: &[(u32, &SymbolMeshData<'_>)],
    ) -> Result<Vec<MeshId>, OutOfSpace> {
        assert!(self.tiles[tile.0 as usize].is_some(), "unknown tile");
        let words: Vec<u32> = meshes.iter().map(|(_, data)| symbol_words(data)).collect();
        let chunks: u32 = meshes
            .iter()
            .map(|(_, data)| data.chunks.len() as u32)
            .sum();
        self.reserve_chunks(gpu, chunks)?;
        let total = words.iter().sum();
        let (block, [mut all, _, _]) = self.alloc_words(gpu, [total, 0, 0], None)?;
        let mut ids = Vec::with_capacity(meshes.len());
        for (&(node, data), words) in meshes.iter().zip(words) {
            let vertices = all.start..all.start + words;
            all.start += words;
            match self.insert_symbol_mesh_in(gpu, tile, (node, block), vertices.clone(), data) {
                Ok(id) => ids.push(id),
                Err(e) => {
                    self.free_words(block, 0, vertices.start..all.end);
                    for id in ids {
                        self.remove_mesh(id);
                    }
                    return Err(e);
                }
            }
        }
        Ok(ids)
    }

    /// A symbol stream of `tile` for render `node`, its header and tables
    /// written at `vertices` of block `block`'s vertex pool. On `Err` the
    /// words are still the caller's.
    fn insert_symbol_mesh_in(
        &mut self,
        gpu: &mut dyn Gpu,
        tile: TileId,
        (node, block): (u32, u32),
        vertices: Range<u32>,
        data: &SymbolMeshData<'_>,
    ) -> Result<MeshId, OutOfSpace> {
        assert!((node as usize) < self.rows, "unknown render node {node}");
        let placed_count = data.placed.len() / PLACED_WORDS;
        debug_assert_eq!(data.placed_quads.len() as u32, placed_count);
        let quads = data.quads.len() / QUAD_WORDS;
        let order: Vec<u32> = (0..quads).collect();
        let tables = symbol_tables(data, &order);
        let props = self.alloc_props(gpu, data.props.len())?;
        let states = match self.alloc_props(gpu, placed_count * SYMBOL_STATE_WORDS) {
            Ok(states) => states,
            Err(e) => {
                self.prop_arena.free(props);
                return Err(e);
            }
        };
        let mut header = [0u32; SYMBOL_HEADER_WORDS];
        let mut at = SYMBOL_HEADER_WORDS as u32;
        for (i, table) in tables.iter().enumerate() {
            header[i] = at;
            self.write_pool(gpu, block, 0, vertices.start + at, *table);
            at += table.len();
        }
        header[symbol_header::STATES] = states.start;
        header[symbol_header::COPIES] = 1;
        header[symbol_header::PARAMS..].copy_from_slice(&data.params);
        self.write_pool(gpu, block, 0, vertices.start, Words::Core(&header));
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
        let symbol = SymbolEntry::new(
            states,
            header[symbol_header::ORDER],
            data.placed_quads,
            quads,
            data.chunks,
        );
        let entry = MeshEntry {
            tile,
            node,
            block,
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
    /// changed: in block `block`'s copies, or (`None`) in any.
    fn write_symbol_masks(&mut self, gpu: &mut dyn Gpu, mesh: MeshId, block: Option<u32>) {
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        let symbol = entry.symbol.as_deref_mut().expect("a symbol mesh");
        let tables = &mut self.tables;
        symbol.update_masks(block, self.labels_faded, |k, mask| {
            write_chunk_word(gpu, tables, entry.chunk_slots[k], SHOWN_WORD, mask);
        });
    }

    /// Follows the frame's label fades (`symbol_fade_change`): once
    /// they have run, labels fading out show nothing, and their chunks
    /// leave the cull at once rather than at the next placement; a
    /// placement starts fades again.
    pub(super) fn follow_label_fades(&mut self, gpu: &mut dyn Gpu, fade_change: f32) {
        let faded = fade_change >= 1.0;
        if faded == self.labels_faded {
            return;
        }
        self.labels_faded = faded;
        let meshes = (self.meshes.iter().enumerate())
            .filter(|(_, m)| m.as_ref().is_some_and(|m| m.symbol.is_some()))
            .map(|(i, _)| MeshId(i as u32));
        for mesh in meshes.collect::<Vec<_>>() {
            self.write_symbol_masks(gpu, mesh, None);
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
        symbol.lay_runs(&entry.chunks);
        let (block, at) = (entry.block, entry.vertices.start + symbol.order_at);
        self.write_pool(gpu, block, 0, at, Words::Core(order));
        self.write_symbol_masks(gpu, mesh, None);
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
        let per_copy = symbol.states.len() as u32 / symbol.copies;
        let states = self.alloc_props(gpu, per_copy * copies)?;
        let zeros = vec![0; states.len()];
        self.tables
            .write_words(gpu, Table::Props, states.start, &zeros);
        let entry = self.meshes[mesh.0 as usize].as_mut().expect("live mesh");
        let symbol = entry.symbol.as_deref_mut().expect("a symbol mesh");
        let old = std::mem::replace(&mut symbol.states, states.clone());
        symbol.copies = copies;
        symbol.shown = vec![0; copies as usize * symbol.block_words()];
        symbol.stayed = symbol.shown.clone();
        let (block, header) = (
            entry.block,
            entry.vertices.start + symbol_header::STATES as u32,
        );
        self.prop_arena.free(old);
        self.write_pool(gpu, block, 0, header, Words::Core(&[states.start, copies]));
        self.write_symbol_masks(gpu, mesh, None);
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
        let own = copy as usize * bits..(copy as usize + 1) * bits;
        shown_bits(
            words,
            &mut symbol.shown[own.clone()],
            &mut symbol.stayed[own],
        );
        self.tables.write(gpu, Table::Props, at, Words::Core(words));
        self.write_symbol_masks(gpu, mesh, Some(copy));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_layout::{Owner, TileBox};

    /// The masks against a read of every quad, over placed symbols of one
    /// to three quads, chunks across them, a reversed draw order, one to
    /// 64 copies, states written copy by copy, and fades running or run.
    #[test]
    fn chunk_masks_match_every_quad() {
        let placed: Vec<[u32; 2]> = (0..150)
            .scan(0, |q, p| {
                let n = p % 3 + 1;
                *q += n;
                Some([*q - n, n])
            })
            .collect();
        let quads = placed.last().map_or(0, |&[f, n]| f + n);
        let chunks: Vec<Chunk> = (0..quads.div_ceil(64))
            .map(|k| Chunk {
                first_index: k * 64,
                index_count: (quads - k * 64).min(64),
                first_vertex: 0,
                bounds: TileBox {
                    min: [0; 2],
                    max: [0; 2],
                },
                sort_key: None,
                owner: Owner::Every,
            })
            .collect();
        for (copies, faded) in [1, 2, 8, 64]
            .into_iter()
            .flat_map(|c| [(c, false), (c, true)])
        {
            for reversed in [false, true] {
                let words = placed.len() as u32 * SYMBOL_STATE_WORDS;
                let mut e = SymbolEntry::new(0..words, 0, &placed, quads, &chunks);
                e.states = 0..words * copies;
                e.copies = copies;
                e.shown = vec![0; copies as usize * e.block_words()];
                e.stayed = e.shown.clone();
                if reversed {
                    e.order = (0..quads).rev().collect();
                    e.lay_runs(&chunks);
                }
                let mut states = vec![vec![0u32; placed.len()]; copies as usize];
                for (round, copy) in (0..copies).chain(0..copies).enumerate() {
                    // A few placed symbols shown, fading or hidden.
                    for (p, s) in states[copy as usize].iter_mut().enumerate() {
                        *s = match (p as u32 * 7 + copy * 13 + round as u32 * 5) % 41 {
                            0 => symbol_state::PLACED,
                            1 => 20,
                            2 => symbol_state::HIDDEN | 127,
                            _ => 0,
                        };
                    }
                    let words: Vec<u32> = (states[copy as usize].iter())
                        .flat_map(|&s| [s, 0, 0])
                        .collect();
                    let bits = e.block_words();
                    let own = copy as usize * bits..(copy as usize + 1) * bits;
                    shown_bits(&words, &mut e.shown[own.clone()], &mut e.stayed[own]);
                    e.update_masks(Some(copy), faded, |_, _| {});
                    let kept = e.masks.clone();
                    e.update_masks(None, faded, |k, _| panic!("chunk {k} changed"));
                    assert_eq!(e.masks, kept);
                    for (k, c) in chunks.iter().enumerate() {
                        let want = (0..32u32).fold(0, |m, bit| {
                            let block = (bit & (copies - 1)) as usize;
                            let shows_any = (c.first_index..c.first_index + c.index_count)
                                .map(|at| e.order.get(at as usize).copied().unwrap_or(at))
                                .map(|q| states[block][e.quad_placed[q as usize] as usize])
                                .any(|s| if faded { stays(s) } else { shows(s) });
                            m | u32::from(shows_any || copies > 32) << bit
                        });
                        assert_eq!(
                            e.masks[k], want,
                            "chunk {k}, {copies} copies, reversed {reversed}, faded {faded}"
                        );
                    }
                }
            }
        }
    }
}
