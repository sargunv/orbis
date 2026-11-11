//! The CPU-driven frame, for devices without compute or indirect draws
//! (WebGL 2, #140): the GPU driver's cull (`cull.wgsl`) on the CPU, over
//! copies of the tables it reads, and direct draws.
//!
//! Each frame whose camera or tables changed, it decides every candidate as
//! the GPU's cull does: a bounds test of each run of one mesh's chunks in
//! one placement, then [`candidate_visible`], `cull.wgsl`'s function
//! written again in Rust. It writes what the GPU's cull writes, word for
//! word: listed nodes' visible lists, indexed nodes' compacted slots (the
//! chunk's indices tagged with its run when it enters the view, hidden ones
//! when it leaves) and each node's draw arguments, which its direct draws
//! take. Slots a layout moved hold what they held (this driver moves them
//! in its own copy of the compacted words), and a bit per compacted word
//! says which of the two, as the GPU's cull reads it from the slot itself.
//! The draws read one of [`COPIES`] index buffers in turn, each written in
//! place before the frames that draw it. Bundles record those draws, so the
//! renderer records them again when a node's arguments or the copy they
//! read change. A test holds all of it to the GPU's cull over the same
//! scene and frames (`Renderer::check_driver`).
//!
//! A commit that changes this file says why in its body: it is the one
//! piece of logic the two drivers have twice.

use std::ops::Range;

use orb_hal::{BufferDesc, BufferId, BufferUsages, Gpu, Pass, PipelineId, RenderCommand};
use orb_shaders::layout;

use crate::scene::{
    ARGS_WORDS, DRAPED, HIDDEN, MOVED, Move, NO_CHUNK, NodeDraw, SURFACE, Scene, Table, UNCLIPPED,
};

use super::Cull;
use super::gpu::{HIDDEN_INDEX, Twin};

/// WebGL rejects element indices from 2^31.
pub(crate) const INDEX_BITS: u32 = 31;

/// The draw arguments: written for diagnostics and tests to read back.
pub(crate) const ARGS_USAGE: BufferUsages = BufferUsages::COPY_DST.union(BufferUsages::COPY_SRC);

/// The scene's compacted indices, which hold nothing on this driver
/// (`Kind::scene_compacted_words`): its copies are its own.
pub(crate) const COMPACTED_USAGE: BufferUsages = BufferUsages::INDEX
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

/// Copies of the compacted indices that draws read in turn.
///
/// WebGL 2 always restarts primitives at the fixed index, and Metal's
/// triangle lists don't, so ANGLE's Metal back end splits each indexed
/// draw at the restart indices of its index buffer. After any change to
/// the buffer, the next indexed draw reads the whole buffer on the CPU to
/// find them, and if the GPU wrote the buffer (a copy into it), it first
/// waits for that copy and everything queued before it: 100-200 ms where
/// a frame's tiles arrive with many uploads (#140). So nothing writes a
/// copy on the GPU: a frame that changes slots writes the words every
/// copy lacks into the one drawn longest ago, in place (`bufferSubData`,
/// [`BufferUsages::IDLE_WRITES`]), when the GPU is done with it, and draws
/// from it: a frame lies between a copy's last draw and its next write.
/// A third copy measured no better on the Globe entry at 2x in Chromium
/// (#140), and a copy is the scene's compacted capacity long.
pub(crate) const COPIES: usize = 2;

/// A copy's usage: written in place, read back by tests.
const COPY_USAGE: BufferUsages = BufferUsages::INDEX
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC)
    .union(BufferUsages::IDLE_WRITES);

/// Unchanged words between two changed runs that are written with them
/// rather than starting another write: each write is a call into the
/// browser.
const WRITE_GAP: u32 = 512;

/// The tables its cull reads.
pub(crate) const MIRRORED: &[Table] = &[
    Table::Tiles,
    Table::Chunks,
    Table::Candidates,
    Table::NodeCull,
    Table::Indices,
];

const CANDIDATE_WORDS: usize = layout::candidate::WORDS;
const CHUNK_WORDS: usize = layout::chunk::WORDS;
const TILE_WORDS: usize = layout::tile::WORDS;
const NODE_WORDS: usize = layout::node_cull::WORDS;
/// `types.wgsl`'s draw kinds.
const DRAW_IMAGES: u32 = 0;
const DRAW_POINTS: u32 = 1;
const DRAW_INDEXED: u32 = 2;
/// What this driver hides with: the primitive restart index, which WebGL 2
/// skips without drawing. WebGL rejects element indices from 2^31, and
/// ANGLE drops a whole draw holding one, such as [`HIDDEN_INDEX`]; where
/// it isn't a restart (Metal's triangle lists), `pull_indexed` takes it
/// as hidden too.
pub(crate) const RESTART_INDEX: u32 = 0xffff_ffff;
/// `types.wgsl`'s vertices an image chunk draws.
const CHUNK_VERTICES: u32 = 768;
/// `types.wgsl`'s metres per unit of a placement's terrain range.
const TERRAIN_RANGE_UNIT: f32 = 4.0;

#[derive(Debug, Default)]
pub(crate) struct CpuDriver {
    /// Keeps every candidate, as `cull.wgsl`'s `keep_all`.
    keep_all: bool,
    /// Per node, [`ARGS_WORDS`] words as `cull.wgsl` writes them: the
    /// draw's arguments, then its counters.
    args: Vec<u32>,
    /// What the last cull read: the frame's words it reads and the copied
    /// tables' revision. `None` culls the next frame whatever it shows.
    seen: Option<(Vec<u32>, u64)>,
    /// Per node, the visible list last written for it.
    listed: Vec<Vec<u32>>,
    /// Per node, its candidates in runs of one mesh's chunks in one
    /// placement, each with the bounds of its chunks, as last laid out.
    runs: Vec<Vec<Run>>,
    /// A bit per compacted word, set where this driver wrote a chunk's
    /// first index and clear where it hid one: what a moved slot holds,
    /// which the GPU's cull reads from the slot itself.
    drawn: Vec<u64>,
    /// The GPU's cull beside this one, for a test to compare.
    twin: Option<Twin>,
    /// The compacted indices as this driver wrote them, which each copy
    /// takes on before it draws.
    shadow: Vec<u32>,
    /// The scene's compacted buffer the shadow and copies were made for:
    /// a new one (a resize) starts empty, every slot written again.
    made_for: Option<BufferId>,
    copies: Vec<Copy>,
    /// The copy draws read.
    current: usize,
}

/// One of the [`COPIES`].
#[derive(Debug)]
struct Copy {
    buffer: BufferId,
    /// Words changed since it was last written.
    stale: Vec<Range<u32>>,
}

/// Consecutive candidates of one mesh in one placement.
#[derive(Clone, Copy, Debug)]
struct Run {
    /// Its candidates, from the node's first.
    end: u32,
    /// Its chunks' bounds, as packed i16 pairs.
    bounds: [u32; 2],
}

impl CpuDriver {
    pub(crate) fn has_twin(&self) -> bool {
        self.twin.is_some()
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        self.twin
            .as_ref()
            .map_or_else(Vec::new, |t| t.cull.pipelines())
    }

    pub(crate) fn rebind(&mut self, gpu: &mut dyn Gpu, scene: &Scene) {
        if let Some(twin) = &mut self.twin {
            twin.rebind(gpu, scene);
        }
    }

    pub(crate) fn set_keep_all(&mut self, gpu: &mut dyn Gpu, on: bool) -> bool {
        if let Some(twin) = &mut self.twin {
            twin.cull.set_keep_all(gpu, on);
        }
        if self.keep_all == on {
            return false;
        }
        self.keep_all = on;
        self.seen = None;
        true
    }

    /// Runs the GPU's cull beside this one into its own buffers, or stops.
    pub(crate) fn set_twin(&mut self, gpu: &mut dyn Gpu, scene: &Scene, on: bool) {
        match (self.twin.take(), on) {
            (Some(twin), false) => twin.destroy(gpu),
            (Some(twin), true) => self.twin = Some(twin),
            (None, true) => {
                let mut twin = Twin::new(gpu, scene);
                twin.cull.set_keep_all(gpu, self.keep_all);
                self.twin = Some(twin);
            }
            (None, false) => {}
        }
    }

    /// Culls every node's candidates, if the camera or the tables changed
    /// since the last cull, and writes what changed. Returns whether a
    /// node's draw changed.
    pub(crate) fn prepare(&mut self, gpu: &mut dyn Gpu, scene: &mut Scene, rebuilt: bool) -> bool {
        let nodes = scene.node_count() as usize;
        let frame = Frame::new(scene.frame());
        let seen = (frame.inputs(scene.frame()), scene.tables().revision());
        let mut args = vec![0u32; nodes * ARGS_WORDS];
        let words = scene.capacity().compacted_words as usize;
        let remade = self.made_for != Some(scene.buffers().compacted);
        if remade {
            self.remake(gpu, words);
            self.made_for = Some(scene.buffers().compacted);
        }
        // Words of the shadow this frame changed.
        let mut changed: Vec<Range<u32>> = Vec::new();
        if rebuilt {
            self.listed.clear();
            self.runs.clear();
            self.drawn.resize(words.div_ceil(64), 0);
            move_bits(&mut self.drawn, scene.moves());
            // The slots of runs that went hide, then slots move, as the
            // GPU driver's do: before this frame's writes to the slots
            // they move from.
            for range in scene.hidden() {
                self.shadow[range.start as usize..range.end as usize].fill(RESTART_INDEX);
                changed.push(range.clone());
            }
            move_words(&mut self.shadow, scene.moves());
            changed.extend((scene.moves().iter()).map(|m| m.to..m.to + m.words));
            // The twin's, on the GPU, in a submission of their own.
            if let Some(twin) = &self.twin {
                let moves = scene.move_passes(gpu, twin.compacted);
                let longest = scene.hidden().iter().map(|r| r.end - r.start).max();
                let words = vec![HIDDEN_INDEX; longest.unwrap_or(0) as usize];
                for range in scene.hidden() {
                    gpu.write_buffer(
                        twin.compacted,
                        u64::from(range.start) * 4,
                        bytemuck::cast_slice(&words[..(range.end - range.start) as usize]),
                    );
                }
                if !moves.is_empty() {
                    gpu.submit(&moves);
                }
            }
        }
        self.listed.resize_with(nodes, Vec::new);
        if !rebuilt && self.seen.as_ref() == Some(&seen) && self.args.len() == args.len() {
            // Nothing moved: the same draws, and no indices written.
            args.copy_from_slice(&self.args);
            for node in args.chunks_mut(ARGS_WORDS) {
                node[6] = 0;
            }
        } else {
            self.cull_nodes(gpu, scene, &frame, &mut args, &mut changed);
        }
        self.seen = Some(seen);
        let draws = |args: &[u32]| -> Vec<[u32; 5]> {
            args.chunks(ARGS_WORDS)
                .map(|a| [a[0], a[1], a[2], a[3], a[4]])
                .collect()
        };
        let mut draws_changed = remade || draws(&args) != draws(&self.args);
        if rebuilt || args != self.args {
            gpu.write_buffer(scene.buffers().args, 0, bytemuck::cast_slice(&args));
        }
        self.args = args;
        if !changed.is_empty() {
            self.turn(gpu, &changed);
            draws_changed = true;
        }
        draws_changed
    }

    /// The copy draws read, once the first frame made them.
    pub(crate) fn compacted(&self) -> Option<BufferId> {
        self.copies.get(self.current).map(|c| c.buffer)
    }

    /// Makes the shadow and copies again, `words` long and empty, for a
    /// new compacted buffer: the next cull writes every slot laid out.
    fn remake(&mut self, gpu: &mut dyn Gpu, words: usize) {
        for copy in self.copies.drain(..) {
            gpu.destroy_buffer(copy.buffer);
        }
        self.shadow = vec![0; words];
        self.copies = (0..COPIES)
            .map(|_| Copy {
                buffer: gpu.create_buffer(&BufferDesc {
                    label: "compacted indices",
                    size: (words as u64 * 4).max(16),
                    usage: COPY_USAGE,
                }),
                stale: Vec::new(),
            })
            .collect();
        self.current = 0;
    }

    /// Moves the draws to the next copy, writing into it the words it
    /// lacks: those `changed` this frame and in the frames since it was
    /// last written.
    fn turn(&mut self, gpu: &mut dyn Gpu, changed: &[Range<u32>]) {
        for copy in &mut self.copies {
            copy.stale.extend_from_slice(changed);
        }
        self.current = (self.current + 1) % self.copies.len();
        let copy = &mut self.copies[self.current];
        let mut stale = std::mem::take(&mut copy.stale);
        stale.sort_unstable_by_key(|r| r.start);
        let mut at = 0;
        while at < stale.len() {
            let start = stale[at].start;
            let mut end = stale[at].end;
            at += 1;
            while at < stale.len() && stale[at].start <= end.saturating_add(WRITE_GAP) {
                end = end.max(stale[at].end);
                at += 1;
            }
            let end = end.min(self.shadow.len() as u32);
            if start < end {
                gpu.write_buffer(
                    copy.buffer,
                    u64::from(start) * 4,
                    bytemuck::cast_slice(&self.shadow[start as usize..end as usize]),
                );
            }
        }
        stale.clear();
        copy.stale = stale;
    }

    /// The cull of every node, into `args`.
    fn cull_nodes(
        &mut self,
        gpu: &mut dyn Gpu,
        scene: &mut Scene,
        frame: &Frame,
        args: &mut [u32],
        changed: &mut Vec<Range<u32>>,
    ) {
        let tables = scene.tables_mut();
        let mut candidates = std::mem::take(tables.mirror_mut(Table::Candidates).expect("copied"));
        let read = |table| tables.mirror(table).expect("copied");
        let (tiles, chunks, node_rows, indices) = (
            read(Table::Tiles),
            read(Table::Chunks),
            read(Table::NodeCull),
            read(Table::Indices),
        );
        let listed = (candidates.len() / CANDIDATE_WORDS) as u32;
        // Slot rewrites, in slot order within a node: (slot, words).
        let mut slots: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut visible: Vec<(u32, Vec<u32>)> = Vec::new();
        self.runs.resize_with(args.len() / ARGS_WORDS, Vec::new);
        for (n, out) in args.chunks_mut(ARGS_WORDS).enumerate() {
            use layout::node_cull as nc;
            let node = rows_at(node_rows, n, NODE_WORDS);
            let base = node[nc::VISIBLE_BASE];
            let count = node[nc::CANDIDATE_COUNT].min(listed - base.min(listed));
            let word = node[nc::DRAW];
            let draw = word & 255;
            if word & HIDDEN != 0 {
                out.fill(0);
                continue;
            }
            let test = Test {
                frame,
                margin_px: f32::from_bits(node[nc::MARGIN_PX]),
                draw,
                elevation: f32::from_bits(node[nc::ELEVATION]),
                draped: word & DRAPED != 0,
                unclipped: word & UNCLIPPED != 0,
            };
            let candidate = |i: u32| -> [u32; CANDIDATE_WORDS] {
                let at = (base + i) as usize * CANDIDATE_WORDS;
                candidates[at..at + CANDIDATE_WORDS]
                    .try_into()
                    .expect("a row")
            };
            if self.runs[n].is_empty() && count > 0 {
                self.runs[n] = runs(count, candidate, chunks);
            }
            let keeps = self.keeps(n, count, &test, tiles, chunks, &candidate);
            if draw == DRAW_INDEXED {
                use layout::candidate as c;
                use layout::chunk as ch;
                let index_base = node[nc::INDEX_BASE];
                let (mut extent, mut kept, mut written, mut drawn) = (0u32, 0u32, 0u32, 0u32);
                for (i, &keep) in keeps.iter().enumerate() {
                    let at = (base as usize + i) * CANDIDATE_WORDS;
                    let candidate = &mut candidates[at..at + CANDIDATE_WORDS];
                    // A hole runs left.
                    if candidate[c::CHUNK] == NO_CHUNK {
                        continue;
                    }
                    let chunk = rows_at(chunks, candidate[c::CHUNK] as usize, CHUNK_WORDS);
                    let index_count = chunk[ch::INDEX_COUNT];
                    if keep {
                        extent = extent.max(
                            candidate[c::SLOT]
                                .wrapping_add(index_count)
                                .wrapping_sub(index_base),
                        );
                        kept += 1;
                        drawn = drawn.wrapping_add(index_count);
                    }
                    let slot = candidate[c::SLOT];
                    if candidate[c::DRAWN] == MOVED {
                        let drawn = index_count > 0 && bit(&self.drawn, slot);
                        candidate[c::DRAWN] = u32::from(drawn);
                    }
                    if candidate[c::DRAWN] != u32::from(keep) {
                        let n = index_count.min(orb_layout::CHUNK_INDICES);
                        // A chunk without indices has no first word: its
                        // slot's is the next candidate's.
                        if n > 0 {
                            set_bit(&mut self.drawn, slot, keep);
                        }
                        let words = if keep {
                            let tag = (candidate[c::RUN] << node[nc::RUN_SHIFT])
                                .wrapping_add(chunk[ch::FIRST_VERTEX]);
                            let first = chunk[ch::FIRST_INDEX];
                            (first..first + n)
                                .map(|i| {
                                    let word = indices.get((i >> 1) as usize).copied().unwrap_or(0);
                                    tag.wrapping_add((word >> ((i & 1) << 4)) & 0xffff)
                                })
                                .collect()
                        } else {
                            vec![RESTART_INDEX; n as usize]
                        };
                        slots.push((candidate[c::SLOT], words));
                        candidate[c::DRAWN] = u32::from(keep);
                        written = written.wrapping_add(index_count);
                    }
                }
                let visible = extent.min(node[nc::INDEX_COUNT]);
                let draw = if visible > 0 {
                    [visible, 1, index_base]
                } else {
                    [0; 3]
                };
                out.copy_from_slice(&[draw[0], draw[1], draw[2], 0, 0, kept, written, drawn]);
            } else {
                use layout::candidate as c;
                let rows: Vec<u32> = (keeps.iter().enumerate())
                    .filter(|(_, keep)| **keep)
                    .flat_map(|(i, _)| {
                        let candidate = candidate(i as u32);
                        [candidate[c::CHUNK], candidate[c::TILE]]
                    })
                    .collect();
                let written = (rows.len() / 2) as u32;
                if rows != self.listed[n] {
                    visible.push((base * 2, rows.clone()));
                    self.listed[n] = rows;
                }
                let points = draw == DRAW_POINTS;
                let cells = if word & SURFACE != 0 {
                    frame.surface_cells
                } else {
                    1
                };
                let vertices = match (written, points) {
                    (0, _) => 0,
                    (_, true) => 6 * cells * cells,
                    (_, false) => CHUNK_VERTICES,
                };
                let instances = written * if points { 256 } else { 1 };
                out.copy_from_slice(&[
                    vertices,
                    instances,
                    0,
                    0,
                    0,
                    written,
                    0,
                    vertices.wrapping_mul(instances),
                ]);
            }
        }
        *tables.mirror_mut(Table::Candidates).expect("copied") = candidates;
        // Each rewritten slot's words, into the shadow.
        for (start, words) in slots {
            let end = start as usize + words.len();
            debug_assert!(end <= self.shadow.len());
            if let Some(shadow) = self.shadow.get_mut(start as usize..end) {
                shadow.copy_from_slice(&words);
                changed.push(start..end as u32);
            }
        }
        for (at, rows) in visible {
            tables.write_words(gpu, Table::Visible, at, &rows);
        }
    }

    /// Whether node `n`'s candidates each draw: every one with
    /// [`keep_all`](Self::keep_all), else those of runs whose bounds meet
    /// the view that [`candidate_visible`] keeps.
    fn keeps(
        &self,
        n: usize,
        count: u32,
        test: &Test<'_>,
        tiles: &[u32],
        chunks: &[u32],
        candidate: &dyn Fn(u32) -> [u32; CANDIDATE_WORDS],
    ) -> Vec<bool> {
        use layout::candidate as c;
        if self.keep_all {
            return vec![true; count as usize];
        }
        let mut keeps = Vec::with_capacity(count as usize);
        let mut start = 0;
        for run in &self.runs[n] {
            let first = candidate(start);
            let tile = rows_at(tiles, first[c::TILE] as usize, TILE_WORDS);
            // A run of one is its chunk's test alone.
            let met = run.end - start == 1 || test.run_meets(tile, run.bounds);
            for i in start..run.end {
                keeps.push(
                    met && {
                        let candidate = candidate(i);
                        candidate[c::CHUNK] != NO_CHUNK && {
                            let chunk = rows_at(chunks, candidate[c::CHUNK] as usize, CHUNK_WORDS);
                            let tile = rows_at(tiles, candidate[c::TILE] as usize, TILE_WORDS);
                            test.candidate_visible(
                                tile,
                                chunk[layout::chunk::BOUNDS_MIN],
                                chunk[layout::chunk::BOUNDS_MAX],
                                0.0,
                                0.0,
                            )
                        }
                    },
                );
            }
            start = run.end;
        }
        debug_assert_eq!(keeps.len(), count as usize);
        keeps
    }

    /// The GPU twin's cull, writing its own buffers.
    pub(crate) fn cull(&self, gpu: &mut dyn Gpu, frame: Cull<'_>, passes: &mut Vec<Pass>) -> u32 {
        self.twin
            .as_ref()
            .map_or(0, |twin| twin.cull(gpu, frame, passes))
    }

    /// Node `node`'s draw, from its arguments.
    pub(crate) fn draw(&self, node: u32, draw: NodeDraw) -> RenderCommand {
        let at = node as usize * ARGS_WORDS;
        let a = self.args.get(at..at + 5).unwrap_or(&[0; 5]);
        if draw == NodeDraw::Indexed {
            RenderCommand::DrawIndexed {
                indices: a[2]..a[2] + a[0],
                base_vertex: a[3] as i32,
                instances: a[4]..a[4] + a[1],
            }
        } else {
            RenderCommand::Draw {
                vertices: a[2]..a[2] + a[0],
                instances: a[3]..a[3] + a[1],
            }
        }
    }

    pub(crate) fn draws_nothing(&self, node: u32) -> bool {
        let at = node as usize * ARGS_WORDS;
        self.args
            .get(at..at + 2)
            .is_none_or(|a| a[0] == 0 || a[1] == 0)
    }

    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        if let Some(twin) = self.twin {
            twin.destroy(gpu);
        }
        for copy in self.copies {
            gpu.destroy_buffer(copy.buffer);
        }
    }
}

/// Whether bit `i` is set (none past the end).
fn bit(bits: &[u64], i: u32) -> bool {
    bits.get(i as usize / 64)
        .is_some_and(|w| w >> (i % 64) & 1 != 0)
}

fn set_bit(bits: &mut [u64], i: u32, on: bool) {
    if let Some(w) = bits.get_mut(i as usize / 64) {
        let mask = 1 << (i % 64);
        *w = if on { *w | mask } else { *w & !mask };
    }
}

/// The 64 bits from bit `at` (none past the end).
fn bits_at(bits: &[u64], at: u32) -> u64 {
    let (w, o) = ((at / 64) as usize, at % 64);
    let word = |w: usize| bits.get(w).copied().unwrap_or(0);
    match o {
        0 => word(w),
        _ => word(w) >> o | word(w + 1) << (64 - o),
    }
}

/// Sets the `n` (1 to 64) bits from bit `at` to `value`'s lowest.
fn put_bits(bits: &mut [u64], at: u32, value: u64, n: u32) {
    let mask = u64::MAX >> (64 - n);
    let value = value & mask;
    let (w, o) = ((at / 64) as usize, at % 64);
    if let Some(x) = bits.get_mut(w) {
        *x = (*x & !(mask << o)) | value << o;
    }
    if o + n > 64
        && let Some(x) = bits.get_mut(w + 1)
    {
        let shift = 64 - o;
        *x = (*x & !(mask >> shift)) | value >> shift;
    }
}

/// Moves bits as `moves` moved their words: every one read before any
/// is written, as the moves copied them.
fn move_bits(bits: &mut [u64], moves: &[Move]) {
    let read: Vec<Vec<u64>> = (moves.iter())
        .map(|m| {
            (0..m.words.div_ceil(64))
                .map(|k| bits_at(bits, m.from + 64 * k))
                .collect()
        })
        .collect();
    for (m, read) in moves.iter().zip(read) {
        for (k, value) in (0u32..).zip(read) {
            put_bits(bits, m.to + 64 * k, value, (m.words - 64 * k).min(64));
        }
    }
}

/// Makes `moves` in `words`: every one read before any is written, as
/// the GPU's copies through a scratch buffer make them.
fn move_words(words: &mut [u32], moves: &[Move]) {
    let read: Vec<Vec<u32>> = (moves.iter())
        .map(|m| {
            let from = m.from as usize..(m.from + m.words) as usize;
            words.get(from).unwrap_or(&[]).to_vec()
        })
        .collect();
    for (m, read) in moves.iter().zip(read) {
        if let Some(to) = words.get_mut(m.to as usize..m.to as usize + read.len()) {
            to.copy_from_slice(&read);
        }
    }
}

/// `rows`' row `i`, `words` long, zeros where it isn't written.
fn rows_at(rows: &[u32], i: usize, words: usize) -> &[u32] {
    static ZERO: [u32; 64] = [0; 64];
    rows.get(i * words..(i + 1) * words)
        .unwrap_or(&ZERO[..words])
}

/// A node's `count` candidates in runs of one mesh (its vertex base) in
/// one placement, each with its chunks' bounds; each hole runs left
/// ([`NO_CHUNK`]) a run of its own.
fn runs(count: u32, candidate: impl Fn(u32) -> [u32; CANDIDATE_WORDS], chunks: &[u32]) -> Vec<Run> {
    use layout::candidate as c;
    use layout::chunk as ch;
    let mut runs: Vec<Run> = Vec::new();
    let mut key = None;
    for i in 0..count {
        let candidate = candidate(i);
        if candidate[c::CHUNK] == NO_CHUNK {
            runs.push(Run {
                end: i + 1,
                bounds: [0; 2],
            });
            key = None;
            continue;
        }
        let chunk = rows_at(chunks, candidate[c::CHUNK] as usize, CHUNK_WORDS);
        let this = (candidate[c::TILE], chunk[ch::VERTEX_BASE]);
        let [lo, hi] = [chunk[ch::BOUNDS_MIN], chunk[ch::BOUNDS_MAX]].map(unpack);
        match runs.last_mut() {
            Some(run) if key == Some(this) => {
                run.end = i + 1;
                let [rlo, rhi] = run.bounds.map(unpack);
                run.bounds = [
                    pack([rlo[0].min(lo[0]), rlo[1].min(lo[1])]),
                    pack([rhi[0].max(hi[0]), rhi[1].max(hi[1])]),
                ];
            }
            _ => runs.push(Run {
                end: i + 1,
                bounds: [chunk[ch::BOUNDS_MIN], chunk[ch::BOUNDS_MAX]],
            }),
        }
        key = Some(this);
    }
    runs
}

/// Every word of the slots the last cull decided, as a cull writing them
/// all would write it (hidden ones [`HIDDEN_INDEX`]): (word, index).
fn decided_slots(
    nodes: &[u32],
    candidates: &[u32],
    chunks: &[u32],
    indices: &[u32],
) -> Vec<(u32, u32)> {
    use layout::candidate as c;
    use layout::chunk as ch;
    use layout::node_cull as nc;
    let mut slots = Vec::new();
    for node in nodes.as_chunks::<NODE_WORDS>().0 {
        let draw = node[nc::DRAW];
        if draw & HIDDEN != 0 || draw & 255 != DRAW_INDEXED {
            continue;
        }
        let base = node[nc::VISIBLE_BASE] as usize;
        for i in base..base + node[nc::CANDIDATE_COUNT] as usize {
            let row = rows_at(candidates, i, CANDIDATE_WORDS);
            if row[c::CHUNK] == NO_CHUNK {
                continue;
            }
            let chunk = rows_at(chunks, row[c::CHUNK] as usize, CHUNK_WORDS);
            let n = chunk[ch::INDEX_COUNT].min(orb_layout::CHUNK_INDICES);
            let tag = (row[c::RUN] << node[nc::RUN_SHIFT]).wrapping_add(chunk[ch::FIRST_VERTEX]);
            for k in 0..n {
                let word = match row[c::DRAWN] {
                    1 => {
                        let i = chunk[ch::FIRST_INDEX] + k;
                        let pair = indices.get((i >> 1) as usize).copied().unwrap_or(0);
                        tag.wrapping_add((pair >> ((i & 1) << 4)) & 0xffff)
                    }
                    _ => HIDDEN_INDEX,
                };
                slots.push((row[c::SLOT] + k, word));
            }
        }
    }
    slots
}

/// `types.wgsl`'s `unpack_i16x2`.
fn unpack(w: u32) -> [f32; 2] {
    [
        f32::from((w & 0xffff) as u16 as i16),
        f32::from((w >> 16) as u16 as i16),
    ]
}

fn pack(p: [f32; 2]) -> u32 {
    u32::from(p[0] as i16 as u16) | u32::from(p[1] as i16 as u16) << 16
}

/// What the cull reads of the frame block.
#[derive(Clone, Copy, Debug)]
struct Frame {
    /// Columns.
    view_proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    world_per_px: f32,
    extrusion_scale: f32,
    shadow_offset: [f32; 2],
    camera_whole: [i32; 2],
    camera_rest: [f32; 2],
    quantum: f32,
    surface_cells: u32,
}

impl Frame {
    /// Its words' ranges in the block.
    const READS: [(usize, usize); 9] = {
        use layout::frame as f;
        [
            (f::VIEW_PROJ, 16),
            (f::VIEWPORT, 2),
            (f::WORLD_PER_PX, 1),
            (f::EXTRUSION_SCALE, 1),
            (f::SHADOW_OFFSET, 2),
            (f::CAMERA_WHOLE, 2),
            (f::CAMERA_REST, 2),
            (f::QUANTUM, 1),
            (f::SURFACE_CELLS, 1),
        ]
    };

    fn new(words: &[u32]) -> Self {
        use layout::frame as f;
        let word = |i: usize| words.get(i).copied().unwrap_or(0);
        let float = |i: usize| f32::from_bits(word(i));
        Self {
            view_proj: std::array::from_fn(|c| {
                std::array::from_fn(|r| float(f::VIEW_PROJ + 4 * c + r))
            }),
            viewport: [float(f::VIEWPORT), float(f::VIEWPORT + 1)],
            world_per_px: float(f::WORLD_PER_PX),
            extrusion_scale: float(f::EXTRUSION_SCALE),
            shadow_offset: [float(f::SHADOW_OFFSET), float(f::SHADOW_OFFSET + 1)],
            camera_whole: [
                word(f::CAMERA_WHOLE) as i32,
                word(f::CAMERA_WHOLE + 1) as i32,
            ],
            camera_rest: [float(f::CAMERA_REST), float(f::CAMERA_REST + 1)],
            quantum: float(f::QUANTUM),
            surface_cells: word(f::SURFACE_CELLS),
        }
    }

    /// The words of the block the cull reads.
    fn inputs(&self, words: &[u32]) -> Vec<u32> {
        Self::READS
            .iter()
            .flat_map(|&(at, n)| (at..at + n).map(|i| words.get(i).copied().unwrap_or(0)))
            .collect()
    }

    /// `view_proj * (x, y, z, w)`, summed as the shader writes it.
    fn project(&self, [x, y, z, w]: [f32; 4]) -> [f32; 4] {
        let m = &self.view_proj;
        std::array::from_fn(|r| m[0][r] * x + m[1][r] * y + m[2][r] * z + m[3][r] * w)
    }
}

/// One node's cull parameters.
struct Test<'a> {
    frame: &'a Frame,
    margin_px: f32,
    draw: u32,
    elevation: f32,
    draped: bool,
    /// Its meshes draw past their clip rectangles.
    unclipped: bool,
}

impl Test<'_> {
    /// Whether any chunk of a run with these bounds may draw: the chunks'
    /// test of its bounds, grown by a data unit and its margin by a pixel,
    /// so that it never drops a chunk [`candidate_visible`] keeps whatever
    /// the rounding. Each corner's outside test is concave in the corner,
    /// so a box wholly outside a plane holds no point inside it.
    fn run_meets(&self, tile: &[u32], [lo, hi]: [u32; 2]) -> bool {
        self.candidate_visible(tile, lo, hi, 1.0, 1.0)
    }

    /// `cull.wgsl`'s `candidate_visible` of a chunk with these bounds in
    /// this placement, the bounds grown by `grow` data units and the
    /// margin by `margin` pixels (0 for the twin itself). Every step is in
    /// f32 in the shader's order.
    fn candidate_visible(&self, tile: &[u32], min: u32, max: u32, grow: f32, margin: f32) -> bool {
        use layout::tile as t;
        let f = self.frame;
        let float = |i: usize| f32::from_bits(tile[i]);
        let margin_px = self.margin_px + margin;
        let points = self.draw == DRAW_POINTS;
        let scale = float(t::SCALE);
        let margin = margin_px * (f.world_per_px * float(t::PX_SCALE)) / scale;
        let before = (if points { 0.0 } else { margin }) + grow;
        let [lx, ly] = unpack(min);
        let [hx, hy] = unpack(max);
        let (lo, hi) = ([lx - before, ly - before], [hx + before, hy + before]);
        let unclipped = self.draw == DRAW_IMAGES || self.unclipped;
        let (mut clip_lo, mut clip_hi) = if unclipped {
            (lo, hi)
        } else {
            ([1e9; 2], [-1e9; 2])
        };
        if !unclipped {
            for i in 0..4 {
                let [cx, cy] = unpack(tile[t::CLIPS + 2 * i]);
                let [dx, dy] = unpack(tile[t::CLIPS + 2 * i + 1]);
                let c_lo = [lo[0].max(cx), lo[1].max(cy)];
                let c_hi = [hi[0].min(dx), hi[1].min(dy)];
                if c_lo[0] <= c_hi[0] && c_lo[1] <= c_hi[1] {
                    clip_lo = [clip_lo[0].min(c_lo[0]), clip_lo[1].min(c_lo[1])];
                    clip_hi = [clip_hi[0].max(c_hi[0]), clip_hi[1].max(c_hi[1])];
                }
            }
        }
        if clip_lo[0] > clip_hi[0] || clip_lo[1] > clip_hi[1] {
            return false;
        }
        if self.draped || tile[t::WARP_CELLS] != 0 || tile[t::GLOBE_CELLS] != 0 {
            return true;
        }
        let after = if points { margin } else { 0.0 };
        let lo = [clip_lo[0] - after, clip_lo[1] - after];
        let hi = [clip_hi[0] + after, clip_hi[1] + after];
        let [g0, g1] = unpack(tile[t::TERRAIN_RANGE]);
        let ground = [
            g0 * TERRAIN_RANGE_UNIT * f.extrusion_scale,
            g1 * TERRAIN_RANGE_UNIT * f.extrusion_scale,
        ];
        // `tile_origin`: whole quanta as integers, then the remainders.
        let origin: [f32; 2] = std::array::from_fn(|k| {
            (tile[t::ORIGIN + k] as i32).wrapping_sub(f.camera_whole[k]) as f32 * f.quantum
                + (float(t::ORIGIN_REST + k) - f.camera_rest[k])
        });
        let affine = |p: [f32; 2]| [origin[0] + p[0] * scale, origin[1] + -p[1] * scale];
        let corner = |p: [f32; 2]| {
            let [x, y] = affine(p);
            f.project([x, y, ground[0], 1.0])
        };
        let corners = [
            corner(lo),
            corner([hi[0], lo[1]]),
            corner([lo[0], hi[1]]),
            corner(hi),
        ];
        let rise = ground[1] - ground[0] + self.elevation * f.extrusion_scale;
        let raised = f.project([0.0, 0.0, rise, 0.0]);
        let m = [
            2.0 * margin_px / f.viewport[0],
            2.0 * margin_px / f.viewport[1],
        ];
        let reach = self.elevation * f.extrusion_scale;
        let shadow = f.project([
            f.shadow_offset[0] * reach,
            f.shadow_offset[1] * reach,
            0.0,
            0.0,
        ]);
        let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] { std::array::from_fn(|k| a[k] + b[k]) };
        let mut outside = u32::MAX;
        for c in corners {
            outside &= outside_planes(c, m)
                & outside_planes(add(c, raised), m)
                & outside_planes(add(c, shadow), m);
        }
        outside == 0
    }
}

/// `cull.wgsl`'s `outside_planes`: the view-volume planes a clip-space
/// corner lies outside, a bit each.
fn outside_planes(p: [f32; 4], margin: [f32; 2]) -> u32 {
    let [x, y, z, w] = p;
    let extra = [margin[0] * w.max(0.0), margin[1] * w.max(0.0)];
    u32::from(x > w + extra[0])
        | u32::from(x < -w - extra[0]) << 1
        | u32::from(y > w + extra[1]) << 2
        | u32::from(y < -w - extra[1]) << 3
        | u32::from(z > w) << 4
        | u32::from(z < 0.0) << 5
}

/// The words the CPU driver wrote and its GPU twin's, after a frame of a
/// renderer whose twin runs (`Renderer::check_driver`): held to each
/// other by [`compare`](Self::compare).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct DriverCheck {
    /// The draw arguments: the CPU's, the GPU's.
    pub args: [orb_hal::BufferId; 2],
    pub compacted: [orb_hal::BufferId; 2],
    pub visible: [orb_hal::BufferId; 2],
    /// The candidates as the GPU's cull left them (their drawn words).
    pub candidates: orb_hal::BufferId,
    /// The candidates as the CPU's left them.
    cpu_candidates: Vec<u32>,
    node_rows: Vec<u32>,
    nodes: usize,
    compacted_words: usize,
    /// What every slot the CPU's cull decided holds, whatever moved it
    /// there: (word, index).
    slots: Vec<(u32, u32)>,
}

/// What [`DriverCheck::compare`] compared.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Compared {
    pub nodes: usize,
    pub candidates: usize,
    /// Candidates of indexed nodes, and those both kept.
    pub indexed: usize,
    pub kept: usize,
    pub compacted_words: usize,
    /// Listed nodes' visible rows both wrote.
    pub visible_rows: usize,
}

impl DriverCheck {
    pub(crate) fn new(driver: &CpuDriver, scene: &Scene) -> Option<Self> {
        let twin = driver.twin.as_ref()?;
        let tables = scene.tables();
        let node_rows = tables.mirror(Table::NodeCull)?;
        let slots = decided_slots(
            node_rows,
            tables.mirror(Table::Candidates)?,
            tables.mirror(Table::Chunks)?,
            tables.mirror(Table::Indices)?,
        );
        Some(Self {
            slots,
            args: [scene.buffers().args, twin.args],
            compacted: [driver.compacted()?, twin.compacted],
            visible: [tables.buffer(Table::Visible), twin.visible],
            candidates: tables.buffer(Table::Candidates),
            cpu_candidates: tables.mirror(Table::Candidates)?.to_vec(),
            node_rows: tables.mirror(Table::NodeCull)?.to_vec(),
            nodes: scene.node_count() as usize,
            compacted_words: scene.stats().compacted_words as usize,
        })
    }

    /// Compares what the two drivers wrote, word for word, `read` giving
    /// a buffer's words: every node's draw arguments and counters, every
    /// candidate's row with what each cull last decided of it, indexed
    /// nodes' compacted slots, and listed nodes' visible lists.
    pub fn compare(
        &self,
        read: &mut dyn FnMut(orb_hal::BufferId) -> Vec<u32>,
    ) -> Result<Compared, String> {
        use layout::node_cull as nc;
        let [cpu_args, gpu_args] = self.args.map(&mut *read);
        for n in 0..self.nodes {
            let at = n * ARGS_WORDS..(n + 1) * ARGS_WORDS;
            if cpu_args.get(at.clone()) != gpu_args.get(at.clone()) {
                return Err(format!(
                    "node {n}'s arguments: CPU {:?}, GPU {:?}",
                    cpu_args.get(at.clone()),
                    gpu_args.get(at)
                ));
            }
        }
        let gpu_candidates = read(self.candidates);
        let candidates = self.cpu_candidates.len() / CANDIDATE_WORDS;
        for i in 0..candidates {
            let at = i * CANDIDATE_WORDS..(i + 1) * CANDIDATE_WORDS;
            let (cpu, gpu) = (&self.cpu_candidates[at.clone()], gpu_candidates.get(at));
            if Some(cpu) != gpu {
                return Err(format!("candidate {i}: CPU {cpu:?}, GPU {gpu:?}"));
            }
        }
        let [cpu_compacted, gpu_compacted] = self.compacted.map(&mut *read);
        // Hidden either way: the GPU cull's index, or this driver's restart.
        let cpu_compacted: Vec<u32> = (cpu_compacted.into_iter())
            .map(|w| if w == RESTART_INDEX { HIDDEN_INDEX } else { w })
            .collect();
        let n = self.compacted_words;
        if let Some(at) = (0..n).find(|&i| cpu_compacted.get(i) != gpu_compacted.get(i)) {
            return Err(format!(
                "compacted word {at} of {n}: CPU {:?}, GPU {:?}",
                cpu_compacted.get(at),
                gpu_compacted.get(at)
            ));
        }
        // Equal, and right: slots that moved hold what they would have
        // held written again.
        if let Some((at, word)) =
            (self.slots.iter()).find(|&&(at, word)| cpu_compacted.get(at as usize) != Some(&word))
        {
            return Err(format!(
                "compacted word {at}: {:?}, where its candidate wants {word}",
                cpu_compacted.get(*at as usize)
            ));
        }
        let [cpu_visible, gpu_visible] = self.visible.map(&mut *read);
        let (mut rows, mut indexed, mut kept) = (0, 0, 0);
        for node in 0..self.nodes {
            let row = rows_at(&self.node_rows, node, NODE_WORDS);
            let word = row[nc::DRAW];
            if word & HIDDEN != 0 {
                continue;
            }
            if word & 255 == DRAW_INDEXED {
                indexed += row[nc::CANDIDATE_COUNT] as usize;
                kept += gpu_args.get(node * ARGS_WORDS + 5).copied().unwrap_or(0) as usize;
                continue;
            }
            let base = row[nc::VISIBLE_BASE] as usize * 2;
            let written = gpu_args.get(node * ARGS_WORDS + 5).copied().unwrap_or(0) as usize;
            let at = base..base + written * 2;
            if cpu_visible.get(at.clone()) != gpu_visible.get(at.clone()) {
                return Err(format!(
                    "node {node}'s visible list differs over words {at:?}"
                ));
            }
            rows += written;
        }
        Ok(Compared {
            nodes: self.nodes,
            candidates,
            indexed,
            kept,
            compacted_words: n,
            visible_rows: rows,
        })
    }
}
