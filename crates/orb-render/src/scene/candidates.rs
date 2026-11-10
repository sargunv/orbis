//! The cull's candidate layout: each node's slice of the candidate and
//! visible lists and, for an indexed node, of the compacted index buffer.
//!
//! An indexed node gives each candidate a slot of the compacted buffer,
//! which the cull fills with the chunk's indices when it comes into view
//! and hides when it leaves, and the candidate's row holds which of the two
//! the slot holds (`drawn`, on the GPU). Laying everything out again gave
//! every candidate a new slot and an unknown state, so the next cull
//! rewrote every slot: 10–18 ms of cull on an Adreno 619 on each frame a
//! tile came or went (#22). So a rebuild keeps the layout where it can:
//!
//! - An unsorted indexed node keeps each run's (a mesh in one placement)
//!   rows, run number and slots for as long as the run lives. A run that
//!   goes leaves its rows as holes the cull skips ([`NO_CHUNK`]) and its
//!   slots hidden. One that comes takes rows anywhere and slots in a hole
//!   between the runs drawn before and after it, or past the last: the
//!   node's slots keep its runs' draw order, which shows where two tiles'
//!   antialiased edges meet on one pixel. With no room there, the runs
//!   after it slide along into the next hole. Only the runs that came are
//!   written, by the next cull: slots that slide move.
//! - A node whose changes don't fit that way (a run with no room in its
//!   slice, holes in over an eighth of its slots, which draw as hidden
//!   indices, a listed or sorted node) is laid out again within its
//!   slices, or in new ones past every node's when it outgrew them, its
//!   runs that stay keeping their run numbers and moving their slots.
//! - Slots that move are copied on the GPU (through a scratch buffer, as
//!   a buffer can't copy into itself), not written again, their rows'
//!   `drawn` word [`MOVED`]: a copy costs the GPU far less than the
//!   cull's rewrite, and the CPU driver would upload what it rewrites.
//! - Each slice has room to grow past what it holds. When a node's runs
//!   outnumber what its indices can address, or the nodes or lists
//!   themselves change, everything is laid out again from the start, the
//!   runs that stay moving as above.
//!
//! A run's slots move only once a cull has written them: until then they
//! hold whatever was there before, and its rows stay `UNKNOWN`.

use std::ops::Range;

use orb_hal::{BufferId, Gpu, Pass};
use orb_layout::Chunk;
use orb_shaders::layout;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::driver::SubmitLimits;

use super::{
    CANDIDATE_WORDS, HIDDEN, INDEXED_BATCH, NODE_CULL_WORDS, NodeDraw, NodeEntry, PlacementTarget,
    Scene, Table, UNKNOWN, VISIBLE_WORDS,
};

/// A candidate row that holds no candidate: a hole a run left in an
/// indexed node's rows, which the cull skips (`cull.wgsl`'s `NO_CHUNK`).
pub(crate) const NO_CHUNK: u32 = u32::MAX;

/// A candidate's `drawn` word once its slot moved with its run: the slot
/// holds the chunk's indices or hidden ones, as the slot it was copied
/// from did, and its first word says which (`cull.wgsl`'s `MOVED`).
pub(crate) const MOVED: u32 = 2;

/// Compacted words a layout moved: `words` from `from` to `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Move {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) words: u32,
}

/// Room in one of a node's slices, relative to its start: the first hole a
/// request fits, else past the last thing in it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Space {
    cap: u32,
    end: u32,
    /// Free ranges below `end`, in order, none touching another.
    holes: Vec<Range<u32>>,
}

impl Space {
    fn new(used: u32, cap: u32) -> Self {
        Self {
            cap,
            end: used,
            holes: Vec::new(),
        }
    }

    fn alloc(&mut self, n: u32) -> Option<u32> {
        if let Some(i) = self.holes.iter().position(|h| h.end - h.start >= n) {
            let at = self.holes[i].start;
            self.holes[i].start += n;
            if self.holes[i].is_empty() {
                self.holes.remove(i);
            }
            return Some(at);
        }
        let at = self.end;
        (u64::from(at) + u64::from(n) <= u64::from(self.cap)).then(|| {
            self.end += n;
            at
        })
    }

    /// Room for `n` within `lo..hi`: the first hole there it fits, else
    /// past the end when `hi` lies past it.
    fn alloc_in(&mut self, n: u32, lo: u32, hi: u32) -> Option<u32> {
        for i in 0..self.holes.len() {
            let hole = self.holes[i].clone();
            let at = hole.start.max(lo);
            if at + n > hole.end.min(hi) {
                continue;
            }
            self.split(i, at..at + n);
            return Some(at);
        }
        let at = self.end.max(lo);
        if at != self.end || u64::from(at) + u64::from(n) > u64::from(self.cap.min(hi)) {
            return None;
        }
        self.end += n;
        Some(at)
    }

    /// Takes `range` out of hole `i`, which holds it.
    fn split(&mut self, i: usize, range: Range<u32>) {
        let hole = self.holes[i].clone();
        // What's left of the hole on either side.
        let (before, after) = (hole.start..range.start, range.end..hole.end);
        match (before.is_empty(), after.is_empty()) {
            (true, true) => {
                self.holes.remove(i);
            }
            (true, false) => self.holes[i] = after,
            (false, true) => self.holes[i] = before,
            (false, false) => {
                self.holes[i] = before;
                self.holes.insert(i + 1, after);
            }
        }
    }

    /// Takes `range`, which is free: in a hole, or past the end within
    /// the slice.
    fn take(&mut self, range: Range<u32>) {
        if range.is_empty() {
            return;
        }
        debug_assert!(range.end <= self.cap);
        let mut inside = range.clone();
        if range.end > self.end {
            if range.start > self.end {
                self.holes.push(self.end..range.start);
            }
            inside.end = self.end.max(range.start);
            self.end = range.end;
        }
        if !inside.is_empty() {
            let i = (self.holes.iter())
                .position(|h| h.start <= inside.start && inside.end <= h.end)
                .expect("a free range");
            self.split(i, inside);
        }
    }

    fn free(&mut self, range: Range<u32>) {
        if range.is_empty() {
            return;
        }
        let i = self.holes.partition_point(|h| h.start < range.start);
        self.holes.insert(i, range);
        // Join it to its neighbours.
        if i + 1 < self.holes.len() && self.holes[i].end == self.holes[i + 1].start {
            self.holes[i].end = self.holes.remove(i + 1).end;
        }
        if i > 0 && self.holes[i - 1].end == self.holes[i].start {
            self.holes[i - 1].end = self.holes.remove(i).end;
        }
        let end = self.end;
        if let Some(last) = self.holes.pop_if(|h| h.end == end) {
            self.end = last.start;
        }
    }
}

/// A run as laid out: its number, its rows and slots in the node's
/// slices, and whether a cull has written its slots since, so they can
/// move.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LaidRun {
    run: u32,
    rows: Range<u32>,
    words: Range<u32>,
    written: bool,
}

/// What identifies a run across rebuilds: its mesh (by the serial it was
/// added with, which no other mesh takes) and its placement slot.
type RunKey = (u64, u32);

/// A node's slices of the lists as last laid out.
#[derive(Clone, Debug)]
pub(super) struct LaidNode {
    /// How it draws, which a layout depends on.
    shape: (NodeDraw, PlacementTarget, bool, bool),
    /// Its candidate rows (and, for an indexed node, its runs' rows of the
    /// visible list).
    row_base: u32,
    rows: Space,
    /// An indexed node's compacted words.
    word_base: u32,
    words: Space,
    run_shift: u32,
    /// For a node that keeps its runs where they are: them, and which run
    /// numbers are taken.
    runs: Option<FxHashMap<RunKey, LaidRun>>,
    numbers: Vec<bool>,
    /// Whether a run's slots wait for a cull to write them.
    unwritten: bool,
    /// Its candidates, holes aside.
    pub(super) live: u32,
}

impl LaidNode {
    /// Its runs, whose slots a layout of the node again can move, if its
    /// indices number runs as they would then.
    fn movable(&self, run_shift: u32) -> Option<Movable<'_>> {
        let runs = self.runs.as_ref().filter(|_| self.run_shift == run_shift)?;
        Some(Movable {
            runs,
            word_base: self.word_base,
        })
    }
}

/// A node's runs as last laid out, whose slots can move.
#[derive(Clone, Copy)]
struct Movable<'a> {
    runs: &'a FxHashMap<RunKey, LaidRun>,
    word_base: u32,
}

/// A run to lay out: a mesh's chunks in one placement.
#[derive(Clone)]
struct RunPlan<'a> {
    key: RunKey,
    placement: u32,
    /// The mesh's chunk slots (the first names the run in its run row) and
    /// chunks.
    slots: &'a [u32],
    chunks: &'a [Chunk],
}

impl RunPlan<'_> {
    /// Its chunks: slot, index count, sort key.
    fn chunks(&self) -> impl Iterator<Item = (u32, u32, Option<f32>)> + '_ {
        (self.slots.iter().zip(self.chunks)).map(|(&slot, c)| (slot, c.index_count, c.sort_key))
    }

    fn first(&self) -> u32 {
        self.slots[0]
    }
}

fn shape(node: &NodeEntry) -> (NodeDraw, PlacementTarget, bool, bool) {
    let c = &node.cull;
    (c.draw, c.target, c.labels, c.unclipped)
}

/// An indexed node's index is `run << run_shift | vertex`.
fn run_shift(vertex_words: u32) -> u32 {
    32 - (vertex_words - 1).leading_zeros().min(31)
}

/// Room a node's slice keeps past what it holds: none while it holds
/// nothing.
fn headroom(used: u32, step: u32) -> u32 {
    if used == 0 { 0 } else { used / 2 + step }
}

impl Scene {
    /// A node's runs, in mesh, placement and chunk order, and the most
    /// vertex words a mesh of it has.
    fn node_runs<'a>(&'a self, node: &NodeEntry) -> (Vec<RunPlan<'a>>, u32) {
        let mut runs = Vec::new();
        let mut vertex_words = 1u32;
        for mesh in &node.meshes {
            let entry = self.meshes[mesh.0 as usize]
                .as_ref()
                .expect("node lists live meshes");
            let tile = self.tiles[entry.tile.0 as usize]
                .as_ref()
                .expect("meshes have tiles");
            if entry.chunk_slots.is_empty() {
                continue;
            }
            vertex_words = vertex_words.max(entry.vertices.len() as u32);
            for placement in &tile.placements {
                if !node.cull.draws(placement.row.target) {
                    continue;
                }
                runs.push(RunPlan {
                    key: (self.mesh_serials[mesh.0 as usize], placement.slot),
                    placement: placement.slot,
                    slots: &entry.chunk_slots,
                    chunks: &entry.chunks,
                });
            }
        }
        (runs, vertex_words)
    }

    /// Lays out every node's candidates contiguously, in mesh, placement
    /// and chunk order (or by sort key): the order the cull pass keeps and
    /// the draws follow. An indexed node's runs go to its slice of the
    /// visible list, and its candidates get consecutive slots of the
    /// compacted index buffer, to be written by the next cull. Each slice
    /// keeps room to grow when nothing was left out.
    /// Returns the compacted index words and candidates left out for want
    /// of room.
    pub(super) fn lay_out(&mut self, gpu: &mut dyn Gpu) -> (u64, u64) {
        let limits = self.limits();
        let mut budget = Budget::of(&limits);
        // Runs that stay move from where the last layout put them.
        let old = std::mem::take(&mut self.laid);
        let old = |n: usize, node: &NodeEntry| {
            (old.len() == self.nodes.len())
                .then(|| &old[n])
                .filter(|o| o.shape == shape(node))
        };
        let plans: Vec<NodePlan> = (self.nodes.iter().enumerate())
            .map(|(n, node)| {
                let indexed = node.cull.draw == NodeDraw::Indexed;
                let (runs, vertex_words) = self.node_runs(node);
                let shift = if indexed { run_shift(vertex_words) } else { 0 };
                let old = old(n, node).and_then(|o| o.movable(shift));
                NodePlan::new(&runs, indexed, shift, &limits, &mut budget, old)
            })
            .collect();

        // Each slice's room to grow, from what the lists have spare.
        let grows = |plan: &NodePlan| match budget.dropped {
            0 => plan.growth(),
            _ => [0; 2],
        };
        let wanted = plans.iter().map(grows).fold([0u64; 2], |[r, w], [a, b]| {
            [r + u64::from(a), w + u64::from(b)]
        });
        let spare = [budget.rows as u64, u64::from(budget.words)];
        let scale = |grow: [u32; 2]| -> [u32; 2] {
            [0, 1].map(|i| match wanted[i] <= spare[i] {
                true => grow[i],
                false => (u64::from(grow[i]) * spare[i] / wanted[i]) as u32,
            })
        };
        let mut writes = Writes::default();
        let mut laid = Vec::with_capacity(self.nodes.len());
        let (mut rows_end, mut words_end) = (0u32, 0u32);
        for (node, plan) in self.nodes.iter().zip(plans) {
            let [grow_rows, grow_words] = scale(grows(&plan));
            let [rows, words] = [plan.rows.len() as u32 + grow_rows, plan.words + grow_words];
            let mut node_laid = LaidNode {
                shape: shape(node),
                row_base: rows_end,
                rows: Space::new(0, rows),
                word_base: words_end,
                words: Space::new(0, words),
                run_shift: 0,
                runs: None,
                numbers: Vec::new(),
                unwritten: false,
                live: 0,
            };
            plan.write(&mut node_laid, &mut writes);
            laid.push(node_laid);
            rows_end += rows;
            words_end += words;
        }
        self.candidates_dropped = budget.dropped;
        self.node_rows = vec![0; self.nodes.len() * NODE_CULL_WORDS];
        self.write_layout(gpu, laid, writes, rows_end, words_end);
        (budget.compacted_short, budget.candidates_short)
    }

    /// Lays out only what changed since the last layout; whether it could
    /// (otherwise nothing is written, and [`lay_out`](Self::lay_out) lays
    /// out everything). A node that keeps its runs where they are writes
    /// only the runs that came or went; any other node whose candidates
    /// changed is laid out again in its slices, or, past their room, in new
    /// ones past every node's (its old ones left unused until everything
    /// is laid out again). When it couldn't, the last layout stays for
    /// [`lay_out`](Self::lay_out) to move runs from, but for nodes laid out
    /// again, whose runs no longer say where their slots are.
    pub(super) fn lay_out_changes(&mut self, gpu: &mut dyn Gpu) -> bool {
        if self.nodes.is_empty()
            || self.laid.len() != self.nodes.len()
            || self.candidates_dropped > 0
        {
            return false;
        }
        let limits = self.limits();
        // Changed in place: a node that can't keep its runs is laid out
        // again whole, and a layout that can't be changed, from the start.
        let mut laid = std::mem::take(&mut self.laid);
        let mut writes = Writes::default();
        let (mut rows_end, mut words_end) = (self.rows_end, self.compacted_words);
        let mut relaid = Vec::new();
        let mut fits = true;
        for (n, (node, laid)) in self.nodes.iter().zip(&mut laid).enumerate() {
            if laid.shape != shape(node) {
                fits = false;
                break;
            }
            let indexed = node.cull.draw == NodeDraw::Indexed;
            let (runs, vertex_words) = self.node_runs(node);
            let shift = if indexed { run_shift(vertex_words) } else { 0 };
            if laid.runs.is_some() && shift == laid.run_shift {
                let mut node_writes = Writes::default();
                if laid.lay_out_changes(&runs, &limits, &mut node_writes) {
                    if !node_writes.moves.is_empty() {
                        relaid.push(n);
                    }
                    writes.append(node_writes);
                    continue;
                }
            }
            // Laid out again: in its slices, else in new ones past
            // everything's.
            let mut budget = Budget::unbounded();
            let old = laid.movable(shift);
            let plan = NodePlan::new(&runs, indexed, shift, &limits, &mut budget, old);
            let plan = match plan {
                _ if budget.dropped > 0 => {
                    fits = false;
                    break;
                }
                plan if plan.fits(laid) => plan,
                plan => {
                    let [rows, words] = plan.room();
                    if u64::from(rows_end) + u64::from(rows) > u64::from(limits.candidates)
                        || u64::from(words_end) + u64::from(words)
                            > u64::from(limits.compacted_words)
                    {
                        fits = false;
                        break;
                    }
                    (laid.row_base, laid.word_base) = (rows_end, words_end);
                    (laid.rows.cap, laid.words.cap) = (rows, words);
                    rows_end += rows;
                    words_end += words;
                    plan
                }
            };
            plan.write(laid, &mut writes);
            relaid.push(n);
        }
        if !fits {
            // Nothing was written: the nodes laid out again, and those
            // whose runs slid, have their slots where they were, which
            // their runs no longer say. Another node changed in place
            // keeps those of its runs that stay.
            for n in relaid {
                laid[n].runs = None;
            }
            self.laid = laid;
            return false;
        }
        self.write_layout(gpu, laid, writes, rows_end, words_end);
        true
    }

    /// The frame's cull ran: it wrote every slot of the nodes it culled
    /// (those not hidden), which can move from now on.
    pub(crate) fn culled(&mut self) {
        use layout::node_cull as nc;
        for (n, laid) in self.laid.iter_mut().enumerate() {
            let draw = self.node_rows.get(n * NODE_CULL_WORDS + nc::DRAW);
            if laid.unwritten && draw.is_some_and(|w| w & HIDDEN == 0) {
                laid.unwritten = false;
                for run in laid.runs.iter_mut().flat_map(|r| r.values_mut()) {
                    run.written = true;
                }
            }
        }
    }

    /// Writes a layout: what it writes of the lists, then the node rows.
    fn write_layout(
        &mut self,
        gpu: &mut dyn Gpu,
        laid: Vec<LaidNode>,
        mut writes: Writes,
        rows_end: u32,
        words_end: u32,
    ) {
        // Runs that went side by side hide in one write: each is a call
        // across to the GPU (to JavaScript, on the web).
        writes.hidden = joined(writes.hidden);
        let longest = writes.hidden.iter().map(|r| r.end - r.start).max();
        let words = vec![self.driver.hidden_index(); longest.unwrap_or(0) as usize];
        for range in &writes.hidden {
            gpu.write_buffer(
                self.buffers.compacted,
                u64::from(range.start) * 4,
                bytemuck::cast_slice(&words[..(range.end - range.start) as usize]),
            );
        }
        for (at, words) in &writes.rows {
            self.tables.write_words(gpu, Table::Candidates, *at, words);
        }
        for (at, words) in &writes.visible {
            self.tables.write_words(gpu, Table::Visible, *at, words);
        }
        let mut batches = 1u32;
        let mut live = 0u32;
        for (n, (node, laid)) in self.nodes.iter().zip(&laid).enumerate() {
            use layout::node_cull as nc;
            let row = &mut self.node_rows[n * NODE_CULL_WORDS..(n + 1) * NODE_CULL_WORDS];
            row[nc::VISIBLE_BASE] = laid.row_base;
            row[nc::CANDIDATE_COUNT] = laid.rows.end;
            row[nc::INDEX_BASE] = laid.word_base;
            row[nc::INDEX_COUNT] = laid.words.end;
            row[nc::RUN_SHIFT] = laid.run_shift;
            row[nc::MARGIN_PX] = node.cull.margin_px.to_bits();
            row[nc::DRAW] = node.cull.draw_word();
            row[nc::ELEVATION] = node.cull.elevation.to_bits();
            if node.cull.draw == NodeDraw::Indexed {
                batches = batches.max(laid.rows.end.div_ceil(INDEXED_BATCH));
            }
            live += laid.live;
        }
        self.tables
            .write_words(gpu, Table::NodeCull, 0, &self.node_rows);
        self.params_dirty = false;
        self.candidate_count = live;
        self.cull_batches = batches;
        self.rows_end = rows_end;
        self.compacted_words = words_end;
        self.laid = laid;
        self.hidden = writes.hidden;
        // Any layout earlier in the frame moved slots of a buffer made
        // again since, which holds nothing to move.
        self.moves = joined_moves(writes.moves);
    }

    /// The passes that make the last layout's moves in `buffer`, the
    /// scene's compacted indices or a copy of them: each move's words
    /// copied from where they were to where they go, all into the scratch
    /// buffer first, as one move's place can be another's old one. They
    /// run before any write to the slots they move from.
    pub(crate) fn move_passes(&mut self, gpu: &mut dyn Gpu, buffer: BufferId) -> Vec<Pass> {
        let words: u64 = self.moves.iter().map(|m| u64::from(m.words)).sum();
        if words == 0 {
            return Vec::new();
        }
        let scratch = self.scratch(gpu, words * 4);
        copies(buffer, scratch, &self.moves)
    }
}

/// The passes that make `moves` in `buffer` through `scratch`, which holds
/// at least everything moved.
fn copies(buffer: BufferId, scratch: BufferId, moves: &[Move]) -> Vec<Pass> {
    let copy = |source, from: u64, destination, to: u64, words: u32| Pass::CopyBuffer {
        source,
        source_offset: from * 4,
        destination,
        destination_offset: to * 4,
        size: u64::from(words) * 4,
    };
    let mut passes = Vec::with_capacity(moves.len() * 2);
    let mut at = 0;
    for m in moves {
        passes.push(copy(buffer, m.from.into(), scratch, at, m.words));
        at += u64::from(m.words);
    }
    let mut at = 0;
    for m in moves {
        passes.push(copy(scratch, at, buffer, m.to.into(), m.words));
        at += u64::from(m.words);
    }
    passes
}

/// `moves` in order, those that carry on from one another at both ends
/// joined.
fn joined_moves(mut moves: Vec<Move>) -> Vec<Move> {
    moves.sort_unstable_by_key(|m| m.from);
    let mut out: Vec<Move> = Vec::with_capacity(moves.len());
    for m in moves.into_iter().filter(|m| m.words > 0) {
        match out.last_mut() {
            Some(last) if last.from + last.words == m.from && last.to + last.words == m.to => {
                last.words += m.words;
            }
            _ => out.push(m),
        }
    }
    out
}

/// What a layout writes: candidate rows and visible words (at their first
/// word), compacted ranges to hide, and slots that move.
#[derive(Default)]
struct Writes {
    rows: Vec<(u32, Vec<u32>)>,
    visible: Vec<(u32, Vec<u32>)>,
    hidden: Vec<Range<u32>>,
    moves: Vec<Move>,
}

impl Writes {
    fn append(&mut self, mut other: Self) {
        self.rows.append(&mut other.rows);
        self.visible.append(&mut other.visible);
        self.hidden.append(&mut other.hidden);
        self.moves.append(&mut other.moves);
    }
}

/// `ranges` in order, those that touch or overlap joined.
fn joined(mut ranges: Vec<Range<u32>>) -> Vec<Range<u32>> {
    ranges.sort_unstable_by_key(|r| r.start);
    let mut out: Vec<Range<u32>> = Vec::with_capacity(ranges.len());
    for range in ranges.into_iter().filter(|r| !r.is_empty()) {
        match out.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => out.push(range),
        }
    }
    out
}

/// Room left in the lists as nodes are laid out, and what didn't fit.
struct Budget {
    rows: usize,
    words: u32,
    dropped: u32,
    compacted_short: u64,
    candidates_short: u64,
}

impl Budget {
    fn of(limits: &SubmitLimits) -> Self {
        Self {
            rows: limits.candidates as usize,
            words: limits.compacted_words,
            dropped: 0,
            compacted_short: 0,
            candidates_short: 0,
        }
    }

    fn unbounded() -> Self {
        Self {
            rows: usize::MAX,
            words: u32::MAX,
            dropped: 0,
            compacted_short: 0,
            candidates_short: 0,
        }
    }
}

/// Makes room for run `i` of `runs` at `want`, where the runs that stay
/// after it begin too soon: each slides along up to the next, as far as
/// the one before it needs, until the slots one would slide into are free
/// (a hole, or the slice past the last run). Whether they could: every run
/// that would slide has had its slots written. `slid` keeps where each
/// run's slots were before the first slide.
fn slide(
    space: &mut Space,
    kept: &mut FxHashMap<RunKey, LaidRun>,
    runs: &[RunPlan<'_>],
    next: &[Option<usize>],
    i: usize,
    want: Range<u32>,
    slid: &mut FxHashMap<usize, Range<u32>>,
) -> bool {
    let mut at = want.end;
    let mut sliding = Vec::new();
    let mut j = next[i];
    while let Some(k) = j {
        let laid = &kept[&runs[k].key];
        if laid.words.start >= at {
            break;
        }
        if !laid.written {
            return false;
        }
        sliding.push((k, at));
        at += laid.words.end - laid.words.start;
        j = next[k];
    }
    if j.is_none() && at > space.cap {
        return false;
    }
    for &(k, _) in &sliding {
        space.free(kept[&runs[k].key].words.clone());
    }
    space.take(want);
    for (k, to) in sliding {
        let laid = kept.get_mut(&runs[k].key).expect("a run that stays");
        let words = laid.words.clone();
        laid.words = to..to + (words.end - words.start);
        space.take(laid.words.clone());
        slid.entry(k).or_insert(words);
    }
    true
}

impl LaidNode {
    /// The runs that went and came since it was laid out, in its slices:
    /// whether they fit, leaving holes in no more than an eighth of its
    /// slots. A run that came with no room between the runs around it
    /// slides those after it along into the next hole, their slots moving
    /// (once a cull wrote them). When they don't fit, it is left part
    /// changed, but its runs still say where their slots are.
    fn lay_out_changes(
        &mut self,
        runs: &[RunPlan<'_>],
        limits: &SubmitLimits,
        writes: &mut Writes,
    ) -> bool {
        // Runs that slid, by where they were in `runs` and their slots
        // before.
        let mut slid = FxHashMap::default();
        let fits = self.change(runs, limits, writes, &mut slid);
        let kept = self.runs.as_mut().expect("a node that keeps its runs");
        for (k, words) in slid {
            let run = kept.get_mut(&runs[k].key).expect("a run that slid");
            if !fits {
                run.words = words;
                continue;
            }
            writes.moves.push(Move {
                from: self.word_base + words.start,
                to: self.word_base + run.words.start,
                words: words.end - words.start,
            });
            let mut rows = Vec::with_capacity(run.rows.len() * CANDIDATE_WORDS);
            let mut at = self.word_base + run.words.start;
            for (chunk, index_count, _) in runs[k].chunks() {
                rows.extend([chunk, runs[k].placement, at, run.run, MOVED]);
                at += index_count;
            }
            writes.rows.push((
                (self.row_base + run.rows.start) * CANDIDATE_WORDS as u32,
                rows,
            ));
        }
        fits
    }

    fn change(
        &mut self,
        runs: &[RunPlan<'_>],
        limits: &SubmitLimits,
        writes: &mut Writes,
        slid: &mut FxHashMap<usize, Range<u32>>,
    ) -> bool {
        let kept = self.runs.as_mut().expect("a node that keeps its runs");
        // Each listed run's slots, if it was laid out before.
        let stays: Vec<Option<Range<u32>>> = (runs.iter())
            .map(|r| kept.get(&r.key).map(|laid| laid.words.clone()))
            .collect();
        let staying = stays.iter().flatten().count();
        // Runs that went: their rows become holes, their slots hidden.
        let gone: Vec<RunKey> = if staying < kept.len() {
            let listed: FxHashSet<RunKey> = runs.iter().map(|r| r.key).collect();
            (kept.keys())
                .filter(|k| !listed.contains(k))
                .copied()
                .collect()
        } else {
            Vec::new()
        };
        for key in gone {
            let run = kept.remove(&key).expect("a laid run");
            let n = run.rows.end - run.rows.start;
            writes.rows.push((
                (self.row_base + run.rows.start) * CANDIDATE_WORDS as u32,
                std::iter::repeat_n([NO_CHUNK, 0, 0, 0, 0], n as usize)
                    .flatten()
                    .collect(),
            ));
            writes
                .hidden
                .push(self.word_base + run.words.start..self.word_base + run.words.end);
            self.rows.free(run.rows);
            self.words.free(run.words);
            self.numbers[run.run as usize] = false;
            self.live -= n;
        }
        // The runs that stay draw in the order they're listed in: their
        // slots must be in it already.
        if !stays.iter().flatten().map(|w| w.start).is_sorted() {
            return false;
        }
        // Runs that came: rows anywhere, a run number, and slots between
        // the runs that stay before and after it, which may slide along.
        let mut next = vec![None; runs.len()];
        let mut later = None;
        for (i, stay) in stays.iter().enumerate().rev() {
            next[i] = later;
            if stay.is_some() {
                later = Some(i);
            }
        }
        let mut before = 0;
        for (i, run) in runs.iter().enumerate() {
            if stays[i].is_some() {
                before = kept[&run.key].words.end;
                continue;
            }
            let after = next[i].map_or(u32::MAX, |j| kept[&runs[j].key].words.start);
            let n = run.slots.len() as u32;
            let words: u32 = run.chunks.iter().map(|c| c.index_count).sum();
            let number = match self.numbers.iter().position(|taken| !taken) {
                Some(free) => free as u32,
                None => {
                    self.numbers.push(false);
                    self.numbers.len() as u32 - 1
                }
            };
            if u64::from(number) >= limits.max_runs(self.run_shift) {
                return false;
            }
            let Some(row) = self.rows.alloc(n) else {
                return false;
            };
            let slot = match self.words.alloc_in(words, before, after) {
                Some(slot) => slot,
                None if slide(
                    &mut self.words,
                    kept,
                    runs,
                    &next,
                    i,
                    before..before + words,
                    slid,
                ) =>
                {
                    before
                }
                None => return false,
            };
            before = slot + words;
            self.numbers[number as usize] = true;
            let mut rows = Vec::with_capacity(n as usize * CANDIDATE_WORDS);
            let mut at = self.word_base + slot;
            for (chunk, index_count, _) in run.chunks() {
                rows.extend([chunk, run.placement, at, number, UNKNOWN]);
                at += index_count;
            }
            writes
                .rows
                .push(((self.row_base + row) * CANDIDATE_WORDS as u32, rows));
            writes.visible.push((
                (self.row_base + number) * VISIBLE_WORDS as u32,
                vec![run.first(), run.placement],
            ));
            kept.insert(
                run.key,
                LaidRun {
                    run: number,
                    rows: row..row + n,
                    words: slot..slot + words,
                    written: false,
                },
            );
            self.unwritten = true;
            self.live += n;
        }
        // Holes below the last run draw as hidden indices, every frame:
        // past an eighth of its slots, the node is laid out again whole
        // instead, its slots moving (on Liberty's Manhattan stutter path,
        // a quarter drew 5% more indices and moved 23% fewer words).
        let holes: u32 = self.words.holes.iter().map(|h| h.end - h.start).sum();
        holes <= self.words.end / 8
    }
}

/// A node's candidates laid out from the start of its slices, everything
/// in them written again but the slots that move.
struct NodePlan {
    /// Rows with slots from the start of its words.
    rows: Vec<[u32; CANDIDATE_WORDS]>,
    /// An indexed node's runs, by number (unused numbers zero), and which
    /// numbers are taken.
    runs: Vec<[u32; 2]>,
    numbers: Vec<bool>,
    words: u32,
    run_shift: u32,
    /// Its runs as laid out, for a node that keeps them where they are.
    kept: Option<FxHashMap<RunKey, LaidRun>>,
    /// Slots that move here: `to` from the start of its words.
    moves: Vec<Move>,
}

impl NodePlan {
    /// Its candidates in run order, or by sort key, as many as `budget`
    /// has room for and its indices can number runs. Runs of an unsorted
    /// indexed node that `old` laid out, and a cull wrote, keep their
    /// numbers and move their slots, when the whole node fits.
    fn new(
        runs: &[RunPlan<'_>],
        indexed: bool,
        run_shift: u32,
        limits: &SubmitLimits,
        budget: &mut Budget,
        old: Option<Movable<'_>>,
    ) -> Self {
        let mut candidates: Vec<_> = (runs.iter().enumerate())
            .flat_map(|(r, run)| {
                run.chunks()
                    .map(move |(chunk, count, key)| (key, chunk, run.placement, count, r as u32))
            })
            .collect();
        let sorted = candidates.iter().any(|c| c.0.is_some());
        if sorted {
            candidates.sort_by(|a, b| a.0.unwrap_or(0.0).total_cmp(&b.0.unwrap_or(0.0)));
        }
        let mut plan = Self {
            rows: Vec::with_capacity(candidates.len()),
            runs: Vec::new(),
            numbers: Vec::new(),
            words: 0,
            run_shift,
            kept: None,
            moves: Vec::new(),
        };
        let max_runs = limits.max_runs(run_shift);
        // Runs that move keep their numbers: below the node's candidates,
        // so they fit its visible slice, which nothing is left out of.
        let words: u64 = candidates.iter().map(|c| u64::from(c.3)).sum();
        let fits = candidates.len() <= budget.rows && words <= u64::from(budget.words);
        let below = (candidates.len() as u64).min(max_runs);
        let moving: Vec<Option<&LaidRun>> = match old {
            Some(old) if indexed && !sorted && fits => (runs.iter())
                .map(|run| {
                    let o = old.runs.get(&run.key)?;
                    (o.written && u64::from(o.run) < below).then_some(o)
                })
                .collect(),
            _ => vec![None; runs.len()],
        };
        let mut taken = vec![false; candidates.len()];
        for o in moving.iter().flatten() {
            taken[o.run as usize] = true;
        }
        // Other runs take the numbers left in order of first use by a kept
        // candidate: no more than the kept candidates, so they fit the
        // node's visible slice too.
        let mut free = (0..candidates.len() as u32)
            .filter(|&n| !taken[n as usize])
            .peekable();
        let mut renumbered = vec![u32::MAX; runs.len()];
        let mut laid: Vec<Option<LaidRun>> = vec![None; runs.len()];
        let dropped = budget.dropped;
        for (_, chunk, tile, index_count, run) in candidates {
            // The cull's copy stops at CHUNK_INDICES.
            debug_assert!(!indexed || index_count <= orb_layout::CHUNK_INDICES);
            let room = index_count <= budget.words;
            let numbered = renumbered[run as usize] != u32::MAX
                || moving[run as usize].is_some()
                || free.peek().is_some_and(|&n| u64::from(n) < max_runs);
            if budget.rows == 0 || (indexed && !(numbered && room)) {
                if indexed && !room {
                    budget.compacted_short += u64::from(index_count);
                }
                if budget.rows == 0 {
                    budget.candidates_short += 1;
                }
                budget.dropped += 1;
                continue;
            }
            budget.rows -= 1;
            let row = plan.rows.len() as u32;
            if !indexed {
                plan.rows.push([chunk, tile, 0, 0, 0]);
                continue;
            }
            budget.words -= index_count;
            let slot = plan.words;
            let moved = moving[run as usize];
            let number = &mut renumbered[run as usize];
            if *number == u32::MAX {
                *number = match moved {
                    Some(o) => {
                        // Its candidates follow one another here as they
                        // did there.
                        plan.moves.push(Move {
                            from: old.map_or(0, |m| m.word_base) + o.words.start,
                            to: slot,
                            words: o.words.end - o.words.start,
                        });
                        o.run
                    }
                    None => free.next().expect("a number was free"),
                };
                let at = *number as usize;
                if plan.runs.len() <= at {
                    plan.runs.resize(at + 1, [0; 2]);
                    plan.numbers.resize(at + 1, false);
                }
                plan.runs[at] = [runs[run as usize].first(), tile];
                plan.numbers[at] = true;
            }
            let drawn = if moved.is_some() { MOVED } else { UNKNOWN };
            plan.rows.push([chunk, tile, slot, *number, drawn]);
            plan.words += index_count;
            let laid = laid[run as usize].get_or_insert(LaidRun {
                run: *number,
                rows: row..row,
                words: slot..slot,
                written: moved.is_some(),
            });
            laid.rows.end = row + 1;
            laid.words.end = plan.words;
        }
        if indexed && !sorted && budget.dropped == dropped {
            plan.kept = Some(
                (runs.iter().zip(laid))
                    .filter_map(|(run, laid)| Some((run.key, laid?)))
                    .collect(),
            );
        }
        plan
    }

    /// Whether it fits a node's slices.
    fn fits(&self, laid: &LaidNode) -> bool {
        self.rows.len() as u32 <= laid.rows.cap && self.words <= laid.words.cap
    }

    /// Rows and words its slices keep past it, to grow into.
    fn growth(&self) -> [u32; 2] {
        let step = match self.kept {
            Some(_) => orb_layout::CHUNK_INDICES,
            None => 0,
        };
        [
            headroom(self.rows.len() as u32, 8),
            headroom(self.words, step),
        ]
    }

    /// Rows and words of new slices for it, with room to grow.
    fn room(&self) -> [u32; 2] {
        let [rows, words] = self.growth();
        [self.rows.len() as u32 + rows, self.words + words]
    }

    /// Writes it into a node's slices.
    fn write(self, laid: &mut LaidNode, writes: &mut Writes) {
        let count = self.rows.len() as u32;
        let indexed = laid.shape.0 == NodeDraw::Indexed;
        // Slots already where they go stay.
        let moves = self.moves.iter().map(|m| Move {
            to: laid.word_base + m.to,
            ..*m
        });
        writes.moves.extend(moves.filter(|m| m.from != m.to));
        let rows: Vec<u32> = (self.rows.into_iter())
            .flat_map(|mut row| {
                if indexed {
                    row[2] += laid.word_base;
                }
                row
            })
            .collect();
        if !rows.is_empty() {
            writes
                .rows
                .push((laid.row_base * CANDIDATE_WORDS as u32, rows));
        }
        if !self.runs.is_empty() {
            writes.visible.push((
                laid.row_base * VISIBLE_WORDS as u32,
                self.runs.into_iter().flatten().collect(),
            ));
        }
        laid.rows = Space::new(count, laid.rows.cap);
        laid.words = Space::new(self.words, laid.words.cap);
        laid.numbers = self.numbers;
        laid.run_shift = self.run_shift;
        laid.unwritten = (self.kept.iter().flat_map(|k| k.values())).any(|r| !r.written);
        laid.runs = self.kept;
        laid.live = count;
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// A run of `chunks` chunks of 3 indices each, mesh `serial`.
    fn run(serial: u64, chunks: u32) -> RunPlan<'static> {
        let chunk = Chunk {
            first_index: 0,
            index_count: 3,
            first_vertex: 0,
            bounds: orb_layout::TileBox {
                min: [0; 2],
                max: [0; 2],
            },
            sort_key: None,
        };
        RunPlan {
            key: (serial, 0),
            placement: 0,
            slots: Vec::leak((0..chunks).map(|c| serial as u32 * 100 + c).collect()),
            chunks: Vec::leak(vec![chunk; chunks as usize]),
        }
    }

    /// An indexed node laid out from `runs`, with room past them.
    fn laid(runs: &[RunPlan<'_>], limits: &SubmitLimits) -> LaidNode {
        let plan = NodePlan::new(runs, true, 8, limits, &mut Budget::unbounded(), None);
        let [rows, words] = plan.room();
        let mut node = LaidNode {
            shape: (NodeDraw::Indexed, PlacementTarget::Map, false, false),
            row_base: 0,
            rows: Space::new(0, rows),
            word_base: 0,
            words: Space::new(0, words),
            run_shift: 8,
            runs: None,
            numbers: Vec::new(),
            unwritten: false,
            live: 0,
        };
        plan.write(&mut node, &mut Writes::default());
        node
    }

    fn slots(node: &LaidNode, serial: u64) -> Range<u32> {
        node.runs.as_ref().unwrap()[&(serial, 0)].words.clone()
    }

    #[test]
    fn runs_that_stay_keep_their_slots_in_draw_order() {
        let limits = SubmitLimits {
            index_bits: 32,
            compacted_words: 1 << 20,
            candidates: 1 << 20,
        };
        // Runs after those the test changes, so that one going leaves
        // holes in under an eighth of the node's slots.
        let tail = |runs: &[RunPlan<'static>]| -> Vec<RunPlan<'static>> {
            let mut runs = runs.to_vec();
            runs.extend((10..15).map(|serial| run(serial, 2)));
            runs
        };
        let runs = tail(&[run(1, 2), run(2, 2), run(3, 2), run(7, 2), run(8, 2)]);
        let mut node = laid(&runs, &limits);
        assert_eq!(slots(&node, 2), 6..12);
        // The second run goes: its rows are holes, its slots hidden, and
        // the others stay.
        let mut writes = Writes::default();
        let runs = tail(&[run(1, 2), run(3, 2), run(7, 2), run(8, 2)]);
        assert!(node.lay_out_changes(&runs, &limits, &mut writes));
        assert_eq!(writes.hidden, vec![6..12]);
        assert_eq!(writes.rows[0].1[..CANDIDATE_WORDS], [NO_CHUNK, 0, 0, 0, 0]);
        assert_eq!((slots(&node, 1), slots(&node, 3)), (0..6, 12..18));
        // A run between them takes the hole; one after them, the room past
        // the last; neither moves a run that stays.
        let mut writes = Writes::default();
        let mut runs = tail(&[run(1, 2), run(4, 1), run(3, 2), run(7, 2), run(8, 2)]);
        runs.push(run(5, 2));
        assert!(node.lay_out_changes(&runs, &limits, &mut writes));
        assert_eq!((slots(&node, 4), slots(&node, 5)), (6..9, 60..66));
        assert_eq!((slots(&node, 1), slots(&node, 3)), (0..6, 12..18));
        assert!(writes.hidden.is_empty());
        let fits = |runs: &[RunPlan<'_>]| {
            node.clone()
                .lay_out_changes(runs, &limits, &mut Writes::default())
        };
        // One with no room between the runs around it can't come this way
        // while their slots wait for a cull.
        let mut more = runs.to_vec();
        more.insert(0, run(6, 1));
        assert!(!fits(&more));
        // Once written, those after it slide along into the next hole,
        // their rows moved, and the runs past it stay.
        let mut written = node.clone();
        for run in written.runs.as_mut().unwrap().values_mut() {
            run.written = true;
        }
        let mut writes = Writes::default();
        assert!(written.lay_out_changes(&more, &limits, &mut writes));
        let at = |serial| slots(&written, serial);
        assert_eq!([at(6), at(1), at(4), at(3)], [0..3, 3..9, 9..12, 12..18]);
        assert_eq!(
            joined_moves(writes.moves),
            [Move {
                from: 0,
                to: 3,
                words: 9
            }]
        );
        let moved: Vec<u32> = (writes.rows.iter())
            .flat_map(|(_, rows)| rows.chunks(CANDIDATE_WORDS).map(|r| r[4]))
            .collect();
        assert_eq!(moved.iter().filter(|&&d| d == MOVED).count(), 3);
        assert!(written.words.holes.is_empty());
        // Unless they'd slide past the slice.
        let mut tight = written.clone();
        tight.words.cap = tight.words.end;
        let mut first = more.to_vec();
        first.insert(0, run(9, 1));
        assert!(!tight.lay_out_changes(&first, &limits, &mut Writes::default()));
        assert_eq!(slots(&tight, 1), 3..9, "a run that couldn't slide stays");
        // Nor can runs that stay but draw in another order.
        assert!(!fits(&tail(&[
            run(3, 2),
            run(1, 2),
            run(4, 1),
            run(7, 2),
            run(8, 2)
        ])));
        // Nor so many going that holes fill over an eighth of its slots.
        assert!(!fits(&[run(1, 2), run(5, 2)]));
    }

    #[test]
    fn a_node_laid_out_again_moves_the_slots_a_cull_wrote() {
        let limits = SubmitLimits {
            index_bits: 32,
            compacted_words: 1 << 20,
            candidates: 1 << 20,
        };
        let mut node = laid(&[run(1, 2), run(2, 2), run(3, 1)], &limits);
        node.word_base = 100;
        // Not yet written by a cull: nothing moves.
        let again = |node: &LaidNode, runs: &[RunPlan<'_>]| {
            NodePlan::new(
                runs,
                true,
                8,
                &limits,
                &mut Budget::unbounded(),
                node.movable(8),
            )
        };
        let runs = [run(4, 1), run(1, 2), run(3, 1)];
        assert!(again(&node, &runs).moves.is_empty());
        for run in node.runs.as_mut().unwrap().values_mut() {
            run.written = true;
        }
        // A run before the others: those that stay move behind it with
        // their numbers, and only its rows wait for the cull.
        let plan = again(&node, &runs);
        let number = |serial| node.runs.as_ref().unwrap()[&(serial, 0)].run;
        assert_eq!(
            plan.moves,
            [
                Move {
                    from: 100,
                    to: 3,
                    words: 6
                },
                Move {
                    from: 112,
                    to: 9,
                    words: 3
                },
            ]
        );
        let drawn: Vec<[u32; 2]> = plan.rows.iter().map(|r| [r[3], r[4]]).collect();
        assert_eq!(
            drawn,
            [
                [1, UNKNOWN],
                [number(1), MOVED],
                [number(1), MOVED],
                [number(3), MOVED],
            ]
        );
        assert_eq!(plan.numbers, [true; 3]);
        // Indices that number runs another way move nothing.
        assert!(node.movable(9).is_none());
    }

    #[test]
    fn moves_join_where_they_carry_on() {
        let m = |from, to, words| Move { from, to, words };
        assert_eq!(
            joined_moves(vec![m(10, 0, 5), m(0, 20, 10), m(15, 5, 2), m(30, 7, 0)]),
            [m(0, 20, 10), m(10, 0, 7)]
        );
    }

    #[test]
    fn hidden_ranges_join_where_they_touch() {
        assert_eq!(
            joined(vec![12..18, 0..6, 6..12, 30..30, 20..24, 22..26]),
            vec![0..18, 20..26]
        );
    }

    #[test]
    fn space_takes_the_first_hole_that_fits_then_room_past_the_end() {
        let mut s = Space::new(0, 100);
        assert_eq!(s.alloc(10), Some(0));
        assert_eq!(s.alloc(20), Some(10));
        assert_eq!(s.alloc(5), Some(30));
        s.free(10..30);
        assert_eq!(s.holes, vec![10..30]);
        assert_eq!(s.alloc(25), Some(35));
        assert_eq!(s.alloc(8), Some(10));
        assert_eq!(s.alloc(12), Some(18));
        assert!(s.holes.is_empty());
        assert_eq!(s.alloc(41), None);
        assert_eq!(s.alloc(40), Some(60));
    }

    #[test]
    fn space_finds_room_between_bounds() {
        let mut s = Space::new(0, 100);
        for _ in 0..5 {
            s.alloc(10);
        }
        s.free(10..20);
        s.free(30..40);
        // Not in the hole below the bound, nor in one too small for it.
        assert_eq!(s.alloc_in(10, 20, 30), None);
        assert_eq!(s.alloc_in(4, 25, 50), Some(30));
        assert_eq!(s.holes, vec![10..20, 34..40]);
        assert_eq!(s.alloc_in(4, 12, 18), Some(12));
        assert_eq!(s.holes, vec![10..12, 16..20, 34..40]);
        // Past the end only when nothing lies after it.
        assert_eq!(s.alloc_in(10, 40, 50), None);
        assert_eq!(s.alloc_in(10, 40, u32::MAX), Some(50));
        assert_eq!(s.alloc_in(41, 60, u32::MAX), None);
    }

    #[test]
    fn space_joins_holes_and_gives_back_its_end() {
        let mut s = Space::new(0, 64);
        for _ in 0..4 {
            s.alloc(8);
        }
        s.free(8..16);
        s.free(24..32);
        assert_eq!(s.end, 24);
        assert_eq!(s.holes, vec![8..16]);
        s.free(16..24);
        assert_eq!(s.end, 8);
        assert!(s.holes.is_empty());
        s.free(0..8);
        assert_eq!(s.end, 0);
    }
}
