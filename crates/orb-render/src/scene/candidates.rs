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
//!   antialiased edges meet on one pixel. Only those runs' rows are
//!   written, and the next cull writes only their slots.
//! - A node whose changes don't fit that way (a run with no room in its
//!   place, holes in over a quarter of its slots, which draw as hidden
//!   indices, a listed or sorted node) is laid out again within its
//!   slices, or in new ones past every node's when it outgrew them.
//! - Each slice has room to grow past what it holds. When a node's runs
//!   outnumber what its indices can address, or the nodes or lists
//!   themselves change, everything is laid out again from the start.

use std::collections::HashMap;
use std::ops::Range;

use orb_hal::Gpu;
use orb_shaders::layout;

use crate::driver::SubmitLimits;

use super::{
    CANDIDATE_WORDS, INDEXED_BATCH, NODE_CULL_WORDS, NodeDraw, NodeEntry, PlacementTarget, Scene,
    Table, UNKNOWN, VISIBLE_WORDS,
};

/// A candidate row that holds no candidate: a hole a run left in an
/// indexed node's rows, which the cull skips (`cull.wgsl`'s `NO_CHUNK`).
pub(crate) const NO_CHUNK: u32 = u32::MAX;

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
            // What's left of the hole on either side.
            let (before, after) = (hole.start..at, at + n..hole.end);
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
            return Some(at);
        }
        let at = self.end.max(lo);
        if at != self.end || u64::from(at) + u64::from(n) > u64::from(self.cap.min(hi)) {
            return None;
        }
        self.end += n;
        Some(at)
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

/// A run as laid out: its number and its rows and slots in the node's
/// slices.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LaidRun {
    run: u32,
    rows: Range<u32>,
    words: Range<u32>,
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
    runs: Option<HashMap<RunKey, LaidRun>>,
    numbers: Vec<bool>,
    /// Its candidates, holes aside.
    pub(super) live: u32,
}

/// A run to lay out: a mesh's chunks in one placement.
#[derive(Clone)]
struct RunPlan {
    key: RunKey,
    /// The mesh's first chunk slot, which its run row names.
    first: u32,
    placement: u32,
    /// Its chunks: slot, index count, sort key.
    chunks: Vec<(u32, u32, Option<f32>)>,
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
    fn node_runs(&self, node: &NodeEntry) -> (Vec<RunPlan>, u32) {
        let mut runs = Vec::new();
        let mut vertex_words = 1u32;
        for mesh in &node.meshes {
            let entry = self.meshes[mesh.0 as usize]
                .as_ref()
                .expect("node lists live meshes");
            let tile = self.tiles[entry.tile.0 as usize]
                .as_ref()
                .expect("meshes have tiles");
            let Some(&first) = entry.chunk_slots.first() else {
                continue;
            };
            vertex_words = vertex_words.max(entry.vertices.len() as u32);
            for placement in &tile.placements {
                if !node.cull.draws(placement.row.target) {
                    continue;
                }
                runs.push(RunPlan {
                    key: (self.mesh_serials[mesh.0 as usize], placement.slot),
                    first,
                    placement: placement.slot,
                    chunks: (entry.chunk_slots.iter().zip(&entry.chunks))
                        .map(|(&chunk, record)| (chunk, record.index_count, record.sort_key))
                        .collect(),
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
        let plans: Vec<NodePlan> = (self.nodes.iter())
            .map(|node| {
                let indexed = node.cull.draw == NodeDraw::Indexed;
                let (runs, vertex_words) = self.node_runs(node);
                let shift = if indexed { run_shift(vertex_words) } else { 0 };
                NodePlan::new(&runs, indexed, shift, &limits, &mut budget)
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
    /// is laid out again).
    pub(super) fn lay_out_changes(&mut self, gpu: &mut dyn Gpu) -> bool {
        if self.nodes.is_empty()
            || self.laid.len() != self.nodes.len()
            || self.candidates_dropped > 0
        {
            return false;
        }
        let limits = self.limits();
        let mut laid = self.laid.clone();
        let mut writes = Writes::default();
        let (mut rows_end, mut words_end) = (self.rows_end, self.compacted_words);
        for (node, laid) in self.nodes.iter().zip(&mut laid) {
            if laid.shape != shape(node) {
                return false;
            }
            let indexed = node.cull.draw == NodeDraw::Indexed;
            let (runs, vertex_words) = self.node_runs(node);
            let shift = if indexed { run_shift(vertex_words) } else { 0 };
            if laid.runs.is_some() && shift == laid.run_shift {
                let mut changed = laid.clone();
                let mut node_writes = Writes::default();
                if changed.lay_out_changes(&runs, &limits, &mut node_writes) {
                    *laid = changed;
                    writes.append(node_writes);
                    continue;
                }
            }
            // Laid out again: in its slices, else in new ones past
            // everything's.
            let mut budget = Budget::unbounded();
            let plan = NodePlan::new(&runs, indexed, shift, &limits, &mut budget);
            let plan = match plan {
                _ if budget.dropped > 0 => return false,
                plan if plan.fits(laid) => plan,
                plan => {
                    let [rows, words] = plan.room();
                    if u64::from(rows_end) + u64::from(rows) > u64::from(limits.candidates)
                        || u64::from(words_end) + u64::from(words)
                            > u64::from(limits.compacted_words)
                    {
                        return false;
                    }
                    (laid.row_base, laid.word_base) = (rows_end, words_end);
                    (laid.rows.cap, laid.words.cap) = (rows, words);
                    rows_end += rows;
                    words_end += words;
                    plan
                }
            };
            plan.write(laid, &mut writes);
        }
        self.write_layout(gpu, laid, writes, rows_end, words_end);
        true
    }

    /// Writes a layout: what it writes of the lists, then the node rows.
    fn write_layout(
        &mut self,
        gpu: &mut dyn Gpu,
        laid: Vec<LaidNode>,
        writes: Writes,
        rows_end: u32,
        words_end: u32,
    ) {
        let hidden = self.driver.hidden_index();
        for range in &writes.hidden {
            let words = vec![hidden; (range.end - range.start) as usize];
            gpu.write_buffer(
                self.buffers.compacted,
                u64::from(range.start) * 4,
                bytemuck::cast_slice(&words),
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
    }
}

/// What a layout of changes writes: candidate rows and visible words (at
/// their first word), and compacted ranges to hide.
#[derive(Default)]
struct Writes {
    rows: Vec<(u32, Vec<u32>)>,
    visible: Vec<(u32, Vec<u32>)>,
    hidden: Vec<Range<u32>>,
}

impl Writes {
    fn append(&mut self, mut other: Self) {
        self.rows.append(&mut other.rows);
        self.visible.append(&mut other.visible);
        self.hidden.append(&mut other.hidden);
    }
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

impl LaidNode {
    /// The runs that went and came since it was laid out, in its slices:
    /// whether they fit, leaving holes in no more than a quarter of its
    /// slots (else it is left part changed).
    fn lay_out_changes(
        &mut self,
        runs: &[RunPlan],
        limits: &SubmitLimits,
        writes: &mut Writes,
    ) -> bool {
        let kept = self.runs.as_mut().expect("a node that keeps its runs");
        let wanted: HashMap<RunKey, &RunPlan> = runs.iter().map(|r| (r.key, r)).collect();
        // Runs that went: their rows become holes, their slots hidden.
        let gone: Vec<RunKey> = kept
            .keys()
            .filter(|k| !wanted.contains_key(k))
            .copied()
            .collect();
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
        let stays = runs.iter().filter_map(|r| kept.get(&r.key));
        if !stays.map(|r| r.words.start).is_sorted() {
            return false;
        }
        // Runs that came: rows anywhere, a run number, and slots between
        // the runs before and after it.
        let mut before = 0;
        for (i, run) in runs.iter().enumerate() {
            if let Some(laid) = kept.get(&run.key) {
                before = laid.words.end;
                continue;
            }
            let after = (runs[i + 1..].iter())
                .find_map(|r| kept.get(&r.key))
                .map_or(u32::MAX, |r| r.words.start);
            let n = run.chunks.len() as u32;
            let words: u32 = run.chunks.iter().map(|c| c.1).sum();
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
            let (Some(row), Some(slot)) = (
                self.rows.alloc(n),
                self.words.alloc_in(words, before, after),
            ) else {
                return false;
            };
            before = slot + words;
            self.numbers[number as usize] = true;
            let mut rows = Vec::with_capacity(n as usize * CANDIDATE_WORDS);
            let mut at = self.word_base + slot;
            for &(chunk, index_count, _) in &run.chunks {
                rows.extend([chunk, run.placement, at, number, UNKNOWN]);
                at += index_count;
            }
            writes
                .rows
                .push(((self.row_base + row) * CANDIDATE_WORDS as u32, rows));
            writes.visible.push((
                (self.row_base + number) * VISIBLE_WORDS as u32,
                vec![run.first, run.placement],
            ));
            kept.insert(
                run.key,
                LaidRun {
                    run: number,
                    rows: row..row + n,
                    words: slot..slot + words,
                },
            );
            self.live += n;
        }
        // Holes below the last run draw as hidden indices: past a quarter
        // of its slots, the node is laid out again whole instead.
        let holes: u32 = self.words.holes.iter().map(|h| h.end - h.start).sum();
        holes <= self.words.end / 4
    }
}

/// A node's candidates laid out from the start of its slices, everything
/// in them written again.
struct NodePlan {
    /// Rows with slots from the start of its words.
    rows: Vec<[u32; CANDIDATE_WORDS]>,
    /// An indexed node's runs, by number.
    runs: Vec<[u32; 2]>,
    words: u32,
    run_shift: u32,
    /// Its runs as laid out, for a node that keeps them where they are.
    kept: Option<HashMap<RunKey, LaidRun>>,
}

impl NodePlan {
    /// Its candidates in run order, or by sort key, as many as `budget`
    /// has room for and its indices can number runs.
    fn new(
        runs: &[RunPlan],
        indexed: bool,
        run_shift: u32,
        limits: &SubmitLimits,
        budget: &mut Budget,
    ) -> Self {
        let mut candidates: Vec<_> = (runs.iter().enumerate())
            .flat_map(|(r, run)| {
                (run.chunks.iter())
                    .map(move |&(chunk, count, key)| (key, chunk, run.placement, count, r as u32))
            })
            .collect();
        let sorted = candidates.iter().any(|c| c.0.is_some());
        if sorted {
            candidates.sort_by(|a, b| a.0.unwrap_or(0.0).total_cmp(&b.0.unwrap_or(0.0)));
        }
        let mut plan = Self {
            rows: Vec::with_capacity(candidates.len()),
            runs: Vec::new(),
            words: 0,
            run_shift,
            kept: None,
        };
        // Runs in order of first use by a kept candidate: no more than
        // the kept candidates, so they fit the node's visible slice.
        let mut renumbered = vec![u32::MAX; runs.len()];
        let mut laid: Vec<Option<LaidRun>> = vec![None; runs.len()];
        let max_runs = limits.max_runs(run_shift);
        let dropped = budget.dropped;
        for (_, chunk, tile, index_count, run) in candidates {
            // The cull's copy stops at CHUNK_INDICES.
            debug_assert!(!indexed || index_count <= orb_layout::CHUNK_INDICES);
            let room = index_count <= budget.words;
            let numbered =
                renumbered[run as usize] != u32::MAX || (plan.runs.len() as u64) < max_runs;
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
            let number = &mut renumbered[run as usize];
            if *number == u32::MAX {
                *number = plan.runs.len() as u32;
                plan.runs.push([runs[run as usize].first, tile]);
            }
            let slot = plan.words;
            plan.rows.push([chunk, tile, slot, *number, UNKNOWN]);
            plan.words += index_count;
            let laid = laid[run as usize].get_or_insert(LaidRun {
                run: *number,
                rows: row..row,
                words: slot..slot,
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
        let numbers = self.runs.len();
        let indexed = laid.shape.0 == NodeDraw::Indexed;
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
        laid.numbers = vec![true; numbers];
        laid.run_shift = self.run_shift;
        laid.runs = self.kept;
        laid.live = count;
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// A run of `chunks` chunks of 3 indices each, mesh `serial`.
    fn run(serial: u64, chunks: u32) -> RunPlan {
        RunPlan {
            key: (serial, 0),
            first: serial as u32 * 100,
            placement: 0,
            chunks: (0..chunks)
                .map(|c| (serial as u32 * 100 + c, 3, None))
                .collect(),
        }
    }

    /// An indexed node laid out from `runs`, with room past them.
    fn laid(runs: &[RunPlan], limits: &SubmitLimits) -> LaidNode {
        let plan = NodePlan::new(runs, true, 8, limits, &mut Budget::unbounded());
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
        let runs = [run(1, 2), run(2, 2), run(3, 2), run(7, 2), run(8, 2)];
        let mut node = laid(&runs, &limits);
        assert_eq!(slots(&node, 2), 6..12);
        // The second run goes: its rows are holes, its slots hidden, and
        // the others stay.
        let mut writes = Writes::default();
        let runs = [run(1, 2), run(3, 2), run(7, 2), run(8, 2)];
        assert!(node.lay_out_changes(&runs, &limits, &mut writes));
        assert_eq!(writes.hidden, vec![6..12]);
        assert_eq!(writes.rows[0].1[..CANDIDATE_WORDS], [NO_CHUNK, 0, 0, 0, 0]);
        assert_eq!((slots(&node, 1), slots(&node, 3)), (0..6, 12..18));
        // A run between them takes the hole; one after them, the room past
        // the last; neither moves a run that stays.
        let mut writes = Writes::default();
        let runs = [
            run(1, 2),
            run(4, 1),
            run(3, 2),
            run(7, 2),
            run(8, 2),
            run(5, 2),
        ];
        assert!(node.lay_out_changes(&runs, &limits, &mut writes));
        assert_eq!((slots(&node, 4), slots(&node, 5)), (6..9, 30..36));
        assert_eq!((slots(&node, 1), slots(&node, 3)), (0..6, 12..18));
        assert!(writes.hidden.is_empty());
        let fits = |runs: &[RunPlan]| {
            node.clone()
                .lay_out_changes(runs, &limits, &mut Writes::default())
        };
        // One with no room between the runs around it can't come this way.
        let mut more = runs.to_vec();
        more.insert(0, run(6, 3));
        assert!(!fits(&more));
        // Nor can runs that stay but draw in another order.
        assert!(!fits(&[
            run(3, 2),
            run(1, 2),
            run(4, 1),
            run(7, 2),
            run(8, 2),
            run(5, 2)
        ]));
        // Nor so many going that holes fill over a quarter of its slots.
        assert!(!fits(&[run(1, 2), run(5, 2)]));
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
