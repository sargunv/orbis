//! Feature state: per source and source layer, feature id →
//! state index; the states set on them; and the GPU table the shaders read
//! through each mesh's slot → index remap.
//!
//! Every feature of a resident mesh whose node reads state gets an index,
//! counted by the meshes that refer to it, so a state write rewrites one
//! table row and no mesh. A row holds, per node that reads state, a flags
//! word (whether the feature has state, and whether what it shows is
//! fading), the frame time a fade started, and per property that reads
//! state its values with the feature's state at [`STATE_ZOOMS`] integer
//! zooms around the camera's, from five below its own to two above (drape
//! slots read their own zoom's value, so they draw alike whatever the
//! camera's), what it showed before the fade at the same zooms, and the
//! property's transition. The layout is compiled per style revision.
//! Index 0 and rows with zero flags draw the value with no state: the
//! node's constant, or for paint that also reads feature data, the
//! feature's property row. That paint evaluates here from the feature data
//! its meshes captured, so a state write shows in the next frame, as GL
//! JS's `updatePaintArrays`.
//!
//! Beyond GL JS and Native, which snap, a state change fades: by each
//! property's transition, from what the feature showed when the frame
//! after the change began, to its new value, eased in the shaders by the
//! frame's time. Nothing runs per frame on the CPU while it fades; once it
//! ends the feature draws as one that always had its state.

use std::collections::HashMap;
use std::ops::Range;

use orb_expr::FeatureData;
use orb_hal::Gpu;
use orb_render::{Renderer, StateLookup};
use orb_style::FeatureStateOp;
use orb_style::Value;

use crate::style::{PaintProperty, StylePlan};

/// Integer zooms a state row holds each value at (`draw.wgsl`'s).
const STATE_ZOOMS: u32 = 8;
/// Words per property in a node's block (`draw.wgsl`'s): its values,
/// what it showed before a fade, and the fade's duration and delay.
const SLOT_WORDS: u32 = 2 * STATE_ZOOMS + 2;
/// A block's flags (`draw.wgsl`'s): the feature has state; what it shows
/// fades.
const HAS: u32 = 1;
const FADING: u32 = 2;
/// Zooms below the camera's (floored) that a state row starts at: drape
/// slots of terrain tiles farther out draw a coarser zoom.
const STATE_ZOOMS_BELOW: f64 = 5.0;

/// A feature's state, and the data paint mixing state and data reads.
pub(crate) type FeatureState<'a> =
    Option<(&'a [(String, Value<'static>)], Option<&'a FeatureData>)>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StateKey {
    source: u32,
    layer: Option<String>,
    id: String,
}

#[derive(Debug, Default)]
struct Entry {
    key: Option<StateKey>,
    state: Vec<(String, Value<'static>)>,
    /// What paint reading both state and data reads of the feature.
    data: Option<FeatureData>,
    /// Mesh feature slots that refer to the entry.
    refs: u32,
    /// Its row on the GPU, while it holds anything.
    row: Option<Row>,
    /// Listed in `dirty`, to write at the next prepare.
    dirty: bool,
}

/// A state row as last written.
#[derive(Debug)]
struct Row {
    /// The state its values were evaluated with.
    state: Vec<(String, Value<'static>)>,
    /// Its words; empty once the layout they were in is gone.
    words: Vec<u32>,
}

/// One node's block of a state row.
#[derive(Debug)]
struct Block {
    /// Word offset within the row.
    offset: u32,
    mask: u32,
    source: u32,
    layer: Option<String>,
    /// The masked property slots, in order.
    slots: Vec<usize>,
    /// Each one's transition: how a state change fades.
    transitions: Vec<orb_style::Transition>,
}

#[derive(Debug)]
pub(crate) struct FeatureStates {
    ids: HashMap<StateKey, u32>,
    entries: Vec<Entry>,
    free: Vec<u32>,
    /// Entries to write at the next prepare, each once: see `Entry::dirty`.
    dirty: Vec<u32>,
    all_dirty: bool,
    /// The plan revision the layout is compiled for.
    revision: Option<u64>,
    /// Per node index, its block.
    blocks: Vec<Option<Block>>,
    stride: u32,
    zoom_dependent: bool,
    /// The first of the zooms the rows hold values at.
    zoom: f64,
    table: Option<Range<u32>>,
    /// Rows of freed entries to clear, whose fade the next feature to take
    /// the index must not inherit.
    zero: Vec<u32>,
    /// The frame time the last fade ends at.
    fading_until: f64,
    /// Rows written to the GPU, for counters.
    pub(crate) rows_written: u64,
    /// State operations applied.
    writes: u64,
}

impl Default for FeatureStates {
    fn default() -> Self {
        Self {
            ids: HashMap::new(),
            // Index 0 is the default state.
            entries: vec![Entry::default()],
            free: Vec::new(),
            dirty: Vec::new(),
            all_dirty: false,
            revision: None,
            blocks: Vec::new(),
            stride: 0,
            zoom_dependent: false,
            zoom: f64::NAN,
            table: None,
            zero: Vec::new(),
            fading_until: f64::NEG_INFINITY,
            rows_written: 0,
            writes: 0,
        }
    }
}

impl FeatureStates {
    fn index(&mut self, key: StateKey) -> u32 {
        if let Some(&index) = self.ids.get(&key) {
            return index;
        }
        let mut entry = Entry {
            key: Some(key.clone()),
            ..Entry::default()
        };
        let index = match self.free.pop() {
            Some(index) => {
                // A freed entry can still be listed in `dirty`.
                entry.dirty = self.entries[index as usize].dirty;
                self.entries[index as usize] = entry;
                index
            }
            None => {
                self.entries.push(entry);
                self.entries.len() as u32 - 1
            }
        };
        self.ids.insert(key, index);
        index
    }

    /// The index of a feature of a resident mesh, counted until
    /// [`release`](Self::release)d, with the data the mesh captured.
    pub(crate) fn intern(
        &mut self,
        source: u32,
        layer: Option<&str>,
        id: &str,
        data: Option<&FeatureData>,
    ) -> u32 {
        let index = self.index(StateKey {
            source,
            layer: layer.map(str::to_string),
            id: id.to_string(),
        });
        let entry = &mut self.entries[index as usize];
        entry.refs += 1;
        if let Some(data) = data {
            match &mut entry.data {
                Some(held) => held.merge(data),
                None => entry.data = Some(data.clone()),
            }
            if !entry.state.is_empty() {
                self.touch(index);
            }
        }
        index
    }

    pub(crate) fn release(&mut self, index: u32) {
        if index == 0 {
            return;
        }
        let entry = &mut self.entries[index as usize];
        entry.refs -= 1;
        self.collect(index);
    }

    /// Frees an entry nothing refers to and that holds no state. Its row
    /// is already zero: rows are cleared when their state is removed.
    fn collect(&mut self, index: u32) {
        let entry = &mut self.entries[index as usize];
        if entry.refs == 0 {
            entry.data = None;
        }
        if entry.refs == 0
            && entry.state.is_empty()
            && let Some(key) = entry.key.take()
        {
            if entry.row.take().is_some() {
                self.zero.push(index);
            }
            self.ids.remove(&key);
            self.free.push(index);
        }
    }

    fn touch(&mut self, index: u32) {
        let entry = &mut self.entries[index as usize];
        if !entry.dirty {
            entry.dirty = true;
            self.dirty.push(index);
        }
    }

    /// Takes the dirty list, clearing each entry's flag.
    fn take_dirty(&mut self) -> Vec<u32> {
        let dirty = std::mem::take(&mut self.dirty);
        for &index in &dirty {
            self.entries[index as usize].dirty = false;
        }
        dirty
    }

    /// Counts state writes, so what feature state draws can be redrawn;
    /// while a state change fades, every frame at `time` is another.
    pub(crate) fn writes(&self, time: f64) -> (u64, u64) {
        match self.fading(time) {
            true => (self.writes, time.to_bits()),
            false => (self.writes, 0),
        }
    }

    /// Whether a state change still fades at frame time `time`.
    pub(crate) fn fading(&self, time: f64) -> bool {
        time < self.fading_until
    }

    /// Whether state values change with the zoom: what draws them then
    /// changes as it moves.
    pub(crate) fn zoom_dependent(&self) -> bool {
        self.zoom_dependent
    }

    pub(crate) fn apply(&mut self, op: &FeatureStateOp) {
        self.writes += 1;
        match op {
            FeatureStateOp::Set {
                source,
                source_layer,
                id,
                state,
            } => {
                let index = self.index(StateKey {
                    source: source.0,
                    layer: source_layer.clone(),
                    id: id.clone(),
                });
                let entry = &mut self.entries[index as usize];
                for (key, value) in state {
                    match entry.state.iter_mut().find(|(k, _)| k == key) {
                        Some((_, v)) => *v = value.clone(),
                        None => entry.state.push((key.clone(), value.clone())),
                    }
                }
                self.touch(index);
            }
            FeatureStateOp::Remove {
                source,
                source_layer,
                id,
                key,
            } => {
                let matches: Vec<u32> = match id {
                    Some(id) => self
                        .ids
                        .get(&StateKey {
                            source: source.0,
                            layer: source_layer.clone(),
                            id: id.clone(),
                        })
                        .copied()
                        .into_iter()
                        .collect(),
                    None => self
                        .ids
                        .iter()
                        .filter(|(k, _)| k.source == source.0 && k.layer == *source_layer)
                        .map(|(_, &i)| i)
                        .collect(),
                };
                for index in matches {
                    let entry = &mut self.entries[index as usize];
                    match key {
                        Some(key) => entry.state.retain(|(k, _)| k != key),
                        None => entry.state.clear(),
                    }
                    self.touch(index);
                }
            }
        }
    }

    /// Every feature's state as edits that set it again: what a style
    /// installed afresh (after a device loss) carries over.
    pub(crate) fn carried(&self) -> Vec<FeatureStateOp> {
        self.entries
            .iter()
            .filter(|e| !e.state.is_empty())
            .filter_map(|e| {
                let key = e.key.as_ref()?;
                Some(FeatureStateOp::Set {
                    source: orb_style::SourceKey(key.source),
                    source_layer: key.layer.clone(),
                    id: key.id.clone(),
                    state: e.state.clone(),
                })
            })
            .collect()
    }

    /// A feature's state, and what paint reading both state and data
    /// reads of it, if it has an entry.
    pub(crate) fn feature(&self, source: u32, layer: Option<&str>, id: &str) -> FeatureState<'_> {
        let key = StateKey {
            source,
            layer: layer.map(str::to_string),
            id: id.to_string(),
        };
        let entry = &self.entries[*self.ids.get(&key)? as usize];
        Some((&entry.state, entry.data.as_ref()))
    }

    /// The states of features of `sources` (keys), as JSON objects by
    /// (source, source layer, id): what queried features report.
    pub(crate) fn snapshot(
        &self,
        sources: &[u32],
    ) -> HashMap<(u32, Option<String>, String), String> {
        self.entries
            .iter()
            .filter(|e| !e.state.is_empty())
            .filter_map(|e| {
                let key = e.key.as_ref().filter(|k| sources.contains(&k.source))?;
                let object = orb_style::Value::Object(std::borrow::Cow::Owned(e.state.clone()));
                Some((
                    (key.source, key.layer.clone(), key.id.clone()),
                    object.to_json_string(),
                ))
            })
            .collect()
    }

    /// Compiles the row layout for a plan revision.
    fn compile(&mut self, plan: &StylePlan) {
        self.revision = Some(plan.revision);
        self.blocks.clear();
        self.zoom_dependent = false;
        let mut offset = 0;
        for node in &plan.nodes {
            if !plan.reads_state(node) {
                self.blocks.push(None);
                continue;
            }
            let properties = plan.paint_properties(node);
            let slots: Vec<usize> = (0..properties.len())
                .filter(|&s| properties[s].reads_state())
                .collect();
            self.zoom_dependent |= slots
                .iter()
                .any(|&s| properties[s].kind() != orb_style::Kind::Source);
            let layer = &plan.style.layers[node.layer];
            let transitions = (slots.iter())
                .map(|&s| {
                    let name = properties[s].id().map(|id| id.name());
                    (layer.transitions.iter())
                        .find(|(n, _)| Some(n.as_str()) == name)
                        .map_or(plan.style.transition, |(_, t)| {
                            t.over(plan.style.transition)
                        })
                })
                .collect();
            let block = Block {
                offset,
                mask: slots.iter().map(|&s| 1 << s).sum(),
                source: plan.sources[node.source].key,
                layer: plan.source_layer(node).map(str::to_string),
                slots,
                transitions,
            };
            offset += 2 + SLOT_WORDS * block.slots.len() as u32;
            self.blocks.push(Some(block));
        }
        self.stride = offset;
        self.all_dirty = true;
        // The rows' words are in the old layout: what is shown carries
        // over, without fades.
        for entry in &mut self.entries {
            if let Some(row) = &mut entry.row {
                row.words.clear();
            }
        }
    }

    /// Brings the GPU table up to date for this frame, at `zoom` and frame
    /// time `time` (the shaders'): the layout for the plan, the values for
    /// `zoom`, room for every index and the rows that changed, fading from
    /// what they showed. Returns whether node lookups changed, so paint
    /// must be written again.
    pub(crate) fn prepare(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        plan: &StylePlan,
        (zoom, time): (f64, f64),
    ) -> bool {
        let mut lookups_changed = false;
        if self.revision != Some(plan.revision) {
            let stride = self.stride;
            self.compile(plan);
            if self.stride != stride
                && let Some(table) = self.table.take()
            {
                renderer.free_table(table);
            }
            lookups_changed = true;
        }
        let zoom = (zoom.floor() - STATE_ZOOMS_BELOW).max(0.0);
        // How far the rows' zooms move: their words move with them.
        let shift = match self.zoom.is_nan() {
            true => 0,
            false => (zoom - self.zoom) as i64,
        };
        if zoom != self.zoom {
            lookups_changed = true;
            self.zoom = zoom;
            self.all_dirty |= self.zoom_dependent;
        }
        if self.stride == 0 {
            self.take_dirty();
            self.zero.clear();
            return lookups_changed;
        }
        let rows = self.entries.len() as u32;
        let capacity = self
            .table
            .as_ref()
            .map_or(0, |t| t.len() as u32 / self.stride);
        if rows > capacity {
            let rows = rows.next_power_of_two().max(64);
            let words = rows * self.stride;
            if let Some(table) = self.table.take() {
                renderer.free_table(table);
            }
            match renderer.alloc_table(gpu, words) {
                Ok(range) => {
                    renderer.write_table(gpu, range.start, &vec![0; words as usize]);
                    self.table = Some(range);
                }
                // Past the device's buffer limit: features draw without
                // state, and the next frame asks again.
                Err(_) => return true,
            }
            self.all_dirty = true;
            lookups_changed = true;
            self.zero.clear();
        }
        let base = self.table.as_ref().expect("allocated above").start;
        let zeros = vec![0u32; self.stride as usize];
        for index in std::mem::take(&mut self.zero) {
            renderer.write_table(gpu, base + index * self.stride, &zeros);
            self.rows_written += 1;
        }
        let indices: Vec<u32> = if std::mem::take(&mut self.all_dirty) {
            // Rows with state or a fade, and those whose state was just
            // removed.
            let indices = (1..self.entries.len() as u32)
                .filter(|&i| {
                    let entry = &self.entries[i as usize];
                    !entry.state.is_empty() || entry.row.is_some() || entry.dirty
                })
                .collect();
            self.take_dirty();
            indices
        } else {
            self.take_dirty()
        };
        let mut row = vec![0u32; self.stride as usize];
        let at = Moment { zoom, time, shift };
        for index in indices {
            row.fill(0);
            let entry = &mut self.entries[index as usize];
            let old = entry.row.take();
            if let Some(key) = entry.key.as_ref() {
                for (node, block) in plan.nodes.iter().zip(&self.blocks) {
                    let Some(block) = block.as_ref() else {
                        continue;
                    };
                    if block.source != key.source || block.layer != key.layer {
                        continue;
                    }
                    let properties = plan.paint_properties(node);
                    let until =
                        write_block(&mut row, (block, &properties), (entry, old.as_ref()), at);
                    self.fading_until = self.fading_until.max(until);
                }
            }
            if row.iter().any(|&w| w != 0) {
                entry.row = Some(Row {
                    state: entry.state.clone(),
                    words: row.clone(),
                });
            }
            renderer.write_table(gpu, base + index * self.stride, &row);
            self.rows_written += 1;
            self.collect(index);
        }
        lookups_changed
    }

    /// Where node `node` (an index into the plan the layout is for) reads
    /// its state lookups.
    pub(crate) fn lookup(&self, node: usize) -> StateLookup {
        match (self.blocks.get(node).and_then(Option::as_ref), &self.table) {
            (Some(block), Some(table)) => StateLookup {
                mask: block.mask,
                base: table.start + block.offset,
                stride: self.stride,
                zoom: self.zoom as f32,
            },
            _ => StateLookup::default(),
        }
    }

    /// Where node `node` of `plan` reads its state lookups once a feature
    /// has state: [`lookup`](Self::lookup)'s, with what it reads before any
    /// feature has. The node's pipelines are made for it, so a first state
    /// asks for none.
    pub(crate) fn reach(&self, plan: &StylePlan, node: usize) -> StateLookup {
        let lookup = self.lookup(node);
        let n = &plan.nodes[node];
        if lookup.mask != 0 || !plan.reads_state(n) {
            return lookup;
        }
        let properties = plan.paint_properties(n);
        let mask = (properties.iter().enumerate())
            .filter(|(_, p)| p.reads_state())
            .map(|(slot, _)| 1 << slot)
            .sum();
        StateLookup { mask, ..lookup }
    }

    /// Lets go of the GPU table, for its owner to free: rows written into
    /// the next one start without fades.
    pub(crate) fn take_table(&mut self) -> Option<std::ops::Range<u32>> {
        for entry in &mut self.entries {
            entry.row = None;
        }
        self.table.take()
    }
}

/// Where and when rows are written: the first of their zooms, the frame
/// time, and how many zooms up from the last rows' theirs start.
#[derive(Clone, Copy, Debug)]
struct Moment {
    zoom: f64,
    time: f64,
    shift: i64,
}

/// Writes `entry`'s block of a node into `row`, from `old`, the row as it
/// was last written: its values with the entry's state, and a fade from
/// what `old` showed at the moment's time when the state changed, or the
/// fade `old` was in, carried over. Returns the frame time its fade ends,
/// or −∞.
fn write_block(
    row: &mut [u32],
    (block, properties): (&Block, &[PaintProperty]),
    (entry, old): (&Entry, Option<&Row>),
    at: Moment,
) -> f64 {
    let z = STATE_ZOOMS as usize;
    let offset = block.offset as usize;
    let has = !entry.state.is_empty();
    let changed = old.map_or(&[][..], |r| &r.state[..]) != &entry.state[..];
    // The old block, its words moved to these zooms.
    let words = old.map_or(&[][..], |r| &r.words[..]);
    let old_block = words.get(offset..offset + 2 + block.slots.len() * SLOT_WORDS as usize);
    let old_flags = old_block.map_or(0, |b| b[0]);
    let old_start = old_block.map_or(0.0, |b| f64::from(f32::from_bits(b[1])));
    let moved = |slot: usize, part: usize, j: usize| {
        let b = old_block.expect("read only with an old block");
        let k = (j as i64 + at.shift).clamp(0, z as i64 - 1) as usize;
        b[2 + slot * SLOT_WORDS as usize + part * z + k]
    };
    // How far an old slot's fade had eased by now, 1 once done.
    let eased = |slot: usize| {
        if old_flags & FADING == 0 {
            return 1.0;
        }
        let b = old_block.expect("fading blocks are read");
        let t = 2 + slot * SLOT_WORDS as usize + 2 * z;
        let (duration, delay) = (f32::from_bits(b[t]), f32::from_bits(b[t + 1]));
        let since = at.time - old_start - f64::from(delay);
        crate::transitions::ease(since / f64::from(duration).max(1e-3))
    };
    let fade = changed && block.transitions.iter().any(|t| t.duration + t.delay > 0.0);
    let carried = !changed && (0..block.slots.len()).any(|i| eased(i) < 1.0);
    let mut flags = u32::from(has) * HAS;
    let mut until = f64::NEG_INFINITY;
    let start = match (fade, carried) {
        (true, _) => at.time,
        (false, true) => old_start,
        (false, false) => 0.0,
    };
    if fade || carried {
        flags |= FADING;
    }
    for (i, (&slot, transition)) in block.slots.iter().zip(&block.transitions).enumerate() {
        let p = &properties[slot];
        let data = entry.data.as_ref().filter(|_| p.mixes_state());
        let words = offset + 2 + SLOT_WORDS as usize * i;
        if has {
            for (j, word) in row[words..][..z].iter_mut().enumerate() {
                *word = p.state_word(at.zoom + j as f64, &entry.state, data);
            }
        }
        let prior = &mut row[words + z..][..z + 2];
        if fade {
            // What the feature showed: its old state's values (or its
            // values without state), eased from the old fade's.
            let t = eased(i);
            for (j, word) in prior[..z].iter_mut().enumerate() {
                let shown = match old_flags & HAS != 0 {
                    true => moved(i, 0, j),
                    false => p.state_word(at.zoom + j as f64, &[], data),
                };
                *word = match t < 1.0 {
                    true => mix_word(p.is_color(), moved(i, 1, j), shown, t),
                    false => shown,
                };
            }
            prior[z] = (transition.duration as f32).to_bits();
            prior[z + 1] = (transition.delay as f32).to_bits();
            until = until.max(at.time + transition.delay + transition.duration);
        } else if carried {
            for (j, word) in prior[..z].iter_mut().enumerate() {
                *word = moved(i, 1, j);
            }
            let b = old_block.expect("carried fades are read");
            let t = 2 + i * SLOT_WORDS as usize + 2 * z;
            prior[z] = b[t];
            prior[z + 1] = b[t + 1];
            let (duration, delay) = (f32::from_bits(b[t]), f32::from_bits(b[t + 1]));
            until = until.max(old_start + f64::from(delay) + f64::from(duration));
        }
    }
    if flags != 0 {
        row[offset] = flags;
        row[offset + 1] = (start as f32).to_bits();
    }
    until
}

/// Row word `b` eased from `a` by `t`: colours (rgba8) or numbers (f32).
fn mix_word(color: bool, a: u32, b: u32, t: f64) -> u32 {
    let t = t as f32;
    if !color {
        let (a, b) = (f32::from_bits(a), f32::from_bits(b));
        return (a + (b - a) * t).to_bits();
    }
    let channel = |w: u32, i: u32| ((w >> (8 * i)) & 0xff) as f32;
    (0..4)
        .map(|i| {
            let (a, b) = (channel(a, i), channel(b, i));
            ((a + (b - a) * t).round() as u32) << (8 * i)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faded_words_mix_numbers_and_colours() {
        let mid = mix_word(false, 2f32.to_bits(), 4f32.to_bits(), 0.5);
        assert_eq!(f32::from_bits(mid), 3.0);
        let red = orb_layout::pack_color([1.0, 0.0, 0.0, 1.0]);
        let blue = orb_layout::pack_color([0.0, 0.0, 1.0, 1.0]);
        assert_eq!(
            mix_word(true, red, blue, 0.5),
            orb_layout::pack_color([128.0 / 255.0, 0.0, 128.0 / 255.0, 1.0])
        );
        assert_eq!(mix_word(true, red, blue, 1.0), blue);
    }
    use orb_style::SourceKey;

    #[test]
    fn a_feature_is_listed_dirty_once_however_often_it_changes() {
        let mut states = FeatureStates::default();
        let set = |id: &str| FeatureStateOp::Set {
            source: SourceKey(1),
            source_layer: None,
            id: id.into(),
            state: vec![("hover".into(), Value::Bool(true))],
        };
        let a = states.intern(1, None, "a", None);
        states.apply(&set("a"));
        states.apply(&set("a"));
        states.apply(&FeatureStateOp::Remove {
            source: SourceKey(1),
            source_layer: None,
            id: None,
            key: None,
        });
        assert_eq!(states.dirty, [a]);
        // Freed while listed, then taken by another feature: still once.
        states.release(a);
        let b = states.intern(1, None, "b", None);
        assert_eq!(b, a, "the index is reused");
        states.apply(&set("b"));
        assert_eq!(states.dirty, [b]);
        assert_eq!(states.take_dirty(), [b]);
        states.apply(&set("b"));
        assert_eq!(states.dirty, [b], "listed again after it was taken");
    }

    #[test]
    fn indices_are_counted_and_states_survive_unloads() {
        let mut states = FeatureStates::default();
        let a = states.intern(1, Some("poi"), "7", None);
        assert_eq!(states.intern(1, Some("poi"), "7", None), a);
        let b = states.intern(1, Some("road"), "7", None);
        assert_ne!(a, b, "ids are per source layer");
        states.apply(&FeatureStateOp::Set {
            source: SourceKey(1),
            source_layer: Some("poi".into()),
            id: "7".into(),
            state: vec![("hover".into(), Value::Bool(true))],
        });
        states.release(a);
        states.release(a);
        states.release(b);
        assert_eq!(states.free, [b], "the stateless entry is freed");
        // The state outlives the meshes, as GL JS keeps it.
        assert_eq!(states.intern(1, Some("poi"), "7", None), a);
        states.apply(&FeatureStateOp::Remove {
            source: SourceKey(1),
            source_layer: Some("poi".into()),
            id: None,
            key: None,
        });
        assert!(states.entries[a as usize].state.is_empty());
    }
}
