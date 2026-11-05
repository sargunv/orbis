//! Feature state (plan §10.2): per source and source layer, feature id →
//! state index; the states set on them; and the GPU table the shaders read
//! through each mesh's slot → index remap.
//!
//! Every feature of a resident mesh whose node reads state gets an index,
//! counted by the meshes that refer to it, so a state write rewrites one
//! table row and no mesh. A row holds, per node that reads state, a flag
//! word (non-zero once the feature has state) and the node's properties
//! that read state evaluated with the feature's state at the table's zoom
//! and the next: the layout is compiled per style revision. Index 0 and
//! rows with a zero flag draw the value with no state: the node's constant,
//! or for paint that also reads feature data, the feature's property row.
//! That paint evaluates here from the feature data its meshes captured, so
//! a state write shows in the next frame, as GL JS's `updatePaintArrays`.

use std::collections::HashMap;
use std::ops::Range;

use orb_expr::FeatureData;
use orb_hal::Gpu;
use orb_render::{Renderer, StateLookup};
use orb_style::FeatureStateOp;
use orb_style::Value;

use crate::style::{PaintProperty, StylePlan};

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
}

#[derive(Debug)]
pub(crate) struct FeatureStates {
    ids: HashMap<StateKey, u32>,
    entries: Vec<Entry>,
    free: Vec<u32>,
    dirty: Vec<u32>,
    all_dirty: bool,
    /// The plan revision the layout is compiled for.
    revision: Option<u64>,
    /// Per node index, its block.
    blocks: Vec<Option<Block>>,
    stride: u32,
    zoom_dependent: bool,
    /// The zoom the rows hold values for.
    zoom: f64,
    table: Option<Range<u32>>,
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
        let entry = Entry {
            key: Some(key.clone()),
            ..Entry::default()
        };
        let index = match self.free.pop() {
            Some(index) => {
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
            self.ids.remove(&key);
            self.free.push(index);
        }
    }

    fn touch(&mut self, index: u32) {
        if !self.dirty.contains(&index) {
            self.dirty.push(index);
        }
    }

    /// Counts state writes, so what feature state draws can be redrawn.
    pub(crate) fn writes(&self) -> u64 {
        self.writes
    }

    /// The zoom the rows hold values for, when any lookup's values depend
    /// on the zoom: what they draw then changes with it.
    pub(crate) fn zoom_key(&self) -> Option<u64> {
        self.zoom_dependent.then(|| self.zoom.to_bits())
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

    /// A feature's state, and what paint reading both state and data
    /// reads of it, if it has an entry.
    pub(crate) fn feature(
        &self,
        source: u32,
        layer: Option<&str>,
        id: &str,
    ) -> Option<(&[(String, Value<'static>)], Option<&FeatureData>)> {
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
            let block = Block {
                offset,
                mask: slots.iter().map(|&s| 1 << s).sum(),
                source: plan.sources[node.source].key,
                layer: plan.source_layer(node).map(str::to_string),
                slots,
            };
            offset += 1 + 2 * block.slots.len() as u32;
            self.blocks.push(Some(block));
        }
        self.stride = offset;
        self.all_dirty = true;
    }

    /// Brings the GPU table up to date for this frame: the layout for the
    /// plan, the values for `zoom`, room for every index and the rows that
    /// changed. Returns whether node lookups changed, so paint must be
    /// written again.
    pub(crate) fn prepare(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        plan: &StylePlan,
        zoom: f64,
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
        let zoom = zoom.floor().max(0.0);
        if zoom != self.zoom {
            lookups_changed = true;
            self.zoom = zoom;
            self.all_dirty |= self.zoom_dependent;
        }
        if self.stride == 0 {
            self.dirty.clear();
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
            match renderer.alloc_table(words) {
                Ok(range) => {
                    renderer.write_table(gpu, range.start, &vec![0; words as usize]);
                    self.table = Some(range);
                }
                // Out of property memory: features draw without state.
                Err(_) => return true,
            }
            self.all_dirty = true;
            lookups_changed = true;
        }
        let base = self.table.as_ref().expect("allocated above").start;
        let indices: Vec<u32> = if std::mem::take(&mut self.all_dirty) {
            // Rows with state, and those whose state was just removed.
            let dirty = std::mem::take(&mut self.dirty);
            (1..self.entries.len() as u32)
                .filter(|&i| !self.entries[i as usize].state.is_empty() || dirty.contains(&i))
                .collect()
        } else {
            std::mem::take(&mut self.dirty)
        };
        let mut row = vec![0u32; self.stride as usize];
        for index in indices {
            row.fill(0);
            let entry = &self.entries[index as usize];
            if let Some(key) = entry.key.as_ref().filter(|_| !entry.state.is_empty()) {
                for (node, block) in plan.nodes.iter().zip(&self.blocks) {
                    let Some(block) = block.as_ref() else {
                        continue;
                    };
                    if block.source != key.source || block.layer != key.layer {
                        continue;
                    }
                    let properties = plan.paint_properties(node);
                    let at = block.offset as usize;
                    row[at] = 1;
                    for (i, &slot) in block.slots.iter().enumerate() {
                        let p: &PaintProperty = &properties[slot];
                        let data = entry.data.as_ref().filter(|_| p.mixes_state());
                        row[at + 1 + 2 * i] = p.state_word(zoom, &entry.state, data);
                        row[at + 2 + 2 * i] = p.state_word(zoom + 1.0, &entry.state, data);
                    }
                }
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

    /// Lets go of the GPU table, for its owner to free.
    pub(crate) fn take_table(&mut self) -> Option<std::ops::Range<u32>> {
        self.table.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_style::SourceKey;

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
