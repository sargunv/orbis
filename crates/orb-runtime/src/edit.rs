//! Style edits on a live map (plan §11.2, S6): a transaction's patch
//! applied without a reload.
//!
//! - Uniform paint (constant and camera values, visibility) is a node
//!   uniform write on the next frame.
//! - A paint value that is, or becomes, per feature re-evaluates the
//!   property rows of resident meshes as a column job from the jobs'
//!   cached tiles; geometry stays.
//! - Feature-state lookups recompile the state table; meshes that need
//!   feature ids get them from a column job.
//! - Filter, layout and zoom-range edits lay out only that layer's nodes
//!   again, from cached tiles.
//! - Layers added, removed or moved renumber the render nodes, keeping
//!   every other node's meshes; new nodes are laid out from cached tiles.
//! - A source added, removed or redefined loads (or drops) only its tiles.
//!
//! Jobs carry the revision they were laid out with; per node, results from
//! before the edit that invalidated it are stale and laid out again.

use std::collections::BTreeMap;
use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_style::{Binding, Change, Class, Transaction};
use orb_v8::LayerKind;

use crate::jobs::{StyleUpdate, TileResult};
use crate::map::{
    Map, NodeState, SourceState, attach_rows, record_maxima, revise, upload_mesh, words,
};

/// The class a paint change costs this renderer for one node:
/// `orb-style`'s, lowered to a program table write when the node's input
/// columns pack the same (`columns_kept`), raised where property rows hold
/// more than the property (line dash and gradient textures, and the width
/// that scales dashes).
fn paint_class(
    kind: &LayerKind,
    name: &str,
    from: Binding,
    to: Binding,
    columns_kept: bool,
) -> Class {
    let mut class = Change::Paint {
        layer: orb_style::LayerKey(0),
        name: String::new(),
        from,
        to,
    }
    .class();
    if class == Class::Column && to == Binding::Bytecode && columns_kept {
        class = Class::Block;
    }
    let line_rows = matches!(kind, LayerKind::Line(l)
        if l.paint.dasharray.as_constant() != Some(&None) || l.paint.gradient.is_some());
    match name {
        "line-dasharray" | "line-gradient" => Class::Column,
        "line-width" if line_rows => Class::Column,
        _ => class,
    }
}

impl Map {
    /// Applies a transaction to the style (plan §11.2): every edit or, on
    /// an error, none. Returns the style generation at which the frame
    /// shows it in full ([`FrameReport::generation`](crate::FrameReport));
    /// paint and feature-state edits show on the next frame.
    pub fn edit(&mut self, tx: &Transaction) -> Result<u64, String> {
        let style = self.style.as_mut().ok_or("no style is loaded")?;
        let patch = style.doc.apply(tx).map_err(|d| d.to_string())?;
        self.warnings
            .extend(patch.warnings.iter().map(ToString::to_string));
        for op in &patch.feature_state {
            style.states.apply(op);
        }
        self.events.wake();
        if patch.changes.is_empty() {
            let generation = self.generation();
            self.events.push(crate::MapEvent::StyleData { generation });
            return Ok(generation);
        }
        if patch.changes.contains(&Change::Restyle) {
            let doc = style.doc.clone();
            self.install(doc);
            return Ok(self.version);
        }
        self.version += 1;
        let version = self.version;
        let mut style = self.style.take().expect("checked above");
        let old = Arc::clone(&style.worker);
        let worker = Arc::new(revise(
            Some(&old),
            &style.doc,
            version,
            self.labels.ideographs.clone(),
            self.surface.tiles(),
        ));
        let mut layers: Vec<u32> = patch
            .changes
            .iter()
            .filter_map(|c| c.layer().map(|l| l.0))
            .collect();
        layers.dedup();
        let doc = &style.doc;
        let update = StyleUpdate {
            revision: version,
            layers: doc.layer_keys().map(|k| k.0).collect(),
            sources: doc.source_keys().map(|k| k.0).collect(),
            layer_json: doc
                .layer_keys()
                .enumerate()
                .filter(|(_, k)| layers.contains(&k.0))
                .map(|(i, k)| (k.0, Arc::new(doc.layer_json(i).clone())))
                .collect(),
            source_json: patch
                .changes
                .iter()
                .filter_map(|c| match c {
                    Change::SourceAdded(key) => {
                        let i = doc.source_index(*key).expect("added sources exist");
                        let id = doc.style().sources[i].id.clone();
                        Some((key.0, id, Arc::new(doc.source_json(i).clone())))
                    }
                    _ => None,
                })
                .collect(),
            state: doc.global_state(),
            glyphs: doc.style().glyphs.clone(),
            font_faces: doc.style().font_faces.clone(),
            local_ideographs: self.labels.ideographs.clone(),
            surface: self.surface.tiles(),
            cost: doc.cost_model(),
        };
        self.services.jobs.set_style(style.id, &update, &worker);
        self.warnings.extend(
            worker
                .plan()
                .warnings
                .iter()
                .filter(|w| !old.plan().warnings.contains(w))
                .cloned(),
        );
        style.worker = Arc::clone(&worker);
        let plan = worker.plan();

        // Sources, by key: removed ones leave the scene next frame.
        let old_keys: Vec<u32> = old.plan().sources.iter().map(|s| s.key).collect();
        let mut old_sources: Vec<Option<_>> = std::mem::take(&mut style.sources)
            .into_iter()
            .zip(std::mem::take(&mut style.placed))
            .map(Some)
            .collect();
        let mut added = Vec::new();
        for (index, source) in plan.sources.iter().enumerate() {
            match old_keys
                .iter()
                .position(|k| *k == source.key)
                .and_then(|i| old_sources[i].take())
            {
                Some((state, placed)) => {
                    style.sources.push(state);
                    style.placed.push(placed);
                }
                None => {
                    style.sources.push(SourceState::Failed);
                    style.placed.push(Vec::new());
                    added.push(index);
                }
            }
        }
        style
            .retired
            .extend(old_sources.into_iter().flatten().map(|(state, _)| state));

        // Nodes, by key: new nodes are stale everywhere.
        let mut relayout: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        let mut columns: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        style.nodes.retain(|key, _| plan.node(*key).is_some());
        for node in &plan.nodes {
            if style.nodes.contains_key(&node.key) {
                continue;
            }
            style.nodes.insert(
                node.key,
                NodeState {
                    maxima: vec![0.0; plan.paint_properties(node).len()],
                    ramp: None,
                    layout_epoch: version,
                    rows_epoch: version,
                },
            );
            if node.source != usize::MAX && plan.image_kind(node.source).is_none() {
                relayout.entry(node.source).or_default().push(node.key);
            }
        }
        for change in &patch.changes {
            let Some(layer) = change.layer() else {
                continue;
            };
            let Some(index) = plan.layer_keys.iter().position(|k| *k == layer.0) else {
                continue;
            };
            let kind = &plan.style.layers[index].kind;
            for node in plan.nodes.iter().filter(|n| n.layer == index) {
                let before = old.plan().node(node.key).map(|i| &old.plan().nodes[i]);
                let layout_changed = before.and_then(|n| n.gpu.as_ref()).map(|g| &g.layout)
                    != node.gpu.as_ref().map(|g| &g.layout);
                // Programs read the columns their layout packs; rows keep
                // their own stride, so columns nothing reads may stay.
                let repack = layout_changed && node.gpu.is_some();
                let class = match change {
                    Change::Paint { name, from, to, .. } => {
                        let class = paint_class(kind, name, *from, *to, !repack);
                        // Meshes need feature ids to read state.
                        let ids = plan.reads_state(node)
                            && !before.is_some_and(|b| old.plan().reads_state(b));
                        if repack || ids {
                            class.max(Class::Column)
                        } else {
                            class
                        }
                    }
                    change => change.class(),
                };
                let state = style.nodes.get_mut(&node.key).expect("added above");
                state.ramp = None;
                if layout_changed {
                    // Input ranges follow the columns; new rows report them.
                    state.maxima.truncate(plan.paint_properties(node).len());
                }
                let image = node.source == usize::MAX || plan.image_kind(node.source).is_some();
                if image || state.layout_epoch == version {
                    continue;
                }
                match class {
                    Class::Column => {
                        state.rows_epoch = version;
                        columns.entry(node.source).or_default().push(node.key);
                    }
                    Class::Relayout => {
                        state.layout_epoch = version;
                        state.rows_epoch = version;
                        relayout.entry(node.source).or_default().push(node.key);
                    }
                    _ => {}
                }
            }
        }
        style.painted = None;
        self.style = Some(style);
        for index in added {
            let state = self.start_source(plan, index);
            self.style.as_mut().expect("restored").sources[index] = state;
        }

        // Jobs for the resident tiles, from the jobs' cached tiles.
        for (source, mut nodes) in relayout {
            nodes.sort_unstable();
            nodes.dedup();
            for tile in self.resident_tiles(source, None) {
                self.send_job(source, tile, Some(nodes.clone()), false, Some(version));
            }
        }
        for (source, mut nodes) in columns {
            nodes.sort_unstable();
            nodes.dedup();
            for tile in self.resident_tiles(source, Some(&nodes)) {
                self.send_job(source, tile, Some(nodes.clone()), true, Some(version));
            }
        }
        self.events.push(crate::MapEvent::StyleData {
            generation: self.version,
        });
        Ok(self.version)
    }

    /// The ready tiles of a source, or those with a mesh of one of `nodes`.
    fn resident_tiles(&self, source: usize, nodes: Option<&[u32]>) -> Vec<TileIndex> {
        let style = self.style.as_ref().expect("edits need a style");
        let SourceState::Ready { pyramid, .. } = &style.sources[source] else {
            return Vec::new();
        };
        pyramid
            .ready()
            .filter(|(_, resident)| {
                nodes.is_none_or(|nodes| resident.meshes.iter().any(|m| nodes.contains(&m.node)))
            })
            .map(|(tile, _)| tile)
            .collect()
    }

    /// Applies a relayout or column job's result to its resident tile.
    pub(crate) fn apply_edit(
        &mut self,
        gpu: &mut dyn Gpu,
        source: u32,
        tile: TileIndex,
        result: Result<TileResult, String>,
    ) {
        let (Some(style), Some((renderer, _))) = (&mut self.style, &mut self.renderer) else {
            return;
        };
        let plan = style.worker.plan();
        let Some(index) = plan.source(source) else {
            return;
        };
        let mut build = match result {
            Ok(build) => build,
            Err(e) => {
                self.warnings.push(format!("tile {tile:?}: {e}"));
                return;
            }
        };
        self.warnings.append(&mut build.warnings);
        let SourceState::Ready { pyramid, .. } = &mut style.sources[index] else {
            return;
        };
        let Some(resident) = pyramid.get_mut(tile) else {
            return;
        };
        let tile_zoom = resident.zoom;
        let mut images = self.services.images.0.lock().expect("images poisoned");
        let atlas = images.atlas.as_mut().expect("initialized");
        let namespace = style.image_namespace.expect("namespace");
        let mut stale_rows = Vec::new();
        if build.rows_only {
            for mesh in &build.meshes {
                let Some(state) = style.nodes.get(&mesh.node) else {
                    continue;
                };
                let Some(current) = resident.meshes.iter_mut().find(|m| m.node == mesh.node) else {
                    continue;
                };
                // A relayout since will bring its own rows.
                if state.layout_epoch > current.layout_rev || build.revision < state.rows_epoch {
                    continue;
                }
                let props = words(&mesh.props, build.host_bytes);
                if renderer
                    .scene()
                    .set_mesh_props(gpu, current.mesh, &props, mesh.row_words)
                    .is_err()
                {
                    self.warnings
                        .push("out of property memory; a paint edit is not drawn".into());
                    continue;
                }
                self.retired_ramps.append(&mut current.ramps);
                for index in current.states.drain(..) {
                    style.states.release(index);
                }
                match attach_rows(
                    gpu,
                    renderer,
                    plan,
                    &build,
                    mesh,
                    current.mesh,
                    atlas,
                    namespace,
                    &mut style.states,
                    &mut self.warnings,
                ) {
                    Ok((ramps, states)) => {
                        current.ramps = ramps;
                        current.states = states;
                    }
                    Err(_) => self
                        .warnings
                        .push("out of property memory; feature state is not drawn".into()),
                }
                current.hits = mesh.hits.clone();
                current.rows_rev = build.revision;
                self.counts.rows_uploaded += 1;
            }
        } else {
            for &key in &build.nodes {
                let Some(state) = style.nodes.get(&key) else {
                    continue;
                };
                if build.revision < state.layout_epoch {
                    continue;
                }
                if build.revision < state.rows_epoch {
                    stale_rows.push(key);
                }
                if let Some(at) = resident.meshes.iter().position(|m| m.node == key) {
                    let mut old = resident.meshes.swap_remove(at);
                    renderer.scene().remove_mesh(old.mesh);
                    for index in old.states.drain(..) {
                        style.states.release(index);
                    }
                    self.retired_ramps.append(&mut old.ramps);
                }
                // A symbol layer comes back whole with its bucket, below.
                if plan
                    .node(key)
                    .is_some_and(|i| plan.nodes[i].family.is_symbol())
                {
                    resident.symbols.retain(|s| s.layer != key / 2);
                    continue;
                }
                let Some(mesh) = build.meshes.iter().find(|m| m.node == key) else {
                    continue;
                };
                if upload_mesh(
                    gpu,
                    renderer,
                    plan,
                    &build,
                    mesh,
                    resident,
                    atlas,
                    namespace,
                    &mut style.states,
                    &mut self.warnings,
                )
                .is_err()
                {
                    self.warnings.push(format!(
                        "tile {tile:?} doesn't fit in scene memory; a layer is not drawn"
                    ));
                    continue;
                }
                self.counts.meshes_uploaded += 1;
            }
            build.symbols.retain(|s| {
                style
                    .nodes
                    .get(&(s.layer * 2))
                    .is_some_and(|state| build.revision >= state.layout_epoch)
            });
            if self
                .labels
                .upload(
                    gpu,
                    renderer,
                    plan,
                    &build,
                    resident,
                    atlas,
                    namespace,
                    &mut style.states,
                    tile_zoom,
                )
                .is_err()
            {
                self.warnings.push(format!(
                    "tile {tile:?} doesn't fit in scene memory; a symbol layer is not drawn"
                ));
            }
            // New meshes and parts take their place in the tile order.
            style.placed[index].clear();
        }
        // Rows from a column job can raise a node's maxima (a radius made
        // per feature, say), which culling pads by; results skipped as
        // stale can only pad more than needed.
        record_maxima(&mut style.nodes, &build.meshes, &mut style.painted);
        drop(images);
        if !stale_rows.is_empty() {
            let version = self.version;
            self.send_job(index, tile, Some(stale_rows), true, Some(version));
        }
    }
}
