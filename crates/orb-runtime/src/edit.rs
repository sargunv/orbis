//! Style edits on a live map: a transaction's patch
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

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_style::v8::Transaction;
use orb_style::{Binding, Change, Class, LayerKey, PropertyId};

use crate::jobs::TileResult;
use crate::map::{Map, NodeState, revise};
use crate::resident::{
    SourceState, evict_unused, mesh_sizes, record_maxima, remove_resident, resolve_rows,
    under_budget, upload_mesh, with_room,
};
use crate::worker::StyleUpdate;

/// The class a paint change costs this renderer for one node:
/// `orb-style`'s, raised where property rows hold more than the property
/// (line dash and gradient textures, and the width that scales dashes).
fn paint_class(layer: &orb_style::Layer, id: PropertyId, from: Binding, to: Binding) -> Class {
    let class = Change::Paint {
        layer: LayerKey(0),
        id,
        from,
        to,
    }
    .class();
    let line_rows = crate::line::has_textures(layer);
    if id == orb_style::LINE_DASHARRAY.id()
        || id == orb_style::LINE_GRADIENT.id()
        || (id == orb_style::LINE_WIDTH.id() && line_rows)
    {
        Class::Column
    } else {
        class
    }
}

impl Map {
    /// Applies a transaction to the style: every edit or, on
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
        // A new projection: the globe changes at once; vector tiles lay
        // out again for its curvature from the jobs' cached data, and the
        // next frame regrids image tiles (nothing loads again).
        let reproject = patch.changes.contains(&Change::Projection);
        if reproject {
            self.surface = crate::surface::Surface::of(style.doc.style());
            // The camera stands on it now: a move started after the edit
            // runs on the new surface, rather than stopping at the next frame.
            self.camera.set_surface(self.surface.view_surface());
        }
        self.version += 1;
        let version = self.version;
        let mut style = self.style.take().expect("checked above");
        let old = Arc::clone(&style.worker);
        // Sources with new data: the jobs index it again, and their layers
        // lay out again from it while their tiles stay drawn.
        let new_data: Vec<u32> = patch
            .changes
            .iter()
            .filter_map(|c| match c {
                Change::SourceData(key) => Some(key.0),
                _ => None,
            })
            .collect();
        let worker = Arc::new(revise(
            Some(&old),
            &style.doc,
            &new_data,
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
                    // Job hosts apply a diff to the data they hold.
                    Change::SourceData(key) if patch.source_diffs.iter().any(|(k, _)| k == key) => {
                        None
                    }
                    Change::SourceAdded(key) | Change::SourceData(key) => {
                        let i = doc.source_index(*key).expect("added sources exist");
                        let id = doc.style().sources[i].id.clone();
                        Some((key.0, id, Arc::new(doc.source_json(i).clone())))
                    }
                    _ => None,
                })
                .collect(),
            source_diffs: patch
                .source_diffs
                .iter()
                .map(|(key, diffs)| (key.0, diffs.clone()))
                .collect(),
            ..StyleUpdate::base(
                doc,
                version,
                self.labels.ideographs.clone(),
                self.surface.tiles(),
            )
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

        // Paint changes ease from the property before the edit, and the
        // root's light, sky and terrain from what the last frame showed.
        let (now, zoom) = (self.frame_time, self.zoom());
        if patch.changes.contains(&Change::Root) {
            self.root.start(&plan.style, zoom);
        }
        let old_plan = old.plan();
        for change in &patch.changes {
            // A layer shown or hidden fades in or out.
            if let Change::Visibility(layer) = change {
                let index = |plan: &crate::style::StylePlan| {
                    plan.layer_keys.iter().position(|k| *k == layer.0)
                };
                if let (Some(before), Some(after)) = (index(old_plan), index(plan)) {
                    let (old, new) = (&old_plan.style.layers[before], &plan.style.layers[after]);
                    let shown = new.visibility == orb_style::Visibility::Visible;
                    if (old.visibility == orb_style::Visibility::Visible) != shown {
                        self.transitions.fade(
                            (&new.id, new.ty),
                            |id| new.slot(id).clone(),
                            shown,
                            plan.style.transition,
                            now,
                        );
                    }
                }
                continue;
            }
            let Change::Paint { layer, id, .. } = change else {
                continue;
            };
            let index =
                |plan: &crate::style::StylePlan| plan.layer_keys.iter().position(|k| *k == layer.0);
            let (Some(before), Some(after)) = (index(old_plan), index(plan)) else {
                continue;
            };
            let new = &plan.style.layers[after];
            let transition = new
                .transitions
                .iter()
                .find(|(n, _)| n == id.name())
                .map_or(plan.style.transition, |(_, t)| {
                    t.over(plan.style.transition)
                });
            let old = &old_plan.style.layers[before];
            self.transitions
                .start(&new.id, *id, old.slot(*id), transition, (now, zoom));
        }

        // Sources, by key: removed ones leave the scene next frame.
        let old_keys: Vec<u32> = old.plan().sources.iter().map(|s| s.key).collect();
        let mut old_sources: Vec<Option<_>> = std::mem::take(&mut style.sources)
            .into_iter()
            .zip(std::mem::take(&mut style.placed))
            .zip(std::mem::take(&mut style.held))
            .map(Some)
            .collect();
        let mut added = Vec::new();
        for (index, source) in plan.sources.iter().enumerate() {
            match old_keys
                .iter()
                .position(|k| *k == source.key)
                .and_then(|i| old_sources[i].take())
            {
                Some(((state, placed), held)) => {
                    style.sources.push(state);
                    style.placed.push(placed);
                    style.held.push(held);
                }
                None => {
                    style.sources.push(SourceState::Failed);
                    style.placed.push(Vec::new());
                    style.held.push(Vec::new());
                    added.push(index);
                }
            }
        }
        style.retired.extend(
            old_sources
                .into_iter()
                .flatten()
                .map(|((state, _), _)| state),
        );

        // Nodes, by key: new nodes are stale everywhere.
        let mut relayout: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        let mut columns: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        style.nodes.retain(|key, _| plan.node(*key).is_some());
        for node in &plan.nodes {
            if style.nodes.contains_key(&node.key) {
                continue;
            }
            style
                .nodes
                .insert(node.key, NodeState::new(plan, node, version));
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
            let layer = &plan.style.layers[index];
            for node in plan.nodes.iter().filter(|n| n.layer == index) {
                let before = old.plan().node(node.key).map(|i| &old.plan().nodes[i]);
                let class = match change {
                    Change::Paint { id, from, to, .. } => {
                        let class = paint_class(layer, *id, *from, *to);
                        // Meshes need feature ids to read state.
                        let ids = plan.reads_state(node)
                            && !before.is_some_and(|b| old.plan().reads_state(b));
                        if ids { class.max(Class::Column) } else { class }
                    }
                    change => change.class(),
                };
                let state = style.nodes.get_mut(&node.key).expect("added above");
                state.ramp = None;
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
        for node in &plan.nodes {
            let image = node.source == usize::MAX || plan.image_kind(node.source).is_some();
            let new_data = !image && new_data.contains(&plan.sources[node.source].key);
            let state = style.nodes.get_mut(&node.key).expect("added above");
            if !(reproject || new_data) || image || state.layout_epoch == version {
                continue;
            }
            state.layout_epoch = version;
            state.rows_epoch = version;
            relayout.entry(node.source).or_default().push(node.key);
        }
        // Another sprite loads over the old one, next frame.
        if patch.changes.contains(&Change::Sprite) {
            style.sprites_started = false;
            let names = std::mem::take(&mut style.sprite_names);
            style
                .stale_sprites
                .get_or_insert_with(Default::default)
                .extend(names);
        }
        style.paint.invalidate();
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

    /// Lays every symbol layer out again from the jobs' cached tiles, in a
    /// revision with nothing new in the document: the style's images
    /// changed. Results from before it are stale, as after an edit.
    pub(crate) fn relayout_symbols(&mut self) {
        let Some(mut style) = self.style.take() else {
            return;
        };
        self.version += 1;
        let version = self.version;
        let worker = Arc::new(revise(
            Some(&style.worker),
            &style.doc,
            &[],
            version,
            self.labels.ideographs.clone(),
            self.surface.tiles(),
        ));
        let update = StyleUpdate::base(
            &style.doc,
            version,
            self.labels.ideographs.clone(),
            self.surface.tiles(),
        );
        self.services.jobs.set_style(style.id, &update, &worker);
        style.worker = Arc::clone(&worker);
        let plan = worker.plan();
        let mut relayout: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for node in plan.nodes.iter().filter(|n| n.family.is_symbol()) {
            if let Some(state) = style.nodes.get_mut(&node.key) {
                state.layout_epoch = version;
                state.rows_epoch = version;
                relayout.entry(node.source).or_default().push(node.key);
            }
        }
        style.paint.invalidate();
        self.style = Some(style);
        for (source, nodes) in relayout {
            for tile in self.resident_tiles(source, None) {
                self.send_job(source, tile, Some(nodes.clone()), false, Some(version));
            }
        }
        self.events.wake();
        self.events.push(crate::MapEvent::StyleData {
            generation: self.version,
        });
    }

    /// The ready tiles of a source, or those with a mesh of one of `nodes`.
    pub(crate) fn resident_tiles(&self, source: usize, nodes: Option<&[u32]>) -> Vec<TileIndex> {
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
        renderer: &mut orb_render::Renderer,
        source: u32,
        tile: TileIndex,
        result: Result<TileResult, String>,
    ) {
        let Some(style) = &mut self.style else {
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
        crate::style_image::report_missing(
            &mut style.reported_missing,
            &style.icons,
            style.sprites_loading,
            &build,
            &mut self.events,
        );
        style.relayout_after_pixels |= style.awaits_sprite_pixels(&build);
        let SourceState::Ready { pyramid, .. } = &style.sources[index] else {
            return;
        };
        if pyramid.get(tile).is_none() {
            return;
        }
        let wanted = pyramid.wants(tile);
        let mut images = self.services.images.0.lock().expect("images poisoned");
        let atlas = images.atlas.as_mut().expect("initialized");
        // Room as for an arriving tile: what's free first; over the budget,
        // tiles out of view go before the pools grow (this one, if it is
        // one, ends the edit's result).
        let mut sizes = mesh_sizes(plan, &build);
        if build.rows_only {
            for size in &mut sizes {
                *size = orb_render::MeshSize {
                    props: size.props,
                    ..Default::default()
                };
            }
        }
        while !renderer.fits(&sizes) && !under_budget(renderer, atlas.bytes(), self.budget) {
            let Some(victim) = evict_unused(&mut style.sources, index) else {
                break;
            };
            self.retired_ramps
                .extend(remove_resident(renderer, victim, &mut style.states));
        }
        let grow = wanted || under_budget(renderer, atlas.bytes(), self.budget);
        let SourceState::Ready { pyramid, .. } = &mut style.sources[index] else {
            unreachable!("checked above");
        };
        let Some(resident) = pyramid.get_mut(tile) else {
            return;
        };
        // An upload that finds no room even so leaves the tile to load
        // again, as a fresh map would, rather than draw without a layer.
        let mut no_room = false;
        let tile_zoom = resident.zoom;
        let namespace = style.image_namespace.expect("namespace");
        let mut stale_rows = Vec::new();
        if build.rows_only {
            // A node's meshes split by features alike in every job, so the
            // rows' n-th mesh of a node is the tile's n-th.
            let mut seen: HashMap<u32, usize> = HashMap::new();
            for mesh in &build.meshes {
                let Some(state) = style.nodes.get(&mesh.node) else {
                    continue;
                };
                let nth = seen.entry(mesh.node).or_default();
                let current = resident
                    .meshes
                    .iter_mut()
                    .filter(|m| m.node == mesh.node)
                    .nth(*nth);
                *nth += 1;
                let Some(current) = current else {
                    continue;
                };
                // A relayout since will bring its own rows.
                if state.layout_epoch > current.layout_rev || build.revision < state.rows_epoch {
                    continue;
                }
                let props = mesh.props.words(build.host_bytes);
                let rows = resolve_rows(
                    gpu,
                    plan,
                    &build,
                    mesh,
                    atlas,
                    namespace,
                    &mut style.states,
                    &mut self.warnings,
                );
                let set = with_room(gpu, renderer, grow, |gpu, renderer| {
                    renderer.set_mesh_rows(
                        gpu,
                        current.mesh,
                        (&props, mesh.row_words),
                        &rows.rows(),
                    )
                });
                if set.is_err() {
                    rows.release(gpu, atlas, &mut style.states);
                    no_room = true;
                    break;
                }
                self.retired_ramps.append(&mut current.ramps);
                for index in current.states.drain(..) {
                    style.states.release(index);
                }
                current.ramps = rows.ramps;
                current.states = rows.states;
                current.hits = mesh.hits.clone();
                current.maxima = mesh.maxima.clone();
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
                let (old, kept) = std::mem::take(&mut resident.meshes)
                    .into_iter()
                    .partition(|m| m.node == key);
                resident.meshes = kept;
                for mut old in old {
                    renderer.remove_mesh(resident.gpu, old.mesh);
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
                    resident
                        .symbols
                        .retain(|s| s.layer != crate::style::layer_key(key));
                    continue;
                }
                for mesh in build.meshes.iter().filter(|m| m.node == key) {
                    let (states, warnings) = (&mut style.states, &mut self.warnings);
                    let uploaded = with_room(gpu, renderer, grow, |gpu, renderer| {
                        upload_mesh(
                            gpu, renderer, plan, &build, mesh, resident, atlas, namespace, states,
                            warnings,
                        )
                    });
                    if uploaded.is_err() {
                        no_room = true;
                        break;
                    }
                    self.counts.meshes_uploaded += 1;
                }
                if no_room {
                    break;
                }
            }
            build.symbols.retain(|s| {
                style
                    .nodes
                    .get(&crate::style::node_key(s.layer, 0))
                    .is_some_and(|state| build.revision >= state.layout_epoch)
            });
            let labels = &mut self.labels;
            no_room = no_room
                || with_room(gpu, renderer, grow, |gpu, renderer| {
                    labels.upload(
                        gpu,
                        renderer,
                        plan,
                        &build,
                        resident,
                        atlas,
                        namespace,
                        &mut style.states,
                        tile_zoom,
                        true,
                    )
                })
                .is_err();
            // New meshes and parts take their place in the tile order.
            style.placed[index].clear();
        }
        if no_room {
            drop(images);
            let Some(resident) = pyramid.remove(tile) else {
                unreachable!("the tile is ready");
            };
            self.retired_ramps
                .extend(remove_resident(renderer, resident, &mut style.states));
            return;
        }
        // Rows from a column job can raise a node's maxima (a radius made
        // per feature, say), which culling pads by; results skipped as
        // stale can only pad more than needed.
        record_maxima(&mut style.nodes, &build.meshes, &mut style.paint);
        resident.cost = crate::resident::tile_cost(renderer, resident);
        drop(images);
        if !stale_rows.is_empty() {
            let version = self.version;
            self.send_job(index, tile, Some(stale_rows), true, Some(version));
        }
    }
}
