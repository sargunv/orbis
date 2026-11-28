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

use rustc_hash::FxHashMap as HashMap;
use std::collections::BTreeMap;
use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_style::v8::{StyleDoc, Transaction};
use orb_style::{Binding, Change, Class, LayerKey, Patch, PropertyId};

use crate::jobs::TileOutcome;
use crate::map::{Judged, LoadedStyle, Map, NodeState, WaitingStyle, revise};
use crate::resident::{
    SourceSlot, SourceState, record_maxima, remove_resident, resolve_rows, upload_mesh,
};
use crate::style::StylePlan;
use crate::worker::{Cause, StyleUpdate, TileJob};

/// The class a paint change costs this renderer for one node:
/// `orb-style`'s, raised where property rows hold more than the property
/// (line dash and gradient textures, and the width that scales dashes) or
/// placement reads them (a symbol's own translate).
fn paint_class(layer: &orb_style::Layer, id: PropertyId, from: Binding, to: Binding) -> Class {
    let class = Change::Paint {
        layer: LayerKey(0),
        id,
        from,
        to,
    }
    .class();
    let line_rows = crate::line::has_textures(layer);
    // Placement reads a symbol's own translate from the tables its bucket
    // builds with the rows.
    let per_feature = |b: Binding| matches!(b, Binding::Source | Binding::Composite);
    let symbol_translate =
        id == orb_style::TEXT_TRANSLATE.id() || id == orb_style::ICON_TRANSLATE.id();
    if symbol_translate && (per_feature(from) || per_feature(to)) {
        return Class::Relayout;
    }
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
        // A style waiting for its modules is the one asked for last: the
        // edit goes to it, and shows once they're in.
        if let Some(WaitingStyle { doc, edits, fresh }) = &mut self.waiting_style {
            let patch = doc.apply(tx).map_err(|d| d.to_string())?;
            self.warnings
                .extend(patch.warnings.iter().map(ToString::to_string));
            *fresh |= patch.changes.contains(&Change::Restyle);
            if !patch.changes.is_empty() {
                edits.push(patch.clone());
            }
            self.take_state_edits(&patch);
            self.events.wake();
            // It may no longer need what it waited for: then it shows at
            // the version it installed at.
            let installed = self.install_waiting();
            return Ok(self.version + u64::from(!installed));
        }
        let style = self.style.as_mut().ok_or("no style is loaded")?;
        let mut doc = style.doc.clone();
        let patch = doc.apply(tx).map_err(|d| d.to_string())?;
        self.warnings
            .extend(patch.warnings.iter().map(ToString::to_string));
        self.events.wake();
        // An edit that adds a feature whose module the host hasn't loaded
        // waits for it, the style drawn as it was: no job runs its code
        // before then ([`crate::modules`]).
        let ready = self.modules.contains(crate::modules::of_style(doc.style()));
        if patch.changes.contains(&Change::Restyle) {
            self.install(doc);
            self.take_state_edits(&patch);
            // The version it's installed at, now or once the modules are in.
            return Ok(self.version + u64::from(!ready));
        }
        if !ready {
            self.waiting_style = Some(WaitingStyle {
                doc,
                edits: vec![patch.clone()],
                fresh: false,
            });
            self.take_state_edits(&patch);
            return Ok(self.version + 1);
        }
        style.doc = doc;
        self.take_state_edits(&patch);
        if patch.changes.is_empty() {
            let generation = self.generation();
            self.events.push(crate::MapEvent::StyleData { generation });
            return Ok(generation);
        }
        Ok(self.apply_patch(&patch))
    }

    /// Applies `patch`, which the drawn style's document has taken, to
    /// its plan, jobs and tiles: the version it shows at.
    pub(crate) fn apply_patch(&mut self, patch: &Patch) -> u64 {
        let style = self.style.as_mut().expect("a patch needs a style");
        // A new projection: the globe changes at once; vector tiles lay
        // out again for its curvature from the jobs' cached data, and the
        // next frame regrids image tiles (nothing loads again).
        let reproject = patch.changes.contains(&Change::Projection);
        if reproject {
            self.surface = crate::surface::Surface::of(style.doc.style());
            // The camera stands on it now: a move started after the edit
            // runs on the new surface, rather than stopping at the next frame.
            self.camera.set_surface(self.surface());
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
            self.options.fonts(),
            self.surface.tiles(),
        ));
        let update = self.edit_update(&style.doc, patch, version);
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

        self.start_transitions(patch, old.plan(), plan);
        let added = carry_sources(&mut style, old.plan(), plan);
        let (relayout, columns) = stale_nodes(
            &mut style,
            (old.plan(), plan),
            patch,
            (version, reproject, &new_data),
        );
        // Another sprite loads over the old one, next frame.
        if patch.changes.contains(&Change::Sprite) {
            style.sprites.reload();
        }
        self.style = Some(style);
        for index in added {
            let state = self.start_source(plan, index);
            self.style.as_mut().expect("restored").sources[index].state = state;
        }

        self.send_relayouts(relayout, columns, version);
        self.events.push(crate::MapEvent::StyleData {
            generation: self.version,
        });
        self.version
    }

    /// The jobs' view of an edit's `patch` of `doc`, at `version`: the
    /// layers it changed, and the sources it added or gave data.
    fn edit_update(&self, doc: &StyleDoc, patch: &Patch, version: u64) -> StyleUpdate {
        let mut layers: Vec<u32> = patch
            .changes
            .iter()
            .filter_map(|c| c.layer().map(|l| l.0))
            .collect();
        layers.dedup();
        StyleUpdate {
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
            ..StyleUpdate::base(doc, version, self.options.fonts(), self.surface.tiles())
        }
    }

    /// Starts the transitions an edit's `patch` from `old_plan` to `plan`
    /// brings: paint eases from the property before the edit, a layer
    /// shown or hidden fades, and the root's light, sky and terrain ease
    /// from what the last frame showed.
    fn start_transitions(&mut self, patch: &Patch, old_plan: &StylePlan, plan: &StylePlan) {
        let (now, zoom) = (self.frame_time, self.frame_zoom());
        if patch.changes.contains(&Change::Root) {
            self.root.start(&plan.style, zoom);
        }
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
            let transition = new.transition(*id, plan.style.transition);
            let old = &old_plan.style.layers[before];
            self.transitions
                .start(&new.id, *id, old.slot(*id), transition, (now, zoom));
        }
    }

    /// Sends the jobs that lay `relayout`'s nodes out again, and that
    /// evaluate `columns`' rows again, by source, for the resident tiles,
    /// from the jobs' cached tiles.
    fn send_relayouts(
        &mut self,
        relayout: BTreeMap<usize, Vec<u32>>,
        columns: BTreeMap<usize, Vec<u32>>,
        version: u64,
    ) {
        for (source, mut nodes) in relayout {
            nodes.sort_unstable();
            nodes.dedup();
            for tile in self.resident_tiles(source, None) {
                let job = TileJob::Relayout {
                    cause: Cause::Edit(version),
                    nodes: nodes.clone(),
                };
                self.send_edit_job(source, tile, job);
            }
        }
        for (source, mut nodes) in columns {
            nodes.sort_unstable();
            nodes.dedup();
            for tile in self.resident_tiles(source, Some(&nodes)) {
                let job = TileJob::Columns {
                    cause: Cause::Edit(version),
                    nodes: nodes.clone(),
                };
                self.send_edit_job(source, tile, job);
            }
        }
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
            self.options.fonts(),
            self.surface.tiles(),
        ));
        let update = StyleUpdate::base(
            &style.doc,
            version,
            self.options.fonts(),
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
        self.style = Some(style);
        for (source, nodes) in relayout {
            for tile in self.resident_tiles(source, None) {
                let job = TileJob::Relayout {
                    cause: Cause::Edit(version),
                    nodes: nodes.clone(),
                };
                self.send_edit_job(source, tile, job);
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
        let SourceState::Ready { pyramid, .. } = &style.sources[source].state else {
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

    /// Applies the result of `job`, a drawn tile's, to its resident tile.
    pub(crate) fn apply_edit(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut orb_render::Renderer,
        (source, tile): (u32, TileIndex),
        job: &TileJob,
        result: Result<TileOutcome, String>,
    ) {
        let Some(style) = &mut self.style else {
            return;
        };
        let plan = style.worker.plan();
        let Some(index) = plan.source(source) else {
            return;
        };
        let mut build = match result {
            // A reload that brought the data drawn again: the tile stays,
            // fresh as long as the fetch said, for good if it gave no
            // lifetime, as a new body without one does. One no server
            // answered tries again when its reload's backoff is due.
            Ok(TileOutcome::Built(build)) if build.renewed => {
                let SourceState::Ready { pyramid, .. } = &mut style.sources[index].state else {
                    return;
                };
                if let Some(resident) = pyramid.get_mut(tile)
                    && !build.stale
                {
                    resident.renew(build.fresh_until);
                }
                return;
            }
            Ok(TileOutcome::Built(build)) => build,
            // The tile's data went: it stays as it was until it reloads.
            Ok(TileOutcome::Missing { mut warnings }) => {
                self.warnings.append(&mut warnings);
                return;
            }
            Ok(TileOutcome::Cancelled) => return,
            // A reload its server puts off is tried again no sooner than
            // it asked, as a first load is.
            Ok(TileOutcome::RetryAfter { error, wait_ms }) => {
                self.warnings.push(format!("tile {tile}: {error}"));
                if *job == TileJob::Revalidate
                    && let SourceState::Ready { pyramid, .. } = &mut style.sources[index].state
                    && let Some(resident) = pyramid.get_mut(tile)
                {
                    resident.reloads.defer(self.services.clock.now(), wait_ms);
                }
                return;
            }
            Err(e) => {
                self.warnings.push(format!("tile {tile}: {e}"));
                return;
            }
        };
        self.warnings.append(&mut build.warnings);
        crate::style_image::report_missing(
            &mut style.reported_missing,
            &style.icons,
            style.sprites.loading,
            &build,
            &mut self.events,
        );
        let SourceState::Ready { pyramid, .. } = &style.sources[index].state else {
            return;
        };
        if pyramid.get(tile).is_none() {
            return;
        }
        let mut images = self.services.images.0.lock().expect("images poisoned");
        let atlas = images.atlas.as_mut().expect("initialized");
        let SourceState::Ready { pyramid, .. } = &mut style.sources[index].state else {
            unreachable!("checked above");
        };
        let Some(resident) = pyramid.get_mut(tile) else {
            return;
        };
        // An upload that finds no room within the device's limits leaves
        // the tile to load again, as a fresh map would, rather than draw
        // without a layer.
        let mut no_room = false;
        let tile_zoom = resident.zoom;
        let namespace = style.image_namespace.expect("namespace");
        let mut stale_rows = Vec::new();
        if matches!(job, TileJob::Columns { .. }) {
            // A node's meshes split by features alike in every job, so the
            // rows' n-th mesh of a node is the tile's n-th.
            let mut seen: HashMap<u32, usize> = HashMap::default();
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
                // A relayout since will bring its own rows, as will a
                // later job.
                if state.layout_epoch > current.layout_rev
                    || state.judge(build.revision) != Judged::Current
                {
                    continue;
                }
                let props = mesh.props.words(&build.payload);
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
                let set = renderer.set_mesh_rows(
                    gpu,
                    current.mesh,
                    (&props, mesh.row_words),
                    &rows.rows(),
                );
                if set.is_err() {
                    rows.release(gpu, atlas, &mut style.states);
                    no_room = true;
                    break;
                }
                self.retired.ramps.append(&mut current.ramps);
                for index in current.states.drain(..) {
                    style.states.release(index);
                }
                current.ramps = rows.ramps;
                current.states = rows.states;
                current.hits = mesh.hits.clone();
                current.maxima = mesh.maxima.clone();
                self.counts.rows_uploaded += 1;
            }
        } else {
            // A whole tile's data loaded again (it expired) brings its
            // lifetime, if it has one.
            if *job == TileJob::Revalidate {
                resident.renew(build.fresh_until);
            }
            // Whether its labels drew a failed glyph range: as the build
            // has it where it laid out every symbol layer, else either.
            let symbols = (plan.nodes.iter()).filter(|n| n.source == index && n.family.is_symbol());
            let mut laid_out = symbols.map(|n| build.nodes.contains(&n.key)).peekable();
            if laid_out.peek().is_some() {
                let had_failed = resident.glyphs_failed;
                resident.glyphs_failed = match laid_out.all(|laid| laid) {
                    true => build.glyphs_failed,
                    false => had_failed || build.glyphs_failed,
                };
                let now = self.services.clock.now();
                (style.glyph_retry).arrived(resident.glyphs_failed, had_failed, now);
            }
            // The old meshes' state rows, released once the new ones hold
            // theirs: a fade in progress carries over.
            let mut released = Vec::new();
            for &key in &build.nodes {
                let Some(state) = style.nodes.get(&key) else {
                    continue;
                };
                // A symbol layer's group comes back whole or not at all:
                // its layers draw one bucket.
                let newer = |key: &u32| {
                    (style.nodes.get(key))
                        .is_some_and(|s| s.judge(build.revision) == Judged::LayoutStale)
                };
                if newer(&key) || plan.with_layout_groups(&[key]).iter().any(newer) {
                    continue;
                }
                if state.judge(build.revision) == Judged::RowsStale {
                    stale_rows.push(key);
                }
                let (old, kept) = std::mem::take(&mut resident.meshes)
                    .into_iter()
                    .partition(|m| m.node == key);
                resident.meshes = kept;
                for mut old in old {
                    renderer.remove_mesh(resident.gpu, old.mesh);
                    released.append(&mut old.states);
                    self.retired.ramps.append(&mut old.ramps);
                }
                // A symbol layer comes back whole with its group's bucket,
                // below.
                if plan
                    .node(key)
                    .is_some_and(|i| plan.nodes[i].family.is_symbol())
                {
                    let layer = crate::style::layer_key(key);
                    crate::labels::drop_symbols(renderer, resident, &mut style.states, &[layer]);
                    continue;
                }
                for mesh in build.meshes.iter().filter(|m| m.node == key) {
                    let (states, warnings) = (&mut style.states, &mut self.warnings);
                    let uploaded = upload_mesh(
                        gpu, renderer, plan, &build, mesh, resident, atlas, namespace, states,
                        warnings,
                    );
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
            // A group's bucket is stale once any of its layers is.
            build.symbols.retain(|s| {
                s.layers.iter().all(|&layer| {
                    (style.nodes.get(&crate::style::node_key(layer, 0)))
                        .is_some_and(|state| state.judge(build.revision) != Judged::LayoutStale)
                })
            });
            no_room = no_room
                || (self.labels)
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
                        true,
                    )
                    .is_err();
            for index in released {
                style.states.release(index);
            }
        }
        if no_room {
            drop(images);
            let Some(resident) = pyramid.remove(tile) else {
                unreachable!("the tile is ready");
            };
            self.retired
                .ramps
                .extend(remove_resident(renderer, resident, &mut style.states));
            return;
        }
        // Rows from a column job can raise a node's maxima (a radius made
        // per feature, say), which culling pads by; results skipped as
        // stale can only pad more than needed.
        record_maxima(&mut style.nodes, &build.meshes, &mut style.maxima);
        resident.cost = crate::resident::tile_cost(renderer, resident);
        drop(images);
        // Labels a whole relayout left out while their glyphs loaded
        // follow, as a load's do.
        let later: Vec<u32> = (build.labels_later.iter().copied())
            .filter(|&key| plan.node(key).is_some())
            .collect();
        if !later.is_empty() {
            let cause = Cause::Upkeep;
            self.send_edit_job(
                index,
                tile,
                TileJob::Relayout {
                    cause,
                    nodes: later,
                },
            );
        }
        if !stale_rows.is_empty() {
            let job = TileJob::Columns {
                cause: Cause::Edit(self.version),
                nodes: stale_rows,
            };
            self.send_edit_job(index, tile, job);
        }
    }
}

/// Carries `style`'s sources over an edit from `old_plan` to `plan`, by
/// key: removed ones leave the scene next frame. Returns the indices of
/// those added, which start loading.
fn carry_sources(style: &mut LoadedStyle, old_plan: &StylePlan, plan: &StylePlan) -> Vec<usize> {
    let old_keys: Vec<u32> = old_plan.sources.iter().map(|s| s.key).collect();
    let mut old_sources: Vec<Option<SourceSlot>> = std::mem::take(&mut style.sources)
        .into_iter()
        .map(Some)
        .collect();
    let mut added = Vec::new();
    for (index, source) in plan.sources.iter().enumerate() {
        let carried =
            (old_keys.iter().position(|k| *k == source.key)).and_then(|i| old_sources[i].take());
        style.sources.push(carried.unwrap_or_else(|| {
            added.push(index);
            SourceSlot::new(SourceState::Failed)
        }));
    }
    (style.retired).extend(old_sources.into_iter().flatten().map(|slot| slot.state));
    added
}

/// Marks `style`'s nodes an edit's `patch` from `old_plan` to `plan` (the
/// style's `version`) left stale, by key: new nodes everywhere, and those
/// a change, a projection (`reproject`) or a source's `new_data` touched.
/// Returns, by source, the nodes to lay out again and those whose rows
/// alone are evaluated again.
#[allow(clippy::type_complexity)]
fn stale_nodes(
    style: &mut LoadedStyle,
    (old_plan, plan): (&StylePlan, &StylePlan),
    patch: &Patch,
    (version, reproject, new_data): (u64, bool, &[u32]),
) -> (BTreeMap<usize, Vec<u32>>, BTreeMap<usize, Vec<u32>>) {
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
            let before = old_plan.node(node.key).map(|i| &old_plan.nodes[i]);
            let class = match change {
                Change::Paint { id, from, to, .. } => {
                    let class = paint_class(layer, *id, *from, *to);
                    // Meshes need feature ids to read state.
                    let ids =
                        plan.reads_state(node) && !before.is_some_and(|b| old_plan.reads_state(b));
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
    // A symbol bucket is its group's: a group that gained or lost a layer
    // lays out again whole, and buckets laid out for it before are stale.
    let group_of = |plan: &StylePlan, leader: u32| {
        let at = plan.layer_keys.iter().position(|&k| k == leader)?;
        let mut group = plan.layout_group(at);
        group.sort_unstable();
        (plan.layout_leaders[at].is_none()).then_some(group)
    };
    let symbols = plan
        .nodes
        .iter()
        .filter(|n| n.family == orb_render::Family::SymbolIcon);
    for node in symbols {
        let leader = plan.layer_keys[node.layer];
        let group = group_of(plan, leader);
        if group.is_none() || group == group_of(old_plan, leader) {
            continue;
        }
        for key in plan.with_layout_groups(&[node.key]) {
            let state = style.nodes.get_mut(&key).expect("added above");
            if state.layout_epoch != version {
                state.layout_epoch = version;
                state.rows_epoch = version;
                relayout.entry(node.source).or_default().push(key);
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
    (relayout, columns)
}
