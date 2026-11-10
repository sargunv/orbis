//! The frame's plan: which tiles cover the
//! view and stand in where they haven't arrived, where they are placed,
//! the background's quads, and which drape slots draw again.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use glam::DVec2;
use orb_camera::View;
use orb_crs::{TileIndex, TileMatrixSet};
use orb_hal::Gpu;
use orb_render::{ClipRect, DemRef, Family, PlacementTarget, Renderer, TileRow, TileWarp};
use orb_source::Draw;

use crate::map::*;
use crate::meshes::{image_mesh, series_patches};
use crate::resident::*;

impl Map {
    /// What a frame draws and loads of each source, over the terrain's
    /// `elevation`: a function of the view
    /// and the resident tiles, which changes nothing.
    pub(crate) fn plan_frame(
        &self,
        renderer: &Renderer,
        view: &View,
        elevation: Option<&crate::terrain::TerrainElevation>,
    ) -> Vec<Option<SourcePlan>> {
        let sources = self.style.as_ref().map_or(0, |s| s.sources.len());
        let at = |index| self.plan_source(renderer, view, index, elevation);
        (0..sources).map(at).collect()
    }

    /// Source `index`'s plan: the tiles covering the view, what of them is
    /// resident to draw and what to load; `None` for a source that draws
    /// nothing this frame.
    fn plan_source(
        &self,
        renderer: &Renderer,
        view: &View,
        index: usize,
        elevation: Option<&crate::terrain::TerrainElevation>,
    ) -> Option<SourcePlan> {
        let style = self.style.as_ref()?;
        let plan = style.worker.plan();
        // A source only the terrain reads.
        if !plan.nodes.iter().any(|n| n.source == index) {
            return None;
        }
        let zoom = self.camera.state().zoom(self.surface.projection());
        // The tallest extrusion or the highest a symbol layer lifts its
        // labels (`symbol-height-offset`): their tiles stay covered while
        // the lifted content can show, as in GL JS. Per feature, the most
        // the expression can give at this zoom, which holds for every tile
        // however the camera got here; one the expression doesn't bound (a
        // bare `get`) is read from the tiles below, once the cover is known.
        // (GL JS keeps the highest any loaded tile has held, which depends
        // on the camera's path.) Otherwise the height as this frame's paint
        // has it.
        let mut unbounded = Unbounded::default();
        let height = (plan.nodes.iter().enumerate())
            .filter(|(_, n)| n.source == index)
            .map(|(i, n)| {
                let layer = &plan.style.layers[n.layer];
                if n.family.is_symbol() {
                    let lift = layer.get(orb_style::SYMBOL_HEIGHT_OFFSET).max_value(zoom);
                    if lift.is_none() {
                        unbounded.labels.push(crate::style::layer_key(n.key));
                    }
                    let lift = lift.unwrap_or(0.0).max(0.0) as f32;
                    return renderer.cull_elevation(i as u32).max(lift);
                }
                // A per-feature extrusion's paint height is the node's row
                // maxima, raised by every tile laid out so far: its bound
                // stands in.
                match (n.family == Family::FillExtrusion)
                    .then(|| extrusion_bound(layer, zoom))
                    .flatten()
                {
                    Some((tallest, bounded)) => {
                        if !bounded {
                            unbounded.extrusions.push(n.key);
                        }
                        tallest
                    }
                    None => renderer.cull_elevation(i as u32),
                }
            })
            .fold(0.0f32, f32::max);
        // Circles and heatmaps draw from the tile their point is in alone,
        // where GL JS draws the copies in its neighbours' buffers too: a
        // neighbour out of view still reaches into it by their reach.
        // Labels, as in GL JS, show from their anchor's tile only.
        let reach = plan
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                n.source == index && matches!(n.family, Family::Circle | Family::Heatmap)
            })
            .map(|(i, _)| renderer.cull_margin_px(i as u32))
            .fold(0.0f32, f32::max);
        let SourceState::Ready {
            endpoint,
            pyramid,
            warped,
            ..
        } = &style.sources[index]
        else {
            return None;
        };
        // A source whose layers all hide at this zoom draws and loads
        // nothing (GL JS's `used`), and its tiles stay cached. Draped
        // layers on terrain draw at their slots' zooms.
        let draping = renderer.terrain_on();
        if !plan.nodes.iter().any(|n| {
            let layer = &plan.style.layers[n.layer];
            n.source == index
                && (layer.is_visible_at(zoom)
                    || self.transitions.fading_out(&layer.id)
                    || (draping
                        && n.family.is_draped()
                        && layer.visibility == orb_style::Visibility::Visible))
        }) {
            return Some(SourcePlan::of(pyramid.plan(&[])));
        }
        // A hillshade the terrain shades drapes its slopes: its source
        // covers each terrain tile's quarters, whatever detail the
        // terrain's cover keeps, so that a slot holds the source's detail
        // at its drape zoom; its other layers, all draped, draw into the
        // terrain tiles' slots from any cover.
        let nodes = || (plan.nodes.iter().enumerate()).filter(|(_, n)| n.source == index);
        let shaded = nodes().any(|(i, _)| renderer.shades_on_terrain(i as u32))
            && nodes().all(|(_, n)| n.family.is_draped());
        if let Some(terrain) = style.terrain.as_ref().filter(|_| shaded) {
            let max = pyramid.max_zoom();
            let mut cover: Vec<orb_crs::CoveredTile> = (terrain.tiles.iter())
                .flat_map(|t| {
                    (t.index.children()).map(|q| orb_crs::CoveredTile {
                        index: crate::terrain::scaled(q, q.z.min(max)),
                        wrap: t.wrap,
                    })
                })
                .collect();
            cover.sort();
            cover.dedup();
            return Some(SourcePlan::of(pyramid.plan(&cover)));
        }
        let tms = &endpoint.tms;
        // Tile mode draws the middle tile of its grid and places labels over
        // the whole grid: a source whose content crosses tile edges (labels,
        // circles, heatmaps; extrusions and fill outlines at the edges)
        // covers the grid, as does a DEM source, whose tiles' borders
        // (slopes at the edges) come from their drawn neighbours. The
        // grid's edges are tile edges: tiles that only touch it hold
        // nothing the middle shows. Of a vector source's tiles around the
        // middle, only what reaches into it is needed: they lay out that
        // alone (`neighbours`, `TileRequest::neighbour`).
        let grid_view;
        let (cover_view, margin, middle) = match &self.mode {
            RenderMode::Tile(_) => {
                let dem = endpoint.dem_unpack.is_some();
                let crossing =
                    nodes().any(|(_, n)| n.family.crosses_tiles() || n.family.reaches_edges());
                let middle = view.viewport();
                grid_view = self.view(orb_camera::Viewport {
                    width: 3.0 * middle.width,
                    height: 3.0 * middle.height,
                });
                match (dem, crossing) {
                    (true, _) => (&grid_view, -0.5, None),
                    (false, true) => (&grid_view, -0.5, Some(view)),
                    (false, false) => (view, -0.5, None),
                }
            }
            _ => (view, f64::from(reach), None),
        };
        let warps = &self.globe_warps;
        let volume = |view: &View, height: f32| {
            view.cover_volume(
                tms,
                endpoint.cover_px,
                pyramid.max_zoom(),
                f64::from(height) * self.extrusion_scale(),
                elevation.map(|e| e as &dyn orb_camera::Elevation),
                margin,
            )
        };
        // Heights no expression bounds: the highest the covered tiles hold,
        // as each laid out, widens the cover. It comes from this cover's own
        // tiles, not from what loaded before, so a map reached by a gesture
        // covers what a fresh one does; in Tile mode from the middle's
        // alone, as its neighbours lay out only what crosses into it.
        let lifted = |cover: &[orb_crs::CoveredTile]| {
            (cover.iter())
                .filter(|_| !unbounded.is_empty())
                .filter_map(|t| pyramid.get(t.index))
                .map(|tile| unbounded.tallest(tile))
                .fold(height, f32::max)
        };
        let cover_of = |view: &View| match (view.low_plane(), *warped) {
            // On a low-zoom plane every set's tiles land through its warp.
            (Some(low), _) => {
                low.cover_warped(tms, endpoint.cover_px, pyramid.max_zoom(), |tile| {
                    warps.footprint(tms, tile)
                })
            }
            (None, false) => {
                let cover = volume(view, height);
                match lifted(&cover) {
                    lift if lift > height => volume(view, lift),
                    _ => cover,
                }
            }
            (None, true) => view.cover_warped(tms, endpoint.cover_px, pyramid.max_zoom(), |tile| {
                warps.surface_footprint(tms, tile)
            }),
        };
        let Some(middle) = middle else {
            return Some(SourcePlan {
                update: pyramid.plan(&cover_of(cover_view)),
                neighbours: Vec::new(),
            });
        };
        let drawn = cover_of(middle);
        let cover = match (cover_view.low_plane(), *warped) {
            (None, false) => volume(cover_view, lifted(&drawn)),
            _ => cover_of(cover_view),
        };
        let neighbours = (cover.iter())
            .map(|t| t.index)
            .filter(|&t| !drawn.iter().any(|d| d.index == t))
            .collect();
        Some(SourcePlan {
            update: pyramid.plan(&cover),
            neighbours,
        })
    }

    /// Applies source `index`'s plan: its pyramid records it, its requests
    /// load (those with nothing to draw are empty at once), and its draws
    /// wait for [`place_source`](Self::place_source).
    pub(crate) fn apply_source(
        &mut self,
        renderer: &mut Renderer,
        index: usize,
        plan: Option<SourcePlan>,
        report: &mut FrameReport,
    ) {
        let style = self.style.as_mut().expect("called with a style");
        style.draws.resize_with(style.sources.len(), || None);
        style.draws[index] = None;
        let Some(SourcePlan {
            mut update,
            neighbours,
        }) = plan
        else {
            return;
        };
        // Tiles held for their labels stay until a placement covers what
        // replaced them ([`place_source`](Self::place_source)).
        update
            .used
            .extend(style.held.get(index).into_iter().flatten().copied());
        let vector = style.worker.plan().image_kind(index).is_none();
        let SourceState::Ready {
            endpoint,
            pyramid,
            loading,
            warped,
            ..
        } = &mut style.sources[index]
        else {
            return;
        };
        // A tile laid out as a neighbour that is drawn now loads again
        // whole, as a fresh map's would; its job finds the tile's data
        // decoded already.
        for &tile in &update.cover {
            if neighbours.contains(&tile) || !pyramid.get(tile).is_some_and(|r| r.neighbour) {
                continue;
            }
            let resident = pyramid.remove(tile).expect("checked above");
            (self.retired_ramps).extend(remove_resident(renderer, resident, &mut style.states));
            update.draw.retain(|d| d.tile != tile);
            update.request.push(tile);
        }
        pyramid.commit(&update);
        let tms = &endpoint.tms;
        let mut requests = Vec::new();
        for &tile in &update.request {
            // Vector tiles on another plane need a series to draw through:
            // planar patches, or the sphere's on a globe.
            let unwarped = vector
                && *warped
                && self.globe_warps.surface_series(tms, tile).is_none()
                && (self.surface.globe().is_none() || self.globe_warps.series(tms, tile).is_none());
            // Outside the source's bounds there is no data, as in GL JS:
            // nothing draws there, not even an ancestor.
            if !endpoint.contains(tile) || unwarped {
                pyramid.insert_empty(tile);
                continue;
            }
            *loading += 1;
            requests.push(tile);
        }
        report.tiles_loading += *loading;
        // Loads nothing wants now give way to those the view waits for.
        let unwanted = pyramid.unwanted_loads();
        style.draws[index] = Some(update.draw);
        let (id, key) = (style.id, style.worker.plan().sources[index].key);
        for tile in unwanted {
            self.services.jobs.cancel(id, key, tile);
        }
        for tile in requests {
            let neighbour = neighbours.contains(&tile);
            self.send_tile_job(index, tile, (None, false, neighbour), None);
        }
    }

    /// Places source `index`'s tiles for this frame: map placements
    /// standing on the terrain, and drape placements into the drape slots
    /// being redrawn.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn place_source(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        view: &View,
        index: usize,
        drape: Option<&DrapePlan>,
        report: &mut FrameReport,
        drawn: &mut Vec<crate::query::DrawnTile>,
    ) {
        let style = self.style.as_mut().expect("called with a style");
        let Some(draws) = style.draws[index].take() else {
            return;
        };
        let draped = drape.filter(|_| {
            let plan = style.worker.plan();
            plan.nodes
                .iter()
                .any(|n| n.source == index && n.family.is_draped())
        });
        // The drape stacks its layers draw into: its rows go into those
        // slots alone.
        let mut stacks: Vec<u8> = (style.worker.plan().nodes.iter().enumerate())
            .filter(|(_, n)| n.source == index)
            .filter_map(|(i, _)| renderer.drape_stack(i as u32))
            .collect();
        stacks.sort_unstable();
        stacks.dedup();
        let terrain = style.terrain.as_ref();
        let source_key = style.worker.plan().sources[index].key;
        let SourceState::Ready {
            pyramid,
            endpoint,
            warped,
            ..
        } = &mut style.sources[index]
        else {
            return;
        };
        let endpoint = Arc::clone(endpoint);
        let tms = &endpoint.tms;
        if endpoint.dem_unpack.is_some() {
            renderer.pad_dems(source_key, &draws.iter().map(|d| d.tile).collect());
        }
        let mut rows: HashMap<TileIndex, Vec<&Draw>> = HashMap::new();
        // Picking reads tiles with features in drawing order, ancestors
        // before descendants.
        let mut order = Vec::new();
        for draw in &draws {
            let copies = rows.entry(draw.tile).or_default();
            if copies.is_empty() {
                order.push(draw.tile);
            }
            copies.push(draw);
        }
        for tile in order {
            let resident = pyramid.get(tile).expect("drawn tiles are ready");
            if resident.meshes.iter().any(|m| !m.hits.is_empty()) {
                drawn.push(crate::query::DrawnTile::new(
                    source_key,
                    (tile, endpoint.data_tile(tile)),
                    &rows[&tile],
                    tms,
                    resident.series.as_deref(),
                ));
            }
        }
        // Each drawn image tile's mesh is the one for this frame's surface
        // and, for a warped tile, the finest drawn tile beside each side,
        // whose vertices it takes so the two meet without cracks.
        let globe = self.surface.globe().is_some();
        for &tile in rows.keys() {
            let edges = match *warped && !globe {
                true => edge_refinement(tile, |t| rows.contains_key(&t)),
                false => [0; 4],
            };
            let resident = pyramid.get_mut(tile).expect("drawn tiles are ready");
            let Some(image) = resident.image.as_mut() else {
                continue;
            };
            if image.globe == Some(globe)
                && image.warped.as_ref().map_or([0; 4], |w| w.edges) == edges
            {
                continue;
            }
            let Ok(mesh) = image_mesh(
                &mut self.globe_warps,
                globe,
                *warped,
                &endpoint,
                tile,
                (&mut image.grid, edges),
            ) else {
                continue;
            };
            image.warped = mesh;
            // A warped image draws through planar patches only on a globe
            // surface; elsewhere through its warp's mesh.
            resident.series = (*warped && globe)
                .then(|| self.globe_warps.surface_series(tms, tile))
                .flatten();
            let patches = resident.series.as_deref().map(series_patches);
            let patches = patches.as_ref().map(|(cells, p)| (*cells, &p[..]));
            let mesh = image.warped.as_ref().map(|w| Arc::clone(&w.mesh));
            let set = renderer.set_image_mesh(gpu, resident.gpu, mesh, patches);
            image.globe = set.is_ok().then_some(globe);
            if set.is_err() {
                self.warnings
                    .push("out of scene memory; an image layer misses tiles".into());
            }
        }
        let period = self.surface.period();
        let whole = whole_draws(tms, &draws);
        let placed = &mut style.placed[index];
        // Labels past a Web Mercator tile go on exactly on a globe.
        let on_mercator = view.globe().is_some() && *tms.projection() == mercator_projection();
        for (&tile, draws) in &rows {
            let resident = pyramid.get(tile).expect("drawn tiles are ready");
            let globe = view
                .globe()
                .and_then(|_| self.globe_warps.series(tms, tile));
            // Each world copy places its labels, as GL JS's wrapped tiles do.
            for symbols in &resident.symbols {
                let mut wraps: Vec<i32> = draws.iter().map(|d| d.wrap).collect();
                wraps.sort_unstable();
                wraps.dedup();
                for wrap in wraps {
                    let to_world = match &resident.series {
                        Some(series) => series.linear(orb_symbol::EXTENT),
                        None => tms.data_to_world(tile, orb_symbol::EXTENT as u32),
                    };
                    style.symbol_tiles.push(crate::labels::SymbolTile {
                        to_world,
                        layer: symbols.layer,
                        bucket: symbols.bucket,
                        tile,
                        wrap,
                        mercator: on_mercator && resident.series.is_none(),
                        globe: globe.clone(),
                    });
                }
            }
            let frame = resident
                .image
                .as_ref()
                .and_then(|image| image.warped.as_ref())
                .map(|warped| warped.frame);
            let ground = terrain.map(|t| t.ground(tile));
            let series = resident.series.clone();
            // A tile on another plane has a place on a globe surface's plane
            // through its planar patches; without them (one at a pole, say)
            // it draws on the globe alone. On a plane surface every tile has
            // one: a warped image draws through its warp's mesh.
            let on_plane = self.surface.globe().is_none() || !*warped || series.is_some();
            // On a globe, a tile draws through its patches on the sphere.
            let globe = match view.globe() {
                Some(_) => self
                    .globe_warps
                    .warp(gpu, renderer, tms, tile)
                    .map(|(warp, anchor)| Some((warp, anchor))),
                None => on_plane.then_some(None),
            };
            // Nowhere to draw it this frame: no placements.
            let Some(globe) = globe else {
                renderer.unplace(resident.gpu);
                continue;
            };
            for extent in renderer.tile_extents(resident.gpu) {
                let at = Placement {
                    extent,
                    period,
                    zoom: self.camera.state().zoom(self.surface.projection()),
                    frame,
                    series: series.as_deref(),
                    globe,
                    on_plane,
                    tile_zoom: resident.zoom,
                };
                let map_rows = |draw: &Draw| -> Vec<TileRow> {
                    let neighbours = neighbours(tms, draw, &whole, period != 0.0);
                    placements(tms, draw, at)
                        .map(|mut row| {
                            if let Some((dem, range)) = ground {
                                row.dem = dem;
                                row.terrain_range = range;
                            }
                            row.neighbours = neighbours;
                            row
                        })
                        .collect()
                };
                let mut rows: Vec<TileRow> = draws.iter().flat_map(|d| map_rows(d)).collect();
                if let Some(drape) = draped {
                    draws
                        .iter()
                        .for_each(|d| drape.rows(tms, d, (at, &stacks), &mut rows));
                }
                report.tiles_drawn += rows.len() as u32;
                if renderer.place(resident.gpu, extent, &rows).is_err() {
                    self.warnings
                        .push("out of tile placements; some tiles are not drawn".into());
                }
            }
        }
        // Tiles held for their labels: drawn again, gone, or done.
        let held = &mut style.held[index];
        let now = self.services.clock.now();
        held.retain(|tile| {
            let resident = pyramid.get(*tile);
            let buckets = || {
                resident
                    .into_iter()
                    .flat_map(|r| &r.symbols)
                    .map(|s| s.bucket)
            };
            let done = rows.contains_key(tile)
                || resident.is_none()
                || buckets().all(|b| self.labels.released(b, now));
            if done {
                for bucket in buckets() {
                    self.labels.unhold(bucket);
                }
                if let Some(r) = resident.filter(|_| !rows.contains_key(tile)) {
                    renderer.unplace(r.gpu);
                }
            }
            !done
        });
        // Tiles leaving the draw list: those with labels keep drawing them
        // until a placement covers what replaced them. Still output waits
        // for placement anyway, and fades nothing.
        let holds = self.mode == RenderMode::Continuous;
        for tile in placed.iter().filter(|t| !rows.contains_key(t)) {
            if let Some(resident) = pyramid.get(*tile) {
                if resident.symbols.is_empty() || !holds {
                    renderer.unplace(resident.gpu);
                } else {
                    renderer.hold_symbols(resident.gpu);
                    for s in &resident.symbols {
                        self.labels.hold(s.bucket);
                    }
                    held.push(*tile);
                }
            }
        }
        // Evicted sooner, held labels would vanish for the frames until
        // that placement commits: arrivals that need room take other tiles.
        for &tile in held.iter() {
            pyramid.keep(tile);
        }
        let mut ordered = Vec::new();
        for draw in &draws {
            // Multiple wraps of a tile share its meshes and placements.
            if ordered.last() != Some(&draw.tile) {
                ordered.push(draw.tile);
            }
        }
        if *placed != ordered {
            let tiles: Vec<_> = ordered
                .iter()
                .map(|tile| pyramid.get(*tile).expect("drawn tiles are ready").gpu)
                .collect();
            renderer.set_tile_order(&tiles);
            *placed = ordered;
        }
    }

    /// Keeps the tiles out of view within the cache budget, a quarter of the
    /// memory budget: the least recently used go first, from any source.
    /// The scene's pools then hold the tiles in view and what's left of the
    /// cache, and arrivals that need room evict more (`evict_unused`).
    pub(crate) fn trim_cache(&mut self, renderer: &mut Renderer) {
        let Some(style) = &mut self.style else {
            return;
        };
        let budget = self.budget / 4;
        // (age, source, tile, cost), oldest first.
        let mut unused: Vec<(u64, usize, TileIndex, u64)> = Vec::new();
        for (index, source) in style.sources.iter().enumerate() {
            if let SourceState::Ready { pyramid, .. } = source {
                unused.extend(
                    pyramid
                        .unused()
                        .map(|(tile, age, r)| (age, index, tile, r.cost)),
                );
            }
        }
        let mut total: u64 = unused.iter().map(|u| u.3).sum();
        if total <= budget {
            return;
        }
        unused.sort_unstable_by_key(|u| std::cmp::Reverse(u.0));
        for (_, index, tile, cost) in unused {
            if total <= budget {
                break;
            }
            let SourceState::Ready { pyramid, .. } = &mut style.sources[index] else {
                continue;
            };
            if let Some(resident) = pyramid.evict_unused(tile) {
                total -= cost;
                self.retired_ramps
                    .extend(remove_resident(renderer, resident, &mut style.states));
            }
        }
    }

    /// Plans this frame's drape: makes room for the terrain tiles' slots,
    /// writes the draped paint at their drape zooms and picks the slots to
    /// redraw, by what is draped under each tile. Nothing
    /// here depends on the camera but which tiles there are, and the
    /// bearing under a viewport-anchored translate. Viewport-anchored
    /// hillshades turn with the map too: they drape their slopes, which
    /// the terrain pass lights from the frame's bearing.
    pub(crate) fn plan_drape(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        view: &View,
    ) -> Option<DrapePlan> {
        let time = self.shader_time();
        let Some(style) = &mut self.style else {
            return None;
        };
        let terrain = style.terrain.as_mut()?;
        if !renderer.terrain_on() {
            return None;
        }
        let worker = Arc::clone(&style.worker);
        let plan = worker.plan();
        let kept = terrain.kept_slots();
        let slots = (terrain.tiles.len() as u32, &kept[..]);
        let (layout, moves) = match renderer.reserve_drape(gpu, slots) {
            Ok(reserved) => reserved,
            Err(_) => {
                // The cover fits the atlas's room (`Terrain::update`)
                // unless its stacks leave none for the view's tiles.
                self.warnings
                    .push("too many drape stacks for the drape atlas; terrain is flat".into());
                renderer.set_terrain(gpu, None);
                return None;
            }
        };
        let surface = self.surface.tms().clone();
        let zooms: BTreeSet<u8> = terrain
            .tiles
            .iter()
            .map(|t| crate::terrain::drape_zoom(&surface, t.index))
            .collect();
        let bearing = self.camera.state().bearing;
        let mut images = self.services.images.0.lock().expect("images poisoned");
        let atlas = images.atlas.as_mut().expect("initialized");
        let at = (
            layout.generation,
            crate::paint::camera_inputs(self.camera.state()),
        );
        crate::paint::drape(gpu, renderer, (style, plan), &zooms, at, atlas);
        let images_revision = atlas.revision();
        drop(images);
        let draped: Vec<usize> = (0..style.sources.len())
            .filter(|&i| (plan.nodes.iter()).any(|n| n.source == i && n.family.is_draped()))
            .collect();
        let period = self.surface.period();
        let sources = &style.sources;
        let draws = &style.draws;
        let paint = &style.paint;
        let revision = (self.version, style.states.writes(time), images_revision);
        // The farthest a draped fill or line is translated at a drape
        // zoom: a slot takes rows from the drawn tiles that far around it,
        // whose content moves into it.
        let reach_px = (plan.nodes.iter())
            .filter(|n| n.family.is_draped())
            .flat_map(|n| {
                let slots = plan.paint_properties(n).len();
                let reach = style.nodes.get(&n.key).and_then(|s| s.maxima.get(slots));
                zooms.iter().map(move |&z| {
                    let [x, y] = crate::paint::translate(plan, n, f64::from(z), reach).px;
                    f64::from(x.hypot(y))
                })
            })
            .fold(0.0, f64::max);
        // A DEM tile's border comes from its drawn neighbours
        // (`Renderer::pad_dems`).
        let drawn_tiles: HashMap<usize, HashSet<TileIndex>> = (draped.iter())
            .filter_map(|&i| Some((i, draws[i].as_ref()?.iter().map(|d| d.tile).collect())))
            .collect();
        // Each draped source's stacks: a slot's stack redraws when what
        // the sources drawing into it drape there changes.
        let stacks = layout.stack_count() as usize;
        let mut source_stacks = vec![Vec::new(); style.sources.len()];
        for (index, node) in plan.nodes.iter().enumerate() {
            if let (Some(stack), Some(stacks)) = (
                renderer.drape_stack(index as u32),
                source_stacks.get_mut(node.source),
            ) {
                stacks.push(usize::from(stack));
            }
        }
        let content = |t: orb_crs::CoveredTile| {
            let zoom = crate::terrain::drape_zoom(&surface, t.index);
            let mut parts = vec![Vec::new(); stacks];
            for &i in &draped {
                let (
                    Some(draws),
                    SourceState::Ready {
                        pyramid, endpoint, ..
                    },
                ) = (&draws[i], &sources[i])
                else {
                    continue;
                };
                for draw in draws {
                    let resident = pyramid.get(draw.tile);
                    let frame = resident
                        .and_then(|r| r.image.as_ref())
                        .and_then(|image| image.warped.as_ref())
                        .map(|warped| warped.frame);
                    let drawn = crate::terrain::Drawn::of(&endpoint.tms, draw, frame, period);
                    if !drawn.overlaps(&surface, t, reach_px) {
                        continue;
                    }
                    // The renderer's tiles stand for their meshes.
                    let meshes: Vec<_> = resident
                        .map(|r| r.meshes.iter().map(|m| m.mesh).collect())
                        .unwrap_or_default();
                    let held = resident.map(|r| r.gpu);
                    // A DEM tile's borders come from its neighbours.
                    let borders = endpoint.dem_unpack.map(|_| {
                        let drawn = &drawn_tiles[&i];
                        orb_render::dem_neighbours(draw.tile, &|n| {
                            (pyramid.get(n).filter(|_| drawn.contains(&n)))
                                .map(|r| crate::terrain::hash(r.gpu) as u32)
                        })
                    });
                    let part = (
                        i,
                        draw.tile,
                        draw.wrap,
                        &draw.regions,
                        meshes,
                        held,
                        borders,
                    );
                    let part = crate::terrain::hash(part);
                    for &stack in source_stacks[i].iter().filter(|&&s| s < stacks) {
                        parts[stack].push(part);
                    }
                }
            }
            let paint = paint.drape_key(zoom, bearing);
            let states = style.states.zoom_key(zoom);
            (parts.into_iter())
                .map(|parts| crate::terrain::hash((revision, parts, zoom, paint, states)))
                .collect()
        };
        let globe_warps = &mut self.globe_warps;
        let mut globe = |tile: TileIndex| {
            let target = view.globe()?.target;
            let (warp, anchor) = globe_warps.warp(gpu, renderer, &surface, tile)?;
            Some(orb_render::TerrainGlobe {
                warp,
                origin: (anchor - target).as_vec3().to_array(),
                poles: orb_crs::sphere::poles(&surface, tile),
            })
        };
        let terrain = style.terrain.as_mut().expect("checked above");
        let (tiles, redraw) = terrain.plan(&surface, view, (layout, &moves), &content, &mut globe);
        renderer.set_terrain_tiles(gpu, &tiles);
        let slots: Vec<(u32, u8, &[u8])> = (redraw.iter())
            .map(|(tile, slot, stacks)| {
                let zoom = crate::terrain::drape_zoom(&surface, tile.index);
                (*slot, zoom, &stacks[..])
            })
            .collect();
        renderer.redraw_drape(&slots);
        Some(DrapePlan {
            surface,
            layout,
            redraw,
            reach_px,
        })
    }

    /// Places the background layers' quads on every covered tile, and on
    /// the drape slots being redrawn.
    pub(crate) fn place_background(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        view: &View,
        elevation: Option<&crate::terrain::TerrainElevation>,
        drape: Option<&DrapePlan>,
        report: &mut FrameReport,
    ) {
        let Some(style) = &mut self.style else {
            return;
        };
        // Backgrounds draw on every covered tile, as GL JS's do: never
        // beyond the world's edge.
        let Some(tile) = style.background else {
            return;
        };
        let zoom = self.camera.state().zoom(self.surface.projection());
        let tms = self.surface.tms();
        let cover = match view.low_plane() {
            Some(low) => low.cover_warped(tms, TILE_PX, 24, |tile| {
                self.globe_warps.footprint(tms, tile)
            }),
            None => view.cover_volume(
                tms,
                TILE_PX,
                24,
                0.0,
                elevation.map(|e| e as &dyn orb_camera::Elevation),
                0.0,
            ),
        };
        let period = self.surface.period();
        let at = Placement {
            extent: 8192,
            period,
            zoom,
            frame: None,
            series: None,
            globe: None,
            on_plane: true,
            tile_zoom: 0.0,
        };
        let globe_warps = &mut self.globe_warps;
        let mut rows: Vec<TileRow> = cover
            .iter()
            .filter_map(|covered| {
                let draw = Draw {
                    tile: covered.index,
                    wrap: covered.wrap,
                    regions: vec![covered.index],
                };
                let globe = match view.globe() {
                    Some(_) => Some(globe_warps.warp(gpu, renderer, tms, covered.index)?),
                    None => None,
                };
                let at = Placement {
                    tile_zoom: tms.style_zoom(covered.index),
                    globe,
                    ..at
                };
                // The cover partitions the world and each quad is exactly
                // its tile, so nothing needs clipping, and a seam pixel is
                // the rasterizer's to give to one tile's triangles. Clipped
                // to the tile, a seam through pixel centres (a globe turned
                // 45°) cracked where the rasterizer gave its pixels to the
                // tile whose far edge it is: the half-open rectangle
                // discarded them. The rectangle reaches everywhere but on
                // to the poles, which only the world's edge rows' caps do
                // (`project.wgsl`'s `reaches_pole`).
                let mut row = placements(tms, &draw, at).next()?;
                let clip = &mut row.clips[0];
                let pole = |y: i16, extreme: i16, short: i16| if y == extreme { y } else { short };
                clip.min = [i16::MIN, pole(clip.min[1], i16::MIN, i16::MIN + 1)];
                clip.max = [i16::MAX, pole(clip.max[1], i16::MAX, i16::MAX - 1)];
                Some(row)
            })
            .collect();
        report.tiles_drawn += rows.len() as u32;
        // Into each slot, the background's quad on its terrain tile.
        for (drape, (terrain, _, _)) in drape
            .iter()
            .flat_map(|d| d.redraw.iter().map(move |r| (d, r)))
        {
            let draw = Draw {
                tile: terrain.index,
                wrap: terrain.wrap,
                regions: vec![terrain.index],
            };
            let tile_zoom = tms.style_zoom(terrain.index);
            let stacks: Vec<u8> = (0..drape.layout.stack_count() as u8).collect();
            let at = (Placement { tile_zoom, ..at }, &stacks[..]);
            drape.rows(tms, &draw, at, &mut rows);
        }
        if renderer.place(tile, 8192, &rows).is_err() {
            self.warnings
                .push("out of tile placements; the background is not drawn".into());
        }
    }
}

/// The drape slots a frame redraws: terrain tiles with their tile slots,
/// and where slots are.
#[derive(Debug)]
pub(crate) struct DrapePlan {
    /// The surface's matrix set, which terrain tiles are of.
    pub(crate) surface: TileMatrixSet,
    pub(crate) layout: orb_render::DrapeLayout,
    pub(crate) redraw: Vec<crate::terrain::Redraw>,
    /// The farthest a draped fill or line is translated, logical pixels.
    pub(crate) reach_px: f64,
}

impl DrapePlan {
    /// The drape rows of `draw`, placed `at`, into every redrawn slot it
    /// overlaps: into the slots of `stacks`, those its layers draw into.
    fn rows(
        &self,
        tms: &TileMatrixSet,
        draw: &Draw,
        (at, stacks): (Placement, &[u8]),
        out: &mut Vec<TileRow>,
    ) {
        let drawn = crate::terrain::Drawn::of(tms, draw, at.frame, at.period);
        for (terrain, slot, redrawn) in &self.redraw {
            let (terrain, slot) = (*terrain, *slot);
            let stacks: Vec<u8> = (stacks.iter().copied())
                .filter(|s| redrawn.contains(s))
                .collect();
            if !stacks.is_empty() && drawn.overlaps(&self.surface, terrain, self.reach_px) {
                let zoom = f64::from(crate::terrain::drape_zoom(&self.surface, terrain.index));
                let map = placements(tms, draw, Placement { zoom, ..at });
                crate::terrain::drape_rows(
                    &self.surface,
                    drawn,
                    at.extent,
                    (terrain, slot),
                    (&self.layout, &stacks),
                    self.reach_px,
                    map,
                    out,
                );
            }
        }
    }
}

/// Per side of `tile` (top, right, bottom, left), how many zooms finer
/// than it the finest `drawn` tile across that side is, up to
/// [`orb_crs::MAX_EDGE_REFINE`].
fn edge_refinement(tile: TileIndex, drawn: impl Fn(TileIndex) -> bool) -> [u8; 4] {
    let TileIndex { z, col, row } = tile;
    [0, 1, 2, 3].map(|side| {
        (1..=orb_crs::MAX_EDGE_REFINE)
            .rev()
            .find(|&d| {
                let (c, r, n) = (col << d, row << d, 1 << d);
                // The tiles `d` zooms down across the side (none past the
                // matrix's first row or column).
                (0..n).any(|k| {
                    drawn(match side {
                        0 => TileIndex::new(z + d, c + k, r.wrapping_sub(1)),
                        1 => TileIndex::new(z + d, c + n, r + k),
                        2 => TileIndex::new(z + d, c + k, r + n),
                        _ => TileIndex::new(z + d, c.wrapping_sub(1), r + k),
                    })
                })
            })
            .unwrap_or(0)
    })
}

/// GL JS's aligned projection: the world offset that moves the camera
/// target to a whole pixel of the world at this zoom (half a pixel off for
/// odd viewport sizes, turned with the bearing), so image tiles at whole
/// pixels draw crisp.
pub(crate) fn pixel_alignment(
    view: &View,
    projection: &orb_crs::Projection,
    bearing: f64,
) -> [f32; 2] {
    let world_per_px = view.world_per_px();
    let bounds = projection.world_bounds();
    let camera = view.camera_world();
    let x = (camera.x - bounds.min.x) / world_per_px;
    let y = (bounds.max.y - camera.y) / world_per_px;
    let viewport = view.viewport();
    let shift = |size: f64| (size.round() as i64 % 2) as f64 / 2.0;
    let (x_shift, y_shift) = (shift(viewport.width), shift(viewport.height));
    let (sin, cos) = (-bearing.to_radians()).sin_cos();
    let dx = x - x.round() + cos * x_shift + sin * y_shift;
    let dy = y - y.round() + cos * y_shift + sin * x_shift;
    let wrap = |d: f64| if d > 0.5 { d - 1.0 } else { d };
    // GL JS's world pixels grow southward; camera-relative world northward.
    [
        (wrap(dx) * world_per_px) as f32,
        (-wrap(dy) * world_per_px) as f32,
    ]
}

/// Where a tile's placement rows put it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Placement<'a> {
    /// Data units per tile side.
    extent: u32,
    /// The surface's wrap period, 0 if it doesn't wrap.
    period: f64,
    /// The camera's style zoom.
    zoom: f64,
    /// A warped tile's frame on the surface, which its mesh spans `extent`
    /// of; tiles on the surface's plane span their own bounds.
    frame: Option<orb_crs::WorldRect>,
    /// A vector tile drawn through a warp's patches, relative to its
    /// anchor (the renderer has the patches).
    series: Option<&'a orb_crs::Series>,
    /// On a globe: the tile's patches on the sphere, and their anchor.
    globe: Option<(TileWarp, glam::DVec3)>,
    /// Whether the tile has a place on the plane (on a globe, a warped
    /// tile without planar patches draws on the sphere alone).
    on_plane: bool,
    /// The style zoom the tile laid out at.
    tile_zoom: f64,
}

/// Placement rows for drawing `draw.tile` in its world copy, clipped to
/// the regions it stands in for.
pub(crate) fn placements<'a>(
    tms: &TileMatrixSet,
    draw: &'a Draw,
    at: Placement<'_>,
) -> impl Iterator<Item = TileRow> + use<'a> {
    let Placement {
        extent,
        period,
        zoom,
        frame,
        series,
        globe,
        on_plane,
        tile_zoom,
    } = at;
    let bounds = frame.unwrap_or_else(|| tms.tile_bounds(draw.tile));
    let wrap = DVec2::new(f64::from(draw.wrap) * period, 0.0);
    let origin = match series {
        Some(series) => series.anchor + wrap,
        None => DVec2::new(bounds.min.x, bounds.max.y) + wrap,
    };
    let tile = draw.tile;
    // On a globe, regions on the world's edge reach on to the poles the
    // set stops short of (caps, `project.wgsl`). Sets are quadtrees.
    let (first, rows0) = (tms.min_zoom(), tms.matrix_size(tms.min_zoom()).1);
    let caps = match globe {
        Some(_) => {
            let bottom = TileIndex::new(first, 0, rows0 - 1);
            [
                orb_crs::sphere::poles(tms, TileIndex::new(first, 0, 0))[0],
                orb_crs::sphere::poles(tms, bottom)[1],
            ]
        }
        None => [false; 2],
    };
    // A tile of a partial matrix reaching past its set's extent draws
    // only what lies inside, in its data units.
    let (own, area) = (tms.tile_bounds(tile), tms.tile_area(tile));
    let to_units = |v: f64| v / own.size().x * f64::from(extent);
    let units = |v: f64| to_units(v).round() as i16;
    let inside = (area != own).then(|| {
        (
            [units(area.min.x - own.min.x), units(own.max.y - area.max.y)],
            [units(area.max.x - own.min.x), units(own.max.y - area.min.y)],
        )
    });
    // The extent's top and bottom edges in the tile's data units: regions
    // reaching them reach on to the caps.
    let world = tms.projection().world_bounds();
    let edges = [
        to_units(own.max.y - world.max.y),
        to_units(own.max.y - world.min.y),
    ];
    let rect = move |region: TileIndex| {
        let depth = region.z - tile.z;
        let size = extent >> depth;
        let x = (region.col - (tile.col << depth)) * size;
        let y = (region.row - (tile.row << depth)) * size;
        let (mut min, mut max) = ([x as i16, y as i16], [(x + size) as i16, (y + size) as i16]);
        if let Some((lo, hi)) = inside {
            min = [min[0].max(lo[0]), min[1].max(lo[1])];
            max = [max[0].min(hi[0]), max[1].min(hi[1])];
        }
        if caps[0] && f64::from(y) <= edges[0] + 0.5 {
            min[1] = i16::MIN;
        }
        if caps[1] && f64::from(y + size) >= edges[1] - 0.5 {
            max[1] = i16::MAX;
        }
        ClipRect::new(min, max)
    };
    // One placement has four clip rectangles. Split larger partitions into
    // disjoint rows instead of expanding to quadrants that overlap loaded
    // descendants (and render translucent geometry twice).
    let pixel_size = TILE_PX * 2.0f64.powf(zoom.floor() - tile_zoom);
    let copy = tms.projection().wrap_period().unwrap_or(0.0) / tms.tile_span(tile.z);
    let x = f64::from(tile.col) + f64::from(draw.wrap) * copy;
    let pattern_origin = [
        (x * pixel_size) as i64 as u32,
        (f64::from(tile.row) * pixel_size) as i64 as u32,
    ];
    let span = match (series, frame) {
        (Some(series), _) => series.scale,
        (None, Some(frame)) => frame.size().x,
        (None, None) => tms.tile_span(tile.z),
    };
    let scale = (span / f64::from(extent)) as f32;
    draw.regions.chunks(4).map(move |regions| {
        let mut clips = [ClipRect::EMPTY; 4];
        for (clip, &region) in clips.iter_mut().zip(regions) {
            *clip = rect(region);
        }
        TileRow {
            extent: extent as f32,
            wrap: draw.wrap,
            pattern_origin,
            origin: origin.to_array(),
            scale,
            zoom: tile_zoom as f32,
            clips,
            bound: ClipRect::EVERYWHERE,
            px_scale: 1.0,
            drape_zoom: 0.0,
            dem: DemRef::NONE,
            terrain_range: [0.0; 2],
            target: PlacementTarget::Map,
            warp: TileWarp::NONE,
            globe: globe.map(|(warp, origin)| orb_render::RowGlobe {
                warp,
                origin: origin.to_array(),
                on_plane,
            }),
            neighbours: 0,
        }
    })
}

/// The draws among `draws` placed alone over their tile's whole square:
/// each draws entire the whole polygons whose centroid it holds.
fn whole_draws(tms: &TileMatrixSet, draws: &[Draw]) -> HashSet<(TileIndex, i32)> {
    // Within rounding of its bounds: not a tile of a partial matrix that
    // reaches past its set's extent.
    let full = |tile| {
        let (area, own) = (tms.tile_area(tile), tms.tile_bounds(tile));
        let off = (area.min - own.min).abs().max((area.max - own.max).abs());
        off.max_element() <= own.size().x * 1e-9
    };
    (draws.iter())
        .filter(|d| d.regions == [d.tile] && full(d.tile))
        .map(|d| (d.tile, d.wrap))
        .collect()
}

/// [`TileRow::neighbours`] of `draw`: which tiles around its own, at its
/// zoom, are among `whole` ([`whole_draws`]), across the antimeridian
/// into the next world copy where the surface wraps. A building whole in
/// a tile is whole in the tile at its zoom holding its centroid too
/// (`orb_tile::Layer::is_whole`), but that tile may be drawn at another
/// zoom, or not at all: then the tile draws its part itself.
fn neighbours(
    tms: &TileMatrixSet,
    draw: &Draw,
    whole: &HashSet<(TileIndex, i32)>,
    wraps: bool,
) -> u16 {
    let TileIndex { z, col, row } = draw.tile;
    let (cols, rows) = tms.matrix_size(z);
    let (cols, rows) = (i64::from(cols), i64::from(rows));
    let mut bits = 0;
    for dy in -1..=1i64 {
        for dx in -1..=1i64 {
            let (mut c, r, mut wrap) = (i64::from(col) + dx, i64::from(row) + dy, draw.wrap);
            if (dx, dy) == (0, 0) || !(0..rows).contains(&r) {
                continue;
            }
            if !(0..cols).contains(&c) {
                if !wraps {
                    continue;
                }
                wrap += c.div_euclid(cols) as i32;
                c = c.rem_euclid(cols);
            }
            let tile = TileIndex::new(z, c as u32, r as u32);
            if whole.contains(&(tile, wrap)) {
                bits |= 1 << (3 * (dy + 1) + dx + 1);
            }
        }
    }
    bits
}

/// Web Mercator, which a globe's sphere wraps.
pub(crate) fn mercator_projection() -> orb_crs::Projection {
    *TileMatrixSet::by_id("WebMercatorQuad")
        .expect("built-in tile matrix set")
        .projection()
}

/// A globe view's frame terms: its matrix, the target, and the horizon
/// plane that depth and the far side's clipping come from.
pub(crate) fn frame_globe(globe: &orb_camera::GlobeView, view: &View) -> orb_render::FrameGlobe {
    let plane = globe.plane.is_some();
    let horizon = if plane {
        // A plane has no horizon: everything faces the eye.
        [0.0, 0.0, 0.0, 1.0]
    } else {
        let radius = orb_crs::sphere::RADIUS;
        let eye = globe.target + globe.eye;
        let distance = eye.length();
        let toward = eye / distance;
        // `facing` is 1 under the eye and 0 at the horizon circle.
        let horizon = radius * radius / distance;
        let scale = 1.0 / (radius - horizon).max(1e-9);
        let n = toward * scale;
        [
            n.x as f32,
            n.y as f32,
            n.z as f32,
            ((globe.target.dot(toward) - horizon) * scale) as f32,
        ]
    };
    orb_render::FrameGlobe {
        view_proj: globe.view_proj.as_mat4().to_cols_array_2d(),
        target: globe.target.to_array(),
        amount: globe.amount as f32,
        horizon,
        metres_per_unit: (globe.metres_per_px / view.world_per_px()) as f32,
        eye: globe.eye.as_vec3().to_array(),
        plane,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_placements_keep_more_than_four_regions_exact() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let draw = Draw {
            tile: TileIndex::new(0, 0, 0),
            wrap: 1,
            regions: (0..8).map(|col| TileIndex::new(3, col, 0)).collect(),
        };
        let at = Placement {
            extent: 4096,
            period: 100.0,
            zoom: 3.0,
            frame: None,
            series: None,
            globe: None,
            on_plane: true,
            tile_zoom: 0.0,
        };
        let rows: Vec<_> = placements(&tms, &draw, at).collect();
        assert_eq!(rows.len(), 2);
        for (col, clip) in rows.iter().flat_map(|row| row.clips).enumerate() {
            assert_eq!(
                clip,
                ClipRect::new([(col * 512) as i16, 0], [((col + 1) * 512) as i16, 512])
            );
        }
        assert_eq!(rows[0].origin, rows[1].origin);
        assert_eq!(rows[0].scale, rows[1].scale);
    }

    /// A tile leaves a whole building to the tile around it holding its
    /// centroid only where that tile is drawn alone over its square at the
    /// same zoom (#143): not where a deeper tile, an ancestor standing in
    /// with regions, or nothing is drawn there. Across the antimeridian the
    /// neighbour is the next world copy's.
    #[test]
    fn whole_buildings_are_left_only_to_neighbours_drawn_whole() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let draw = |z, col, row, wrap| Draw {
            tile: TileIndex::new(z, col, row),
            wrap,
            regions: vec![TileIndex::new(z, col, row)],
        };
        let draws = [
            // The tile, its neighbours east and south-west.
            draw(3, 4, 4, 0),
            draw(3, 5, 4, 0),
            draw(3, 3, 5, 0),
            // North of it, deeper tiles.
            draw(4, 8, 6, 0),
            draw(4, 9, 6, 0),
            // West, an ancestor's region.
            Draw {
                regions: vec![TileIndex::new(3, 3, 4)],
                ..draw(2, 1, 2, 0)
            },
            // At the world's west edge, and the copy west of it.
            draw(3, 0, 1, 0),
            draw(3, 7, 1, -1),
            draw(3, 7, 2, 0),
        ];
        let whole = whole_draws(&tms, &draws);
        let bits = |d: &Draw| neighbours(&tms, d, &whole, true);
        // East (dx 1, dy 0) is bit 5, south-west (dx -1, dy 1) bit 6.
        assert_eq!(bits(&draws[0]), 1 << 5 | 1 << 6);
        // West (bit 3) is the copy west of it; south-west (bit 6) isn't.
        assert_eq!(bits(&draws[6]), 1 << 3);
        assert_eq!(neighbours(&tms, &draws[6], &whole, false), 0);
    }

    #[test]
    fn an_extrusion_bound_keeps_its_constant_part() {
        let bound = |paint: &str| {
            let style = orb_style::v8::parse(&format!(
                r#"{{"version": 8,
                    "sources": {{"b": {{"type": "geojson", "data": {{"type": "FeatureCollection", "features": []}}}}}},
                    "layers": [{{"id": "e", "type": "fill-extrusion", "source": "b", "paint": {paint}}}]}}"#
            ))
            .unwrap();
            extrusion_bound(&style.layers[0], 10.0)
        };
        assert_eq!(bound(r#"{"fill-extrusion-height": 300}"#), None);
        // A base no expression bounds leaves the constant height standing.
        assert_eq!(
            bound(r#"{"fill-extrusion-height": 300, "fill-extrusion-base": ["get", "b"]}"#),
            Some((300.0, false))
        );
        assert_eq!(
            bound(r#"{"fill-extrusion-height": ["min", ["get", "h"], 800]}"#),
            Some((800.0, true))
        );
    }
}

/// For an extrusion layer whose height or base varies per feature, the
/// most either can give at `zoom`, a constant one's value included, and
/// whether both set a bound: where one doesn't (a bare `get`), the tiles
/// the cover holds tell the rest. `None` when every feature shares both.
fn extrusion_bound(layer: &orb_style::Layer, zoom: f64) -> Option<(f32, bool)> {
    let top = layer.get(orb_style::EXTRUSION_HEIGHT);
    let base = layer.get(orb_style::EXTRUSION_BASE);
    let per_feature = |kind| matches!(kind, orb_style::Kind::Source | orb_style::Kind::Composite);
    if !per_feature(top.kind()) && !per_feature(base.kind()) {
        return None;
    }
    let bounds = [top.max_value(zoom), base.max_value(zoom)];
    let tallest = bounds.into_iter().flatten().fold(0.0, f64::max) as f32;
    Some((tallest, bounds.iter().all(Option::is_some)))
}

/// A source's plan for a frame ([`Map::plan_source`]).
pub(crate) struct SourcePlan {
    pub(crate) update: orb_source::Update,
    /// Tiles of the cover only what reaches into the tiles beside them is
    /// needed of: in Tile mode, the drawn tile's neighbours, which lay out
    /// that alone (`TileRequest::neighbour`).
    pub(crate) neighbours: Vec<TileIndex>,
}

impl SourcePlan {
    fn of(update: orb_source::Update) -> Self {
        Self {
            update,
            neighbours: Vec::new(),
        }
    }
}

/// A source's layers whose height in the cover no expression bounds: the
/// tiles they stand in tell.
#[derive(Default)]
struct Unbounded {
    /// Symbol layers' keys, lifted by `symbol-height-offset`.
    labels: Vec<u32>,
    /// Extrusion nodes' keys.
    extrusions: Vec<u32>,
}

impl Unbounded {
    fn is_empty(&self) -> bool {
        self.labels.is_empty() && self.extrusions.is_empty()
    }

    /// The highest these stand in `tile`, as it laid out, style metres.
    fn tallest(&self, tile: &ResidentTile) -> f32 {
        let labels = (tile.symbols.iter())
            .filter(|s| self.labels.contains(&s.layer))
            .map(|s| s.max_height_offset as f32);
        // An extrusion's paint slots: colour, height, base.
        let extrusions = (tile.meshes.iter())
            .filter(|m| self.extrusions.contains(&m.node))
            .flat_map(|m| m.maxima.get(1..3).into_iter().flatten().copied());
        labels.chain(extrusions).fold(0.0, f32::max)
    }
}
