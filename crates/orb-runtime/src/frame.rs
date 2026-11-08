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
        shadow: glam::DVec2,
    ) -> Vec<Option<orb_source::Update>> {
        let sources = self.style.as_ref().map_or(0, |s| s.sources.len());
        let at = |index| self.plan_source(renderer, view, index, elevation, shadow);
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
        shadow: glam::DVec2,
    ) -> Option<orb_source::Update> {
        let style = self.style.as_ref()?;
        let plan = style.worker.plan();
        // A source only the terrain reads.
        if !plan.nodes.iter().any(|n| n.source == index) {
            return None;
        }
        // Symbol layout needs the sprites' metadata, so a source with symbol
        // layers waits for them (GL JS's worker awaits `getImages` alike).
        if style.sprites_loading > 0
            && plan
                .nodes
                .iter()
                .any(|n| n.source == index && n.family == Family::SymbolIcon)
        {
            return None;
        }
        let zoom = self.camera.state().zoom(self.surface.projection());
        // The tallest extrusion, as this frame's paint has it.
        let height = (plan.nodes.iter().enumerate())
            .filter(|(_, n)| n.source == index)
            .map(|(i, _)| renderer.cull_elevation(i as u32))
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
        // A hillshade the terrain shades reads the slopes over each terrain
        // tile's quarters (`terrain::slopes`): its source covers those,
        // whatever detail the terrain's cover keeps, as its other layers,
        // all draped, draw into the terrain tiles' slots from any cover.
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
            return Some(pyramid.plan(&cover));
        }
        let tms = &endpoint.tms;
        // Where the source's visible layers are shown, their tiles hold
        // their data: the cover keeps to the least zoom all of them have
        // data at, wherever that detail still shows (orb_camera's
        // `DATA_TILE_PX`), and picks finer by distance above it.
        let data_zoom = plan
            .nodes
            .iter()
            .filter(|n| n.source == index)
            .map(|n| &plan.style.layers[n.layer])
            .filter(|l| l.is_visible_at(zoom))
            .map(|l| l.minzoom.floor().max(0.0) as u8)
            .max()
            .unwrap_or(0)
            .clamp(pyramid.min_zoom(), pyramid.max_zoom());
        let warps = &self.globe_warps;
        let cover = match (view.low_plane(), *warped) {
            // On a low-zoom plane every set's tiles land through its warp.
            (Some(low), _) => {
                low.cover_warped(tms, endpoint.cover_px, pyramid.max_zoom(), |tile| {
                    warps.footprint(tms, tile)
                })
            }
            (None, false) => view.cover_volume(
                tms,
                endpoint.cover_px,
                data_zoom..=pyramid.max_zoom(),
                orb_camera::Standing {
                    height: f64::from(height) * self.extrusion_scale(),
                    shadow,
                },
                elevation.map(|e| e as &dyn orb_camera::Elevation),
                f64::from(reach),
            ),
            (None, true) => view.cover_warped(tms, endpoint.cover_px, pyramid.max_zoom(), |tile| {
                warps.surface_footprint(tms, tile)
            }),
        };
        Some(pyramid.plan(&cover))
    }

    /// Applies source `index`'s plan: its pyramid records it, its requests
    /// load (those with nothing to draw are empty at once), and its draws
    /// wait for [`place_source`](Self::place_source).
    pub(crate) fn apply_source(
        &mut self,
        index: usize,
        update: Option<orb_source::Update>,
        report: &mut FrameReport,
    ) {
        let style = self.style.as_mut().expect("called with a style");
        style.draws.resize_with(style.sources.len(), || None);
        style.draws[index] = None;
        let Some(update) = update else {
            return;
        };
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
        style.draws[index] = Some(update.draw);
        for tile in requests {
            self.send_job(index, tile, None, false, None);
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
                    tile,
                    &rows[&tile],
                    tms,
                    resident.series.as_deref(),
                ));
            }
        }
        // Each drawn image tile's mesh is the one for this frame's surface
        // and, for a warped tile, the finest drawn tile beside each side,
        // whose vertices it takes so the two meet without cracks (#32).
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
            if image.globe == globe && image.warped.as_ref().map_or([0; 4], |w| w.edges) == edges {
                continue;
            }
            let Ok(mesh) = image_mesh(
                &mut self.globe_warps,
                globe,
                *warped,
                &endpoint,
                tile,
                edges,
            ) else {
                continue;
            };
            image.globe = globe;
            image.warped = mesh;
            // A warped image draws through planar patches only on a globe
            // surface; elsewhere through its warp's mesh.
            resident.series = (*warped && globe)
                .then(|| self.globe_warps.surface_series(tms, tile))
                .flatten();
            let patches = resident.series.as_deref().map(series_patches);
            let patches = patches.as_ref().map(|(cells, p)| (*cells, &p[..]));
            let mesh = image.warped.as_ref().map(|w| Arc::clone(&w.mesh));
            if renderer
                .set_image_mesh(gpu, resident.gpu, mesh, patches)
                .is_err()
            {
                self.warnings
                    .push("out of scene memory; an image layer misses tiles".into());
            }
        }
        let period = self.surface.period();
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
                    placements(tms, draw, at)
                        .map(|mut row| {
                            if let Some((dem, range)) = ground {
                                row.dem = dem;
                                row.terrain_range = range;
                            }
                            row
                        })
                        .collect()
                };
                let mut rows: Vec<TileRow> = draws.iter().flat_map(|d| map_rows(d)).collect();
                if let Some(drape) = draped {
                    draws.iter().for_each(|d| drape.rows(tms, d, at, &mut rows));
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
    /// hillshades turn with the map too: the terrain pass shades them
    /// between drape stacks, from the frame's bearing.
    pub(crate) fn plan_drape(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        view: &View,
    ) -> Option<DrapePlan> {
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
        let at = (layout.generation, bearing);
        crate::paint::drape(gpu, renderer, (style, plan), &zooms, at, atlas);
        let images_revision = atlas.revision();
        drop(images);
        let draped: Vec<usize> = (0..style.sources.len())
            .filter(|&i| {
                (plan.nodes.iter().enumerate()).any(|(index, n)| {
                    n.source == i
                        && n.family.is_draped()
                        && !renderer.shades_on_terrain(index as u32)
                })
            })
            .collect();
        let period = self.surface.period();
        let sources = &style.sources;
        let draws = &style.draws;
        let paint = &style.paint;
        let revision = (self.version, style.states.writes(), images_revision);
        // The farthest a draped fill or line is translated at a drape
        // zoom: a slot takes rows from the drawn tiles that far around it,
        // whose content moves into it.
        let reach_px = (plan.nodes.iter())
            .filter(|n| n.family.is_draped())
            .filter_map(|n| {
                let layer = &plan.style.layers[n.layer];
                match layer.ty {
                    orb_style::LayerType::Fill => Some(layer.get(orb_style::FILL_TRANSLATE)),
                    orb_style::LayerType::Line => Some(layer.get(orb_style::LINE_TRANSLATE)),
                    _ => None,
                }
            })
            .flat_map(|translate| {
                zooms.iter().map(move |&z| {
                    let [x, y] = translate.evaluate(f64::from(z));
                    f64::from(x.hypot(y))
                })
            })
            .fold(0.0, f64::max);
        // A DEM tile's border comes from its drawn neighbours
        // (`Renderer::pad_dems`).
        let drawn_tiles: HashMap<usize, HashSet<TileIndex>> = (draped.iter())
            .filter_map(|&i| Some((i, draws[i].as_ref()?.iter().map(|d| d.tile).collect())))
            .collect();
        let content = |t: orb_crs::CoveredTile| {
            let zoom = crate::terrain::drape_zoom(&surface, t.index);
            let mut parts = Vec::new();
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
                            (drawn.contains(&n) && pyramid.get(n).is_some()).then_some(0)
                        })
                    });
                    let regions = draw.regions.clone();
                    parts.push((i, draw.tile, draw.wrap, regions, meshes, held, borders));
                }
            }
            let paint = paint.drape_key(zoom, bearing);
            let states = style.states.zoom_key(zoom);
            crate::terrain::hash((revision, parts, zoom, paint, states))
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
        // A shaded hillshade draws where its layer does at the tile's
        // drape zoom, as draped layers do.
        let covered = &terrain.tiles;
        let drawn: HashMap<usize, HashSet<TileIndex>> = (plan.nodes.iter().enumerate())
            .filter(|&(index, _)| renderer.shades_on_terrain(index as u32))
            .map(|(_, n)| {
                let tiles = draws[n.source].iter().flatten().map(|d| d.tile).collect();
                (n.source, tiles)
            })
            .collect();
        let slopes = |node: u32, i: usize| {
            let node = &plan.nodes[node as usize];
            let tile = covered[i].index;
            let zoom = crate::terrain::drape_zoom(&surface, tile);
            match (&sources[node.source], drawn.get(&node.source)) {
                (
                    SourceState::Ready {
                        endpoint, pyramid, ..
                    },
                    Some(drawn),
                ) if plan.style.layers[node.layer].is_visible_at(f64::from(zoom)) => tile
                    .children()
                    .map(|q| crate::terrain::slopes((endpoint, pyramid), drawn, |r| r.gpu, q)),
                _ => [None; 4],
            }
        };
        renderer.set_terrain_tiles(gpu, &tiles, slopes);
        let slots: Vec<(u32, u8)> = redraw
            .iter()
            .map(|&(tile, slot)| (slot, crate::terrain::drape_zoom(&surface, tile.index)))
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
                0..=24,
                orb_camera::Standing::default(),
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
                placements(tms, &draw, at).next()
            })
            .collect();
        report.tiles_drawn += rows.len() as u32;
        // Into each slot, the background's quad on its terrain tile.
        for (drape, (terrain, _)) in drape
            .iter()
            .flat_map(|d| d.redraw.iter().map(move |r| (d, r)))
        {
            let draw = Draw {
                tile: terrain.index,
                wrap: terrain.wrap,
                regions: vec![terrain.index],
            };
            let tile_zoom = tms.style_zoom(terrain.index);
            drape.rows(tms, &draw, Placement { tile_zoom, ..at }, &mut rows);
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
    pub(crate) redraw: Vec<(orb_crs::CoveredTile, u32)>,
    /// The farthest a draped fill or line is translated, logical pixels.
    pub(crate) reach_px: f64,
}

impl DrapePlan {
    /// The drape rows of `draw`, placed `at`, into every redrawn slot it
    /// overlaps.
    fn rows(&self, tms: &TileMatrixSet, draw: &Draw, at: Placement, out: &mut Vec<TileRow>) {
        let drawn = crate::terrain::Drawn::of(tms, draw, at.frame, at.period);
        for &(terrain, slot) in &self.redraw {
            if drawn.overlaps(&self.surface, terrain, self.reach_px) {
                let zoom = f64::from(crate::terrain::drape_zoom(&self.surface, terrain.index));
                let map = placements(tms, draw, Placement { zoom, ..at });
                crate::terrain::drape_rows(
                    &self.surface,
                    drawn,
                    at.extent,
                    (terrain, slot),
                    &self.layout,
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
    let rect = move |region: TileIndex| {
        let depth = region.z - tile.z;
        let size = extent >> depth;
        let x = (region.col - (tile.col << depth)) * size;
        let y = (region.row - (tile.row << depth)) * size;
        let (mut min, mut max) = ([x as i16, y as i16], [(x + size) as i16, (y + size) as i16]);
        if caps[0] && region.row == 0 {
            min[1] = i16::MIN;
        }
        if caps[1] && region.row + 1 == rows0 << (region.z - first) {
            max[1] = i16::MAX;
        }
        ClipRect::new(min, max)
    };
    // One placement has four clip rectangles. Split larger partitions into
    // disjoint rows instead of expanding to quadrants that overlap loaded
    // descendants (and render translucent geometry twice).
    let pixel_size = TILE_PX * 2.0f64.powf(zoom.floor() - tile_zoom);
    let x = f64::from(tile.col) + f64::from(draw.wrap) * f64::from(tms.matrix_size(tile.z).0);
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
        }
    })
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
}
