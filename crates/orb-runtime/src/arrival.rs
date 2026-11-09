//! What arrives between frames: the replies the map's jobs,
//! fetches and lanes send, applied at the start of a frame within its
//! budget, and arrived tiles made resident.

use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_render::{ImageUpload, OutOfSpace, Renderer, TileKey};
use orb_style::{LayerType, SourceKind};

use crate::MapEvent;
use crate::image::ResidentImage;
use crate::jobs::TileResult;
use crate::map::*;
use crate::meshes::*;
use crate::resident::*;
use orb_source::endpoint::TileEndpoint;

impl Map {
    /// Applies arrived results in order. New tiles apply within
    /// [`TILE_APPLY_MS`], at least one a frame: tiles that arrive together
    /// spread their uploads over frames rather than stalling one. Edits'
    /// results apply whole.
    pub(crate) fn apply_messages(&mut self, gpu: &mut dyn Gpu, renderer: &mut Renderer) {
        let started = self.services.clock.now();
        self.tiles_deferred = false;
        while let Ok(message) = self.receiver.try_recv() {
            let new_tile = matches!(message, Message::Tile { edit: None, .. });
            match message {
                // Replies to the labels a GPU reset replaced are dropped.
                Message::Placed { epoch, committed } => {
                    if epoch == self.label_epoch {
                        match committed {
                            Ok(committed) => {
                                let started = self.services.clock.now();
                                self.labels.committed(gpu, renderer, committed);
                                self.placement_ms += self.services.clock.now() - started;
                            }
                            // The labels ask again with the next frame.
                            Err(message) => {
                                self.labels.place_failed();
                                self.warnings.push(format!("placement: {message}"));
                            }
                        }
                    }
                }
                Message::Raster {
                    epoch,
                    rows,
                    result,
                } => {
                    if epoch == self.label_epoch {
                        self.labels.rastered(
                            gpu,
                            renderer.glyphs(),
                            &rows,
                            result,
                            &mut self.warnings,
                        );
                    }
                }
                Message::Style {
                    request,
                    url,
                    result,
                } => self.apply_style(request, &url, result),
                Message::SpriteIcons {
                    generation,
                    epoch,
                    index,
                    result,
                } => {
                    let mut done = false;
                    if let Some(style) = self
                        .style
                        .as_mut()
                        .filter(|s| generation == self.generation && epoch == s.sprite_epoch)
                    {
                        style.sprites_loading -= 1;
                        match result {
                            Err(e) => {
                                let message = format!("sprite: {e}");
                                self.events.push(MapEvent::Error { message });
                            }
                            Ok(icons) => {
                                for icon in icons {
                                    style.sprite_names.insert(icon.name.clone());
                                    style.icons.add(
                                        &icon.name,
                                        [icon.width, icon.height],
                                        &icon.metadata,
                                    );
                                    style.sprite_icon_names[index].push(icon.name);
                                }
                            }
                        }
                        done = style.sprites_loading == 0;
                    }
                    if done {
                        self.sprites_done(gpu);
                    }
                }
                Message::Sprite {
                    generation,
                    epoch,
                    index,
                    result,
                } => self.apply_sprite_pixels(gpu, (generation, epoch, index), result),
                Message::TileJson {
                    generation,
                    source,
                    result,
                    head,
                } => {
                    let Some(style) = self
                        .style
                        .as_mut()
                        .filter(|_| generation == self.generation)
                    else {
                        continue;
                    };
                    if let Some((url, head)) = head {
                        self.services.jobs.add_archive_head(&url, &Arc::new(head));
                    }
                    let plan = style.worker.plan();
                    let Some(index) = plan.source(source) else {
                        continue;
                    };
                    if !matches!(style.sources[index], SourceState::Resolving) {
                        continue;
                    }
                    let def = plan.source_def(index);
                    let tms = self.surface.source_tms(&def.id, false);
                    let endpoint = match (result, self.surface.plan(&tms, plan.image_kind(index))) {
                        (Err(e), _) => Err(format!("TileJSON: {e}")),
                        (Ok(_), Err(e)) => Err(e),
                        (Ok(text), Ok(warp)) => TileEndpoint::resolve(&def.kind, Some(&text), tms)
                            .map(|endpoint| (endpoint, warp)),
                    };
                    let image = plan.image_kind(index).is_some();
                    style.sources[index] = ready_source(endpoint, &def.id, image, &mut self.events);
                    self.events.sources.insert(index);
                    // Color-relief ramps wait for their source's encoding.
                    style.paint.invalidate();
                    for node in plan.nodes.iter().filter(|n| n.source == index) {
                        style.nodes.entry(node.key).or_default().ramp = None;
                    }
                }
                Message::Tile {
                    generation,
                    source,
                    tile,
                    edit,
                    terrain,
                    result,
                } => {
                    let result = *result;
                    let host = result.as_ref().ok().map(HostData::of);
                    if let Some(edit) = edit {
                        self.finish_job(edit);
                    }
                    let mut kept = false;
                    if generation == self.tile_generation {
                        // Loads given up, missing or failed brought nothing.
                        self.tiles_arrived += u32::from(
                            edit.is_none()
                                && result.as_ref().is_ok_and(|r| !r.cancelled && !r.missing),
                        );
                        match edit {
                            None if terrain => {
                                self.apply_terrain_tile(gpu, renderer, source, tile, result)
                            }
                            None => {
                                if let Some(build) =
                                    self.apply_tile(gpu, renderer, source, tile, result, true)
                                {
                                    kept = true;
                                    let tried = Some(self.room_state(renderer));
                                    self.waiting.push(WaitingTile {
                                        generation,
                                        source,
                                        tile,
                                        build,
                                        tried,
                                    });
                                }
                            }
                            Some(_) => self.apply_edit(gpu, renderer, source, tile, result),
                        }
                    }
                    // Uploaded or dropped, the result's host data is done.
                    if let Some(host) = host.filter(|_| !kept) {
                        host.release(gpu);
                    }
                }
            }
            if new_tile && self.services.clock.now() - started > TILE_APPLY_MS {
                self.tiles_deferred = true;
                break;
            }
        }
        self.retry_waiting(gpu, renderer, started);
    }

    /// A sprite's pixels, into the atlas, for the icons its JSON gave. A
    /// sprite whose image failed takes those icons back, and symbol layers
    /// lay out again without them.
    fn apply_sprite_pixels(
        &mut self,
        gpu: &mut dyn Gpu,
        (generation, epoch, index): (u64, u64, usize),
        result: Result<Vec<crate::sprite::SpriteImage>, String>,
    ) {
        let Some(style) = self
            .style
            .as_mut()
            .filter(|s| generation == self.generation && epoch == s.sprite_epoch)
        else {
            return;
        };
        style.sprite_pixels_loading -= 1;
        let mut relayout = false;
        match result {
            Err(e) => {
                let message = format!("sprite: {e}");
                self.events.push(MapEvent::Error { message });
                let mut withdrawn = false;
                for name in std::mem::take(&mut style.sprite_icon_names[index]) {
                    if !style.added_images.contains_key(&name) {
                        style.sprite_names.remove(&name);
                        withdrawn |= style.icons.remove(&name);
                    }
                }
                // While other sprites' JSON loads, the set goes with theirs.
                if withdrawn && style.sprites_loading == 0 {
                    style.icons.epoch += 1;
                    let icons = Arc::new(style.icons.clone());
                    self.services
                        .jobs
                        .set_icons(style.id, &icons, &style.worker);
                    relayout = true;
                }
            }
            Ok(images) => {
                let namespace = style.image_namespace.expect("sprite namespace");
                let mut atlas = self.services.images.0.lock().expect("images poisoned");
                let atlas = atlas.atlas.as_mut().expect("initialized");
                for image in images {
                    // Another sprite's image of the name gives way, keeping
                    // its atlas id.
                    let held = atlas.resolve(namespace, &image.name).ok();
                    if held.and_then(|id| atlas.size(id)).is_some() {
                        atlas.clear(gpu, namespace, &image.name);
                    }
                    // Patterns and icons pad differently.
                    let mut upload = |pattern: bool| {
                        let id = if pattern {
                            atlas.resolve(namespace, &image.name)?
                        } else {
                            atlas.resolve_icon(namespace, &image.name)?
                        };
                        atlas.upload(
                            gpu,
                            id,
                            ImageUpload {
                                size: [image.width, image.height],
                                pixels: &image.pixels,
                                metadata: image.metadata.clone(),
                                pattern,
                            },
                        )
                    };
                    let uploaded = upload(true).and_then(|()| upload(false));
                    if let Err(e) = uploaded {
                        self.warnings.push(format!("sprite {}: {e}", image.name));
                    }
                }
            }
        }
        if style.sprite_pixels_loading == 0 {
            relayout |= std::mem::take(&mut style.relayout_after_pixels);
        }
        if relayout {
            self.relayout_symbols();
        }
    }

    /// Tries the tiles waiting for room again, in the frame's budget left,
    /// each only if room may have changed since it last failed: memory
    /// freed or grown, or other tiles' jobs done.
    pub(crate) fn retry_waiting(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        started: f64,
    ) {
        let mut state = self.room_state(renderer);
        for mut waiting in std::mem::take(&mut self.waiting) {
            let host = HostData::of(&waiting.build);
            if waiting.generation != self.tile_generation {
                host.release(gpu);
                continue;
            }
            if self.tiles_deferred || waiting.tried == Some(state) {
                self.waiting.push(waiting);
                continue;
            }
            match self.apply_tile(
                gpu,
                renderer,
                waiting.source,
                waiting.tile,
                Ok(waiting.build),
                false,
            ) {
                Some(build) => {
                    state = self.room_state(renderer);
                    waiting.build = build;
                    waiting.tried = Some(state);
                    self.waiting.push(waiting);
                }
                None => {
                    host.release(gpu);
                    state = self.room_state(renderer);
                }
            }
            self.tiles_deferred = self.services.clock.now() - started > TILE_APPLY_MS;
        }
    }

    /// What a waiting tile depends on: the scene's pools in use and their
    /// buffers, tile page layers in use, tile jobs in flight, and sprites
    /// whose pixels are on their way.
    pub(crate) fn room_state(&self, renderer: &Renderer) -> [u64; 7] {
        let stats = renderer.stats();
        let loading: u64 = self
            .style
            .iter()
            .flat_map(|s| &s.sources)
            .map(|s| match s {
                SourceState::Ready { loading, .. } => u64::from(*loading),
                _ => 0,
            })
            .sum();
        [
            u64::from(stats.vertex_words),
            u64::from(stats.index_words),
            u64::from(stats.prop_words),
            renderer.pools_revision(),
            renderer.used_layers(),
            loading,
            self.style
                .as_ref()
                .map_or(0, |s| u64::from(s.sprite_pixels_loading)),
        ]
    }

    pub(crate) fn finish_job(&mut self, edit: u64) {
        if let Some(count) = self.pending.get_mut(&edit) {
            *count -= 1;
            if *count == 0 {
                self.pending.remove(&edit);
            }
        }
    }

    pub(crate) fn apply_tile(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        source: u32,
        tile: TileIndex,
        result: Result<TileResult, String>,
        arrived: bool,
    ) -> Option<TileResult> {
        let Some(style) = &mut self.style else {
            return None;
        };
        let plan = style.worker.plan();
        let source = plan.source(source)?;
        let SourceState::Ready {
            pyramid,
            loading,
            endpoint,
            warped,
            ..
        } = &mut style.sources[source]
        else {
            return None;
        };
        let warped = *warped;
        if arrived {
            *loading = loading.saturating_sub(1);
            self.events.sources.insert(source);
        }
        let mut build = match result {
            Ok(build) => build,
            Err(e) => {
                let message = format!("tile {tile:?}: {e}");
                self.events.push(MapEvent::Error { message });
                pyramid.insert_failed(tile, self.services.clock.now());
                return None;
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
        if build.missing {
            pyramid.insert_missing(tile);
            return None;
        }
        // Given up before it loaded: a later plan that wants it asks again.
        if build.cancelled {
            pyramid.forget_loading(tile);
            return None;
        }
        // Neighbours' prepares refer to an image tile's layer, so only a
        // tile the pyramid still wants is uploaded.
        if !pyramid.is_loading(tile) {
            return None;
        }
        // Nodes the tile was laid out for before an edit changed them are
        // laid out again from the job's cached tile.
        let mut stale = Vec::new();
        for node in plan.nodes.iter().filter(|n| n.source == source) {
            let state = style.nodes.entry(node.key).or_default();
            if build.image.is_none() && build.revision < state.layout_epoch.max(state.rows_epoch) {
                stale.push(node.key);
            }
        }
        drop_stale(&mut build, plan, &stale);
        let footprint = warped
            .then(|| self.globe_warps.surface_footprint(&endpoint.tms, tile))
            .flatten();
        let tile_zoom = self.surface.tile_zoom(&endpoint.tms, tile, footprint);
        let globe = self.surface.globe().is_some();
        let key = TileKey {
            source: plan.sources[source].key,
            tile,
        };
        let mut image = None;
        let mut fade_ms = 0.0;
        let now = self.frame_time;
        let mut grid = build.warp_grid.take();
        if let Some(tile_image) = &build.image {
            let Ok(warped) = image_mesh(
                &mut self.globe_warps,
                globe,
                warped,
                endpoint,
                tile,
                (&mut grid, [0; 4]),
            ) else {
                // Nothing of the tile reaches the surface.
                pyramid.insert_empty(tile);
                return None;
            };
            // Raster tiles fade in over their layers' longest duration;
            // still output, and a map not yet idle once (as GL JS's first
            // view, which has no tiles to fade from), shows them at once.
            let fades = self.mode == RenderMode::Continuous
                && self.events.idled
                && !matches!(plan.source_def(source).kind, SourceKind::Image(_));
            fade_ms = plan
                .nodes
                .iter()
                .filter(|n| fades && n.source == source)
                .map(|n| &plan.style.layers[n.layer])
                .filter(|r| r.ty == LayerType::Raster)
                .map(|r| f64::from(r.get(orb_style::RASTER_FADE_DURATION).evaluate(tile_zoom)))
                .fold(0.0, f64::max);
            image = Some((tile_image, warped));
        }
        // A tile on another plane draws through its series' patches; a
        // warped image only on a globe surface.
        let series = (warped && (image.is_none() || globe))
            .then(|| self.globe_warps.surface_series(&endpoint.tms, tile))
            .flatten();
        let patches = series.as_deref().map(series_patches);
        let patches = patches.as_ref().map(|(cells, p)| (*cells, &p[..]));
        let image_build = image.as_ref().map(|(tile_image, warped)| {
            let kind = plan
                .image_kind(source)
                .expect("image results are for image sources");
            orb_render::ImageBuild {
                key,
                kind: match kind {
                    crate::style::ImageKind::Raster => orb_render::PageKind::Raster,
                    crate::style::ImageKind::Dem => orb_render::PageKind::Dem,
                },
                size: [tile_image.width, tile_image.height],
                pixels: tile_image
                    .pixels
                    .upload(kind == crate::style::ImageKind::Raster),
                mesh: warped.as_ref().map(|w| Arc::clone(&w.mesh)),
                patches,
                fade_from: (fade_ms > 0.0).then_some((now - self.born.unwrap_or(now)) as f32),
                unpack: endpoint.dem_unpack.map(|u| u.map(|v| v as f32)),
                area: inside_texels(&endpoint.tms, tile, [tile_image.width, tile_image.height]),
            }
        });
        // Labels take their images' atlas rectangles as they upload: a tile
        // whose labels show images waits for the sprites' pixels (layout
        // needed only their JSON).
        if style.awaits_sprite_pixels(&build) {
            return Some(build);
        }
        let sizes = mesh_sizes(plan, &build);
        // Room: what's free first; then unused tiles go, this source's
        // before others'; under the budget the pools grow before they do.
        // While other tiles load (they free the stand-ins they replace),
        // this one waits. Otherwise, for a tile of the cover, the tiles it
        // replaces go, then tiles that only stand in (a fresh view of this
        // cover would hold none of them), its own ancestors first. A tile
        // outside the cover takes nothing from it.
        // A tile the plan no longer wants (neither in the cover nor
        // standing in for a tile of it that failed) is forgotten; one it
        // wants gives way, to be tried again later, and its ancestor is
        // asked to stand in.
        let others_loading = style
            .sources
            .iter()
            .any(|s| matches!(s, SourceState::Ready { loading, .. } if *loading > 0));
        let mut images = self.services.images.0.lock().expect("images poisoned");
        let atlas = images.atlas.as_mut().expect("initialized");
        let mut grow = under_budget(renderer, atlas.bytes(), self.budget);
        let uploading = self.services.clock.now();
        let uploaded = loop {
            atlas.release_ramps(gpu, self.retired_ramps.drain(..));
            use orb_render::UploadError::{Invalid, NoRoom};
            let attempt = match &image_build {
                Some(image_build) => renderer
                    .upload_image(gpu, image_build, grow)
                    .map(|handle| ResidentTile::new(handle, tile_zoom)),
                // Room for the meshes first, then their rows and labels.
                None if !(renderer.fits(&sizes) || grow && renderer.grow_for(gpu, &sizes)) => {
                    Err(NoRoom)
                }
                None => upload(
                    gpu,
                    renderer,
                    plan,
                    (key, &build, patches),
                    atlas,
                    style.image_namespace.expect("namespace"),
                    &mut style.states,
                    &mut self.labels,
                    tile_zoom,
                    &mut self.warnings,
                )
                .map_err(|OutOfSpace| NoRoom),
            };
            let victim = match attempt {
                Ok(resident) => break Ok(resident),
                Err(Invalid(e)) => break Err(Some(e)),
                // Labels' sizes aren't known ahead: both pools grow.
                Err(NoRoom) if grow && image_build.is_none() && renderer.grow(gpu) => continue,
                Err(NoRoom) => evict_unused(&mut style.sources, source),
            };
            let victim = match victim {
                Some(victim) => Some(victim),
                None if !grow => {
                    grow = true;
                    continue;
                }
                None if others_loading => None,
                None => match &mut style.sources[source] {
                    SourceState::Ready { pyramid, .. } if pyramid.covers(tile) => pyramid
                        .evict_within(tile)
                        .or_else(|| evict_stand_in(&mut style.sources, source, tile)),
                    _ => None,
                },
            };
            match victim {
                Some(victim) => {
                    self.retired_ramps
                        .extend(remove_resident(renderer, victim, &mut style.states))
                }
                None => break Err(None),
            }
        };
        drop(images);
        self.upload_ms += self.services.clock.now() - uploading;
        let SourceState::Ready { pyramid, .. } = &mut style.sources[source] else {
            unreachable!("checked above");
        };
        let wanted = pyramid.wants(tile);
        let mut resident = match uploaded {
            Ok(resident) => resident,
            Err(Some(e)) => {
                self.warnings.push(format!("tile {tile:?}: {e}"));
                pyramid.insert_empty(tile);
                return None;
            }
            Err(None) if others_loading => return Some(build),
            Err(None) if !wanted => {
                pyramid.forget_loading(tile);
                return None;
            }
            // Even the tiles in view don't leave room: what's loaded stands
            // in, and the tile is tried again after a while (the view may
            // have changed), as after a failed fetch.
            Err(None) => {
                let message = format!(
                    "tile {tile:?} doesn't fit in scene memory with the tiles in view; \
                     what's loaded stands in until it's tried again"
                );
                self.events.push(MapEvent::Error { message });
                pyramid.insert_failed(tile, self.services.clock.now());
                return None;
            }
        };
        if fade_ms > 0.0 {
            self.image_fades_until = self.image_fades_until.max(now + fade_ms);
        }
        resident.image = image.map(|(_, warped)| ResidentImage {
            warped,
            grid,
            globe: Some(globe),
        });
        resident.series = series;
        self.counts.meshes_uploaded += build.meshes.len() as u32;
        record_maxima(&mut style.nodes, &build.meshes, &mut style.paint);
        resident.cost = tile_cost(renderer, &resident);
        if let Some(unwanted) = pyramid.insert(tile, resident) {
            self.retired_ramps
                .extend(remove_resident(renderer, unwanted, &mut style.states));
        }
        if !stale.is_empty() {
            let version = self.version;
            self.send_job(source, tile, Some(stale), false, Some(version));
        }
        None
    }
}

/// The texels of a `size` image of `tile` inside its set's extent,
/// `[x0, y0, x1, y1)`, for a tile of a partial matrix that reaches past it;
/// `None` for a tile wholly inside.
fn inside_texels(
    tms: &orb_crs::TileMatrixSet,
    tile: TileIndex,
    [width, height]: [u32; 2],
) -> Option<[u32; 4]> {
    let (own, area) = (tms.tile_bounds(tile), tms.tile_area(tile));
    if area == own {
        return None;
    }
    let size = own.size();
    let at = |v: f64, side: u32| (v * f64::from(side)).round().clamp(0.0, f64::from(side)) as u32;
    let x = |w: f64| at((w - own.min.x) / size.x, width);
    let y = |w: f64| at((own.max.y - w) / size.y, height);
    let texels = [x(area.min.x), y(area.max.y), x(area.max.x), y(area.min.y)];
    (texels[0] < texels[2] && texels[1] < texels[3]).then_some(texels)
}
