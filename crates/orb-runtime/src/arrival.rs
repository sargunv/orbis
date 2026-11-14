//! What arrives between frames: the replies the map's jobs,
//! fetches and lanes send, applied at the start of a frame within its
//! budget, and arrived tiles made resident.

use std::sync::Arc;

use orb_crs::TileIndex;
use orb_hal::Gpu;
use orb_render::{ImageUpload, OutOfSpace, Renderer, TileKey};
use orb_style::{LayerType, SourceKind};

use crate::image::ResidentImage;
use crate::jobs::{TileOutcome, TileResult};
use crate::map::*;
use crate::meshes::*;
use crate::resident::*;
use crate::schedule::upload_bytes;
use crate::{MapEvent, Progress};
use orb_source::Awaited;
use orb_source::endpoint::TileEndpoint;

impl Map {
    /// Applies what arrived until `deadline` on the services' clock and
    /// within the frame's bytes ([`Budget`], [`crate::schedule`]):
    /// every message in the order it came but tiles' results, which queue;
    /// then the tile going up in pieces, edits' results in their order,
    /// loads nearest the view's focus first, and the tiles waiting for
    /// room. Tiles that arrive together spread their uploads over frames
    /// rather than stalling one, and a tile larger than what a frame has
    /// left goes up over several ([`Uploading`]).
    ///
    /// [`Budget`]: crate::Budget
    pub(crate) fn apply_messages(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        deadline: f64,
    ) {
        let clock = Arc::clone(&self.services.clock);
        while let Ok(message) = self.receiver.try_recv() {
            match message {
                Message::Tile { edit: Some(_), .. } => self.inbox.edits.push_back(message),
                Message::Tile { .. } => self.inbox.loads.push(message),
                message => self.apply_message(gpu, renderer, message),
            }
        }
        self.continue_upload(gpu, renderer);
        self.apply_heights(gpu, renderer, deadline);
        while let Some(message) = self.inbox.edits.front() {
            let bytes = self.upload_cost(message);
            if !self.beat.affords(clock.now() < deadline, bytes) {
                break;
            }
            let message = self.inbox.edits.pop_front().expect("in front");
            self.apply_tile_message(gpu, renderer, message);
            (self.beat).spend(bytes);
            self.beat.count();
        }
        // Farthest first, to take the nearest from the end.
        let mut loads = std::mem::take(&mut self.inbox.loads);
        loads.sort_by_cached_key(|m| std::cmp::Reverse(self.load_rank(m)));
        while let Some(message) = loads.last() {
            let bytes = self.upload_cost(message);
            let in_time = clock.now() < deadline;
            if !self.beat.takes(in_time, bytes, self.splits(message)) {
                break;
            }
            let message = loads.pop().expect("last");
            self.apply_tile_message(gpu, renderer, message);
            self.beat.count();
        }
        self.inbox.loads = loads;
        let deferred = self.retry_waiting(gpu, renderer, deadline);
        self.tiles_deferred = deferred
            || self.host_heights.pending()
            || self.uploading.is_some()
            || !self.inbox.edits.is_empty()
            || !self.inbox.loads.is_empty();
    }

    /// Writes what the frame's bytes allow of the tile going up in pieces,
    /// and makes the tile of them once they're all in: until then it draws
    /// nothing, and what stands in for it stays. A tile the view no longer
    /// waits for goes.
    fn continue_upload(&mut self, gpu: &mut dyn Gpu, renderer: &mut Renderer) {
        let Some(mut up) = self.uploading.take() else {
            return;
        };
        let wanted = up.generation == self.tile_generation
            && self.awaited(up.source, up.tile) == Awaited::Wanted;
        if !wanted {
            renderer.unreserve(up.reserved);
            HostData::of(&up.build).release(gpu);
            return;
        }
        let words = up.build.payload.words();
        let written = renderer.write_reserved(gpu, words, &mut up.reserved, self.beat.left());
        self.beat.spend(written);
        self.beat.count();
        // Its labels go up with it, whole: in a frame with room for them,
        // or one that has written nothing else.
        let labels = upload_bytes(&up.build).saturating_sub(up.reserved.written());
        let room = labels <= self.beat.left() || self.beat.bytes() == 0;
        if up.reserved.left() > 0 || !room {
            self.uploading = Some(up);
            return;
        }
        // Making the tile spends its labels' bytes (`apply_tile_in`).
        let host = HostData::of(&up.build);
        let mut reserved = Some(up.reserved);
        let result = Ok(TileOutcome::Built(up.build));
        let at = (up.source, up.tile);
        let allowed = (false, self.beat.left());
        let applied = self.apply_tile_with(gpu, renderer, at, result, allowed, &mut reserved);
        let at = (up.generation, up.source, up.tile);
        if !self.settle_applied(renderer, at, applied) {
            host.release(gpu);
        }
    }

    /// Where `applied`, what became of tile `tile` of source `source` in
    /// tile generation `generation`, goes: waiting for room (`tried` now),
    /// parked or going up in pieces. Whether its result is kept (its host
    /// data held).
    fn settle_applied(
        &mut self,
        renderer: &Renderer,
        (generation, source, tile): (u64, u32, TileIndex),
        applied: Applied,
    ) -> bool {
        match applied {
            Applied::Done => false,
            Applied::Waits(build) => {
                let tried = Some(self.room_state(renderer));
                self.waiting.push(WaitingTile {
                    generation,
                    source,
                    tile,
                    build,
                    tried,
                });
                true
            }
            Applied::Parks(build) => {
                let at = (generation, source, tile);
                let cap = self.cache_limit();
                (self.parked).park(at, build, cap, &mut self.retired.host);
                true
            }
            Applied::Uploading(build, reserved) => {
                self.uploading = Some(Uploading {
                    generation,
                    source,
                    tile,
                    build,
                    reserved,
                });
                true
            }
        }
    }

    /// Whether `message` is a vector tile's result that may go up in
    /// pieces: one the view waits for, while no other tile does.
    fn splits(&self, message: &Message) -> bool {
        let Message::Tile {
            result, terrain, ..
        } = message
        else {
            return false;
        };
        let built = matches!(&**result, Ok(TileOutcome::Built(b)) if b.image.is_none());
        built && !terrain && self.uploading.is_none()
    }

    /// What applying a result may write of a vector tile's words this
    /// frame: the frame's bytes left, or all of them while another tile
    /// goes up in pieces (the beat then took it whole).
    fn allowed(&self) -> u64 {
        match self.uploading {
            None => self.beat.left(),
            Some(_) => u64::MAX,
        }
    }

    /// What applying `message`, a tile's result, uploads: nothing for a
    /// result nothing waits for.
    fn upload_cost(&self, message: &Message) -> u64 {
        let Message::Tile {
            generation,
            source,
            tile,
            edit,
            terrain,
            result,
        } = message
        else {
            return 0;
        };
        let Ok(TileOutcome::Built(build)) = &**result else {
            return 0;
        };
        let wanted = *generation == self.tile_generation
            && (edit.is_some() || *terrain || self.awaited(*source, *tile) == Awaited::Wanted);
        if wanted { upload_bytes(build) } else { 0 }
    }

    /// What arrived for tile `tile` of the source keyed `source` finds.
    fn awaited(&self, source: u32, tile: TileIndex) -> Awaited {
        let Some(style) = &self.style else {
            return Awaited::No;
        };
        let index = style.worker.plan().source(source);
        match index.map(|i| &style.sources[i]) {
            Some(SourceState::Ready { pyramid, .. }) => pyramid.awaited(tile),
            _ => Awaited::No,
        }
    }

    /// A load's place in the order loads apply in: the fetch order
    /// ([`crate::frame::fetch_order`]), coarse stand-ins first, then by
    /// its tile's distance from the view's focus; results nothing waits
    /// for (given up, of a replaced style), `None`, first, since they
    /// take no upload.
    fn load_rank(&self, message: &Message) -> Option<u64> {
        let Message::Tile {
            generation,
            source,
            tile,
            terrain,
            ..
        } = message
        else {
            return None;
        };
        let style = self
            .style
            .as_ref()
            .filter(|_| *generation == self.tile_generation);
        let (endpoint, tier) = style.and_then(|style| match terrain {
            true => (style.terrain.as_ref())
                .filter(|t| t.key == *source)
                .map(|t| (&t.endpoint, crate::frame::Tier::View)),
            false => {
                let index = style.worker.plan().source(*source)?;
                match &style.sources[index] {
                    SourceState::Ready {
                        endpoint, pyramid, ..
                    } => {
                        let coarse = style.coarse.get(index).copied().flatten();
                        let used = pyramid.used_last(*tile);
                        Some((endpoint, crate::frame::Tier::of(*tile, coarse, used)))
                    }
                    _ => None,
                }
            }
        })?;
        (*terrain || self.awaited(*source, *tile) == Awaited::Wanted).then_some(())?;
        let order = crate::frame::fetch_order(&endpoint.tms, self.camera.state().target);
        Some(order(*tile, tier))
    }

    /// Applies an arrived message other than a tile's result.
    fn apply_message(&mut self, gpu: &mut dyn Gpu, renderer: &mut Renderer, message: Message) {
        match message {
            // Replies to the labels a GPU reset replaced are dropped.
            Message::Placed { epoch, committed } => {
                if epoch == self.label_epoch {
                    match committed {
                        // A commit applies whole, before the frame's
                        // tiles, which take what it leaves of the budget.
                        Ok(committed) => {
                            let started = self.services.clock.now();
                            let bytes = (committed.changed.iter())
                                .map(|(_, _, s)| {
                                    let words = [&s.text, &s.icon, &s.text_order, &s.icon_order];
                                    4 * words.iter().map(|w| w.len() as u64).sum::<u64>()
                                })
                                .sum();
                            self.labels.committed(gpu, renderer, committed);
                            self.beat.spend(bytes);
                            self.beat.count();
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
                    self.labels
                        .rastered(gpu, renderer.glyphs(), &rows, result, &mut self.warnings);
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
                    .filter(|s| generation == self.generation && epoch == s.sprites.epoch)
                {
                    style.sprites.loading -= 1;
                    match result {
                        Err(e) => {
                            let sprite = &style.worker.plan().style.sprites[index];
                            if !std::mem::replace(&mut style.sprites.failed[index], true) {
                                self.events.push(MapEvent::Sprite {
                                    id: sprite.id.clone(),
                                    url: sprite.url.clone(),
                                    progress: Progress::Failed(e.clone()),
                                });
                            }
                            let message = format!("sprite: {e}");
                            self.events.push(MapEvent::Error { message });
                        }
                        Ok(icons) => {
                            for icon in icons {
                                style.sprites.names.insert(icon.name.clone());
                                style.icons.add(
                                    &icon.name,
                                    [icon.width, icon.height],
                                    &icon.metadata,
                                );
                                style.sprites.icon_names[index].push(icon.name);
                            }
                        }
                    }
                    done = style.sprites.loading == 0;
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
                    return;
                };
                if let Some((url, head)) = head {
                    self.services.jobs.add_archive_head(&url, &Arc::new(head));
                }
                let plan = style.worker.plan();
                let Some(index) = plan.source(source) else {
                    return;
                };
                if !matches!(style.sources[index], SourceState::Resolving) {
                    return;
                }
                let def = plan.source_def(index);
                let tms = self.surface.source_tms(&def.id, false);
                let endpoint = match (&result, self.surface.plan(&tms, plan.image_kind(index))) {
                    (Err(e), _) => Err(format!("TileJSON: {e}")),
                    (Ok(_), Err(e)) => Err(e),
                    (Ok(text), Ok(warp)) => TileEndpoint::resolve(&def.kind, Some(text), tms)
                        .map(|endpoint| (endpoint, warp)),
                };
                // Tiles in MLT wait for its decoders.
                let mlt = crate::modules::Modules::MLT;
                if let Ok((endpoint, _)) = &endpoint
                    && endpoint.encoding == orb_style::Encoding::Mlt
                    && plan.image_kind(index).is_none()
                    && matches!(def.kind, orb_style::SourceKind::Vector(_))
                    && !self.modules.contains(mlt)
                {
                    self.modules_wanted |= mlt;
                    let message = Message::TileJson {
                        generation,
                        source,
                        result,
                        head: None,
                    };
                    (self.for_modules).push((mlt, crate::map::ModuleWait::TileJson(message)));
                    self.events.wake();
                    return;
                }
                let image = plan.image_kind(index).is_some();
                if endpoint.is_ok() {
                    self.events.push(MapEvent::SourceChanged {
                        source: def.id.clone(),
                    });
                }
                style.sources[index] = ready_source(endpoint, &def.id, image, &mut self.events);
                self.events.sources.insert(source);
                // Color-relief ramps wait for their source's encoding.
                style.paint.invalidate();
                for node in plan.nodes.iter().filter(|n| n.source == index) {
                    style.nodes.entry(node.key).or_default().ramp = None;
                }
            }
            Message::Tile { .. } => unreachable!("tiles' results queue"),
        }
    }

    /// Applies a tile's result, a load's or an edit's: what a load
    /// uploads counts against the frame's bytes.
    fn apply_tile_message(&mut self, gpu: &mut dyn Gpu, renderer: &mut Renderer, message: Message) {
        let allowed = self.allowed();
        let Message::Tile {
            generation,
            source,
            tile,
            edit,
            terrain,
            result,
        } = message
        else {
            unreachable!("a tile's result");
        };
        let mut result = *result;
        self.applied += 1;
        // A tile of heights the app gives stands on those shown now: a
        // batch shown since its load began shows on it too.
        if let Ok(TileOutcome::Built(build)) = &mut result {
            self.host_heights.refresh(source, tile, build);
        }
        let host = match &result {
            Ok(TileOutcome::Built(build)) => Some(HostData::of(build)),
            _ => None,
        };
        if let Some(edit) = edit {
            self.finish_job(edit);
        }
        let mut kept = false;
        if generation == self.tile_generation {
            // Loads given up, missing or failed brought nothing.
            self.tiles_arrived +=
                u32::from(edit.is_none() && matches!(result, Ok(TileOutcome::Built(_))));
            match edit {
                // A DEM tile's texels go up whole, against the frame's bytes.
                None if terrain => {
                    let bytes = match &result {
                        Ok(TileOutcome::Built(build)) => upload_bytes(build),
                        _ => 0,
                    };
                    self.apply_terrain_tile(gpu, renderer, source, tile, result);
                    self.beat.spend(bytes);
                }
                None => {
                    let mut reserved = None;
                    let applied = self.apply_tile_with(
                        gpu,
                        renderer,
                        (source, tile),
                        result,
                        (true, allowed),
                        &mut reserved,
                    );
                    kept = self.settle_applied(renderer, (generation, source, tile), applied);
                }
                Some(_) => self.apply_edit(gpu, renderer, source, tile, result),
            }
        }
        // Uploaded or dropped, the result's host data is done.
        if let Some(host) = host.filter(|_| !kept) {
            host.release(gpu);
        }
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
            .filter(|s| generation == self.generation && epoch == s.sprites.epoch)
        else {
            return;
        };
        style.sprites.pixels_loading -= 1;
        let mut relayout = false;
        let sprite = &style.worker.plan().style.sprites[index];
        let failed = std::mem::replace(&mut style.sprites.failed[index], result.is_err());
        if !failed {
            self.events.push(MapEvent::Sprite {
                id: sprite.id.clone(),
                url: sprite.url.clone(),
                progress: match &result {
                    Ok(_) => Progress::Loaded,
                    Err(e) => Progress::Failed(e.clone()),
                },
            });
        }
        match result {
            Err(e) => {
                let message = format!("sprite: {e}");
                self.events.push(MapEvent::Error { message });
                let mut withdrawn = false;
                for name in std::mem::take(&mut style.sprites.icon_names[index]) {
                    if !style.added_images.contains_key(&name) {
                        style.sprites.names.remove(&name);
                        withdrawn |= style.icons.remove(&name);
                    }
                }
                // While other sprites' JSON loads, the set goes with theirs.
                if withdrawn && style.sprites.loading == 0 {
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
        if style.sprites.pixels_loading == 0 {
            relayout |= std::mem::take(&mut style.sprites.relayout_after_pixels);
        }
        if relayout {
            self.relayout_symbols();
        }
    }

    /// Tries the tiles waiting for room again (and those parked that the
    /// view came back to), in the frame's budget left (until `deadline`),
    /// each only if room may have changed since it last failed: memory
    /// freed or grown, or other tiles' jobs done. Whether the budget left
    /// some untried.
    pub(crate) fn retry_waiting(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        deadline: f64,
    ) -> bool {
        let mut state = self.room_state(renderer);
        let mut deferred = false;
        for mut waiting in std::mem::take(&mut self.waiting) {
            let host = HostData::of(&waiting.build);
            if waiting.generation != self.tile_generation {
                host.release(gpu);
                continue;
            }
            if deferred || waiting.tried == Some(state) {
                self.waiting.push(waiting);
                continue;
            }
            let bytes = upload_bytes(&waiting.build);
            let in_time = self.services.clock.now() < deadline;
            let splits = waiting.build.image.is_none() && self.uploading.is_none();
            if !self.beat.takes(in_time, bytes, splits) {
                deferred = true;
                self.waiting.push(waiting);
                continue;
            }
            self.applied += 1;
            let (allowed, mut reserved) = (self.allowed(), None);
            let result = Ok(TileOutcome::Built(waiting.build));
            let at = (waiting.source, waiting.tile);
            let applied =
                self.apply_tile_with(gpu, renderer, at, result, (false, allowed), &mut reserved);
            self.beat.count();
            match applied {
                Applied::Waits(build) => {
                    state = self.room_state(renderer);
                    waiting.build = build;
                    waiting.tried = Some(state);
                    self.waiting.push(waiting);
                }
                Applied::Done => {
                    host.release(gpu);
                    state = self.room_state(renderer);
                }
                applied => {
                    let at = (waiting.generation, waiting.source, waiting.tile);
                    self.settle_applied(renderer, at, applied);
                }
            }
        }
        deferred
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
                SourceState::Ready { pyramid, .. } => pyramid.loading() as u64,
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
                .map_or(0, |s| u64::from(s.sprites.pixels_loading)),
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

    /// Applies a tile's result whole: [`apply_tile_with`](Self::apply_tile_with)
    /// with no bound on what it writes.
    pub(crate) fn apply_tile(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        source: u32,
        tile: TileIndex,
        result: Result<TileOutcome, String>,
        arrived: bool,
    ) -> Applied {
        let mut reserved = None;
        let at = (source, tile);
        self.apply_tile_with(
            gpu,
            renderer,
            at,
            result,
            (arrived, u64::MAX),
            &mut reserved,
        )
    }

    /// Applies tile `tile` of the source keyed `source`'s result, whose
    /// load just answered if `arrived`. A vector tile whose words come to
    /// more than `allowed` bytes takes room for them and writes that many
    /// ([`Applied::Uploading`]); one whose words are all written in
    /// `reserved` is made of them. What it doesn't take of `reserved`
    /// goes back.
    pub(crate) fn apply_tile_with(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        at: (u32, TileIndex),
        result: Result<TileOutcome, String>,
        allowed: (bool, u64),
        reserved: &mut Option<orb_render::Reserved>,
    ) -> Applied {
        let applied = self.apply_tile_in(gpu, renderer, at, result, allowed, reserved);
        if let Some(reserved) = reserved.take() {
            renderer.unreserve(reserved);
        }
        applied
    }

    fn apply_tile_in(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        (source, tile): (u32, TileIndex),
        result: Result<TileOutcome, String>,
        (arrived, allowed): (bool, u64),
        reserved: &mut Option<orb_render::Reserved>,
    ) -> Applied {
        // A tile made of words written over frames spent them as they went;
        // if it isn't made of them after all, they go up again and count.
        let mut prepaid = reserved.as_ref().map_or(0, orb_render::Reserved::written);
        let Some(style) = &mut self.style else {
            return Applied::Done;
        };
        let plan = style.worker.plan();
        let key = source;
        let Some(source) = plan.source(key) else {
            return Applied::Done;
        };
        let SourceState::Ready {
            pyramid,
            endpoint,
            warped,
            ..
        } = &mut style.sources[source]
        else {
            return Applied::Done;
        };
        let warped = *warped;
        if arrived {
            self.events.sources.insert(key);
            let progress = match &result {
                Err(e) => Progress::Failed(e.clone()),
                Ok(TileOutcome::Cancelled) => Progress::Cancelled,
                Ok(_) => Progress::Loaded,
            };
            self.events.push(MapEvent::Tile {
                source: plan.source_def(source).id.clone(),
                tile,
                progress,
            });
            // An app's answer a reload forgot: asked for again instead.
            if self.host_tiles.landed_stale(key, tile) {
                pyramid.forget_loading(tile);
                self.events.wake();
                return Applied::Done;
            }
        }
        let mut build = match result {
            Ok(TileOutcome::Built(build)) => build,
            Err(e) => {
                let message = format!("tile {tile}: {e}");
                self.events.push(MapEvent::Error { message });
                pyramid.insert_failed(tile, self.services.clock.now());
                return Applied::Done;
            }
            Ok(TileOutcome::Missing { mut warnings }) => {
                self.warnings.append(&mut warnings);
                // A vector tile left out of its tileset may mean no data
                // there ([`MissingTiles`]); an image tile's ancestor fills
                // the hole.
                if self.missing_tiles == crate::MissingTiles::Empty
                    && plan.image_kind(source).is_none()
                {
                    pyramid.insert_empty(tile);
                } else {
                    pyramid.insert_missing(tile);
                }
                return Applied::Done;
            }
            // Given up before it loaded: a later plan that wants it asks
            // again.
            Ok(TileOutcome::Cancelled) => {
                pyramid.forget_loading(tile);
                return Applied::Done;
            }
        };
        match pyramid.awaited(tile) {
            // No longer in flight: other tiles' loads are what this one
            // waits for, if it waits.
            Awaited::Wanted => pyramid.arrived(tile),
            // Built after its load was given up, it isn't uploaded: parked,
            // so that a view that comes back to it costs an upload, not a
            // fetch and layout. An image's decode is cheap to do again.
            Awaited::GivenUp => {
                pyramid.forget_loading(tile);
                let parks = build.image.is_none() && self.mode == RenderMode::Continuous;
                return if parks {
                    Applied::Parks(build)
                } else {
                    Applied::Done
                };
            }
            Awaited::No => return Applied::Done,
        }
        self.warnings.append(&mut build.warnings);
        crate::style_image::report_missing(
            &mut style.reported_missing,
            &style.icons,
            style.sprites.loading,
            &build,
            &mut self.events,
        );
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
        // Labels left out while their glyphs loaded follow as a relayout,
        // as stale nodes do.
        for &key in &build.labels_later {
            if !stale.contains(&key) && plan.node(key).is_some() {
                stale.push(key);
            }
        }
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
                return Applied::Done;
            };
            // Raster tiles fade in over their layers' longest duration;
            // still output, and a map not yet idle once (as GL JS's first
            // view, which has no tiles to fade from), shows them at once.
            let fades = self.mode == RenderMode::Continuous
                && self.events.idled
                && !matches!(plan.source_def(source).kind, SourceKind::Image(_));
            if fades {
                fade_ms = raster_fade_ms(plan, source, tile_zoom);
            }
            image = Some((tile_image, warped));
        }
        // A tile on another plane draws through its series' patches; a
        // warped image only on a globe surface.
        let series = (warped && (image.is_none() || globe))
            .then(|| self.globe_warps.surface_series(&endpoint.tms, tile))
            .flatten();
        let patches = series.as_deref().map(series_patches);
        let patches = patches.as_ref().map(|(cells, p)| (*cells, &p[..]));
        let fade_from = (fade_ms > 0.0).then_some((now - self.born.unwrap_or(now)) as f32);
        let image_build = image.as_ref().map(|&(tile_image, ref warped)| {
            let mesh = warped.as_ref().map(|w| Arc::clone(&w.mesh));
            let at = (plan, source, key, tile);
            image_build(at, tile_image, mesh, patches, (fade_from, endpoint))
        });
        // Labels take their images' atlas rectangles as they upload: a tile
        // whose labels show images waits for the sprites' pixels (layout
        // needed only their JSON).
        if style.awaits_sprite_pixels(&build) {
            return Applied::Waits(build);
        }
        // Room: the scene's blocks, tables and the source's page grow to
        // hold it, whatever the camera does. Past the device's limits,
        // unused tiles go, this source's before others': the cache, within
        // a view of the tiles in view (`trim_cache`), is held too. While
        // other tiles load (they free the stand-ins they replace), this one
        // waits. Otherwise, for a tile of the cover, the tiles it replaces
        // go, then tiles that only stand in (a fresh view of this cover
        // would hold none of them), its own ancestors first. A tile outside
        // the cover takes nothing from it. A tile the plan no longer wants
        // (neither in the cover nor standing in for a tile of it that
        // failed) is forgotten; one it wants gives way, to be tried again
        // later, and its ancestor is asked to stand in.
        let others_loading = style
            .sources
            .iter()
            .any(|s| matches!(s, SourceState::Ready { pyramid, .. } if pyramid.loading() > 0));
        let mut images = self.services.images.0.lock().expect("images poisoned");
        let atlas = images.atlas.as_mut().expect("initialized");
        let uploading = self.services.clock.now();
        // An image tile's page grows only once its source's tiles kept out
        // of view leave it no layer: those give a layer back, where other
        // sources' tiles hold none of the page.
        let mut grow = false;
        let uploaded = loop {
            atlas.release_ramps(gpu, self.retired.ramps.drain(..));
            use orb_render::UploadError::{Invalid, NoRoom};
            let attempt = match &image_build {
                Some(image_build) => renderer
                    .upload_image(gpu, image_build, grow)
                    .map(|handle| Upload::Resident(ResidentTile::new(handle, tile_zoom))),
                // Its words written, the tile is made of them.
                None if let Some(written) = reserved.take() => {
                    let made = upload(
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
                        Some(written),
                    );
                    // Its room went back: its words go up again.
                    if made.is_err() {
                        prepaid = 0;
                    }
                    made.map(Upload::Resident).map_err(|OutOfSpace| NoRoom)
                }
                None if upload_bytes(&build) > allowed => {
                    crate::resident::reserve(gpu, renderer, plan, key, &build).map(Upload::Reserved)
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
                    None,
                )
                .map(Upload::Resident)
                .map_err(|OutOfSpace| NoRoom),
            };
            let victim = match attempt {
                Ok(uploaded) => break Ok(uploaded),
                Err(Invalid(e)) => break Err(Some(e)),
                Err(NoRoom) if image_build.is_some() && !grow => match &mut style.sources[source] {
                    SourceState::Ready { pyramid, .. } => pyramid.evict_lru(),
                    _ => None,
                },
                Err(NoRoom) => evict_unused(&mut style.sources, source),
            };
            let victim = match victim {
                Some(victim) => Some(victim),
                None if image_build.is_some() && !grow => {
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
                    // A stand-in may be held for its labels: they go with it
                    // (`update_holds` finds it gone).
                    for symbols in &victim.symbols {
                        self.labels.unhold(symbols.bucket);
                    }
                    self.retired
                        .ramps
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
            Ok(Upload::Resident(resident)) => {
                self.beat
                    .spend(upload_bytes(&build).saturating_sub(prepaid));
                resident
            }
            Ok(Upload::Reserved(mut reserved)) => {
                let words = build.payload.words();
                let written = renderer.write_reserved(gpu, words, &mut reserved, allowed);
                self.beat.spend(written);
                return Applied::Uploading(build, reserved);
            }
            Err(Some(e)) => {
                self.warnings.push(format!("tile {tile}: {e}"));
                pyramid.insert_empty(tile);
                return Applied::Done;
            }
            Err(None) if others_loading => return Applied::Waits(build),
            Err(None) if !wanted => {
                pyramid.forget_loading(tile);
                return Applied::Done;
            }
            // Even the tiles in view don't leave room: what's loaded stands
            // in, and the tile is tried again after a while (the view may
            // have changed), as after a failed fetch.
            Err(None) => {
                let message = format!(
                    "tile {tile} doesn't fit in scene memory with the tiles in view; \
                     what's loaded stands in until it's tried again"
                );
                self.events.push(MapEvent::Error { message });
                pyramid.insert_failed(tile, self.services.clock.now());
                return Applied::Done;
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
        resident.neighbour = build.neighbour;
        resident.glyphs_failed = build.glyphs_failed;
        (style.glyph_retry).arrived(build.glyphs_failed, false, self.services.clock.now());
        self.counts.meshes_uploaded += build.meshes.len() as u32;
        record_maxima(&mut style.nodes, &build.meshes, &mut style.paint);
        resident.cost = tile_cost(renderer, &resident);
        if let Some(unwanted) = pyramid.insert(tile, resident) {
            self.retired
                .ramps
                .extend(remove_resident(renderer, unwanted, &mut style.states));
        }
        if !stale.is_empty() {
            let version = self.version;
            self.send_edit_job(source, tile, Some(stale), false, version);
        }
        Applied::Done
    }
}

/// What became of a tile's result ([`Map::apply_tile`]).
pub(crate) enum Applied {
    /// Uploaded, or settled without (missing, empty, failed or dropped).
    Done,
    /// Waits for other tiles to load, which free the stand-ins they
    /// replace, or for the sprites' pixels its labels show.
    Waits(TileResult),
    /// Built after its load was given up ([`Parked`](crate::schedule::Parked)).
    Parks(TileResult),
    /// Has room for its words and goes up over frames ([`Uploading`]).
    Uploading(TileResult, orb_render::Reserved),
}

/// What an attempt at a tile's upload made room for.
enum Upload {
    Resident(ResidentTile),
    Reserved(orb_render::Reserved),
}

/// A vector tile going up over frames, its words written into room it
/// holds ([`orb_render::Reserved`]): it shows once every word is in, so
/// that what a frame shows never has part of it, and no frame writes more
/// of it than its budget's bytes.
#[derive(Debug)]
pub(crate) struct Uploading {
    pub(crate) generation: u64,
    /// The plan source's key.
    pub(crate) source: u32,
    pub(crate) tile: TileIndex,
    pub(crate) build: TileResult,
    pub(crate) reserved: orb_render::Reserved,
}

/// The longest fade of `source`'s raster layers at `tile_zoom`, in
/// milliseconds: what its tiles fade in over.
fn raster_fade_ms(plan: &crate::style::StylePlan, source: usize, tile_zoom: f64) -> f64 {
    (plan.nodes.iter())
        .filter(|n| n.source == source)
        .map(|n| &plan.style.layers[n.layer])
        .filter(|r| r.ty == LayerType::Raster)
        .map(|r| f64::from(r.get(orb_style::RASTER_FADE_DURATION).evaluate(tile_zoom)))
        .fold(0.0, f64::max)
}

/// What the renderer uploads of image tile `tile` (`key`) of `plan`'s
/// source `source`: its pixels, drawn through `mesh` (its square without)
/// and on a globe through `patches`, and when it arrived if it fades in
/// (`fade_from`).
fn image_build<'a>(
    (plan, source, key, tile): (&crate::style::StylePlan, usize, TileKey, TileIndex),
    tile_image: &'a crate::jobs::TileImage,
    mesh: Option<Arc<orb_render::ImageMesh>>,
    patches: Option<(u32, &'a [f32])>,
    (fade_from, endpoint): (Option<f32>, &TileEndpoint),
) -> orb_render::ImageBuild<'a> {
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
        mesh,
        patches,
        fade_from,
        unpack: endpoint.dem_unpack.map(|u| u.map(|v| v as f32)),
        area: inside_texels(&endpoint.tms, tile, [tile_image.width, tile_image.height]),
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
