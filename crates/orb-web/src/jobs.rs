//! Tile jobs across workers: both ends of the
//! protocol between the page and its job workers.
//!
//! The page's JS owns the transport, one `MessagePort` per job worker, and
//! passes messages through unread. The page's [`WorkerJobs`] posts each
//! request as a [`JobRequest`] in [`orb_runtime::wire`]'s bytes; a job
//! worker's [`JobWorker`] answers each one with an id as
//! `{message, buffer?, image?}`: a [`JobReply`], with a packed tile ([`pack`])
//! or a DEM tile's pixels as the transferred `buffer` and an image tile as
//! the transferred `image` bitmap. The page copies only a packed tile's
//! header into wasm memory; the payload goes from the buffer to the GPU.
//! A tile goes to the job worker with the fewest requests in flight (an
//! overzoomed one to its ancestor's), and its relayouts and queries follow
//! it there, where its data is; a glyph unit goes to its owner worker. A
//! worker that is lost (its wasm trapped, or it died) fails the queries it
//! had, which the map retries, and its rasters go again (their glyphs'
//! atlas rows wait for them). The page puts another in its place, which
//! this sends the style; or, past the page's limit, its share goes to the
//! next worker on. The loads it had go again
//! at once, as suspects: one at a time, with no other suspect in flight, so
//! a loss then is likely its own. A suspect lost again is quarantined: it
//! loads as missing, so its ancestor stands in, rather than take down a
//! worker each time it is loaded.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use js_sys::{ArrayBuffer, Object, Reflect};
use orb_crs::TileIndex;
use orb_hal::Gpu as _;
use orb_runtime::wire;
use orb_runtime::{
    DemUnpack, GlyphRaster, ImageKind, ImagePixels, JobHost, JobReply, JobRequest, Jobs, Loaded,
    RasterDone, SpriteIcons, StyleUpdate, TileBuild, TileDone, TileImage, TileRequest, TileResult,
    TileWorker, pack, packed_header, unpack,
};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

thread_local! {
    /// Each map's `self.post(worker, message)`, which sends a message to its job
    /// worker `worker`, by map. Wasm in a page is single-threaded, so the
    /// runtime's `Send` jobs reach it here.
    static POSTS: std::cell::RefCell<HashMap<u32, js_sys::Function>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Forgets map `map`'s job workers' `post`.
pub(crate) fn forget(map: u32) {
    POSTS.with(|posts| posts.borrow_mut().remove(&map));
}

/// [`Jobs`] for the map: each tile goes to the job worker with
/// the fewest requests in flight, and its relayouts and queries follow it
/// there, where its data was decoded.
pub(crate) struct WorkerJobs {
    /// The map whose job workers these are.
    map: u32,
    workers: u32,
    state: Mutex<State>,
    /// Itself, for fetches that answer later.
    me: std::sync::Weak<WorkerJobs>,
}

impl std::fmt::Debug for WorkerJobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerJobs")
            .field("workers", &self.workers)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct State {
    next_id: u32,
    pending: HashMap<u32, TileDone>,
    rasters: HashMap<u32, RasterDone>,
    /// Each raster's request, to send again if its worker is lost.
    raster_requests: HashMap<u32, GlyphRaster>,
    features: HashMap<u32, orb_runtime::FeaturesDone>,
    answers: HashMap<u32, orb_runtime::QueryDone>,
    /// The worker each request went to, until it answers, and a load's
    /// request, to send again if the worker is lost.
    sent_to: HashMap<u32, (u32, Option<TileRequest>)>,
    /// Tiles whose loads were lost with a worker, by how many times: one
    /// is a suspect, two quarantine it.
    lost_loads: HashMap<LoadKey, u8>,
    /// The suspect's load in flight, and the loads of those waiting.
    suspect: Option<u32>,
    suspects: VecDeque<(TileRequest, TileDone)>,
    /// The PMTiles archive heads the map read, for a replacement worker.
    heads: Vec<(String, Arc<Vec<u8>>)>,
    /// Workers lost.
    lost: Vec<bool>,
    /// The worker each tile (by source; an overzoomed tile by the ancestor
    /// it slices, so that they share its decode) went to, the latest
    /// [`HOMES`] of them, oldest first in `home_order`.
    homes: HashMap<(u32, TileIndex), u32>,
    home_order: VecDeque<(u32, TileIndex)>,
    /// Bytes of packed headers copied into this worker's wasm memory.
    header_bytes: u64,
    /// Replies received.
    done: u32,
}

/// A load's style, source and tile.
type LoadKey = (u64, u32, TileIndex);

fn load_key(request: &TileRequest) -> LoadKey {
    (request.style, request.source, request.tile)
}

/// Losses of one tile's loads that quarantine it.
const QUARANTINE: u8 = 2;

/// What the map's jobs have done so far.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JobStats {
    pub header_bytes: u64,
    pub done: u32,
}

impl WorkerJobs {
    /// Map `map`'s jobs on `workers` job workers, reached through `post`.
    pub(crate) fn new(map: u32, workers: u32, post: js_sys::Function) -> Arc<Self> {
        POSTS.with(|posts| posts.borrow_mut().insert(map, post));
        let workers = workers.max(1);
        Arc::new_cyclic(|me| Self {
            map,
            workers,
            state: Mutex::new(State {
                lost: vec![false; workers as usize],
                ..State::default()
            }),
            me: me.clone(),
        })
    }

    fn post(&self, worker: u32, message: &JsValue) {
        POSTS.with(|posts| {
            if let Some(f) = posts.borrow().get(&self.map) {
                let _ = f.call2(&JsValue::NULL, &worker.into(), message);
            }
        });
    }

    pub(crate) fn stats(&self) -> JobStats {
        let state = self.state.lock().expect("jobs poisoned");
        JobStats {
            header_bytes: state.header_bytes,
            done: state.done,
        }
    }

    /// Completes a request with a job worker's reply. A packed tile's
    /// buffer is registered with `gpu`; only its header enters wasm memory.
    pub(crate) fn receive(
        &self,
        gpu: &mut orb_hal_wgpu::WgpuGpu,
        reply: &JsValue,
    ) -> Result<(), JsError> {
        let message = field(reply, "message")
            .dyn_into::<js_sys::Uint8Array>()
            .map_err(|_| JsError::new("job reply without a message"))?;
        let message: JobReply = wire::decode(&message.to_vec()).map_err(|e| JsError::new(&e))?;
        let buffer = field(reply, "buffer").dyn_into::<ArrayBuffer>().ok();
        self.complete(Some(gpu), message, buffer, reply)
    }

    /// Job worker `worker` is lost. With `replacement`, another worker
    /// took its place and gets the style first (if one has loaded) and the
    /// archive heads; without, the next worker on takes its share. The
    /// loads it had go again, as suspects, or answer missing once
    /// quarantined; its other requests fail.
    pub(crate) fn lost(
        &self,
        worker: u32,
        why: &str,
        replacement: Option<Option<(u64, StyleUpdate, SpriteIcons)>>,
    ) {
        let (failed, loads, rasters, heads) = {
            let mut state = self.state.lock().expect("jobs poisoned");
            let Some(lost) = state.lost.get_mut(worker as usize) else {
                return;
            };
            *lost = replacement.is_none();
            let ids: Vec<u32> = state
                .sent_to
                .iter()
                .filter(|&(_, &(w, _))| w == worker)
                .map(|(&id, _)| id)
                .collect();
            let (mut failed, mut loads, mut rasters) = (Vec::new(), Vec::new(), Vec::new());
            for id in ids {
                let Some((_, request)) = state.sent_to.remove(&id) else {
                    continue;
                };
                if state.suspect == Some(id) {
                    state.suspect = None;
                }
                if let Some(raster) = state.raster_requests.remove(&id) {
                    if let Some(done) = state.rasters.remove(&id) {
                        rasters.push((raster, done));
                    }
                    continue;
                }
                let Some(request) = request else {
                    failed.push(id);
                    continue;
                };
                let n = state.lost_loads.entry(load_key(&request)).or_default();
                *n = n.saturating_add(1);
                if let Some(done) = state.pending.remove(&id) {
                    loads.push((request, done));
                }
            }
            (failed, loads, rasters, state.heads.clone())
        };
        if let Some(style) = replacement {
            if let Some((style, update, icons)) = style {
                let update = Box::new(update);
                self.post(
                    worker,
                    &bytes(&wire::encode(&JobRequest::Style { style, update })),
                );
                self.post(
                    worker,
                    &bytes(&wire::encode(&JobRequest::Icons { style, icons })),
                );
            }
            for (url, head) in heads {
                let head = (*head).clone();
                self.post(
                    worker,
                    &bytes(&wire::encode(&JobRequest::ArchiveHead { url, head })),
                );
            }
        }
        let message = format!("job worker {worker} lost: {why}");
        for id in failed {
            let message = message.clone();
            let _ = self.complete(
                None,
                JobReply::Error { id, message },
                None,
                &JsValue::UNDEFINED,
            );
        }
        for (request, done) in rasters {
            self.raster(request, done);
        }
        for (request, done) in loads {
            if self.quarantined(load_key(&request)) {
                done(Ok(quarantine(format!(
                    "tile {:?} is left out from now on: {message} loading it a second time",
                    request.tile
                ))));
            } else {
                self.load(request, done);
            }
        }
        self.next_suspect();
    }

    fn quarantined(&self, load: LoadKey) -> bool {
        let state = self.state.lock().expect("jobs poisoned");
        state
            .lost_loads
            .get(&load)
            .is_some_and(|&n| n >= QUARANTINE)
    }

    /// Sends the next suspect's load, unless one is in flight.
    fn next_suspect(&self) {
        let next = {
            let mut state = self.state.lock().expect("jobs poisoned");
            if state.suspect.is_some() {
                return;
            }
            state.suspects.pop_front()
        };
        if let Some((request, done)) = next {
            self.load(request, done);
        }
    }

    /// Completes the request `message` answers; tile and image replies
    /// register their buffers with `gpu`.
    fn complete(
        &self,
        gpu: Option<&mut orb_hal_wgpu::WgpuGpu>,
        message: JobReply,
        buffer: Option<ArrayBuffer>,
        reply: &JsValue,
    ) -> Result<(), JsError> {
        let id = message.id();
        let mut state = self.state.lock().expect("jobs poisoned");
        let sent = state.sent_to.remove(&id);
        if let Some(done) = state.features.remove(&id) {
            drop(state);
            done(match message {
                JobReply::Features { features, .. } => Ok(features),
                JobReply::Error { message, .. } => Err(message),
                _ => Err("malformed features reply".into()),
            });
            return Ok(());
        }
        if let Some(done) = state.answers.remove(&id) {
            drop(state);
            done(match message {
                JobReply::Answer { json, .. } => Ok(json),
                JobReply::Error { message, .. } => Err(message),
                _ => Err("malformed cluster reply".into()),
            });
            return Ok(());
        }
        if let Some(done) = state.rasters.remove(&id) {
            state.raster_requests.remove(&id);
            drop(state);
            done(match message {
                JobReply::Glyphs { bitmaps, .. } => Ok(bitmaps),
                JobReply::Error { message, .. } => Err(message),
                _ => Err("malformed raster reply".into()),
            });
            return Ok(());
        }
        state.done += 1;
        let done = state
            .pending
            .remove(&id)
            .ok_or_else(|| JsError::new(&format!("reply to unknown job {id}")))?;
        // A suspect answered: it didn't take its worker down.
        let answered = state.suspect == Some(id);
        if answered {
            state.suspect = None;
            if let Some((_, Some(request))) = &sent {
                state.lost_loads.remove(&load_key(request));
            }
        }
        let result = match message {
            JobReply::Missing { .. } => Ok(TileResult {
                missing: true,
                ..TileResult::default()
            }),
            JobReply::Cancelled { .. } => Ok(TileResult {
                cancelled: true,
                ..TileResult::default()
            }),
            // A DEM tile, decoded by the core in the job worker.
            JobReply::Dem {
                width, elevations, ..
            } => {
                let pixels = buffer.map(|b| js_sys::Uint8Array::new(&b).to_vec());
                orb_tile::DemGrid::from_elevations(width, elevations).map(|dem| TileResult {
                    image: Some(TileImage {
                        width,
                        height: width,
                        pixels: ImagePixels::Core(pixels.unwrap_or_default()),
                    }),
                    dem: Some(Arc::new(dem)),
                    ..TileResult::default()
                })
            }
            JobReply::Image { warp_grid, .. } => {
                match (
                    field(reply, "image").dyn_into::<web_sys::ImageBitmap>(),
                    gpu,
                ) {
                    (Ok(bitmap), Some(gpu)) => Ok(TileResult {
                        image: Some(TileImage {
                            width: bitmap.width(),
                            height: bitmap.height(),
                            pixels: ImagePixels::Host(gpu.register_image_bitmap(bitmap)),
                        }),
                        warp_grid: warp_grid.map(Arc::new),
                        ..TileResult::default()
                    }),
                    _ => Err("image reply without a bitmap".into()),
                }
            }
            JobReply::Tile { .. } => match (buffer, gpu) {
                (Some(buffer), Some(gpu)) => {
                    let header = header_bytes(&buffer);
                    if let Ok(header) = &header {
                        state.header_bytes += header.len() as u64;
                    }
                    header.and_then(|header| {
                        let bytes = gpu.register_array_buffer(buffer);
                        let result = unpack(&header, bytes);
                        if result.is_err() {
                            gpu.release_host_bytes(bytes);
                        }
                        result
                    })
                }
                _ => Err("tile reply without a buffer".into()),
            },
            JobReply::Error { message, .. } => Err(message),
            _ => Err("malformed job reply".into()),
        };
        drop(state);
        done(result);
        if answered {
            self.next_suspect();
        }
        Ok(())
    }
}

/// A quarantined tile's load: missing, so its ancestor stands in, with why
/// as a warning.
fn quarantine(why: String) -> TileResult {
    TileResult {
        missing: true,
        warnings: (!why.is_empty()).then_some(why).into_iter().collect(),
        ..TileResult::default()
    }
}

/// Tiles whose worker is remembered: far more than the workers keep
/// decoded. A tile forgotten goes to a worker afresh, which fetches it again.
const HOMES: usize = 8192;

impl State {
    /// The worker `tile` of `source` went to, or the one with the fewest
    /// requests in flight, which becomes its home.
    fn home(&mut self, source: u32, tile: TileIndex) -> u32 {
        let key = (source, tile);
        if let Some(&worker) = self.homes.get(&key) {
            return worker;
        }
        let mut busy = vec![0usize; self.lost.len()];
        for &(worker, _) in self.sent_to.values() {
            busy[worker as usize] += 1;
        }
        let worker = (0..self.lost.len())
            .filter(|&w| !self.lost[w])
            .min_by_key(|&w| busy[w])
            .unwrap_or(0) as u32;
        if self.home_order.len() == HOMES
            && let Some(oldest) = self.home_order.pop_front()
        {
            self.homes.remove(&oldest);
        }
        self.homes.insert(key, worker);
        self.home_order.push_back(key);
        worker
    }
}

/// Copies a packed tile's header out of `buffer`: its length, then its
/// bytes ([`packed_header`]).
fn header_bytes(buffer: &ArrayBuffer) -> Result<Vec<u8>, String> {
    let bad = || "malformed packed tile".to_string();
    if buffer.byte_length() < 4 {
        return Err(bad());
    }
    let len = js_sys::Uint8Array::new_with_byte_offset_and_length(buffer, 0, 4).to_vec();
    let len = u32::from_le_bytes(len.try_into().expect("four bytes"));
    if 4 + len > buffer.byte_length() {
        return Err(bad());
    }
    let prefix = js_sys::Uint8Array::new_with_byte_offset_and_length(buffer, 0, 4 + len).to_vec();
    Ok(packed_header(&prefix)?.to_vec())
}

/// The first worker not lost from `preferred` on, if any is left.
fn route(lost: &[bool], preferred: u32) -> Option<u32> {
    let n = lost.len() as u32;
    (0..n)
        .map(|i| (preferred + i) % n)
        .find(|&w| !lost[w as usize])
}

impl WorkerJobs {
    /// Sends the request `make` builds with its id to the worker
    /// `preferred` picks (or the next one on, while it's lost), its done
    /// kept by `keep`. With every worker lost, the request fails at once.
    fn send(
        &self,
        preferred: impl FnOnce(&mut State) -> u32,
        load: Option<TileRequest>,
        keep: impl FnOnce(&mut State, u32),
        make: impl FnOnce(u32) -> JobRequest,
    ) {
        let mut state = self.state.lock().expect("jobs poisoned");
        let preferred = preferred(&mut state);
        let id = state.next_id;
        state.next_id = id.wrapping_add(1);
        keep(&mut state, id);
        let Some(worker) = route(&state.lost, preferred) else {
            drop(state);
            let message = "every job worker was lost".to_string();
            let _ = self.complete(
                None,
                JobReply::Error { id, message },
                None,
                &JsValue::UNDEFINED,
            );
            return;
        };
        state.sent_to.insert(id, (worker, load));
        drop(state);
        self.post(worker, &bytes(&wire::encode(&make(id))));
    }

    /// Posts `message` to every worker not lost.
    fn broadcast(&self, message: &JsValue) {
        let lost = self.state.lock().expect("jobs poisoned").lost.clone();
        for worker in (0..self.workers).filter(|&w| !lost[w as usize]) {
            self.post(worker, message);
        }
    }
}

impl Jobs for WorkerJobs {
    fn set_style(&self, style: u64, update: &StyleUpdate, _: &Arc<TileWorker>) {
        self.broadcast(&bytes(&wire::encode(&JobRequest::Style {
            style,
            update: Box::new(update.clone()),
        })));
    }

    fn add_archive_head(&self, url: &str, head: &Arc<Vec<u8>>) {
        {
            let mut state = self.state.lock().expect("jobs poisoned");
            state.heads.retain(|(u, _)| u != url);
            state.heads.push((url.to_string(), Arc::clone(head)));
            // As many as a job host keeps.
            if state.heads.len() > 8 {
                state.heads.remove(0);
            }
        }
        self.broadcast(&bytes(&wire::encode(&JobRequest::ArchiveHead {
            url: url.to_string(),
            head: (**head).clone(),
        })));
    }

    fn set_icons(&self, style: u64, icons: &Arc<SpriteIcons>, _: &Arc<TileWorker>) {
        self.broadcast(&bytes(&wire::encode(&JobRequest::Icons {
            style,
            icons: (**icons).clone(),
        })));
    }

    fn prefetch_glyphs(&self, style: u64, urls: Vec<String>) {
        // The page fetches each range once and hands it to every worker,
        // which lays out with its own metrics; their tiles that need one
        // meanwhile wait for it rather than fetch it again.
        let request = JobRequest::Glyphs {
            style,
            urls: urls.clone(),
        };
        self.broadcast(&bytes(&wire::encode(&request)));
        for url in urls {
            let (me, fetching) = (self.me.clone(), url.clone());
            let done = Box::new(move |fetched: Result<Vec<u8>, orb_host::FetchError>| {
                let Some(jobs) = me.upgrade() else {
                    return;
                };
                let bytes = fetched.map_err(|e| format!("glyphs {url}: {e}"));
                let range = JobRequest::GlyphRange { style, url, bytes };
                jobs.broadcast(&self::bytes(&wire::encode(&range)));
            });
            crate::regions::service().get(orb_host::ResourceKind::Glyphs, &fetching, done);
        }
    }

    fn raster(&self, request: GlyphRaster, done: RasterDone) {
        // A unit's owner worker rasterizes it.
        let owner = request.owner(self.workers);
        let kept = request.clone();
        self.send(
            |_| owner,
            None,
            |state, id| {
                state.rasters.insert(id, done);
                state.raster_requests.insert(id, kept);
            },
            |id| JobRequest::Raster { id, request },
        );
    }

    fn load(&self, request: TileRequest, done: TileDone) {
        let suspect = {
            let mut state = self.state.lock().expect("jobs poisoned");
            match state.lost_loads.get(&load_key(&request)) {
                Some(&n) if n >= QUARANTINE => {
                    drop(state);
                    return done(Ok(quarantine(String::new())));
                }
                Some(_) if state.suspect.is_some() => {
                    state.suspects.push_back((request, done));
                    return;
                }
                found => found.is_some(),
            }
        };
        let (source, tile) = (request.source, request.data_tile());
        self.send(
            |state| state.home(source, tile),
            Some(request.clone()),
            |state, id| {
                state.pending.insert(id, done);
                if suspect {
                    state.suspect = Some(id);
                }
            },
            |id| JobRequest::Load { id, request },
        );
    }

    fn cancel(&self, style: u64, source: u32, tile: TileIndex) {
        let key = (style, source, tile);
        let worker = {
            let state = self.state.lock().expect("jobs poisoned");
            state.sent_to.values().find_map(|(worker, request)| {
                let request = request.as_ref()?;
                (load_key(request) == key && request.nodes.is_none()).then_some(*worker)
            })
        };
        if let Some(worker) = worker {
            self.post(
                worker,
                &bytes(&wire::encode(&JobRequest::Cancel {
                    style,
                    source,
                    tile,
                })),
            );
        }
    }

    fn features(&self, request: orb_runtime::FeatureRequest, done: orb_runtime::FeaturesDone) {
        let (source, tile) = (request.tile.source, request.tile.data_tile());
        self.send(
            |state| state.home(source, tile),
            None,
            |state, id| drop(state.features.insert(id, done)),
            |id| JobRequest::Features { id, request },
        );
    }

    fn cluster(&self, request: orb_runtime::ClusterRequest, done: orb_runtime::QueryDone) {
        let (source, tile) = (request.tile.source, request.tile.data_tile());
        self.send(
            |state| state.home(source, tile),
            None,
            |state, id| drop(state.answers.insert(id, done)),
            |id| JobRequest::Cluster { id, request },
        );
    }
}

/// A job worker: a [`JobHost`] behind the page's messages.
#[wasm_bindgen]
#[derive(Debug)]
pub struct JobWorker {
    host: Arc<JobHost>,
}

#[wasm_bindgen]
impl JobWorker {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let host = JobHost::new(crate::regions::service(), Arc::new(orb_host::jobs::Inline));
        host.set_local_glyphs(Arc::new(crate::host::CanvasGlyphs));
        host.set_font_files(Arc::new(crate::host::ShaperFiles));
        if let Some(words) = crate::host::browser_words() {
            host.set_word_breaker(words);
        }
        Self { host }
    }

    /// How this worker loads `shaper.wasm` for `font-faces` files:
    /// `loader()` returns a promise of a `Shaper`.
    #[wasm_bindgen(js_name = setShaperLoader)]
    pub fn set_shaper_loader(&mut self, loader: js_sys::Function) {
        crate::host::ShaperFiles::set_loader(loader);
    }

    /// Traps, as a bug would: the page replaces the worker. To exercise
    /// that recovery.
    pub fn trap(&self) {
        panic!("a trap asked for");
    }

    /// Handles a request from the page: a [`JobRequest`] in
    /// [`orb_runtime::wire`]'s bytes. Returns a promise of the reply to
    /// post back, transferring its `buffer` and `image`, or nothing.
    pub fn receive(
        &mut self,
        message: &js_sys::Uint8Array,
    ) -> Result<Option<js_sys::Promise>, JsError> {
        let request: JobRequest = wire::decode(&message.to_vec()).map_err(|e| JsError::new(&e))?;
        let (sender, answer) = futures_channel::oneshot::channel::<JobAnswer>();
        let (id, dem_unpack) = match request {
            JobRequest::Style { style, update } => {
                self.host
                    .update_style(style, &update)
                    .map_err(|e| JsError::new(&e))?;
                return Ok(None);
            }
            JobRequest::Icons { style, icons } => {
                self.host.set_icons(style, Arc::new(icons));
                return Ok(None);
            }
            JobRequest::ArchiveHead { url, head } => {
                self.host.add_archive_head(&url, &head);
                return Ok(None);
            }
            JobRequest::Cancel {
                style,
                source,
                tile,
            } => {
                self.host.cancel(style, source, tile);
                return Ok(None);
            }
            JobRequest::Glyphs { style, urls } => {
                self.host.expect_glyphs(style, urls);
                return Ok(None);
            }
            JobRequest::GlyphRange { style, url, bytes } => {
                self.host.glyph_range(style, url, bytes);
                return Ok(None);
            }
            JobRequest::Cluster { id, request } => {
                self.host.cluster(
                    request,
                    Box::new(move |r| drop(sender.send(JobAnswer::Cluster(r)))),
                );
                (id, None)
            }
            JobRequest::Load { id, request } => {
                let dem_unpack = request.dem_unpack;
                self.host.load(
                    request,
                    Box::new(move |r| drop(sender.send(JobAnswer::Load(r)))),
                );
                (id, dem_unpack)
            }
            JobRequest::Features { id, request } => {
                self.host.features(
                    request,
                    Box::new(move |r| drop(sender.send(JobAnswer::Features(r)))),
                );
                (id, None)
            }
            JobRequest::Raster { id, request } => {
                self.host.raster(
                    request,
                    Box::new(move |r| drop(sender.send(JobAnswer::Raster(r)))),
                );
                (id, None)
            }
        };
        Ok(Some(wasm_bindgen_futures::future_to_promise(async move {
            let error = |message: String| reply(&JobReply::Error { id, message }, None, None);
            let Ok(answer) = answer.await else {
                return Ok(error("the job host dropped the request".into()));
            };
            Ok(match answer {
                JobAnswer::Load(Ok(Loaded::Tile(build))) => tile_reply(id, build),
                JobAnswer::Load(Ok(Loaded::Missing)) => {
                    reply(&JobReply::Missing { id }, None, None)
                }
                JobAnswer::Load(Ok(Loaded::Cancelled)) => {
                    reply(&JobReply::Cancelled { id }, None, None)
                }
                // A terrain DEM tile's texels, for the GPU and its
                // elevations for the CPU.
                JobAnswer::Load(Ok(Loaded::Image {
                    bytes,
                    kind: ImageKind::Dem,
                    ..
                })) if let Some(DemUnpack(unpack)) = dem_unpack => {
                    let decoded = crate::host::decode_image(&bytes)
                        .await
                        .map_err(|e| format!("{e:?}"))
                        .and_then(|image| {
                            let grid =
                                orb_tile::DemGrid::from_rgba(image.width, &image.pixels, unpack)?;
                            Ok((image, grid))
                        });
                    match decoded {
                        Ok((image, grid)) => reply(
                            &JobReply::Dem {
                                id,
                                width: image.width,
                                elevations: grid.elevations().to_vec(),
                            },
                            Some(js_sys::Uint8Array::from(image.pixels.as_slice()).buffer()),
                            None,
                        ),
                        Err(e) => error(format!("image decode: {e}")),
                    }
                }
                JobAnswer::Load(Ok(Loaded::Image {
                    bytes, warp_grid, ..
                })) => match crate::host::decode_bitmap(&bytes, true).await {
                    Ok(bitmap) => {
                        let warp_grid = warp_grid.map(Arc::unwrap_or_clone);
                        let message = JobReply::Image { id, warp_grid };
                        reply(&message, None, Some(bitmap.into()))
                    }
                    Err(e) => error(format!("image decode: {e:?}")),
                },
                JobAnswer::Features(Ok(features)) => {
                    reply(&JobReply::Features { id, features }, None, None)
                }
                JobAnswer::Raster(Ok(bitmaps)) => {
                    reply(&JobReply::Glyphs { id, bitmaps }, None, None)
                }
                JobAnswer::Cluster(Ok(json)) => reply(&JobReply::Answer { id, json }, None, None),
                JobAnswer::Load(Err(e))
                | JobAnswer::Features(Err(e))
                | JobAnswer::Raster(Err(e))
                | JobAnswer::Cluster(Err(e)) => error(e),
            })
        })))
    }
}

/// What the host answered a request with.
enum JobAnswer {
    Load(Result<Loaded, String>),
    Features(Result<Vec<orb_runtime::TileFeature>, String>),
    Raster(Result<Vec<orb_runtime::GlyphBitmap>, String>),
    Cluster(Result<String, String>),
}

/// A reply for the page: the message, and the buffer and bitmap it
/// transfers.
fn reply(message: &JobReply, buffer: Option<ArrayBuffer>, image: Option<JsValue>) -> JsValue {
    object(&[
        ("message", bytes(&wire::encode(message))),
        ("buffer", buffer.map_or(JsValue::UNDEFINED, Into::into)),
        ("image", image.unwrap_or(JsValue::UNDEFINED)),
    ])
}

/// A packed tile's reply.
fn tile_reply(id: u32, build: TileBuild) -> JsValue {
    let packed = pack(build);
    let buffer = js_sys::Uint8Array::from(packed.as_slice()).buffer();
    reply(&JobReply::Tile { id }, Some(buffer), None)
}

/// A message's bytes as a JS array.
fn bytes(bytes: &[u8]) -> JsValue {
    js_sys::Uint8Array::from(bytes).into()
}

fn field(object: &JsValue, key: &str) -> JsValue {
    Reflect::get(object, &JsValue::from_str(key)).unwrap_or(JsValue::UNDEFINED)
}

fn object(fields: &[(&str, JsValue)]) -> JsValue {
    let object = Object::new();
    for (key, value) in fields {
        let _ = Reflect::set(&object, &JsValue::from_str(key), value);
    }
    object.into()
}

impl Default for JobWorker {
    fn default() -> Self {
        Self::new()
    }
}
