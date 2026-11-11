//! One job host for every platform: what a map's tile,
//! query and glyph jobs need beyond a worker's CPU work. It keeps the style
//! revisions requests name, shares fetches between the requests that need
//! them (a source's data document, glyph ranges, word data, font files), and
//! answers each request once. Tile fetches take turns, [`FETCHES`] at once:
//! a load its map gives up ([`cancel`](JobHost::cancel)) before its turn
//! never starts, and one fetching is aborted. Natively
//! [`PoolJobs`](crate::PoolJobs) runs one in the map's process; on the web
//! every job worker runs one behind the page's messages. Image tiles come
//! out as their encoded bytes: each platform decodes them its own way.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use orb_host::jobs::Executor;
use orb_host::{Abort, Fetch, FetchError, FontFiles, ResourceKind};

use crate::glyphs::{GlyphRaster, LocalGlyphs, RasterDone};
use crate::loader::TileLoader;
use crate::style::ImageKind;
use crate::worker::{Built, StyleUpdate, TileRequest, TileWorker};
use crate::{FeatureRequest, FeaturesDone, SpriteIcons, TileBuild};
use orb_crs::TileIndex;

/// What a tile load came to.
#[derive(Debug)]
pub enum Loaded {
    Tile(TileBuild),
    /// An image source's tile, still encoded, and the warp grid its
    /// request asked for ([`TileRequest::warp`]).
    Image {
        bytes: Vec<u8>,
        kind: ImageKind,
        warp_grid: Option<Arc<orb_crs::WarpGrid>>,
    },
    /// The source doesn't have the tile.
    Missing,
    /// Its map gave the load up before it arrived ([`JobHost::cancel`]).
    Cancelled,
}

/// Tile fetches a job host runs at once. Natively they feed the fetch
/// threads; in a browser they keep the network queue short, so that loads
/// given up wait here, where they can be dropped, rather than there.
pub(crate) const FETCHES: usize = 16;

/// A load its map can give up: its style, source and tile.
type LoadKey = (u64, u32, TileIndex);

/// Tile fetches running, and those waiting for a turn in request order,
/// with the aborts that give each up.
#[derive(Default)]
struct Fetches {
    running: Vec<(Option<LoadKey>, Abort)>,
    waiting: VecDeque<(Option<LoadKey>, Abort, Waiting)>,
    /// A thread starts waiting fetches: one whose fetch completes as it
    /// starts hands its turn on without starting more itself.
    starting: bool,
}

/// Starts a fetch with its turn, or, given none, answers it cancelled.
type Waiting = Box<dyn FnOnce(Option<Turn>) + Send>;

/// A running fetch's turn, and the abort its fetch takes: dropped, the
/// next waiting fetch starts.
struct Turn {
    fetches: Arc<Mutex<Fetches>>,
    abort: Abort,
}

impl Drop for Turn {
    fn drop(&mut self) {
        let mut fetches = self.fetches.lock().expect("fetches poisoned");
        let running = &mut fetches.running;
        if let Some(i) = running.iter().position(|(_, a)| a.same(&self.abort)) {
            running.swap_remove(i);
        }
        drop(fetches);
        start_waiting(&self.fetches);
    }
}

/// Starts waiting fetches while turns are free, unless another call
/// already does.
fn start_waiting(fetches: &Arc<Mutex<Fetches>>) {
    let mut state = fetches.lock().expect("fetches poisoned");
    if state.starting {
        return;
    }
    state.starting = true;
    while state.running.len() < FETCHES {
        let Some((key, abort, fetch)) = state.waiting.pop_front() else {
            break;
        };
        state.running.push((key, abort.clone()));
        drop(state);
        fetch(Some(Turn {
            fetches: Arc::clone(fetches),
            abort,
        }));
        state = fetches.lock().expect("fetches poisoned");
    }
    state.starting = false;
}

/// Called once with a load's outcome, on any thread.
pub(crate) type LoadedDone = Box<dyn FnOnce(Result<Loaded, String>) + Send>;

/// Loads of a source's data document in flight, by style, source and URL:
/// the requests waiting on each, with the worker of the revision each was
/// asked for in, which a later revision with other data doesn't share.
type DataLoads = HashMap<(u64, u32, String), Vec<DataWaiter>>;
type DataWaiter = (Arc<TileWorker>, TileRequest, LoadedDone);

/// Glyph range, word data and font file fetches in flight, by style and
/// URL: what to run when each arrives.
type GlyphWaiters = HashMap<(u64, String), Vec<Box<dyn FnOnce() + Send>>>;

/// A job host: see the module.
pub struct JobHost {
    fetch: Arc<dyn Fetch>,
    loader: Arc<TileLoader>,
    fetches: Arc<Mutex<Fetches>>,
    executor: Arc<dyn Executor>,
    /// The latest revision of the current style and of the one before it,
    /// whose requests may still be in flight.
    styles: Mutex<Vec<(u64, Arc<TileWorker>)>>,
    /// The newest style's id: a load for an older one whose bytes arrive
    /// after it was replaced isn't built, since the map drops its result.
    newest: Arc<AtomicU64>,
    data_loads: Arc<Mutex<DataLoads>>,
    glyphs: Arc<GlyphLoads>,
    /// The host's local fonts, if any.
    local: Mutex<Option<Arc<dyn LocalGlyphs>>>,
    files: Mutex<Option<Arc<dyn FontFiles>>>,
    /// The host's word breaker, if it has one.
    words: Mutex<Option<Arc<orb_text::WordBreaker>>>,
}

impl std::fmt::Debug for JobHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobHost").finish_non_exhaustive()
    }
}

/// Fetches what tiles wait on, once per URL.
struct GlyphLoads {
    fetch: Arc<dyn Fetch>,
    executor: Arc<dyn Executor>,
    waiting: Mutex<GlyphWaiters>,
    /// Glyph ranges fetched lately, whole: the atlas's raster jobs draw
    /// their bitmaps from these rather than fetch them again (workers keep
    /// metrics only).
    recent: Mutex<RecentRanges>,
}

/// The glyph ranges fetched last, up to [`RecentRanges::BUDGET`] bytes.
#[derive(Default)]
struct RecentRanges {
    ranges: std::collections::VecDeque<(String, Arc<Vec<u8>>)>,
    bytes: usize,
}

impl RecentRanges {
    /// A few fonts' worth of ranges: a Latin range is 60–80 KB.
    const BUDGET: usize = 4 << 20;

    fn get(&self, url: &str) -> Option<Arc<Vec<u8>>> {
        let found = self.ranges.iter().find(|(u, _)| u == url);
        found.map(|(_, bytes)| Arc::clone(bytes))
    }

    fn insert(&mut self, url: &str, bytes: Arc<Vec<u8>>) {
        if bytes.len() > Self::BUDGET || self.get(url).is_some() {
            return;
        }
        self.bytes += bytes.len();
        self.ranges.push_back((url.to_string(), bytes));
        while self.bytes > Self::BUDGET {
            let (_, oldest) = self.ranges.pop_front().expect("over budget");
            self.bytes -= oldest.len();
        }
    }
}

impl GlyphLoads {
    /// Runs `ready` once `worker` has `url`, fetching it unless that's
    /// already under way.
    fn wait(
        self: &Arc<Self>,
        worker: &Arc<TileWorker>,
        style: u64,
        url: String,
        ready: Box<dyn FnOnce() + Send>,
    ) {
        let key = (style, url);
        let mut waiting = self.waiting.lock().expect("glyph loads poisoned");
        if let Some(waiters) = waiting.get_mut(&key) {
            waiters.push(ready);
            return;
        }
        // A fetch that finished after the build asked would be fetched again.
        if worker.data.glyphs.has(&key.1) {
            drop(waiting);
            ready();
            return;
        }
        waiting.insert(key.clone(), vec![ready]);
        drop(waiting);
        let (this, worker) = (Arc::clone(self), Arc::clone(worker));
        self.fetch.get(
            ResourceKind::Glyphs,
            &key.1.clone(),
            &Abort::NEVER,
            Box::new(move |fetched| {
                this.arrived(worker, key, fetched.map_err(|e| e.to_string()));
            }),
        );
    }

    /// Marks style `style`'s `url` as coming from elsewhere ([`arrived`](Self::arrived)):
    /// layouts that need it meanwhile wait for it rather than fetch it.
    fn expect(&self, worker: &TileWorker, style: u64, url: String) {
        if worker.data.glyphs.has(&url) {
            return;
        }
        let mut waiting = self.waiting.lock().expect("glyph loads poisoned");
        waiting.entry((style, url)).or_default();
    }

    /// `key`'s resource is in (or failed): `worker` takes it, and the
    /// layouts waiting for it go on.
    fn arrived(
        self: &Arc<Self>,
        worker: Arc<TileWorker>,
        key: (u64, String),
        fetched: Result<Vec<u8>, String>,
    ) {
        let url = key.1.clone();
        let files = worker
            .data
            .glyphs
            .files()
            .filter(|_| worker.declares_font(&url));
        // Shared with the recent ranges, not copied into them.
        let fetched = fetched.map(Arc::new);
        if let Ok(bytes) = &fetched
            && worker.is_glyph_range(&url)
        {
            let mut recent = self.recent.lock().expect("ranges poisoned");
            recent.insert(&url, Arc::clone(bytes));
        }
        let this = Arc::clone(self);
        let finish = move |prepared: Result<(), String>| {
            let fetched = fetched.as_ref().map(|b| b.as_slice());
            let bytes = prepared.and(fetched.map_err(Clone::clone));
            worker.add_resource(&url, bytes);
            let waiters = this
                .waiting
                .lock()
                .expect("glyph loads poisoned")
                .remove(&key)
                .unwrap_or_default();
            for ready in waiters {
                ready();
            }
        };
        // A font file opens once the engine is in.
        match files {
            Some(files) => files.prepare(Box::new(finish)),
            None => finish(Ok(())),
        }
    }

    /// Builds a tile on this thread, fetching the glyph ranges and word
    /// data its labels need first and building again once they are in.
    fn complete(
        self: &Arc<Self>,
        worker: Arc<TileWorker>,
        bytes: Arc<Vec<u8>>,
        request: TileRequest,
        done: LoadedDone,
    ) {
        // Fetched and decoded while the sprites loaded, a tile with symbol
        // layers lays out once their icons are in.
        if worker.awaits_icons(request.icons) {
            // Decoded now, the tile's frame waits in the worker's cache.
            if !bytes.is_empty() {
                let _ = worker.frame(&bytes, request.encoding, &request);
            }
            let (this, waiting) = (Arc::clone(self), Arc::clone(&worker));
            let executor = Arc::clone(&self.executor);
            let epoch = request.icons;
            let ready = Box::new(move || {
                executor.spawn(Box::new(move || {
                    this.complete(waiting, bytes, request, done);
                }));
            });
            worker.with_icons(epoch, ready);
            return;
        }
        let urls = match worker.build(&bytes, request.encoding, &request) {
            Ok(Built::Tile(build)) => return done(Ok(Loaded::Tile(build))),
            Err(e) => return done(Err(e)),
            Ok(Built::Fetch(urls)) => urls,
        };
        let style = request.style;
        let remaining = Arc::new(std::sync::atomic::AtomicUsize::new(urls.len()));
        let retry = Arc::new(Mutex::new(Some((bytes, request, done))));
        for url in urls {
            let (this, worker2) = (Arc::clone(self), Arc::clone(&worker));
            let (remaining, retry) = (Arc::clone(&remaining), Arc::clone(&retry));
            let ready = Box::new(move || {
                if remaining.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) != 1 {
                    return;
                }
                let (bytes, request, done) = retry
                    .lock()
                    .expect("retry poisoned")
                    .take()
                    .expect("retried once");
                let executor = Arc::clone(&this.executor);
                executor.spawn(Box::new(move || {
                    this.complete(worker2, bytes, request, done);
                }));
            });
            self.wait(&worker, style, url, ready);
        }
    }
}

/// A fetch's outcome as a load's: a tile the source doesn't have (HTTP
/// 404) is [`Loaded::Missing`], one given up [`Loaded::Cancelled`]; other
/// fetch errors are errors.
fn missing(error: FetchError) -> Result<Loaded, String> {
    match error {
        FetchError::NotFound => Ok(Loaded::Missing),
        FetchError::Aborted => Ok(Loaded::Cancelled),
        e => Err(e.to_string()),
    }
}

impl JobHost {
    pub fn new(fetch: Arc<dyn Fetch>, executor: Arc<dyn Executor>) -> Arc<Self> {
        Arc::new(Self {
            loader: TileLoader::new(Arc::clone(&fetch)),
            fetches: Arc::default(),
            glyphs: Arc::new(GlyphLoads {
                fetch: Arc::clone(&fetch),
                executor: Arc::clone(&executor),
                waiting: Mutex::default(),
                recent: Mutex::default(),
            }),
            fetch,
            executor,
            styles: Mutex::new(Vec::new()),
            newest: Arc::default(),
            data_loads: Arc::default(),
            local: Mutex::new(None),
            files: Mutex::new(None),
            words: Mutex::new(None),
        })
    }

    /// Draws local glyphs with the host's fonts: system
    /// fonts natively, the canvas on the web.
    pub fn set_local_glyphs(&self, local: Arc<dyn LocalGlyphs>) {
        *self.local.lock().expect("local glyphs poisoned") = Some(local);
    }

    /// Draws declared font files with the host's engine: `orb-text`
    /// natively, `shaper.wasm` in the web's job workers.
    pub fn set_font_files(&self, files: Arc<dyn FontFiles>) {
        *self.files.lock().expect("font files poisoned") = Some(files);
    }

    /// Breaks words in scripts written without spaces with the host's
    /// segmenter (the browser's `Intl.Segmenter`, as GL JS does), for the
    /// styles set from now on, rather than with compiled dictionaries.
    pub fn set_word_breaker(&self, breaker: orb_text::WordBreaker) {
        *self.words.lock().expect("words poisoned") = Some(Arc::new(breaker));
    }

    /// Makes a revision of style `id` available to later requests.
    /// Requests for the previous style may still be in flight; requests for
    /// an older revision of this one use the newest.
    pub fn set_style(&self, id: u64, worker: Arc<TileWorker>) {
        if let Some(local) = &*self.local.lock().expect("local glyphs poisoned") {
            worker.set_local_glyphs(Arc::clone(local));
        }
        if let Some(files) = &*self.files.lock().expect("font files poisoned") {
            worker.set_font_files(Arc::clone(files));
        }
        if let Some(words) = &*self.words.lock().expect("words poisoned") {
            worker.set_word_breaker(Arc::clone(words));
        }
        let mut styles = self.styles.lock().expect("styles poisoned");
        styles.retain(|(i, _)| *i + 1 >= id && *i != id);
        styles.push((id, worker));
        self.newest.fetch_max(id, Ordering::Relaxed);
    }

    /// Applies a revision of style `id` sent from another worker.
    pub fn update_style(&self, id: u64, update: &StyleUpdate) -> Result<(), String> {
        let previous = self.worker(id);
        let worker = TileWorker::update(previous.as_deref(), update)?;
        self.set_style(id, Arc::new(worker));
        Ok(())
    }

    /// Sets the sprite metadata style `id` lays icons out with. Revisions
    /// share their worker data, so this reaches every one.
    pub fn set_icons(&self, id: u64, icons: Arc<SpriteIcons>) {
        if let Some(worker) = self.worker(id) {
            worker.set_icons(icons);
        }
    }

    fn worker(&self, style: u64) -> Option<Arc<TileWorker>> {
        self.styles
            .lock()
            .expect("styles poisoned")
            .iter()
            .find(|(id, _)| *id == style)
            .map(|(_, w)| Arc::clone(w))
    }

    /// Starts fetching glyph ranges for style `id`, as tiles that need
    /// them would: tiles that ask meanwhile wait on these fetches.
    pub fn prefetch_glyphs(&self, id: u64, urls: Vec<String>) {
        let Some(worker) = self.worker(id) else {
            return;
        };
        for url in urls {
            self.glyphs.wait(&worker, id, url, Box::new(|| {}));
        }
    }

    /// Lets the glyph ranges that failed be fetched again by the tiles that
    /// need them ([`Jobs::retry_glyphs`](crate::Jobs::retry_glyphs)).
    pub fn retry_glyphs(&self) {
        for (_, worker) in self.styles.lock().expect("styles poisoned").iter() {
            worker.data.glyphs.retry_failed();
        }
    }

    /// Glyph ranges style `id`'s tiles are about to need that another
    /// context fetches and hands in with [`glyph_range`](Self::glyph_range):
    /// tiles that need them meanwhile wait for them rather than fetch them.
    pub fn expect_glyphs(&self, id: u64, urls: Vec<String>) {
        let Some(worker) = self.worker(id) else {
            return;
        };
        for url in urls {
            self.glyphs.expect(&worker, id, url);
        }
    }

    /// A glyph range [`expect_glyphs`](Self::expect_glyphs) announced,
    /// fetched (or failed) elsewhere.
    pub fn glyph_range(&self, id: u64, url: String, fetched: Result<Vec<u8>, String>) {
        let Some(worker) = self.worker(id) else {
            return;
        };
        self.glyphs.arrived(worker, (id, url), fetched);
    }

    /// PMTiles archive `url` begins with `head`: loads of its tiles start
    /// from these bytes.
    pub fn add_archive_head(&self, url: &str, head: &[u8]) {
        self.loader.add_head(url, head);
    }

    /// Loads and lays out a tile.
    pub fn load(&self, request: TileRequest, done: LoadedDone) {
        let Some(worker) = self.worker(request.style) else {
            return done(Err(format!("style {} was replaced", request.style)));
        };
        let glyphs = Arc::clone(&self.glyphs);
        if worker.has_data(&request) {
            self.executor.spawn(Box::new(move || {
                glyphs.complete(worker, Arc::default(), request, done);
            }));
            return;
        }
        if let Some(url) = worker.data_url(request.source).map(str::to_string) {
            self.load_data(worker, url, request, done);
            return;
        }
        let (executor, loader) = (Arc::clone(&self.executor), Arc::clone(&self.loader));
        // A whole tile's load, which its map may give up; not a relayout.
        let key = (request.nodes.is_none() && !request.rows_only).then_some((
            request.style,
            request.source,
            request.tile,
        ));
        if let Some(kind) = worker.image_kind(request.source) {
            let (source, tile, warp) = (request.source, request.tile, request.warp);
            let resource = worker.resource_kind(source);
            let fetch: Waiting = Box::new(move |turn| {
                let Some(turn) = turn else {
                    return done(Ok(Loaded::Cancelled));
                };
                let abort = turn.abort.clone();
                loader.load(
                    resource,
                    request.location,
                    request.tile,
                    request.encoding,
                    &abort,
                    Box::new(move |fetched| {
                        drop(turn);
                        match fetched {
                            // An archive's left-out tile draws nothing.
                            Ok((bytes, _)) if bytes.is_empty() => {
                                done(Ok(Loaded::Tile(TileBuild::default())))
                            }
                            // Its mesh on the surface's plane is the costly
                            // part of drawing it there: the grid is made here.
                            Ok((bytes, _)) if warp => executor.spawn(Box::new(move || {
                                let warp_grid = worker.warp_grid(source, tile).map(Arc::new);
                                done(Ok(Loaded::Image {
                                    bytes,
                                    kind,
                                    warp_grid,
                                }))
                            })),
                            Ok((bytes, _)) => done(Ok(Loaded::Image {
                                bytes,
                                kind,
                                warp_grid: None,
                            })),
                            Err(e) => done(missing(e)),
                        }
                    }),
                );
            });
            return self.take_turn(key, fetch);
        }
        let newest = Arc::clone(&self.newest);
        let fetch: Waiting = Box::new(move |turn| {
            let Some(turn) = turn else {
                return done(Ok(Loaded::Cancelled));
            };
            let data_tile = request.data_tile();
            let abort = turn.abort.clone();
            loader.load(
                ResourceKind::Tile,
                request.location.clone(),
                data_tile,
                request.encoding,
                &abort,
                Box::new(move |fetched| {
                    drop(turn);
                    executor.spawn(Box::new(move || match fetched {
                        Err(FetchError::Aborted) => done(Ok(Loaded::Cancelled)),
                        _ if newest.load(Ordering::Relaxed) > request.style => {
                            done(Err(format!("style {} was replaced", request.style)))
                        }
                        Ok((bytes, encoding)) => {
                            let request = TileRequest {
                                encoding,
                                ..request
                            };
                            glyphs.complete(worker, Arc::new(bytes), request, done)
                        }
                        Err(e) => done(missing(e)),
                    }));
                }),
            );
        });
        self.take_turn(key, fetch);
    }

    /// Runs `fetch` in its turn: at once while fewer than [`FETCHES`] run,
    /// or once one ends.
    fn take_turn(&self, key: Option<LoadKey>, fetch: Waiting) {
        // Each turn its own, which names its place among the running.
        let abort = Abort::new();
        let mut fetches = self.fetches.lock().expect("fetches poisoned");
        fetches.waiting.push_back((key, abort, fetch));
        drop(fetches);
        start_waiting(&self.fetches);
    }

    /// Drops what style `style`'s worker keeps of `tile` of its source
    /// `source` (its decoded frames), so that its next load fetches it
    /// again ([`Jobs::forget`](crate::Jobs::forget)).
    pub fn forget(&self, style: u64, source: u32, tile: TileIndex) {
        let styles = self.styles.lock().expect("styles poisoned");
        if let Some((_, worker)) = styles.iter().find(|(id, _)| *id == style) {
            worker.forget(source, tile);
        }
    }

    /// Gives up the load of `tile` of style `style`'s source `source`: if
    /// its fetch hasn't started, it never does; if it has, it is aborted
    /// (a fetch other loads share goes on for them). Either way the load
    /// answers [`Loaded::Cancelled`], unless its bytes already came.
    pub fn cancel(&self, style: u64, source: u32, tile: TileIndex) {
        let key = Some((style, source, tile));
        let (given_up, running): (Vec<Waiting>, Vec<Abort>) = {
            let mut fetches = self.fetches.lock().expect("fetches poisoned");
            let (given_up, kept) = std::mem::take(&mut fetches.waiting)
                .into_iter()
                .partition(|(k, _, _)| *k == key);
            fetches.waiting = kept;
            let running = fetches.running.iter().filter(|(k, _)| *k == key);
            (
                given_up.into_iter().map(|(_, _, fetch)| fetch).collect(),
                running.map(|(_, abort)| abort.clone()).collect(),
            )
        };
        for fetch in given_up {
            fetch(None);
        }
        for abort in running {
            abort.abort();
        }
    }

    /// Loads a tile of a source whose data is one document (GeoJSON by
    /// URL): every tile of the source waits on its one fetch, which indexes
    /// the document for them all.
    fn load_data(
        &self,
        worker: Arc<TileWorker>,
        url: String,
        request: TileRequest,
        done: LoadedDone,
    ) {
        let key = (request.style, request.source, url.clone());
        let glyphs = Arc::clone(&self.glyphs);
        let mut loads = self.data_loads.lock().expect("data loads poisoned");
        if let Some(waiting) = loads.get_mut(&key) {
            waiting.push((worker, request, done));
            return;
        }
        // Recheck while holding the registry: a previous load may have
        // indexed the document between the checks.
        if worker.has_data(&request) {
            drop(loads);
            self.executor.spawn(Box::new(move || {
                glyphs.complete(worker, Arc::default(), request, done);
            }));
            return;
        }
        loads.insert(key.clone(), vec![(worker, request, done)]);
        drop(loads);
        let loads = Arc::clone(&self.data_loads);
        let executor = Arc::clone(&self.executor);
        self.fetch.get(
            ResourceKind::Source,
            &url,
            &Abort::NEVER,
            Box::new(move |fetched| {
                executor.spawn(Box::new(move || {
                    // Index the document before removing the in-flight
                    // entry, without holding the registry during CPU work.
                    let first = loads
                        .lock()
                        .expect("data loads poisoned")
                        .get_mut(&key)
                        .expect("registered fetch")
                        .pop()
                        .expect("first request");
                    let fetched = fetched.map(Arc::new).map_err(|e| e.to_string());
                    // Every waiter gets the bytes: a revision that set the
                    // same URL again has data of its own to index, and one
                    // already indexed ignores them.
                    let build = |(worker, request, done): DataWaiter| match &fetched {
                        Ok(bytes) => glyphs.complete(worker, Arc::clone(bytes), request, done),
                        Err(e) => done(Err(e.clone())),
                    };
                    build(first);
                    let waiting = loads
                        .lock()
                        .expect("data loads poisoned")
                        .remove(&key)
                        .expect("registered fetch");
                    waiting.into_iter().for_each(build);
                }));
            }),
        );
    }

    /// Reads features of a tile for a query, from its cached frame or the
    /// tile fetched again.
    pub fn features(&self, request: FeatureRequest, done: FeaturesDone) {
        let Some(worker) = self.worker(request.tile.style) else {
            return done(Err(format!("style {} was replaced", request.tile.style)));
        };
        let executor = Arc::clone(&self.executor);
        if worker.has_data(&request.tile) {
            executor.spawn(Box::new(move || {
                done(worker.features(&[], request.tile.encoding, &request));
            }));
            return;
        }
        let tile = &request.tile;
        self.loader.load(
            worker.resource_kind(tile.source),
            tile.location.clone(),
            tile.data_tile(),
            tile.encoding,
            &Abort::NEVER,
            Box::new(move |fetched| {
                executor.spawn(Box::new(move || match fetched {
                    Ok((bytes, encoding)) => done(worker.features(&bytes, encoding, &request)),
                    Err(FetchError::NotFound) => done(Ok(Vec::new())),
                    Err(e) => done(Err(e.to_string())),
                }));
            }),
        );
    }

    /// Answers a cluster query from the source's index, fetching its
    /// document when this host has none yet.
    pub fn cluster(&self, request: crate::ClusterRequest, done: crate::QueryDone) {
        let Some(worker) = self.worker(request.tile.style) else {
            return done(Err(format!("style {} was replaced", request.tile.style)));
        };
        let executor = Arc::clone(&self.executor);
        if worker.has_data(&request.tile) {
            executor.spawn(Box::new(move || done(worker.cluster(&[], &request))));
            return;
        }
        let tile = &request.tile;
        self.loader.load(
            worker.resource_kind(tile.source),
            tile.location.clone(),
            tile.data_tile(),
            tile.encoding,
            &Abort::NEVER,
            Box::new(move |fetched| {
                executor.spawn(Box::new(move || match fetched {
                    Ok((bytes, _)) => done(worker.cluster(&bytes, &request)),
                    Err(e) => done(Err(e.to_string())),
                }));
            }),
        );
    }

    /// Rasterizes glyphs for the atlas.
    pub fn raster(&self, request: GlyphRaster, done: RasterDone) {
        let executor = Arc::clone(&self.executor);
        match &request {
            GlyphRaster::Pbf { url, .. } => {
                let recent = self.glyphs.recent.lock().expect("ranges poisoned").get(url);
                if let Some(bytes) = recent {
                    executor.spawn(Box::new(move || done(request.run(&bytes))));
                    return;
                }
                let (url, glyphs) = (url.clone(), Arc::clone(&self.glyphs));
                self.fetch.get(
                    ResourceKind::Glyphs,
                    &url.clone(),
                    &Abort::NEVER,
                    Box::new(move |fetched| match fetched {
                        Ok(bytes) => {
                            let bytes = Arc::new(bytes);
                            let mut recent = glyphs.recent.lock().expect("ranges poisoned");
                            recent.insert(&url, Arc::clone(&bytes));
                            drop(recent);
                            executor.spawn(Box::new(move || done(request.run(&bytes))));
                        }
                        Err(e) => done(Err(format!("glyphs {url}: {e}"))),
                    }),
                );
            }
            GlyphRaster::Local { .. } => {
                match self.local.lock().expect("local glyphs poisoned").clone() {
                    Some(local) => {
                        executor.spawn(Box::new(move || done(request.draw(local.as_ref()))));
                    }
                    None => done(Err("this host draws no local glyphs".into())),
                }
            }
            // The file opened when a tile resolved its glyphs, here or in
            // another worker, which then fetches it (the HTTP cache,
            // usually) once the engine is in.
            GlyphRaster::File { url, .. } => {
                let Some(files) = self.files.lock().expect("font files poisoned").clone() else {
                    return done(Err("this host draws no font files".into()));
                };
                let (url, fetch) = (url.clone(), Arc::clone(&self.fetch));
                files.clone().prepare(Box::new(move |ready| {
                    if let Err(e) = ready {
                        return done(Err(e));
                    }
                    if files.is_open(&url) {
                        return done(request.draw_file(files.as_ref(), None));
                    }
                    fetch.get(
                        ResourceKind::Glyphs,
                        &url.clone(),
                        &Abort::NEVER,
                        Box::new(move |fetched| match fetched {
                            Ok(bytes) => done(request.draw_file(files.as_ref(), Some(&bytes))),
                            Err(e) => done(Err(format!("font face {url}: {e}"))),
                        }),
                    );
                }));
            }
        }
    }
}
