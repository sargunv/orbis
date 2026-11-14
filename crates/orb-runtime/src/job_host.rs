//! One job host for every platform: what a map's tile,
//! query and glyph jobs need beyond a worker's CPU work. It keeps the style
//! revisions requests name, shares fetches between the requests that need
//! them (a source's data document, glyph ranges, word data, font files), and
//! answers each request once. Tile fetches take turns ([`Turns`]), by
//! their requests' rank, which the map sets again each frame
//! ([`reorder`](JobHost::reorder)): a load its map gives up
//! ([`cancel`](JobHost::cancel)) before its turn never starts, and one
//! fetching is aborted. Natively [`PoolJobs`](crate::PoolJobs) runs one
//! in the map's process, [`orb_host::tile_fetches`] at once; on the web every job worker
//! runs one behind the page's messages, fetching whatever the page sends,
//! since the page's turns order fetches across the workers. Image tiles
//! come out as their encoded bytes: each platform decodes them its own
//! way.

use rustc_hash::FxHashMap as HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use orb_host::jobs::Executor;
use orb_host::{Abort, Fetch, FetchError, FontFiles, ResourceKind};

use crate::glyphs::{GlyphRaster, LocalGlyphs, RasterDone};
use crate::loader::TileLoader;
use crate::modules::{Load, Modules};
use crate::style::ImageKind;
use crate::turns::{FetchCounts, LoadKey, Start, Turns};
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

/// Called once a load's fetch has ended, or at once for a load that
/// fetches nothing: a browser's job worker tells the page, whose turns
/// ([`Turns`]) its fetch held.
pub type Fetched = Box<dyn FnOnce() + Send>;

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
    turns: Turns,
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
    wants: Arc<Wants>,
    /// The bidirectional algorithm's module, where the host loads it apart
    /// ([`JobHost::set_module_loader`]); `None` where it always has it.
    bidi: Mutex<Option<Arc<BidiLoad>>>,
}

/// What a layout waiting for the bidirectional algorithm runs once it
/// loaded, or with why it didn't.
type BidiWaiter = Box<dyn FnOnce(Result<(), String>) + Send>;

/// The bidirectional algorithm's module, loaded into this context the
/// first time a tile's labels read right to left ([`Modules::BIDI`]).
struct BidiLoad {
    load: Load,
    /// Whether it's in: every style's worker reads this.
    held: Arc<AtomicBool>,
    /// The layouts waiting while it loads; `None` while it isn't loading.
    waiting: Mutex<Option<Vec<BidiWaiter>>>,
}

impl BidiLoad {
    /// Runs `ready` once the module is in, loading it unless that's under
    /// way, or with why it didn't load.
    fn wait(self: &Arc<Self>, ready: BidiWaiter) {
        if self.held.load(Ordering::Acquire) {
            return ready(Ok(()));
        }
        let mut waiting = self.waiting.lock().expect("bidi waiters poisoned");
        if let Some(waiters) = waiting.as_mut() {
            waiters.push(ready);
            return;
        }
        *waiting = Some(vec![ready]);
        drop(waiting);
        let this = Arc::clone(self);
        (self.load)(
            Modules::BIDI,
            Box::new(move |loaded| {
                if loaded.is_ok() {
                    this.held.store(true, Ordering::Release);
                }
                let waiters = this.waiting.lock().expect("bidi waiters poisoned").take();
                for ready in waiters.unwrap_or_default() {
                    ready(loaded.clone());
                }
            }),
        );
    }
}

/// The whole-tile loads under way, each with whether its map still wants
/// it: one it gave up ([`JobHost::cancel`]) after its fetch is neither
/// decoded nor laid out, at whatever stage it waits, and answers
/// [`Loaded::Cancelled`].
#[derive(Default)]
struct Wants(Mutex<HashMap<LoadKey, bool>>);

impl Wants {
    /// Counts `request`'s load as under way, wanted, until `done` answers.
    fn track(self: &Arc<Self>, request: &TileRequest, done: LoadedDone) -> LoadedDone {
        let Some(key) = load_key(request) else {
            return done;
        };
        self.0.lock().expect("wants poisoned").insert(key, true);
        let wants = Arc::clone(self);
        Box::new(move |loaded| {
            wants.0.lock().expect("wants poisoned").remove(&key);
            done(loaded);
        })
    }

    /// Whether `request`'s map still wants it: a relayout always.
    fn wanted(&self, request: &TileRequest) -> bool {
        let wants = self.0.lock().expect("wants poisoned");
        load_key(request).is_none_or(|key| wants.get(&key).copied().unwrap_or(true))
    }

    fn give_up(&self, key: LoadKey) {
        if let Some(wanted) = self.0.lock().expect("wants poisoned").get_mut(&key) {
            *wanted = false;
        }
    }
}

/// A whole tile's load, which its map may give up; not a relayout.
fn load_key(request: &TileRequest) -> Option<LoadKey> {
    request
        .loads_whole()
        .then_some((request.style, request.source, request.tile))
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
        // Given up while it was fetched, or waited: nothing more is done.
        if !self.wants.wanted(&request) {
            return done(Ok(Loaded::Cancelled));
        }
        // Fetched and decoded while the sprites loaded, a tile with symbol
        // layers lays out once their icons are in.
        if worker.awaits_icons(request.icons) {
            // Decoded now, the tile's frame waits in the worker's cache.
            if !bytes.is_empty() {
                let _ = worker.frame(&bytes, request.encoding, &request);
            }
            let (this, waiting) = (Arc::clone(self), Arc::clone(&worker));
            let executor = Arc::clone(&self.executor);
            let (epoch, rank) = (request.icons, request.rank);
            let ready = Box::new(move || {
                executor.spawn_ranked(
                    rank,
                    Box::new(move || this.complete(waiting, bytes, request, done)),
                );
            });
            worker.with_icons(epoch, ready);
            return;
        }
        let (urls, bidi) = match worker.build(&bytes, request.encoding, &request) {
            Ok(Built::Tile(build)) => return done(Ok(Loaded::Tile(build))),
            Err(e) => return done(Err(e)),
            Ok(Built::Fetch(urls)) => (urls, false),
            Ok(Built::Bidi) => (Vec::new(), true),
        };
        let style = request.style;
        // Labels never hold up a whole tile's other layers: those go now,
        // and its map lays the labels out again once their glyphs are in,
        // which these fetches bring, and the bidirectional algorithm.
        if load_key(&request).is_some()
            && !request.neighbour
            && let Some((others, labels)) = worker.labels_apart(request.source)
        {
            let geometry = TileRequest {
                nodes: Some(others),
                ..request.clone()
            };
            if let Ok(Built::Tile(mut build)) = worker.build(&bytes, request.encoding, &geometry) {
                for url in urls {
                    self.wait(&worker, style, url, Box::new(|| {}));
                }
                if bidi {
                    self.wait_bidi(Box::new(|_| {}));
                }
                build.labels_later = labels;
                return done(Ok(Loaded::Tile(build)));
            }
        }
        let remaining = std::sync::atomic::AtomicUsize::new(urls.len() + usize::from(bidi));
        let remaining = Arc::new(remaining);
        let retry = Arc::new(Mutex::new(Some((bytes, request, done))));
        let ready = || -> Box<dyn FnOnce() + Send> {
            let (this, worker) = (Arc::clone(self), Arc::clone(&worker));
            let (remaining, retry) = (Arc::clone(&remaining), Arc::clone(&retry));
            Box::new(move || {
                if remaining.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) != 1 {
                    return;
                }
                let Some((bytes, request, done)) = retry.lock().expect("retry poisoned").take()
                else {
                    return;
                };
                let executor = Arc::clone(&this.executor);
                executor.spawn_ranked(
                    request.rank,
                    Box::new(move || this.complete(worker, bytes, request, done)),
                );
            })
        };
        if bidi {
            let (ready, retry) = (ready(), Arc::clone(&retry));
            self.wait_bidi(Box::new(move |loaded| match loaded {
                Ok(()) => ready(),
                // Its labels can't lay out without it: the load fails, as
                // one whose fetch failed does.
                Err(e) => {
                    if let Some((.., done)) = retry.lock().expect("retry poisoned").take() {
                        done(Err(format!("labels read right to left: {e}")));
                    }
                }
            }));
        }
        for url in urls {
            self.wait(&worker, style, url, ready());
        }
    }

    /// Runs `ready` once this context has the bidirectional algorithm,
    /// loading it if its host loads it apart.
    fn wait_bidi(&self, ready: BidiWaiter) {
        let bidi = self.bidi.lock().expect("bidi poisoned").clone();
        match bidi {
            Some(bidi) => bidi.wait(ready),
            None => ready(Err("no module loader".into())),
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
    /// A map's job host, its tile fetches taking turns
    /// [`orb_host::tile_fetches`] at once.
    pub fn new(fetch: Arc<dyn Fetch>, executor: Arc<dyn Executor>) -> Arc<Self> {
        Self::with_turns(fetch, executor, Turns::new(orb_host::tile_fetches()))
    }

    /// A job host fetching every load as it comes: a browser's job
    /// worker's, whose page sends loads in their turns.
    pub fn unlimited(fetch: Arc<dyn Fetch>, executor: Arc<dyn Executor>) -> Arc<Self> {
        Self::with_turns(fetch, executor, Turns::new(usize::MAX))
    }

    fn with_turns(fetch: Arc<dyn Fetch>, executor: Arc<dyn Executor>, turns: Turns) -> Arc<Self> {
        Arc::new(Self {
            loader: TileLoader::new(Arc::clone(&fetch)),
            turns,
            glyphs: Arc::new(GlyphLoads {
                fetch: Arc::clone(&fetch),
                executor: Arc::clone(&executor),
                waiting: Mutex::default(),
                recent: Mutex::default(),
                wants: Arc::default(),
                bidi: Mutex::new(None),
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

    /// How this host loads feature modules into its context: the
    /// bidirectional algorithm's ([`Modules::BIDI`]), once a tile's labels
    /// read right to left, for the styles set from now on. Without one it
    /// has every module.
    pub fn set_module_loader(&self, load: Load) {
        *self.glyphs.bidi.lock().expect("bidi poisoned") = Some(Arc::new(BidiLoad {
            load,
            held: Arc::new(AtomicBool::new(false)),
            waiting: Mutex::new(None),
        }));
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
        if let Some(bidi) = &*self.glyphs.bidi.lock().expect("bidi poisoned") {
            worker.set_bidi(Arc::clone(&bidi.held));
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

    /// Loads and lays out a tile; `fetched` hears when its fetch ends.
    pub fn load(&self, request: TileRequest, fetched: Option<Fetched>, done: LoadedDone) {
        let fetched = fetched.unwrap_or_else(|| Box::new(|| {}));
        let Some(worker) = self.worker(request.style) else {
            fetched();
            return done(Err(format!("style {} was replaced", request.style)));
        };
        let done = self.glyphs.wants.track(&request, done);
        let glyphs = Arc::clone(&self.glyphs);
        let rank = request.rank;
        if worker.has_data(&request) {
            fetched();
            self.executor.spawn_ranked(
                rank,
                Box::new(move || glyphs.complete(worker, Arc::default(), request, done)),
            );
            return;
        }
        // A document's one fetch serves every tile of its source.
        if let Some(url) = worker.data_url(request.source).map(str::to_string) {
            fetched();
            self.load_data(worker, url, request, done);
            return;
        }
        let (executor, loader) = (Arc::clone(&self.executor), Arc::clone(&self.loader));
        let key = load_key(&request);
        if let Some(kind) = worker.image_kind(request.source) {
            let (source, tile, warp) = (request.source, request.tile, request.warp);
            let resource = worker.resource_kind(source);
            let fetch: Start = Box::new(move |turn| {
                let Some(turn) = turn else {
                    fetched();
                    return done(Ok(Loaded::Cancelled));
                };
                let (abort, rank) = (turn.abort().clone(), turn.rank());
                loader.load(
                    resource,
                    request.location.clone(),
                    request.tile,
                    request.encoding,
                    &abort,
                    Box::new(move |bytes| {
                        drop(turn);
                        fetched();
                        match bytes {
                            // Given up while it was fetched: not decoded.
                            Ok(_) if !glyphs.wants.wanted(&request) => done(Ok(Loaded::Cancelled)),
                            // An archive's left-out tile draws nothing.
                            Ok((bytes, _)) if bytes.is_empty() => {
                                done(Ok(Loaded::Tile(TileBuild::default())))
                            }
                            // Its mesh on the surface's plane is the costly
                            // part of drawing it there: the grid is made here.
                            Ok((bytes, _)) if warp => executor.spawn_ranked(
                                rank,
                                Box::new(move || {
                                    let warp_grid = worker.warp_grid(source, tile).map(Arc::new);
                                    done(Ok(Loaded::Image {
                                        bytes,
                                        kind,
                                        warp_grid,
                                    }))
                                }),
                            ),
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
            return self.turns.wait(rank, key, fetch);
        }
        let newest = Arc::clone(&self.newest);
        let fetch: Start = Box::new(move |turn| {
            let Some(turn) = turn else {
                fetched();
                return done(Ok(Loaded::Cancelled));
            };
            let data_tile = request.data_tile();
            let (abort, rank) = (turn.abort().clone(), turn.rank());
            loader.load(
                ResourceKind::Tile,
                request.location.clone(),
                data_tile,
                request.encoding,
                &abort,
                Box::new(move |bytes| {
                    drop(turn);
                    fetched();
                    executor.spawn_ranked(
                        rank,
                        Box::new(move || match bytes {
                            Err(FetchError::Aborted) => done(Ok(Loaded::Cancelled)),
                            _ if newest.load(Ordering::Relaxed) > request.style => {
                                done(Err(format!("style {} was replaced", request.style)))
                            }
                            // A sparse tileset has no tile there: an
                            // ancestor stands in.
                            Ok((bytes, _)) if bytes.is_empty() && worker.sparse(request.source) => {
                                done(Ok(Loaded::Missing))
                            }
                            Ok((bytes, encoding)) => {
                                let request = TileRequest {
                                    encoding,
                                    ..request
                                };
                                glyphs.complete(worker, Arc::new(bytes), request, done)
                            }
                            Err(e) => done(missing(e)),
                        }),
                    );
                }),
            );
        });
        self.turns.wait(rank, key, fetch);
    }

    /// Ranks the loads waiting for their turns again
    /// ([`Turns::reorder`]).
    pub fn reorder(&self, rank: &dyn Fn(LoadKey) -> Option<u64>) {
        self.turns.reorder(rank);
    }

    /// The tile fetches this host started, aborted and completed so far.
    pub fn fetch_counts(&self) -> FetchCounts {
        self.turns.counts()
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
        self.glyphs.wants.give_up((style, source, tile));
        self.turns.cancel((style, source, tile));
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
