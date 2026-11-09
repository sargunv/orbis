//! One job host for every platform: what a map's tile,
//! query and glyph jobs need beyond a worker's CPU work. It keeps the style
//! revisions requests name, shares fetches between the requests that need
//! them (a source's data document, glyph ranges, word data, font files), and
//! answers each request once. Natively [`PoolJobs`](crate::PoolJobs) runs one
//! in the map's process; on the web every job worker runs one behind the
//! page's messages. Image tiles come out as their encoded bytes: each
//! platform decodes them its own way.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use orb_host::jobs::Executor;
use orb_host::{Fetch, FetchError, FontFiles};

use crate::glyphs::{GlyphRaster, LocalGlyphs, RasterDone};
use crate::style::ImageKind;
use crate::worker::{Built, StyleUpdate, TileRequest, TileWorker};
use crate::{FeatureRequest, FeaturesDone, SpriteIcons, TileBuild, TileLoader};

/// What a tile load came to.
#[derive(Debug)]
pub enum Loaded {
    Tile(TileBuild),
    /// An image source's tile, still encoded.
    Image {
        bytes: Vec<u8>,
        kind: ImageKind,
    },
    /// The source doesn't have the tile.
    Missing,
}

/// Called once with a load's outcome, on any thread.
pub type LoadedDone = Box<dyn FnOnce(Result<Loaded, String>) + Send>;

/// Loads of a source's data document in flight, by style and source: the
/// requests waiting on each.
type DataLoads = HashMap<(u64, u32), Vec<(TileRequest, LoadedDone)>>;

/// Glyph range, word data and font file fetches in flight, by style and
/// URL: what to run when each arrives.
type GlyphWaiters = HashMap<(u64, String), Vec<Box<dyn FnOnce() + Send>>>;

/// A job host: see the module.
pub struct JobHost {
    fetch: Arc<dyn Fetch>,
    loader: Arc<TileLoader>,
    executor: Arc<dyn Executor>,
    /// The latest revision of the current style and of the one before it,
    /// whose requests may still be in flight.
    styles: Mutex<Vec<(u64, Arc<TileWorker>)>>,
    data_loads: Arc<Mutex<DataLoads>>,
    glyphs: Arc<GlyphLoads>,
    /// The host's local fonts, if any.
    local: Mutex<Option<Arc<dyn LocalGlyphs>>>,
    files: Mutex<Option<Arc<dyn FontFiles>>>,
    /// Where the Southeast Asian word data is, for hosts that load it
    /// lazily.
    word_data_url: Mutex<Option<String>>,
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
    /// The word data once a style's worker has loaded it, for the others.
    words: Mutex<Option<Arc<orb_text::WordBreaker>>>,
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
        let url = key.1.clone();
        self.fetch.get(
            &key.1.clone(),
            Box::new(move |fetched| {
                let files = worker
                    .data
                    .glyphs
                    .files()
                    .filter(|_| worker.declares_font(&url));
                if let Ok(bytes) = &fetched
                    && worker.is_glyph_range(&url)
                {
                    let mut recent = this.recent.lock().expect("ranges poisoned");
                    recent.insert(&url, Arc::new(bytes.clone()));
                }
                let finish = move |prepared: Result<(), String>| {
                    let bytes = prepared.and(fetched.as_deref().map_err(ToString::to_string));
                    worker.add_resource(&url, bytes);
                    if let Some(words) = worker.word_breaker() {
                        this.words
                            .lock()
                            .expect("words poisoned")
                            .get_or_insert(words);
                    }
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
            }),
        );
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
/// 404) is [`Loaded::Missing`]; other fetch errors are errors.
fn missing(error: FetchError) -> Result<Loaded, String> {
    match error {
        FetchError::NotFound => Ok(Loaded::Missing),
        e => Err(e.to_string()),
    }
}

impl JobHost {
    pub fn new(fetch: Arc<dyn Fetch>, executor: Arc<dyn Executor>) -> Arc<Self> {
        Arc::new(Self {
            loader: TileLoader::new(Arc::clone(&fetch)),
            glyphs: Arc::new(GlyphLoads {
                fetch: Arc::clone(&fetch),
                executor: Arc::clone(&executor),
                waiting: Mutex::default(),
                words: Mutex::new(None),
                recent: Mutex::default(),
            }),
            fetch,
            executor,
            styles: Mutex::new(Vec::new()),
            data_loads: Arc::default(),
            local: Mutex::new(None),
            files: Mutex::new(None),
            word_data_url: Mutex::new(None),
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

    /// Fetches the Southeast Asian word data from `url` when labels first
    /// need it, rather than compiling it in.
    pub fn set_word_data_url(&self, url: &str) {
        *self.word_data_url.lock().expect("word data poisoned") = Some(url.to_string());
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
        if let Some(words) = &*self.glyphs.words.lock().expect("words poisoned") {
            worker.set_word_breaker(Arc::clone(words));
        } else if let Some(url) = &*self.word_data_url.lock().expect("word data poisoned") {
            worker.set_word_data_url(url);
        }
        let mut styles = self.styles.lock().expect("styles poisoned");
        styles.retain(|(i, _)| *i + 1 >= id && *i != id);
        styles.push((id, worker));
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
        let executor = Arc::clone(&self.executor);
        if let Some(kind) = worker.image_kind(request.source) {
            self.loader.load(
                request.location,
                request.tile,
                request.encoding,
                Box::new(move |fetched| match fetched {
                    // An archive's left-out tile draws nothing.
                    Ok((bytes, _)) if bytes.is_empty() => {
                        done(Ok(Loaded::Tile(TileBuild::default())))
                    }
                    Ok((bytes, _)) => done(Ok(Loaded::Image { bytes, kind })),
                    Err(e) => done(missing(e)),
                }),
            );
            return;
        }
        let data_tile = request.data_tile();
        self.loader.load(
            request.location.clone(),
            data_tile,
            request.encoding,
            Box::new(move |fetched| {
                executor.spawn(Box::new(move || match fetched {
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
        let key = (request.style, request.source);
        let glyphs = Arc::clone(&self.glyphs);
        let mut loads = self.data_loads.lock().expect("data loads poisoned");
        if let Some(waiting) = loads.get_mut(&key) {
            waiting.push((request, done));
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
        loads.insert(key, vec![(request, done)]);
        drop(loads);
        let loads = Arc::clone(&self.data_loads);
        let executor = Arc::clone(&self.executor);
        self.fetch.get(
            &url,
            Box::new(move |fetched| {
                executor.spawn(Box::new(move || {
                    // Index the document before removing the in-flight
                    // entry, without holding the registry during CPU work.
                    let (request, done) = loads
                        .lock()
                        .expect("data loads poisoned")
                        .get_mut(&key)
                        .expect("registered fetch")
                        .pop()
                        .expect("first request");
                    let fetched = fetched.map(Arc::new).map_err(|e| e.to_string());
                    let build = |request: TileRequest, done: LoadedDone, bytes| match &fetched {
                        Ok(_) => glyphs.complete(Arc::clone(&worker), bytes, request, done),
                        Err(e) => done(Err(e.clone())),
                    };
                    build(request, done, fetched.clone().unwrap_or_default());
                    let waiting = loads
                        .lock()
                        .expect("data loads poisoned")
                        .remove(&key)
                        .expect("registered fetch");
                    // The document is indexed: later requests ignore bytes.
                    for (request, done) in waiting {
                        build(request, done, Arc::default());
                    }
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
            tile.location.clone(),
            tile.data_tile(),
            tile.encoding,
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
            tile.location.clone(),
            tile.data_tile(),
            tile.encoding,
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
                    &url.clone(),
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
                        &url.clone(),
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
