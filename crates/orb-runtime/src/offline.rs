//! Offline regions: a style's resources for an area and a zoom
//! range, downloaded into [`Storage`] and served from it before the
//! network.
//!
//! A region keeps, under `regions/<id>/`:
//! - `manifest.json`: its definition, whether it's complete, its size, the
//!   resources and the tile archives;
//! - `resources`: the bodies of everything fetched by URL (or URL and byte
//!   range) but tiles from `{z}/{x}/{y}` templates: the style, TileJSON,
//!   glyph ranges, sprites, GeoJSON and image data, and a `pmtiles://`
//!   source's header, directories and tiles, as the runtime reads them;
//! - `<n>.pmtiles`: an extract (`orb_source::pmtiles::write_head`) of each
//!   templated source's tiles, as the server sent them.
//!
//! [`RegionFetch`] answers a map's fetches from complete regions, so a map
//! of a downloaded area opens with no network, and passes the rest on.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use orb_crs::{TileIndex, TileMatrixSet};
use orb_expr::json::{self, Json, JsonValue};
use orb_host::{Abort, Fetch, FetchDone, FetchError, ResourceKind, Storage, StorageDone};
use orb_source::endpoint::TileEndpoint;
use orb_source::pmtiles::{self, Contents, Entry, Lookup, Reader};
use orb_style::{GeoJsonData, Scheme, SourceKind};

/// What a region holds: a style's map of `bounds` (west, south, east,
/// north, degrees) between map zooms `zooms`.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionDefinition {
    pub name: String,
    pub style_url: String,
    pub bounds: [f64; 4],
    pub zooms: [f64; 2],
    /// What MapLibre Native's definitions add, kept and given back: the
    /// app's bytes, the pixel ratio, whether ideographs are downloaded,
    /// and a GeoJSON geometry the region covers within `bounds`.
    pub metadata: Vec<u8>,
    pub pixel_ratio: f32,
    pub include_ideographs: bool,
    pub geometry: Option<String>,
}

impl Default for RegionDefinition {
    fn default() -> Self {
        Self {
            name: String::new(),
            style_url: String::new(),
            bounds: [-180.0, -85.051_128_779_806_59, 180.0, 85.051_128_779_806_59],
            zooms: [0.0, 0.0],
            metadata: Vec::new(),
            pixel_ratio: 1.0,
            include_ideographs: false,
            geometry: None,
        }
    }
}

/// A region in storage.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionInfo {
    pub id: u32,
    pub definition: RegionDefinition,
    /// Whether its download finished; only complete regions are served.
    pub complete: bool,
    pub bytes: u64,
}

/// How a download goes.
#[derive(Clone, Debug, PartialEq)]
pub enum RegionEvent {
    /// Requests done of those known so far (the total grows as TileJSON
    /// arrives), and bytes kept.
    Progress {
        id: u32,
        done: u64,
        total: u64,
        bytes: u64,
    },
    /// The region is stored and served; `failed` requests (after retries)
    /// are missing from it.
    Complete { id: u32, bytes: u64, failed: u64 },
    /// The download stopped: storage failed, or the style didn't load.
    Failed { id: u32, message: String },
}

/// Called with each event of a download, on any thread.
pub type RegionEvents = Arc<dyn Fn(RegionEvent) + Send + Sync>;

/// The most tiles one region may hold.
pub(crate) const MAX_REGION_TILES: usize = 250_000;
/// Requests in flight at once.
const PARALLEL: usize = 8;
/// Bytes buffered before they're appended to storage.
const FLUSH_BYTES: usize = 1 << 20;
/// Network failures retried per request.
const RETRIES: u8 = 2;

/// The default `text-font`, as GL JS's.
const DEFAULT_FONT: [&str; 2] = ["Open Sans Regular", "Arial Unicode MS Regular"];

fn region_dir(id: u32) -> String {
    format!("regions/{id}")
}

/// Starts downloading `definition` through `fetch` (the network, or the
/// HTTP cache) into `storage` as a new region, and calls `done` with its
/// id once its id is chosen; `events` follow.
pub fn download(
    fetch: Arc<dyn Fetch>,
    storage: Arc<dyn Storage>,
    definition: RegionDefinition,
    events: RegionEvents,
    done: StorageDone<Download>,
) {
    let listed = storage.clone();
    listed.list(
        "regions",
        Box::new(move |names| {
            let names = match names {
                Ok(names) => names,
                Err(e) => return done(Err(e)),
            };
            let id = next_id(&names);
            let job = Arc::new(Job {
                id,
                fetch,
                storage,
                definition,
                events,
                state: Mutex::new(State::default()),
                abort: Abort::new(),
            });
            let download = Download(job.clone());
            // The manifest goes first, so an interrupted download is listed
            // (and can be deleted).
            let manifest = job.manifest(false);
            let storage = job.storage.clone();
            storage.write(
                &format!("{}/manifest.json", region_dir(id)),
                manifest.into_bytes(),
                false,
                Box::new(move |written| match written {
                    Ok(()) => {
                        job.enqueue(Task::Resource {
                            url: job.definition.style_url.clone(),
                            range: None,
                            then: Then::Style,
                        });
                        done(Ok(download));
                    }
                    Err(e) => done(Err(e)),
                }),
            );
        }),
    );
}

/// A download in progress.
#[derive(Clone)]
pub struct Download(Arc<Job>);

impl std::fmt::Debug for Download {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Download").field(&self.0.id).finish()
    }
}

impl Download {
    pub fn id(&self) -> u32 {
        self.0.id
    }

    /// Stops the download, aborting its fetches; its region stays
    /// incomplete until deleted.
    pub fn cancel(&self) {
        self.0.lock().cancelled = true;
        self.0.abort.abort();
    }
}

/// Deletes region `id` and everything it holds.
pub fn delete(storage: &dyn Storage, id: u32, done: StorageDone<()>) {
    storage.remove(&region_dir(id), done);
}

/// The regions in `storage`.
pub fn list(storage: Arc<dyn Storage>, done: StorageDone<Vec<RegionInfo>>) {
    regions(
        storage,
        Box::new(move |found| {
            done(found.map(|found| found.into_iter().map(|(info, _)| info).collect()))
        }),
    );
}

/// A region written whole from what another store holds, as a merge
/// reads it: its resources and templated tiles go in as they're read, and
/// [`Import::finish`] stores it complete.
#[derive(Clone)]
pub struct Import(Arc<Job>);

impl std::fmt::Debug for Import {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Import").field(&self.0.id).finish()
    }
}

/// Fetches nothing: an import's region reads nothing from the network.
struct NoFetch;

impl Fetch for NoFetch {
    fn get(&self, _: ResourceKind, _: &str, _: &Abort, done: FetchDone) {
        done(Err(FetchError::NotFound));
    }

    fn get_range(&self, _: ResourceKind, _: &str, _: u64, _: u64, _: &Abort, done: FetchDone) {
        done(Err(FetchError::NotFound));
    }
}

/// Starts region `definition` in `storage`, written by the [`Import`]
/// `done` gets; `events` get its completion.
pub fn import(
    storage: Arc<dyn Storage>,
    definition: RegionDefinition,
    events: RegionEvents,
    done: StorageDone<Import>,
) {
    let listed = storage.clone();
    listed.list(
        "regions",
        Box::new(move |names| {
            let id = match names {
                Ok(names) => next_id(&names),
                Err(e) => return done(Err(e)),
            };
            let job = Arc::new(Job {
                id,
                fetch: Arc::new(NoFetch),
                storage,
                definition,
                events,
                state: Mutex::new(State::default()),
                abort: Abort::new(),
            });
            let manifest = job.manifest(false);
            let storage = job.storage.clone();
            storage.write(
                &format!("{}/manifest.json", region_dir(id)),
                manifest.into_bytes(),
                false,
                Box::new(move |written| done(written.map(|()| Import(job)))),
            );
        }),
    );
}

/// The id after the greatest of the regions `names`.
fn next_id(names: &[String]) -> u32 {
    names
        .iter()
        .filter_map(|n| n.parse::<u32>().ok())
        .max()
        .map_or(1, |n| n + 1)
}

impl Import {
    pub fn id(&self) -> u32 {
        self.0.id
    }

    /// Keeps the body of `url`, as a download keeps a resource.
    pub fn resource(&self, url: &str, bytes: &[u8]) {
        self.0.keep(orb_host::cache::key(url, None), bytes);
    }

    /// Keeps `tile` of the source whose tiles are at `template`, a URL with
    /// `{z}`, `{x}` and `{y}` and no other tokens, `y` counted from the
    /// north.
    pub fn tile(&self, template: &str, tile: TileIndex, bytes: Vec<u8>) {
        let archive = {
            let mut state = self.0.lock();
            let found = state
                .archives
                .iter()
                .position(|a| a.templates.iter().any(|t| t == template));
            let index = found.unwrap_or_else(|| {
                state.archives.push(Archive {
                    templates: vec![template.to_string()],
                    scheme: Scheme::Xyz,
                    tile_type: tile_type(&bytes),
                    zooms: [tile.z, tile.z],
                    bounds: self.0.definition.bounds,
                    entries: Vec::new(),
                    data_len: 0,
                    buffer: Vec::new(),
                });
                state.archives.len() - 1
            });
            let zooms = &mut state.archives[index].zooms;
            *zooms = [zooms[0].min(tile.z), zooms[1].max(tile.z)];
            index
        };
        self.0.keep_tile(archive, tile, bytes);
    }

    /// Stores the region complete, then tells its events.
    pub fn finish(self) {
        self.0.lock().stopped = true;
        self.0.finish();
    }
}

/// What a tile's first bytes say it is, for an extract's header: tiles
/// serve by URL whatever it says.
fn tile_type(bytes: &[u8]) -> pmtiles::TileType {
    match bytes {
        [0x89, b'P', b'N', b'G', ..] => pmtiles::TileType::Png,
        [0xff, 0xd8, ..] => pmtiles::TileType::Jpeg,
        [b'R', b'I', b'F', b'F', ..] => pmtiles::TileType::Webp,
        _ => pmtiles::TileType::Mvt,
    }
}

/// The regions in `storage`, with their manifests.
fn regions(storage: Arc<dyn Storage>, done: StorageDone<Vec<(RegionInfo, Json)>>) {
    let reader = storage.clone();
    storage.list(
        "regions",
        Box::new(move |names| match names {
            Ok(names) => {
                let ids = names.iter().filter_map(|n| n.parse().ok()).collect();
                manifests(reader, ids, Vec::new(), done);
            }
            Err(e) => done(Err(e)),
        }),
    );
}

/// Reads regions' manifests, one after another.
fn manifests(
    storage: Arc<dyn Storage>,
    mut ids: Vec<u32>,
    mut out: Vec<(RegionInfo, Json)>,
    done: StorageDone<Vec<(RegionInfo, Json)>>,
) {
    let Some(id) = ids.pop() else {
        return done(Ok(out));
    };
    let next = storage.clone();
    storage.read(
        &format!("{}/manifest.json", region_dir(id)),
        None,
        Box::new(move |read| {
            match read {
                Ok(Some(bytes)) => {
                    if let Some(found) = parse_manifest(id, &bytes) {
                        out.push(found);
                    }
                }
                Ok(None) => {}
                Err(e) => return done(Err(e)),
            }
            manifests(next, ids, out, done);
        }),
    );
}

fn parse_manifest(id: u32, bytes: &[u8]) -> Option<(RegionInfo, Json)> {
    let doc = json::parse(std::str::from_utf8(bytes).ok()?).ok()?;
    let numbers = |key: &str| -> Option<Vec<f64>> {
        doc.get(key)?.as_array()?.iter().map(Json::as_f64).collect()
    };
    let bounds = numbers("bounds")?;
    let zooms = numbers("zooms")?;
    let info = RegionInfo {
        id,
        definition: RegionDefinition {
            name: doc.get("name")?.as_str()?.to_string(),
            style_url: doc.get("style")?.as_str()?.to_string(),
            bounds: bounds.try_into().ok()?,
            zooms: zooms.try_into().ok()?,
            metadata: numbers("metadata")
                .unwrap_or_default()
                .into_iter()
                .map(|b| b as u8)
                .collect(),
            pixel_ratio: doc
                .get("pixel_ratio")
                .and_then(Json::as_f64)
                .map_or(1.0, |r| r as f32),
            include_ideographs: doc
                .get("ideographs")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            geometry: doc
                .get("geometry")
                .and_then(Json::as_str)
                .map(str::to_string),
        },
        complete: doc.get("complete")?.as_bool()?,
        bytes: doc.get("bytes")?.as_f64()? as u64,
    };
    Some((info, doc))
}

// ---------------------------------------------------------------------
// Downloading

#[derive(Debug)]
enum Task {
    /// Fetch a URL (or a byte range of it) and keep it as a resource.
    Resource {
        url: String,
        range: Option<(u64, u64)>,
        then: Then,
    },
    /// A templated source's tile, into archive `archive`.
    Tile {
        archive: usize,
        tile: TileIndex,
        url: String,
    },
    /// A `pmtiles://` source's tile, read through its directories.
    ArchiveTile { source: usize, tile: TileIndex },
}

/// What a resource is for, once it arrives.
#[derive(Debug)]
enum Then {
    /// Kept for the region, a resource of this kind, and nothing follows.
    Keep(ResourceKind),
    Style,
    /// TileJSON of a source.
    TileJson(String),
    /// A `pmtiles://` source's header.
    Header(String),
}

impl Then {
    /// The kind of resource that arrives for this.
    fn kind(&self) -> ResourceKind {
        match self {
            &Then::Keep(kind) => kind,
            Then::Style => ResourceKind::Style,
            Then::TileJson(_) | Then::Header(_) => ResourceKind::Source,
        }
    }
}

#[derive(Default)]
struct State {
    queue: VecDeque<Task>,
    in_flight: usize,
    done: u64,
    total: u64,
    failed: u64,
    cancelled: bool,
    stopped: bool,
    /// A pump is running, and was asked to look again.
    pumping: bool,
    again: bool,
    /// The style, parsed, once it arrives.
    style: Option<Arc<orb_style::Style>>,
    /// Keys of resources already kept: `resources` offset and length.
    resources: HashMap<String, (u64, u64)>,
    resources_len: u64,
    /// Bytes waiting to be appended to `resources`.
    resources_buffer: Vec<u8>,
    archives: Vec<Archive>,
    /// `pmtiles://` sources' readers, by archive URL.
    readers: Vec<(String, Arc<Mutex<Reader>>)>,
    /// Appends to storage in order: one in flight at a time.
    writes: VecDeque<(String, Vec<u8>)>,
    writing: bool,
    bytes: u64,
}

struct Archive {
    templates: Vec<String>,
    scheme: Scheme,
    tile_type: pmtiles::TileType,
    zooms: [u8; 2],
    bounds: [f64; 4],
    entries: Vec<Entry>,
    data_len: u64,
    buffer: Vec<u8>,
}

struct Job {
    id: u32,
    fetch: Arc<dyn Fetch>,
    storage: Arc<dyn Storage>,
    definition: RegionDefinition,
    events: RegionEvents,
    state: Mutex<State>,
    /// Fires when the download is cancelled: its fetches stop.
    abort: Abort,
}

impl Job {
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("region download poisoned")
    }

    fn enqueue(self: &Arc<Self>, task: Task) {
        {
            let mut state = self.lock();
            state.total += 1;
            state.queue.push_back(task);
        }
        self.pump();
    }

    /// Starts queued tasks up to the parallel limit, or finishes. One
    /// call pumps at a time: a call while another runs (a fetch answered
    /// at once) only asks it to look again, so the stack stays flat.
    fn pump(self: &Arc<Self>) {
        {
            let mut state = self.lock();
            if state.pumping {
                state.again = true;
                return;
            }
            state.pumping = true;
        }
        loop {
            let task = {
                let mut state = self.lock();
                let next = if state.stopped || state.in_flight >= PARALLEL {
                    None
                } else if state.cancelled {
                    state.stopped = true;
                    None
                } else {
                    state.queue.pop_front()
                };
                match next {
                    Some(task) => {
                        state.in_flight += 1;
                        task
                    }
                    None if state.again => {
                        state.again = false;
                        continue;
                    }
                    None => {
                        state.pumping = false;
                        let finished =
                            !state.stopped && state.queue.is_empty() && state.in_flight == 0;
                        if finished {
                            state.stopped = true;
                            drop(state);
                            self.finish();
                        }
                        return;
                    }
                }
            };
            self.start(task, 0);
        }
    }

    fn start(self: &Arc<Self>, task: Task, tries: u8) {
        let job = self.clone();
        match &task {
            Task::Resource { url, range, then } => {
                let (url, range, kind) = (url.clone(), *range, then.kind());
                let done: FetchDone = Box::new(move |result| job.arrived(task, tries, result));
                match range {
                    Some((offset, length)) => {
                        (self.fetch).get_range(kind, &url, offset, length, &self.abort, done);
                    }
                    None => self.fetch.get(kind, &url, &self.abort, done),
                }
            }
            Task::Tile { url, .. } => {
                let url = url.clone();
                self.fetch.get(
                    ResourceKind::Tile,
                    &url,
                    &self.abort,
                    Box::new(move |result| job.arrived(task, tries, result)),
                );
            }
            Task::ArchiveTile { source, tile } => {
                let (url, reader) = self.lock().readers[*source].clone();
                let lookup = reader.lock().expect("reader poisoned").lookup(*tile);
                match lookup {
                    Ok(Lookup::Read(range)) => {
                        let reader = reader.clone();
                        let key = orb_host::cache::key(&url, Some((range.offset, range.length)));
                        self.fetch.get_range(
                            ResourceKind::Source,
                            &url,
                            range.offset,
                            range.length,
                            &self.abort,
                            Box::new(move |result| match result {
                                Ok(bytes) => {
                                    let inserted = reader
                                        .lock()
                                        .expect("reader poisoned")
                                        .insert(range, &bytes);
                                    job.keep(key, &bytes);
                                    match inserted {
                                        // The same tile again, one level deeper.
                                        Ok(()) => job.start(task, tries),
                                        Err(e) => {
                                            job.arrived(task, RETRIES, Err(FetchError::Range(e)))
                                        }
                                    }
                                }
                                // An archive shorter than the first read is
                                // read header first.
                                Err(e)
                                    if crate::loader::falls_back(&e)
                                        && reader
                                            .lock()
                                            .expect("reader poisoned")
                                            .read_failed(range) =>
                                {
                                    job.start(task, tries)
                                }
                                Err(e) => job.arrived(task, tries, Err(e)),
                            }),
                        );
                    }
                    Ok(Lookup::Tile { range, .. }) => {
                        let resource = Task::Resource {
                            url,
                            range: Some((range.offset, range.length)),
                            then: Then::Keep(ResourceKind::Tile),
                        };
                        // Counted once: as this task.
                        self.start(resource, tries);
                    }
                    Ok(Lookup::Missing) => self.arrived(task, tries, Err(FetchError::NotFound)),
                    Err(e) => self.arrived(task, RETRIES, Err(FetchError::Range(e))),
                }
            }
        }
    }

    fn arrived(self: &Arc<Self>, task: Task, tries: u8, result: Result<Vec<u8>, FetchError>) {
        match result {
            Ok(bytes) => match task {
                Task::Resource { url, range, then } => {
                    self.keep(orb_host::cache::key(&url, range), &bytes);
                    self.then(then, &url, &bytes);
                }
                Task::Tile { archive, tile, .. } => self.keep_tile(archive, tile, bytes),
                Task::ArchiveTile { .. } => unreachable!("archive tiles arrive as resources"),
            },
            // Not in the tileset, or no such glyph range: nothing to keep.
            // Aborted, the download is stopping.
            Err(FetchError::NotFound | FetchError::Aborted) => {}
            Err(FetchError::Network(_) | FetchError::Status(500..)) if tries < RETRIES => {
                return self.start(task, tries + 1);
            }
            Err(e) => {
                let mut state = self.lock();
                state.failed += 1;
                if matches!(
                    task,
                    Task::Resource {
                        then: Then::Style,
                        ..
                    }
                ) {
                    state.stopped = true;
                    drop(state);
                    return (self.events)(RegionEvent::Failed {
                        id: self.id,
                        message: format!("style: {e}"),
                    });
                }
            }
        }
        let (done, total, bytes) = {
            let mut state = self.lock();
            state.in_flight -= 1;
            state.done += 1;
            (state.done, state.total, state.bytes)
        };
        if done % 32 == 0 || done == total {
            (self.events)(RegionEvent::Progress {
                id: self.id,
                done,
                total,
                bytes,
            });
        }
        self.pump();
    }

    /// Keeps a resource's body, once per key.
    fn keep(self: &Arc<Self>, key: String, bytes: &[u8]) {
        let flush = {
            let mut state = self.lock();
            if state.resources.contains_key(&key) {
                return;
            }
            let at = state.resources_len;
            state.resources.insert(key, (at, bytes.len() as u64));
            state.resources_len += bytes.len() as u64;
            state.bytes += bytes.len() as u64;
            state.resources_buffer.extend_from_slice(bytes);
            (state.resources_buffer.len() >= FLUSH_BYTES)
                .then(|| std::mem::take(&mut state.resources_buffer))
        };
        if let Some(buffer) = flush {
            self.append(format!("{}/resources", region_dir(self.id)), buffer);
        }
    }

    fn keep_tile(self: &Arc<Self>, archive: usize, tile: TileIndex, bytes: Vec<u8>) {
        // Archives hold no empty bodies; the network answers those.
        if bytes.is_empty() {
            return;
        }
        let flush = {
            let mut state = self.lock();
            state.bytes += bytes.len() as u64;
            let a = &mut state.archives[archive];
            a.entries.push(Entry {
                id: pmtiles::tile_id(tile).expect("a valid tile"),
                offset: a.data_len,
                length: bytes.len() as u64,
                run: 1,
            });
            a.data_len += bytes.len() as u64;
            a.buffer.extend(bytes);
            (a.buffer.len() >= FLUSH_BYTES).then(|| std::mem::take(&mut a.buffer))
        };
        if let Some(buffer) = flush {
            self.append(format!("{}/{archive}.data", region_dir(self.id)), buffer);
        }
    }

    /// Appends to a storage key, in order with the other appends.
    fn append(self: &Arc<Self>, key: String, bytes: Vec<u8>) {
        self.append_all(vec![(key, bytes)]);
    }

    /// Queues appends together, so none finds the queue drained between
    /// them (storage may call back at once).
    fn append_all(self: &Arc<Self>, appends: Vec<(String, Vec<u8>)>) {
        let next = {
            let mut state = self.lock();
            state.writes.extend(appends);
            if state.writing {
                return;
            }
            let Some(next) = state.writes.pop_front() else {
                return;
            };
            state.writing = true;
            next
        };
        self.write_next(next);
    }

    fn write_next(self: &Arc<Self>, (key, bytes): (String, Vec<u8>)) {
        let job = self.clone();
        self.storage.write(
            &key,
            bytes,
            true,
            Box::new(move |result| {
                if let Err(e) = result {
                    job.fail(e);
                }
                let next = {
                    let mut state = job.lock();
                    let next = state.writes.pop_front();
                    state.writing = next.is_some();
                    next
                };
                match next {
                    Some(next) => job.write_next(next),
                    // Writes drained: finishing may be waiting for them.
                    None => job.pump_finish(),
                }
            }),
        );
    }

    fn fail(&self, message: String) {
        let mut state = self.lock();
        if !state.cancelled {
            state.cancelled = true;
            drop(state);
            self.abort.abort();
            (self.events)(RegionEvent::Failed {
                id: self.id,
                message,
            });
        }
    }

    fn then(self: &Arc<Self>, then: Then, url: &str, bytes: &[u8]) {
        match then {
            Then::Keep(_) => {}
            Then::Style => match std::str::from_utf8(bytes)
                .map_err(|e| e.to_string())
                .and_then(|text| {
                    // What a style could use is fetched, whatever previews
                    // the maps that read the region enable.
                    orb_style::v8::parse_with(text, orb_style::Previews::ALL, &[])
                        .map_err(|e| e.to_string())
                        .map(|s| (s, text))
                }) {
                Ok((style, text)) => self.style(style, text),
                Err(e) => self.fail(format!("style: {e}")),
            },
            Then::TileJson(source) => {
                let text = String::from_utf8_lossy(bytes);
                self.tiles(&source, Some(&text), None);
            }
            Then::Header(source) => match orb_source::endpoint::archive_tilejson(bytes) {
                Ok(tilejson) => self.tiles(&source, Some(&tilejson), Some(url)),
                Err(e) => self.fail(format!("{url}: {e}")),
            },
        }
    }

    /// Queues everything the style needs but tiles, which follow their
    /// sources' TileJSON.
    fn style(self: &Arc<Self>, style: orb_style::Style, text: &str) {
        let style = Arc::new(style);
        self.lock().style = Some(style.clone());
        for sprite in &style.sprites {
            for ratio in ["", "@2x"] {
                for (ext, kind) in [
                    (".json", ResourceKind::SpriteJson),
                    (".png", ResourceKind::SpriteImage),
                ] {
                    let split = sprite.url.find(['?', '#']).unwrap_or(sprite.url.len());
                    let (path, query) = sprite.url.split_at(split);
                    self.resource(format!("{path}{ratio}{ext}{query}"), Then::Keep(kind));
                }
            }
        }
        if let Some(template) = &style.glyphs {
            for stack in font_stacks(text) {
                let stack = orb_text::FontStack::new(&stack);
                for range in 0..256u32 {
                    let first = char::from_u32(range * 256);
                    let last = char::from_u32(range * 256 + 255);
                    // Ranges the map draws with local fonts are never fetched.
                    let local = |c: Option<char>| {
                        c.is_none_or(|c| {
                            orb_text::unicode::uses_local_ideograph_font(c)
                                || orb_text::unicode::is_complex_shaping(c)
                        })
                    };
                    if local(first) && local(last) {
                        continue;
                    }
                    self.resource(
                        orb_text::glyphs_url(template, &stack, range),
                        Then::Keep(ResourceKind::Glyphs),
                    );
                }
            }
        }
        for source in &style.sources {
            match &source.kind {
                SourceKind::Vector(v) => self.tileset(&source.id, &v.tileset),
                SourceKind::Raster(r) => self.tileset(&source.id, &r.tileset),
                SourceKind::RasterDem(d) => self.tileset(&source.id, &d.raster.tileset),
                SourceKind::GeoJson(g) => {
                    if let GeoJsonData::Url(url) = &g.data {
                        self.resource(url.clone(), Then::Keep(ResourceKind::Source));
                    }
                }
                SourceKind::Image(image) => {
                    // A video streams from its host's player: not stored.
                    if let orb_style::ImageContent::Image(url) = &image.content {
                        self.resource(url.clone(), Then::Keep(ResourceKind::Image));
                    }
                }
            }
        }
    }

    fn resource(self: &Arc<Self>, url: String, then: Then) {
        self.enqueue(Task::Resource {
            url,
            range: None,
            then,
        });
    }

    fn tileset(self: &Arc<Self>, source: &str, tileset: &orb_style::TileSet) {
        match (&tileset.tiles, &tileset.url) {
            (Some(_), _) | (None, None) => self.tiles(source, None, None),
            (None, Some(url)) => match url.strip_prefix("pmtiles://") {
                Some(archive) => self.enqueue(Task::Resource {
                    url: archive.to_string(),
                    range: Some((0, pmtiles::HEADER_LEN)),
                    then: Then::Header(source.to_string()),
                }),
                None => self.resource(url.clone(), Then::TileJson(source.to_string())),
            },
        }
    }

    /// Queues a source's tiles in the region, once its TileJSON (or
    /// archive header) is known.
    fn tiles(self: &Arc<Self>, source: &str, tilejson: Option<&str>, archive_url: Option<&str>) {
        let Some(style) = self.lock().style.clone() else {
            return;
        };
        let Some(def) = style.sources.iter().find(|s| s.id == source) else {
            return;
        };
        let tms = TileMatrixSet::web_mercator().clone();
        // A source without tiles (bad TileJSON) leaves them out.
        let Ok(endpoint) = TileEndpoint::resolve(&def.kind, tilejson, tms) else {
            self.lock().failed += 1;
            return;
        };
        let tiles = endpoint.region_tiles(self.definition.bounds, self.definition.zooms);
        if tiles.len() > MAX_REGION_TILES {
            return self.fail(format!(
                "source {source}: {} tiles, over {MAX_REGION_TILES}",
                tiles.len()
            ));
        }
        if let Some(url) = archive_url {
            let index = {
                let mut state = self.lock();
                state.readers.push((url.to_string(), Arc::default()));
                state.readers.len() - 1
            };
            for tile in tiles {
                self.enqueue(Task::ArchiveTile {
                    source: index,
                    tile,
                });
            }
            return;
        }
        // Templates with only {z}, {x} and {y} go into an extract; others
        // are kept as resources by URL.
        let templated = endpoint.templates.iter().all(|t| {
            let rest = t.replace("{z}", "").replace("{x}", "").replace("{y}", "");
            !rest.contains('{')
        });
        if !templated {
            for tile in tiles {
                self.resource(endpoint.url(tile), Then::Keep(ResourceKind::Tile));
            }
            return;
        }
        let tile_type = match def.kind {
            SourceKind::Vector(_) => match endpoint.encoding {
                orb_style::Encoding::Mvt => pmtiles::TileType::Mvt,
                orb_style::Encoding::Mlt => pmtiles::TileType::Mlt,
            },
            _ => pmtiles::TileType::Png,
        };
        let archive = {
            let mut state = self.lock();
            state.archives.push(Archive {
                templates: endpoint.templates.clone(),
                scheme: endpoint.scheme,
                tile_type,
                zooms: [endpoint.minzoom, endpoint.maxzoom],
                bounds: self.definition.bounds,
                entries: Vec::new(),
                data_len: 0,
                buffer: Vec::new(),
            });
            state.archives.len() - 1
        };
        for tile in tiles {
            let url = endpoint.url(tile);
            self.enqueue(Task::Tile { archive, tile, url });
        }
    }

    /// Called when writes drain: finishes if the downloads had.
    fn pump_finish(self: &Arc<Self>) {
        let ready = {
            let state = self.lock();
            state.stopped && !state.cancelled && state.queue.is_empty() && state.in_flight == 0
        };
        if ready {
            self.finish();
        }
    }

    /// Flushes buffers, writes the archives' heads and the manifest.
    fn finish(self: &Arc<Self>) {
        let (buffers, writing) = {
            let mut state = self.lock();
            let mut buffers = Vec::new();
            if !state.resources_buffer.is_empty() {
                buffers.push((
                    format!("{}/resources", region_dir(self.id)),
                    std::mem::take(&mut state.resources_buffer),
                ));
            }
            for (i, a) in state.archives.iter_mut().enumerate() {
                if !a.buffer.is_empty() {
                    buffers.push((
                        format!("{}/{i}.data", region_dir(self.id)),
                        std::mem::take(&mut a.buffer),
                    ));
                }
            }
            (buffers, state.writing || !state.writes.is_empty())
        };
        if !buffers.is_empty() || writing {
            self.append_all(buffers);
            // `pump_finish` comes back once the appends are done.
            return;
        }
        self.assemble(0);
    }

    /// Writes archive `index` as its head followed by its data, then the
    /// next, then the manifest.
    fn assemble(self: &Arc<Self>, index: usize) {
        let head = {
            let mut state = self.lock();
            let Some(a) = state.archives.get_mut(index) else {
                drop(state);
                return self.complete();
            };
            let contents = Contents {
                tile_type: a.tile_type,
                tile_compression: pmtiles::Compression::None,
                minzoom: a.zooms[0],
                maxzoom: a.zooms[1],
                bounds: a.bounds,
                metadata: "{}".into(),
            };
            (pmtiles::write_head(&mut a.entries, &contents), a.data_len)
        };
        let (head, data_len) = head;
        let head = match head {
            Ok(head) => head,
            Err(e) => return self.fail(e),
        };
        let dir = region_dir(self.id);
        let (archive, data) = (
            format!("{dir}/{index}.pmtiles"),
            format!("{dir}/{index}.data"),
        );
        let job = self.clone();
        self.storage.write(
            &archive.clone(),
            head,
            false,
            Box::new(move |result| match result {
                Ok(()) => job.copy(index, archive, data, 0, data_len),
                Err(e) => job.fail(e),
            }),
        );
    }

    /// Appends `data`'s bytes from `at` to `archive`, a chunk at a time.
    fn copy(self: &Arc<Self>, index: usize, archive: String, data: String, at: u64, len: u64) {
        if at >= len {
            let job = self.clone();
            return self
                .storage
                .remove(&data, Box::new(move |_| job.assemble(index + 1)));
        }
        let chunk = (len - at).min(16 << 20);
        let job = self.clone();
        self.storage.read(
            &data.clone(),
            Some((at, chunk)),
            Box::new(move |result| match result {
                Ok(Some(bytes)) => {
                    let next = job.clone();
                    job.storage.write(
                        &archive.clone(),
                        bytes,
                        true,
                        Box::new(move |result| match result {
                            Ok(()) => next.copy(index, archive, data, at + chunk, len),
                            Err(e) => next.fail(e),
                        }),
                    );
                }
                Ok(None) => job.fail(format!("{data} vanished")),
                Err(e) => job.fail(e),
            }),
        );
    }

    fn complete(self: &Arc<Self>) {
        let manifest = self.manifest(true);
        let job = self.clone();
        self.storage.write(
            &format!("{}/manifest.json", region_dir(self.id)),
            manifest.into_bytes(),
            false,
            Box::new(move |result| match result {
                Ok(()) => {
                    let (bytes, failed) = {
                        let state = job.lock();
                        (state.bytes, state.failed)
                    };
                    (job.events)(RegionEvent::Complete {
                        id: job.id,
                        bytes,
                        failed,
                    });
                }
                Err(e) => job.fail(e),
            }),
        );
    }

    fn manifest(&self, complete: bool) -> String {
        let state = self.lock();
        let d = &self.definition;
        let j = |value| Json::new(value, json::Span::default());
        let num = |n: f64| j(JsonValue::Number(n));
        let text = |s: &str| j(JsonValue::String(s.into()));
        let array = |items: Vec<Json>| j(JsonValue::Array(items));
        let object = |members: Vec<(&str, Json)>| {
            j(JsonValue::Object(
                members
                    .into_iter()
                    .map(|(key, value)| json::Member {
                        key: key.into(),
                        key_span: json::Span::default(),
                        value,
                    })
                    .collect(),
            ))
        };
        let resources = state
            .resources
            .iter()
            .map(|(key, (at, len))| array(vec![text(key), num(*at as f64), num(*len as f64)]))
            .collect();
        let archives = state
            .archives
            .iter()
            .map(|a| {
                object(vec![
                    (
                        "templates",
                        array(a.templates.iter().map(|t| text(t)).collect()),
                    ),
                    ("tms", j(JsonValue::Bool(a.scheme == Scheme::Tms))),
                ])
            })
            .collect();
        let mut members = vec![
            ("name", text(&d.name)),
            ("style", text(&d.style_url)),
            ("bounds", array(d.bounds.iter().map(|v| num(*v)).collect())),
            ("zooms", array(d.zooms.iter().map(|v| num(*v)).collect())),
            ("complete", j(JsonValue::Bool(complete))),
            ("bytes", num(state.bytes as f64)),
            ("resources", array(resources)),
            ("archives", array(archives)),
            (
                "metadata",
                array(d.metadata.iter().map(|&b| num(b.into())).collect()),
            ),
            ("pixel_ratio", num(d.pixel_ratio.into())),
            ("ideographs", j(JsonValue::Bool(d.include_ideographs))),
        ];
        if let Some(geometry) = &d.geometry {
            members.push(("geometry", text(geometry)));
        }
        object(members).to_string()
    }
}

/// The font stacks a style's symbol layers name: every literal array of
/// strings in a `text-font`, or the default for symbol layers with text
/// and no `text-font`.
fn font_stacks(style: &str) -> Vec<Vec<String>> {
    let mut stacks = HashSet::default();
    let Ok(doc) = json::parse(style) else {
        return Vec::new();
    };
    fn strings(value: &Json, out: &mut HashSet<Vec<String>>) {
        if let Some(items) = value.as_array() {
            let names: Option<Vec<String>> = items
                .iter()
                .map(|i| i.as_str().map(str::to_string))
                .collect();
            match names {
                Some(names) if !names.is_empty() && !is_operator(&names[0]) => {
                    out.insert(names);
                }
                _ => items.iter().for_each(|i| strings(i, out)),
            }
        }
    }
    fn is_operator(name: &str) -> bool {
        matches!(
            name,
            "get" | "literal" | "concat" | "to-string" | "var" | "global-state"
        )
    }
    for layer in doc
        .get("layers")
        .and_then(Json::as_array)
        .unwrap_or_default()
    {
        if layer.get("type").and_then(Json::as_str) != Some("symbol") {
            continue;
        }
        let layout = layer.get("layout");
        match layout.and_then(|l| l.get("text-font")) {
            Some(font) => strings(font, &mut stacks),
            None if layout.is_some_and(|l| l.get("text-field").is_some()) => {
                stacks.insert(DEFAULT_FONT.map(String::from).to_vec());
            }
            None => {}
        }
    }
    let mut stacks: Vec<_> = stacks.into_iter().collect();
    stacks.sort();
    stacks
}

// ---------------------------------------------------------------------
// Serving

/// Answers fetches from the complete regions in `storage` before passing
/// them to `inner`: resources by URL (and range), templated tiles from the
/// extracts. Fetches wait for the regions to be read at first.
pub struct RegionFetch {
    inner: Arc<dyn Fetch>,
    storage: Arc<dyn Storage>,
    served: Mutex<Served>,
}

impl std::fmt::Debug for RegionFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegionFetch").finish_non_exhaustive()
    }
}

/// A fetch waiting for the regions to be read.
type Pending = (ResourceKind, String, Option<(u64, u64)>, Abort, FetchDone);

enum Served {
    Loading(Vec<Pending>),
    Ready(Arc<Vec<ServedRegion>>),
}

struct ServedRegion {
    dir: String,
    resources: HashMap<String, (u64, u64)>,
    archives: Vec<ServedArchive>,
}

struct ServedArchive {
    key: String,
    templates: Vec<Vec<Part>>,
    tms: bool,
    reader: Arc<Mutex<Reader>>,
}

#[derive(Debug, PartialEq)]
enum Part {
    Text(String),
    Z,
    X,
    Y,
}

impl RegionFetch {
    pub fn new(inner: Arc<dyn Fetch>, storage: Arc<dyn Storage>) -> Arc<Self> {
        let this = Arc::new(Self {
            inner,
            storage,
            served: Mutex::new(Served::Loading(Vec::new())),
        });
        this.reload();
        this
    }

    /// Reads the regions again: after a download completes or a region is
    /// deleted.
    pub fn reload(self: &Arc<Self>) {
        let this = self.clone();
        regions(
            self.storage.clone(),
            Box::new(move |found| {
                // Without readable storage (an insecure page), no regions.
                let regions = found
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|(info, _)| info.complete)
                    .map(|(info, doc)| served_region(info.id, &doc))
                    .collect();
                let pending = {
                    let mut served = this.served.lock().expect("regions poisoned");
                    match std::mem::replace(&mut *served, Served::Ready(Arc::new(regions))) {
                        Served::Loading(pending) => pending,
                        Served::Ready(_) => Vec::new(),
                    }
                };
                for (kind, url, range, abort, done) in pending {
                    this.serve(kind, &url, range, &abort, done);
                }
            }),
        );
    }

    fn serve(
        &self,
        kind: ResourceKind,
        url: &str,
        range: Option<(u64, u64)>,
        abort: &Abort,
        done: FetchDone,
    ) {
        let regions = {
            let mut served = self.served.lock().expect("regions poisoned");
            match &mut *served {
                Served::Loading(pending) => {
                    let pending_abort = abort.clone();
                    return pending.push((kind, url.to_string(), range, pending_abort, done));
                }
                Served::Ready(regions) => regions.clone(),
            }
        };
        let key = orb_host::cache::key(url, range);
        for region in regions.iter() {
            if let Some(&(at, length)) = region.resources.get(&key) {
                let (inner, url, abort) = (self.inner.clone(), url.to_string(), abort.clone());
                return self.storage.read(
                    &format!("{}/resources", region.dir),
                    Some((at, length)),
                    Box::new(move |read| match read {
                        Ok(Some(bytes)) => done(Ok(bytes)),
                        _ => pass(&*inner, kind, &url, range, &abort, done),
                    }),
                );
            }
        }
        if range.is_none() {
            for region in regions.iter() {
                for archive in &region.archives {
                    if let Some(tile) = archive.tile(url) {
                        let served = Tile {
                            inner: self.inner.clone(),
                            storage: self.storage.clone(),
                            key: archive.key.clone(),
                            reader: archive.reader.clone(),
                            url: url.to_string(),
                            kind,
                            abort: abort.clone(),
                        };
                        return served.read(tile, 0, done);
                    }
                }
            }
        }
        pass(&*self.inner, kind, url, range, abort, done);
    }
}

fn pass(
    inner: &dyn Fetch,
    kind: ResourceKind,
    url: &str,
    range: Option<(u64, u64)>,
    abort: &Abort,
    done: FetchDone,
) {
    match range {
        Some((offset, length)) => inner.get_range(kind, url, offset, length, abort, done),
        None => inner.get(kind, url, abort, done),
    }
}

impl Fetch for RegionFetch {
    fn get(&self, kind: ResourceKind, url: &str, abort: &Abort, done: FetchDone) {
        self.serve(kind, url, None, abort, done);
    }
    fn get_range(
        &self,
        kind: ResourceKind,
        url: &str,
        offset: u64,
        length: u64,
        abort: &Abort,
        done: FetchDone,
    ) {
        self.serve(kind, url, Some((offset, length)), abort, done);
    }
}

/// A templated tile read from a region's extract.
struct Tile {
    inner: Arc<dyn Fetch>,
    storage: Arc<dyn Storage>,
    key: String,
    reader: Arc<Mutex<Reader>>,
    url: String,
    kind: ResourceKind,
    abort: Abort,
}

impl Tile {
    fn read(self, tile: TileIndex, reads: u8, done: FetchDone) {
        let lookup = self.reader.lock().expect("reader poisoned").lookup(tile);
        match lookup {
            Ok(Lookup::Read(range)) if reads < 6 => {
                let storage = self.storage.clone();
                storage.read(
                    &self.key.clone(),
                    Some((range.offset, range.length)),
                    Box::new(move |read| {
                        let mut reader = self.reader.lock().expect("reader poisoned");
                        let kept = match read {
                            Ok(Some(bytes)) => reader.insert(range, &bytes).is_ok(),
                            _ => false,
                        };
                        // An extract shorter than the first read is read
                        // header first.
                        let again = kept || reader.read_failed(range);
                        drop(reader);
                        if again {
                            self.read(tile, reads + 1, done);
                        } else {
                            pass(&*self.inner, self.kind, &self.url, None, &self.abort, done);
                        }
                    }),
                );
            }
            Ok(Lookup::Tile { range, .. }) => {
                let storage = self.storage.clone();
                storage.read(
                    &self.key.clone(),
                    Some((range.offset, range.length)),
                    Box::new(move |read| match read {
                        Ok(Some(bytes)) => done(Ok(bytes)),
                        _ => pass(&*self.inner, self.kind, &self.url, None, &self.abort, done),
                    }),
                );
            }
            // Not in the region: the network.
            _ => pass(&*self.inner, self.kind, &self.url, None, &self.abort, done),
        }
    }
}

fn served_region(id: u32, doc: &Json) -> ServedRegion {
    let dir = region_dir(id);
    let resources = doc
        .get("resources")
        .and_then(Json::as_array)
        .unwrap_or_default()
        .iter()
        .filter_map(|r| {
            let [key, at, length] = r.as_array()? else {
                return None;
            };
            Some((
                key.as_str()?.to_string(),
                (at.as_f64()? as u64, length.as_f64()? as u64),
            ))
        })
        .collect();
    let archives = doc
        .get("archives")
        .and_then(Json::as_array)
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(i, a)| ServedArchive {
            key: format!("{dir}/{i}.pmtiles"),
            templates: a
                .get("templates")
                .and_then(Json::as_array)
                .unwrap_or_default()
                .iter()
                .filter_map(|t| Some(parse_template(t.as_str()?)))
                .collect(),
            tms: a.get("tms").and_then(Json::as_bool).unwrap_or(false),
            reader: Arc::default(),
        })
        .collect();
    ServedRegion {
        dir,
        resources,
        archives,
    }
}

fn parse_template(template: &str) -> Vec<Part> {
    let mut parts = Vec::new();
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        let (part, kind) = match &rest[start..] {
            t if t.starts_with("{z}") => (Part::Z, 3),
            t if t.starts_with("{x}") => (Part::X, 3),
            t if t.starts_with("{y}") => (Part::Y, 3),
            _ => {
                parts.push(Part::Text(rest[..=start].to_string()));
                rest = &rest[start + 1..];
                continue;
            }
        };
        if start > 0 {
            parts.push(Part::Text(rest[..start].to_string()));
        }
        parts.push(part);
        rest = &rest[start + kind..];
    }
    if !rest.is_empty() {
        parts.push(Part::Text(rest.to_string()));
    }
    // Adjacent text parts, from `{` that started no placeholder.
    let mut merged: Vec<Part> = Vec::new();
    for part in parts {
        match (merged.last_mut(), part) {
            (Some(Part::Text(a)), Part::Text(b)) => a.push_str(&b),
            (_, part) => merged.push(part),
        }
    }
    merged
}

impl ServedArchive {
    /// The tile a URL names under one of the templates.
    fn tile(&self, url: &str) -> Option<TileIndex> {
        self.templates.iter().find_map(|t| {
            let (mut z, mut x, mut y) = (None, None, None);
            let mut rest = url;
            for part in t {
                match part {
                    Part::Text(text) => rest = rest.strip_prefix(text.as_str())?,
                    number => {
                        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
                        let value: u32 = rest[..digits].parse().ok()?;
                        rest = &rest[digits..];
                        match number {
                            Part::Z => z = Some(value),
                            Part::X => x = Some(value),
                            _ => y = Some(value),
                        }
                    }
                }
            }
            let (z, x, y) = (u8::try_from(z?).ok()?, x?, y?);
            if !rest.is_empty() || z > 30 || x >> z != 0 || y >> z != 0 {
                return None;
            }
            let row = if self.tms { (1 << z) - 1 - y } else { y };
            Some(TileIndex::new(z, x, row))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    /// A server with a style, its TileJSON, tiles at every z/x/y, one glyph
    /// range and a sprite; or, `offline`, no network at all.
    #[derive(Debug, Default)]
    struct Server {
        offline: bool,
        requests: Mutex<Vec<String>>,
    }

    const STYLE: &str = r#"{"version": 8,
        "sources": {"v": {"type": "vector", "url": "https://s/tiles.json"},
                    "r": {"type": "raster", "tiles": ["https://r/{z}/{x}/{y}.png?k=1"], "maxzoom": 1}},
        "glyphs": "https://s/fonts/{fontstack}/{range}.pbf",
        "sprite": "https://s/sprite",
        "layers": [{"id": "l", "type": "symbol", "source": "v", "source-layer": "x",
                    "layout": {"text-field": "a", "text-font": ["literal", ["Sans Bold"]]}}]}"#;

    impl Fetch for Server {
        fn get(&self, _: orb_host::ResourceKind, url: &str, _: &Abort, done: FetchDone) {
            self.requests.lock().unwrap().push(url.to_string());
            if self.offline {
                return done(Err(FetchError::Network("offline".into())));
            }
            done(match url {
                "https://s/style.json" => Ok(STYLE.into()),
                "https://s/tiles.json" => {
                    Ok(br#"{"tiles": ["https://t/{z}/{x}/{y}.pbf"], "maxzoom": 3}"#.to_vec())
                }
                "https://s/fonts/Sans Bold/0-255.pbf" => Ok(b"glyphs".to_vec()),
                u if u.starts_with("https://s/sprite") => Ok(u.as_bytes().to_vec()),
                u if u.starts_with("https://t/") || u.starts_with("https://r/") => {
                    Ok(format!("tile {u}").into_bytes())
                }
                _ => Err(FetchError::NotFound),
            })
        }
        fn get_range(
            &self,
            _: orb_host::ResourceKind,
            _: &str,
            _: u64,
            _: u64,
            _: &Abort,
            done: FetchDone,
        ) {
            done(Err(FetchError::NotFound));
        }
    }

    fn get(fetch: &dyn Fetch, url: &str) -> Result<Vec<u8>, FetchError> {
        let (tx, rx) = mpsc::channel();
        fetch.get(
            ResourceKind::Unknown,
            url,
            &Abort::NEVER,
            Box::new(move |r| tx.send(r).unwrap()),
        );
        rx.recv().unwrap()
    }

    #[test]
    fn regions_download_and_open_offline() {
        let root = std::env::temp_dir().join(format!("orb-offline-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let storage: Arc<dyn Storage> = Arc::new(orb_host_native::FileStorage::new(&root));
        let server = Arc::new(Server::default());
        let (tx, rx) = mpsc::channel();
        let definition = RegionDefinition {
            name: "Tokyo".into(),
            style_url: "https://s/style.json".into(),
            bounds: [139.7, 35.6, 139.8, 35.7],
            zooms: [0.0, 2.0],
            ..RegionDefinition::default()
        };
        let events: RegionEvents = Arc::new(move |e| tx.send(e).unwrap());
        download(
            server.clone(),
            storage.clone(),
            definition.clone(),
            events,
            Box::new(drop),
        );
        let complete = rx
            .iter()
            .find(|e| !matches!(e, RegionEvent::Progress { .. }))
            .unwrap();
        let RegionEvent::Complete {
            id: 1,
            bytes,
            failed: 0,
        } = complete
        else {
            panic!("{complete:?}");
        };
        assert!(bytes > 0);
        let requests = server.requests.lock().unwrap().clone();
        // Vector tiles at zooms 0–2; 256-pixel rasters a zoom deeper, to
        // their maxzoom.
        for url in [
            "https://t/0/0/0.pbf",
            "https://t/2/3/1.pbf",
            "https://r/1/1/0.png?k=1",
        ] {
            assert!(requests.iter().any(|r| r == url), "{url} not requested");
        }
        assert!(!requests.iter().any(|r| r.starts_with("https://t/3/")));
        assert!(requests.iter().any(|r| r == "https://s/sprite@2x.png"));

        let (tx, rx) = mpsc::channel();
        list(storage.clone(), Box::new(move |r| tx.send(r).unwrap()));
        let regions = rx.recv().unwrap().unwrap();
        assert_eq!(regions.len(), 1);
        assert_eq!(
            (regions[0].complete, &regions[0].definition),
            (true, &definition)
        );

        let offline = Arc::new(Server {
            offline: true,
            ..Server::default()
        });
        let fetch = RegionFetch::new(offline.clone(), storage.clone());
        assert_eq!(
            get(&*fetch, "https://s/style.json").unwrap(),
            STYLE.as_bytes()
        );
        assert_eq!(
            get(&*fetch, "https://t/2/3/1.pbf").unwrap(),
            b"tile https://t/2/3/1.pbf"
        );
        assert_eq!(
            get(&*fetch, "https://r/1/1/0.png?k=1").unwrap(),
            b"tile https://r/1/1/0.png?k=1"
        );
        assert_eq!(
            get(&*fetch, "https://s/fonts/Sans Bold/0-255.pbf").unwrap(),
            b"glyphs"
        );
        assert!(
            offline.requests.lock().unwrap().is_empty(),
            "served without the network"
        );
        // Outside the region: the network, which is gone.
        assert!(get(&*fetch, "https://t/2/0/0.pbf").is_err());

        let (tx, rx) = mpsc::channel();
        delete(&*storage, 1, Box::new(move |r| tx.send(r).unwrap()));
        rx.recv().unwrap().unwrap();
        fetch.reload();
        assert!(get(&*fetch, "https://s/style.json").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn templates_match_their_tiles() {
        let archive = ServedArchive {
            key: String::new(),
            templates: vec![parse_template("https://a/{z}/{x}/{y}.pbf?key={k}")],
            tms: true,
            reader: Arc::default(),
        };
        assert_eq!(
            archive.tile("https://a/3/2/1.pbf?key={k}"),
            Some(TileIndex::new(3, 2, 6))
        );
        assert_eq!(archive.tile("https://a/3/2/1.pbf?key=x"), None);
        assert_eq!(archive.tile("https://a/3/9/1.pbf?key={k}"), None);
    }
}
