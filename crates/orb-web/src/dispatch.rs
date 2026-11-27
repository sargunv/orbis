//! Which job worker each of the map's requests goes to, and what happens
//! when one is lost: plain Rust over a [`Post`] of
//! [`orb_wire`]'s bytes, so its tests run natively. `jobs`
//! carries the messages between the page and the workers.
//!
//! A tile goes to the job worker with the fewest requests in flight (an
//! overzoomed one to its ancestor's), and its relayouts and queries follow
//! it there, where its data is; a glyph unit goes to its owner worker. A
//! worker that is lost (its wasm trapped, or it died) fails the queries it
//! had, which the map retries, and its rasters go again (their glyphs'
//! atlas rows wait for them). The page puts another in its place, which
//! this sends the style; or, past the page's limit, its share goes to the
//! next worker on. The loads it had go again at once, as suspects: one at
//! a time, with no other suspect in flight, so a loss then is likely its
//! own. A suspect lost again is quarantined: it loads as missing, so its
//! ancestor stands in, rather than take down a worker each time it is
//! loaded.
//!
//! Tile loads take turns across the workers ([`Turns`], the map's
//! `tileFetches` at once, [`orb_host::TILE_FETCHES`] by default): a load
//! goes to its worker in its turn, which passes on once the worker says
//! its fetch ended ([`JobReply::Fetched`]), so the map's order, which it
//! sets again each frame, holds across them all and the browser's network
//! queue stays short.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Weak};

use orb_crs::TileIndex;
use orb_host::Fetch;
use orb_runtime::{
    FeaturesDone, FetchCounts, GlyphRaster, JobReply, JobRequest, Jobs, QueryDone, RasterDone,
    SpriteIcons, StyleUpdate, TileDone, TileOutcome, TileRequest, TileResult, TileWorker, Turn,
    Turns,
};
use orb_wire as wire;

/// Posts a message, a [`JobRequest`] in [`wire`]'s bytes, to each of the
/// job workers named.
pub(crate) type Post = Box<dyn Fn(&[u32], &[u8]) + Send + Sync>;

/// [`Jobs`] for the map: each tile goes to the job worker with
/// the fewest requests in flight, and its relayouts and queries follow it
/// there, where its data was decoded.
pub(crate) struct WorkerJobs {
    workers: u32,
    post: Post,
    /// Where glyph ranges are fetched, once for every worker.
    fetch: Arc<dyn Fetch>,
    state: Mutex<State>,
    /// Tile loads taking turns across the workers.
    turns: Turns,
    /// Itself, for fetches that answer later.
    me: Weak<WorkerJobs>,
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
    /// Each request sent, by id, until its worker answers or is lost.
    in_flight: HashMap<u32, Sent>,
    /// Tiles whose loads were lost with a worker, by how many times: one
    /// is a suspect, two quarantine it.
    lost_loads: HashMap<LoadKey, u8>,
    /// The loads of suspects waiting for the one in flight.
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
    /// Sources whose every tile slices one document (GeoJSON), by key,
    /// as the style has them: their tiles go to one worker, which parses
    /// and indexes it once, its home in `whole_homes` once it has one.
    whole: HashSet<u32>,
    whole_homes: HashMap<u32, u32>,
    /// Loads answered, those failed without a reply included.
    done: u32,
}

/// A request sent to job worker `worker`, until it answers.
struct Sent {
    worker: u32,
    job: InFlight,
}

/// What a request in flight answers, and what's needed to send it again
/// or fail it if its worker is lost.
enum InFlight {
    /// A tile's load, sent again if its worker is lost, holding its `turn`
    /// until the worker's fetch ends. A `suspect`'s tile was lost with a
    /// worker before: no other suspect is in flight beside it.
    Load {
        request: TileRequest,
        done: TileDone,
        turn: Option<Turn>,
        suspect: bool,
    },
    /// A glyph unit's raster, sent again if its worker is lost.
    Raster {
        request: GlyphRaster,
        done: RasterDone,
    },
    /// A tile's features, failed if its worker is lost (the map retries).
    Features(FeaturesDone),
    /// A cluster's answer, failed if its worker is lost.
    Cluster(QueryDone),
}

/// A job's style, source and tile.
type LoadKey = (u64, u32, TileIndex);

/// The tile `request`'s job is of, whatever its kind: losing its worker
/// counts against the tile ([`QUARANTINE`]).
fn tile_key(request: &TileRequest) -> LoadKey {
    (request.style, request.source, request.tile)
}

/// Losses of one tile's loads that quarantine it.
const QUARANTINE: u8 = 2;

impl WorkerJobs {
    /// Jobs on `workers` job workers, reached through `post`, with
    /// `fetches` tile fetches in flight at once; glyph ranges come from
    /// `fetch`.
    pub(crate) fn new(
        workers: u32,
        fetches: usize,
        post: Post,
        fetch: Arc<dyn Fetch>,
    ) -> Arc<Self> {
        let workers = workers.max(1);
        Arc::new_cyclic(|me| Self {
            workers,
            post,
            fetch,
            state: Mutex::new(State {
                lost: vec![false; workers as usize],
                ..State::default()
            }),
            turns: Turns::new(fetches.max(1)),
            me: me.clone(),
        })
    }

    /// Loads answered so far, those failed without a reply included.
    pub(crate) fn done(&self) -> u32 {
        self.state.lock().expect("jobs poisoned").done
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
        let (jobs, heads, turns) = {
            let mut state = self.state.lock().expect("jobs poisoned");
            let Some(lost) = state.lost.get_mut(worker as usize) else {
                return;
            };
            *lost = replacement.is_none();
            let State {
                in_flight,
                lost_loads,
                ..
            } = &mut *state;
            let mut jobs: Vec<(u32, InFlight)> = (in_flight.extract_if(|_, s| s.worker == worker))
                .map(|(id, sent)| (id, sent.job))
                .collect();
            // Their turns pass on: the loads go again, in turns of their own.
            let mut turns = Vec::new();
            for (_, job) in &mut jobs {
                if let InFlight::Load { request, turn, .. } = job {
                    let n = lost_loads.entry(tile_key(request)).or_default();
                    *n = n.saturating_add(1);
                    turns.extend(turn.take());
                }
            }
            (jobs, state.heads.clone(), turns)
        };
        if let Some(style) = replacement {
            if let Some((style, update, icons)) = style {
                let update = Box::new(update);
                self.post(worker, &JobRequest::Style { style, update });
                self.post(worker, &JobRequest::Icons { style, icons });
            }
            for (url, head) in heads {
                let head = (*head).clone();
                self.post(worker, &JobRequest::ArchiveHead { url, head });
            }
        }
        // Their turns pass on once a replacement has the style.
        drop(turns);
        let message = format!("job worker {worker} lost: {why}");
        for (id, job) in jobs {
            match job {
                InFlight::Load { request, done, .. } if self.quarantined(tile_key(&request)) => {
                    done(Ok(quarantine(format!(
                        "tile {} is left out from now on: {message} loading it a second time",
                        request.tile
                    ))));
                }
                InFlight::Load { request, done, .. } => self.load(request, done),
                InFlight::Raster { request, done } => self.raster(request, done),
                job @ (InFlight::Features(_) | InFlight::Cluster(_)) => {
                    self.fail(id, job, message.clone());
                }
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
            if state.suspect_in_flight() {
                return;
            }
            state.suspects.pop_front()
        };
        if let Some((request, done)) = next {
            self.load(request, done);
        }
    }

    /// Completes the request `message` answers. A load answered with a
    /// tile, an image or a DEM tile, whose data travels beside the
    /// message, becomes its result through `payload`.
    pub(crate) fn complete(
        &self,
        message: JobReply,
        payload: impl FnOnce(JobReply) -> Result<TileResult, String>,
    ) -> Result<(), String> {
        let id = message.id();
        let mut state = self.state.lock().expect("jobs poisoned");
        if let JobReply::Fetched { .. } = message {
            // A load's fetch ended: its turn, dropped once the state is
            // let go, sends the next load.
            let turn = match state.in_flight.get_mut(&id) {
                Some(Sent {
                    job: InFlight::Load { turn, .. },
                    ..
                }) => turn.take(),
                _ => None,
            };
            drop(state);
            drop(turn);
            return Ok(());
        }
        let sent =
            (state.in_flight.remove(&id)).ok_or_else(|| format!("reply to unknown job {id}"))?;
        // A suspect answered: it didn't take its worker down.
        if let InFlight::Load {
            request,
            suspect: true,
            ..
        } = &sent.job
        {
            state.lost_loads.remove(&tile_key(request));
        }
        drop(state);
        self.answer(sent.job, message, payload);
        Ok(())
    }

    /// Answers `job` with `message`, a load's data through `payload`.
    fn answer(
        &self,
        job: InFlight,
        message: JobReply,
        payload: impl FnOnce(JobReply) -> Result<TileResult, String>,
    ) {
        let malformed = |what: &str| format!("malformed {what} reply");
        match job {
            InFlight::Features(done) => done(match message {
                JobReply::Features { features, .. } => Ok(features),
                JobReply::Error { message, .. } => Err(message),
                _ => Err(malformed("features")),
            }),
            InFlight::Cluster(done) => done(match message {
                JobReply::Answer { json, .. } => Ok(json),
                JobReply::Error { message, .. } => Err(message),
                _ => Err(malformed("cluster")),
            }),
            InFlight::Raster { done, .. } => done(match message {
                JobReply::Glyphs { bitmaps, .. } => Ok(bitmaps),
                JobReply::Error { message, .. } => Err(message),
                _ => Err(malformed("raster")),
            }),
            InFlight::Load {
                done,
                turn,
                suspect,
                ..
            } => {
                // The next load's turn sends it, the state let go.
                drop(turn);
                self.state.lock().expect("jobs poisoned").done += 1;
                done(match message {
                    JobReply::Missing { .. } => Ok(TileOutcome::Missing {
                        warnings: Vec::new(),
                    }),
                    JobReply::Cancelled { .. } => Ok(TileOutcome::Cancelled),
                    JobReply::RetryAfter { error, wait_ms, .. } => {
                        Ok(TileOutcome::RetryAfter { error, wait_ms })
                    }
                    JobReply::Error { message, .. } => Err(message),
                    JobReply::Tile { .. } | JobReply::Image { .. } | JobReply::Dem { .. } => {
                        payload(message).map(TileOutcome::Built)
                    }
                    _ => Err(malformed("job")),
                });
                if suspect {
                    self.next_suspect();
                }
            }
        }
    }

    /// Fails request `id`'s `job` with `message`.
    fn fail(&self, id: u32, job: InFlight, message: String) {
        self.answer(job, JobReply::Error { id, message }, |_| {
            Err("malformed job reply".into())
        });
    }

    /// Sends the request `make` builds with its id to the worker
    /// `preferred` picks (or the next one on, while it's lost), keeping
    /// `job` until it answers. With every worker lost, the request fails
    /// at once.
    fn send(
        &self,
        preferred: impl FnOnce(&mut State) -> u32,
        job: InFlight,
        make: impl FnOnce(u32) -> JobRequest,
    ) {
        let mut state = self.state.lock().expect("jobs poisoned");
        let preferred = preferred(&mut state);
        let id = state.next_id;
        state.next_id = id.wrapping_add(1);
        let Some(worker) = route(&state.lost, preferred) else {
            drop(state);
            self.fail(id, job, "every job worker was lost".into());
            return;
        };
        state.in_flight.insert(id, Sent { worker, job });
        drop(state);
        self.post(worker, &make(id));
    }

    /// Posts `request` to `worker`.
    fn post(&self, worker: u32, request: &JobRequest) {
        (self.post)(&[worker], &wire::encode(request));
    }

    /// Posts `request` to every worker not lost.
    fn broadcast(&self, request: &JobRequest) {
        let lost = self.state.lock().expect("jobs poisoned").lost.clone();
        let workers: Vec<u32> = (0..self.workers).filter(|&w| !lost[w as usize]).collect();
        (self.post)(&workers, &wire::encode(request));
    }
}

/// A quarantined tile's load: missing, so its ancestor stands in, with why
/// as a warning.
fn quarantine(why: String) -> TileOutcome {
    TileOutcome::Missing {
        warnings: (!why.is_empty()).then_some(why).into_iter().collect(),
    }
}

/// Tiles whose worker is remembered: far more than the workers keep
/// decoded. A tile forgotten goes to a worker afresh, which fetches it again.
const HOMES: usize = 8192;

impl State {
    /// The worker `tile` of `source` went to, or the one with the fewest
    /// requests in flight, which becomes its home.
    fn home(&mut self, source: u32, tile: TileIndex) -> u32 {
        // A whole source's home stays while the style has the source.
        let whole = self.whole.contains(&source);
        let home = match whole {
            true => self.whole_homes.get(&source),
            false => self.homes.get(&(source, tile)),
        };
        if let Some(&worker) = home {
            return worker;
        }
        let mut busy = vec![0usize; self.lost.len()];
        for sent in self.in_flight.values() {
            busy[sent.worker as usize] += 1;
        }
        let worker = (0..self.lost.len())
            .filter(|&w| !self.lost[w])
            .min_by_key(|&w| busy[w])
            .unwrap_or(0) as u32;
        if whole {
            self.whole_homes.insert(source, worker);
            return worker;
        }
        if self.home_order.len() == HOMES
            && let Some(oldest) = self.home_order.pop_front()
        {
            self.homes.remove(&oldest);
        }
        self.homes.insert((source, tile), worker);
        self.home_order.push_back((source, tile));
        worker
    }

    /// Whether a suspect's load is in flight.
    fn suspect_in_flight(&self) -> bool {
        (self.in_flight.values()).any(|s| matches!(s.job, InFlight::Load { suspect: true, .. }))
    }
}

impl WorkerJobs {
    /// Sends a job to its tile's worker, in its `turn` (none for a
    /// relayout or a column job), which it holds until the worker's fetch
    /// ends.
    fn send_load(
        &self,
        mut request: TileRequest,
        done: TileDone,
        suspect: bool,
        turn: Option<Turn>,
    ) {
        request.rank = turn.as_ref().map_or(request.rank, Turn::rank);
        let (source, tile) = (request.source, request.data_tile());
        let abort = (request.load_key()).zip(turn.as_ref().map(|t| t.abort().clone()));
        let job = InFlight::Load {
            request: request.clone(),
            done,
            turn,
            suspect,
        };
        self.send(
            |state| state.home(source, tile),
            job,
            |id| JobRequest::Load { id, request },
        );
        // A turn taken back (a load beside the view yielding to one of
        // the view's, or given up) stops the worker's fetch.
        if let Some((key, abort)) = abort {
            let me = self.me.clone();
            abort.on_abort(move || {
                if let Some(jobs) = me.upgrade() {
                    jobs.stop_sent(key);
                }
            });
        }
    }

    /// Tells the worker a sent load of `key` went to that it's given up.
    fn stop_sent(&self, key: LoadKey) {
        let worker = {
            let state = self.state.lock().expect("jobs poisoned");
            (state.in_flight.values()).find_map(|sent| match &sent.job {
                InFlight::Load { request, .. } if request.load_key() == Some(key) => {
                    Some(sent.worker)
                }
                _ => None,
            })
        };
        if let Some(worker) = worker {
            let (style, source, tile) = key;
            self.post(
                worker,
                &JobRequest::Cancel {
                    style,
                    source,
                    tile,
                },
            );
        }
    }
}

/// The first worker not lost from `preferred` on, if any is left.
fn route(lost: &[bool], preferred: u32) -> Option<u32> {
    let n = lost.len() as u32;
    (0..n)
        .map(|i| (preferred + i) % n)
        .find(|&w| !lost[w as usize])
}

impl Jobs for WorkerJobs {
    fn set_style(&self, style: u64, update: &StyleUpdate, worker: &Arc<TileWorker>) {
        {
            // Keys are the style's own: another style's source may have
            // a GeoJSON source's key, and spreads over the workers.
            let mut state = self.state.lock().expect("jobs poisoned");
            state.whole = worker.whole_sources().into_iter().collect();
            let State {
                whole, whole_homes, ..
            } = &mut *state;
            whole_homes.retain(|source, _| whole.contains(source));
        }
        self.broadcast(&JobRequest::Style {
            style,
            update: Box::new(update.clone()),
        });
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
        self.broadcast(&JobRequest::ArchiveHead {
            url: url.to_string(),
            head: (**head).clone(),
        });
    }

    fn set_icons(&self, style: u64, icons: &Arc<SpriteIcons>, _: &Arc<TileWorker>) {
        self.broadcast(&JobRequest::Icons {
            style,
            icons: (**icons).clone(),
        });
    }

    fn prefetch_glyphs(&self, style: u64, urls: Vec<String>) {
        // The page fetches each range once and hands it to every worker,
        // which lays out with its own metrics; their tiles that need one
        // meanwhile wait for it rather than fetch it again.
        self.broadcast(&JobRequest::Glyphs {
            style,
            urls: urls.clone(),
        });
        for url in urls {
            let (me, fetching) = (self.me.clone(), url.clone());
            let done = Box::new(move |fetched: Result<orb_host::Fetched, _>| {
                let Some(jobs) = me.upgrade() else {
                    return;
                };
                let bytes = (fetched.map(|f| f.bytes)).map_err(|e| format!("glyphs {url}: {e}"));
                jobs.broadcast(&JobRequest::GlyphRange { style, url, bytes });
            });
            let never = &orb_host::Abort::NEVER;
            (self.fetch).get(orb_host::ResourceKind::Glyphs, &fetching, never, done);
        }
    }

    fn retry_glyphs(&self) {
        // Ahead of the relayouts that need the ranges: each worker takes
        // its messages in order.
        self.broadcast(&JobRequest::RetryGlyphs);
    }

    fn raster(&self, request: GlyphRaster, done: RasterDone) {
        // A unit's owner worker rasterizes it.
        let owner = request.owner(self.workers);
        let job = InFlight::Raster {
            request: request.clone(),
            done,
        };
        self.send(|_| owner, job, |id| JobRequest::Raster { id, request });
    }

    fn load(&self, request: TileRequest, done: TileDone) {
        let suspect = {
            let mut state = self.state.lock().expect("jobs poisoned");
            match state.lost_loads.get(&tile_key(&request)) {
                Some(&n) if n >= QUARANTINE => {
                    drop(state);
                    return done(Ok(quarantine(String::new())));
                }
                Some(_) if state.suspect_in_flight() => {
                    state.suspects.push_back((request, done));
                    return;
                }
                found => found.is_some(),
            }
        };
        if !request.loads_whole() {
            return self.send_load(request, done, suspect, None);
        }
        let (me, key, rank) = (self.me.clone(), request.load_key(), request.rank);
        let start: orb_runtime::Start = Box::new(move |turn| {
            let Some(turn) = turn else {
                return done(Ok(TileOutcome::Cancelled));
            };
            match me.upgrade() {
                Some(jobs) => jobs.send_load(request, done, suspect, Some(turn)),
                None => done(Ok(TileOutcome::Cancelled)),
            }
        });
        self.turns.wait(rank, key, start);
    }

    fn cancel(&self, style: u64, source: u32, tile: TileIndex) {
        let key = (style, source, tile);
        // One waiting for its turn is answered here; one in its turn hears
        // from its worker, which its turn's abort tells; one sent without
        // a turn too.
        if !self.turns.cancel(key) {
            self.stop_sent(key);
        }
    }

    fn fetch_counts(&self) -> FetchCounts {
        self.turns.counts()
    }

    fn reorder(&self, rank: &dyn Fn(orb_runtime::LoadKey) -> Option<u64>) {
        self.turns.reorder(rank);
    }

    fn features(&self, request: orb_runtime::FeatureRequest, done: FeaturesDone) {
        let (source, tile) = (request.tile.source, request.tile.data_tile());
        self.send(
            |state| state.home(source, tile),
            InFlight::Features(done),
            |id| JobRequest::Features { id, request },
        );
    }

    fn cluster(&self, request: orb_runtime::ClusterRequest, done: QueryDone) {
        let (source, tile) = (request.tile.source, request.tile.data_tile());
        self.send(
            |state| state.home(source, tile),
            InFlight::Cluster(done),
            |id| JobRequest::Cluster { id, request },
        );
    }
}

#[cfg(test)]
mod tests;
