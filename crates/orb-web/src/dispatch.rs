//! Which job worker each of the map's requests goes to, and what happens
//! when one is lost: plain Rust over a [`Post`] of
//! [`orb_runtime::wire`]'s bytes, so its tests run natively. `jobs`
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
use orb_runtime::wire;
use orb_runtime::{
    FetchCounts, GlyphRaster, JobReply, JobRequest, Jobs, RasterDone, SpriteIcons, StyleUpdate,
    TileDone, TileOutcome, TileRequest, TileResult, TileWorker, Turn, Turns,
};

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
    pending: HashMap<u32, TileDone>,
    rasters: HashMap<u32, RasterDone>,
    /// Each raster's request, to send again if its worker is lost.
    raster_requests: HashMap<u32, GlyphRaster>,
    features: HashMap<u32, orb_runtime::FeaturesDone>,
    answers: HashMap<u32, orb_runtime::QueryDone>,
    /// The worker each request went to, until it answers, and a load's
    /// request, to send again if the worker is lost.
    sent_to: HashMap<u32, (u32, Option<TileRequest>)>,
    /// The turns of loads sent whose fetch hasn't ended.
    turns: HashMap<u32, Turn>,
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
    /// Sources whose every tile slices one document (GeoJSON), by key,
    /// as the style has them: their tiles go to one worker, which parses
    /// and indexes it once, its home in `whole_homes` once it has one.
    whole: HashSet<u32>,
    whole_homes: HashMap<u32, u32>,
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

    /// Replies to loads received so far.
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
        let (failed, loads, rasters, heads, turns) = {
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
            // Their turns pass on: the loads go again, in turns of their own.
            let mut turns = Vec::new();
            for id in ids {
                let Some((_, request)) = state.sent_to.remove(&id) else {
                    continue;
                };
                turns.extend(state.turns.remove(&id));
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
            (failed, loads, rasters, state.heads.clone(), turns)
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
        for id in failed {
            self.fail(id, message.clone());
        }
        for (request, done) in rasters {
            self.raster(request, done);
        }
        for (request, done) in loads {
            if self.quarantined(load_key(&request)) {
                done(Ok(quarantine(format!(
                    "tile {} is left out from now on: {message} loading it a second time",
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
        // Dropped once the state is let go: the next load's turn sends it.
        let turn = state.turns.remove(&id);
        if let JobReply::Fetched { .. } = message {
            drop(state);
            drop(turn);
            return Ok(());
        }
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
        let pending = state.pending.remove(&id);
        drop(state);
        drop(turn);
        let done = pending.ok_or_else(|| format!("reply to unknown job {id}"))?;
        let mut state = self.state.lock().expect("jobs poisoned");
        // A suspect answered: it didn't take its worker down.
        let answered = state.suspect == Some(id);
        if answered {
            state.suspect = None;
            if let Some((_, Some(request))) = &sent {
                state.lost_loads.remove(&load_key(request));
            }
        }
        drop(state);
        done(match message {
            JobReply::Missing { .. } => Ok(TileOutcome::Missing {
                warnings: Vec::new(),
            }),
            JobReply::Cancelled { .. } => Ok(TileOutcome::Cancelled),
            JobReply::Error { message, .. } => Err(message),
            JobReply::Tile { .. } | JobReply::Image { .. } | JobReply::Dem { .. } => {
                payload(message).map(TileOutcome::Built)
            }
            _ => Err("malformed job reply".into()),
        });
        if answered {
            self.next_suspect();
        }
        Ok(())
    }

    /// Fails request `id` with `message`.
    fn fail(&self, id: u32, message: String) {
        let _ = self.complete(JobReply::Error { id, message }, |_| {
            Err("malformed job reply".into())
        });
    }

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
            self.fail(id, "every job worker was lost".into());
            return;
        };
        state.sent_to.insert(id, (worker, load));
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
        for &(worker, _) in self.sent_to.values() {
            busy[worker as usize] += 1;
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
}

impl WorkerJobs {
    /// Sends a load to its tile's worker, in its `turn` (none for a
    /// relayout), which it holds until the worker's fetch ends.
    fn send_load(
        &self,
        mut request: TileRequest,
        done: TileDone,
        suspect: bool,
        turn: Option<Turn>,
    ) {
        request.rank = turn.as_ref().map_or(request.rank, Turn::rank);
        let (source, tile) = (request.source, request.data_tile());
        let (key, abort) = (load_key(&request), turn.as_ref().map(|t| t.abort().clone()));
        self.send(
            |state| state.home(source, tile),
            Some(request.clone()),
            |state, id| {
                state.pending.insert(id, done);
                if let Some(turn) = turn {
                    state.turns.insert(id, turn);
                }
                if suspect {
                    state.suspect = Some(id);
                }
            },
            |id| JobRequest::Load { id, request },
        );
        // A turn taken back (a load beside the view yielding to one of
        // the view's, or given up) stops the worker's fetch.
        if let Some(abort) = abort {
            let me = self.me.clone();
            abort.on_abort(move || {
                if let Some(jobs) = me.upgrade() {
                    jobs.stop_sent(key);
                }
            });
        }
    }

    /// Tells the worker a sent whole load of `key` went to that it's given
    /// up.
    fn stop_sent(&self, key: LoadKey) {
        let worker = {
            let state = self.state.lock().expect("jobs poisoned");
            state.sent_to.values().find_map(|(worker, request)| {
                let request = request.as_ref()?;
                (load_key(request) == key && request.nodes.is_none()).then_some(*worker)
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
            let done = Box::new(move |fetched: Result<Vec<u8>, orb_host::FetchError>| {
                let Some(jobs) = me.upgrade() else {
                    return;
                };
                let bytes = fetched.map_err(|e| format!("glyphs {url}: {e}"));
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
        if !request.loads_whole() {
            return self.send_load(request, done, suspect, None);
        }
        let (me, key, rank) = (self.me.clone(), load_key(&request), request.rank);
        let start: orb_runtime::Start = Box::new(move |turn| {
            let Some(turn) = turn else {
                return done(Ok(TileOutcome::Cancelled));
            };
            match me.upgrade() {
                Some(jobs) => jobs.send_load(request, done, suspect, Some(turn)),
                None => done(Ok(TileOutcome::Cancelled)),
            }
        });
        self.turns.wait(rank, Some(key), start);
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

#[cfg(test)]
mod tests;
