//! Tile sources: given the ideal cover of a view, decide
//! which tiles to load and what to draw while they load.
//!
//! A [`TilePyramid`] caches one source's tiles. Each frame,
//! [`TilePyramid::update`] takes the cover and returns the tiles to draw,
//! each with the regions it stands in for, and the tiles to request. A tile
//! that isn't ready yet retains loaded descendants, then fills uncovered
//! regions with its nearest ready ancestor. Replacements publish only
//! after they are ready, without discarding finer fallback coverage.

#![forbid(unsafe_code)]

pub mod endpoint;
mod gzip;
#[cfg(feature = "mbtiles")]
pub mod mbtiles;
pub mod pmtiles;

use std::collections::BTreeMap;

use rustc_hash::{FxHashMap, FxHashSet};

use orb_crs::{CoveredTile, TileIndex};

/// What to draw for one tile this frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draw {
    pub tile: TileIndex,
    pub wrap: i32,
    /// The areas this tile draws in, as tiles at its zoom or deeper. Just
    /// `tile` itself when it draws whole.
    pub regions: Vec<TileIndex>,
}

/// A frame's plan for one source ([`TilePyramid::plan`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Update {
    /// Tiles to draw, ancestors before descendants.
    pub draw: Vec<Draw>,
    /// Tiles to start loading, in cover order.
    pub request: Vec<TileIndex>,
    /// Tiles the plan uses, drawn or standing by: the cache keeps them.
    pub used: Vec<TileIndex>,
    /// The cover's tiles the source has (at or above its least zoom).
    pub cover: Vec<TileIndex>,
    /// Tiles kept loaded to stand in ([`TilePyramid::plan_keeping`]).
    pub kept: Vec<TileIndex>,
}

#[derive(Debug)]
enum State<T> {
    /// Requested, and wanted by the last plan or one before it that
    /// hasn't given it up: its result is drawn when it comes.
    Loading,
    /// Requested, then left by the view, its load given up
    /// ([`unwanted_loads`](TilePyramid::unwanted_loads)): its result
    /// isn't drawn unless a plan wants it again first, which makes it
    /// [`Loading`](Self::Loading) once more.
    GivenUp,
    /// Its result came, wanted, and waits its turn to be drawn (for room,
    /// say): no longer in flight.
    Arrived,
    Ready(T),
    /// Loaded with nothing to draw (HTTP 204, outside the source's
    /// bounds): counts as ready.
    Empty,
    /// Not in the tileset (HTTP 404): its nearest ancestor in the tileset
    /// stands in.
    Missing,
    /// Failed for a reason that may pass, such as the network: ancestors or
    /// children stand in, as for a loading tile, its nearest ancestor
    /// requested if none is loaded, and it is requested again
    /// once [`TilePyramid::retry_failed`] finds `retry_at` (ms) has passed.
    Failed {
        retry_at: f64,
    },
}

/// The first retry's delay after a failure; each further failure doubles
/// it, up to [`RETRY_MAX_MS`].
const RETRY_MS: f64 = 1000.0;
const RETRY_MAX_MS: f64 = 60_000.0;

/// How long to wait (ms) before retrying what failed `failures` times in
/// a row before: [`RETRY_MS`], doubling with each, up to [`RETRY_MAX_MS`].
pub fn retry_delay(failures: u32) -> f64 {
    (RETRY_MS * 2f64.powi(failures.min(16) as i32)).min(RETRY_MAX_MS)
}

/// `tile`'s place along the Z-order curve over the whole tileset, at
/// zoom 31: tiles that follow each other on it lie together, and a tile
/// comes before its descendants' places that follow.
fn z_order(tile: TileIndex) -> u64 {
    // Each of the 32 bits of `v` to every other bit of the result.
    fn spread(v: u32) -> u64 {
        let mut x = u64::from(v);
        x = (x | x << 16) & 0x0000_ffff_0000_ffff;
        x = (x | x << 8) & 0x00ff_00ff_00ff_00ff;
        x = (x | x << 4) & 0x0f0f_0f0f_0f0f_0f0f;
        x = (x | x << 2) & 0x3333_3333_3333_3333;
        (x | x << 1) & 0x5555_5555_5555_5555
    }
    let shift = 31u32.saturating_sub(u32::from(tile.z));
    spread(tile.col << shift) | spread(tile.row << shift) << 1
}

/// Whether `tile` is one of `cover`'s or beside one at its zoom, across
/// the antimeridian too where the world is `columns` tiles wide.
fn beside(cover: &FxHashSet<TileIndex>, tile: TileIndex, columns: Option<u32>) -> bool {
    (-1i64..=1).any(|dc| {
        let col = i64::from(tile.col) + dc;
        let col = match columns {
            Some(n) if n > 0 => col.rem_euclid(i64::from(n)),
            _ => col,
        };
        (-1i64..=1).any(|dr| {
            let row = i64::from(tile.row) + dr;
            let (Ok(col), Ok(row)) = (u32::try_from(col), u32::try_from(row)) else {
                return false;
            };
            cover.contains(&TileIndex::new(tile.z, col, row))
        })
    })
}

/// What stands in for a missing tile.
enum StandIn {
    Ready(TileIndex),
    Loading,
    Nothing,
}

impl<T> State<T> {
    /// Requested and not yet settled: in flight, or arrived.
    fn is_loading(&self) -> bool {
        matches!(self, Self::Loading | Self::GivenUp | Self::Arrived)
    }

    /// Requested and not yet answered, wanted or given up.
    fn in_flight(&self) -> bool {
        matches!(self, Self::Loading | Self::GivenUp)
    }
}

#[derive(Debug)]
struct Entry<T> {
    state: State<T>,
    last_used: u64,
}

/// What a load's answer for a tile finds in its pyramid
/// ([`TilePyramid::awaited`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Awaited {
    /// The plan wants it: it's drawn once inserted.
    Wanted,
    /// Its load was given up and no plan wanted it since: nothing draws
    /// it, so it needn't go to the GPU.
    GivenUp,
    /// No load of it is under way (it was drained, or answered already).
    No,
}

/// One source's tiles: loading, ready and cached.
#[derive(Debug)]
pub struct TilePyramid<T> {
    tiles: FxHashMap<TileIndex, Entry<T>>,
    min_zoom: u8,
    max_zoom: u8,
    /// Ready tiles kept beyond those in use.
    cache_size: usize,
    frame: u64,
    /// Failures in a row of tiles that last failed, for their backoff.
    failures: FxHashMap<TileIndex, u32>,
    /// The last committed plan's cover, and the tiles it kept to stand
    /// in.
    cover: FxHashSet<TileIndex>,
    kept: Vec<TileIndex>,
    /// Entries in flight, wanted or given up.
    loading: usize,
    /// Counts changes to the tiles held and their states.
    revision: u64,
}

impl<T> TilePyramid<T> {
    pub fn new(min_zoom: u8, max_zoom: u8, cache_size: usize) -> Self {
        Self {
            tiles: FxHashMap::default(),
            min_zoom,
            max_zoom,
            cache_size,
            frame: 0,
            failures: FxHashMap::default(),
            cover: FxHashSet::default(),
            kept: Vec::new(),
            loading: 0,
            revision: 0,
        }
    }

    /// Counts changes to the tiles held and their states: the same cover
    /// plans alike until it changes.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Settles loading `tile` in `state`; gives `state` back if `tile`
    /// isn't loading.
    fn settle(&mut self, tile: TileIndex, state: State<T>) -> Result<(), State<T>> {
        match self.tiles.get_mut(&tile) {
            Some(entry) if entry.state.is_loading() => {
                self.loading -= usize::from(entry.state.in_flight());
                entry.state = state;
                self.revision += 1;
                Ok(())
            }
            _ => Err(state),
        }
    }

    /// Marks wanted `tile`'s result as come: no longer in flight
    /// ([`loading`](Self::loading)), it waits to be inserted.
    pub fn arrived(&mut self, tile: TileIndex) {
        if let Some(entry) = self.tiles.get_mut(&tile)
            && matches!(entry.state, State::Loading)
        {
            entry.state = State::Arrived;
            self.loading -= 1;
        }
    }

    /// The least zoom the source has tiles at.
    pub fn min_zoom(&self) -> u8 {
        self.min_zoom
    }

    pub fn max_zoom(&self) -> u8 {
        self.max_zoom
    }

    pub fn get(&self, tile: TileIndex) -> Option<&T> {
        match &self.tiles.get(&tile)?.state {
            State::Ready(value) => Some(value),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, tile: TileIndex) -> Option<&mut T> {
        match &mut self.tiles.get_mut(&tile)?.state {
            State::Ready(value) => Some(value),
            _ => None,
        }
    }

    /// The ready tiles, to change in place.
    pub fn ready_mut(&mut self) -> impl Iterator<Item = (TileIndex, &mut T)> {
        self.tiles
            .iter_mut()
            .filter_map(|(&tile, entry)| match &mut entry.state {
                State::Ready(value) => Some((tile, value)),
                _ => None,
            })
    }

    /// The ready tiles, in no particular order.
    pub fn ready(&self) -> impl Iterator<Item = (TileIndex, &T)> {
        self.tiles
            .iter()
            .filter_map(|(&tile, entry)| match &entry.state {
                State::Ready(value) => Some((tile, value)),
                _ => None,
            })
    }

    /// Whether `tile`'s load is under way, wanted or given up.
    pub fn is_loading(&self, tile: TileIndex) -> bool {
        self.tiles.get(&tile).is_some_and(|e| e.state.is_loading())
    }

    /// What an answer to `tile`'s load finds: whether the plan still
    /// wants it.
    pub fn awaited(&self, tile: TileIndex) -> Awaited {
        match self.tiles.get(&tile).map(|e| &e.state) {
            Some(State::Loading | State::Arrived) => Awaited::Wanted,
            Some(State::GivenUp) => Awaited::GivenUp,
            _ => Awaited::No,
        }
    }

    /// Plans a frame and records it ([`plan`](Self::plan), then
    /// [`commit`](Self::commit)).
    pub fn update(&mut self, cover: &[CoveredTile]) -> Update {
        let update = self.plan(cover);
        self.commit(&update);
        update
    }

    /// What a frame draws for `cover`, the ideal tiles (deepest at most
    /// `max_zoom`; tiles below `min_zoom` are not drawn), from the tiles
    /// resident now, and what it needs loaded: a function of the cover
    /// and the pyramid alone.
    pub fn plan(&self, cover: &[CoveredTile]) -> Update {
        let mut update = Update::default();
        // Tiles this plan requests, loading from here on.
        let mut requested = FxHashSet::default();
        // (tile, wrap) → regions, ordered so ancestors draw first.
        let mut draws: BTreeMap<(u8, TileIndex, i32), Vec<TileIndex>> = BTreeMap::new();

        // Taken when an ideal tile isn't ready, which most frames have none of.
        let mut branches = None;

        for covered in cover {
            let ideal = covered.index;
            if ideal.z < self.min_zoom {
                continue;
            }
            debug_assert!(ideal.z <= self.max_zoom, "cover deeper than the source");
            update.used.push(ideal);
            update.cover.push(ideal);
            match self.tiles.get(&ideal).map(|e| &e.state) {
                None => {
                    if requested.insert(ideal) {
                        update.request.push(ideal);
                    }
                }
                Some(State::Ready(_)) => {
                    draws.insert((ideal.z, ideal, covered.wrap), vec![ideal]);
                    continue;
                }
                Some(State::Empty) => continue,
                Some(State::Loading | State::GivenUp | State::Arrived | State::Failed { .. }) => {}
                // A missing tile's nearest ancestor in the tileset stands
                // in for it, whatever else is resident: other tiles stand
                // in only while that ancestor loads.
                Some(State::Missing) => match self.stand_in(ideal, &mut requested, &mut update) {
                    StandIn::Ready(ancestor) => {
                        draws
                            .entry((ancestor.z, ancestor, covered.wrap))
                            .or_default()
                            .push(ideal);
                        continue;
                    }
                    StandIn::Nothing => continue,
                    StandIn::Loading => {}
                },
            }

            let failed = matches!(
                self.tiles.get(&ideal).map(|e| &e.state),
                Some(State::Failed { .. })
            );
            let mut descendants = Vec::new();
            let mut missing = Vec::new();
            let branches = branches.get_or_insert_with(|| self.branches());
            self.descendant_coverage(ideal, branches, &mut descendants, &mut missing);
            for tile in descendants {
                update.used.push(tile);
                if self.is_ready(tile) {
                    draws.insert((tile.z, tile, covered.wrap), vec![tile]);
                }
            }
            if missing.is_empty() {
                continue;
            }
            match self.ready_ancestor(ideal) {
                Some(ancestor) => {
                    update.used.push(ancestor);
                    draws
                        .entry((ancestor.z, ancestor, covered.wrap))
                        .or_default()
                        .extend(missing);
                }
                // A failed tile with no ancestor loaded asks for its
                // nearest one, as GL JS loads errored tiles' parents: the
                // region shows coarser data while the tile waits for its
                // retry, not the background.
                None if failed => {
                    self.stand_in(ideal, &mut requested, &mut update);
                }
                None => {}
            }
        }

        update.draw = draws
            .into_iter()
            .map(|((_, tile, wrap), mut regions)| {
                let regions = if regions.contains(&tile) {
                    vec![tile]
                } else {
                    // In Z order, not the cover's: a placement clips to
                    // four regions, and draws over their box (an image
                    // tile's whole mesh shades it), so its four lie
                    // together; nor does their grouping depend on the
                    // order the view's tiles came in.
                    regions.sort_unstable_by_key(|&r| z_order(r));
                    regions
                };
                Draw {
                    tile,
                    wrap,
                    regions,
                }
            })
            .collect();
        update
    }

    /// As [`plan`](Self::plan), keeping `kept` loaded too, requested ahead
    /// of the cover's tiles: coarse tiles that stand in, as any loaded
    /// ancestor does, wherever the cover's own haven't loaded. Nothing
    /// draws them otherwise.
    pub fn plan_keeping(&self, cover: &[CoveredTile], kept: &[TileIndex]) -> Update {
        let mut update = self.plan(cover);
        let mut request = Vec::new();
        for &tile in kept {
            if tile.z < self.min_zoom || tile.z > self.max_zoom {
                continue;
            }
            update.used.push(tile);
            update.kept.push(tile);
            if !self.tiles.contains_key(&tile) && !update.request.contains(&tile) {
                request.push(tile);
            }
        }
        if !request.is_empty() {
            request.append(&mut update.request);
            update.request = request;
        }
        update
    }

    /// Records a frame's plan: its requests load from now on, and the tiles
    /// it used are the cache's most recent.
    pub fn commit(&mut self, update: &Update) {
        self.frame += 1;
        let frame = self.frame;
        self.cover.clear();
        self.cover.extend(&update.cover);
        self.kept.clone_from(&update.kept);
        for &tile in &update.request {
            self.tiles.entry(tile).or_insert_with(|| {
                self.loading += 1;
                self.revision += 1;
                Entry {
                    state: State::Loading,
                    last_used: frame,
                }
            });
        }
        for tile in &update.used {
            if let Some(entry) = self.tiles.get_mut(tile) {
                entry.last_used = frame;
                // Wanted again before its given-up load answered.
                if matches!(entry.state, State::GivenUp) {
                    entry.state = State::Loading;
                    self.revision += 1;
                }
            }
        }
    }

    /// Wanted loads the last update didn't use, given up, each once:
    /// nothing draws them or waits for them now, so their loads may be
    /// cancelled, as GL JS aborts the requests of tiles it drops. With
    /// `beside`, a tile beside one of the cover's at its zoom loads on, as
    /// MapLibre Native keeps it: a view turned back and forth brings its
    /// corners back before their loads would come again, for a caller that
    /// keeps what such a load brings. `beside` gives the world's width in
    /// columns at a zoom, for neighbours across the antimeridian (`None`
    /// on a plane that doesn't wrap). One a later update wants again loads
    /// on until its result comes back; a cancelled one is then forgotten
    /// ([`forget_loading`](Self::forget_loading)) and requested anew.
    pub fn unwanted_loads(&mut self, beside: Option<&dyn Fn(u8) -> Option<u32>>) -> Vec<TileIndex> {
        let frame = self.frame;
        let cover = std::mem::take(&mut self.cover);
        let near = |tile: TileIndex| {
            beside.is_some_and(|columns| self::beside(&cover, tile, columns(tile.z)))
        };
        let given_up = self.give_up(|tile, e| e.last_used != frame && !near(tile));
        self.cover = cover;
        given_up
    }

    /// Whether the last update used `tile`: loads that load on beside the
    /// view ([`unwanted_loads`](Self::unwanted_loads)) it didn't.
    pub fn used_last(&self, tile: TileIndex) -> bool {
        self.tiles
            .get(&tile)
            .is_some_and(|e| e.last_used == self.frame)
    }

    /// Every wanted load, given up, as
    /// [`unwanted_loads`](Self::unwanted_loads) gives up those out of
    /// view: for a caller that drops the pyramid, or stops drawing.
    pub fn give_up_loads(&mut self) -> Vec<TileIndex> {
        self.give_up(|_, _| true)
    }

    /// Gives up the wanted loads `which` picks.
    fn give_up(&mut self, which: impl Fn(TileIndex, &Entry<T>) -> bool) -> Vec<TileIndex> {
        let given_up: Vec<TileIndex> = (self.tiles.iter_mut())
            .filter(|(tile, e)| matches!(e.state, State::Loading) && which(**tile, e))
            .map(|(&tile, e)| {
                e.state = State::GivenUp;
                tile
            })
            .collect();
        self.revision += u64::from(!given_up.is_empty());
        given_up
    }

    /// Counts `tile` as used by the last update, as if its plan had used
    /// it: no eviction takes it before the next.
    pub fn keep(&mut self, tile: TileIndex) {
        if let Some(entry) = self.tiles.get_mut(&tile) {
            entry.last_used = self.frame;
        }
    }

    /// Stores a loaded tile. Absent or already completed entries return
    /// their value back. After a drain, callers must reject old-generation
    /// results before they can reach newly requested entries.
    pub fn insert(&mut self, tile: TileIndex, value: T) -> Option<T> {
        match self.settle(tile, State::Ready(value)) {
            Ok(()) => {
                self.failures.remove(&tile);
                None
            }
            Err(State::Ready(value)) => Some(value),
            Err(_) => unreachable!("given back as given"),
        }
    }

    /// Marks a loading tile as loaded with nothing to draw.
    pub fn insert_empty(&mut self, tile: TileIndex) {
        if self.settle(tile, State::Empty).is_ok() {
            self.failures.remove(&tile);
        }
    }

    /// Marks a loading tile as failed at `now` (ms) for a reason that may
    /// pass: ancestors or children stand in until it's retried, after a
    /// delay that doubles with each failure in a row.
    pub fn insert_failed(&mut self, tile: TileIndex, now: f64) {
        if !self.is_loading(tile) {
            return;
        }
        let failures = self.failures.entry(tile).or_default();
        let delay = retry_delay(*failures);
        *failures = failures.saturating_add(1);
        let retry_at = now + delay;
        let _ = self.settle(tile, State::Failed { retry_at });
    }

    /// Forgets failed tiles due for a retry at `now` (ms), so the next
    /// [`update`](Self::update) requests those it covers again; every
    /// failed tile with `now` infinite, as when the network returns.
    /// Whether any failed tile remains.
    pub fn retry_failed(&mut self, now: f64) -> bool {
        let (mut waiting, before) = (false, self.tiles.len());
        self.tiles.retain(|_, entry| match entry.state {
            State::Failed { retry_at } if retry_at <= now => false,
            State::Failed { .. } => {
                waiting = true;
                true
            }
            _ => true,
        });
        self.revision += u64::from(self.tiles.len() != before);
        waiting
    }

    /// Forgets the settled tiles `which` picks that hold nothing (empty,
    /// missing or failed), as if they had never been requested: the next
    /// plan that needs one requests it again.
    pub fn forget_settled(&mut self, which: &dyn Fn(TileIndex) -> bool) {
        let before = self.tiles.len();
        self.tiles.retain(|&tile, entry| {
            !matches!(
                entry.state,
                State::Empty | State::Missing | State::Failed { .. }
            ) || !which(tile)
        });
        self.revision += u64::from(self.tiles.len() != before);
    }

    /// The tiles whose loads are under way, wanted or given up.
    pub fn loading_tiles(&self) -> impl Iterator<Item = TileIndex> + '_ {
        self.tiles
            .iter()
            .filter(|(_, entry)| entry.state.in_flight())
            .map(|(&tile, _)| tile)
    }

    /// Forgets a loading tile whose load completed unused, as if it had
    /// never been requested: a later plan that needs it requests it again.
    pub fn forget_loading(&mut self, tile: TileIndex) {
        if self.is_loading(tile) {
            let entry = self.tiles.remove(&tile).expect("loading");
            self.loading -= usize::from(entry.state.in_flight());
            self.revision += 1;
        }
    }

    /// Marks a loading tile as absent from the tileset: its nearest
    /// ancestor in the tileset stands in, loaded if need be; while that
    /// loads, ready ancestors and descendants stand in as for a loading
    /// tile.
    pub fn insert_missing(&mut self, tile: TileIndex) {
        if self.settle(tile, State::Missing).is_ok() {
            self.failures.remove(&tile);
        }
    }

    /// Evicts ready and empty tiles not used in the last update, least
    /// recently used first, until at most `cache_size` unused ones remain.
    /// In-flight entries remain until completion: eviction does not cancel
    /// host jobs, and forgetting them would allow duplicate requests whose
    /// out-of-order results could complete the wrong entry. Loading entries
    /// do not consume the ready cache budget. Returns the evicted values.
    pub fn evict(&mut self) -> Vec<T> {
        let frame = self.frame;
        let mut unused: Vec<(u64, TileIndex)> = self
            .tiles
            .iter()
            .filter(|(_, e)| e.last_used != frame && !e.state.is_loading())
            .map(|(&t, e)| (e.last_used, t))
            .collect();
        if unused.len() <= self.cache_size {
            return Vec::new();
        }
        unused.sort_unstable();
        let excess = unused.len() - self.cache_size;
        self.revision += 1;
        unused[..excess]
            .iter()
            .filter_map(|(_, t)| {
                self.failures.remove(t);
                match self.tiles.remove(t)?.state {
                    State::Ready(value) => Some(value),
                    _ => None,
                }
            })
            .collect()
    }

    /// How many tiles the last update used, in any state: those the view
    /// covers, and those standing in for its tiles.
    pub fn used(&self) -> usize {
        let frame = self.frame;
        (self.tiles.values())
            .filter(|entry| entry.last_used == frame)
            .count()
    }

    /// The ready tiles the last update didn't use, with how many updates
    /// ago each was last used: comparable across pyramids updated together.
    pub fn unused(&self) -> impl Iterator<Item = (TileIndex, u64, &T)> {
        let frame = self.frame;
        self.tiles
            .iter()
            .filter_map(move |(&tile, entry)| match &entry.state {
                State::Ready(value) if entry.last_used != frame => {
                    Some((tile, frame - entry.last_used, value))
                }
                _ => None,
            })
    }

    /// Removes `tile` if it's ready, used or not.
    pub fn remove(&mut self, tile: TileIndex) -> Option<T> {
        self.is_ready(tile).then(|| self.take_ready(tile))
    }

    /// Evicts `tile` if it's ready and the last update didn't use it.
    pub fn evict_unused(&mut self, tile: TileIndex) -> Option<T> {
        let unused = self.tiles.get(&tile)?.last_used != self.frame;
        (unused && self.is_ready(tile)).then(|| self.take_ready(tile))
    }

    /// Removes `tile`, which is ready, and gives its value.
    fn take_ready(&mut self, tile: TileIndex) -> T {
        self.revision += 1;
        match self.tiles.remove(&tile).map(|e| e.state) {
            Some(State::Ready(value)) => value,
            _ => unreachable!("a ready tile"),
        }
    }

    /// Evicts the least recently used ready tile that the last update didn't
    /// use, to make room for a new one. Returns `None` when every ready tile
    /// is in use.
    pub fn evict_lru(&mut self) -> Option<T> {
        let frame = self.frame;
        let (&tile, _) = self
            .tiles
            .iter()
            .filter(|(_, e)| e.last_used != frame && matches!(e.state, State::Ready(_)))
            .min_by_key(|(_, e)| e.last_used)?;
        Some(self.take_ready(tile))
    }

    /// Evicts a ready tile inside `tile`'s area. While `tile` loads, its
    /// ready descendants stand in for it; once it's ready they draw nothing,
    /// so they're the first to make room for it.
    pub fn evict_within(&mut self, tile: TileIndex) -> Option<T> {
        let (&found, _) = self.tiles.iter().find(|(t, e)| {
            t.z > tile.z && matches!(e.state, State::Ready(_)) && t.is_within(tile)
        })?;
        Some(self.take_ready(found))
    }

    /// Whether a tile of the last committed plan's cover that `which`
    /// picks is still loading.
    pub fn cover_loading(&self, which: impl Fn(TileIndex) -> bool) -> bool {
        self.cover.iter().any(|&t| which(t) && self.is_loading(t))
    }

    /// Whether the last committed plan's cover holds `tile`.
    pub fn covers(&self, tile: TileIndex) -> bool {
        self.cover.contains(&tile)
    }

    /// Whether the last plan wants `tile`: a tile of its cover, or the
    /// ancestor of one that failed (even while it's retried) or is
    /// missing, which it asks for to stand in.
    pub fn wants(&self, tile: TileIndex) -> bool {
        self.cover.contains(&tile)
            || self.cover.iter().any(|&t| {
                t.z > tile.z
                    && t.is_within(tile)
                    && (self.failures.contains_key(&t)
                        || matches!(
                            self.tiles.get(&t).map(|e| &e.state),
                            Some(State::Failed { .. } | State::Missing)
                        ))
            })
    }

    /// Evicts a ready tile that only stands in: one the last plan's cover
    /// doesn't hold or keep, standing in for no missing tile. A fresh view
    /// of the cover would never have loaded it, so it makes room for the
    /// cover's tiles; `near`'s ancestors, which draw in its place, go
    /// first.
    pub fn evict_stand_in(&mut self, near: Option<TileIndex>) -> Option<T> {
        let stands_in = |tile: TileIndex| {
            self.is_ready(tile)
                && !self.cover.contains(&tile)
                && !self.kept.contains(&tile)
                && !self
                    .tiles
                    .iter()
                    .any(|(&t, e)| matches!(e.state, State::Missing) && t.is_within(tile))
        };
        let ancestors = std::iter::successors(near, |t| t.parent());
        let found = ancestors
            .skip(1)
            .take_while(|t| t.z >= self.min_zoom)
            .find(|&t| stands_in(t))
            .or_else(|| self.tiles.keys().copied().find(|&t| stands_in(t)))?;
        Some(self.take_ready(found))
    }

    /// Forgets every tile, returning the ready values.
    pub fn drain(&mut self) -> Vec<T> {
        self.failures.clear();
        self.loading = 0;
        self.revision += 1;
        self.tiles
            .drain()
            .filter_map(|(_, entry)| match entry.state {
                State::Ready(value) => Some(value),
                _ => None,
            })
            .collect()
    }

    /// Tiles requested and not yet answered, wanted or given up.
    pub fn loading(&self) -> usize {
        self.loading
    }

    /// [`loading`](Self::loading), but for loads beside the view that
    /// the last update didn't use ([`unwanted_loads`](Self::unwanted_loads)):
    /// what the view waits for.
    pub fn loading_for_view(&self) -> usize {
        let beside = (self.tiles.values())
            .filter(|e| matches!(e.state, State::Loading) && e.last_used != self.frame)
            .count();
        self.loading - beside
    }

    fn is_ready(&self, tile: TileIndex) -> bool {
        matches!(
            self.tiles.get(&tile),
            Some(Entry {
                state: State::Ready(_),
                ..
            })
        )
    }

    /// The tiles with a known descendant, the only branches worth
    /// descending: walking the cached tiles' ancestors bounds this by cache
    /// size × depth, not 4^depth.
    fn branches(&self) -> FxHashSet<TileIndex> {
        let mut branches = FxHashSet::default();
        let known = (self.tiles.iter()).filter(|(_, e)| !e.state.is_loading());
        for (&tile, _) in known {
            let ancestors = std::iter::successors(tile.parent(), |t| t.parent());
            for parent in ancestors.take_while(|t| t.z >= self.min_zoom) {
                branches.insert(parent);
            }
        }
        branches
    }

    /// Partitions a missing ideal into loaded descendants and uncovered
    /// regions. Empty descendants cover their region without drawing it.
    fn descendant_coverage(
        &self,
        tile: TileIndex,
        branches: &FxHashSet<TileIndex>,
        ready: &mut Vec<TileIndex>,
        missing: &mut Vec<TileIndex>,
    ) {
        match self.tiles.get(&tile).map(|entry| &entry.state) {
            Some(State::Ready(_) | State::Empty) => ready.push(tile),
            _ if branches.contains(&tile) => {
                for child in tile.children() {
                    self.descendant_coverage(child, branches, ready, missing);
                }
            }
            _ => missing.push(tile),
        }
    }

    /// The nearest ancestor of a missing or failed tile that is neither,
    /// as GL JS loads the parents of tiles that failed: requested when not
    /// yet known. Ancestors on the way stay in use.
    fn stand_in(
        &self,
        mut tile: TileIndex,
        requested: &mut FxHashSet<TileIndex>,
        update: &mut Update,
    ) -> StandIn {
        while let Some(parent) = tile.parent() {
            if parent.z < self.min_zoom {
                break;
            }
            update.used.push(parent);
            match self.tiles.get(&parent).map(|e| &e.state) {
                None => {
                    if requested.insert(parent) {
                        update.request.push(parent);
                    }
                    return StandIn::Loading;
                }
                Some(State::Missing | State::Failed { .. }) => tile = parent,
                Some(State::Loading | State::GivenUp | State::Arrived) => {
                    return StandIn::Loading;
                }
                Some(State::Ready(_)) => return StandIn::Ready(parent),
                Some(State::Empty) => break,
            }
        }
        StandIn::Nothing
    }

    fn ready_ancestor(&self, mut tile: TileIndex) -> Option<TileIndex> {
        while let Some(parent) = tile.parent() {
            if parent.z < self.min_zoom {
                return None;
            }
            if self.is_ready(parent) {
                return Some(parent);
            }
            tile = parent;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cover(tiles: &[(u8, u32, u32)]) -> Vec<CoveredTile> {
        tiles
            .iter()
            .map(|&(z, col, row)| CoveredTile {
                index: TileIndex::new(z, col, row),
                wrap: 0,
            })
            .collect()
    }

    fn t(z: u8, col: u32, row: u32) -> TileIndex {
        TileIndex::new(z, col, row)
    }

    #[test]
    fn requests_then_draws_ready_tiles() {
        let mut p = TilePyramid::new(0, 14, 8);
        let view = cover(&[(1, 0, 0), (1, 1, 0)]);
        let first = p.update(&view);
        assert_eq!(first.request, [t(1, 0, 0), t(1, 1, 0)]);
        assert!(first.draw.is_empty());

        assert_eq!(p.insert(t(1, 0, 0), "a"), None);
        let second = p.update(&view);
        assert!(
            second.request.is_empty(),
            "loading tiles aren't requested twice"
        );
        assert_eq!(
            second.draw,
            [Draw {
                tile: t(1, 0, 0),
                wrap: 0,
                regions: vec![t(1, 0, 0)]
            }]
        );
    }

    /// Tiles kept load ahead of the cover's and stay, drawn only where a
    /// cover tile hasn't loaded: a view whose tiles are in looks the same
    /// with them or without.
    #[test]
    fn kept_tiles_load_first_and_draw_only_as_stand_ins() {
        let mut p = TilePyramid::new(0, 14, 0);
        let (view, kept) = (cover(&[(3, 6, 2), (3, 7, 2)]), [t(0, 0, 0)]);
        let plan = |p: &mut TilePyramid<&str>| {
            let update = p.plan_keeping(&view, &kept);
            p.commit(&update);
            update
        };
        assert_eq!(plan(&mut p).request, [t(0, 0, 0), t(3, 6, 2), t(3, 7, 2)]);
        p.insert(t(0, 0, 0), "world");
        p.insert(t(3, 6, 2), "tile");
        let standing_in = plan(&mut p);
        assert!(standing_in.request.is_empty());
        let drawn: Vec<_> = (standing_in.draw.iter())
            .map(|d| (d.tile, d.regions.clone()))
            .collect();
        assert_eq!(
            drawn,
            [
                (t(0, 0, 0), vec![t(3, 7, 2)]),
                (t(3, 6, 2), vec![t(3, 6, 2)])
            ]
        );
        p.insert(t(3, 7, 2), "tile");
        let loaded = plan(&mut p);
        assert!(loaded.draw.iter().all(|d| d.tile.z == 3));
        assert_eq!(loaded.draw, p.plan(&view).draw);
        // Kept, the cache holds it even with no room past what's used,
        // and it never makes room for the cover's tiles.
        p.evict();
        assert!(p.evict_stand_in(None).is_none());
        assert!(p.get(t(0, 0, 0)).is_some());
    }

    /// An ancestor's regions come in Z order, whatever order the cover
    /// gave its tiles in: each run of four lies together.
    #[test]
    fn regions_come_in_z_order() {
        let mut p = TilePyramid::new(0, 22, 64);
        p.update(&cover(&[(1, 0, 0)]));
        p.insert(t(1, 0, 0), "ancestor");
        let mut ideals: Vec<(u8, u32, u32)> = (0..4)
            .flat_map(|c| (0..4).map(move |r| (3, c, r)))
            .collect();
        // Outward from the middle, as `focus_first` orders a cover.
        ideals.sort_by_key(|&(_, c, r)| (c.abs_diff(2) + r.abs_diff(2), c, r));
        let update = p.update(&cover(&ideals));
        let draw = &update.draw[0];
        assert_eq!(draw.tile, t(1, 0, 0));
        let mut sorted = draw.regions.clone();
        sorted.sort_by_key(|&r| z_order(r));
        assert_eq!(draw.regions, sorted);
        // The first four are one quadrant's.
        assert!(draw.regions[..4].iter().all(|r| r.is_within(t(2, 0, 0))));
    }

    #[test]
    fn ready_parent_stands_in_for_missing_children() {
        let mut p = TilePyramid::new(0, 14, 8);
        p.update(&cover(&[(1, 0, 0)]));
        p.insert(t(1, 0, 0), "parent");

        // Zoom in: two of the parent's children are ideal, one is ready.
        let view = cover(&[(2, 0, 0), (2, 1, 0)]);
        p.update(&view);
        p.insert(t(2, 0, 0), "child");
        let update = p.update(&view);
        assert_eq!(
            update.draw,
            [
                Draw {
                    tile: t(1, 0, 0),
                    wrap: 0,
                    regions: vec![t(2, 1, 0)]
                },
                Draw {
                    tile: t(2, 0, 0),
                    wrap: 0,
                    regions: vec![t(2, 0, 0)]
                },
            ]
        );
    }

    #[test]
    fn ready_children_stand_in_for_a_missing_parent() {
        let mut p = TilePyramid::new(0, 14, 8);
        p.update(&cover(&[(2, 0, 0), (2, 1, 1)]));
        p.insert(t(2, 0, 0), "a");
        p.insert(t(2, 1, 1), "b");

        // Zoom out: the parent isn't loaded yet.
        let update = p.update(&cover(&[(1, 0, 0)]));
        assert_eq!(update.request, [t(1, 0, 0)]);
        let drawn: Vec<_> = update.draw.iter().map(|d| d.tile).collect();
        assert_eq!(drawn, [t(2, 0, 0), t(2, 1, 1)]);
    }

    // Record the complete cover and draw set at each scripted zoom step.
    // A delayed coarse tile must not replace finer, already visible fallback.
    #[test]
    fn zoom_out_retains_descendants_when_a_late_ancestor_arrives() {
        let mut p = TilePyramid::new(0, 14, 0);
        p.update(&cover(&[(0, 0, 0)])); // slow ancestor request
        let fine = cover(&[(3, 0, 0), (3, 1, 0)]);
        p.update(&fine);
        p.insert(t(3, 0, 0), "a");
        p.insert(t(3, 1, 0), "b");
        let mut frames = vec![(fine.clone(), p.update(&fine).draw)];
        let middle = cover(&[(2, 0, 0)]);
        frames.push((middle.clone(), p.update(&middle).draw));
        // This result arrives between frames of the same zoom-out gesture.
        p.insert(t(0, 0, 0), "late ancestor");
        frames.push((middle.clone(), p.update(&middle).draw));
        assert!(
            frames[2].1.iter().any(|d| d.tile == t(3, 0, 0)),
            "{frames:#?}"
        );
        assert!(
            frames[2].1.iter().any(|d| d.tile == t(3, 1, 0)),
            "{frames:#?}"
        );
        let coarse = cover(&[(1, 0, 0)]);
        frames.push((coarse.clone(), p.update(&coarse).draw));
        p.evict();
        assert!(
            frames[3].1.iter().any(|d| d.tile == t(3, 0, 0)),
            "{frames:#?}"
        );
        assert_eq!(p.get(t(3, 0, 0)), Some(&"a"));
        // Publish the requested replacement, then zoom in again. Each ready
        // ideal replaces its fallbacks; missing ideals keep their coverage.
        p.insert(t(1, 0, 0), "replacement");
        assert_eq!(p.update(&coarse).draw[0].tile, t(1, 0, 0));
        assert!(p.update(&middle).draw.iter().any(|d| d.tile == t(3, 0, 0)));
        assert_eq!(p.update(&fine).draw, frames[0].1);
    }

    #[test]
    fn ready_ideal_replaces_descendants_and_makes_them_evictable() {
        let mut p = TilePyramid::new(0, 14, 0);
        let children = cover(&[(13, 7274, 3224), (13, 7275, 3225)]);
        p.update(&children);
        p.insert(t(13, 7274, 3224), "first child");
        p.insert(t(13, 7275, 3225), "second child");
        let ideal = t(12, 3637, 1612);
        let view = cover(&[(ideal.z, ideal.col, ideal.row)]);
        assert_eq!(p.update(&view).draw.len(), 2);
        assert!(p.evict().is_empty(), "fallback children stay while loading");
        p.insert(ideal, "ideal");
        assert_eq!(
            p.update(&view).draw,
            [Draw {
                tile: ideal,
                wrap: 0,
                regions: vec![ideal],
            }]
        );
        let mut evicted = p.evict();
        evicted.sort_unstable();
        assert_eq!(evicted, ["first child", "second child"]);
        assert_eq!(p.get(ideal), Some(&"ideal"));
    }

    #[test]
    fn zoom_out_two_levels_keeps_coverage_until_the_ideal_arrives() {
        let mut p = TilePyramid::new(0, 14, 0);
        p.update(&cover(&[(3, 0, 0)]));
        p.insert(t(3, 0, 0), "fine");
        let coarse = cover(&[(1, 0, 0)]);
        let waiting = p.update(&coarse);
        assert_eq!(
            waiting.draw,
            [Draw {
                tile: t(3, 0, 0),
                wrap: 0,
                regions: vec![t(3, 0, 0)]
            }]
        );
        p.evict();
        p.insert(t(1, 0, 0), "coarse");
        assert_eq!(p.update(&coarse).draw[0].tile, t(1, 0, 0));
    }

    #[test]
    fn descendant_and_ancestor_regions_partition_each_wrap_without_overlap() {
        let mut p = TilePyramid::new(0, 14, 0);
        p.update(&cover(&[(0, 0, 0)]));
        p.insert(t(0, 0, 0), "ancestor");
        p.update(&cover(&[(3, 0, 0), (3, 3, 3)]));
        p.insert(t(3, 0, 0), "fine");
        p.insert_empty(t(3, 3, 3));
        let mut view = cover(&[(1, 0, 0)]);
        view.push(CoveredTile {
            index: t(1, 0, 0),
            wrap: 1,
        });
        for _ in 0..2 {
            let update = p.update(&view);
            for wrap in [0, 1] {
                for col in 0..4 {
                    for row in 0..4 {
                        let count = update
                            .draw
                            .iter()
                            .filter(|d| d.wrap == wrap)
                            .flat_map(|d| &d.regions)
                            .filter(|r| {
                                let depth = 3 - r.z;
                                col >> depth == r.col && row >> depth == r.row
                            })
                            .count();
                        assert_eq!(count, usize::from((col, row) != (3, 3)), "{update:?}");
                    }
                }
            }
            // Empty fallback coverage must be retained too, or the ancestor
            // fills its region on the next frame and makes old data flash.
            p.evict();
        }
    }

    #[test]
    fn empty_tiles_draw_nothing_and_need_no_fallback() {
        let mut p = TilePyramid::new(0, 14, 8);
        p.update(&cover(&[(0, 0, 0)]));
        p.insert(t(0, 0, 0), "world");
        p.update(&cover(&[(1, 0, 0)]));
        p.insert_empty(t(1, 0, 0));
        assert!(p.update(&cover(&[(1, 0, 0)])).draw.is_empty());
    }

    #[test]
    fn failed_tiles_fall_back_and_retry_with_backoff() {
        let mut p = TilePyramid::new(0, 14, 8);
        p.update(&cover(&[(0, 0, 0)]));
        p.insert(t(0, 0, 0), "world");
        p.update(&cover(&[(1, 0, 0)]));
        p.insert_failed(t(1, 0, 0), 0.0);
        assert_eq!(p.loading(), 0, "a failed tile doesn't keep the map loading");
        let update = p.update(&cover(&[(1, 0, 0)]));
        assert_eq!(update.draw[0].tile, t(0, 0, 0), "the ancestor stands in");
        assert!(update.request.is_empty());
        assert!(p.retry_failed(999.0), "not due before a second");
        assert!(p.update(&cover(&[(1, 0, 0)])).request.is_empty());
        assert!(!p.retry_failed(1000.0));
        assert_eq!(p.update(&cover(&[(1, 0, 0)])).request, [t(1, 0, 0)]);
        // The second failure waits twice as long; the network's return
        // retries at once.
        p.insert_failed(t(1, 0, 0), 1000.0);
        assert!(p.retry_failed(2999.0));
        assert!(!p.retry_failed(f64::INFINITY));
        assert_eq!(p.update(&cover(&[(1, 0, 0)])).request, [t(1, 0, 0)]);
        p.insert(t(1, 0, 0), "tile");
        assert!(p.failures.is_empty());
    }

    #[test]
    fn missing_tiles_load_their_ancestors() {
        let mut pyramid = TilePyramid::new(0, 4, 8);
        pyramid.update(&cover(&[(2, 0, 0)]));
        pyramid.insert_missing(t(2, 0, 0));
        let update = pyramid.update(&cover(&[(2, 0, 0)]));
        assert_eq!(update.request, [t(1, 0, 0)]);
        pyramid.insert_missing(t(1, 0, 0));
        assert_eq!(pyramid.update(&cover(&[(2, 0, 0)])).request, [t(0, 0, 0)]);
        pyramid.insert(t(0, 0, 0), ());
        let update = pyramid.update(&cover(&[(2, 0, 0)]));
        assert!(update.request.is_empty());
        assert_eq!(update.draw[0].tile, t(0, 0, 0));
        assert!(pyramid.evict().is_empty());
        assert!(pyramid.get(t(0, 0, 0)).is_some());
    }

    #[test]
    fn missing_tiles_fall_back_without_reloading() {
        let mut pyramid = TilePyramid::new(0, 4, 8);
        pyramid.update(&cover(&[(0, 0, 0)]));
        pyramid.insert(t(0, 0, 0), ());
        pyramid.update(&cover(&[(1, 0, 0)]));
        pyramid.insert_missing(t(1, 0, 0));
        let update = pyramid.update(&cover(&[(1, 0, 0)]));
        assert!(update.request.is_empty());
        assert_eq!(update.draw.len(), 1);
        assert_eq!(update.draw[0].tile, t(0, 0, 0));
        assert_eq!(update.draw[0].regions, vec![t(1, 0, 0)]);
        assert_eq!(pyramid.loading(), 0);
    }

    /// A missing tile's stand-in doesn't depend on what else is resident:
    /// its nearest ancestor in the tileset, not a coarser one cached
    /// earlier, nor descendants.
    #[test]
    fn missing_tiles_draw_their_nearest_ancestor_whatever_is_cached() {
        let mut pyramid = TilePyramid::new(0, 4, 8);
        pyramid.update(&cover(&[(0, 0, 0)]));
        pyramid.insert(t(0, 0, 0), ());
        pyramid.update(&cover(&[(3, 0, 0)]));
        pyramid.insert(t(3, 0, 0), ());
        pyramid.update(&cover(&[(2, 0, 0)]));
        pyramid.insert_missing(t(2, 0, 0));
        let update = pyramid.update(&cover(&[(2, 0, 0)]));
        assert_eq!(update.request, [t(1, 0, 0)]);
        pyramid.insert(t(1, 0, 0), ());
        let update = pyramid.update(&cover(&[(2, 0, 0)]));
        assert!(update.request.is_empty());
        assert_eq!(update.draw.len(), 1);
        assert_eq!(update.draw[0].tile, t(1, 0, 0));
        assert_eq!(update.draw[0].regions, vec![t(2, 0, 0)]);
    }

    #[test]
    fn evicts_least_recently_used_beyond_the_cache() {
        let mut p = TilePyramid::new(0, 14, 1);
        for col in 0..3 {
            p.update(&cover(&[(2, col, 0)]));
            p.insert(t(2, col, 0), col);
        }
        p.update(&cover(&[(2, 3, 0)]));
        // Three unused ready tiles, cache of one: the two oldest go.
        let mut evicted = p.evict();
        evicted.sort_unstable();
        assert_eq!(evicted, [0, 1]);
        assert!(p.get(t(2, 2, 0)).is_some());
        assert!(p.is_loading(t(2, 3, 0)));
    }

    #[test]
    fn kept_tiles_outlast_the_update_that_dropped_them() {
        let mut p = TilePyramid::new(0, 14, 0);
        for col in 0..2 {
            p.update(&cover(&[(2, col, 0)]));
            p.insert(t(2, col, 0), col);
        }
        p.update(&cover(&[(2, 2, 0)]));
        p.keep(t(2, 1, 0));
        assert_eq!(p.evict_lru(), Some(0));
        assert_eq!(p.evict_lru(), None, "kept until the next update");
        assert!(p.evict().is_empty());
        p.update(&cover(&[(2, 2, 0)]));
        assert_eq!(p.evict(), [1]);
    }

    #[test]
    fn revisiting_an_inflight_tile_does_not_duplicate_or_lose_its_result() {
        let mut p = TilePyramid::new(0, 14, 0);
        let view = cover(&[(3, 0, 0)]);
        assert_eq!(p.update(&view).request, [t(3, 0, 0)]);
        // Pan away and back before the request completes. No cancellation is
        // sent to the host, so this request is still alive across eviction.
        p.update(&cover(&[(3, 1, 0)]));
        p.evict();
        let revisited = p.update(&view);
        // Replay a host in which earlier duplicate attempts fail and the
        // latest attempt succeeds. Before retaining in-flight entries, the
        // older failure marked the new entry Empty and rejected its success:
        // a stationary view then stayed empty indefinitely.
        for _ in &revisited.request {
            p.insert_empty(t(3, 0, 0));
        }
        assert_eq!(p.insert(t(3, 0, 0), "arrived"), None);
        assert!(
            revisited.request.is_empty(),
            "duplicate request: {revisited:?}"
        );
        assert_eq!(p.update(&view).draw[0].tile, t(3, 0, 0));
    }

    #[test]
    fn results_for_drained_tiles_come_back() {
        let mut p = TilePyramid::new(0, 14, 0);
        p.update(&cover(&[(3, 0, 0)]));
        p.update(&cover(&[(3, 1, 0)]));
        p.evict();
        p.drain();
        assert_eq!(p.insert(t(3, 0, 0), "late"), Some("late"));
    }

    #[test]
    fn a_second_result_for_one_tile_comes_back() {
        let mut p = TilePyramid::new(0, 14, 0);
        p.update(&cover(&[(3, 0, 0)]));
        // Duplicate replies cannot replace a published tile.
        assert_eq!(p.insert(t(3, 0, 0), "first"), None);
        assert_eq!(p.insert(t(3, 0, 0), "second"), Some("second"));
        p.insert_empty(t(3, 0, 0));
        assert_eq!(p.get(t(3, 0, 0)), Some(&"first"));
    }

    #[test]
    fn evicts_the_least_recently_used_unused_tile_on_demand() {
        let mut p = TilePyramid::new(0, 14, 8);
        for col in 0..3 {
            p.update(&cover(&[(2, col, 0)]));
            p.insert(t(2, col, 0), col);
        }
        // Tile 2 is in use; 0 is the oldest unused.
        p.update(&cover(&[(2, 2, 0)]));
        assert_eq!(p.evict_lru(), Some(0));
        assert_eq!(p.evict_lru(), Some(1));
        assert_eq!(p.evict_lru(), None, "tiles in use stay");
        assert_eq!(p.get(t(2, 2, 0)), Some(&2));
    }

    #[test]
    fn evicts_descendants_a_tile_replaces() {
        let mut p = TilePyramid::new(0, 14, 8);
        p.update(&cover(&[(3, 0, 0), (3, 4, 4)]));
        p.insert(t(3, 0, 0), "inside");
        p.insert(t(3, 4, 4), "outside");
        p.update(&cover(&[(2, 0, 0), (3, 4, 4)]));
        assert_eq!(p.evict_within(t(2, 0, 0)), Some("inside"));
        assert_eq!(p.evict_within(t(2, 0, 0)), None);
        assert_eq!(p.get(t(3, 4, 4)), Some(&"outside"));
    }

    #[test]
    fn evicts_tiles_that_only_stand_in() {
        let mut p = TilePyramid::new(0, 14, 8);
        p.update(&cover(&[(1, 0, 0), (2, 3, 0), (3, 7, 7)]));
        p.insert(t(1, 0, 0), "grandparent");
        p.insert(t(2, 3, 0), "in the cover");
        p.insert(t(3, 7, 7), "elsewhere");
        p.update(&cover(&[(2, 0, 0)]));
        p.insert(t(2, 0, 0), "parent");
        // Zoomed in: the parent and grandparent stand in for the children,
        // and a finer tile for its ancestor.
        let view = cover(&[
            (3, 0, 0),
            (3, 1, 0),
            (3, 1, 1),
            (3, 0, 1),
            (2, 3, 0),
            (2, 3, 3),
        ]);
        p.update(&view);
        assert!(p.covers(t(3, 0, 0)) && !p.covers(t(2, 0, 0)));
        let near = Some(t(3, 0, 0));
        assert_eq!(p.evict_stand_in(near), Some("parent"));
        assert_eq!(p.evict_stand_in(near), Some("grandparent"));
        assert_eq!(p.evict_stand_in(near), Some("elsewhere"));
        // A tile of the cover never goes.
        assert_eq!(p.evict_stand_in(near), None);
        assert_eq!(p.get(t(2, 3, 0)), Some(&"in the cover"));
    }

    #[test]
    fn failed_tiles_ask_for_their_nearest_ancestor() {
        let mut p = TilePyramid::new(0, 14, 8);
        let view = cover(&[(3, 1, 2)]);
        p.update(&view);
        p.insert_failed(t(3, 1, 2), 0.0);
        // Nothing loaded stands in: the parent is requested, then drawn.
        let update = p.update(&view);
        assert_eq!(update.request, [t(2, 0, 1)]);
        assert!(update.draw.is_empty());
        p.insert_failed(t(2, 0, 1), 0.0);
        assert_eq!(p.update(&view).request, [t(1, 0, 0)]);
        p.insert(t(1, 0, 0), "grandparent");
        assert_eq!(
            p.update(&view).draw,
            [Draw {
                tile: t(1, 0, 0),
                wrap: 0,
                regions: vec![t(3, 1, 2)]
            }]
        );
    }

    /// Loads the view stopped wanting are given up once each; a given-up
    /// load the view wants again stays loading until its result, then is
    /// asked for anew.
    #[test]
    fn loads_the_view_left_are_given_up_once() {
        let mut p: TilePyramid<()> = TilePyramid::new(0, 14, 8);
        let (here, there) = (cover(&[(3, 1, 2)]), cover(&[(3, 6, 6)]));
        p.update(&here);
        assert!(p.unwanted_loads(Some(&quad)).is_empty());
        p.update(&there);
        assert_eq!(p.unwanted_loads(Some(&quad)), [t(3, 1, 2)]);
        assert!(p.unwanted_loads(Some(&quad)).is_empty());
        // Back before the cancelled load answers: nothing asks twice.
        assert!(p.update(&here).request.is_empty());
        p.forget_loading(t(3, 1, 2));
        assert_eq!(p.update(&here).request, [t(3, 1, 2)]);
    }

    /// The columns of a world one tile wide at zoom 0.
    fn quad(z: u8) -> Option<u32> {
        Some(1 << z)
    }

    /// A load the view left beside one of its tiles, across the
    /// antimeridian too, loads on where its caller keeps such loads, and
    /// goes last in the fetch order's eyes: the last plan didn't use it.
    #[test]
    fn loads_beside_the_view_load_on_when_asked_to() {
        let mut p: TilePyramid<()> = TilePyramid::new(0, 14, 8);
        let near = [t(3, 7, 2), t(3, 2, 3), t(3, 1, 3)];
        p.update(&cover(&[
            (3, 7, 2),
            (3, 2, 3),
            (3, 1, 3),
            (3, 0, 4),
            (3, 4, 6),
        ]));
        p.update(&cover(&[(3, 0, 2), (3, 2, 2)]));
        let mut given_up = p.unwanted_loads(Some(&quad));
        given_up.sort();
        assert_eq!(given_up, [t(3, 0, 4), t(3, 4, 6)]);
        assert!(
            near.iter()
                .all(|&tile| p.is_loading(tile) && !p.used_last(tile))
        );
        assert_eq!((p.loading(), p.loading_for_view()), (7, 4));
        assert_eq!(p.awaited(near[0]), Awaited::Wanted);
        let mut given_up = p.unwanted_loads(None);
        given_up.sort();
        assert_eq!(given_up, [t(3, 1, 3), t(3, 2, 3), t(3, 7, 2)]);
    }

    /// Across the antimeridian, a tile's neighbour is where its world
    /// wraps: past the last of WorldCRS84Quad's two columns a zoom-0 tile
    /// holds, not a quadtree's one; on a plane that doesn't wrap, nowhere.
    #[test]
    fn loads_beside_the_view_wrap_where_the_world_does() {
        let crs84 = |z: u8| Some(2u32 << z);
        let left = |beside: Option<&dyn Fn(u8) -> Option<u32>>| {
            let mut p: TilePyramid<()> = TilePyramid::new(0, 14, 8);
            p.update(&cover(&[(3, 7, 2), (3, 15, 2)]));
            p.update(&cover(&[(3, 0, 2)]));
            let mut given_up = p.unwanted_loads(beside);
            given_up.sort();
            given_up
        };
        assert_eq!(left(Some(&crs84)), [t(3, 7, 2)]);
        assert_eq!(left(Some(&|_| None)), [t(3, 7, 2), t(3, 15, 2)]);
    }

    /// A given-up load's answer finds it given up, unless a plan wanted
    /// it again first.
    #[test]
    fn loads_wanted_again_are_awaited_again() {
        let mut p: TilePyramid<()> = TilePyramid::new(0, 14, 8);
        let (here, there) = (cover(&[(3, 1, 2)]), cover(&[(3, 6, 6)]));
        p.update(&here);
        assert_eq!(p.awaited(t(3, 1, 2)), Awaited::Wanted);
        p.update(&there);
        p.unwanted_loads(None);
        assert_eq!(p.awaited(t(3, 1, 2)), Awaited::GivenUp);
        assert_eq!(p.loading(), 2, "given up, still in flight");
        p.update(&here);
        assert_eq!(p.awaited(t(3, 1, 2)), Awaited::Wanted);
        assert_eq!(p.awaited(t(5, 0, 0)), Awaited::No);
    }

    /// An arrived result waiting its turn is out of flight, but stands in
    /// as loading until it's inserted.
    #[test]
    fn arrived_tiles_are_not_in_flight() {
        let mut p = TilePyramid::new(0, 14, 8);
        p.update(&cover(&[(0, 0, 0)]));
        p.insert(t(0, 0, 0), "world");
        let view = cover(&[(1, 0, 0)]);
        p.update(&view);
        p.arrived(t(1, 0, 0));
        assert_eq!(p.loading(), 0);
        let update = p.update(&view);
        assert!(update.request.is_empty(), "not asked for twice");
        assert_eq!(update.draw[0].tile, t(0, 0, 0), "the ancestor stands in");
        let revision = p.revision();
        assert_eq!(p.insert(t(1, 0, 0), "tile"), None);
        assert!(p.revision() > revision);
        assert_eq!(p.update(&view).draw[0].tile, t(1, 0, 0));
    }

    #[test]
    fn forgotten_loads_are_requested_again() {
        let mut p: TilePyramid<()> = TilePyramid::new(0, 14, 8);
        let view = cover(&[(1, 0, 0)]);
        assert_eq!(p.update(&view).request, [t(1, 0, 0)]);
        p.forget_loading(t(1, 0, 0));
        assert_eq!(p.update(&view).request, [t(1, 0, 0)]);
    }

    #[test]
    fn keeps_ancestors_standing_in_for_missing_tiles() {
        let mut p = TilePyramid::new(0, 14, 8);
        p.update(&cover(&[(1, 0, 0)]));
        p.insert(t(1, 0, 0), "parent");
        let view = cover(&[(2, 0, 0), (2, 1, 0)]);
        p.update(&view);
        p.insert_missing(t(2, 1, 0));
        p.update(&view);
        assert_eq!(p.evict_stand_in(Some(t(2, 0, 0))), None);
        assert_eq!(p.get(t(1, 0, 0)), Some(&"parent"));
    }

    #[test]
    fn skips_tiles_below_min_zoom() {
        let mut p: TilePyramid<()> = TilePyramid::new(2, 14, 8);
        let update = p.update(&cover(&[(1, 0, 0)]));
        assert!(update.request.is_empty() && update.draw.is_empty());
    }
}
