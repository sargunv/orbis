//! Tile sources (plan §7.3, §9.3): given the ideal cover of a view, decide
//! which tiles to load and what to draw while they load.
//!
//! A [`TilePyramid`] caches one source's tiles. Each frame,
//! [`TilePyramid::update`] takes the cover and returns the tiles to draw,
//! each with the regions it stands in for, and the tiles to request. A tile
//! that isn't ready yet retains loaded descendants, then fills uncovered
//! regions with its nearest ready ancestor. Replacements publish only
//! after they are ready, without discarding finer fallback coverage.

#![forbid(unsafe_code)]

mod gzip;
pub mod pmtiles;

use std::collections::{BTreeMap, HashMap, HashSet};

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

/// The outcome of [`TilePyramid::update`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Update {
    /// Tiles to draw, ancestors before descendants.
    pub draw: Vec<Draw>,
    /// Tiles to start loading, in cover order.
    pub request: Vec<TileIndex>,
}

#[derive(Debug)]
enum State<T> {
    Loading,
    Ready(T),
    /// Failed or empty: counts as ready and draws nothing.
    Empty,
    /// Not in the tileset (HTTP 404): ancestors or children stand in, as
    /// for a loading tile.
    Missing,
}

#[derive(Debug)]
struct Entry<T> {
    state: State<T>,
    last_used: u64,
}

/// One source's tiles: loading, ready and cached.
#[derive(Debug)]
pub struct TilePyramid<T> {
    tiles: HashMap<TileIndex, Entry<T>>,
    min_zoom: u8,
    max_zoom: u8,
    /// Ready tiles kept beyond those in use.
    cache_size: usize,
    frame: u64,
}

impl<T> TilePyramid<T> {
    pub fn new(min_zoom: u8, max_zoom: u8, cache_size: usize) -> Self {
        Self {
            tiles: HashMap::new(),
            min_zoom,
            max_zoom,
            cache_size,
            frame: 0,
        }
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

    pub fn is_loading(&self, tile: TileIndex) -> bool {
        matches!(
            self.tiles.get(&tile),
            Some(Entry {
                state: State::Loading,
                ..
            })
        )
    }

    /// Plans a frame. `cover` holds the ideal tiles, deepest at most
    /// `max_zoom`; tiles below `min_zoom` are not drawn. Requested tiles are
    /// marked loading until [`insert`](Self::insert) or
    /// [`insert_empty`](Self::insert_empty).
    pub fn update(&mut self, cover: &[CoveredTile]) -> Update {
        self.frame += 1;
        let frame = self.frame;
        let mut update = Update::default();
        // (tile, wrap) → regions, ordered so ancestors draw first.
        let mut draws: BTreeMap<(u8, TileIndex, i32), Vec<TileIndex>> = BTreeMap::new();

        // Only descend branches containing loaded tiles. Walking the cached
        // ancestor paths bounds this by cache size × depth, not 4^depth.
        let mut branches = HashSet::new();
        for (&tile, entry) in &self.tiles {
            if matches!(entry.state, State::Loading) {
                continue;
            }
            let mut ancestor = tile;
            while let Some(parent) = ancestor.parent() {
                if parent.z < self.min_zoom {
                    break;
                }
                branches.insert(parent);
                ancestor = parent;
            }
        }

        for covered in cover {
            let ideal = covered.index;
            if ideal.z < self.min_zoom {
                continue;
            }
            debug_assert!(ideal.z <= self.max_zoom, "cover deeper than the source");
            let entry = self.tiles.entry(ideal).or_insert_with(|| {
                update.request.push(ideal);
                Entry {
                    state: State::Loading,
                    last_used: frame,
                }
            });
            entry.last_used = frame;
            // A missing tile, unlike a loading one, has its ancestors
            // requested.
            let missing_tile = match entry.state {
                State::Ready(_) => {
                    draws.insert((ideal.z, ideal, covered.wrap), vec![ideal]);
                    continue;
                }
                State::Empty => continue,
                State::Loading => false,
                State::Missing => true,
            };

            let mut descendants = Vec::new();
            let mut missing = Vec::new();
            self.descendant_coverage(ideal, &branches, &mut descendants, &mut missing);
            for tile in descendants {
                self.touch(tile);
                if self.is_ready(tile) {
                    draws.insert((tile.z, tile, covered.wrap), vec![tile]);
                }
            }
            if !missing.is_empty()
                && let Some(ancestor) = self.ready_ancestor(ideal)
            {
                self.touch(ancestor);
                draws
                    .entry((ancestor.z, ancestor, covered.wrap))
                    .or_default()
                    .extend(missing);
            } else if missing_tile && !missing.is_empty() {
                self.request_ancestor(ideal, &mut update.request);
            }
        }

        update.draw = draws
            .into_iter()
            .map(|((_, tile, wrap), regions)| {
                let regions = if regions.contains(&tile) {
                    vec![tile]
                } else {
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

    /// Stores a loaded tile. Absent or already completed entries return
    /// their value back. After a drain, callers must reject old-generation
    /// results before they can reach newly requested entries.
    pub fn insert(&mut self, tile: TileIndex, value: T) -> Option<T> {
        match self.tiles.get_mut(&tile) {
            Some(entry) if matches!(entry.state, State::Loading) => {
                entry.state = State::Ready(value);
                None
            }
            _ => Some(value),
        }
    }

    /// Marks a loading tile as loaded with nothing to draw, such as a 404
    /// or an error.
    pub fn insert_empty(&mut self, tile: TileIndex) {
        if let Some(entry) = self.tiles.get_mut(&tile)
            && matches!(entry.state, State::Loading)
        {
            entry.state = State::Empty;
        }
    }

    /// Marks a loading tile as absent from the tileset: it draws nothing,
    /// and its nearest ready ancestor, or else its ready children, stand in.
    pub fn insert_missing(&mut self, tile: TileIndex) {
        if let Some(entry) = self.tiles.get_mut(&tile)
            && matches!(entry.state, State::Loading)
        {
            entry.state = State::Missing;
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
            .filter(|(_, e)| e.last_used != frame && !matches!(e.state, State::Loading))
            .map(|(&t, e)| (e.last_used, t))
            .collect();
        if unused.len() <= self.cache_size {
            return Vec::new();
        }
        unused.sort_unstable();
        let excess = unused.len() - self.cache_size;
        unused[..excess]
            .iter()
            .filter_map(|(_, t)| match self.tiles.remove(t)?.state {
                State::Ready(value) => Some(value),
                _ => None,
            })
            .collect()
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
        match self.tiles.remove(&tile)?.state {
            State::Ready(value) => Some(value),
            _ => unreachable!("filtered to ready tiles"),
        }
    }

    /// Evicts a ready tile inside `tile`'s area. While `tile` loads, its
    /// ready descendants stand in for it; once it's ready they draw nothing,
    /// so they're the first to make room for it.
    pub fn evict_within(&mut self, tile: TileIndex) -> Option<T> {
        let (&found, _) = self.tiles.iter().find(|(t, e)| {
            t.z > tile.z
                && matches!(e.state, State::Ready(_))
                && TileIndex::new(tile.z, t.col >> (t.z - tile.z), t.row >> (t.z - tile.z)) == tile
        })?;
        match self.tiles.remove(&found)?.state {
            State::Ready(value) => Some(value),
            _ => unreachable!("filtered to ready tiles"),
        }
    }

    /// Forgets every tile, returning the ready values.
    pub fn drain(&mut self) -> Vec<T> {
        self.tiles
            .drain()
            .filter_map(|(_, entry)| match entry.state {
                State::Ready(value) => Some(value),
                _ => None,
            })
            .collect()
    }

    /// Tiles requested and not yet inserted.
    pub fn loading(&self) -> usize {
        self.tiles
            .values()
            .filter(|e| matches!(e.state, State::Loading))
            .count()
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

    /// Partitions a missing ideal into loaded descendants and uncovered
    /// regions. Empty descendants cover their region without drawing it.
    fn descendant_coverage(
        &self,
        tile: TileIndex,
        branches: &HashSet<TileIndex>,
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

    /// Requests the nearest ancestor of a missing tile not yet known, past
    /// missing ones, as GL JS loads the parents of tiles that failed.
    /// Ancestors on the way stay in use.
    fn request_ancestor(&mut self, mut tile: TileIndex, request: &mut Vec<TileIndex>) {
        let frame = self.frame;
        while let Some(parent) = tile.parent() {
            if parent.z < self.min_zoom {
                return;
            }
            let Some(entry) = self.tiles.get_mut(&parent) else {
                self.tiles.insert(
                    parent,
                    Entry {
                        state: State::Loading,
                        last_used: frame,
                    },
                );
                request.push(parent);
                return;
            };
            entry.last_used = frame;
            if !matches!(entry.state, State::Missing) {
                return;
            }
            tile = parent;
        }
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

    fn touch(&mut self, tile: TileIndex) {
        if let Some(entry) = self.tiles.get_mut(&tile) {
            entry.last_used = self.frame;
        }
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
        evicted.sort();
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
        evicted.sort();
        assert_eq!(evicted, [0, 1]);
        assert!(p.get(t(2, 2, 0)).is_some());
        assert!(p.is_loading(t(2, 3, 0)));
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
    fn skips_tiles_below_min_zoom() {
        let mut p: TilePyramid<()> = TilePyramid::new(2, 14, 8);
        let update = p.update(&cover(&[(1, 0, 0)]));
        assert!(update.request.is_empty() && update.draw.is_empty());
    }
}
