//! Sources the app feeds: vector sources whose `tiles` are
//! [`HOST_TILES`](orb_source::endpoint::HOST_TILES). The map asks for each
//! tile with [`MapEvent::TileRequest`](crate::MapEvent::TileRequest) and
//! holds its jobs until the app answers with
//! [`Map::host_tile`](crate::Map::host_tile): the tile encoded, its source
//! layers as GeoJSON (a custom geometry source), or empty. The answer goes
//! to the jobs with each request, and is kept for relayouts.

use std::collections::{HashMap, HashSet, VecDeque};

use orb_crs::TileIndex;
use orb_source::endpoint::HostTile;

use crate::TileRequest;

/// Answers kept for relayouts; one the map needs again after it went is
/// asked for again.
const KEPT: usize = 1024;

/// A source's key and a data tile.
type Key = (u32, TileIndex);

#[derive(Debug, Default)]
pub(crate) struct HostTiles {
    answers: HashMap<Key, HostTile>,
    /// Oldest answer first.
    order: VecDeque<Key>,
    /// Jobs held for an answer, and their edits.
    waiting: HashMap<Key, Vec<(TileRequest, Option<u64>)>>,
    /// Loads under way with an answer a reload forgot, by source and
    /// tile: each lands nowhere, and its tile is asked for again.
    stale: HashSet<Key>,
}

impl HostTiles {
    pub(crate) fn answer(&self, source: u32, tile: TileIndex) -> Option<&HostTile> {
        self.answers.get(&(source, tile))
    }

    /// Holds a job until the app answers; true when the app must be asked.
    pub(crate) fn wait(
        &mut self,
        source: u32,
        tile: TileIndex,
        request: TileRequest,
        edit: Option<u64>,
    ) -> bool {
        let waiting = self.waiting.entry((source, tile)).or_default();
        waiting.push((request, edit));
        waiting.len() == 1
    }

    /// Keeps the app's answer, and returns the jobs that waited for it.
    pub(crate) fn answered(
        &mut self,
        source: u32,
        tile: TileIndex,
        answer: HostTile,
    ) -> Vec<(TileRequest, Option<u64>)> {
        let key = (source, tile);
        if self.answers.insert(key, answer).is_none() {
            self.order.push_back(key);
            while self.order.len() > KEPT {
                if let Some(old) = self.order.pop_front() {
                    self.answers.remove(&old);
                }
            }
        }
        self.waiting.remove(&key).unwrap_or_default()
    }

    /// Forgets the answers of `source`'s data tiles that `which` picks.
    /// Returns the data tiles `which` picks whose jobs still wait, which
    /// the app is asked for again, and the tiles of those jobs.
    pub(crate) fn forget(
        &mut self,
        source: u32,
        which: &dyn Fn(TileIndex) -> bool,
    ) -> (Vec<TileIndex>, HashSet<TileIndex>) {
        let picked = |(s, t): &Key| *s == source && which(*t);
        self.answers.retain(|k, _| !picked(k));
        self.order.retain(|k| !picked(k));
        let mut asked = Vec::new();
        let mut held = HashSet::new();
        for (key, jobs) in &self.waiting {
            if key.0 == source {
                held.extend(jobs.iter().map(|(request, _)| request.tile));
                if which(key.1) {
                    asked.push(key.1);
                }
            }
        }
        (asked, held)
    }

    /// Marks a load under way as stale: it lands nowhere.
    pub(crate) fn stale(&mut self, source: u32, tile: TileIndex) {
        self.stale.insert((source, tile));
    }

    /// Whether a load that landed was stale, which it no longer is.
    pub(crate) fn landed_stale(&mut self, source: u32, tile: TileIndex) -> bool {
        self.stale.remove(&(source, tile))
    }
}
