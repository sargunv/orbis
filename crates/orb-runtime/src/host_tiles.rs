//! Sources the app feeds: vector sources whose `tiles` are
//! [`HOST_TILES`](orb_source::endpoint::HOST_TILES). The map asks for each
//! tile with [`MapEvent::TileRequest`](crate::MapEvent::TileRequest) and
//! holds its jobs until the app answers with
//! [`Map::host_tile`](crate::Map::host_tile): the tile encoded, its source
//! layers as GeoJSON (a custom geometry source), or empty. The answer goes
//! to the jobs with each request, and is kept for relayouts.

use std::collections::{HashMap, VecDeque};

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

    /// Forgets a source's answers: its tiles are asked for again.
    pub(crate) fn forget(&mut self, source: u32) {
        self.answers.retain(|(s, _), _| *s != source);
        self.order.retain(|(s, _)| *s != source);
    }
}
