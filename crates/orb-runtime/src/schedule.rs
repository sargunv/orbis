//! What each frame applies of what arrived, and in which order: the
//! [`Budget`] a frame spends on arrivals ([`FrameBudget`]), what it has
//! spent, and the tiles parked after the view left them.
//!
//! Frames are the one time the map works ([`Map::render`]). Arrived
//! loads wait in the map's inbox and apply nearest the view's focus
//! first; edits' results apply in the order they came. A frame applies at
//! least one result, so arrivals never stall, and stops once its time or
//! its bytes run out.
//!
//! [`FrameBudget`]: crate::FrameBudget
//! [`Map::render`]: crate::Map::render

use std::collections::VecDeque;

use orb_crs::TileIndex;

use crate::jobs::TileResult;
use crate::map::Message;
use crate::resident::HostData;

/// What a frame may spend applying what arrived: tiles' and edits' results
/// (their uploads, and the room made for them) and label commits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    /// Time on the map's thread from the frame's start, in milliseconds.
    pub ms: f64,
    /// Bytes of tile data uploaded.
    pub bytes: u64,
    /// Drape slot texels redrawn beyond the blank ones, which a frame
    /// draws whatever they take: what slots a zoom step or an arrival
    /// left stale redraw over the frames after, the longest stale first
    /// ([`Terrain::plan`](crate::terrain::Terrain::plan)).
    pub drape_texels: u64,
}

/// What the current frame has spent of its [`Budget`]'s bytes: its time
/// is a deadline on the services' clock.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Beat {
    /// The budget's bytes.
    cap: u64,
    bytes: u64,
    /// Tiles, edits and label commits applied.
    applied: u32,
}

impl Beat {
    /// A frame's, with `cap` bytes to upload.
    pub(crate) fn new(cap: u64) -> Self {
        Self {
            cap,
            ..Self::default()
        }
    }

    /// Whether the frame, `in_time` (before its deadline), affords
    /// applying a result that uploads `bytes` whole: its first always, so
    /// that arrivals never stall.
    pub(crate) fn affords(&self, in_time: bool, bytes: u64) -> bool {
        self.applied == 0 || (in_time && self.bytes + bytes <= self.cap)
    }

    /// Whether the frame takes a result that uploads `bytes`, whole or,
    /// if it `splits`, the bytes the frame has left of it: a vector tile's
    /// words go up over frames, so that none writes more than its cap.
    pub(crate) fn takes(&self, in_time: bool, bytes: u64, splits: bool) -> bool {
        match splits && bytes > self.left() {
            true => (self.applied == 0 || in_time) && self.left() > 0,
            false => self.affords(in_time, bytes),
        }
    }

    /// The bytes spent.
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The bytes it has left.
    pub(crate) fn left(&self) -> u64 {
        self.cap.saturating_sub(self.bytes)
    }

    /// Counts a result applied, or a piece of one.
    pub(crate) fn count(&mut self) {
        self.applied += 1;
    }

    /// Counts bytes uploaded.
    pub(crate) fn spend(&mut self, bytes: u64) {
        self.bytes += bytes;
    }
}

/// The bytes a tile result uploads: its meshes' and symbol tables' words
/// and its image's texels.
pub(crate) fn upload_bytes(build: &TileResult) -> u64 {
    let meshes: u64 = (build.meshes.iter())
        .map(|m| u64::from(m.vertices.len() + m.indices.len() + m.props.len()))
        .sum();
    let symbols: u64 = (build.symbols.iter())
        .flat_map(|s| &s.tables)
        .map(|t| u64::from(t.len()))
        .sum();
    let image = (build.image.as_ref()).map_or(0, |i| u64::from(i.width) * u64::from(i.height));
    4 * (meshes + symbols + image)
}

/// Tiles' results arrived and waiting their turn: edits' in the order they
/// came, loads in any (each beat takes them nearest the focus first).
#[derive(Default)]
pub(crate) struct Inbox {
    pub(crate) edits: VecDeque<Message>,
    pub(crate) loads: Vec<Message>,
}

/// Vector tiles built after the view gave their loads up, kept off the GPU
/// up to [`Parked::bytes`] of the map's memory budget, oldest out first: a
/// view that comes back to one uploads it rather than fetching and laying
/// it out again.
#[derive(Debug, Default)]
pub(crate) struct Parked {
    tiles: VecDeque<ParkedTile>,
    bytes: u64,
}

#[derive(Debug)]
struct ParkedTile {
    generation: u64,
    /// The plan source's key.
    source: u32,
    tile: TileIndex,
    build: TileResult,
    bytes: u64,
}

impl Parked {
    /// What a map whose memory budget is `budget` parks at most: an eighth
    /// of it, a few views' worth of tiles (64 MiB on a desktop, 24 on a
    /// phone).
    pub(crate) fn bytes(budget: u64) -> u64 {
        budget / 8
    }

    /// Parks `build`, tile `tile` of source `source` in tile generation
    /// `generation`, within `cap` bytes; the host data of those it pushes
    /// out goes to `release`.
    pub(crate) fn park(
        &mut self,
        (generation, source, tile): (u64, u32, TileIndex),
        build: TileResult,
        cap: u64,
        release: &mut Vec<HostData>,
    ) {
        let bytes = upload_bytes(&build);
        self.bytes += bytes;
        self.tiles.push_back(ParkedTile {
            generation,
            source,
            tile,
            build,
            bytes,
        });
        while self.bytes > cap
            && let Some(oldest) = self.tiles.pop_front()
        {
            self.bytes -= oldest.bytes;
            release.push(HostData::of(&oldest.build));
        }
    }

    /// Takes the tile parked for `tile` of source `source` in
    /// `generation`.
    pub(crate) fn take(
        &mut self,
        (generation, source, tile): (u64, u32, TileIndex),
    ) -> Option<TileResult> {
        let at = (self.tiles.iter())
            .position(|p| (p.generation, p.source, p.tile) == (generation, source, tile))?;
        let parked = self.tiles.remove(at).expect("found");
        self.bytes -= parked.bytes;
        Some(parked.build)
    }

    /// Drops the tiles `which` picks by source and tile, their host data
    /// to `release`.
    pub(crate) fn drop_where(
        &mut self,
        which: impl Fn(u32, TileIndex) -> bool,
        release: &mut Vec<HostData>,
    ) {
        let (dropped, kept) = std::mem::take(&mut self.tiles)
            .into_iter()
            .partition::<VecDeque<_>, _>(|p| which(p.source, p.tile));
        self.tiles = kept;
        for parked in dropped {
            self.bytes -= parked.bytes;
            release.push(HostData::of(&parked.build));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A result that uploads `texels` texels.
    fn build(texels: u32) -> TileResult {
        TileResult {
            image: Some(crate::TileImage {
                width: texels,
                height: 1,
                pixels: crate::ImagePixels::Core(Vec::new()),
            }),
            ..TileResult::default()
        }
    }

    #[test]
    fn parked_tiles_come_back_once_and_push_out_the_oldest() {
        let mut parked = Parked::default();
        let mut released = Vec::new();
        let cap = 48 << 20;
        let texels = (cap / 4 / 3) as u32 + 1;
        for col in 0..3 {
            let at = (1, 7, TileIndex::new(5, col, 0));
            parked.park(at, build(texels), cap, &mut released);
        }
        assert_eq!(released.len(), 1, "the third pushes the first out");
        assert!(parked.take((1, 7, TileIndex::new(5, 0, 0))).is_none());
        assert!(
            parked.take((2, 7, TileIndex::new(5, 1, 0))).is_none(),
            "another generation's"
        );
        assert!(parked.take((1, 7, TileIndex::new(5, 1, 0))).is_some());
        assert!(
            parked.take((1, 7, TileIndex::new(5, 1, 0))).is_none(),
            "taken once"
        );
        parked.drop_where(|source, _| source == 7, &mut released);
        assert_eq!(
            (parked.tiles.len(), parked.bytes, released.len()),
            (0, 0, 2)
        );
    }

    /// A vector tile larger than what the frame has left is taken in
    /// pieces; one that can't split is taken whole only as the first.
    #[test]
    fn a_frame_takes_what_splits_within_its_cap() {
        let mut beat = Beat::new(10);
        assert!(beat.takes(false, 100, true));
        beat.spend(10);
        beat.count();
        assert!(!beat.takes(true, 100, true), "nothing left");
        let mut beat = Beat::new(10);
        beat.spend(4);
        beat.count();
        assert!(beat.takes(true, 100, true));
        assert!(!beat.takes(false, 100, true), "out of time");
        assert!(!beat.takes(true, 100, false));
        assert!(beat.takes(true, 6, false));
    }

    #[test]
    fn a_frame_applies_its_first_result_whatever_it_costs() {
        let mut beat = Beat::new(10);
        assert!(beat.affords(false, 100));
        beat.spend(100);
        beat.count();
        assert!(!beat.affords(true, 0));
        let mut beat = Beat::new(10);
        beat.spend(4);
        beat.count();
        assert!(beat.affords(true, 6));
        assert!(!beat.affords(true, 7));
        assert!(!beat.affords(false, 0));
    }
}
