//! What each frame applies of what arrived, and in which order: the
//! [`Budget`] a frame spends on arrivals, what it has spent, and the
//! tiles parked after the view left them.
//!
//! Frames are the one time the map works ([`Map::render`]). Arrived
//! loads wait in the map's inbox and apply nearest the view's focus
//! first; edits' results apply in the order they came. A frame applies at
//! least one result, so arrivals never stall, and stops once its time or
//! its bytes run out.
//!
//! [`Map::render`]: crate::Map::render

use std::collections::VecDeque;

use orb_crs::TileIndex;

use crate::jobs::TileResult;
use crate::map::Message;
use crate::resident::HostData;

/// What a frame may spend applying what arrived since the last before it
/// draws: tiles' and edits' results (their uploads, and the room made for
/// them), label commits and stale drape slots, counted from the frame's
/// start. However often the host hears of arrivals, a frame spends no
/// more, and tiles that arrive together spread over frames rather than
/// stalling one, at least one a frame. It is one budget whatever the
/// camera does: a map held still loads as one let go. The web's frames,
/// on the display's refresh, are the one time its map works: what
/// arrives between them only queues, its wake asking for the next. Native
/// hosts also spend it in [`Map::prepare`](crate::Map::prepare) and,
/// drawing nothing, in [`Map::apply_arrivals`](crate::Map::apply_arrivals).
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
    /// Drape sheets those slots lie in, beyond those of the blank ones:
    /// each replays the drape's draws on the CPU, every draped layer's.
    pub drape_sheets: usize,
}

impl Budget {
    /// What a frame may redraw of the drape: its texels and sheets.
    pub(crate) fn drape(&self) -> (u64, usize) {
        (self.drape_texels, self.drape_sheets)
    }

    /// A desktop's, the default: a third of a 60 Hz frame and 8 MiB. What
    /// a frame applies costs it again in the cull's candidates and drape
    /// slots, so the bytes bound it as much as the time. Drape slots:
    /// eight whole colour slots (an M6 redraws one in about 0.4 ms).
    pub const DESKTOP: Self = Self {
        ms: 6.0,
        bytes: 8 << 20,
        drape_texels: 8 << 20,
        drape_sheets: 2,
    };
    /// A phone's or a tablet's.
    pub const MOBILE: Self = Self {
        ms: 3.0,
        bytes: 2 << 20,
        drape_texels: 2 << 20,
        drape_sheets: 1,
    };
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
/// up to the map's [`cache_limit`](crate::Map::cache_limit) of tiles,
/// oldest out first: a view that comes back to one uploads it rather than fetching
/// and laying it out again.
#[derive(Debug, Default)]
pub(crate) struct Parked {
    tiles: VecDeque<ParkedTile>,
}

#[derive(Debug)]
struct ParkedTile {
    generation: u64,
    /// The plan source's key.
    source: u32,
    tile: TileIndex,
    build: TileResult,
}

impl Parked {
    /// Parks `build`, tile `tile` of source `source` in tile generation
    /// `generation`, within `cap` tiles; the host data of those it pushes
    /// out goes to `release`.
    pub(crate) fn park(
        &mut self,
        (generation, source, tile): (u64, u32, TileIndex),
        build: TileResult,
        cap: usize,
        release: &mut Vec<HostData>,
    ) {
        self.tiles.push_back(ParkedTile {
            generation,
            source,
            tile,
            build,
        });
        while self.tiles.len() > cap
            && let Some(oldest) = self.tiles.pop_front()
        {
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
        Some(self.tiles.remove(at).expect("found").build)
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
        for col in 0..3 {
            let at = (1, 7, TileIndex::new(5, col, 0));
            parked.park(at, build(16), 2, &mut released);
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
        assert_eq!((parked.tiles.len(), released.len()), (0, 2));
    }
}
