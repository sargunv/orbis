//! Tile archives, PMTiles and MBTiles, read by byte ranges with no I/O: a
//! [`Reader`] answers what it can from the ranges read so far and
//! [`Step::Read`] for the next one; the caller reads it,
//! [`Reader::insert`]s it and asks again.

use orb_crs::TileIndex;
use orb_style::Encoding;

use crate::endpoint::TileJson;

/// The most bytes one read takes, or a body decompresses to (64 MiB).
pub const MAX_BYTES: u64 = 64 << 20;

/// A tile archive a source's `url` names.
#[derive(Clone, Debug, PartialEq, Eq, Hash, orb_wire::Wire)]
pub enum Archive {
    /// `pmtiles://`: the archive's URL without the protocol.
    Pmtiles(String),
    /// `mbtiles://`: the archive's URL, a path (`mbtiles:///data/a.mbtiles`)
    /// as a `file://` URL.
    Mbtiles(String),
}

impl Archive {
    /// The archive a source's `url` names, if it names one.
    pub fn of(url: &str) -> Option<Self> {
        if let Some(rest) = url.strip_prefix("pmtiles://") {
            return Some(Self::Pmtiles(rest.to_string()));
        }
        let rest = url.strip_prefix("mbtiles://")?;
        Some(Self::Mbtiles(match rest.starts_with('/') {
            true => format!("file://{rest}"),
            false => rest.to_string(),
        }))
    }

    /// Where its bytes are.
    pub fn url(&self) -> &str {
        match self {
            Self::Pmtiles(url) | Self::Mbtiles(url) => url,
        }
    }
}

/// Bytes of an archive: `length` at `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Range {
    pub offset: u64,
    pub length: u64,
}

impl Range {
    pub fn new(offset: u64, length: u64) -> Result<Self, String> {
        if length == 0 || length > MAX_BYTES || offset.checked_add(length).is_none() {
            return Err("invalid or oversized archive range".into());
        }
        Ok(Self { offset, length })
    }

    fn of<'a>(&self, file: &'a [u8]) -> &'a [u8] {
        let at = (self.offset as usize).min(file.len());
        let end = (self.offset.saturating_add(self.length) as usize).min(file.len());
        &file[at..end]
    }
}

/// What a read of an archive needs or gives.
#[derive(Clone, Debug, PartialEq)]
pub enum Step<T> {
    /// Read the range, [`insert`](Reader::insert) it and ask again.
    Read(Range),
    Done(T),
}

/// A tile an archive holds.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    pub body: Body,
    /// A vector tile's encoding; `None` for an image, decoded by content.
    pub encoding: Option<Encoding>,
}

/// Where a tile's bytes are.
#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    /// At this range of the archive, compressed so.
    At(Range, crate::pmtiles::Compression),
    /// Here, decompressed.
    Bytes(Vec<u8>),
}

/// A tile's bytes, decompressed, and its encoding (`None`: an image).
pub type TileBytes = (Vec<u8>, Option<Encoding>);

/// One archive's reader: what it has read of the archive so far.
#[derive(Debug)]
#[allow(clippy::large_enum_variant, reason = "one per archive, kept")]
pub enum Reader {
    Pmtiles(crate::pmtiles::Reader),
    #[cfg(feature = "mbtiles")]
    Mbtiles(crate::mbtiles::Reader),
}

impl Reader {
    /// A reader of `archive`, which has read nothing.
    pub fn new(archive: &Archive) -> Result<Self, String> {
        match archive {
            Archive::Pmtiles(_) => Ok(Self::Pmtiles(Default::default())),
            #[cfg(feature = "mbtiles")]
            Archive::Mbtiles(_) => Ok(Self::Mbtiles(Default::default())),
            #[cfg(not(feature = "mbtiles"))]
            Archive::Mbtiles(url) => Err(format!(
                "{url}: mbtiles:// needs Orbis built with its `sqlite` feature"
            )),
        }
    }

    /// Keeps the bytes a [`Step::Read`] of `range` asked for: the first
    /// read may be shorter, ending where the archive does.
    pub fn insert(&mut self, range: Range, bytes: &[u8]) -> Result<(), String> {
        match self {
            Self::Pmtiles(r) => r.insert(range, bytes),
            #[cfg(feature = "mbtiles")]
            Self::Mbtiles(r) => r.insert(range, bytes),
        }
    }

    /// Whether the read of `range` a step asked for is still wanted: false
    /// once bytes inserted hold it.
    pub fn wants(&self, range: Range) -> bool {
        match self {
            Self::Pmtiles(r) => r.wants(range),
            #[cfg(feature = "mbtiles")]
            Self::Mbtiles(r) => r.wants(range),
        }
    }

    /// The read of `range` failed as one past an archive's end does: true
    /// when that was the first read, which steps from now on ask shorter.
    pub fn read_failed(&mut self, range: Range) -> bool {
        match self {
            Self::Pmtiles(r) => r.read_failed(range),
            #[cfg(feature = "mbtiles")]
            Self::Mbtiles(r) => r.read_failed(range),
        }
    }

    /// What the archive says of its tiles.
    pub fn tilejson(&mut self) -> Result<Step<TileJson>, String> {
        match self {
            Self::Pmtiles(r) => Ok(r.tilejson()),
            #[cfg(feature = "mbtiles")]
            Self::Mbtiles(r) => r.tilejson(),
        }
    }

    /// Tile `tile`, or `None` when the archive leaves it out.
    pub fn tile(&mut self, tile: TileIndex) -> Result<Step<Option<Tile>>, String> {
        match self {
            Self::Pmtiles(r) => r.tile(tile),
            #[cfg(feature = "mbtiles")]
            Self::Mbtiles(r) => r.tile(tile),
        }
    }

    /// Answers `step` from `file`, an archive in memory, and tile bodies
    /// with it: a tile's bytes, decompressed.
    pub fn read_in<T>(
        &mut self,
        file: &[u8],
        mut step: impl FnMut(&mut Self) -> Result<Step<T>, String>,
    ) -> Result<T, String> {
        for _ in 0..MAX_READS {
            match step(self)? {
                Step::Done(value) => return Ok(value),
                Step::Read(range) => self.insert(range, range.of(file))?,
            }
        }
        Err("too many archive reads".into())
    }

    /// Tile `tile` of `file`, an archive in memory: its bytes and
    /// encoding, or `None` when it leaves the tile out.
    pub fn tile_in(&mut self, file: &[u8], tile: TileIndex) -> Result<Option<TileBytes>, String> {
        let Some(found) = self.read_in(file, |r| r.tile(tile))? else {
            return Ok(None);
        };
        let bytes = match found.body {
            Body::At(range, compression) => compression.decode(range.of(file))?,
            Body::Bytes(bytes) => bytes,
        };
        Ok(Some((bytes, found.encoding)))
    }
}

/// The most reads one step takes: a lookup reads a page per level of two
/// B-trees and an overflow run, or PMTiles' head and three leaves.
pub const MAX_READS: u32 = 64;
