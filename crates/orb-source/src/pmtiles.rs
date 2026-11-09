//! PMTiles v3 indexing, with no I/O. Callers satisfy [`Lookup::Read`] and
//! cache the decoded directory with [`Reader::insert`]. All allocations and
//! decompressed bodies are bounded; unsupported codecs fail explicitly.

use std::collections::VecDeque;

use orb_crs::TileIndex;

pub const HEADER_LEN: u64 = 127;
/// Maximum compressed or decompressed directory/tile body (64 MiB).
pub(crate) const MAX_BYTES: u64 = 64 << 20;
const CACHE_BYTES: usize = 8 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Range {
    pub offset: u64,
    pub length: u64,
}

impl Range {
    pub fn new(offset: u64, length: u64) -> Result<Self, String> {
        if length == 0 || length > MAX_BYTES || offset.checked_add(length).is_none() {
            return Err("invalid or oversized PMTiles range".into());
        }
        Ok(Self { offset, length })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    None,
    Gzip,
}

impl Compression {
    fn parse(value: u8) -> Result<Self, String> {
        match value {
            1 => Ok(Self::None),
            2 => Ok(Self::Gzip),
            _ => Err(format!("unsupported PMTiles compression {value}")),
        }
    }

    pub fn decode(self, bytes: &[u8]) -> Result<Vec<u8>, String> {
        if bytes.len() as u64 > MAX_BYTES {
            return Err("oversized PMTiles body".into());
        }
        match self {
            Self::None => Ok(bytes.to_vec()),
            Self::Gzip => crate::gzip::decode(bytes, MAX_BYTES as usize)
                .map_err(|e| format!("PMTiles gzip: {e}")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileType {
    Mvt,
    Mlt,
    /// Raster images, decoded by the host.
    Png,
    Jpeg,
    Webp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub root: Range,
    pub metadata: Range,
    pub(crate) leaves_offset: u64,
    pub(crate) leaves_length: u64,
    pub tiles_offset: u64,
    pub(crate) tiles_length: u64,
    pub internal_compression: Compression,
    pub tile_compression: Compression,
    pub tile_type: TileType,
    pub minzoom: u8,
    pub maxzoom: u8,
    pub bounds: [f64; 4],
}

impl Header {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != HEADER_LEN as usize || &bytes[..8] != b"PMTiles\x03" {
            return Err("expected a PMTiles v3 header".into());
        }
        let u64_at = |at| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        let bounds =
            |at| f64::from(i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())) / 1e7;
        let header = Self {
            root: Range::new(u64_at(8), u64_at(16))?,
            metadata: Range::new(u64_at(24), u64_at(32))?,
            leaves_offset: u64_at(40),
            leaves_length: u64_at(48),
            tiles_offset: u64_at(56),
            tiles_length: u64_at(64),
            internal_compression: Compression::parse(bytes[97])?,
            tile_compression: Compression::parse(bytes[98])?,
            tile_type: match bytes[99] {
                1 => TileType::Mvt,
                2 => TileType::Png,
                3 => TileType::Jpeg,
                4 => TileType::Webp,
                6 => TileType::Mlt,
                other => return Err(format!("unsupported PMTiles tile type {other}")),
            },
            minzoom: bytes[100],
            maxzoom: bytes[101],
            bounds: [bounds(102), bounds(106), bounds(110), bounds(114)],
        };
        if header.root.offset < HEADER_LEN || header.root.offset + header.root.length > 16384 {
            return Err("PMTiles root must fit in the first 16 KiB".into());
        }
        if header.minzoom > header.maxzoom
            || header.maxzoom > 31
            || header.bounds[0] < -180.0
            || header.bounds[2] > 180.0
            || header.bounds[1] < -90.0
            || header.bounds[3] > 90.0
            || header.bounds[0] > header.bounds[2]
            || header.bounds[1] > header.bounds[3]
        {
            return Err("invalid PMTiles zooms or bounds".into());
        }
        for (offset, length) in [
            (header.leaves_offset, header.leaves_length),
            (header.tiles_offset, header.tiles_length),
        ] {
            offset
                .checked_add(length)
                .ok_or("PMTiles section overflow")?;
        }
        Ok(header)
    }
}

/// Hilbert order within each zoom, preceded by every lower zoom.
pub fn tile_id(tile: TileIndex) -> Result<u64, String> {
    if tile.z > 31 || u64::from(tile.col) >= 1u64 << tile.z || u64::from(tile.row) >= 1u64 << tile.z
    {
        return Err("invalid PMTiles tile coordinate".into());
    }
    let (mut x, mut y) = (u64::from(tile.col), u64::from(tile.row));
    let mut id = ((1u64 << (2 * tile.z)) - 1) / 3;
    let mut side = (1u64 << tile.z) / 2;
    while side > 0 {
        let (rx, ry) = (u64::from(x & side != 0), u64::from(y & side != 0));
        id += side * side * ((3 * rx) ^ ry);
        if ry == 0 {
            if rx == 1 {
                // Only the bits below `side` matter in later iterations.
                x = side - 1 - (x & (side - 1));
                y = side - 1 - (y & (side - 1));
            }
            std::mem::swap(&mut x, &mut y);
        }
        side /= 2;
    }
    Ok(id)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub id: u64,
    pub offset: u64,
    pub length: u64,
    /// Zero points to a leaf directory; otherwise consecutive tile ids.
    pub run: u64,
}

fn varint(bytes: &mut &[u8]) -> Result<u64, String> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let (&byte, rest) = bytes.split_first().ok_or("truncated PMTiles varint")?;
        *bytes = rest;
        if shift == 63 && byte > 1 {
            return Err("PMTiles varint overflow".into());
        }
        value |= u64::from(byte & 127) << shift;
        if byte < 128 {
            return Ok(value);
        }
    }
    Err("PMTiles varint overflow".into())
}

pub fn directory(mut bytes: &[u8]) -> Result<Vec<Entry>, String> {
    let count = usize::try_from(varint(&mut bytes)?).map_err(|_| "PMTiles directory count")?;
    if count > bytes.len() / 4 || count > CACHE_BYTES / std::mem::size_of::<Entry>() {
        return Err("oversized PMTiles directory".into());
    }
    let mut entries = Vec::with_capacity(count);
    let mut id = 0u64;
    for i in 0..count {
        let delta = varint(&mut bytes)?;
        if i > 0 && delta == 0 {
            return Err("unsorted PMTiles directory".into());
        }
        id = id.checked_add(delta).ok_or("PMTiles id overflow")?;
        entries.push(Entry {
            id,
            offset: 0,
            length: 0,
            run: 0,
        });
    }
    for entry in &mut entries {
        entry.run = varint(&mut bytes)?;
    }
    for entry in &mut entries {
        entry.length = varint(&mut bytes)?;
        if entry.length == 0 || entry.length > MAX_BYTES {
            return Err("invalid PMTiles entry length".into());
        }
    }
    let mut end = 0u64;
    for (i, entry) in entries.iter_mut().enumerate() {
        let offset = varint(&mut bytes)?;
        entry.offset = if offset == 0 && i > 0 {
            end
        } else {
            offset.checked_sub(1).ok_or("invalid PMTiles offset")?
        };
        end = entry
            .offset
            .checked_add(entry.length)
            .ok_or("PMTiles offset overflow")?;
        entry
            .id
            .checked_add(entry.run)
            .ok_or("PMTiles run overflow")?;
    }
    if !bytes.is_empty() {
        return Err("trailing PMTiles directory bytes".into());
    }
    Ok(entries)
}

/// What an archive's header says of its tiles.
#[derive(Clone, Debug, PartialEq)]
pub struct Contents {
    pub tile_type: TileType,
    pub tile_compression: Compression,
    pub minzoom: u8,
    pub maxzoom: u8,
    /// West, south, east, north.
    pub bounds: [f64; 4],
    /// The JSON metadata object.
    pub metadata: String,
}

/// The bytes of an archive that come before its tile data: the header, the
/// root directory, the metadata and leaf directories. `tiles` place each
/// tile's body by `offset` and `length` within the tile data, in any order
/// (they are sorted here; `run` is ignored), so bodies can be appended as
/// they arrive and the archive is this head followed by them. Directories
/// that don't fit the first 16 KiB split into leaves.
pub fn write_head(tiles: &mut [Entry], contents: &Contents) -> Result<Vec<u8>, String> {
    tiles.sort_unstable_by_key(|t| t.id);
    for tile in tiles.iter_mut() {
        tile.run = 1;
    }
    if tiles.windows(2).any(|w| w[0].id == w[1].id) {
        return Err("a tile twice in one archive".into());
    }
    let metadata = crate::gzip::encode(contents.metadata.as_bytes());
    let room = 16384 - HEADER_LEN as usize;
    let (root, leaves) = {
        let root = crate::gzip::encode(&write_directory(tiles));
        if root.len() <= room {
            (root, Vec::new())
        } else {
            let mut per_leaf = 4096;
            loop {
                let mut leaves = Vec::new();
                let mut pointers = Vec::new();
                for chunk in tiles.chunks(per_leaf) {
                    let leaf = crate::gzip::encode(&write_directory(chunk));
                    pointers.push(Entry {
                        id: chunk[0].id,
                        offset: leaves.len() as u64,
                        length: leaf.len() as u64,
                        run: 0,
                    });
                    leaves.extend(leaf);
                }
                let root = crate::gzip::encode(&write_directory(&pointers));
                if root.len() <= room {
                    break (root, leaves);
                }
                per_leaf *= 2;
            }
        }
    };
    let root_at = HEADER_LEN;
    let metadata_at = root_at + root.len() as u64;
    let leaves_at = metadata_at + metadata.len() as u64;
    let tiles_at = leaves_at + leaves.len() as u64;
    let tiles_length = tiles.iter().map(|t| t.offset + t.length).max().unwrap_or(0);
    let mut head = vec![0u8; HEADER_LEN as usize];
    head[..8].copy_from_slice(b"PMTiles\x03");
    let count = tiles.len() as u64;
    for (at, value) in [
        (8, root_at),
        (16, root.len() as u64),
        (24, metadata_at),
        (32, metadata.len() as u64),
        (40, leaves_at),
        (48, leaves.len() as u64),
        (56, tiles_at),
        (64, tiles_length),
        (72, count),
        (80, count),
        (88, count),
    ] {
        head[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }
    let compression = |c: Compression| match c {
        Compression::None => 1,
        Compression::Gzip => 2,
    };
    let tile_type = match contents.tile_type {
        TileType::Mvt => 1,
        TileType::Png => 2,
        TileType::Jpeg => 3,
        TileType::Webp => 4,
        TileType::Mlt => 6,
    };
    // Clustered when the bodies are contiguous in tile id order.
    let clustered = tiles
        .iter()
        .try_fold(0, |end, t| (t.offset == end).then_some(end + t.length));
    head[96] = u8::from(clustered.is_some());
    head[97] = compression(Compression::Gzip);
    head[98] = compression(contents.tile_compression);
    head[99] = tile_type;
    head[100] = contents.minzoom;
    head[101] = contents.maxzoom;
    let [west, south, east, north] = contents.bounds;
    let e7 = |v: f64| ((v * 1e7).round() as i32).to_le_bytes();
    for (i, v) in [west, south, east, north].into_iter().enumerate() {
        head[102 + i * 4..106 + i * 4].copy_from_slice(&e7(v));
    }
    head[118] = contents.minzoom;
    head[119..123].copy_from_slice(&e7((west + east) / 2.0));
    head[123..127].copy_from_slice(&e7((south + north) / 2.0));
    head.extend(root);
    head.extend(metadata);
    head.extend(leaves);
    Ok(head)
}

/// A directory's bytes, before compression.
fn write_directory(entries: &[Entry]) -> Vec<u8> {
    fn varint(mut value: u64, out: &mut Vec<u8>) {
        while value >= 128 {
            out.push(value as u8 | 128);
            value >>= 7;
        }
        out.push(value as u8);
    }
    let mut out = Vec::new();
    varint(entries.len() as u64, &mut out);
    let mut previous = 0;
    for e in entries {
        varint(e.id - previous, &mut out);
        previous = e.id;
    }
    for e in entries {
        varint(e.run, &mut out);
    }
    for e in entries {
        varint(e.length, &mut out);
    }
    let mut end = None;
    for e in entries {
        // Zero continues from the previous entry's end.
        varint(
            if end == Some(e.offset) {
                0
            } else {
                e.offset + 1
            },
            &mut out,
        );
        end = Some(e.offset + e.length);
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    Read(Range),
    Tile {
        range: Range,
        compression: Compression,
        tile_type: TileType,
    },
    Missing,
}

/// What a reader reads first: the header and, the spec requires, the root
/// directory within the first 16 KiB, in one round trip.
pub const HEAD: Range = Range {
    offset: 0,
    length: 16384,
};
const HEADER: Range = Range {
    offset: 0,
    length: HEADER_LEN,
};

/// Reads an archive's directories as [`Lookup::Read`] asks: [`HEAD`] first,
/// then the leaves a tile's lookup needs, keeping the last 32 used.
#[derive(Debug)]
pub struct Reader {
    header: Option<Header>,
    root: Option<Vec<Entry>>,
    /// Least recently used first.
    leaves: VecDeque<(Range, Vec<Entry>)>,
    /// The first read: [`HEAD`], or the header alone for an archive
    /// shorter than that.
    first: Range,
}

impl Default for Reader {
    fn default() -> Self {
        Self {
            header: None,
            root: None,
            leaves: VecDeque::new(),
            first: HEAD,
        }
    }
}

impl Reader {
    /// Keeps the bytes a [`Lookup::Read`] of `range` asked for. The first
    /// read may be any prefix of the archive at least a header long: the
    /// root directory is kept too when it is within it.
    pub fn insert(&mut self, range: Range, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() as u64 != range.length {
            return Err("short PMTiles range".into());
        }
        if range.offset == 0 && range.length >= HEADER_LEN && range.length <= HEAD.length {
            let header = Header::parse(&bytes[..HEADER_LEN as usize])?;
            let root = header.root;
            let end = (root.offset + root.length) as usize;
            if end <= bytes.len() {
                let decoded = header
                    .internal_compression
                    .decode(&bytes[root.offset as usize..end])?;
                self.root = Some(directory(&decoded)?);
            }
            self.header = Some(header);
            return Ok(());
        }
        let header = self.header.as_ref().ok_or("PMTiles header missing")?;
        let entries = directory(&header.internal_compression.decode(bytes)?)?;
        if range == header.root {
            self.root = Some(entries);
        } else {
            self.leaves.retain(|(r, _)| *r != range);
            let size = |d: &Vec<Entry>| d.len() * std::mem::size_of::<Entry>();
            let mut bytes: usize = self.leaves.iter().map(|(_, d)| size(d)).sum();
            while self.leaves.len() >= 32 || bytes + size(&entries) > CACHE_BYTES {
                let Some((_, oldest)) = self.leaves.pop_front() else {
                    break;
                };
                bytes -= size(&oldest);
            }
            self.leaves.push_back((range, entries));
        }
        Ok(())
    }

    /// Whether the read of `range` a lookup asked for is still wanted:
    /// false once another read's bytes were inserted.
    pub fn wants(&self, range: Range) -> bool {
        match &self.header {
            None => range == self.first,
            Some(_) if range.offset == 0 => false,
            Some(header) if range == header.root => self.root.is_none(),
            Some(_) => !self.leaves.iter().any(|(r, _)| *r == range),
        }
    }

    /// The read of `range` failed. True when that was the first read of
    /// [`HEAD`], which an archive shorter than 16 KiB can't answer: lookups
    /// read the header alone from now on.
    pub fn read_failed(&mut self, range: Range) -> bool {
        if range != HEAD || self.header.is_some() {
            return false;
        }
        self.first = HEADER;
        true
    }

    /// Where `tile` is, or the read that finds out. A leaf directory used
    /// becomes the most recently used.
    pub fn lookup(&mut self, tile: TileIndex) -> Result<Lookup, String> {
        let Some(header) = &self.header else {
            return Ok(Lookup::Read(self.first));
        };
        if tile.z < header.minzoom || tile.z > header.maxzoom {
            return Ok(Lookup::Missing);
        }
        let id = tile_id(tile)?;
        let mut range = header.root;
        for depth in 0..4 {
            let dir = if depth == 0 {
                self.root.as_ref()
            } else if let Some(at) = self.leaves.iter().position(|(r, _)| *r == range) {
                let used = self.leaves.remove(at).expect("found");
                self.leaves.push_back(used);
                self.leaves.back().map(|(_, d)| d)
            } else {
                None
            };
            let Some(dir) = dir else {
                return Ok(Lookup::Read(range));
            };
            let index = dir.partition_point(|e| e.id <= id);
            let Some(entry) = index.checked_sub(1).map(|i| &dir[i]) else {
                return Ok(Lookup::Missing);
            };
            let (base, size) = if entry.run == 0 {
                (header.leaves_offset, header.leaves_length)
            } else {
                (header.tiles_offset, header.tiles_length)
            };
            if entry
                .offset
                .checked_add(entry.length)
                .is_none_or(|end| end > size)
            {
                return Err("PMTiles entry outside section".into());
            }
            range = Range::new(
                base.checked_add(entry.offset)
                    .ok_or("PMTiles offset overflow")?,
                entry.length,
            )?;
            if entry.run > 0 {
                return Ok(if id - entry.id < entry.run {
                    Lookup::Tile {
                        range,
                        compression: header.tile_compression,
                        tile_type: header.tile_type,
                    }
                } else {
                    Lookup::Missing
                });
            }
        }
        Err("PMTiles directory nesting exceeds four levels".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An archive of every tile at zoom `z` but some, written with their
    /// bodies out of order (as tiles may arrive): each body names its tile.
    fn archive(z: u8) -> Vec<u8> {
        let side = 1u32 << z;
        let mut tiles = Vec::new();
        let mut data = Vec::new();
        for row in (0..side).rev() {
            for col in (0..side).rev().filter(|c| (c + row) % 7 != 3) {
                let body = format!("{z}/{col}/{row}").into_bytes();
                tiles.push(Entry {
                    id: tile_id(TileIndex::new(z, col, row)).unwrap(),
                    offset: data.len() as u64,
                    length: body.len() as u64,
                    run: 0,
                });
                data.extend(body);
            }
        }
        let contents = Contents {
            tile_type: TileType::Mvt,
            tile_compression: Compression::None,
            minzoom: z,
            maxzoom: z,
            bounds: [-180.0, -85.0, 180.0, 85.0],
            metadata: "{}".into(),
        };
        let mut archive = write_head(&mut tiles, &contents).unwrap();
        archive.extend(data);
        archive
    }

    /// Looks `tile` up in `archive` as a server would answer, failing reads
    /// past its end; the directory reads go in `reads`.
    fn read(
        reader: &mut Reader,
        archive: &[u8],
        tile: TileIndex,
        reads: &mut Vec<Range>,
    ) -> Option<Vec<u8>> {
        loop {
            match reader.lookup(tile).unwrap() {
                Lookup::Read(range) => {
                    reads.push(range);
                    let (at, end) = (
                        range.offset as usize,
                        (range.offset + range.length) as usize,
                    );
                    match archive.get(at..end) {
                        Some(bytes) => reader.insert(range, bytes).unwrap(),
                        None => assert!(reader.read_failed(range), "{range:?}"),
                    }
                }
                Lookup::Tile { range, .. } => {
                    let at = range.offset as usize;
                    break Some(archive[at..at + range.length as usize].to_vec());
                }
                Lookup::Missing => break None,
            }
        }
    }

    /// Every tile reads back through the reader, from a small archive and
    /// from one with leaf directories.
    #[test]
    fn written_archives_read_back() {
        for z in [3u8, 9] {
            let side = 1u32 << z;
            let archive = archive(z);
            let header = Header::parse(&archive[..HEADER_LEN as usize]).unwrap();
            assert_eq!(header.leaves_length > 0, z == 9, "leaves only when needed");
            let mut reader = Reader::default();
            for (col, row) in [(0, 0), (side - 1, side - 1), (5, 2), (2, 5), (side / 2, 1)] {
                let found = read(
                    &mut reader,
                    &archive,
                    TileIndex::new(z, col, row),
                    &mut Vec::new(),
                );
                let expected =
                    ((col + row) % 7 != 3).then(|| format!("{z}/{col}/{row}").into_bytes());
                assert_eq!(found, expected, "{z}/{col}/{row}");
            }
        }
    }

    /// The first read brings the header and the root together; an archive
    /// shorter than 16 KiB fails that read and is read header first.
    #[test]
    fn first_read_has_the_root() {
        let large = archive(9);
        let mut reads = Vec::new();
        let mut reader = Reader::default();
        read(&mut reader, &large, TileIndex::new(9, 1, 1), &mut reads);
        assert_eq!(reads.len(), 2, "the head and a leaf: {reads:?}");
        assert_eq!(reads[0], HEAD);
        assert!(!reader.wants(HEAD) && !reader.wants(reads[1]));

        let small = archive(3);
        assert!(small.len() < HEAD.length as usize);
        let mut reads = Vec::new();
        let mut reader = Reader::default();
        read(&mut reader, &small, TileIndex::new(3, 1, 1), &mut reads);
        let root = Header::parse(&small[..HEADER_LEN as usize]).unwrap().root;
        assert_eq!(reads, [HEAD, HEADER, root]);
        assert!(!reader.read_failed(root), "only the first read falls back");
    }

    /// The 32 leaf directories kept are the ones used last.
    #[test]
    fn leaves_are_kept_least_recently_used_first() {
        let archive = archive(9);
        let mut reader = Reader::default();
        // One tile in each of 33 leaves.
        let mut tiles: Vec<(TileIndex, Range)> = Vec::new();
        'find: for row in 0..512 {
            for col in 0..512 {
                let tile = TileIndex::new(9, col, row);
                let mut reads = Vec::new();
                read(&mut reader, &archive, tile, &mut reads);
                if let Some(&leaf) = reads.iter().find(|r| **r != HEAD) {
                    tiles.push((tile, leaf));
                    if tiles.len() == 33 {
                        break 'find;
                    }
                }
            }
        }
        assert_eq!(tiles.len(), 33);
        let mut reader = Reader::default();
        let mut reads = Vec::new();
        for (tile, _) in &tiles[..32] {
            read(&mut reader, &archive, *tile, &mut reads);
        }
        // Using the first leaf again keeps it over the second.
        read(&mut reader, &archive, tiles[0].0, &mut reads);
        read(&mut reader, &archive, tiles[32].0, &mut reads);
        assert!(!reader.wants(tiles[0].1));
        assert!(reader.wants(tiles[1].1));
        assert!(!reader.wants(tiles[2].1));
    }

    #[test]
    fn hilbert_ids() {
        assert_eq!(tile_id(TileIndex::new(0, 0, 0)).unwrap(), 0);
        for (x, y, id) in [(0, 0, 1), (0, 1, 2), (1, 1, 3), (1, 0, 4)] {
            assert_eq!(tile_id(TileIndex::new(1, x, y)).unwrap(), id);
        }
        // Official PMTiles z2 Hilbert traversal.
        for (i, (x, y)) in [
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 3),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 3),
            (3, 2),
            (3, 1),
            (2, 1),
            (2, 0),
            (3, 0),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(tile_id(TileIndex::new(2, x, y)).unwrap(), 5 + i as u64);
        }
        assert!(tile_id(TileIndex::new(32, 0, 0)).is_err());
        assert!(tile_id(TileIndex::new(1, 2, 0)).is_err());
        assert!(tile_id(TileIndex::new(31, u32::MAX / 2, u32::MAX / 2)).is_ok());
    }

    #[test]
    fn directory_runs_leaves_and_contiguous_offsets() {
        // Entries: id 0 run 2 len 3 offset 0; id 4 leaf len 4 offset 3.
        let entries = directory(&[2, 0, 4, 2, 0, 3, 4, 1, 0]).unwrap();
        let root = Range::new(127, 9).unwrap();
        let header = Header {
            root,
            metadata: Range::new(136, 2).unwrap(),
            leaves_offset: 200,
            leaves_length: 20,
            tiles_offset: 300,
            tiles_length: 20,
            internal_compression: Compression::None,
            tile_compression: Compression::None,
            tile_type: TileType::Mvt,
            minzoom: 0,
            maxzoom: 2,
            bounds: [-180., -90., 180., 90.],
        };
        let mut reader = Reader {
            header: Some(header),
            root: Some(entries),
            leaves: VecDeque::new(),
            first: HEAD,
        };
        assert!(matches!(
            reader.lookup(TileIndex::new(1, 0, 0)).unwrap(),
            Lookup::Tile {
                range: Range {
                    offset: 300,
                    length: 3
                },
                ..
            }
        ));
        assert_eq!(
            reader.lookup(TileIndex::new(1, 1, 1)).unwrap(),
            Lookup::Missing
        );
        assert_eq!(
            reader.lookup(TileIndex::new(1, 1, 0)).unwrap(),
            Lookup::Read(Range::new(203, 4).unwrap())
        );
        // A leaf entry id4 run1 len2 offset5 (six bytes).
        reader.header.as_mut().unwrap().root = root;
        reader.leaves.push_back((
            Range::new(203, 4).unwrap(),
            vec![Entry {
                id: 4,
                run: 1,
                length: 2,
                offset: 5,
            }],
        ));
        assert!(matches!(
            reader.lookup(TileIndex::new(1, 1, 0)).unwrap(),
            Lookup::Tile {
                range: Range {
                    offset: 305,
                    length: 2
                },
                ..
            }
        ));
    }

    #[test]
    fn malformed_inputs_fail_without_large_allocations() {
        for bytes in [
            &[][..],
            &[255; 10],
            &[1, 0, 1, 0, 0],
            &[2, 0, 0, 1, 1, 1, 1, 1, 0],
            &[1, 0, 1, 1, 0],
        ] {
            assert!(directory(bytes).is_err());
        }
        assert!(Header::parse(&[0; 127]).is_err());
        assert!(Range::new(u64::MAX, 2).is_err());
        assert!(Range::new(0, MAX_BYTES + 1).is_err());
        assert!(Compression::Gzip.decode(b"invalid gzip").is_err());
    }
}
