//! Host-driven tile loading. The PMTiles reader stays pure in orb-source;
//! this adapter satisfies its range reads and owns per-job-executor caches.
//! Loads that need the same bytes at once share one fetch: the first view
//! of an archive reads its head and each leaf directory once, and
//! overzoomed tiles of one data tile fetch it once.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use orb_crs::TileIndex;
use orb_host::{Fetch, FetchDone, FetchError, ResourceKind};
use orb_source::endpoint::HostTile;
use orb_source::pmtiles::{self, Lookup, Range, Reader, TileType};
use orb_style::Encoding;

pub use orb_source::endpoint::TileLocation;

pub(crate) type LoadDone =
    Box<dyn FnOnce(Result<(Vec<u8>, Encoding), FetchError>) + Send + 'static>;
type Archives = VecDeque<(String, Arc<Mutex<Reader>>)>;
/// A shared fetch's bytes, for each load that waits on it.
type Shared = Result<Arc<Vec<u8>>, FetchError>;
type Waiter = Box<dyn FnOnce(Shared) + Send>;
/// Fetches in flight, by URL and byte range: the loads waiting on each.
type InFlight = HashMap<(String, Option<Range>), Vec<Waiter>>;

pub(crate) struct TileLoader {
    fetch: Arc<dyn Fetch>,
    archives: Mutex<Archives>,
    in_flight: Mutex<InFlight>,
}

impl std::fmt::Debug for TileLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileLoader").finish_non_exhaustive()
    }
}

/// Reads an archive's first bytes, [`pmtiles::HEAD`], or its header alone
/// when it is too short for that: `done` gets a prefix of the archive.
pub(crate) fn read_head(fetch: &Arc<dyn Fetch>, url: &str, done: FetchDone) {
    let (head, header) = (pmtiles::HEAD, pmtiles::HEADER_LEN);
    let (again, url_again) = (Arc::clone(fetch), url.to_string());
    fetch.get_range(
        ResourceKind::Source,
        url,
        head.offset,
        head.length,
        Box::new(move |result| match result {
            Err(e) if falls_back(&e) => {
                again.get_range(ResourceKind::Source, &url_again, 0, header, done)
            }
            result => done(result),
        }),
    );
}

/// Whether a failed first read may succeed as the header alone: anything
/// but an archive that isn't there.
pub(crate) fn falls_back(error: &FetchError) -> bool {
    !matches!(error, FetchError::NotFound | FetchError::UnsupportedUrl(_))
}

impl TileLoader {
    pub(crate) fn new(fetch: Arc<dyn Fetch>) -> Arc<Self> {
        Arc::new(Self {
            fetch,
            archives: Mutex::new(VecDeque::new()),
            in_flight: Mutex::new(HashMap::new()),
        })
    }

    /// The reader of archive `url`, the most recently used of the eight
    /// kept.
    fn reader(&self, url: &str) -> Arc<Mutex<Reader>> {
        let mut archives = self.archives.lock().expect("archives poisoned");
        if let Some(index) = archives.iter().position(|(u, _)| u == url) {
            let entry = archives.remove(index).expect("found");
            let reader = entry.1.clone();
            archives.push_back(entry);
            return reader;
        }
        let reader = Arc::new(Mutex::new(Reader::default()));
        if archives.len() == 8 {
            archives.pop_front();
        }
        archives.push_back((url.to_string(), reader.clone()));
        reader
    }

    /// Archive `url` begins with `head`, which the map read for its
    /// header: its loads start from these bytes rather than read them.
    pub(crate) fn add_head(&self, url: &str, head: &[u8]) {
        if let Ok(range) = Range::new(0, head.len() as u64) {
            let reader = self.reader(url);
            let mut reader = reader.lock().expect("reader poisoned");
            if reader.wants(range) {
                let _ = reader.insert(range, head);
            }
        }
    }

    /// Fetches `url` (`range` of it), or waits on the fetch of it already
    /// under way.
    fn share(
        self: &Arc<Self>,
        kind: ResourceKind,
        url: &str,
        range: Option<Range>,
        waiter: Waiter,
    ) {
        let key = (url.to_string(), range);
        {
            let mut in_flight = self.in_flight.lock().expect("fetches poisoned");
            if let Some(waiting) = in_flight.get_mut(&key) {
                waiting.push(waiter);
                return;
            }
            in_flight.insert(key.clone(), vec![waiter]);
        }
        let loader = Arc::clone(self);
        let done: FetchDone = Box::new(move |result| {
            let waiting = loader
                .in_flight
                .lock()
                .expect("fetches poisoned")
                .remove(&key)
                .unwrap_or_default();
            let result = result.map(Arc::new);
            // The last waiter takes the bytes: one load alone copies none.
            let mut waiting = waiting.into_iter();
            let last = waiting.next_back();
            for waiter in waiting {
                waiter(result.clone());
            }
            if let Some(waiter) = last {
                waiter(result);
            }
        });
        match range {
            None => self.fetch.get(kind, url, done),
            Some(range) => self
                .fetch
                .get_range(kind, url, range.offset, range.length, done),
        }
    }

    pub(crate) fn load(
        self: &Arc<Self>,
        kind: ResourceKind,
        location: TileLocation,
        tile: TileIndex,
        encoding: Encoding,
        done: LoadDone,
    ) {
        match location {
            // The app answered: its tile, or none.
            TileLocation::Host(HostTile::Encoded {
                bytes,
                encoding: own,
            }) => done(Ok((bytes, own.unwrap_or(encoding)))),
            TileLocation::Host(HostTile::GeoJson(_) | HostTile::Empty) => {
                done(Ok((Vec::new(), encoding)))
            }
            // A query's refetch of an app's tile the map no longer holds.
            TileLocation::Url(url) if url.starts_with(orb_source::endpoint::HOST_TILES) => {
                done(Err(FetchError::NotFound))
            }
            TileLocation::Url(url) => self.share(
                kind,
                &url,
                None,
                Box::new(move |r| done(r.map(|bytes| (owned(bytes), encoding)))),
            ),
            TileLocation::Pmtiles(url) => {
                let reader = self.reader(&url);
                self.archive(url, reader, tile, encoding, 0, done);
            }
        }
    }

    fn archive(
        self: &Arc<Self>,
        url: String,
        reader: Arc<Mutex<Reader>>,
        tile: TileIndex,
        encoding: Encoding,
        reads: u8,
        done: LoadDone,
    ) {
        if reads > 8 {
            done(Err(FetchError::Range(
                "too many PMTiles directory reads".into(),
            )));
            return;
        }
        let lookup = reader.lock().expect("reader poisoned").lookup(tile);
        match lookup {
            Err(error) => done(Err(FetchError::Range(error))),
            // Archives leave out empty tiles: draw nothing, as the
            // PMTiles protocol for GL JS answers empty data.
            Ok(Lookup::Missing) => done(Ok((Vec::new(), encoding))),
            Ok(Lookup::Tile {
                range,
                compression,
                tile_type,
            }) => {
                let encoding = match tile_type {
                    TileType::Mvt => Encoding::Mvt,
                    TileType::Mlt => Encoding::Mlt,
                    // Images decode by content.
                    TileType::Png | TileType::Jpeg | TileType::Webp => encoding,
                };
                // An archive holds tiles alone: an image source is one image.
                self.share(
                    ResourceKind::Tile,
                    &url,
                    Some(range),
                    Box::new(move |r| {
                        done(
                            r.and_then(|bytes| {
                                compression.decode(&bytes).map_err(FetchError::Range)
                            })
                            .map(|bytes| (bytes, encoding)),
                        );
                    }),
                );
            }
            Ok(Lookup::Read(range)) => {
                let loader = Arc::clone(self);
                self.share(
                    ResourceKind::Source,
                    &url.clone(),
                    Some(range),
                    Box::new(move |r| {
                        let result = {
                            let mut reader = reader.lock().expect("reader poisoned");
                            match r {
                                // The first waiter keeps the bytes for all.
                                Ok(bytes) if reader.wants(range) => {
                                    reader.insert(range, &bytes).map_err(FetchError::Range)
                                }
                                Ok(_) => Ok(()),
                                Err(e) if falls_back(&e) && reader.read_failed(range) => Ok(()),
                                Err(e) => Err(e),
                            }
                        };
                        match result {
                            Ok(()) => loader.archive(url, reader, tile, encoding, reads + 1, done),
                            Err(error) => done(Err(error)),
                        }
                    }),
                );
            }
        }
    }
}

/// A shared fetch's bytes, copied unless this was the last load waiting.
fn owned(bytes: Arc<Vec<u8>>) -> Vec<u8> {
    Arc::try_unwrap(bytes).unwrap_or_else(|shared| (*shared).clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_source::pmtiles::{Compression, Contents, Entry, write_head};

    /// Answers reads from `archive` once released, counting them; reads
    /// past its end fail as a server's 416 would.
    #[derive(Default)]
    struct Held {
        archive: Vec<u8>,
        reads: Mutex<Vec<(u64, u64)>>,
        held: Mutex<Vec<(u64, u64, FetchDone)>>,
    }

    impl Held {
        fn release(&self) -> usize {
            let held = std::mem::take(&mut *self.held.lock().unwrap());
            let n = held.len();
            for (offset, length, done) in held {
                let (at, end) = (offset as usize, (offset + length) as usize);
                done(match self.archive.get(at..end) {
                    Some(bytes) => Ok(bytes.to_vec()),
                    None => Err(FetchError::Status(416)),
                });
            }
            n
        }
    }

    impl Fetch for Held {
        fn get(&self, _: orb_host::ResourceKind, _: &str, done: FetchDone) {
            done(Err(FetchError::NotFound));
        }
        fn get_range(
            &self,
            _: orb_host::ResourceKind,
            _: &str,
            offset: u64,
            length: u64,
            done: FetchDone,
        ) {
            self.reads.lock().unwrap().push((offset, length));
            self.held.lock().unwrap().push((offset, length, done));
        }
    }

    fn archive(z: u8) -> Vec<u8> {
        let side = 1u32 << z;
        let mut tiles = Vec::new();
        let mut data = Vec::new();
        for row in 0..side {
            for col in 0..side {
                let body = format!("{col}/{row}").into_bytes();
                tiles.push(Entry {
                    id: pmtiles::tile_id(TileIndex::new(z, col, row)).unwrap(),
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

    /// Loads `tiles` at once, releasing reads round by round: the reads
    /// each round made, and what each load got.
    fn load_all(
        fetch: &Arc<Held>,
        loader: &Arc<TileLoader>,
        tiles: &[TileIndex],
    ) -> (Vec<usize>, Vec<Vec<u8>>) {
        let got = Arc::new(Mutex::new(vec![Vec::new(); tiles.len()]));
        for (i, &tile) in tiles.iter().enumerate() {
            let got = Arc::clone(&got);
            loader.load(
                ResourceKind::Tile,
                TileLocation::Pmtiles("a".into()),
                tile,
                Encoding::Mvt,
                Box::new(move |r| got.lock().unwrap()[i] = r.unwrap().0),
            );
        }
        let mut rounds = Vec::new();
        loop {
            match fetch.release() {
                0 => break,
                n => rounds.push(n),
            }
        }
        let got = got.lock().unwrap().clone();
        (rounds, got)
    }

    /// Tiles loaded at once share the head read, each leaf read and each
    /// tile read.
    #[test]
    fn loads_at_once_share_their_reads() {
        let fetch = Arc::new(Held {
            archive: archive(9),
            ..Held::default()
        });
        let loader = TileLoader::new(fetch.clone());
        let tiles: Vec<_> = (0..6).map(|i| TileIndex::new(9, i % 3, i / 3)).collect();
        let mut doubled = tiles.clone();
        doubled.extend(&tiles);
        let (rounds, got) = load_all(&fetch, &loader, &doubled);
        // The head, a leaf (the six tiles are in the first), the tiles.
        assert_eq!(rounds, [1, 1, 6], "{:?}", fetch.reads.lock().unwrap());
        for (tile, got) in doubled.iter().zip(&got) {
            assert_eq!(*got, format!("{}/{}", tile.col, tile.row).into_bytes());
        }
    }

    /// A short archive's first read fails, and its loads read the header
    /// alone, then the root, once.
    #[test]
    fn short_archives_read_their_header_alone() {
        let fetch = Arc::new(Held {
            archive: archive(2),
            ..Held::default()
        });
        let loader = TileLoader::new(fetch.clone());
        let tiles = [TileIndex::new(2, 0, 0), TileIndex::new(2, 3, 1)];
        let (rounds, got) = load_all(&fetch, &loader, &tiles);
        assert_eq!(rounds, [1, 1, 1, 2]);
        assert_eq!(got[1], b"3/1");
        let reads = fetch.reads.lock().unwrap();
        assert_eq!(reads[..2], [(0, 16384), (0, pmtiles::HEADER_LEN)]);
    }

    /// An archive's head handed over reads nothing again.
    #[test]
    fn a_head_added_is_not_read() {
        let archive = archive(9);
        let fetch = Arc::new(Held {
            archive: archive.clone(),
            ..Held::default()
        });
        let loader = TileLoader::new(fetch.clone());
        loader.add_head("a", &archive[..16384]);
        let (rounds, _) = load_all(&fetch, &loader, &[TileIndex::new(9, 0, 0)]);
        assert_eq!(rounds, [1, 1], "a leaf, then the tile");
    }
}
