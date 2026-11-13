//! Host-driven tile loading. The PMTiles reader stays pure in orb-source;
//! this adapter satisfies its range reads and owns per-job-executor caches.
//! Loads that need the same bytes at once share one fetch: the first view
//! of an archive reads its head and each leaf directory once, and
//! overzoomed tiles of one data tile fetch it once. A load aborted leaves
//! the fetch it shares, which is aborted once no load waits on it.

use rustc_hash::FxHashMap as HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use orb_crs::TileIndex;
use orb_host::{Abort, Fetch, FetchDone, FetchError, ResourceKind};
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
/// A fetch's URL and byte range.
type FetchKey = (String, Option<Range>);
/// Fetches in flight, by URL and byte range.
type InFlight = HashMap<FetchKey, Sharing>;

/// A fetch in flight: the loads waiting on it, by number, and its abort.
struct Sharing {
    /// Tells this fetch from a later one of the same bytes.
    id: u64,
    abort: Abort,
    waiters: Vec<(u64, Waiter)>,
}

pub(crate) struct TileLoader {
    fetch: Arc<dyn Fetch>,
    archives: Mutex<Archives>,
    /// MBTiles archives' readers, as [`Self::archives`].
    #[cfg(feature = "sqlite")]
    mbtiles: Mutex<VecDeque<(String, Arc<Mutex<orb_source::mbtiles::Reader>>)>>,
    in_flight: Mutex<InFlight>,
    /// Numbers shared fetches and their waiters.
    next: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for TileLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileLoader").finish_non_exhaustive()
    }
}

/// Reads an archive's first bytes, [`pmtiles::HEAD`], or its header alone
/// when it is too short for that: `done` gets a prefix of the archive.
pub(crate) fn read_head(fetch: &Arc<dyn Fetch>, url: &str, abort: &Abort, done: FetchDone) {
    let (head, header) = (pmtiles::HEAD, pmtiles::HEADER_LEN);
    let (again, url_again, abort_again) = (Arc::clone(fetch), url.to_string(), abort.clone());
    fetch.get_range(
        ResourceKind::Source,
        url,
        head.offset,
        head.length,
        abort,
        Box::new(move |result| match result {
            Err(e) if falls_back(&e) => again.get_range(
                ResourceKind::Source,
                &url_again,
                0,
                header,
                &abort_again,
                done,
            ),
            result => done(result),
        }),
    );
}

/// Whether a failed first read may succeed as the header alone: anything
/// but an archive that isn't there, or a read given up.
pub(crate) fn falls_back(error: &FetchError) -> bool {
    !matches!(
        error,
        FetchError::NotFound | FetchError::UnsupportedUrl(_) | FetchError::Aborted
    )
}

impl TileLoader {
    pub(crate) fn new(fetch: Arc<dyn Fetch>) -> Arc<Self> {
        Arc::new(Self {
            fetch,
            archives: Mutex::new(VecDeque::new()),
            #[cfg(feature = "sqlite")]
            mbtiles: Mutex::default(),
            in_flight: Mutex::new(HashMap::default()),
            next: Default::default(),
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
    /// under way. Aborted, `waiter` gets [`FetchError::Aborted`] at once
    /// and the fetch goes on for the loads still waiting on it.
    fn share(
        self: &Arc<Self>,
        kind: ResourceKind,
        url: &str,
        range: Option<Range>,
        abort: &Abort,
        waiter: Waiter,
    ) {
        if abort.is_aborted() {
            return waiter(Err(FetchError::Aborted));
        }
        let key = (url.to_string(), range);
        let number = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (id, shared) = {
            let mut in_flight = self.in_flight.lock().expect("fetches poisoned");
            match in_flight.get_mut(&key) {
                Some(sharing) => {
                    sharing.waiters.push((number, waiter));
                    (sharing.id, None)
                }
                None => {
                    let shared = Abort::new();
                    let sharing = Sharing {
                        id: number,
                        abort: shared.clone(),
                        waiters: vec![(number, waiter)],
                    };
                    in_flight.insert(key.clone(), sharing);
                    (number, Some(shared))
                }
            }
        };
        let (loader, given_up) = (Arc::downgrade(self), key.clone());
        abort.on_abort(move || {
            if let Some(loader) = loader.upgrade() {
                loader.give_up(&given_up, id, number);
            }
        });
        let Some(shared) = shared else {
            return;
        };
        let loader = Arc::clone(self);
        let done: FetchDone = Box::new(move |result| {
            let waiting = {
                let mut in_flight = loader.in_flight.lock().expect("fetches poisoned");
                match in_flight.get(&key) {
                    Some(sharing) if sharing.id == id => in_flight.remove(&key),
                    _ => None,
                }
            };
            let waiting = waiting.map(|s| s.waiters).unwrap_or_default();
            let waiting = waiting.into_iter().map(|(_, waiter)| waiter);
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
            None => self.fetch.get(kind, url, &shared, done),
            Some(range) => {
                let (offset, length) = (range.offset, range.length);
                self.fetch
                    .get_range(kind, url, offset, length, &shared, done)
            }
        }
    }

    /// Takes waiter `number` off shared fetch `id` of `key`, answering it
    /// aborted, and aborts the fetch if it was the last.
    fn give_up(&self, key: &FetchKey, id: u64, number: u64) {
        let (waiter, abort) = {
            let mut in_flight = self.in_flight.lock().expect("fetches poisoned");
            let Some(sharing) = in_flight.get_mut(key).filter(|s| s.id == id) else {
                return;
            };
            let Some(i) = sharing.waiters.iter().position(|(n, _)| *n == number) else {
                return;
            };
            let (_, waiter) = sharing.waiters.remove(i);
            let last = sharing.waiters.is_empty();
            let abort = last.then(|| in_flight.remove(key).expect("found").abort);
            (waiter, abort)
        };
        waiter(Err(FetchError::Aborted));
        if let Some(abort) = abort {
            abort.abort();
        }
    }

    /// Loads a tile, unless `abort` fires first.
    pub(crate) fn load(
        self: &Arc<Self>,
        kind: ResourceKind,
        location: TileLocation,
        tile: TileIndex,
        encoding: Encoding,
        abort: &Abort,
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
                abort,
                Box::new(move |r| done(r.map(|bytes| (owned(bytes), encoding)))),
            ),
            TileLocation::Pmtiles(url) => {
                let reader = self.reader(&url);
                self.archive(url, reader, tile, encoding, 0, abort.clone(), done);
            }
            #[cfg(feature = "sqlite")]
            TileLocation::Mbtiles(url) => {
                let reader = {
                    let mut readers = self.mbtiles.lock().expect("archives poisoned");
                    most_recent(&mut readers, &url)
                };
                let step = move |r: &mut orb_source::mbtiles::Reader| r.tile(tile);
                drive(
                    self,
                    url,
                    reader,
                    step,
                    abort.clone(),
                    Box::new(move |result| {
                        use orb_source::mbtiles::Format;
                        done(result.map(|found| match found {
                            // Left out of the archive: empty, as PMTiles.
                            None => (Vec::new(), encoding),
                            Some((bytes, Format::Vector(own))) => (bytes, own),
                            Some((bytes, Format::Raster)) => (bytes, encoding),
                        }))
                    }),
                );
            }
            #[cfg(not(feature = "sqlite"))]
            TileLocation::Mbtiles(_) => done(Err(FetchError::UnsupportedUrl(
                "mbtiles:// needs Orbis built with its `sqlite` feature".into(),
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn archive(
        self: &Arc<Self>,
        url: String,
        reader: Arc<Mutex<Reader>>,
        tile: TileIndex,
        encoding: Encoding,
        reads: u8,
        abort: Abort,
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
                    &abort,
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
                let shared = abort.clone();
                self.share(
                    ResourceKind::Source,
                    &url.clone(),
                    Some(range),
                    &shared,
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
                            Ok(()) => {
                                loader.archive(url, reader, tile, encoding, reads + 1, abort, done)
                            }
                            Err(error) => done(Err(error)),
                        }
                    }),
                );
            }
        }
    }
}

/// The reader of archive `url` in `readers`, made if need be: the most
/// recently used of the eight kept.
#[cfg(feature = "sqlite")]
fn most_recent<R: Default>(
    readers: &mut VecDeque<(String, Arc<Mutex<R>>)>,
    url: &str,
) -> Arc<Mutex<R>> {
    if let Some(index) = readers.iter().position(|(u, _)| u == url) {
        let entry = readers.remove(index).expect("found");
        let reader = entry.1.clone();
        readers.push_back(entry);
        return reader;
    }
    let reader = Arc::<Mutex<R>>::default();
    if readers.len() == 8 {
        readers.pop_front();
    }
    readers.push_back((url.to_string(), reader.clone()));
    reader
}

/// What an MBTiles read gives, or why it failed.
#[cfg(feature = "sqlite")]
pub(crate) type MbtilesDone<T> = Box<dyn FnOnce(Result<T, FetchError>) + Send>;

/// Runs `step` on the MBTiles archive `url`'s `reader` until it answers,
/// reading the pages it asks for through `loader`'s shared fetches.
#[cfg(feature = "sqlite")]
pub(crate) fn drive<T: 'static>(
    loader: &Arc<TileLoader>,
    url: String,
    reader: Arc<Mutex<orb_source::mbtiles::Reader>>,
    step: impl Fn(&mut orb_source::mbtiles::Reader) -> Result<orb_source::mbtiles::Step<T>, String>
    + Send
    + 'static,
    abort: Abort,
    done: MbtilesDone<T>,
) {
    read_until(loader, url, reader, step, abort, done, 0);
}

#[cfg(feature = "sqlite")]
fn read_until<T: 'static>(
    loader: &Arc<TileLoader>,
    url: String,
    reader: Arc<Mutex<orb_source::mbtiles::Reader>>,
    step: impl Fn(&mut orb_source::mbtiles::Reader) -> Result<orb_source::mbtiles::Step<T>, String>
    + Send
    + 'static,
    abort: Abort,
    done: MbtilesDone<T>,
    reads: u32,
) {
    use orb_source::mbtiles::Step;
    // A lookup reads a page per level of two trees and an overflow run.
    if reads > 64 {
        return done(Err(FetchError::Range(format!(
            "{url}: too many MBTiles reads"
        ))));
    }
    let answer = step(&mut reader.lock().expect("reader poisoned"));
    let (offset, length) = match answer {
        Ok(Step::Done(value)) => return done(Ok(value)),
        Err(e) => return done(Err(FetchError::Range(format!("{url}: {e}")))),
        Ok(Step::Read { offset, length }) => (offset, length),
    };
    let range = match Range::new(offset, length) {
        Ok(range) => range,
        Err(e) => return done(Err(FetchError::Range(e))),
    };
    let next = Arc::clone(loader);
    loader.share(
        ResourceKind::Tile,
        &url.clone(),
        Some(range),
        &abort.clone(),
        Box::new(move |result| {
            let inserted = {
                let mut reader = reader.lock().expect("reader poisoned");
                match result {
                    Ok(bytes) => reader
                        .insert(offset, &bytes)
                        .map_err(|e| FetchError::Range(format!("{url}: {e}"))),
                    // A file shorter than the first read is read again.
                    Err(e) if falls_back(&e) && reader.read_failed(offset) => Ok(()),
                    Err(e) => Err(e),
                }
            };
            match inserted {
                Ok(()) => read_until(&next, url, reader, step, abort, done, reads + 1),
                Err(e) => done(Err(e)),
            }
        }),
    );
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
        /// Each read's abort.
        aborts: Mutex<Vec<Abort>>,
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
        fn get(&self, _: orb_host::ResourceKind, _: &str, _: &orb_host::Abort, done: FetchDone) {
            done(Err(FetchError::NotFound));
        }
        fn get_range(
            &self,
            _: orb_host::ResourceKind,
            _: &str,
            offset: u64,
            length: u64,
            abort: &Abort,
            done: FetchDone,
        ) {
            self.reads.lock().unwrap().push((offset, length));
            self.aborts.lock().unwrap().push(abort.clone());
            let done = abort.answer(done);
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
                &Abort::NEVER,
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

    /// A load aborted leaves the read it shares to the others, and the
    /// read is aborted once none waits on it.
    #[test]
    fn a_shared_read_is_aborted_with_its_last_load() {
        let fetch = Arc::new(Held {
            archive: archive(9),
            ..Held::default()
        });
        let loader = TileLoader::new(fetch.clone());
        let got = Arc::new(Mutex::new(Vec::new()));
        let aborts = [Abort::new(), Abort::new()];
        for (i, abort) in aborts.iter().enumerate() {
            let got = Arc::clone(&got);
            loader.load(
                ResourceKind::Tile,
                TileLocation::Pmtiles("a".into()),
                TileIndex::new(9, 0, 0),
                Encoding::Mvt,
                abort,
                Box::new(move |r| got.lock().unwrap().push((i, r.map(|(b, _)| b)))),
            );
        }
        assert_eq!(fetch.reads.lock().unwrap().len(), 1, "one head read");
        aborts[0].abort();
        assert_eq!(*got.lock().unwrap(), [(0, Err(FetchError::Aborted))]);
        assert!(!fetch.aborts.lock().unwrap()[0].is_aborted());
        aborts[1].abort();
        assert_eq!(got.lock().unwrap()[1], (1, Err(FetchError::Aborted)));
        assert!(fetch.aborts.lock().unwrap()[0].is_aborted());
        // The aborted read's answer goes nowhere, and starts no more.
        fetch.release();
        assert_eq!(got.lock().unwrap().len(), 2);
        assert_eq!(fetch.reads.lock().unwrap().len(), 1);
    }

    /// An MBTiles archive on disk loads its TileJSON and tiles through
    /// range reads of the file, as a `mbtiles:///path` source does.
    #[cfg(feature = "sqlite")]
    #[test]
    fn mbtiles_archives_load_by_range_reads() {
        let path = std::env::temp_dir().join(format!("orb-loader-{}.mbtiles", std::process::id()));
        let _ = std::fs::remove_file(&path);
        {
            let db = rusqlite::Connection::open(&path).unwrap();
            db.execute_batch(
                "CREATE TABLE metadata (name text, value text);
                 CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);
                 CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, tile_row);
                 INSERT INTO metadata VALUES ('format', 'pbf'), ('minzoom', '0'), ('maxzoom', '2');",
            )
            .unwrap();
            for (z, x, y) in [(0, 0, 0), (2, 1, 2), (2, 3, 0)] {
                let row = (1 << z) - 1 - y;
                let body = vec![z as u8 * 16 + x as u8; 5000];
                db.execute(
                    "INSERT INTO tiles VALUES (?, ?, ?, ?)",
                    rusqlite::params![z, x, row, body],
                )
                .unwrap();
            }
        }
        let fetch: Arc<dyn Fetch> = orb_host_native::NativeFetch::new();
        let loader = TileLoader::new(fetch);
        let url = format!("file://{}", path.display());
        let (tx, rx) = std::sync::mpsc::channel();
        drive(
            &loader,
            url.clone(),
            Arc::default(),
            orb_source::mbtiles::Reader::tilejson,
            Abort::new(),
            Box::new(move |r| tx.send(r).unwrap()),
        );
        let tilejson = rx.recv().unwrap().unwrap();
        assert!(tilejson.contains(r#""maxzoom":2"#), "{tilejson}");
        for (z, x, y, found) in [(2, 1, 2, true), (2, 3, 0, true), (2, 0, 0, false)] {
            let (tx, rx) = std::sync::mpsc::channel();
            loader.load(
                ResourceKind::Tile,
                TileLocation::Mbtiles(url.clone()),
                TileIndex::new(z, x, y),
                Encoding::Mvt,
                &Abort::new(),
                Box::new(move |r| tx.send(r).unwrap()),
            );
            let (bytes, encoding) = rx.recv().unwrap().unwrap();
            let want = if found {
                vec![z * 16 + x as u8; 5000]
            } else {
                Vec::new()
            };
            assert_eq!((bytes, encoding), (want, Encoding::Mvt), "{z}/{x}/{y}");
        }
        std::fs::remove_file(&path).unwrap();
    }
}
