//! Host-driven tile loading: tiles by URL, from the app, or read out of an
//! archive by the reader `orb-source` keeps pure, its range reads
//! satisfied here. Loads that need the same bytes at once share one fetch:
//! the first view of an archive reads its head and each directory once,
//! and overzoomed tiles of one data tile fetch it once. A load aborted
//! leaves the fetch it shares, which is aborted once no load waits on it.

use rustc_hash::FxHashMap as HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use orb_crs::TileIndex;
use orb_host::{Abort, Fetch, FetchDone, FetchError, Fetched, Read, ResourceKind, TileFormat};
use orb_source::archive::{self, Archive, Body, Range, Step, TileBytes};
use orb_source::endpoint::HostTile;
use orb_style::Encoding;

pub use orb_source::endpoint::TileLocation;

/// A tile's bytes, with their lifetime where they have one (a tile in an
/// archive has none), and their encoding.
pub(crate) type LoadDone =
    Box<dyn FnOnce(Result<(Fetched, Encoding), FetchError>) + Send + 'static>;
/// A shared fetch's answer, for each load that waits on it.
type Shared = Result<Arc<Fetched>, FetchError>;
type Waiter = Box<dyn FnOnce(Shared) + Send>;
/// A fetch's URL and what it reads of it.
type FetchKey = (String, Read);
/// An archive's reader, shared by the loads that read it.
pub(crate) type ArchiveReader = Arc<Mutex<archive::Reader>>;

/// Called once with the bytes of a range read of an archive.
pub(crate) type ReadDone = Box<dyn FnOnce(Result<&[u8], FetchError>) + Send>;
/// Reads a range of an archive: its directories and pages as
/// [`ResourceKind::Source`], a tile's body as an archive tile.
pub(crate) type Reads = Arc<dyn Fn(ResourceKind, Range, ReadDone) + Send + Sync>;

/// A fetch in flight: the loads waiting on it, by number, and its abort.
struct Sharing {
    /// Tells this fetch from a later one of the same bytes.
    id: u64,
    abort: Abort,
    waiters: Vec<(u64, Waiter)>,
}

pub(crate) struct TileLoader {
    fetch: Arc<dyn Fetch>,
    /// The readers of the last eight archives read, most recent last.
    archives: Mutex<VecDeque<(Archive, ArchiveReader)>>,
    in_flight: Mutex<HashMap<FetchKey, Sharing>>,
    /// Numbers shared fetches and their waiters.
    next: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for TileLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileLoader").finish_non_exhaustive()
    }
}

/// The format of a vector tile in `encoding`.
pub(crate) fn tile_format(encoding: Encoding) -> TileFormat {
    match encoding {
        Encoding::Mvt => TileFormat::Mvt,
        Encoding::Mlt => TileFormat::Mlt,
    }
}

/// Whether a failed first read of an archive may succeed shorter: anything
/// but an archive that isn't there, or a read given up.
fn falls_back(error: &FetchError) -> bool {
    !matches!(
        error,
        FetchError::NotFound | FetchError::UnsupportedUrl(_) | FetchError::Aborted
    )
}

/// Reads into `reader` the ranges `step` asks for, through `read`, until
/// it answers; a first read past a short archive's end is asked again
/// shorter, and a read another load's bytes already answered is kept once.
pub(crate) fn drive<T: 'static>(
    reader: ArchiveReader,
    step: impl Fn(&mut archive::Reader) -> Result<Step<T>, String> + Send + 'static,
    read: Reads,
    done: Box<dyn FnOnce(Result<T, FetchError>) + Send>,
) {
    fn go<T: 'static>(
        reader: ArchiveReader,
        step: impl Fn(&mut archive::Reader) -> Result<Step<T>, String> + Send + 'static,
        read: Reads,
        done: Box<dyn FnOnce(Result<T, FetchError>) + Send>,
        reads: u32,
    ) {
        if reads > archive::MAX_READS {
            return done(Err(FetchError::Range("too many archive reads".into())));
        }
        let range = match step(&mut reader.lock().expect("reader poisoned")) {
            Ok(Step::Done(value)) => return done(Ok(value)),
            Ok(Step::Read(range)) => range,
            Err(e) => return done(Err(FetchError::Range(e))),
        };
        let again = Arc::clone(&read);
        read(
            ResourceKind::Source,
            range,
            Box::new(move |result| {
                let kept = {
                    let mut reader = reader.lock().expect("reader poisoned");
                    match result {
                        Ok(bytes) if reader.wants(range) => {
                            reader.insert(range, bytes).map_err(FetchError::Range)
                        }
                        Ok(_) => Ok(()),
                        Err(e) if falls_back(&e) && reader.read_failed(range) => Ok(()),
                        Err(e) => Err(e),
                    }
                };
                match kept {
                    Ok(()) => go(reader, step, again, done, reads + 1),
                    Err(e) => done(Err(e)),
                }
            }),
        );
    }
    go(reader, step, read, done, 0);
}

/// Reads `tile` out of an archive with `reader`, its body too, through
/// `read`: `None` when the archive leaves it out.
pub(crate) fn read_tile(
    reader: ArchiveReader,
    tile: TileIndex,
    read: Reads,
    done: Box<dyn FnOnce(Result<Option<TileBytes>, FetchError>) + Send>,
) {
    let body = Arc::clone(&read);
    let found: Box<dyn FnOnce(_) + Send> = Box::new(move |found| match found {
        Ok(Some(archive::Tile {
            body: Body::At(range, compression),
            encoding,
        })) => body(
            ResourceKind::Tile(TileFormat::Archive),
            range,
            Box::new(move |read| {
                let bytes = read.and_then(|b| compression.decode(b).map_err(FetchError::Range));
                done(bytes.map(|bytes| Some((bytes, encoding))))
            }),
        ),
        Ok(Some(archive::Tile {
            body: Body::Bytes(bytes),
            encoding,
        })) => done(Ok(Some((bytes, encoding)))),
        Ok(None) => done(Ok(None)),
        Err(e) => done(Err(e)),
    });
    drive(reader, move |r| r.tile(tile), read, found);
}

/// [`Reads`] of `url` through `fetch`, `then` seeing each read's bytes.
pub(crate) fn fetch_reads(
    fetch: Arc<dyn Fetch>,
    url: String,
    abort: Abort,
    then: impl Fn(Range, &[u8]) + Send + Sync + 'static,
) -> Reads {
    let then = Arc::new(then);
    Arc::new(move |kind, range, done: ReadDone| {
        let then = Arc::clone(&then);
        let (offset, length) = (range.offset, range.length);
        fetch.get_range(
            kind,
            &url,
            offset,
            length,
            &abort,
            Box::new(move |result| match result {
                Ok(fetched) => {
                    then(range, &fetched.bytes);
                    done(Ok(&fetched.bytes))
                }
                Err(e) => done(Err(e)),
            }),
        );
    })
}

impl TileLoader {
    pub(crate) fn new(fetch: Arc<dyn Fetch>) -> Arc<Self> {
        Arc::new(Self {
            fetch,
            archives: Mutex::default(),
            in_flight: Mutex::default(),
            next: Default::default(),
        })
    }

    /// The reader of `archive`, made if need be: the most recently used of
    /// the eight kept.
    fn reader(&self, archive: &Archive) -> Result<ArchiveReader, FetchError> {
        let mut readers = self.archives.lock().expect("archives poisoned");
        if let Some(index) = readers.iter().position(|(a, _)| a == archive) {
            let entry = readers.remove(index).expect("found");
            let reader = Arc::clone(&entry.1);
            readers.push_back(entry);
            return Ok(reader);
        }
        let reader = archive::Reader::new(archive).map_err(FetchError::UnsupportedUrl)?;
        let reader = Arc::new(Mutex::new(reader));
        if readers.len() == 8 {
            readers.pop_front();
        }
        readers.push_back((archive.clone(), Arc::clone(&reader)));
        Ok(reader)
    }

    /// `archive` begins with `head`, which the map read for its header:
    /// its loads start from these bytes rather than read them.
    pub(crate) fn add_head(&self, archive: &Archive, head: &[u8]) {
        if let (Ok(range), Ok(reader)) = (Range::new(0, head.len() as u64), self.reader(archive)) {
            let mut reader = reader.lock().expect("reader poisoned");
            if reader.wants(range) {
                let _ = reader.insert(range, head);
            }
        }
    }

    /// Fetches `read` of `url`, or waits on the fetch of it already under
    /// way. Aborted, `waiter` gets [`FetchError::Aborted`] at once and the
    /// fetch goes on for the loads still waiting on it.
    fn share(
        self: &Arc<Self>,
        kind: ResourceKind,
        url: &str,
        read: Read,
        abort: &Abort,
        waiter: Waiter,
    ) {
        if abort.is_aborted() {
            return waiter(Err(FetchError::Aborted));
        }
        let key = (url.to_string(), read);
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
            let result = result.map(Arc::new);
            // The last waiter takes the answer: one load alone copies none.
            let mut waiting = waiting.into_iter().map(|(_, waiter)| waiter);
            let last = waiting.next_back();
            for waiter in waiting {
                waiter(result.clone());
            }
            if let Some(waiter) = last {
                waiter(result);
            }
        });
        self.fetch.read(kind, url, read, &shared, done);
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

    /// [`Reads`] of `url` through the shared fetches.
    fn reads(self: &Arc<Self>, url: String, abort: Abort) -> Reads {
        let loader = Arc::clone(self);
        Arc::new(move |kind, range, done: ReadDone| {
            let read = Read::Range(range.offset, range.length);
            let shared: Waiter = Box::new(move |result| match result {
                Ok(fetched) => done(Ok(&fetched.bytes)),
                Err(e) => done(Err(e)),
            });
            loader.share(kind, &url, read, &abort, shared);
        })
    }

    /// Loads a tile, unless `abort` fires first; from a URL, taking a
    /// stale cached copy where `stale` says ([`Fetch::get_or_stale`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load(
        self: &Arc<Self>,
        kind: ResourceKind,
        location: TileLocation,
        tile: TileIndex,
        encoding: Encoding,
        stale: bool,
        abort: &Abort,
        done: LoadDone,
    ) {
        match location {
            // The app answered: its tile, or none.
            TileLocation::Host(HostTile::Encoded {
                bytes,
                encoding: own,
            }) => done(Ok((bytes.into(), own.unwrap_or(encoding)))),
            TileLocation::Host(HostTile::GeoJson(_) | HostTile::Empty) => {
                done(Ok((Fetched::default(), encoding)))
            }
            TileLocation::Url(url) => self.share(
                kind,
                &url,
                Read::Whole { stale },
                abort,
                Box::new(move |r| done(r.map(|fetched| (owned(fetched), encoding)))),
            ),
            TileLocation::Archive(archive) => {
                let reader = match self.reader(&archive) {
                    Ok(reader) => reader,
                    Err(e) => return done(Err(e)),
                };
                let reads = self.reads(archive.url().to_string(), abort.clone());
                let found = Box::new(move |found: Result<Option<TileBytes>, _>| {
                    done(found.map(|found| match found {
                        // Archives leave out empty tiles: draw nothing,
                        // as the PMTiles protocol for GL JS answers.
                        None => (Fetched::default(), encoding),
                        // Images decode by content.
                        Some((bytes, own)) => (bytes.into(), own.unwrap_or(encoding)),
                    }))
                });
                read_tile(reader, tile, reads, found);
            }
        }
    }
}

/// A shared fetch's answer, copied unless this was the last load waiting.
fn owned(fetched: Arc<Fetched>) -> Fetched {
    Arc::try_unwrap(fetched).unwrap_or_else(|shared| (*shared).clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_source::pmtiles::{self, Compression, Contents, Entry, TileType, write_head};

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
                    Some(bytes) => Ok(bytes.to_vec().into()),
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
                ResourceKind::Tile(TileFormat::Mvt),
                TileLocation::Archive(Archive::Pmtiles("a".into())),
                tile,
                Encoding::Mvt,
                false,
                &Abort::NEVER,
                Box::new(move |r| got.lock().unwrap()[i] = r.unwrap().0.bytes),
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
        loader.add_head(&Archive::Pmtiles("a".into()), &archive[..16384]);
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
                ResourceKind::Tile(TileFormat::Mvt),
                TileLocation::Archive(Archive::Pmtiles("a".into())),
                TileIndex::new(9, 0, 0),
                Encoding::Mvt,
                false,
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
        let loader = TileLoader::new(Arc::clone(&fetch));
        let url = orb_host::file_url(&path);
        let archive = Archive::Mbtiles(url.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        drive(
            Arc::new(Mutex::new(archive::Reader::new(&archive).unwrap())),
            archive::Reader::tilejson,
            fetch_reads(fetch, url.clone(), Abort::new(), |_, _| {}),
            Box::new(move |r| tx.send(r).unwrap()),
        );
        let tilejson = rx.recv().unwrap().unwrap();
        assert_eq!(tilejson.maxzoom, Some(2.0));
        for (z, x, y, found) in [(2, 1, 2, true), (2, 3, 0, true), (2, 0, 0, false)] {
            let (tx, rx) = std::sync::mpsc::channel();
            loader.load(
                ResourceKind::Tile(TileFormat::Mvt),
                TileLocation::Archive(archive.clone()),
                TileIndex::new(z, x, y),
                Encoding::Mvt,
                false,
                &Abort::new(),
                Box::new(move |r| tx.send(r).unwrap()),
            );
            let (fetched, encoding) = rx.recv().unwrap().unwrap();
            let bytes = fetched.bytes;
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
