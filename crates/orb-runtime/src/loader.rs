//! Host-driven tile loading. The PMTiles reader stays pure in orb-source;
//! this adapter satisfies its range reads and owns per-job-executor caches.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use orb_crs::TileIndex;
use orb_host::{Fetch, FetchError};
use orb_source::pmtiles::{Lookup, Reader, TileType};
use orb_style::Encoding;

pub use orb_source::endpoint::TileLocation;

pub type LoadDone = Box<dyn FnOnce(Result<(Vec<u8>, Encoding), FetchError>) + Send + 'static>;
type Archives = VecDeque<(String, Arc<Mutex<Reader>>)>;

pub struct TileLoader {
    fetch: Arc<dyn Fetch>,
    archives: Mutex<Archives>,
}

impl std::fmt::Debug for TileLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileLoader").finish_non_exhaustive()
    }
}

impl TileLoader {
    pub fn new(fetch: Arc<dyn Fetch>) -> Arc<Self> {
        Arc::new(Self {
            fetch,
            archives: Mutex::new(VecDeque::new()),
        })
    }

    pub fn load(
        self: &Arc<Self>,
        location: TileLocation,
        tile: TileIndex,
        encoding: Encoding,
        done: LoadDone,
    ) {
        match location {
            TileLocation::Url(url) => self.fetch.get(
                &url,
                Box::new(move |r| done(r.map(|bytes| (bytes, encoding)))),
            ),
            TileLocation::Pmtiles(url) => {
                let reader = {
                    let mut archives = self.archives.lock().expect("archives poisoned");
                    if let Some(index) = archives.iter().position(|(u, _)| *u == url) {
                        let entry = archives.remove(index).unwrap();
                        let reader = entry.1.clone();
                        archives.push_back(entry);
                        reader
                    } else {
                        let reader = Arc::new(Mutex::new(Reader::default()));
                        if archives.len() == 8 {
                            archives.pop_front();
                        }
                        archives.push_back((url.clone(), reader.clone()));
                        reader
                    }
                };
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
                self.fetch.get_range(
                    &url,
                    range.offset,
                    range.length,
                    Box::new(move |r| {
                        done(
                            r.and_then(|bytes| {
                                compression.decode(&bytes).map_err(FetchError::Range)
                            })
                            .map(|bytes| {
                                (
                                    bytes,
                                    match tile_type {
                                        TileType::Mvt => Encoding::Mvt,
                                        TileType::Mlt => Encoding::Mlt,
                                        // Images decode by content.
                                        TileType::Png | TileType::Jpeg | TileType::Webp => encoding,
                                    },
                                )
                            }),
                        );
                    }),
                );
            }
            Ok(Lookup::Read(range)) => {
                let loader = Arc::clone(self);
                self.fetch.get_range(
                    &url.clone(),
                    range.offset,
                    range.length,
                    Box::new(move |r| {
                        let result = r.and_then(|bytes| {
                            reader
                                .lock()
                                .expect("reader poisoned")
                                .insert(range, &bytes)
                                .map_err(FetchError::Range)
                        });
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
