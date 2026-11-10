//! Test inputs shared across modules.

use std::sync::{Arc, Mutex};

use orb_crs::TileIndex;
use orb_host::jobs::Inline;
use orb_host::{Fetch, FetchDone, ImageDecode, ResourceKind};
use orb_style::Encoding;

use crate::jobs::TileResult;
use crate::loader::TileLocation;
use crate::worker::{Built, TileRequest, TileWorker};
use crate::{PoolJobs, Services};

/// The committed test data, served for `local://` URLs (`tests/data/README.md`).
pub(crate) const ASSETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

/// The tile [`SYMBOLS`] labels, under [`ASSETS`].
pub(crate) const SYMBOL_TILE: (u8, u32, u32) = (14, 14552, 6450);

/// A style with one text layer over a central Tokyo place tile.
pub(crate) const SYMBOLS: &str = r#"{"version":8,"glyphs":"local://glyphs/{fontstack}/{range}.pbf",
    "sources":{"s":{"type":"vector","tiles":["local://tiles/{z}-{x}-{y}.mvt"]}},
    "layers":[{"id":"labels","type":"symbol","source":"s","source-layer":"place",
        "layout":{"text-field":"{name_en}","text-font":["Noto Sans Regular"]},
        "paint":{"text-color":["match",["get","class"],"city","red","blue"]}}]}"#;

/// [`SYMBOLS`]' worker and its [`SYMBOL_TILE`], laid out with glyphs.
pub(crate) fn symbol_tile() -> (TileWorker, TileResult) {
    let worker = TileWorker::new(SYMBOLS).unwrap();
    let (z, x, y) = SYMBOL_TILE;
    let tile = TileIndex::new(z, x, y);
    let location = TileLocation::Url(format!("local://tiles/{z}-{x}-{y}.mvt"));
    let request = TileRequest::whole(1, 1, tile, location, Encoding::Mvt);
    let bytes = std::fs::read(format!("{ASSETS}/tiles/{z}-{x}-{y}.mvt")).unwrap();
    let Built::Fetch(urls) = worker.build(&bytes, Encoding::Mvt, &request).unwrap() else {
        panic!("labels need glyph ranges first")
    };
    for url in &urls {
        let path = format!("{ASSETS}/{}", url.strip_prefix("local://").unwrap());
        worker.add_resource(url, Ok(&std::fs::read(path).unwrap()));
    }
    let build = worker.build(&[], Encoding::Mvt, &request).unwrap().tile();
    (worker, TileResult::from(build))
}

/// A map's services for a test: fetches through `fetch`, tile jobs inline
/// on the thread that asks, no images, and a clock from now.
pub(crate) fn services(fetch: Arc<dyn Fetch>) -> Services {
    Services {
        clock: Arc::new(Clock(std::time::Instant::now())),
        image_decode: Arc::new(NoDecode),
        images: Default::default(),
        jobs: PoolJobs::new(fetch.clone(), Arc::new(NoDecode), Arc::new(Inline)),
        fetch,
        wake: Arc::new(|| {}),
    }
}

/// A fetch that holds requests until the test answers them.
#[derive(Default)]
pub(crate) struct HeldFetch {
    pub(crate) pending: Mutex<Vec<(String, FetchDone)>>,
}

impl Fetch for HeldFetch {
    fn get(&self, _: ResourceKind, url: &str, done: FetchDone) {
        self.pending.lock().unwrap().push((url.to_string(), done));
    }

    fn get_range(&self, _: ResourceKind, url: &str, _: u64, _: u64, done: FetchDone) {
        self.get(ResourceKind::Unknown, url, done);
    }
}

/// A fetch a test doesn't expect.
pub(crate) struct NoFetch;

impl Fetch for NoFetch {
    fn get(&self, _: ResourceKind, url: &str, _: FetchDone) {
        panic!("unexpected fetch of {url}");
    }

    fn get_range(&self, _: ResourceKind, url: &str, _: u64, _: u64, _: FetchDone) {
        panic!("unexpected fetch of {url}");
    }
}

/// Image decoding a test doesn't expect: tests load no images.
pub(crate) struct NoDecode;

impl ImageDecode for NoDecode {
    fn decode(&self, _: Vec<u8>, _: orb_host::ImageDone) {
        panic!("unexpected image");
    }
}

/// Milliseconds since the clock was made.
struct Clock(std::time::Instant);

impl orb_host::Clock for Clock {
    fn now(&self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1000.0
    }
}
