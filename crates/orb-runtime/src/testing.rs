//! Test inputs shared across modules.

use orb_crs::TileIndex;
use orb_v8::Encoding;

use crate::jobs::{Built, TileRequest, TileResult, TileWorker};
use crate::loader::TileLocation;

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
    let request = TileRequest {
        style: 1,
        source: 1,
        tile,
        zoom: f64::from(tile.z),
        bend: None,
        data_zoom: tile.z,
        location: TileLocation::Url(format!("local://tiles/{z}-{x}-{y}.mvt")),
        encoding: Encoding::Mvt,
        nodes: None,
        rows_only: false,
        dem_unpack: None,
    };
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
