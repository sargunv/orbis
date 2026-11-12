//! What the fuzz targets share: a tile worker whose style has a layer of
//! every type over the source layers of the GL JS fixtures' MVT tiles and
//! OpenMapTiles' MLT ones, with expressions reading feature properties,
//! and labels whose glyphs come from Latin's range.
//!
//! `cargo xtask fuzz <target> [--seconds n]` runs one, seeded from the
//! fixture mirrors. Input must never panic: a malformed input fails with
//! an error or warning, as the runtime reports it. A tile laid out also
//! crosses to the map as a job worker sends it, packed, and must read
//! back whole.

#![forbid(unsafe_code)]

use std::sync::OnceLock;

use orb_crs::TileIndex;
use orb_runtime::{Built, HostTile, TileLocation, TileRequest, TileWorker};
use orb_style::Encoding;

/// Glyph range 0–255 of the style's font; other ranges are missing.
const LATIN: &[u8] =
    include_bytes!("../../orb-runtime/tests/data/glyphs/Noto Sans Regular/0-255.pbf");

/// Layers of every type, over source layers by the names MVT (Mapbox
/// Streets) and MLT (OpenMapTiles) tiles use.
const STYLE: &str = r#"{"version":8,"glyphs":"fuzz://{fontstack}/{range}.pbf",
"sources":{"s":{"type":"vector","tiles":["fuzz://{z}/{x}/{y}"]}},
"layers":[
{"id":"water","type":"fill","source":"s","source-layer":"water",
 "paint":{"fill-color":["match",["get","class"],"lake","blue","red"],"fill-outline-color":"black"}},
{"id":"landuse","type":"fill","source":"s","source-layer":"landuse",
 "layout":{"fill-sort-key":["to-number",["get","rank"],0]},
 "paint":{"fill-opacity":["coalesce",["get","opacity"],0.5]}},
{"id":"landcover","type":"fill","source":"s","source-layer":"landcover","paint":{"fill-pattern":"missing"}},
{"id":"building","type":"fill-extrusion","source":"s","source-layer":"building",
 "paint":{"fill-extrusion-height":["coalesce",["get","render_height"],["get","height"],10],
  "fill-extrusion-base":["coalesce",["get","render_min_height"],["get","min_height"],0]}},
{"id":"road","type":"line","source":"s","source-layer":"road",
 "layout":{"line-cap":"round","line-join":"round","line-sort-key":["to-number",["get","layer"],0]},
 "paint":{"line-width":["interpolate",["linear"],["zoom"],5,1,18,["coalesce",["get","width"],8]],
  "line-dasharray":[2,1]}},
{"id":"transportation","type":"line","source":"s","source-layer":"transportation",
 "paint":{"line-gradient":["interpolate",["linear"],["line-progress"],0,"red",1,"blue"],"line-width":3}},
{"id":"waterway","type":"line","source":"s","source-layer":"waterway","paint":{"line-offset":2}},
{"id":"boundary","type":"line","source":"s","source-layer":"boundary","filter":["==",["get","admin_level"],2]},
{"id":"admin","type":"line","source":"s","source-layer":"admin"},
{"id":"poi","type":"circle","source":"s","source-layer":"poi",
 "paint":{"circle-radius":["coalesce",["get","rank"],3],"circle-color":["to-color",["get","color"],"red"]}},
{"id":"poi-heat","type":"heatmap","source":"s","source-layer":"poi_label",
 "paint":{"heatmap-weight":["coalesce",["get","scalerank"],1]}},
{"id":"place","type":"symbol","source":"s","source-layer":"place",
 "layout":{"text-field":["coalesce",["get","name_en"],["get","name"]],"text-font":["Noto Sans Regular"],
  "text-variable-anchor":["top","bottom","left","right"],"text-max-width":6}},
{"id":"place_label","type":"symbol","source":"s","source-layer":"place_label",
 "layout":{"text-field":"{name}","text-font":["Noto Sans Regular"],"text-writing-mode":["vertical","horizontal"]}},
{"id":"road_label","type":"symbol","source":"s","source-layer":"road_label",
 "layout":{"symbol-placement":"line","text-field":"{name}","text-font":["Noto Sans Regular"]}},
{"id":"transportation_name","type":"symbol","source":"s","source-layer":"transportation_name",
 "layout":{"symbol-placement":"line-center","text-field":["get","name"],"text-font":["Noto Sans Regular"]}},
{"id":"housenumber","type":"symbol","source":"s","source-layer":"housenumber",
 "layout":{"text-field":["format",["get","housenumber"],{"font-scale":0.8}],"text-font":["Noto Sans Regular"]}}
]}"#;

/// Lays out `bytes`, a tile in `encoding`, as tile `z/0/0` of [`STYLE`]'s
/// source: `options` picks the zoom (its low bits) and whether the tile
/// is bent as a globe's tiles are (its high bit).
pub fn lay_out(bytes: &[u8], encoding: Encoding, options: u8) {
    static WORKER: OnceLock<TileWorker> = OnceLock::new();
    let worker = WORKER.get_or_init(|| TileWorker::new(STYLE).expect("the fuzz style parses"));
    let (z, bend) = (options % 23, options & 0x80 != 0);
    // As an app's tile: decoded for each input, where a fetched one's
    // decode would be kept for the next.
    let location = TileLocation::Host(HostTile::Empty);
    let mut request = TileRequest::whole(1, 1, TileIndex::new(z, 0, 0), location, encoding);
    if bend {
        request.bend = Some(orb_runtime::Bend {
            cells: [4, 4],
            poles: [true, false],
        });
    }
    let mut built = worker.build(bytes, encoding, &request);
    if let Ok(Built::Fetch(urls)) = &built {
        for url in urls {
            let range = if url.ends_with("/0-255.pbf") {
                Ok(LATIN)
            } else {
                Err("missing".to_string())
            };
            worker.add_resource(url, range);
        }
        built = worker.build(bytes, encoding, &request);
    }
    if let Ok(Built::Tile(build)) = built {
        let packed = orb_runtime::pack(build).to_bytes();
        assert!(unpack(&packed), "a packed tile reads back");
    }
}

/// Reads `bytes` as a packed tile from a job worker
/// ([`orb_runtime::unpack`]): whether it holds together.
pub fn unpack(bytes: &[u8]) -> bool {
    let Ok(header) = orb_runtime::packed_header(bytes) else {
        return false;
    };
    let id = orb_hal::HostBytesId(0);
    orb_runtime::unpack(header, id, bytes.len() as u64).is_ok()
}
