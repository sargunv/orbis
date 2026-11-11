//! A GeoJSON document, parsed and tiled: the world, and tiles down its
//! north-west corner.

#![no_main]
#![forbid(unsafe_code)]

use orb_tile::geojson::{Index, Options};

libfuzzer_sys::fuzz_target!(|text: &str| {
    let Ok(features) = orb_style::v8::parse_geojson(text) else {
        return;
    };
    let options = Options {
        line_metrics: true,
        generate_id: true,
        ..Options::default()
    };
    if let Ok(mut index) = Index::new(features, options) {
        for z in [0, 1, 5, 14] {
            let _ = index.tile(z, 0, 0);
        }
    }
});
