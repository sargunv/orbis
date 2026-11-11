//! A Mapbox Vector Tile, decoded and laid out: the first byte picks the
//! zoom and whether the tile is bent as a globe's tiles are.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    if let Some((&options, tile)) = data.split_first() {
        orb_fuzz::lay_out(tile, orb_style::Encoding::Mvt, options);
    }
});
