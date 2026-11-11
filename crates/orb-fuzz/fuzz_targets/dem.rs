//! A DEM tile: its side from the first two bytes and its texels the rest,
//! decoded with terrarium's or Mapbox's unpack vector, and sampled.

#![no_main]
#![forbid(unsafe_code)]

libfuzzer_sys::fuzz_target!(|data: &[u8]| {
    let Some((&[a, b], texels)) = data.split_first_chunk::<2>() else {
        return;
    };
    let dim = u32::from(u16::from_le_bytes([a, b]) >> 1);
    let unpack = match b & 0x80 {
        0 => [256.0, 1.0, 1.0 / 256.0, 32768.0],
        _ => [6553.6, 25.6, 0.1, 10000.0],
    };
    if let Ok(dem) = orb_tile::DemGrid::from_rgba(dim, texels, unpack) {
        let side = i64::from(dim);
        for (i, j) in [(0, 0), (-1, -1), (side, side / 2)] {
            let _ = dem.texel(i, j);
        }
    }
});
