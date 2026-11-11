//! A tile the tileset doesn't have (a missing file, as HTTP 404): a vector
//! tile draws its parent or nothing, as `MapOptions::missing_tiles` says;
//! a raster tile draws its parent either way.

use std::path::Path;

mod common;

use orb_api::{Map, MapOptions, MissingTiles, Runtime, RuntimeOptions};

fn varint(mut n: u64, out: &mut Vec<u8>) {
    while n >= 128 {
        out.push(n as u8 | 128);
        n >>= 7;
    }
    out.push(n as u8);
}

/// A length-delimited protobuf field.
fn message(field: u64, bytes: &[u8], out: &mut Vec<u8>) {
    varint(field << 3 | 2, out);
    varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

fn uint(field: u64, n: u64, out: &mut Vec<u8>) {
    varint(field << 3, out);
    varint(n, out);
}

/// A vector tile whose layer `l` holds one polygon over the whole tile.
fn full_tile() -> Vec<u8> {
    let zigzag = |n: i64| ((n << 1) ^ (n >> 63)) as u64;
    let mut geometry = Vec::new();
    // MoveTo(-64, -64), LineTo ×3 around the buffered square, ClosePath.
    for n in [9, zigzag(-64), zigzag(-64), 26]
        .into_iter()
        .chain(
            [(4224, 0), (0, 4224), (-4224, 0)]
                .into_iter()
                .flat_map(|(x, y)| [zigzag(x), zigzag(y)]),
        )
        .chain([15])
    {
        varint(n, &mut geometry);
    }
    let mut feature = Vec::new();
    uint(3, 3, &mut feature);
    message(4, &geometry, &mut feature);
    let mut layer = Vec::new();
    uint(15, 2, &mut layer);
    message(1, b"l", &mut layer);
    message(2, &feature, &mut layer);
    uint(5, 4096, &mut layer);
    let mut tile = Vec::new();
    message(3, &layer, &mut tile);
    tile
}

/// A 256-pixel blue PNG.
fn blue_png(path: &Path) {
    let file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut encoder = png::Encoder::new(file, 256, 256);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let pixels: Vec<u8> = [0, 0, 255, 255].repeat(256 * 256);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&pixels)
        .unwrap();
}

/// The middle pixel of a still at zoom 1.5, where only zoom 0's tile exists.
fn middle(source: &str, layer: &str, missing_tiles: MissingTiles) -> [u8; 4] {
    let style = format!(
        r##"{{"version": 8, "center": [10, 10], "zoom": 1.5,
            "sources": {{"s": {source}}},
            "layers": [
              {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {layer}]}}"##
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut map = Map::new(
        &runtime,
        MapOptions {
            missing_tiles,
            ..MapOptions::default()
        },
    );
    map.set_viewport(64.0, 64.0);
    map.set_style_json(&style).unwrap();
    let image = map.snapshot(64, 64, 1.0).unwrap();
    let i = (32 * 64 + 32) * 4;
    image.pixels[i..i + 4].try_into().unwrap()
}

#[test]
fn a_child_tile_that_404s_draws_its_parent_or_nothing() {
    let dir = common::Scratch::new("missing-tiles");
    std::fs::create_dir_all(dir.join("v/0/0")).unwrap();
    std::fs::create_dir_all(dir.join("r/0/0")).unwrap();
    std::fs::write(dir.join("v/0/0/0.pbf"), full_tile()).unwrap();
    blue_png(&dir.join("r/0/0/0.png"));

    let vector = format!(
        r#"{{"type": "vector", "maxzoom": 4, "tiles": ["file://{}/v/{{z}}/{{x}}/{{y}}.pbf"]}}"#,
        dir.display()
    );
    let fill = r##"{"id": "f", "type": "fill", "source": "s", "source-layer": "l",
        "paint": {"fill-color": "#ff0000"}}"##;
    let raster = format!(
        r#"{{"type": "raster", "maxzoom": 4, "tiles": ["file://{}/r/{{z}}/{{x}}/{{y}}.png"]}}"#,
        dir.display()
    );
    let image =
        r#"{"id": "r", "type": "raster", "source": "s", "paint": {"raster-fade-duration": 0}}"#;

    let red = [255, 0, 0, 255];
    let white = [255, 255, 255, 255];
    let blue = [0, 0, 255, 255];
    let results = [
        middle(&vector, fill, MissingTiles::Parent),
        middle(&vector, fill, MissingTiles::Empty),
        middle(&raster, image, MissingTiles::Parent),
        middle(&raster, image, MissingTiles::Empty),
    ];
    assert_eq!(results[0], red, "the vector parent stands in");
    assert_eq!(results[1], white, "the vector tile is empty");
    assert_eq!(results[2], blue, "the raster parent stands in");
    assert_eq!(results[3], blue, "the raster parent stands in either way");
}

/// Fails each URL's first request, as a network that dropped out does,
/// and serves `full_tile` after.
struct FlakyNetwork(std::sync::Mutex<std::collections::HashSet<String>>);

impl orb_api::NetworkHook for FlakyNetwork {
    fn request(&self, request: orb_api::NetworkRequest, done: orb_api::NetworkDone) {
        let failed_before = !self.0.lock().unwrap().insert(request.url);
        done(if failed_before {
            orb_api::NetworkAnswer::Body {
                bytes: full_tile(),
                fresh_until: None,
                etag: None,
                modified: None,
            }
        } else {
            orb_api::NetworkAnswer::Failed(orb_api::FetchError::Network("dropped".into()))
        });
    }
}

#[test]
fn a_still_requests_tiles_that_failed_before_again() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    runtime.set_network_hook(Some(std::sync::Arc::new(FlakyNetwork(Default::default()))));
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_viewport(64.0, 64.0);
    map.set_style_json(
        r##"{"version": 8, "center": [10, 10], "zoom": 0,
            "sources": {"s": {"type": "vector", "maxzoom": 0,
                "tiles": ["https://tiles.invalid/{z}/{x}/{y}.pbf"]}},
            "layers": [
              {"id": "bg", "type": "background", "paint": {"background-color": "#ffffff"}},
              {"id": "f", "type": "fill", "source": "s", "source-layer": "l",
                "paint": {"fill-color": "#ff0000"}}]}"##,
    )
    .unwrap();
    let middle = |map: &mut Map| -> [u8; 4] {
        let image = map.snapshot(64, 64, 1.0).unwrap();
        let i = (32 * 64 + 32) * 4;
        image.pixels[i..i + 4].try_into().unwrap()
    };
    // The tile's request fails: the still finishes without it.
    assert_eq!(middle(&mut map), [255, 255, 255, 255]);
    // The next still asks again rather than waiting out the backoff.
    assert_eq!(middle(&mut map), [255, 0, 0, 255]);
}
