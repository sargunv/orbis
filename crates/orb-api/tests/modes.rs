//! Still and Tile output: the same inputs draw the same bytes,
//! and tiles drawn apart agree on the labels across their edges.

use std::path::PathBuf;

use orb_api::{
    CameraState, CameraTarget, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, TileMatrixSet,
    TileRequest,
};

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../web/demo/data")
        .canonicalize()
        .unwrap()
}

/// The demo's Tokyo extract with Liberty's labels.
fn tokyo() -> String {
    let data = data();
    std::fs::read_to_string(data.join("tokyo-mvt.json"))
        .unwrap()
        .replace(
            "pmtiles://./",
            &format!("pmtiles://file://{}/", data.display()),
        )
        .replace("\"./", &format!("\"file://{}/", data.display()))
}

/// A label's text and anchor (longitude, latitude).
type Anchor = (String, f64, f64);

/// Labels a few pixels apart over a few tiles at zoom 10, each longer than
/// the gap: a random field, and a row in which each overlaps the next, so
/// that which ones show along it runs in one chain of collisions from its
/// west end. Returns the style and each label's anchor.
fn crowd() -> (String, Vec<Anchor>) {
    let mut anchors = Vec::new();
    // 41 pixels apart at zoom 10, west to east: placed against the view, the
    // row shows another label at an edge when its west end is out of view.
    for i in 0..60 {
        anchors.push((format!("chain{i:03}"), -0.6 + f64::from(i) * 0.0283, 0.6));
    }
    let mut seed = 7u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        f64::from(seed >> 8) / f64::from(1u32 << 24)
    };
    for i in 0..1200 {
        let (lon, lat) = (-0.5 + next() * 1.4, -0.5 + next() * 0.8);
        anchors.push((format!("{}{i}", "label".repeat(1 + i % 3)), lon, lat));
    }
    let features: Vec<String> = anchors
        .iter()
        .map(|(name, lon, lat)| {
            format!(
                r#"{{"type":"Feature","properties":{{"name":"{name}"}},"geometry":{{"type":"Point","coordinates":[{lon},{lat}]}}}}"#
            )
        })
        .collect();
    let style = format!(
        r#"{{"version":8,
            "glyphs":"file://{}/glyphs/{{fontstack}}/{{range}}.pbf",
            "sources":{{"crowd":{{"type":"geojson","data":{{"type":"FeatureCollection","features":[{}]}}}}}},
            "layers":[
              {{"id":"bg","type":"background","paint":{{"background-color":"white"}}}},
              {{"id":"names","type":"symbol","source":"crowd",
                "layout":{{"text-field":["get","name"],"text-font":["Noto Sans Regular"],"text-size":14}}}}
            ]}}"#,
        data().display(),
        features.join(",")
    );
    (style, anchors)
}

fn mercator() -> TileMatrixSet {
    TileMatrixSet::by_id("WebMercatorQuad").unwrap()
}

/// The tile of `z` that holds `at`.
fn tile_at(at: Geodetic, z: u8) -> orb_crs::TileIndex {
    let p = mercator().projection().forward(at);
    mercator()
        .tiles_in(orb_crs::WorldRect { min: p, max: p }, z)
        .next()
        .unwrap()
}

/// Draws one tile of `z` at 512 pixels.
fn render_tile(map: &mut Map, z: u8, col: u32, row: u32) -> Vec<u8> {
    let request = TileRequest {
        tile_matrix_set: mercator(),
        z,
        col,
        row,
        size: TILE,
        scale: 1.0,
    };
    map.render_tile(&request).unwrap().pixels
}

const TILE: u32 = 512;

#[test]
fn still_output_repeats_byte_for_byte() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let still = |map: &mut Map| map.snapshot(512, 384, 1.0).unwrap().pixels;
    let station = Geodetic::new(139.7671, 35.6812);
    let mut map = Map::new(
        &runtime,
        MapOptions {
            camera: Some(CameraState::new(station, 15.0, mercator().projection())),
            ..MapOptions::default()
        },
    );
    map.set_style_json(&tokyo()).unwrap();
    let first = still(&mut map);
    // Elsewhere and back: what was placed there doesn't carry over.
    for (center, zoom) in [(Geodetic::new(139.75, 35.69), 13.5), (station, 15.0)] {
        map.jump_to(&CameraTarget {
            center: Some(center),
            zoom: Some(zoom),
            ..Default::default()
        });
        still(&mut map);
    }
    assert!(still(&mut map) == first, "a still differs on repeat");
}

#[test]
fn tiles_repeat_byte_for_byte() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&tokyo()).unwrap();
    let center = tile_at(Geodetic::new(139.755, 35.686), 15);
    let first = render_tile(&mut map, 15, center.col, center.row);
    // A neighbour between: what was placed there doesn't carry over.
    render_tile(&mut map, 15, center.col + 1, center.row);
    assert!(
        render_tile(&mut map, 15, center.col, center.row) == first,
        "a tile differs on repeat"
    );
}

/// A tile laid out first as a neighbour (only what reaches into the tile
/// drawn) loads whole once drawn: it draws what a fresh map's does. And a
/// neighbour laid out whole, drawn before, serves as one laid out as a
/// neighbour does.
#[test]
fn a_tile_drawn_after_its_neighbour_matches_a_fresh_one() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let center = tile_at(Geodetic::new(139.755, 35.686), 15);
    let mut fresh = Map::new(&runtime, MapOptions::default());
    fresh.set_style_json(&tokyo()).unwrap();
    let first = render_tile(&mut fresh, 15, center.col, center.row);
    let beside_whole = render_tile(&mut fresh, 15, center.col - 1, center.row);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&tokyo()).unwrap();
    assert!(
        render_tile(&mut map, 15, center.col - 1, center.row) == beside_whole,
        "a tile beside a neighbour laid out whole differs from one beside a neighbour's edges"
    );
    assert!(
        render_tile(&mut map, 15, center.col, center.row) == first,
        "a tile drawn after its neighbour differs from a fresh map's"
    );
}

/// Each tile places its labels with its neighbours' (tile scope), so
/// two tiles that share an edge both see a label anchored a few pixels from
/// it, which crosses it: both show it or neither does.
#[test]
fn tiles_agree_on_labels_across_edges() {
    const Z: u8 = 10;
    let (style, anchors) = crowd();
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    let center = tile_at(Geodetic::new(0.2, 0.2), Z);
    let mut tiles = Vec::new();
    for row in center.row - 1..=center.row + 1 {
        for col in center.col - 1..=center.col + 1 {
            render_tile(&mut map, Z, col, row);
            let shown: std::collections::HashSet<Box<str>> = map
                .shown_labels()
                .into_iter()
                .filter(|&(_, opacity)| opacity > 0.5)
                .map(|(text, _)| text)
                .collect();
            tiles.push(((col, row), shown));
        }
    }
    // Web Mercator pixels of the grid at `Z`.
    let world = f64::from(TILE) * f64::from(1u32 << Z);
    let pixel = |lon: f64, lat: f64| {
        let y = lat.to_radians().tan().asinh();
        [
            (lon + 180.0) / 360.0 * world,
            (1.0 - y / std::f64::consts::PI) / 2.0 * world,
        ]
    };
    let s = f64::from(TILE);
    let (mut agree, mut differ) = (0, Vec::new());
    for (i, ((ca, ra), a)) in tiles.iter().enumerate() {
        for ((cb, rb), b) in &tiles[i + 1..] {
            if ca.abs_diff(*cb) + ra.abs_diff(*rb) != 1 {
                continue;
            }
            // The shared edge, `edge` across it, starting `start` along it.
            let vertical = ca != cb;
            let (edge, start) = if vertical {
                (f64::from(*ca.max(cb)) * s, f64::from(*ra) * s)
            } else {
                (f64::from(*ra.max(rb)) * s, f64::from(*ca) * s)
            };
            for (name, lon, lat) in &anchors {
                let [x, y] = pixel(*lon, *lat);
                let (across, along) = if vertical { (x, y) } else { (y, x) };
                // Well within the shortest label's half width, and away
                // from the edge's ends.
                if (across - edge).abs() > 8.0 || along < start + 20.0 || along > start + s - 20.0 {
                    continue;
                }
                match (a.contains(name.as_str()), b.contains(name.as_str())) {
                    (true, true) => agree += 1,
                    (false, false) => {}
                    _ => differ.push(name.clone()),
                }
            }
        }
    }
    eprintln!("{agree} labels agree across edges, {} differ", differ.len());
    assert!(agree > 0, "no labels crossed the edges");
    assert!(differ.is_empty(), "labels differ across edges: {differ:?}");
}

#[test]
fn a_style_that_fails_to_load_fails_a_snapshot_at_once() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let dir = std::env::temp_dir().join(format!("orbis-bad-style-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("style.json");
    // An array output without `literal` doesn't parse.
    std::fs::write(
        &path,
        r#"{"version":8,"sources":{},"layers":[{"id":"bg","type":"background",
            "paint":{"background-color":["match",1,1,[1,2],"red"]}}]}"#,
    )
    .unwrap();
    let mut map = Map::new(&runtime, MapOptions::default());
    map.load_style_url(&format!("file://{}", path.display()));
    let started = std::time::Instant::now();
    let error = map.snapshot(64, 64, 1.0).unwrap_err();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(
        error.message().starts_with("style: "),
        "{}",
        error.message()
    );
    assert!(
        started.elapsed().as_secs() < 10,
        "took {:?}",
        started.elapsed()
    );
}

/// A hillshade tile's edge pixels shade from its neighbours' elevations, as
/// in a view: over a plane sloping evenly across tiles, the tile shades
/// evenly to its edges.
#[test]
fn hillshade_tiles_shade_their_edges_from_their_neighbours() {
    const Z: u8 = 12;
    let (col, row) = (2048u32, 2048u32);
    let dir = std::env::temp_dir().join(format!("orbis-tile-dem-{}", std::process::id()));
    // Terrarium tiles of the 3×3 block around the tile, rising 8 m a DEM
    // pixel (some 10 m wide) eastward and 4 m southward.
    for x in col - 1..=col + 1 {
        for y in row - 1..=row + 1 {
            let mut pixels = Vec::with_capacity((TILE * TILE * 4) as usize);
            for py in 0..TILE {
                for px in 0..TILE {
                    let east = (x + 1 - col) * TILE + px;
                    let south = (y + 1 - row) * TILE + py;
                    let metres = 8 * east + 4 * south;
                    let v = (metres + 32768) * 256;
                    pixels.extend([(v >> 16) as u8, (v >> 8) as u8, v as u8, 255]);
                }
            }
            let path = dir.join(format!("{Z}/{x}/{y}.png"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
            let mut encoder = png::Encoder::new(file, TILE, TILE);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&pixels).unwrap();
        }
    }
    let style = format!(
        r#"{{"version":8,
            "sources":{{"dem":{{"type":"raster-dem","encoding":"terrarium","tileSize":{TILE},
              "maxzoom":{Z},"tiles":["file://{}/{{z}}/{{x}}/{{y}}.png"]}}}},
            "layers":[
              {{"id":"bg","type":"background","paint":{{"background-color":"white"}}}},
              {{"id":"shade","type":"hillshade","source":"dem"}}
            ]}}"#,
        dir.display()
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    let pixels = render_tile(&mut map, Z, col, row);
    std::fs::remove_dir_all(&dir).unwrap();
    let at = |x: u32, y: u32| {
        let i = ((y * TILE + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    let middle = at(TILE / 2, TILE / 2);
    let mut worst = (0, 0, 0);
    for i in 0..TILE {
        for (x, y) in [(0, i), (TILE - 1, i), (i, 0), (i, TILE - 1)] {
            let step = (0..3)
                .map(|c| at(x, y)[c].abs_diff(middle[c]))
                .max()
                .unwrap();
            worst = worst.max((step, x, y));
        }
    }
    assert!(middle != [255, 255, 255], "the plane isn't shaded");
    assert!(
        worst.0 <= 2,
        "an edge pixel shades apart: {worst:?} against {middle:?}"
    );
}
