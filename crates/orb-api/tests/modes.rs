//! Still and Tile output (plan §14.2): the same inputs draw the same bytes,
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

/// Labels a few pixels apart over a few tiles at zoom 10, each longer than
/// the gap: a random field, and a row in which each overlaps the next, so
/// that which ones show along it runs in one chain of collisions from its
/// west end.
fn crowd() -> String {
    let mut features = Vec::new();
    let mut feature = |name: String, lon: f64, lat: f64| {
        features.push(format!(
            r#"{{"type":"Feature","properties":{{"name":"{name}"}},"geometry":{{"type":"Point","coordinates":[{lon},{lat}]}}}}"#
        ));
    };
    // 41 pixels apart at zoom 10, west to east: placed against the view, the
    // row shows another label at an edge when its west end is out of view.
    for i in 0..60 {
        feature(format!("chain{i:03}"), -0.6 + f64::from(i) * 0.0283, 0.6);
    }
    let mut seed = 7u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        f64::from(seed >> 8) / f64::from(1u32 << 24)
    };
    for i in 0..1200 {
        let (lon, lat) = (-0.5 + next() * 1.4, -0.5 + next() * 0.8);
        feature(format!("{}{i}", "label".repeat(1 + i % 3)), lon, lat);
    }
    format!(
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
    )
}

fn mercator() -> TileMatrixSet {
    TileMatrixSet::by_id("WebMercatorQuad").unwrap()
}

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

/// A label: text, layer and bounds in pixels of the grid at its zoom.
type Label = (String, String, [f64; 4]);

/// Draws the 3×3 tiles around `at`, and the middle one again. Returns the
/// labels that cross an edge two tiles share and show on both sides, those
/// that show on one, and whether the middle tile drew the same bytes again.
fn edges(style: &str, at: Geodetic, z: u8, size: u32) -> (usize, Vec<Label>, bool) {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).unwrap();
    let tms = mercator();
    let p = tms.projection().forward(at);
    let center = tms
        .tiles_in(orb_crs::WorldRect { min: p, max: p }, z)
        .next()
        .unwrap();
    let mut tile = |col: u32, row: u32| -> (Vec<u8>, Vec<Label>) {
        let request = TileRequest {
            tile_matrix_set: tms.clone(),
            z,
            col,
            row,
            size,
            scale: 1.0,
        };
        let pixels = map.render_tile(&request).unwrap().pixels;
        // The view holds the tile and a tile around it.
        let s = f64::from(size);
        let (x0, y0) = ((f64::from(col) - 1.0) * s, (f64::from(row) - 1.0) * s);
        let [left, top] = [f64::from(col) * s, f64::from(row) * s];
        let labels = map
            .semantic_snapshot()
            .map(|s| s.nodes)
            .unwrap_or_default()
            .into_iter()
            .map(|n| {
                let b = n.bounds;
                (
                    n.text,
                    n.layer,
                    [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0],
                )
            })
            .filter(|(_, _, b)| b[2] > left && b[0] < left + s && b[3] > top && b[1] < top + s)
            .collect();
        (pixels, labels)
    };
    let first = tile(center.col, center.row);
    let mut tiles = Vec::new();
    for row in center.row - 1..=center.row + 1 {
        for col in center.col - 1..=center.col + 1 {
            tiles.push(((col, row), tile(col, row).1));
        }
    }
    let repeats = tile(center.col, center.row).0 == first.0;

    let same = |a: &Label, b: &Label| {
        a.0 == b.0 && a.1 == b.1 && a.2.iter().zip(b.2).all(|(x, y)| (x - y).abs() < 0.01)
    };
    let (mut agree, mut differ) = (0, Vec::new());
    for (i, ((ca, ra), a)) in tiles.iter().enumerate() {
        for ((cb, rb), b) in &tiles[i + 1..] {
            if ca.abs_diff(*cb) + ra.abs_diff(*rb) != 1 {
                continue;
            }
            let edge = f64::from(if ca != cb { ca.max(cb) } else { ra.max(rb) } * size);
            let crosses = |l: &&Label| {
                let (lo, hi) = if ca != cb {
                    (l.2[0], l.2[2])
                } else {
                    (l.2[1], l.2[3])
                };
                lo < edge && hi > edge
            };
            for (x, y) in [(a, b), (b, a)] {
                for l in x.iter().filter(crosses) {
                    if y.iter().any(|m| same(l, m)) {
                        agree += 1;
                    } else {
                        differ.push(l.clone());
                    }
                }
            }
        }
    }
    (agree / 2, differ, repeats)
}

#[test]
fn tiles_repeat_and_agree_across_edges() {
    for (name, style, at, z, size) in [
        ("tokyo", tokyo(), Geodetic::new(139.755, 35.686), 15, 512),
        ("crowd", crowd(), Geodetic::new(0.2, 0.2), 10, 512),
    ] {
        let (agree, differ, repeats) = edges(&style, at, z, size);
        eprintln!(
            "{name}: {agree} labels agree across edges, {} differ",
            differ.len()
        );
        assert!(repeats, "{name}: a tile differs on repeat");
        assert!(agree > 0, "{name}: no labels crossed the edges");
        assert!(
            differ.is_empty(),
            "{name}: labels differ across edges: {differ:?}"
        );
    }
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
