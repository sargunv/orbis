use orb_camera::{CameraState, View, Viewport};
use orb_crs::TileMatrixSet;
use orb_geo::Geodetic;

#[test]
fn covers_match_pinned_gl_js() {
    let golden: serde_json::Value = serde_json::from_str(include_str!("data/covers.json")).unwrap();
    let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
    let mut failures = Vec::new();
    for case in golden["cases"].as_array().unwrap() {
        let n = |key: &str| case[key].as_f64().unwrap();
        let mut camera = CameraState::new(Geodetic::new(n("lon"), n("lat")), n("zoom"));
        camera.pitch = n("pitch");
        camera.bearing = n("bearing");
        camera.fov = n("fov");
        let view = View::new(
            &camera,
            Viewport {
                width: n("width"),
                height: n("height"),
            },
            tms.projection(),
        );
        let mut actual: Vec<_> = view
            .cover_volume(
                &tms,
                n("tileSize"),
                n("maxzoom") as u8,
                n("maxContentElevation") * tms.projection().style_height_scale(camera.target),
                None,
            )
            .iter()
            .filter(|t| t.index.z >= n("minzoom") as u8)
            .map(|t| {
                [
                    i64::from(t.wrap),
                    i64::from(t.index.z),
                    i64::from(t.index.col),
                    i64::from(t.index.row),
                ]
            })
            .collect();
        actual.sort();
        let expected: Vec<[i64; 4]> = serde_json::from_value(case["tiles"].clone()).unwrap();
        if actual != expected {
            let extra: Vec<_> = actual.iter().filter(|t| !expected.contains(t)).collect();
            let missing: Vec<_> = expected.iter().filter(|t| !actual.contains(t)).collect();
            failures.push(format!(
                "{}: extra {extra:?}, missing {missing:?}",
                case["name"]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} differing covers:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manhattan_cover_counts() {
    let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
    for pitch in [60., 80., 85.] {
        let mut camera = CameraState::new(Geodetic::new(-74.0113, 40.7075), 14.48);
        camera.bearing = 30.;
        camera.pitch = pitch;
        let view = View::new(
            &camera,
            Viewport {
                width: 1024.,
                height: 768.,
            },
            tms.projection(),
        );
        for maxzoom in [14, 24] {
            let mut counts = std::collections::BTreeMap::new();
            for tile in view.cover(&tms, 512., maxzoom) {
                *counts.entry(tile.index.z).or_insert(0) += 1;
            }
            println!("Manhattan pitch {pitch}, maxzoom {maxzoom}: {counts:?}");
        }
    }
}
