use orb_camera::{CameraState, View, Viewport};
use orb_crs::Geodetic;
use orb_crs::Surface;
use orb_crs::TileMatrixSet;

#[test]
fn covers_match_pinned_gl_js() {
    let golden: serde_json::Value = serde_json::from_str(include_str!("data/covers.json")).unwrap();
    let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
    let mut failures = Vec::new();
    for case in golden["cases"].as_array().unwrap() {
        let n = |key: &str| case[key].as_f64().unwrap();
        let mut camera = CameraState::new(
            Geodetic::new(n("lon"), n("lat")),
            n("zoom"),
            tms.projection(),
        );
        camera.pitch = n("pitch");
        camera.bearing = n("bearing");
        camera.fov = n("fov");
        let view = View::new(
            &camera,
            Viewport {
                width: n("width"),
                height: n("height"),
            },
            &Surface::from(tms.projection()),
        );
        let mut actual: Vec<_> = view
            .cover_volume(
                &tms,
                n("tileSize"),
                n("maxzoom") as u8,
                n("maxContentElevation") * tms.projection().style_height_scale(camera.target),
                None,
                0.0,
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

/// The globe's cover against Web Mercator's for the same cameras (plan
/// §18's budget: globe cover within 1.5× Mercator's time). Prints counts and
/// times; run with `--release -- --nocapture`.
#[test]
fn globe_cover_cost() {
    let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
    let projection = *tms.projection();
    let viewport = Viewport {
        width: 1024.0,
        height: 768.0,
    };
    let (mut plane_total, mut globe_total) = (0.0, 0.0);
    for zoom in [1.0, 3.0, 5.0, 8.0, 11.0] {
        for pitch in [0.0, 45.0, 70.0] {
            let mut camera = CameraState::new(Geodetic::new(139.7, 35.7), zoom, &projection);
            camera.pitch = pitch;
            camera.bearing = 20.0;
            let time = |surface: Surface| {
                let view = View::new(&camera, viewport, &surface);
                let start = std::time::Instant::now();
                let mut tiles = 0;
                for _ in 0..50 {
                    tiles = view.cover(&tms, 512.0, 22).len();
                }
                (tiles, start.elapsed().as_secs_f64() * 1e3 / 50.0)
            };
            let (plane_tiles, plane_ms) = time(Surface::plane(projection));
            let (globe_tiles, globe_ms) = time(Surface {
                projection,
                globe: Some(orb_crs::Globe::V8),
            });
            plane_total += plane_ms;
            globe_total += globe_ms;
            println!(
                "z{zoom} pitch {pitch}: Mercator {plane_tiles} tiles {plane_ms:.3} ms, globe {globe_tiles} tiles {globe_ms:.3} ms"
            );
            assert!(globe_tiles > 0 && globe_tiles < 200);
        }
    }
    println!(
        "total: Mercator {plane_total:.2} ms, globe {globe_total:.2} ms ({:.2}×)",
        globe_total / plane_total
    );
}
