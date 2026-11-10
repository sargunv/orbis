//! A snapshot leaves the live map as it was: hit tests and rendered
//! queries answer for the live view, and its labels stay placed.

mod common;

use std::time::{Duration, Instant};

use glam::DVec2;
use orb_api::{
    CameraState, Event, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, Target, TileMatrixSet,
};

fn query(map: &mut Map, at: DVec2) -> String {
    let id = map.query_rendered(at, at);
    let started = Instant::now();
    loop {
        if let Some(Event::Query { id: got, result }) = map.poll_event()
            && got == id
        {
            return result.unwrap();
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "no query result"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn snapshots_leave_the_live_view_as_it_was() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mercator = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
    let mut map = Map::new(
        &runtime,
        MapOptions {
            camera: Some(CameraState::new(
                Geodetic::new(139.755, 35.686),
                15.0,
                mercator.projection(),
            )),
            ..MapOptions::default()
        },
    );
    map.load_style_url(&common::tokyo_url());
    let mut target = Target::texture(&runtime, 400, 300, 1.0);
    common::settle(&mut map, &mut target);
    while map.poll_event().is_some() {}
    let points: Vec<DVec2> = (0..12)
        .flat_map(|i| {
            (0..9).map(move |j| DVec2::new(16.0 + 33.0 * f64::from(i), 16.0 + 33.0 * f64::from(j)))
        })
        .collect();
    let hits = |map: &Map| -> Vec<Option<String>> {
        points
            .iter()
            .map(|p| map.hit_test(*p).map(|h| format!("{} {:?}", h.layer, h.id)))
            .collect()
    };
    let (before, features, pixels) = (
        hits(&map),
        query(&mut map, DVec2::new(200.0, 150.0)),
        target.read_pixels().unwrap(),
    );
    assert!(features.contains("\"type\":\"Feature\""), "{features}");
    assert!(
        before.iter().filter(|h| h.is_some()).count() > 20,
        "{before:?}"
    );

    // A snapshot of another size than the live view.
    map.snapshot(256, 512, 2.0).unwrap();
    assert_eq!(hits(&map), before, "hit tests after the snapshot");
    assert_eq!(query(&mut map, DVec2::new(200.0, 150.0)), features);

    common::settle(&mut map, &mut target);
    let after = target.read_pixels().unwrap();
    let differing = pixels
        .chunks(4)
        .zip(after.chunks(4))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "the live view drew differently after the snapshot"
    );
}
