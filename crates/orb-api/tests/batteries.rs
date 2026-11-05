//! Markers and the location layer: drawn over the style, named in hits,
//! and back after the style is replaced.

use std::time::{Duration, Instant};

use glam::DVec2;
use orb_api::{Event, Location, Map, MapOptions, Marker, Runtime, RuntimeOptions, Target};

const STYLE: &str = r##"{"version": 8, "center": [0, 0], "zoom": 4, "sources": {},
    "layers": [{"id": "bg", "type": "background", "paint": {"background-color": "#ffffff"}}]}"##;

fn settle(map: &mut Map, target: &mut Target) {
    map.render_until_idle(target, Duration::from_secs(30))
        .expect("the map idles");
}

/// Pixels of a colour: red-ish (the pin) or blue-ish (the location).
fn count(target: &Target, test: impl Fn(&[u8]) -> bool) -> usize {
    target
        .read_pixels()
        .unwrap()
        .chunks(4)
        .filter(|p| test(p))
        .count()
}
fn red(p: &[u8]) -> bool {
    p[0] > 180 && p[1] < 90 && p[2] < 90
}
fn blue(p: &[u8]) -> bool {
    p[2] > 180 && p[0] < 90
}

#[test]
fn markers_and_the_location_draw_over_any_style() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 256, 256, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(STYLE).unwrap();
    settle(&mut map, &mut target);

    let id = map.add_marker(Marker {
        position: [0.0, 0.0],
        image: None,
        anchor: None,
    });
    map.set_location(Some(Location {
        position: [0.0, -4.0],
        accuracy: 300_000.0,
        heading: Some(45.0),
    }));
    settle(&mut map, &mut target);
    assert!(count(&target, red) > 100, "the pin");
    assert!(count(&target, blue) > 100, "the location dot");
    // The pin's head is above the point: its tip is at the centre.
    let pin = DVec2::new(128.0, 112.0);
    let hit = map.hit_test(pin).expect("the pin");
    assert_eq!(hit.layer, "orbis-markers");
    // Rendered queries name the marker by its id.
    let query = map.query_rendered(pin, pin);
    let started = Instant::now();
    let features = loop {
        if let Some(Event::Query { id: got, result }) = map.poll_event()
            && got == query
        {
            break result.unwrap();
        }
        assert!(started.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(1));
    };
    assert!(features.contains(&format!(r#""id":{id}"#)), "{features}");

    // A new style: they come back.
    map.set_style_json(&STYLE.replace("#ffffff", "#eeeeee"))
        .unwrap();
    settle(&mut map, &mut target);
    assert!(count(&target, red) > 100 && count(&target, blue) > 100);

    map.remove_marker(id).unwrap();
    map.set_location(None);
    settle(&mut map, &mut target);
    assert_eq!((count(&target, red), count(&target, blue)), (0, 0));
    assert!(map.remove_marker(id).is_err());
}
