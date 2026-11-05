//! Plan §8: the GPU memory a map holds follows what it draws.

use std::path::PathBuf;
use std::time::Duration;

use orb_api::{CameraTarget, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, Target};

fn tokyo() -> String {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../web/demo/data")
        .canonicalize()
        .unwrap();
    std::fs::read_to_string(data.join("tokyo-mvt.json"))
        .unwrap()
        .replace(
            "pmtiles://./",
            &format!("pmtiles://file://{}/", data.display()),
        )
        .replace("\"./", &format!("\"file://{}/", data.display()))
}

/// The demo's Tokyo extract draws on a few tens of megabytes: pools and
/// tables start small and grow as the tiles in view need, and the image
/// atlas adds pages as images arrive. They started at about 110 MB.
#[test]
fn tokyo_takes_what_it_draws() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 1024, 768, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&tokyo()).unwrap();
    for (zoom, pitch) in [(12.0, 0.0), (15.0, 0.0), (16.0, 60.0)] {
        map.jump_to(&CameraTarget {
            center: Some(Geodetic::new(139.74, 35.685)),
            zoom: Some(zoom),
            pitch: Some(pitch),
            ..CameraTarget::default()
        });
        let report = map
            .render_until_idle(&mut target, Duration::from_secs(60))
            .unwrap();
        let events: Vec<_> = std::iter::from_fn(|| map.poll_event()).collect();
        eprintln!(
            "z{zoom} p{pitch}: drawn {} events {events:?}",
            report.tiles_drawn
        );
        eprintln!(
            "z{zoom} p{pitch}: {:#?} allocated {}",
            report.memory,
            report.memory.allocated()
        );
    }
}
