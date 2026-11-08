//! The GPU memory a map holds follows what it draws.

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
        let memory = report.memory;
        assert!(report.tiles_drawn > 0, "z{zoom}: Tokyo draws");
        eprintln!(
            "z{zoom} pitch {pitch}: {:.1} MB allocated",
            memory.allocated() as f64 / 1e6
        );
        assert!(memory.allocated() < 64 << 20, "z{zoom}: {memory:#?}");
    }
}

/// A dropped map frees what its frames made: maps made, drawn and
/// dropped one after another leave the device holding what the runtime
/// shares (its target, the image atlas) and nothing more.
#[test]
fn dropped_maps_leave_nothing_behind() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 512, 512, 1.0);
    let live: Vec<usize> = (0..20)
        .map(|_| {
            let mut map = Map::new(&runtime, MapOptions::default());
            map.set_style_json(&tokyo()).unwrap();
            map.jump_to(&CameraTarget {
                center: Some(Geodetic::new(139.74, 35.685)),
                zoom: Some(15.0),
                ..CameraTarget::default()
            });
            map.render_until_idle(&mut target, Duration::from_secs(60))
                .unwrap();
            drop(map);
            runtime.stats().live_resources
        })
        .collect();
    eprintln!("live resources after each map: {live:?}");
    assert!(live.iter().all(|&n| n == live[0]), "{live:?}");
}
