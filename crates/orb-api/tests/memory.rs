//! The GPU memory a map holds follows what it draws.

mod common;

use std::time::Duration;

use orb_api::{CameraTarget, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, Target};

/// The demo's Tokyo extract draws on a few tens of megabytes: pools and
/// tables start small and grow as the tiles in view need, and the image
/// atlas adds pages as images arrive. They started at about 110 MB.
#[test]
fn tokyo_takes_what_it_draws() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 1024, 768, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.load_style_url(&common::tokyo_url());
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
            map.load_style_url(&common::tokyo_url());
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

/// The background's meshes take room in the scene's pools as a tile's do,
/// growing them when they're full: on the globe each background layer
/// draws a grid of 65 × 65 vertices, and forty of them outgrow the pools
/// a new map starts with. Making it used to panic, its pools ungrown, as
/// a style switch's could in the browser once tiles had filled them.
#[test]
fn a_background_grows_the_pools_it_needs() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 64, 64, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    let layers: Vec<String> = (0..40)
        .map(|i| {
            format!(
                r##"{{"id":"bg{i}","type":"background","paint":{{"background-color":"#00ff00","background-opacity":0.1}}}}"##
            )
        })
        .collect();
    map.set_style_json(&format!(
        r#"{{"version":8,"projection":{{"type":"globe"}},"sources":{{}},"layers":[{}]}}"#,
        layers.join(",")
    ))
    .unwrap();
    common::settle(&mut map, &mut target);
}
