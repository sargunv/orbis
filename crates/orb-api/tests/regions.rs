//! Offline regions: a downloaded area opens with no network and no HTTP
//! cache. Downloads from OpenFreeMap: `cargo test -p orb-api --test
//! regions -- --ignored`.

use std::time::{Duration, Instant};

use orb_api::{
    CameraState, Geodetic, Map, MapOptions, RegionDefinition, RegionEvent, Runtime, RuntimeOptions,
    TileMatrixSet,
};

const LIBERTY: &str = "https://tiles.openfreemap.org/styles/liberty";

#[test]
#[ignore = "downloads from the network"]
fn a_downloaded_region_opens_offline() {
    let dir = std::env::temp_dir().join(format!("orb-regions-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let options = RuntimeOptions {
        state_dir: Some(dir.clone()),
        ..RuntimeOptions::default()
    };
    let started = Instant::now();
    let runtime = Runtime::new(&options).unwrap();
    let id = runtime
        .download_region(RegionDefinition {
            name: "Imperial Palace".into(),
            style_url: LIBERTY.into(),
            bounds: [139.745, 35.676, 139.760, 35.690],
            zooms: [11.0, 14.0],
        })
        .unwrap();
    let complete = loop {
        match runtime.poll_region_event() {
            Some(RegionEvent::Progress { .. }) => {}
            Some(event) => break event,
            None => {
                assert!(
                    started.elapsed() < Duration::from_secs(180),
                    "download timed out"
                );
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    };
    let RegionEvent::Complete { bytes, failed, .. } = complete else {
        panic!("{complete:?}");
    };
    eprintln!(
        "region {id}: {bytes} bytes in {:.1} s, {failed} failed",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(failed, 0);
    let regions = runtime.regions().unwrap();
    assert!(regions.iter().any(|r| r.id == id && r.complete));
    drop(runtime);

    // Only the region now: no HTTP cache, and no network.
    std::fs::remove_dir_all(dir.join("http")).unwrap();
    // SAFETY: no other thread of this test binary reads the environment.
    unsafe {
        std::env::set_var("ALL_PROXY", "http://127.0.0.1:9");
        std::env::set_var("HTTPS_PROXY", "http://127.0.0.1:9");
    }
    let runtime = Runtime::new(&options).unwrap();
    let mercator = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
    let mut map = Map::new(
        &runtime,
        MapOptions {
            camera: Some(CameraState::new(
                Geodetic::new(139.752, 35.683),
                13.5,
                mercator.projection(),
            )),
            ..MapOptions::default()
        },
    );
    map.load_style_url(LIBERTY);
    let image = map.snapshot(256, 256, 1.0).unwrap();
    let warnings = map.take_warnings();
    // Drawn: more than the background's one colour.
    let first = &image.pixels[..4];
    assert!(image.pixels.chunks(4).any(|p| p != first));
    assert!(
        !warnings.iter().any(|w| w.contains("network")),
        "{warnings:#?}"
    );
    runtime.delete_region(id).unwrap();
    assert!(runtime.regions().unwrap().is_empty());
    drop(runtime);
    std::fs::remove_dir_all(dir).unwrap();
}
