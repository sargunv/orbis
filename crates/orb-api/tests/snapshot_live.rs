//! A snapshot leaves the live map as it was: hit tests and rendered
//! queries answer for the live view, and its labels stay placed.

mod common;

use glam::DVec2;
use orb_api::{
    CameraState, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, Target, TileMatrixSet,
};

fn query(map: &mut Map, target: &mut Target, at: DVec2) -> String {
    let id = map.query_rendered(at, at);
    common::answer(map, target, id).unwrap()
}

#[test]
fn snapshots_leave_the_live_view_as_it_was() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mercator = TileMatrixSet::web_mercator();
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
        query(&mut map, &mut target, DVec2::new(200.0, 150.0)),
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
    assert_eq!(
        query(&mut map, &mut target, DVec2::new(200.0, 150.0)),
        features
    );

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

/// A target drawing into a BGRA8 texture, as Metal's and most desktops'
/// windows are.
fn bgra_window(runtime: &Runtime, width: u32, height: u32) -> Target {
    use orb_hal_wgpu::wgpu;
    let texture = runtime
        .with_device(|gpu| {
            gpu.device().create_texture(&wgpu::TextureDescriptor {
                label: Some("bgra window"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Bgra8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        })
        .unwrap();
    Target::borrowed(runtime, &texture, 1.0).unwrap()
}

fn tokyo_map(runtime: &Runtime) -> Map {
    let mut map = Map::new(
        runtime,
        MapOptions {
            camera: Some(CameraState::new(
                Geodetic::new(139.755, 35.686),
                15.0,
                TileMatrixSet::web_mercator().projection(),
            )),
            ..MapOptions::default()
        },
    );
    map.load_style_url(&common::tokyo_url());
    map
}

/// Whether any event the map has queued says it laid its tiles out again
/// on a new renderer.
fn restored(map: &mut Map) -> bool {
    std::iter::from_fn(|| map.poll_event())
        .any(|e| matches!(e, orb_api::Event::Map(orb_api::MapEvent::Restored)))
}

#[test]
fn targets_of_another_format_keep_the_maps_tiles() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let want = tokyo_map(&runtime).snapshot(256, 256, 1.0).unwrap();

    // Frames into an RGBA8 texture, then a BGRA8 window: the renderer
    // takes the window's format on, its tiles kept.
    let mut map = tokyo_map(&runtime);
    let mut texture = Target::texture(&runtime, 256, 256, 1.0);
    common::settle(&mut map, &mut texture);
    let mut window = bgra_window(&runtime, 256, 256);
    common::settle(&mut map, &mut window);
    assert!(
        !restored(&mut map),
        "the window's frames took a new renderer"
    );

    // Still output draws in the window's format between its frames.
    for _ in 0..2 {
        let got = map.snapshot(256, 256, 1.0).unwrap();
        common::assert_same("bgra snapshot", 256, (got.pixels, want.pixels.clone()));
        common::settle(&mut map, &mut window);
        assert!(!restored(&mut map), "the snapshot took a new renderer");
    }
}
