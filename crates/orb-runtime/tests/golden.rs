//! The modern path's golden pass streams: what a settled frame of each
//! scene submits (every pass, and each bundle's commands, through
//! `orb_hal::Recording`), against `tests/data/golden`. A change to how
//! frames are encoded shows here as a diff to review, so refactors that
//! should encode the same frames (the WebGL 2 path's driver seam, #140)
//! prove they do. `UPDATE_GOLDEN=1` rewrites the files.

mod common;

use std::sync::Arc;

use common::scenes::scenes;
use common::{Geodetic, Map, TargetFormat, Viewport, WgpuGpu};
use orb_hal::{Gpu, Recording};

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/golden");
const SIZE: (u32, u32) = (512, 384);

/// A map of the style at `url`, settled, and the text of one frame more,
/// after `edit` if given (which a frame of a settled map may not draw
/// otherwise: the terrain's drape redraws on a paint change).
fn stream(url: &str, at: (f64, f64, f64), previews: bool, edit: Option<&str>) -> String {
    let mut gpu = Recording::new(pollster::block_on(WgpuGpu::offscreen()).unwrap());
    let format = TargetFormat {
        color: orb_hal::TextureFormat::Rgba8Unorm,
        sample_count: 1,
    };
    let target = gpu.create_texture(&orb_hal::TextureDesc {
        label: "target",
        width: SIZE.0,
        height: SIZE.1,
        array_layers: 1,
        dimension: orb_hal::TextureViewDimension::D2,
        format: format.color,
        usage: orb_hal::TextureUsages::RENDER_ATTACHMENT | orb_hal::TextureUsages::COPY_SRC,
        sample_count: 1,
        mip_levels: 1,
        view_formats: &[],
    });
    let (zoom, bearing, pitch) = at;
    let camera = common::camera(&common::Options {
        style: None,
        encoding: "mvt".into(),
        at: Some((Geodetic::new(139.74, 35.685), zoom, bearing, pitch)),
    });
    let mut map = Map::new(common::native_services(Arc::new(|| {})), camera);
    map.set_clear_color([1.0; 4]);
    map.set_stable_labels(false);
    if previews {
        map.set_previews(orb_runtime::Previews::ALL);
    }
    map.load_style_url(url);
    let viewport = Viewport {
        width: f64::from(SIZE.0),
        height: f64::from(SIZE.1),
    };
    let mut renderer = map.create_renderer(&mut gpu, format);
    let mut time = 0.0;
    let mut frame = |map: &mut Map, gpu: &mut Recording<WgpuGpu>| {
        time += 1000.0 / 60.0;
        map.render(gpu, &mut renderer, target, viewport, 1.0, time)
    };
    let started = std::time::Instant::now();
    let mut idle = 0;
    while idle < 3 {
        let report = frame(&mut map, &mut gpu);
        for warning in common::take_warnings(&mut map) {
            eprintln!("{url}: {warning}");
        }
        idle = if report.idle { idle + 1 } else { 0 };
        assert!(
            started.elapsed().as_secs() < 60,
            "{url}: timed out settling"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    if let Some(edit) = edit {
        let tx = orb_runtime::Transaction::from_commands(edit).unwrap();
        map.edit(&tx).unwrap();
    }
    gpu.start();
    frame(&mut map, &mut gpu);
    gpu.finish()
}

fn check(name: &str, text: &str) {
    let path = format!("{GOLDEN}/{name}.txt");
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(GOLDEN).unwrap();
        std::fs::write(&path, text).unwrap();
        return;
    }
    let golden = std::fs::read_to_string(&path).unwrap_or_default();
    if golden != text {
        let first = (golden.lines().zip(text.lines()))
            .position(|(a, b)| a != b)
            .unwrap_or(golden.lines().count().min(text.lines().count()));
        let line = |s: &str| s.lines().nth(first).unwrap_or("").to_string();
        panic!(
            "{name}: the frame's passes differ from {path} at line {}:\n  golden: {}\n  now:    {}\nUPDATE_GOLDEN=1 rewrites it, once the change is the one meant",
            first + 1,
            line(&golden),
            line(text),
        );
    }
}

/// Checks the streams of the scenes named `names` (`common::scenes`).
fn check_scenes(names: &[&str]) {
    for scene in scenes(names) {
        check(
            scene.name,
            &stream(&scene.url, scene.at, scene.previews, scene.edit),
        );
    }
}

/// Liberty's vector layers and labels, flat and pitched.
#[test]
fn liberty() {
    check_scenes(&["liberty-z12", "liberty-z15-pitched"]);
}

/// The globe from afar, with its sky.
#[test]
fn globe() {
    check_scenes(&["globe"]);
}

/// Terrain from a DEM, the vector layers draped on it, and a hillshade.
#[test]
fn terrain() {
    check_scenes(&["terrain", "terrain-drape"]);
}

/// A heatmap of the points of interest, and circles.
#[test]
fn heatmap() {
    check_scenes(&["heatmap"]);
}

/// Raster imagery under the vector layers.
#[test]
fn raster() {
    check_scenes(&["raster"]);
}

/// Liberty's 3D buildings, multisampled, and with shadows and ambient
/// occlusion (previews).
#[test]
fn extrusions() {
    check_scenes(&["extrusions", "extrusions-lit"]);
}

/// Colours mixed in linear light (`dev.sargunv.orbis:color-space`).
#[test]
fn linear_light() {
    check_scenes(&["linear"]);
}
