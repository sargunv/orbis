//! The modern path's golden pass streams: what a settled frame of each
//! scene submits (every pass, and each bundle's commands, through
//! `orb_hal::Recording`), against `tests/data/golden`. A change to how
//! frames are encoded shows here as a diff to review, so refactors that
//! should encode the same frames (the WebGL 2 path's driver seam, #140)
//! prove they do. `UPDATE_GOLDEN=1` rewrites the files.

mod common;

use common::WgpuGpu;
use common::scenes::{scenes, settle};
use orb_hal::Recording;

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/golden");

/// The text of `scene`'s last frame (`scenes::settle`).
fn stream(scene: &common::scenes::Scene) -> String {
    let mut gpu = Recording::new(pollster::block_on(WgpuGpu::offscreen()).unwrap());
    settle(&mut gpu, scene, Recording::start);
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
        check(scene.name, &stream(&scene));
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

/// Liberty's 3D buildings, and multisampled under a light of their own.
#[test]
fn extrusions() {
    check_scenes(&["extrusions", "extrusions-lit"]);
}
