//! What the integration tests share: the demo's Tokyo extract, terrain to
//! put under it ([`dem`]), drawing until idle, and comparing pictures.

// Each test binary uses some of these.
#![allow(dead_code)]

pub(crate) mod dem;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use orb_api::{CameraState, CameraTarget, Constraints, Geodetic, Map, MapOptions, Runtime, Target};

/// The demo's data directory: the Tokyo extract, its glyphs and sprites.
pub(crate) fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../web/demo/data")
        .canonicalize()
        .unwrap()
}

/// The demo's Tokyo extract with Liberty's style, as a URL its relative
/// URLs resolve against.
pub(crate) fn tokyo_url() -> String {
    format!("file://{}", data().join("tokyo-mvt.json").display())
}

/// The Tokyo extract's style as text, its relative URLs made absolute: for
/// tests that change it before setting it.
pub(crate) fn tokyo_json() -> String {
    let data = data();
    std::fs::read_to_string(data.join("tokyo-mvt.json"))
        .unwrap()
        .replace(
            "pmtiles://./",
            &format!("pmtiles://file://{}/", data.display()),
        )
        .replace("\"./", &format!("\"file://{}/", data.display()))
}

/// Draws until the map is loaded and idle.
pub(crate) fn settle(map: &mut Map, target: &mut Target) {
    map.render_until_idle(target, Duration::from_secs(60))
        .expect("the map idles");
}

/// A directory of a test's own under the system's temporary one, removed
/// with what it holds when it drops, as a failing test unwinds too.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    pub(crate) fn new(name: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("orb-{name}-{}-{n}", std::process::id()));
        // What a killed run with this process id left.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl std::ops::Deref for Scratch {
    type Target = std::path::Path;
    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

impl AsRef<std::path::Path> for Scratch {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The demo's Tokyo extract, with `extra` spliced into the document's top
/// level (a projection, a terrain) and `sources` into its sources.
pub(crate) fn tokyo(extra: &str, sources: &str) -> String {
    let style = tokyo_json().replacen('{', &format!("{{{extra}"), 1);
    style.replacen("\"sources\": {", &format!("\"sources\": {{{sources}"), 1)
}

/// `style` (Tokyo's) with its buildings standing.
pub(crate) fn extruded(style: String) -> String {
    let style = style.replacen(
        "{\n      \"id\": \"building\",",
        r##"{"id": "building-3d", "type": "fill-extrusion", "source": "openmaptiles",
            "source-layer": "building", "minzoom": 14,
            "paint": {"fill-extrusion-color": "#d9d0c9",
                      "fill-extrusion-height": ["get", "render_height"]}},
        {"id": "building","##,
        1,
    );
    assert!(style.contains("building-3d"));
    style
}

/// Inside the extract's data, which covers `[139.725, 35.677, 139.755,
/// 35.697]` (web/demo/data/README.md).
pub(crate) const STATION: Geodetic = Geodetic {
    lon: 139.745,
    lat: 35.6852,
};

/// Lets a camera tilt to 85 degrees, past the default 60.
pub(crate) fn tilted() -> Constraints {
    Constraints {
        max_pitch: 85.0,
        ..Constraints::default()
    }
}

/// Lets a globe's camera look past its horizon.
pub(crate) fn steep() -> Constraints {
    Constraints {
        max_pitch: 179.0,
        ..Constraints::default()
    }
}

/// A map's options for tests that compare pictures: labels that depend on
/// the camera alone, which MapLibre Native's stabilisers (on by default)
/// don't, and every preview, for the styles that use one.
pub(crate) fn options() -> MapOptions {
    MapOptions {
        stable_labels: false,
        previews: orb_api::Previews::ALL,
        ..MapOptions::default()
    }
}

/// The cull dispatches a frame runs: one on the GPU driver, none on the
/// CPU's (`cargo xtask check` runs every test here on it too).
pub(crate) fn culls() -> u32 {
    let driver = std::env::var("ORB_DRIVER").unwrap_or_default();
    u32::from(driver.is_empty() || driver == "gpu")
}

/// A camera: centre, zoom, bearing, pitch.
pub(crate) type At = (Geodetic, f64, f64, f64);

pub(crate) fn camera((center, zoom, bearing, pitch): At) -> CameraTarget {
    CameraTarget {
        center: Some(center),
        zoom: Some(zoom),
        bearing: Some(bearing),
        pitch: Some(pitch),
        ..CameraTarget::default()
    }
}

/// What a fresh map draws at rest at `at`, live, once idle, and its camera.
pub(crate) fn fresh_at(
    runtime: &Runtime,
    target: &mut Target,
    style: &str,
    at: At,
) -> (Vec<u8>, CameraState) {
    fresh(runtime, target, style, |map| map.jump_to(&camera(at)))
}

/// What a fresh map draws at rest where `place` puts it, live, once idle,
/// and its camera.
pub(crate) fn fresh(
    runtime: &Runtime,
    target: &mut Target,
    style: &str,
    place: impl FnOnce(&mut Map),
) -> (Vec<u8>, CameraState) {
    let mut fresh = Map::new(runtime, options());
    fresh.set_constraints(steep());
    fresh.set_style_json(style).unwrap();
    place(&mut fresh);
    settle(&mut fresh, target);
    (target.read_pixels().unwrap(), fresh.camera())
}

/// Asserts the two pictures, `width` pixels wide, equal: what a test drew
/// and what it expected. On failure, writes both and their difference to
/// a fresh directory and names it.
pub(crate) fn assert_same(name: &str, width: u32, (got, want): (Vec<u8>, Vec<u8>)) {
    let height = (want.len() / 4) as u32 / width;
    // A map that loaded nothing draws its background alike both ways.
    let colours: HashSet<&[u8]> = want.chunks(4).collect();
    if colours.len() <= 64 {
        let at = std::env::temp_dir().join(format!("orb-{name}-{}.png", std::process::id()));
        write_png(&at, width, height, &want);
        panic!(
            "{name}: the map drew next to nothing ({} colours): {}",
            colours.len(),
            at.display()
        );
    }
    let differ: Vec<(usize, u8)> = got
        .chunks(4)
        .zip(want.chunks(4))
        .enumerate()
        .map(|(i, (a, b))| {
            (
                i,
                a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap(),
            )
        })
        .filter(|&(_, d)| d > 0)
        .collect();
    if differ.is_empty() {
        return;
    }
    let largest = differ.iter().map(|d| d.1).max().unwrap();
    let first: Vec<_> = differ
        .iter()
        .take(4)
        .map(|&(i, _)| {
            let (x, y) = (i % width as usize, i / width as usize);
            (x, y, &got[i * 4..][..4], &want[i * 4..][..4])
        })
        .collect();
    let dir = std::env::temp_dir().join(format!("orb-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let diff: Vec<u8> = got
        .chunks(4)
        .zip(want.chunks(4))
        .flat_map(|(a, b)| {
            if a == b {
                [0, 0, 0, 255]
            } else {
                [255, 0, 255, 255]
            }
        })
        .collect();
    for (suffix, pixels) in [("got", &got), ("want", &want), ("diff", &diff)] {
        write_png(&dir.join(format!("{suffix}.png")), width, height, pixels);
    }
    panic!(
        "{name}: {} of {} pixels differ, by up to {largest}; first (x, y, got, want): \
         {first:?}; see {}",
        differ.len(),
        width * height,
        dir.display()
    );
}

/// Writes `rgba`, `width` by `height` pixels, to a PNG at `path`.
pub(crate) fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    let file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(rgba)
        .unwrap();
}
