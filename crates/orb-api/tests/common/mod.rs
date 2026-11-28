//! What the integration tests share: the demo's Tokyo extract, terrain to
//! put under it ([`dem`]), vector tiles ([`mvt`]), drawing until idle or
//! until something happens, and comparing pictures.

// Each test binary uses some of these.
#![allow(dead_code)]

pub(crate) mod dem;
pub(crate) mod mvt;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use orb_api::{
    CameraState, CameraTarget, Constraints, Event, Geodetic, Map, MapOptions, Runtime, Target,
};

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
    orb_host::file_url(&data().join("tokyo-mvt.json"))
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

/// How long a test waits for what it waits on before it fails, or, given
/// to `render_until_idle`, how long the map may go with nothing arriving:
/// long past anything a loaded machine takes, so that reaching it means a
/// hang.
pub(crate) const DEADLINE: Duration = Duration::from_secs(60);

/// Draws until the map is loaded and idle.
pub(crate) fn settle(map: &mut Map, target: &mut Target) {
    map.render_until_idle(target, DEADLINE)
        .expect("the map idles");
}

/// Milliseconds on the test's clock, which frames and input share.
pub(crate) fn now() -> f64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// Draws the frames the map asks for, on the test's clock, as a host
/// does, waiting for its wakes in between, until `ready` gives a value:
/// `what` names it should it never come.
pub(crate) fn until<T>(
    map: &mut Map,
    target: &mut Target,
    what: &str,
    ready: impl FnMut(&mut Map, &Target) -> Option<T>,
) -> T {
    frames_until(map, target, now, what, ready)
}

/// [`until`] with every frame at `time`, for tests that hold the clock
/// still.
pub(crate) fn until_at<T>(
    map: &mut Map,
    target: &mut Target,
    time: f64,
    what: &str,
    ready: impl FnMut(&mut Map, &Target) -> Option<T>,
) -> T {
    frames_until(map, target, || time, what, ready)
}

fn frames_until<T>(
    map: &mut Map,
    target: &mut Target,
    clock: impl Fn() -> f64,
    what: &str,
    mut ready: impl FnMut(&mut Map, &Target) -> Option<T>,
) -> T {
    let deadline = Instant::now() + DEADLINE;
    loop {
        if let Some(value) = ready(map, target) {
            return value;
        }
        assert!(Instant::now() < deadline, "no {what} in {DEADLINE:?}");
        if map.needs_frame() {
            map.render(target, clock()).unwrap();
        }
        // A display's frame at most: a wake since the frame ends it now.
        map.wait_woken(Duration::from_millis(16));
    }
}

/// The answer to query `id`, drawing the frames the map asks for while it
/// waits; other events are dropped.
pub(crate) fn answer(map: &mut Map, target: &mut Target, id: u64) -> Result<String, String> {
    until(map, target, "query answer", |map, _| {
        std::iter::from_fn(|| map.poll_event()).find_map(|event| match event {
            Event::Query { id: done, result } if done == id => Some(result),
            _ => None,
        })
    })
}

/// Waits, drawing nothing, until something wakes the idle map: for what
/// holds between a wake and the frame it asks for.
pub(crate) fn wake(map: &Map) {
    let deadline = Instant::now() + DEADLINE;
    while !map.needs_frame() {
        assert!(Instant::now() < deadline, "no wake in {DEADLINE:?}");
        map.wait_woken(Duration::from_secs(1));
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
