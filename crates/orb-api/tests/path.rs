//! Every frame is right for its camera: a map brought to a camera through
//! a gesture draws, once loaded and faded, the pixels a fresh map at rest
//! at that camera draws. Nothing the path left behind (resident tiles of
//! other zooms, rows, cull state, warps, paint written for other zooms)
//! may show.

use std::path::{Path, PathBuf};
use std::time::Duration;

use orb_api::{CameraTarget, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, Target};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 384;

fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../web/demo/data")
        .canonicalize()
        .unwrap()
}

/// The demo's Tokyo extract, with `extra` spliced into the document's top
/// level (a projection, a terrain) and `sources` into its sources.
fn tokyo(extra: &str, sources: &str) -> String {
    let data = data();
    let style = std::fs::read_to_string(data.join("tokyo-mvt.json"))
        .unwrap()
        .replace(
            "pmtiles://./",
            &format!("pmtiles://file://{}/", data.display()),
        )
        .replace("\"./", &format!("\"file://{}/", data.display()));
    let style = style.replacen('{', &format!("{{{extra}"), 1);
    style.replacen("\"sources\": {", &format!("\"sources\": {{{sources}"), 1)
}

/// Tokyo's buildings standing, as extrusions under the v8 `light`.
fn extruded_tokyo() -> String {
    let style = tokyo("", "").replacen(
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

/// A camera: centre, zoom, bearing, pitch.
type At = (Geodetic, f64, f64, f64);

fn camera((center, zoom, bearing, pitch): At) -> CameraTarget {
    CameraTarget {
        center: Some(center),
        zoom: Some(zoom),
        bearing: Some(bearing),
        pitch: Some(pitch),
        ..CameraTarget::default()
    }
}

/// Draws until loaded and idle.
fn settle(map: &mut Map, target: &mut Target) {
    map.render_until_idle(target, Duration::from_secs(60))
        .expect("the map idles");
}

/// What a live map draws at `end` once loaded and faded to idle, after
/// moving there frame by frame from `start` (a gesture: one frame per
/// step, nothing waits for data), and what a fresh map draws at rest at
/// `end`. Labels too: what idle shows depends on the camera and the data
/// alone, not on the order tiles arrived in or what showed on the way.
fn both_ways(style: &str, start: At, end: At, steps: u32) -> (Vec<u8>, Vec<u8>) {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);

    let mut moved = Map::new(&runtime, MapOptions::default());
    moved.set_style_json(style).unwrap();
    moved.jump_to(&camera(start));
    settle(&mut moved, &mut target);
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    for step in 1..=steps {
        let t = f64::from(step) / f64::from(steps);
        let (a, b) = (start.0, end.0);
        moved.jump_to(&camera((
            Geodetic::new(lerp(a.lon, b.lon, t), lerp(a.lat, b.lat, t)),
            lerp(start.1, end.1, t),
            lerp(start.2, end.2, t),
            lerp(start.3, end.3, t),
        )));
        moved.render(&mut target, 0.0).unwrap();
        std::thread::sleep(Duration::from_millis(8));
    }
    settle(&mut moved, &mut target);
    let gestured = target.read_pixels().unwrap();
    let rest = moved.camera();
    drop(moved);
    let (fresh, fresh_rest) = fresh_at(&runtime, &mut target, style, end);
    // At rest, the camera too is what a fresh one is: over terrain, it
    // stands on the same ground at the same height (#104).
    assert_eq!(
        rest, fresh_rest,
        "the camera at rest differs from a fresh map's"
    );
    (gestured, fresh)
}

/// What a fresh map draws at rest at `at`, live, once idle, and its camera.
fn fresh_at(
    runtime: &Runtime,
    target: &mut Target,
    style: &str,
    at: At,
) -> (Vec<u8>, orb_api::CameraState) {
    let mut fresh = Map::new(runtime, MapOptions::default());
    fresh.set_style_json(style).unwrap();
    fresh.jump_to(&camera(at));
    settle(&mut fresh, target);
    (target.read_pixels().unwrap(), fresh.camera())
}

/// Asserts the two pictures equal, to one step in a channel: the GPU cull
/// compacts chunks in no fixed order, so overlapping translucent pieces of
/// one layer may blend in either order and round apart, even between two
/// fresh maps. On failure, writes both and their difference to a fresh
/// directory and names it.
fn assert_same(name: &str, pair: (Vec<u8>, Vec<u8>)) {
    assert_within(name, pair, 1);
}

/// [`assert_same`] to `steps` in a channel.
fn assert_within(name: &str, (gestured, fresh): (Vec<u8>, Vec<u8>), steps: u8) {
    // A map that loaded nothing draws its background alike both ways.
    let colours: std::collections::HashSet<&[u8]> = fresh.chunks(4).collect();
    if colours.len() <= 64 {
        let at = std::env::temp_dir().join(format!("orb-path-{name}-{}.png", std::process::id()));
        write_png(&at, WIDTH, HEIGHT, &fresh);
        panic!(
            "{name}: the map drew next to nothing ({} colours): {}",
            colours.len(),
            at.display()
        );
    }
    let differ: Vec<(usize, u8)> = gestured
        .chunks(4)
        .zip(fresh.chunks(4))
        .enumerate()
        .map(|(i, (a, b))| {
            (
                i,
                a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap(),
            )
        })
        .filter(|&(_, d)| d > steps)
        .collect();
    if differ.is_empty() {
        return;
    }
    let largest = differ.iter().map(|d| d.1).max().unwrap();
    let first: Vec<_> = differ
        .iter()
        .take(4)
        .map(|&(i, _)| {
            let (x, y) = (i % WIDTH as usize, i / WIDTH as usize);
            (x, y, &gestured[i * 4..][..4], &fresh[i * 4..][..4])
        })
        .collect();
    let dir = std::env::temp_dir().join(format!("orb-path-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let diff: Vec<u8> = gestured
        .chunks(4)
        .zip(fresh.chunks(4))
        .flat_map(|(a, b)| {
            let close = a.iter().zip(b).all(|(x, y)| x.abs_diff(*y) <= steps);
            if close {
                [0, 0, 0, 255]
            } else {
                [255, 0, 255, 255]
            }
        })
        .collect();
    for (suffix, pixels) in [("gestured", &gestured), ("fresh", &fresh), ("diff", &diff)] {
        write_png(
            &dir.join(format!("{name}-{suffix}.png")),
            WIDTH,
            HEIGHT,
            pixels,
        );
    }
    panic!(
        "{name}: {} of {} pixels differ from a fresh map's, by up to {largest}; first \
         (x, y, gestured, fresh): {first:?}; see {}",
        differ.len(),
        WIDTH * HEIGHT,
        dir.display()
    );
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
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

/// Inside the extract's data, which covers `[139.725, 35.677, 139.755,
/// 35.697]` (web/demo/data/README.md).
const STATION: Geodetic = Geodetic {
    lon: 139.745,
    lat: 35.6852,
};

#[test]
fn zooming_in_with_a_turn_and_tilt_draws_what_a_fresh_map_does() {
    let start = (Geodetic::new(139.72, 35.67), 12.6, 0.0, 0.0);
    let end = (STATION, 15.4, 35.0, 50.0);
    assert_same("mercator-in", both_ways(&tokyo("", ""), start, end, 30));
}

#[test]
fn extrusions_after_a_gesture_are_a_fresh_map_s() {
    let start = (Geodetic::new(139.72, 35.67), 14.2, 0.0, 0.0);
    let end = (STATION, 15.6, -40.0, 60.0);
    assert_same("extrusions", both_ways(&extruded_tokyo(), start, end, 30));
}

/// Panning around a circle frame by frame, as the bench's moving camera
/// does, stopping six times to idle and compare with a fresh map there.
#[test]
fn panning_frame_by_frame_draws_what_a_fresh_map_does() {
    let style = tokyo("", "");
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    // A circle of about 300 m around the station.
    let at = |f: u32| {
        let angle = std::f64::consts::TAU * f64::from(f) / 120.0;
        let center = Geodetic::new(
            STATION.lon + 0.003 * (angle.cos() - 1.0),
            STATION.lat + 0.0025 * angle.sin(),
        );
        (center, 15.0, 0.0, 0.0)
    };
    map.jump_to(&camera(at(0)));
    settle(&mut map, &mut target);
    for stop in 1..=6 {
        for f in (stop - 1) * 20 + 1..=stop * 20 {
            map.jump_to(&camera(at(f)));
            map.render(&mut target, 0.0).unwrap();
            std::thread::sleep(Duration::from_millis(8));
        }
        settle(&mut map, &mut target);
        let panned = target.read_pixels().unwrap();
        let rest = map.camera();
        let (fresh, fresh_rest) = fresh_at(&runtime, &mut target, &style, at(stop * 20));
        assert_eq!(
            rest, fresh_rest,
            "the camera at rest differs from a fresh map's"
        );
        assert_same(&format!("pan-{stop}"), (panned, fresh));
    }
}

#[test]
fn zooming_out_draws_what_a_fresh_map_does() {
    let start = (STATION, 16.6, -20.0, 30.0);
    let end = (Geodetic::new(139.75, 35.69), 14.3, 10.0, 0.0);
    assert_same("mercator-out", both_ways(&tokyo("", ""), start, end, 30));
}

#[test]
fn a_globe_gesture_draws_what_a_fresh_globe_does() {
    let style = tokyo(r#""projection": {"type": "globe"},"#, "");
    let start = (Geodetic::new(139.6, 35.6), 9.0, 0.0, 0.0);
    let end = (STATION, 12.3, -25.0, 40.0);
    assert_same("globe", both_ways(&style, start, end, 30));
}

/// A directory of DEM tiles, removed when the test is done with it.
struct DemDir(PathBuf);

impl Drop for DemDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Terrarium DEM tiles of `metres(lon, lat)` over `(west, south, east,
/// north)`, zooms 0–`max_zoom`, written to a fresh directory; the source
/// `name`'s JSON, and the directory to keep while it's read.
fn dem(
    name: &str,
    (west, south, east, north): (f64, f64, f64, f64),
    max_zoom: u8,
    metres: impl Fn(f64, f64) -> f64,
) -> (String, DemDir) {
    let dir = std::env::temp_dir().join(format!(
        "orb-path-dem-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    const SIZE: u32 = 256;
    for z in 0..=max_zoom {
        let n = f64::from(1u32 << z);
        let col = |lon: f64| ((lon + 180.0) / 360.0 * n).floor() as u32;
        let row = |lat: f64| {
            let lat = f64::to_radians(lat);
            ((1.0 - lat.tan().asinh() / std::f64::consts::PI) / 2.0 * n).floor() as u32
        };
        for x in col(west)..=col(east) {
            for y in row(north)..=row(south) {
                let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
                for py in 0..SIZE {
                    for px in 0..SIZE {
                        let fx = (f64::from(x) + (f64::from(px) + 0.5) / f64::from(SIZE)) / n;
                        let fy = (f64::from(y) + (f64::from(py) + 0.5) / f64::from(SIZE)) / n;
                        let lon = fx * 360.0 - 180.0;
                        let lat = (std::f64::consts::PI * (1.0 - 2.0 * fy))
                            .sinh()
                            .atan()
                            .to_degrees();
                        let v = ((metres(lon, lat) + 32768.0) * 256.0).round() as u32;
                        pixels.extend([(v >> 16) as u8, (v >> 8) as u8, v as u8, 255]);
                    }
                }
                let path = dir.join(format!("{z}/{x}/{y}.png"));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                write_png(&path, SIZE, SIZE, &pixels);
            }
        }
    }
    let source = format!(
        r#""{name}": {{"type": "raster-dem", "encoding": "terrarium", "tileSize": {SIZE},
            "maxzoom": {max_zoom}, "bounds": [{west}, {south}, {east}, {north}],
            "tiles": ["file://{}/{{z}}/{{x}}/{{y}}.png"]}},"#,
        dir.display()
    );
    (source, DemDir(dir))
}

/// Rolling hills over central Tokyo, zooms 0–11.
fn hills() -> (String, DemDir) {
    dem("hills", (139.2, 35.2, 140.3, 36.2), 11, |lon, lat| {
        600.0
            + 500.0 * (lon * 40.0).sin() * (lat * 55.0).cos()
            + 150.0 * (lon * 170.0 + lat * 90.0).sin()
    })
}

/// Steep ridges over the Valais Alps, 700 to about 4,500 m, zooms 0–11,
/// hillshaded on the terrain under a sky: ridge silhouettes stand against
/// it at a high pitch. Crests are rounded: a crease one DEM pixel wide
/// running across the pixel grid shades as dotted lines (#103).
fn alps() -> (String, DemDir) {
    let (dem, dir) = dem("alps", (7.4, 45.7, 8.4, 46.3), 11, |lon, lat| {
        let ridge = |v: f64| 1.0 - (v.sin().powi(2) + 0.04).sqrt();
        700.0
            + 2600.0 * ridge(lon * 31.0 + lat * 9.0) * ridge(lat * 23.0 - lon * 5.0)
            + 1200.0 * ridge(lon * 97.0 - lat * 61.0)
    });
    let dem = dem.trim_end_matches(',');
    let style = format!(
        r##"{{"version": 8,
            "sources": {{{dem}}},
            "terrain": {{"source": "alps", "exaggeration": 1.2}},
            "sky": {{"sky-color": "#4f8fd9", "horizon-color": "#d6e8fb", "fog-color": "#e4eef8"}},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#d8e0c8"}}}},
              {{"id": "relief", "type": "hillshade", "source": "alps"}}
            ]}}"##
    );
    (style, dir)
}

/// Terrain over local hills: DEM borders come from the neighbours the view
/// draws, not from those a gesture saw (#84).
#[test]
fn a_terrain_gesture_draws_what_a_fresh_map_does() {
    let (hills, _dem) = hills();
    let style = tokyo(
        r#""terrain": {"source": "hills", "exaggeration": 1.5},"#,
        &hills,
    );
    let start = (Geodetic::new(139.70, 35.66), 12.0, 0.0, 30.0);
    let end = (STATION, 13.8, 40.0, 60.0);
    assert_same("terrain", both_ways(&style, start, end, 30));
}

/// Ridges against the sky over the Alps: a turn and tilt to a pitch of
/// 75° draws the silhouettes a fresh map at rest draws (#84).
#[test]
fn an_alpine_gesture_draws_what_a_fresh_map_does() {
    let start = (Geodetic::new(7.95, 46.0), 11.2, 0.0, 30.0);
    let end = (Geodetic::new(7.85, 45.95), 12.4, 60.0, 75.0);
    let (alps, _dem) = alps();
    assert_same("alps", both_ways(&alps, start, end, 30));
}

/// The terrain's cover fits its drape atlas in any view: at a pitch of 85°
/// over the Alps, and at zoom 22 straight down onto a crest, where slopes
/// fall away a kilometre and more past the screen's edges, the cover
/// coarsens its farthest tiles until it fits, rather than draw the terrain
/// flat. The hillshade covers the terrain tiles' quarters, coarser or not,
/// so a gesture down onto the crest draws what a fresh map does.
#[test]
fn every_terrain_view_fits_the_drape_atlas() {
    let (alps, _dem) = alps();
    let crest = Geodetic::new(7.552, 45.897);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 1920, 1080, 1.0);
    for at in [
        (Geodetic::new(7.57349, 45.87721), 14.58, 43.5, 85.0),
        (crest, 22.0, -167.1, 0.0),
    ] {
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&alps).unwrap();
        map.jump_to(&camera(at));
        settle(&mut map, &mut target);
        let warnings: Vec<String> = std::iter::from_fn(|| map.poll_event())
            .filter_map(|event| match event {
                orb_api::Event::Map(orb_api::MapEvent::Warning { message }) => Some(message),
                _ => None,
            })
            .collect();
        assert!(warnings.is_empty(), "at {at:?}: {warnings:?}");
    }
    // Detail enough to compare: the DEM's 11 zooms overzoomed by 8.
    let start = (crest, 16.0, -120.0, 30.0);
    let end = (crest, 19.0, -167.1, 0.0);
    assert_same("crest", both_ways(&alps, start, end, 20));
}

/// The terrain demo's Alps (live tiles), turned at a pitch of 72° with the
/// camera never stopping: once loaded, a frame mid-turn draws what a fresh
/// map at rest at its camera draws, ridges against the sky and every drape
/// slot's edge. What a slot's filter reads past its tile never depends on
/// which tile holds the next slot (#84). Labels, which settle by their own
/// history, are hidden. Needs the network:
/// `cargo test --release -p orb-api --test path -- --ignored alps`.
#[test]
#[ignore = "fetches tiles over the network"]
fn turning_over_the_live_alps_draws_what_a_fresh_map_does() {
    let data = data();
    let style = std::fs::read_to_string(data.join("alps.json")).unwrap();
    let hide = orb_api::Transaction::from_commands(
        r#"[["setLayoutProperty", "peaks", "visibility", "none"],
            ["setLayoutProperty", "places", "visibility", "none"]]"#,
    )
    .unwrap();
    let at = |bearing| (Geodetic::new(7.7491, 46.0207), 11.6, bearing, 72.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut moved = Map::new(&runtime, MapOptions::default());
    moved.set_style_json(&style).unwrap();
    moved.edit(&hide).unwrap();
    moved.jump_to(&camera(at(-72.0)));
    settle(&mut moved, &mut target);
    // Load what the arrival shows, then turn back without waiting.
    moved.jump_to(&camera(at(-32.0)));
    settle(&mut moved, &mut target);
    moved.jump_to(&camera(at(-72.0)));
    settle(&mut moved, &mut target);
    for step in 1..10 {
        moved.jump_to(&camera(at(-72.0 + 4.0 * f64::from(step))));
        moved.render(&mut target, 0.0).unwrap();
    }
    // The camera never stops: it turns half a degree and back each frame
    // until a frame at the bearing has everything loaded.
    for frame in 0.. {
        moved.jump_to(&camera(at(-32.0)));
        let report = moved.render(&mut target, 0.0).unwrap();
        if report.is_some_and(|r| r.tiles_loading == 0 && r.loaded) {
            break;
        }
        assert!(frame < 4000, "tiles did not load");
        moved.jump_to(&camera(at(-31.5)));
        moved.render(&mut target, 0.0).unwrap();
        std::thread::sleep(Duration::from_millis(5));
    }
    let turned = target.read_pixels().unwrap();
    drop(moved);
    let mut fresh = Map::new(&runtime, MapOptions::default());
    fresh.set_style_json(&style).unwrap();
    fresh.edit(&hide).unwrap();
    fresh.jump_to(&camera(at(-32.0)));
    settle(&mut fresh, &mut target);
    // The terrain filters several drape texels into a pixel, each of which
    // may round a step apart (`assert_same`'s translucent pieces): two.
    assert_within("live-alps", (turned, target.read_pixels().unwrap()), 2);
}

/// A draped fill whose hovered colour varies with zoom shows the colour of
/// its new zoom in the first frame after a jump of six zooms: the state
/// rows' zooms follow the camera that frame (#85), as a fresh map at rest
/// draws it.
#[test]
fn a_fast_zoom_drapes_state_paint_of_its_new_zoom() {
    let (lon, lat) = (139.76, 35.68);
    let (hills, _dem) = hills();
    let style = format!(
        r##"{{"version": 8,
            "sources": {{{}
              "square": {{"type": "geojson", "data": {{"type": "Feature", "id": 1, "properties": {{}},
                "geometry": {{"type": "Polygon", "coordinates": [[[{w}, {s}], [{e}, {s}], [{e}, {n}], [{w}, {n}], [{w}, {s}]]]}}}}}}
            }},
            "terrain": {{"source": "hills", "exaggeration": 1}},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {{"id": "square", "type": "fill", "source": "square", "paint": {{"fill-color":
                ["interpolate", ["linear"], ["zoom"],
                  10, ["case", ["boolean", ["feature-state", "hover"], false], "#ff0000", "#00ff00"],
                  18, ["case", ["boolean", ["feature-state", "hover"], false], "#0000ff", "#00ff00"]]}}}}
            ]}}"##,
        hills,
        w = lon - 0.05,
        e = lon + 0.05,
        s = lat - 0.05,
        n = lat + 0.05,
    );
    let hover = orb_api::Transaction::from_commands(
        r#"[["setFeatureState", {"source": "square", "id": 1}, {"hover": true}]]"#,
    )
    .unwrap();
    let centre = |pixels: &[u8]| {
        let at = ((HEIGHT / 2 * WIDTH + WIDTH / 2) * 4) as usize;
        pixels[at..at + 3].to_vec()
    };
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let at = |zoom| (Geodetic::new(lon, lat), zoom, 0.0, 0.0);
    let mut moved = Map::new(&runtime, MapOptions::default());
    moved.set_style_json(&style).unwrap();
    moved.edit(&hover).unwrap();
    moved.jump_to(&camera(at(10.0)));
    settle(&mut moved, &mut target);
    moved.jump_to(&camera(at(16.0)));
    moved.render(&mut target, 0.0).unwrap();
    let jumped = centre(&target.read_pixels().unwrap());
    drop(moved);
    let mut fresh = Map::new(&runtime, MapOptions::default());
    fresh.set_style_json(&style).unwrap();
    fresh.edit(&hover).unwrap();
    fresh.jump_to(&camera(at(16.0)));
    settle(&mut fresh, &mut target);
    let rest = centre(&target.read_pixels().unwrap());
    assert!(
        jumped.iter().zip(&rest).all(|(a, b)| a.abs_diff(*b) <= 2),
        "the first frame at z16 draws {jumped:?}, a fresh map at rest {rest:?}"
    );
}

/// A frame that only moves the camera, with nothing arriving or leaving
/// the view, writes no tile rows and lays out no cull candidates: rows
/// are of tiles' places in the world.
#[test]
fn camera_only_frames_write_no_rows() {
    for style in [
        tokyo("", ""),
        tokyo(r#""projection": {"type": "globe"},"#, ""),
        extruded_tokyo(),
    ] {
        let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
        let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&style).unwrap();
        let at = |step: f64| (STATION, 14.2 + step * 0.01, step, 20.0);
        map.jump_to(&camera(at(0.0)));
        settle(&mut map, &mut target);
        for step in 1..=3 {
            map.jump_to(&camera(at(f64::from(step))));
            let report = map.render(&mut target, 0.0).unwrap().unwrap();
            assert_eq!(report.tiles_loading, 0, "the move loads nothing");
            assert_eq!(report.frame.placements_written, 0, "step {step}");
            assert!(!report.frame.candidates_rebuilt, "step {step}");
            // One cull, and the recorded bundles replayed.
            assert_eq!(report.frame.dispatches, 1, "step {step}");
            assert!(!report.frame.bundle_recorded, "step {step}");
        }
    }
}

/// At zoom 22 a square's corner draws where the CPU projects it, to a
/// pixel: rows' origins and the camera subtract as hi and lo parts on the
/// GPU.
#[test]
fn zoom_22_draws_to_the_pixel() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    // A square far from the world's origin, its south-west corner near
    // the view's centre.
    let (lon, lat) = (139.767_123_4, 35.681_234_5);
    let style = format!(
        r##"{{"version": 8, "sources": {{"square": {{"type": "geojson", "data":
            {{"type": "Polygon", "coordinates": [[[{lon}, {lat}], [{e}, {lat}], [{e}, {n}],
            [{lon}, {n}], [{lon}, {lat}]]]}}}}}},
          "layers": [{{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
            {{"id": "square", "type": "fill", "source": "square",
              "paint": {{"fill-color": "#000000", "fill-antialias": false}}}}]}}"##,
        e = lon + 0.01,
        n = lat + 0.01,
    );
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    // A few metres apart: tens of pixels at zoom 22.
    for (dlon, dlat) in [(0.0, 0.0), (2.37e-6, -1.13e-6), (-3.11e-6, 0.9e-6)] {
        let center = Geodetic::new(lon + dlon, lat + dlat);
        map.jump_to(&camera((center, 22.0, 0.0, 0.0)));
        settle(&mut map, &mut target);
        let pixels = target.read_pixels().unwrap();
        let corner = map.project(Geodetic::new(lon, lat)).expect("in view");
        let dark = |x: u32, y: u32| pixels[((y * WIDTH + x) * 4) as usize] < 128;
        // A row just inside the square, and a column just inside it.
        let row = (corner.y - 4.0) as u32;
        let column = (corner.x + 4.0) as u32;
        let left = (0..WIDTH)
            .find(|&x| dark(x, row))
            .expect("the square shows");
        let bottom = (0..HEIGHT)
            .rev()
            .find(|&y| dark(column, y))
            .expect("the square shows")
            + 1;
        assert!(
            (f64::from(left) - corner.x).abs() <= 1.0
                && (f64::from(bottom) - corner.y).abs() <= 1.0,
            "the corner draws at ({left}, {bottom}), not at {corner}"
        );
    }
}

/// A dense view's memory comes back when nothing holds it: Tokyo at a few
/// views, then open ocean (the extract has no tiles there), where the tile
/// cache still holds the city, then an empty style.
#[test]
fn an_emptied_view_gives_its_memory_back() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&tokyo("", "")).expect("style");
    let mut memory = |map: &mut Map, at: At| {
        map.jump_to(&camera(at));
        let report = map
            .render_until_idle(&mut target, Duration::from_secs(60))
            .expect("the map idles");
        report.memory.allocated()
    };
    let mut dense = 0;
    for (lon, lat, zoom) in [
        (139.767, 35.681, 13.0),
        (139.70, 35.66, 13.5),
        (139.75, 35.70, 14.5),
        (139.79, 35.67, 12.5),
    ] {
        dense = dense.max(memory(
            &mut map,
            (Geodetic::new(lon, lat), zoom, 30.0, 60.0),
        ));
    }
    let ocean = memory(&mut map, (Geodetic::new(150.0, 30.0), 15.5, 30.0, 60.0));
    // The cache keeps what it holds of the city (a quarter of the budget):
    // the pools hold it too.
    assert!(ocean <= dense);
    // Nothing left to draw or keep: the pools go back to a fresh map's.
    let empty = r#"{"version": 8, "sources": {}, "layers": []}"#;
    map.set_style_json(empty).expect("style");
    let emptied = memory(&mut map, (Geodetic::new(150.0, 30.0), 15.5, 30.0, 60.0));
    let pools = |map: &mut Map, target: &mut Target| {
        let report = map
            .render_until_idle(target, Duration::from_secs(60))
            .unwrap();
        let s = report.memory.scene;
        [s.vertices.allocated, s.indices.allocated, s.props.allocated]
    };
    let mut fresh = Map::new(&runtime, MapOptions::default());
    fresh.set_style_json(empty).expect("style");
    assert_eq!(
        pools(&mut map, &mut target),
        pools(&mut fresh, &mut target),
        "dense {} MiB, ocean {} MiB, emptied {} MiB",
        dense >> 20,
        ocean >> 20,
        emptied >> 20
    );
}
