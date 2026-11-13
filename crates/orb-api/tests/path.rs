//! Every frame is right for its camera: a map brought to a camera through
//! a gesture draws, once loaded and faded, the pixels a fresh map at rest
//! at that camera draws. Nothing the path left behind (resident tiles of
//! other zooms, rows, cull state, warps, paint written for other zooms)
//! may show.

mod common;

use std::path::Path;
use std::time::Duration;

use orb_api::{
    CameraTarget, Constraints, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, Target,
};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 384;

/// The demo's Tokyo extract, with `extra` spliced into the document's top
/// level (a projection, a terrain) and `sources` into its sources.
fn tokyo(extra: &str, sources: &str) -> String {
    let style = common::tokyo_json().replacen('{', &format!("{{{extra}"), 1);
    style.replacen("\"sources\": {", &format!("\"sources\": {{{sources}"), 1)
}

/// Tokyo's buildings standing, as extrusions under the v8 `light`.
fn extruded_tokyo() -> String {
    extruded(tokyo("", ""))
}

/// `style` (Tokyo's) with its buildings standing.
fn extruded(style: String) -> String {
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

/// Lets a globe's camera look past its horizon.
fn steep() -> Constraints {
    Constraints {
        max_pitch: 179.0,
        ..Constraints::default()
    }
}

/// A map's options for these tests: labels that depend on the camera
/// alone, which MapLibre Native's stabilisers (on by default) don't, and
/// every preview, for the styles that use one.
fn options() -> MapOptions {
    MapOptions {
        stable_labels: false,
        previews: orb_api::Previews::ALL,
        ..MapOptions::default()
    }
}

/// The cull dispatches a frame runs: one on the GPU driver, none on the
/// CPU's (`ORB_DRIVER=cpu` runs these tests on it).
fn culls() -> u32 {
    let driver = std::env::var("ORB_DRIVER").unwrap_or_default();
    u32::from(driver.is_empty() || driver == "gpu")
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

/// What a live map draws at `end` once loaded and faded to idle, after
/// moving there frame by frame from `start` (a gesture: one frame per
/// step, nothing waits for data), and what a fresh map draws at rest at
/// `end`. Labels too: what idle shows depends on the camera and the data
/// alone, not on the order tiles arrived in or what showed on the way.
fn both_ways(style: &str, start: At, end: At, steps: u32) -> (Vec<u8>, Vec<u8>) {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);

    let mut moved = Map::new(&runtime, options());
    moved.set_constraints(steep());
    moved.set_style_json(style).unwrap();
    moved.jump_to(&camera(start));
    common::settle(&mut moved, &mut target);
    // The style's pipeline set came with it: what the path shows on the
    // way, and at its end, asks for none.
    let requested = runtime.stats().counters.pipelines_requested;
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
    common::settle(&mut moved, &mut target);
    assert_eq!(
        runtime.stats().counters.pipelines_requested,
        requested,
        "the path's frames asked for pipelines the style's set left out"
    );
    let gestured = target.read_pixels().unwrap();
    let rest = moved.camera();
    drop(moved);
    let (fresh, fresh_rest) = fresh_at(&runtime, &mut target, style, end);
    // At rest, the camera too is what a fresh one is: over terrain, it
    // stands on the same ground at the same height.
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
    let mut fresh = Map::new(runtime, options());
    fresh.set_constraints(steep());
    fresh.set_style_json(style).unwrap();
    fresh.jump_to(&camera(at));
    common::settle(&mut fresh, target);
    (target.read_pixels().unwrap(), fresh.camera())
}

/// Asserts the two pictures equal. On failure, writes both and their
/// difference to a fresh directory and names it.
fn assert_same(name: &str, (gestured, fresh): (Vec<u8>, Vec<u8>)) {
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
            if a == b {
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
    let mut map = Map::new(&runtime, options());
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
    common::settle(&mut map, &mut target);
    for stop in 1..=6 {
        for f in (stop - 1) * 20 + 1..=stop * 20 {
            map.jump_to(&camera(at(f)));
            map.render(&mut target, 0.0).unwrap();
            std::thread::sleep(Duration::from_millis(8));
        }
        common::settle(&mut map, &mut target);
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

/// A label of a tile that left the draw list, drawn as last placed while
/// it fades out, stays where it is in the world as the globe comes in
/// below zoom 12. Zooming out of zoom 12 with zoom 11's tiles resident, a
/// label only zoom 12's tile has (its filter wants zoom 12) is held from a
/// tile last drawn on the plane: it draws where a fresh map at the end
/// draws that label from zoom 11's tile, and nothing else does (rows placed
/// on the plane once mixed toward the camera's target as the globe came
/// in, and the label flew to the middle of the view). A label both tiles
/// have is drawn once, by zoom 11's tile, at full.
#[test]
fn held_labels_stay_put_as_the_globe_comes_in() {
    let data = common::data();
    let (lon, lat) = (STATION.lon, STATION.lat);
    let style = |filter: &str| {
        format!(
            r##"{{"version": 8, "projection": {{"type": "globe"}},
                "glyphs": "file://{data}/glyphs/{{fontstack}}/{{range}}.pbf",
                "sources": {{"points": {{"type": "geojson", "data":
                    {{"type": "FeatureCollection", "features": [
                    {{"type": "Feature", "properties": {{"name": "Held"}},
                      "geometry": {{"type": "Point", "coordinates": [{}, {}]}}}},
                    {{"type": "Feature", "properties": {{"name": "Shared"}},
                      "geometry": {{"type": "Point", "coordinates": [{}, {}]}}}}]}}}}}},
                "layers": [
                    {{"id": "background", "type": "background",
                      "paint": {{"background-color": "#ffffff"}}}},
                    {{"id": "labels", "type": "symbol", "source": "points",
                      "filter": {filter},
                      "layout": {{"text-field": ["get", "name"], "text-size": 24,
                                 "text-font": ["Noto Sans Regular"]}}}}]}}"##,
            lon + 0.03,
            lat + 0.012,
            lon - 0.03,
            lat + 0.006,
            data = data.display(),
        )
    };
    let at = |zoom| (STATION, zoom, 0.0, 62.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    // One clock for the frames drawn here and the ones settling draws.
    let start = std::time::Instant::now();
    let now = || start.elapsed().as_secs_f64() * 1000.0;
    let map = |target: &mut Target, style: &str| {
        let mut map = Map::new(
            &runtime,
            MapOptions {
                fade_duration: 4000.0,
                ..options()
            },
        );
        map.set_style_json(style).unwrap();
        map.jump_to(&camera(at(11.4)));
        map.render(target, now()).unwrap();
        common::settle(&mut map, target);
        map
    };
    let only_12 = r#"["any", ["!=", ["get", "name"], "Held"], [">=", ["zoom"], 12]]"#;
    let mut moved = map(&mut target, &style(only_12));
    moved.jump_to(&camera(at(12.05)));
    common::settle(&mut moved, &mut target);
    for step in 1..=12 {
        let zoom = 12.05 - 0.65 * f64::from(step) / 12.0;
        moved.jump_to(&camera(at(zoom)));
        moved.render(&mut target, now()).unwrap();
        std::thread::sleep(Duration::from_millis(8));
    }
    // Drawn once zoom 11's tile is placed, the held label fading out.
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut loaded = 0;
    while loaded < 2 {
        assert!(std::time::Instant::now() < deadline, "the map loads");
        let report = moved.render(&mut target, now()).unwrap();
        loaded = if report.is_some_and(|r| r.loaded) {
            loaded + 1
        } else {
            0
        };
        std::thread::sleep(Duration::from_millis(4));
    }
    let held = target.read_pixels().unwrap();
    drop(moved);
    drop(map(&mut target, &style("true")));
    let fresh = target.read_pixels().unwrap();
    // Each half's ink: its weight and its centre, weighted by darkness, so
    // that the opacity doesn't move it (zooming out speeds fades, as GL
    // JS's zoom adjustment does: the held label is partway out). "Shared"
    // is on the left, "Held" on the right.
    let ink = |rgba: &[u8], right: bool| {
        let (mut weight, mut x, mut y) = (0.0, 0.0, 0.0);
        for (i, p) in rgba.chunks(4).enumerate() {
            let (px, py) = ((i % WIDTH as usize) as f64, (i / WIDTH as usize) as f64);
            if (px >= f64::from(WIDTH) / 2.0) == right {
                let w = f64::from(255 - p[0]);
                (weight, x, y) = (weight + w, x + w * px, y + w * py);
            }
        }
        (weight, x / weight, y / weight)
    };
    let mut problems = Vec::new();
    for (right, name) in [(false, "Shared"), (true, "Held")] {
        let (weight, x, y) = ink(&held, right);
        let (want, wx, wy) = ink(&fresh, right);
        if want < 20_000.0 || weight < 0.25 * want || (x - wx).hypot(y - wy) > 1.0 {
            problems.push(format!(
                "{name} inks {weight:.0} around ({x:.1}, {y:.1}), a fresh map {want:.0} \
                 around ({wx:.1}, {wy:.1})"
            ));
        }
        // Drawn once: no copy fading out over the one shown.
        if !right && (weight / want - 1.0).abs() > 0.02 {
            problems.push(format!("{name} inks {weight:.0}, a fresh map {want:.0}"));
        }
    }
    if !problems.is_empty() {
        let dir = std::env::temp_dir().join(format!("orb-path-held-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_png(&dir.join("held.png"), WIDTH, HEIGHT, &held);
        write_png(&dir.join("fresh.png"), WIDTH, HEIGHT, &fresh);
        panic!("{}; see {}", problems.join("; "), dir.display());
    }
}

/// Satellites lifted by a data-driven `symbol-height-offset`, and towers as
/// tall, over a globe looking past its horizon: the far ones' ground is out
/// of view. A gesture that passed over one, and saw how high it stands,
/// covers no more than a fresh map, which knows only what the covered tiles
/// hold (GL JS keeps the highest it has seen).
#[test]
fn lifted_content_covers_what_a_fresh_globe_does() {
    // At each: a point lifted to its altitude, and a tower that tall.
    let at = |lon: f64, lat: f64, altitude: u32| {
        let (w, e, s, n) = (lon - 0.3, lon + 0.3, lat - 0.3, lat + 0.3);
        [
            format!(
                r#"{{"type": "Feature", "properties": {{"altitude": {altitude}}},
                    "geometry": {{"type": "Point", "coordinates": [{lon}, {lat}]}}}}"#
            ),
            format!(
                r#"{{"type": "Feature", "properties": {{"altitude": {altitude}}},
                    "geometry": {{"type": "Polygon", "coordinates":
                    [[[{w}, {s}], [{e}, {s}], [{e}, {n}], [{w}, {n}], [{w}, {s}]]]}}}}"#
            ),
        ]
    };
    // Shows, its tiles kept by the height of `near` on the covered ground.
    let far = at(2.3522, 45.9188, 500_000);
    let near = at(2.3522, 60.0, 500_000);
    // Higher than anything the covered tiles hold, so its tiles go
    // uncovered and it doesn't show, though the gesture starts over it.
    let lone = at(-3.0, 18.0, 3_000_000);
    let features = |i: usize| [&far[i], &near[i], &lone[i]].map(String::as_str).join(", ");
    let style = format!(
        r##"{{"version": 8, "projection": {{"type": "globe"}},
            "glyphs": "file://{data}/glyphs/{{fontstack}}/{{range}}.pbf",
            "sources": {{
                "satellites": {{"type": "geojson", "data":
                    {{"type": "FeatureCollection", "features": [{points}]}}}},
                "towers": {{"type": "geojson", "data":
                    {{"type": "FeatureCollection", "features": [{towers}]}}}}}},
            "layers": [
                {{"id": "background", "type": "background",
                  "paint": {{"background-color": "#c9d9e8"}}}},
                {{"id": "towers", "type": "fill-extrusion", "source": "towers",
                  "paint": {{"fill-extrusion-height": ["get", "altitude"],
                             "fill-extrusion-color": "#4a6"}}}},
                {{"id": "satellites", "type": "symbol", "source": "satellites",
                  "layout": {{"symbol-height-offset": ["get", "altitude"],
                              "symbol-height-anchor": "absolute",
                              "text-field": ["to-string", ["get", "altitude"]],
                              "text-font": ["Noto Sans Regular"], "text-size": 16,
                              "text-allow-overlap": true, "text-ignore-placement": true}},
                  "paint": {{"text-color": "#c00", "text-halo-color": "#fff",
                             "text-halo-width": 2}}}}]}}"##,
        data = common::data().display(),
        points = features(0),
        towers = features(1),
    );
    let start = (Geodetic::new(-3.0, 18.0), 4.0, 0.0, 0.0);
    let end = (Geodetic::new(2.3522, 52.0566), 4.3, 180.0, 95.0);
    assert_same("lifted", both_ways(&style, start, end, 20));
}

/// Terrarium DEM tiles of `metres(lon, lat)` over `(west, south, east,
/// north)`, zooms 0–`max_zoom`, written to a fresh directory; the source
/// `name`'s JSON, and the directory to keep while it's read.
fn dem(
    name: &str,
    (west, south, east, north): (f64, f64, f64, f64),
    max_zoom: u8,
    metres: impl Fn(f64, f64) -> f64,
) -> (String, common::Scratch) {
    let dir = common::Scratch::new(&format!("path-dem-{name}"));
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
    (source, dir)
}

/// Rolling hills over central Tokyo, zooms 0–11.
fn hills() -> (String, common::Scratch) {
    dem("hills", (139.2, 35.2, 140.3, 36.2), 11, |lon, lat| {
        600.0
            + 500.0 * (lon * 40.0).sin() * (lat * 55.0).cos()
            + 150.0 * (lon * 170.0 + lat * 90.0).sin()
    })
}

/// Steep ridges over the Valais Alps, 700 to about 4,500 m, zooms 0–11,
/// hillshaded on the terrain under a sky: ridge silhouettes stand against
/// it at a high pitch. Crests are rounded: a crease one DEM pixel wide
/// running across the pixel grid shades as dotted lines.
fn alps() -> (String, common::Scratch) {
    let (dem, dir) = alps_dem();
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

/// The ridges' DEM source `alps`, and its directory.
fn alps_dem() -> (String, common::Scratch) {
    dem("alps", (7.4, 45.7, 8.4, 46.3), 11, |lon, lat| {
        let ridge = |v: f64| 1.0 - (v.sin().powi(2) + 0.04).sqrt();
        700.0
            + 2600.0 * ridge(lon * 31.0 + lat * 9.0) * ridge(lat * 23.0 - lon * 5.0)
            + 1200.0 * ridge(lon * 97.0 - lat * 61.0)
    })
}

/// A line draped on flat terrain is as wide as on a flat map at any
/// zoom, in screen pixels as the spec has it, with a zoom-dependent width
/// too. Drape slots drawn at their tile's whole zoom magnified it by up to
/// twice between zoom steps (#151).
#[test]
fn a_draped_line_is_as_wide_as_a_flat_one() {
    let (lon, lat) = (139.76, 35.68);
    let (flat, _dem) = dem("flat", (139.2, 35.2, 140.3, 36.2), 11, |_, _| 0.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    // The dark rows down the middle column: the line's width there.
    let width = |pixels: &[u8]| {
        (0..HEIGHT)
            .filter(|&y| pixels[((y * WIDTH + WIDTH / 2) * 4) as usize] < 128)
            .count()
    };
    let mut widths = |zoom: f64| {
        [false, true].map(|terrain| {
            let style = format!(
                r##"{{"version": 8,
                "sources": {{{flat}
                  "line": {{"type": "geojson", "data": {{"type": "LineString",
                    "coordinates": [[{w}, {lat}], [{e}, {lat}]]}}}}
                }},
                {}
                "layers": [
                  {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
                  {{"id": "line", "type": "line", "source": "line", "paint": {{"line-color": "#000000",
                      "line-width": ["interpolate", ["linear"], ["zoom"], 12, 6, 14, 14]}}}}
                ]}}"##,
                match terrain {
                    true => r#""terrain": {"source": "flat", "exaggeration": 1},"#,
                    false => "",
                },
                w = lon - 0.2,
                e = lon + 0.2,
            );
            let mut map = Map::new(&runtime, options());
            map.set_style_json(&style).unwrap();
            map.jump_to(&camera((Geodetic::new(lon, lat), zoom, 0.0, 0.0)));
            common::settle(&mut map, &mut target);
            width(&target.read_pixels().unwrap())
        })
    };
    for zoom in [12.0, 12.3, 12.6, 12.9] {
        let [flat, draped] = widths(zoom);
        // Each is the zoom's width, give or take an edge's rounding.
        let expected = 6.0 + 4.0 * (zoom - 12.0);
        assert!(
            (flat as f64 - expected).abs() <= 1.0,
            "z{zoom}: flat {flat}"
        );
        assert!(
            flat.abs_diff(draped) <= 1,
            "z{zoom}: draped {draped} px against flat {flat} px"
        );
    }
}

/// A draped line's slots draw at the zoom's drape step, a 32nd of a level:
/// a camera zooming within one redraws none of them, and one into the
/// next redraws them, whichever way it zooms.
#[test]
fn a_zoom_within_a_drape_step_redraws_no_slots() {
    let (lon, lat) = (139.76, 35.68);
    let (flat, _dem) = dem("flat", (139.2, 35.2, 140.3, 36.2), 11, |_, _| 0.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let style = format!(
        r##"{{"version": 8,
        "sources": {{{flat}
          "line": {{"type": "geojson", "data": {{"type": "LineString",
            "coordinates": [[{w}, {lat}], [{e}, {lat}]]}}}}
        }},
        "terrain": {{"source": "flat", "exaggeration": 1}},
        "layers": [
          {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "line", "type": "line", "source": "line",
              "paint": {{"line-color": "#000000", "line-width": 6}}}}
        ]}}"##,
        w = lon - 0.2,
        e = lon + 0.2,
    );
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&style).unwrap();
    let at = |zoom: f64| camera((Geodetic::new(lon, lat), zoom, 0.0, 0.0));
    map.jump_to(&at(12.1));
    common::settle(&mut map, &mut target);
    let mut slots = |zoom: f64| {
        map.jump_to(&at(zoom));
        let report = map.render(&mut target, 0.0).unwrap().unwrap();
        report.frame.drape_slots
    };
    // 12.1 is in the step from 12.09375.
    for zoom in [12.095, 12.11, 12.12, 12.1] {
        assert_eq!(slots(zoom), 0, "z{zoom}");
    }
    assert!(slots(12.13) > 0, "z12.13");
    assert!(slots(12.09) > 0, "z12.09");
}

/// A zoom that crosses a drape step every frame, as a fast wheel zoom
/// does, redraws no more slots a frame than its budget holds (eight whole
/// colour slots on a desktop), the slots it leaves stale drawn as they
/// were meanwhile, the longest stale first; at rest the map draws what a
/// fresh one does.
#[test]
fn a_zoom_across_a_drape_step_each_frame_redraws_within_its_budget() {
    let (lon, lat) = (139.76, 35.68);
    let (flat, _dem) = dem("flat", (139.2, 35.2, 140.3, 36.2), 11, |_, _| 0.0);
    let lines: Vec<String> = (0..9)
        .map(|i| {
            let y = lat - 0.08 + 0.02 * f64::from(i);
            format!("[[{}, {y}], [{}, {y}]]", lon - 0.3, lon + 0.3)
        })
        .collect();
    let style = format!(
        r##"{{"version": 8,
        "sources": {{{flat}
          "lines": {{"type": "geojson", "data": {{"type": "MultiLineString",
            "coordinates": [{lines}]}}}}
        }},
        "terrain": {{"source": "flat", "exaggeration": 1}},
        "layers": [
          {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "lines", "type": "line", "source": "lines",
              "paint": {{"line-color": "#000000", "line-width": 6}}}}
        ]}}"##,
        lines = lines.join(", "),
    );
    // Straight down, a cover of a dozen tiles that only leave as it zooms
    // in: none is blank.
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 3 * WIDTH, 3 * HEIGHT, 1.0);
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&style).unwrap();
    let at = |zoom: f64| camera((Geodetic::new(lon, lat), zoom, 0.0, 0.0));
    let end = 12.1 + 20.0 / 32.0;
    map.jump_to(&at(12.1));
    common::settle(&mut map, &mut target);
    let (mut most, mut redrawn) = (0, 0);
    for step in 1..=20 {
        map.jump_to(&at(12.1 + f64::from(step) / 32.0));
        let report = map.render(&mut target, 0.0).unwrap().unwrap();
        most = most.max(report.frame.drape_slots);
        redrawn += report.frame.drape_slots;
    }
    common::settle(&mut map, &mut target);
    let gestured = target.read_pixels().unwrap();
    assert!(most <= 8, "{most} slots redrawn in a frame");
    assert!(redrawn >= 20, "{redrawn} slots redrawn over the zoom");
    let (fresh, _) = fresh_at(
        &runtime,
        &mut target,
        &style,
        (Geodetic::new(lon, lat), end, 0.0, 0.0),
    );
    let dark = |pixels: &[u8]| pixels.chunks(4).filter(|p| p[0] < 128).count();
    assert!(dark(&fresh) > 0, "the lines drew nothing");
    let differ = (gestured.chunks(4).zip(fresh.chunks(4))).filter(|(a, b)| a != b);
    assert_eq!(differ.count(), 0, "the zoomed map differs from a fresh one");
}

/// A per-feature line width the zoom blends, which the drape pass works
/// out on the GPU, draws into a slot at its drape step too: a map zoomed
/// within the step, which redraws no slot, draws what a fresh map there
/// does. At the camera's own zoom, the slot kept the width of the frame
/// that drew it.
#[test]
fn a_zoom_blended_feature_width_drapes_at_its_step() {
    let (lon, lat) = (139.76, 35.68);
    let (flat, _dem) = dem("flat", (139.2, 35.2, 140.3, 36.2), 11, |_, _| 0.0);
    let style = format!(
        r##"{{"version": 8,
        "sources": {{{flat}
          "line": {{"type": "geojson", "data": {{"type": "Feature", "properties": {{"w": 2}},
            "geometry": {{"type": "LineString", "coordinates": [[{w}, {lat}], [{e}, {lat}]]}}}}}}
        }},
        "terrain": {{"source": "flat", "exaggeration": 1}},
        "layers": [
          {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "line", "type": "line", "source": "line", "paint": {{"line-color": "#000000",
              "line-width": ["interpolate", ["linear"], ["zoom"],
                12, ["get", "w"], 13, ["*", 30, ["get", "w"]]]}}}}
        ]}}"##,
        w = lon - 0.2,
        e = lon + 0.2,
    );
    // 12.1 and 12.12 are in the step from 12.09375.
    let start = (Geodetic::new(lon, lat), 12.1, 0.0, 0.0);
    let end = (Geodetic::new(lon, lat), 12.12, 0.0, 0.0);
    // A line on white has too few colours for `assert_same`.
    let (gestured, fresh) = both_ways(&style, start, end, 2);
    let dark = |pixels: &[u8]| pixels.chunks(4).filter(|p| p[0] < 128).count();
    assert!(dark(&fresh) > 0, "the line drew nothing");
    let differ = (gestured.chunks(4).zip(fresh.chunks(4))).filter(|(a, b)| a != b);
    assert_eq!(
        differ.count(),
        0,
        "{} dark pixels gestured against {} fresh",
        dark(&gestured),
        dark(&fresh)
    );
}

/// A fill draped on terrain and translated far, in a pitched view where
/// near terrain tiles are of zooms past the camera's, draws whole: a slot
/// takes the rows of drawn tiles as far around it as the translation
/// reaches at the camera's zoom, which the slot is drawn at. Reckoned at
/// each terrain tile's own zoom, the reach fell short in near slots, and
/// bands past a drawn tile's edge went blank. (A flat map clips what a
/// tile translates to the tile, as GL JS's stencil does.)
#[test]
fn a_draped_translate_reaches_as_far_as_the_camera_s_pixels() {
    let (lon, lat) = (139.76, 35.68);
    let (flat, _dem) = dem("flat", (139.2, 35.2, 140.3, 36.2), 11, |_, _| 0.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let style = format!(
        r##"{{"version": 8,
        "sources": {{{flat}
          "area": {{"type": "geojson", "data": {{"type": "Polygon",
            "coordinates": [[[{w}, {s}], [{e}, {s}], [{e}, {n}], [{w}, {n}], [{w}, {s}]]]}}}}
        }},
        "terrain": {{"source": "flat", "exaggeration": 1}},
        "layers": [
          {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "area", "type": "fill", "source": "area",
            "paint": {{"fill-color": "#ff0000", "fill-translate": [160, 0]}}}}
        ]}}"##,
        w = lon - 0.5,
        e = lon - 0.01,
        s = lat - 0.3,
        n = lat + 0.3,
    );
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&style).unwrap();
    map.jump_to(&camera((Geodetic::new(lon, lat), 12.3, 0.0, 60.0)));
    common::settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    let red = |x: u32, y: u32| {
        let p = &pixels[((y * WIDTH + x) * 4) as usize..][..4];
        p[0] > 200 && p[1] < 80 && p[2] < 80
    };
    // Each row: red from the left edge to the fill's translated edge, and
    // none past the first pixel that isn't (give or take its edge's).
    for y in 0..HEIGHT {
        let end = (0..WIDTH).find(|&x| !red(x, y)).unwrap_or(WIDTH);
        assert!(end > WIDTH / 4, "row {y}: red to {end} only");
        let past = (end + 2..WIDTH).filter(|&x| red(x, y)).count();
        assert_eq!(past, 0, "row {y}: a gap at {end}, red again past it");
    }
}

/// Terrain that comes on after the style's first frame, its DEM source
/// behind a TileJSON: a fresh map that never moves drapes the buildings
/// its near drape slots' zoom shows (they show from zoom 13; the view is
/// north of the extract, which lies in its near slots), as a map that
/// moved there after the terrain came on does. Paint written before
/// the terrain was on must not stay.
#[test]
fn a_terrain_that_comes_on_late_drapes_what_a_moved_map_does() {
    let (hills, dem) = hills();
    let tilejson = dem.join("tiles.json");
    let tiles = format!("file://{}/{{z}}/{{x}}/{{y}}.png", dem.display());
    std::fs::write(
        &tilejson,
        format!(r#"{{"tilejson": "3.0.0", "tiles": ["{tiles}"], "maxzoom": 11}}"#),
    )
    .unwrap();
    let late = hills.replace(
        &format!(r#""tiles": ["{tiles}"]"#),
        &format!(r#""url": "file://{}""#, tilejson.display()),
    );
    assert_ne!(late, hills);
    let style = tokyo(
        r#""terrain": {"source": "hills", "exaggeration": 1.5},"#,
        &late,
    );
    let north = Geodetic::new(STATION.lon, 35.70);
    let start = (north, 12.5, 0.0, 60.0);
    let end = (north, 12.9, 0.0, 60.0);
    assert_same("late-terrain", both_ways(&style, start, end, 5));
}

/// Terrain over local hills: DEM borders come from the neighbours the view
/// draws, not from those a gesture saw.
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

/// Tokyo's buildings on hills, lit by a map-anchored light: they stand
/// on the terrain and draw as a fresh map does after a gesture over the
/// hills.
#[test]
fn extrusions_on_terrain_after_a_gesture_draw_what_a_fresh_map_does() {
    let (hills, _dem) = hills();
    let style = extruded(tokyo(
        r#""terrain": {"source": "hills", "exaggeration": 1.5},
           "light": {"anchor": "map", "position": [1.15, 210, 60]},"#,
        &hills,
    ));
    let start = (Geodetic::new(139.735, 35.68), 14.5, -20.0, 40.0);
    let end = (STATION, 16.0, 30.0, 60.0);
    assert_same("terrain-extrusions", both_ways(&style, start, end, 20));
}

/// A map over terrain resized between frames draws its next frame at
/// the new size: the terrain's depth for what it hides, its layers, the
/// extrusions' depth and targets are remade at the new size, and what the
/// frame binds is the new ones, not the destroyed (a submit that used one
/// drew nothing, a black map in browsers).
#[test]
fn a_resized_terrain_map_draws_its_next_frame() {
    let (hills, _dem) = hills();
    let style = extruded(tokyo(
        r#""terrain": {"source": "hills", "exaggeration": 1.5},"#,
        &hills,
    ));
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&style).unwrap();
    map.jump_to(&camera((STATION, 15.0, 20.0, 50.0)));
    common::settle(&mut map, &mut target);
    for (width, height) in [
        (WIDTH + 96, HEIGHT - 64),
        (WIDTH / 2, HEIGHT),
        (WIDTH, HEIGHT),
    ] {
        target.resize(width, height, 1.0);
        map.render(&mut target, 0.0).unwrap();
        let pixels = target.read_pixels().unwrap();
        let colours: std::collections::HashSet<&[u8]> = pixels.chunks(4).collect();
        assert!(
            colours.len() > 64,
            "the frame after a resize to {width}×{height} drew {} colours",
            colours.len()
        );
    }
}

/// Turning terrain on raises it from flat over the style's transition,
/// and turning it off eases it flat and then drops it, with the map's
/// tiles drawn on every frame between: a fill under Tokyo's layers covers
/// the ground, so no frame shows the background (once, removing terrain
/// left the drape no room while the terrain fell, and every frame of the
/// fall drew the labels alone); the frames between differ from both ends
/// (they ease rather than jump), and at rest the map draws what a fresh
/// map with or without the terrain does at that camera.
#[test]
fn turning_terrain_on_and_off_eases_with_every_tile_drawn() {
    let (hills, _dem) = hills();
    let ground = r#""ground": {"type": "geojson", "data": {"type": "Polygon",
        "coordinates": [[[139.2, 35.2], [140.3, 35.2], [140.3, 36.2], [139.2, 36.2], [139.2, 35.2]]]}},"#;
    let style = |terrain: &str| {
        let style = tokyo(terrain, &format!("{hills}{ground}"));
        let background = r##""background-color": "#f8f4f0"
      }
    },"##;
        assert!(style.contains(background));
        style.replacen(
            background,
            r##""background-color": "#ff00ff"}},
            {"id": "ground", "type": "fill", "source": "ground",
             "paint": {"fill-color": "#f8f4f0", "fill-antialias": false}},"##,
            1,
        )
    };
    let terrain = r#"{"source": "hills", "exaggeration": 1.5}"#;
    let with = style(&format!(r#""terrain": {terrain},"#));
    let without = style("");
    let at = (STATION, 14.5, 30.0, 55.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, options());
    map.set_constraints(steep());
    map.set_style_json(&without).unwrap();
    map.jump_to(&camera(at));
    // The frames' clock, in milliseconds: settling times its frames on
    // from the last one's.
    let started = std::time::Instant::now();
    let clock = || started.elapsed().as_secs_f64() * 1000.0;
    map.render(&mut target, clock()).unwrap();
    common::settle(&mut map, &mut target);
    // The background's magenta, however the terrain shades it: nothing in
    // Tokyo's style is near it.
    let bare = |pixels: &[u8]| {
        (pixels.chunks(4))
            .filter(|p| u16::from(p[1]) + 96 < u16::from(p[0].min(p[2])))
            .count()
    };
    let mut turn = |terrain: &str, fresh_style: &str, name: &str| {
        let mut frames = vec![target.read_pixels().unwrap()];
        let tx = format!(r#"[["setTerrain", {terrain}]]"#);
        map.edit(&orb_api::Transaction::from_commands(&tx).unwrap())
            .unwrap();
        // Past the style's transition, 300 ms, at 60 frames a second.
        let mut t = clock();
        for _ in 0..30 {
            map.render(&mut target, t).unwrap();
            frames.push(target.read_pixels().unwrap());
            t += 1000.0 / 60.0;
        }
        common::settle(&mut map, &mut target);
        frames.push(target.read_pixels().unwrap());
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(bare(frame), 0, "{name}: frame {i} shows the background");
        }
        let (first, last) = (&frames[0], frames.last().unwrap());
        assert_ne!(first, last, "{name} changed nothing");
        let middle = &frames[frames.len() / 3];
        assert!(
            middle != first && middle != last,
            "{name}: the map jumped rather than eased"
        );
        let camera = map.camera();
        let mut other = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
        let (fresh, fresh_camera) = fresh_at(&runtime, &mut other, fresh_style, at);
        assert_eq!(
            camera, fresh_camera,
            "{name}: the camera differs from a fresh map's"
        );
        assert_same(name, (last.clone(), fresh));
    };
    turn(terrain, &with, "terrain-on");
    turn("null", &without, "terrain-off");
}

/// Labels of tiles a view stopped drawing are held until a placement
/// covers what replaced them and they've faded, on frame time: a host's
/// display link has a clock of its own, not the runtime's. Let go by the
/// runtime's clock, a map whose display link ran ahead of it kept them
/// placed, faded out, for as long as it stayed idle: once idle, it culls
/// what a fresh map there does, and draws the same.
#[test]
fn labels_left_behind_go_by_the_display_links_clock() {
    let style = tokyo("", "");
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let idle = |map: &mut Map, target: &mut Target| {
        (map.render_until_idle(target, Duration::from_secs(60)))
            .unwrap()
            .scene
            .candidates
    };
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&style).unwrap();
    map.jump_to(&camera((STATION, 15.8, 30.0, 0.0)));
    let started = std::time::Instant::now();
    map.render(
        &mut target,
        3.6e6 + started.elapsed().as_secs_f64() * 1000.0,
    )
    .unwrap();
    idle(&mut map, &mut target);
    let at = (STATION, 14.5, 30.0, 0.0);
    map.jump_to(&camera(at));
    let culled = idle(&mut map, &mut target);
    let moved = target.read_pixels().unwrap();
    drop(map);
    let mut fresh = Map::new(&runtime, options());
    fresh.set_style_json(&style).unwrap();
    fresh.jump_to(&camera(at));
    assert_eq!(culled, idle(&mut fresh, &mut target), "candidates culled");
    assert_same("held-labels", (moved, target.read_pixels().unwrap()));
}

/// Tokyo's buildings (unless `buildings` is false) standing on low
/// rolling ground under a sky, over a plain ground and under the
/// comma-separated `above`, and nothing else: a frame of them pitched far
/// enough for fog, which begins near (fog reaches from sea level's depth
/// at the target, so the ground stays low).
fn foggy_extrusions_under(buildings: bool, above: &str) -> Vec<u8> {
    let (lowland, _dem) = dem("lowland", (139.2, 35.2, 140.3, 36.2), 11, |lon, lat| {
        20.0 + 15.0 * (lon * 170.0).sin() * (lat * 210.0).cos()
    });
    let style = tokyo(
        r##""terrain": {"source": "lowland"},
            "sky": {"sky-color": "#4f8fd9", "horizon-color": "#d6e8fb", "fog-color": "#c8d8f0",
                    "fog-ground-blend": 0.2},"##,
        &lowland,
    );
    let mut layers = vec![
        r##"{"id": "ground", "type": "background", "paint": {"background-color": "#e8e4dc"}}"##,
    ];
    if buildings {
        layers.push(
            r##"{"id": "building-3d", "type": "fill-extrusion", "source": "openmaptiles",
                 "source-layer": "building", "minzoom": 14,
                 "paint": {"fill-extrusion-color": "#d9d0c9",
                           "fill-extrusion-height": ["get", "render_height"]}}"##,
        );
    }
    layers.extend(Some(above).filter(|a| !a.is_empty()));
    let at = style.find("\"layers\": [").unwrap();
    let style = format!("{}\"layers\": [{}]}}", &style[..at], layers.join(","));
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    fresh_at(&runtime, &mut target, &style, (STATION, 15.5, 30.0, 80.0)).0
}

/// A draped layer over a fill-extrusion layer on terrain that draws
/// nothing changes nothing: its terrain layer fogs only what it draws, so
/// the fog doesn't land on the ground twice, nor on the buildings in
/// front (the demo's hover highlights, inserted before the first label,
/// hazed the Globe entry's buildings toward the horizon).
#[test]
fn a_draped_layer_that_draws_nothing_over_extrusions_changes_nothing() {
    let plain = foggy_extrusions_under(true, "");
    let highlight = r##"{"id": "highlight", "type": "fill", "source": "openmaptiles",
        "source-layer": "building",
        "paint": {"fill-color": "#e5007e",
                  "fill-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.3, 0]}}"##;
    assert_same(
        "empty-drape-over-3d",
        (foggy_extrusions_under(true, highlight), plain),
    );
}

/// Buildings hide a draped layer above their layer where they stand in
/// front of it, as GL JS draws such a layer's terrain over their depth:
/// an opaque background above them covers the ground and leaves the
/// buildings as they were.
#[test]
fn buildings_hide_the_draped_layers_above_them() {
    let cover =
        r##"{"id": "cover", "type": "background", "paint": {"background-color": "#ff00ff"}}"##;
    let bare = foggy_extrusions_under(false, "");
    let plain = foggy_extrusions_under(true, "");
    let covered = foggy_extrusions_under(true, cover);
    let px = |frame: &[u8], i: usize| frame[i * 4..i * 4 + 4].to_vec();
    let total = plain.len() / 4;
    let building: Vec<bool> = (0..total).map(|i| px(&plain, i) != px(&bare, i)).collect();
    // Their edges blend with what lies behind them, which the cover
    // changes: only pixels inside a building count.
    let w = WIDTH as usize;
    let buildings: Vec<usize> = (w..total - w)
        .filter(|&i| [i, i - 1, i + 1, i - w, i + w].iter().all(|&j| building[j]))
        .collect();
    let kept = (buildings.iter())
        .filter(|&&i| px(&covered, i) == px(&plain, i))
        .count();
    let ground = (0..total)
        .filter(|&i| px(&covered, i) == [255, 0, 255, 255])
        .count();
    assert!(
        buildings.len() > total / 20,
        "buildings drew {} of {total} pixels",
        buildings.len()
    );
    assert!(
        ground > total / 20,
        "the cover drew {ground} of {total} pixels"
    );
    assert!(
        kept * 20 > buildings.len() * 19,
        "buildings kept {kept} of their {} pixels",
        buildings.len()
    );
}

/// Houses on a hillside each stand on the ground under them: a row of
/// them climbs a slope 300 m high across a few overzoom slices, each house
/// a block of two (OpenMapTiles merges a block's buildings into one
/// multipolygon) in a hue of its own, and every one shows, looked at from
/// downhill and from uphill. Houses raised by one elevation per tile sink
/// into the hill above that point.
#[test]
fn houses_on_a_slope_stand_on_the_ground() {
    const HOUSES: usize = 12;
    let (lat, west) = (35.68, 139.745);
    // A plane rising east, 3,000 m a tenth of a degree: a third.
    let (slope, _dem) = dem("slope", (139.6, 35.6, 139.9, 35.76), 12, |lon, _| {
        (lon - 139.6) * 30_000.0
    });
    let square = |w: f64, s: f64| {
        let (e, n) = (w + 0.00013, s + 0.0003);
        format!("[[[{w}, {s}], [{e}, {s}], [{e}, {n}], [{w}, {n}], [{w}, {s}]]]")
    };
    let houses: Vec<String> = (0..HOUSES)
        .map(|i| {
            let w = west + 0.001 * i as f64;
            format!(
                r#"{{"type": "Feature", "properties": {{"hue": {}}}, "geometry":
                    {{"type": "MultiPolygon", "coordinates": [{}, {}]}}}}"#,
                30 * i,
                square(w, lat),
                square(w + 0.00017, lat),
            )
        })
        .collect();
    let style = format!(
        r##"{{"version": 8,
            "sources": {{{slope}
              "houses": {{"type": "geojson", "maxzoom": 12, "data":
                {{"type": "FeatureCollection", "features": [{}]}}}}}},
            "terrain": {{"source": "slope"}},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {{"id": "houses", "type": "fill-extrusion", "source": "houses",
                "paint": {{"fill-extrusion-height": 15,
                           "fill-extrusion-color": ["to-color", ["concat",
                             "hsl(", ["to-string", ["get", "hue"]], ", 100%, 50%)"]]}}}}]}}"##,
        houses.join(", "),
    );
    // How many pixels show each house's hue: saturated pixels within 10°
    // of it, whatever the light makes of its faces.
    let shown = |pixels: &[u8]| {
        let mut count = [0u32; HOUSES];
        for p in pixels.chunks(4) {
            let [r, g, b] = [p[0], p[1], p[2]].map(|c| f64::from(c) / 255.0);
            let (hi, lo) = (r.max(g).max(b), r.min(g).min(b));
            if hi - lo < 0.3 {
                continue;
            }
            let d = hi - lo;
            let hue = 60.0
                * if hi == r {
                    ((g - b) / d).rem_euclid(6.0)
                } else if hi == g {
                    (b - r) / d + 2.0
                } else {
                    (r - g) / d + 4.0
                };
            let i = (hue / 30.0).round() as usize % HOUSES;
            if (hue - 30.0 * (hue / 30.0).round()).abs() < 10.0 {
                count[i] += 1;
            }
        }
        count
    };
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let middle = west + 0.0005 * (HOUSES - 1) as f64;
    for (name, at) in [
        (
            "downhill",
            (Geodetic::new(middle, lat - 0.003), 15.2, 0.0, 60.0),
        ),
        (
            "uphill",
            (Geodetic::new(middle, lat + 0.003), 15.2, 180.0, 60.0),
        ),
    ] {
        let (pixels, _) = fresh_at(&runtime, &mut target, &style, at);
        let count = shown(&pixels);
        if count.iter().any(|&n| n < 20) {
            let path = std::env::temp_dir()
                .join(format!("orb-path-slope-{name}-{}.png", std::process::id()));
            write_png(&path, WIDTH, HEIGHT, &pixels);
            panic!(
                "{name}: pixels per house {count:?}, some sunk or missing: {}",
                path.display()
            );
        }
    }
}

/// Ridges against the sky over the Alps: a turn and tilt to a pitch of
/// 75° draws the silhouettes a fresh map at rest draws.
#[test]
fn an_alpine_gesture_draws_what_a_fresh_map_does() {
    let start = (Geodetic::new(7.95, 46.0), 11.2, 0.0, 30.0);
    let end = (Geodetic::new(7.85, 45.95), 12.4, 60.0, 75.0);
    let (alps, _dem) = alps();
    assert_same("alps", both_ways(&alps, start, end, 30));
}

/// The same ridges on a globe, into the middle of its handover to Web
/// Mercator at a pitch of 75°: the cover bounds each tile by the DEM's
/// range under it and measures detail at the camera's elevation, both
/// learned along the way, and the fog fades with the handover, yet the
/// picture is a fresh globe's.
#[test]
fn an_alpine_globe_gesture_draws_what_a_fresh_globe_does() {
    let start = (Geodetic::new(7.95, 46.0), 10.4, 0.0, 30.0);
    let end = (Geodetic::new(7.85, 45.95), 11.6, 60.0, 75.0);
    let (alps, _dem) = alps();
    let globe = alps.replacen(
        "{\"version\": 8,",
        "{\"version\": 8, \"projection\": {\"type\": \"globe\"},",
        1,
    );
    assert_ne!(globe, alps);
    assert_same("alps-globe", both_ways(&globe, start, end, 30));
}

/// Terrain fog fades in through a globe's handover to Web Mercator rather
/// than snapping on where it ends (#154): across zoom 12, where the
/// handover finishes, the picture changes no more than over the same
/// zoom step on the plane past it.
#[test]
fn terrain_fog_fades_through_the_globe_handover() {
    let (alps, _dem) = alps();
    let globe = alps.replacen(
        "{\"version\": 8,",
        "{\"version\": 8, \"projection\": {\"type\": \"globe\"},",
        1,
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut at = |zoom: f64| {
        let camera = (Geodetic::new(7.85, 45.95), zoom, 60.0, 75.0);
        fresh_at(&runtime, &mut target, &globe, camera).0
    };
    let [before, after, past] = [11.97, 12.0, 12.03].map(&mut at);
    // The picture's mean colour, which the fog lightens: a step's zoom
    // shifts every pixel, but barely moves it.
    let mean = |pixels: &[u8]| {
        let mut sum = [0.0; 3];
        for p in pixels.chunks(4) {
            for (s, &c) in sum.iter_mut().zip(p) {
                *s += f64::from(c);
            }
        }
        sum.map(|s| s / (pixels.len() / 4) as f64)
    };
    let change = |a: &[u8], b: &[u8]| {
        let (a, b) = (mean(a), mean(b));
        (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max)
    };
    let (handover, plane) = (change(&before, &after), change(&after, &past));
    assert!(
        handover < 2.0 * plane + 0.2,
        "across the handover's end the mean colour changes by {handover:.2} a channel, \
         over the same step on the plane by {plane:.2}"
    );
}

/// A view at rest over the Alps stays put: frames after idle cover the
/// same terrain tiles and read the same DEM tiles, so they pad no DEM
/// borders, write no rows, keep the cull's candidates and dispatch the
/// same cull every frame (at most one, and none once the drape is drawn
/// and nothing else has candidates to cull), from the first frame after
/// idle on. Here the DEM tile under one
/// cover tile culls that tile, which must not let go of the DEM tile and
/// bring the cover tile back on the next frame.
#[test]
fn an_alpine_view_at_rest_redoes_nothing() {
    let (alps, _dem) = alps();
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&alps).unwrap();
    map.jump_to(&camera((Geodetic::new(7.95, 46.0), 11.2, 0.0, 30.0)));
    common::settle(&mut map, &mut target);
    let mut dispatches = None;
    for frame in 1..=4 {
        let report = map.render(&mut target, 0.0).unwrap().unwrap();
        assert!(report.idle, "frame {frame}");
        assert_eq!(report.frame.prepared, 0, "frame {frame}");
        assert_eq!(report.frame.placements_written, 0, "frame {frame}");
        assert!(report.frame.dispatches <= culls(), "frame {frame}");
        let first = *dispatches.get_or_insert(report.frame.dispatches);
        assert_eq!(report.frame.dispatches, first, "frame {frame}");
        assert!(!report.frame.candidates_rebuilt, "frame {frame}");
    }
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
        let mut map = Map::new(&runtime, options());
        map.set_style_json(&alps).unwrap();
        map.jump_to(&camera(at));
        common::settle(&mut map, &mut target);
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

/// A moving view's drape pages only grow. Its cover takes more tiles and
/// fewer as it turns and tilts, and pages made anew allocate the whole
/// atlas in one frame, so pages larger than the cover are made to fit it
/// only once the view holds still: then exactly, as a fresh map's pages
/// fit it. Here the view over the Alps at pitch 80 looking north holds
/// larger pages than looking north-east, which a slow turn there, its
/// cover steady, keeps.
#[test]
fn a_moving_view_only_grows_its_drape_pages() {
    let (alps, _dem) = alps();
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 1440, 900, 1.0);
    let at = |bearing, pitch| (Geodetic::new(7.9, 46.0), 12.0, bearing, pitch);
    let settled = |map: &mut Map, target: &mut Target| {
        let report = map.render_until_idle(target, Duration::from_secs(60));
        report.expect("the map idles").memory.drape_atlas
    };
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&alps).unwrap();
    map.jump_to(&camera(at(45.0, 80.0)));
    let rest = settled(&mut map, &mut target);
    map.jump_to(&camera(at(0.0, 80.0)));
    let grown = settled(&mut map, &mut target);
    assert!(
        grown > rest,
        "looking north holds no larger pages: the test shows nothing"
    );
    let mut shrank = Vec::new();
    for step in 0..90 {
        map.jump_to(&camera(at(45.0 + f64::from(step) * 0.05, 80.0)));
        let report = map.render(&mut target, 0.0).unwrap();
        if report.is_some_and(|r| r.memory.drape_atlas < grown) {
            shrank.push(step);
        }
        std::thread::sleep(Duration::from_millis(8));
    }
    assert!(
        shrank.is_empty(),
        "the turning view's pages shrank in frames {shrank:?}"
    );
    let turned = settled(&mut map, &mut target);
    let mut fresh = Map::new(&runtime, options());
    fresh.set_style_json(&alps).unwrap();
    fresh.jump_to(&camera(at(45.0 + 89.0 * 0.05, 80.0)));
    let fresh = settled(&mut fresh, &mut target);
    assert_eq!(
        turned, fresh,
        "the turned map's pages differ from a fresh map's"
    );
}

/// Ground below the horizon at a high pitch over steep terrain, as over
/// Chongqing's hills on the demo's globe: a pitched view's cover wants
/// many more terrain tiles than a view straight down, and the drape pages
/// hold every one at rest, so the ground draws everywhere under the sky
/// (no pixel is left clear) and a gesture there draws what a fresh map
/// does. Pages sized for a fixed count of tiles coarsened the cover down
/// to the tiles under the eye, whose coarser ground rose over the camera
/// standing on the finer: black below the horizon.
#[test]
fn a_steep_view_over_terrain_shows_ground_below_the_horizon() {
    let (hills, _dem) = hills();
    let style = tokyo(
        r##""terrain": {"source": "hills", "exaggeration": 1.5},
            "projection": {"type": "globe"},
            "sky": {"sky-color": "#4f8fd9", "horizon-color": "#d6e8fb", "fog-color": "#e4eef8"},"##,
        &hills,
    );
    // Hillshaded, as the demo's globe is: a slope stack beside the colour.
    let style = style.replacen(
        "{\n      \"id\": \"park\",",
        r#"{"id": "relief", "type": "hillshade", "source": "hills"},
        {"id": "park","#,
        1,
    );
    assert!(style.contains("\"relief\""));
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    for (size, scale, at) in [
        ([3200, 2000], 2.0, (STATION, 16.12, -31.5, 85.0)),
        ([3200, 2000], 2.0, (STATION, 15.65, -52.9, 74.0)),
        ([1920, 1080], 1.0, (STATION, 15.02, -52.9, 80.0)),
    ] {
        let mut target = Target::texture(&runtime, size[0], size[1], scale);
        let (pixels, _) = fresh_at(&runtime, &mut target, &style, at);
        let dump = |what: &str, pixels: &[u8]| {
            let path = std::env::temp_dir().join(format!(
                "orb-path-steep-{}-{what}-{}.png",
                at.3,
                std::process::id()
            ));
            write_png(&path, size[0], size[1], pixels);
            path
        };
        let bare = pixels.chunks(4).filter(|p| p[3] < 255).count();
        assert!(
            bare == 0,
            "at {at:?}: {bare} pixels show no ground or sky: {}",
            dump("live", &pixels).display()
        );
        // A still map's pages always held its whole cover: at rest, a
        // live map's do too, and draw what it draws.
        let mut still = Map::new(&runtime, options());
        still.set_constraints(steep());
        still.set_style_json(&style).unwrap();
        still.jump_to(&camera(at));
        let logical = size.map(|v| (f64::from(v) / scale) as u32);
        let image = still.snapshot(logical[0], logical[1], scale).unwrap();
        // Within rounding: the CPU driver's blends differ by a unit here
        // and there between the two.
        let near = |(a, b): (&[u8], &[u8])| a.iter().zip(b).all(|(x, y)| x.abs_diff(*y) <= 2);
        let differ = (pixels.chunks(4).zip(image.pixels.chunks(4)))
            .filter(|&p| !near(p))
            .count();
        assert!(
            differ == 0,
            "at {at:?}: {differ} pixels differ from a still map's: {} against {}",
            dump("live", &pixels).display(),
            dump("still", &image.pixels).display()
        );
    }
    let start = (STATION, 14.0, 0.0, 40.0);
    let end = (STATION, 16.12, -31.5, 85.0);
    assert_same("steep", both_ways(&style, start, end, 20));
}

/// The Alps terrain style's view (live tiles), turned at a pitch of 72° with the
/// camera never stopping: once loaded, a frame mid-turn draws what a fresh
/// map at rest at its camera draws, ridges against the sky and every drape
/// slot's edge. What a slot's filter reads past its tile never depends on
/// which tile holds the next slot. Labels, which settle by their own
/// history, are hidden. Needs the network:
/// `cargo test --release -p orb-api --test path -- --ignored alps`.
#[test]
#[ignore = "fetches tiles over the network"]
fn turning_over_the_live_alps_draws_what_a_fresh_map_does() {
    let data = common::data();
    let style = std::fs::read_to_string(data.join("alps.json")).unwrap();
    let hide = orb_api::Transaction::from_commands(
        r#"[["setLayoutProperty", "peaks", "visibility", "none"],
            ["setLayoutProperty", "places", "visibility", "none"]]"#,
    )
    .unwrap();
    let at = |bearing| (Geodetic::new(7.7491, 46.0207), 11.6, bearing, 72.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut moved = Map::new(&runtime, options());
    moved.set_style_json(&style).unwrap();
    moved.edit(&hide).unwrap();
    moved.jump_to(&camera(at(-72.0)));
    common::settle(&mut moved, &mut target);
    // Load what the arrival shows, then turn back without waiting.
    moved.jump_to(&camera(at(-32.0)));
    common::settle(&mut moved, &mut target);
    moved.jump_to(&camera(at(-72.0)));
    common::settle(&mut moved, &mut target);
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
    let mut fresh = Map::new(&runtime, options());
    fresh.set_style_json(&style).unwrap();
    fresh.edit(&hide).unwrap();
    fresh.jump_to(&camera(at(-32.0)));
    common::settle(&mut fresh, &mut target);
    assert_same("live-alps", (turned, target.read_pixels().unwrap()));
}

/// A draped fill whose hovered colour varies with zoom shows the colour of
/// its new zoom in the first frame after a jump of six zooms: the state
/// rows' zooms follow the camera that frame, as a fresh map at rest
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
    let mut moved = Map::new(&runtime, options());
    moved.set_style_json(&style).unwrap();
    moved.edit(&hover).unwrap();
    moved.jump_to(&camera(at(10.0)));
    common::settle(&mut moved, &mut target);
    moved.jump_to(&camera(at(16.0)));
    // Its time runs on from the settled frames', past the hover's fade.
    moved.render(&mut target, 1e6).unwrap();
    let jumped = centre(&target.read_pixels().unwrap());
    drop(moved);
    let mut fresh = Map::new(&runtime, options());
    fresh.set_style_json(&style).unwrap();
    fresh.edit(&hover).unwrap();
    fresh.jump_to(&camera(at(16.0)));
    common::settle(&mut fresh, &mut target);
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
        let mut map = Map::new(&runtime, options());
        map.set_style_json(&style).unwrap();
        let at = |step: f64| (STATION, 14.2 + step * 0.01, step, 20.0);
        map.jump_to(&camera(at(0.0)));
        common::settle(&mut map, &mut target);
        for step in 1..=3 {
            map.jump_to(&camera(at(f64::from(step))));
            let report = map.render(&mut target, 0.0).unwrap().unwrap();
            assert_eq!(report.tiles_loading, 0, "the move loads nothing");
            assert_eq!(report.frame.placements_written, 0, "step {step}");
            assert!(!report.frame.candidates_rebuilt, "step {step}");
            // The recorded bundles replayed (the CPU driver's record again
            // when what a node draws changes).
            assert!(!report.frame.bundle_recorded || culls() == 0, "step {step}");
        }
    }
}

/// At zoom 22 a square's corner draws where the CPU projects it, to a
/// pixel: rows' origins and the camera subtract as hi and lo parts on the
/// GPU.
/// A GeoJSON source past its maxzoom: dashed lines, a dashed ring, a
/// gradient and line labels that run across its slices' edges, laid out
/// again at each zoom along the data tile's whole lines. Zooming into it
/// draws what a fresh map there does.
#[test]
fn overscaled_geojson_after_a_gesture_is_a_fresh_map_s() {
    let (lon, lat) = (STATION.lon, STATION.lat);
    // A spiral of about 40 turns' worth of vertices, and a ring.
    let spiral: Vec<String> = (0..160)
        .map(|i| {
            let a = f64::from(i) * 0.4;
            let r = 0.0004 + 0.00004 * f64::from(i);
            format!("[{}, {}]", lon + r * a.cos(), lat + r * a.sin())
        })
        .collect();
    let ring: Vec<String> = (0..=24)
        .map(|i| {
            let a = f64::from(i % 24) * std::f64::consts::TAU / 24.0;
            format!("[{}, {}]", lon + 0.004 * a.cos(), lat + 0.003 * a.sin())
        })
        .collect();
    let style = format!(
        r##"{{"version": 8,
            "glyphs": "file://{data}/glyphs/{{fontstack}}/{{range}}.pbf",
            "sources": {{
                "lines": {{"type": "geojson", "maxzoom": 11, "data":
                    {{"type": "FeatureCollection", "features": [
                    {{"type": "Feature", "properties": {{"name": "Spiral Street"}},
                      "geometry": {{"type": "LineString", "coordinates": [{spiral}]}}}},
                    {{"type": "Feature", "properties": {{}},
                      "geometry": {{"type": "Polygon", "coordinates": [[{ring}]]}}}}]}}}},
                "metered": {{"type": "geojson", "maxzoom": 11, "lineMetrics": true, "data":
                    {{"type": "LineString", "coordinates": [{spiral}]}}}}}},
            "layers": [
                {{"id": "background", "type": "background",
                  "paint": {{"background-color": "#f4f1ea"}}}},
                {{"id": "gradient", "type": "line", "source": "metered",
                  "paint": {{"line-width": 10, "line-offset": 8,
                             "line-gradient": ["interpolate", ["linear"], ["line-progress"],
                                               0, "#1060c0", 0.5, "#10c060", 1, "#c01060"]}}}},
                {{"id": "dashes", "type": "line", "source": "lines",
                  "paint": {{"line-width": 4, "line-color": "#202020",
                             "line-dasharray": [3, 2]}}}},
                {{"id": "names", "type": "symbol", "source": "lines",
                  "layout": {{"symbol-placement": "line", "symbol-spacing": 120,
                              "text-field": ["get", "name"],
                              "text-font": ["Noto Sans Regular"], "text-size": 14}},
                  "paint": {{"text-color": "#800000", "text-halo-color": "#fff",
                             "text-halo-width": 1.5}}}}]}}"##,
        data = common::data().display(),
        spiral = spiral.join(", "),
        ring = ring.join(", "),
    );
    let start = (Geodetic::new(lon - 0.004, lat - 0.002), 13.4, 0.0, 0.0);
    let end = (Geodetic::new(lon + 0.0011, lat + 0.0007), 16.6, 30.0, 40.0);
    assert_same("overscaled-geojson", both_ways(&style, start, end, 30));
}

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
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&style).unwrap();
    // A few metres apart: tens of pixels at zoom 22.
    for (dlon, dlat) in [(0.0, 0.0), (2.37e-6, -1.13e-6), (-3.11e-6, 0.9e-6)] {
        let center = Geodetic::new(lon + dlon, lat + dlat);
        map.jump_to(&camera((center, 22.0, 0.0, 0.0)));
        common::settle(&mut map, &mut target);
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
    let mut map = Map::new(&runtime, options());
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
    // The cache keeps what it holds of the city (two views' worth): the
    // pools hold it too.
    assert!(ocean <= dense);
    // Nothing left to draw or keep: the pools, the lists and tables, and
    // the tile pages go back to a fresh map's.
    let empty = r#"{"version": 8, "sources": {}, "layers": []}"#;
    map.set_style_json(empty).expect("style");
    let emptied = memory(&mut map, (Geodetic::new(150.0, 30.0), 15.5, 30.0, 60.0));
    let pools = |map: &mut Map, target: &mut Target| {
        let report = map
            .render_until_idle(target, Duration::from_secs(60))
            .unwrap();
        let (m, s) = (&report.memory, report.memory.scene);
        [
            s.vertices.allocated,
            s.indices.allocated,
            s.compacted.allocated,
            s.props.allocated,
            s.tables,
            m.tile_textures,
        ]
    };
    let mut fresh = Map::new(&runtime, options());
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

/// MapLibre Native's label stabilisers, on by default, make what labels
/// show depend on how the camera got there; they must still settle. After
/// a turning, tilting zoom the labels at rest stay put however often the
/// same view is placed again.
#[test]
fn stable_labels_converge_and_stay_put() {
    assert!(MapOptions::default().stable_labels);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&tokyo("", "")).unwrap();
    let start = (Geodetic::new(139.72, 35.67), 12.6, 0.0, 0.0);
    let end = (STATION, 15.4, 35.0, 50.0);
    map.jump_to(&camera(start));
    common::settle(&mut map, &mut target);
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    for step in 1..=30 {
        let t = f64::from(step) / 30.0;
        map.jump_to(&camera((
            Geodetic::new(
                lerp(start.0.lon, end.0.lon, t),
                lerp(start.0.lat, end.0.lat, t),
            ),
            lerp(start.1, end.1, t),
            lerp(start.2, end.2, t),
            lerp(start.3, end.3, t),
        )));
        map.render(&mut target, 0.0).unwrap();
        std::thread::sleep(Duration::from_millis(8));
    }
    common::settle(&mut map, &mut target);
    let shown = |map: &Map| {
        let mut names: Vec<Box<str>> = map
            .shown_labels()
            .into_iter()
            .filter(|&(_, opacity)| opacity > 0.5)
            .map(|(name, _)| name)
            .collect();
        names.sort();
        names
    };
    let rest = shown(&map);
    assert!(rest.len() > 10, "labels show at rest: {rest:?}");
    // A bearing a millionth of a degree off places the view again.
    for i in 0..4 {
        let nudge = if i % 2 == 0 { 1e-6 } else { 0.0 };
        map.jump_to(&camera((end.0, end.1, end.2 + nudge, end.3)));
        common::settle(&mut map, &mut target);
        assert_eq!(
            shown(&map),
            rest,
            "placing the view again ({i}) moved labels"
        );
    }
}

/// Ground that showed its hillshade keeps it while the terrain under it
/// turns coarser: zoomed out half a level, the terrain's tiles there are a
/// zoom coarser and their hillshade tiles haven't arrived, so the finer
/// ones the hillshade drew before stand in, as they would on a flat map.
/// The first frame after the zoom, before anything can arrive, shades the
/// middle of the view (ground the first view showed) as the settled view
/// does: no more of it is bare background.
#[test]
fn coarser_terrain_keeps_the_hillshade_it_drew() {
    const GROUND: [u8; 3] = [0xd8, 0xe0, 0xc8];
    let at = Geodetic::new(7.85, 45.95);
    let (dem, _dir) = dem("relief", (7.82, 45.93, 7.88, 45.97), 15, |lon, lat| {
        1500.0 + 400.0 * (lon * 900.0).sin() * (lat * 700.0).cos()
    });
    let dem = dem.trim_end_matches(',');
    let style = format!(
        r##"{{"version": 8, "sources": {{{dem}}},
            "terrain": {{"source": "relief"}},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#d8e0c8"}}}},
              {{"id": "relief", "type": "hillshade", "source": "relief"}}
            ]}}"##
    );
    // Bare ground in the middle half of the view, which the first view
    // covers too.
    let bare = |pixels: &[u8]| {
        let (w, h) = (WIDTH as usize, HEIGHT as usize);
        (h / 4..h * 3 / 4)
            .flat_map(|y| (w / 4..w * 3 / 4).map(move |x| (y * w + x) * 4))
            .filter(|&i| pixels[i..i + 3] == GROUND)
            .count()
    };
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    map.jump_to(&camera((at, 14.4, 0.0, 0.0)));
    common::settle(&mut map, &mut target);
    map.jump_to(&camera((at, 13.9, 0.0, 0.0)));
    map.render(&mut target, 0.0).unwrap();
    let first = bare(&target.read_pixels().unwrap());
    common::settle(&mut map, &mut target);
    let settled = bare(&target.read_pixels().unwrap());
    let middle = WIDTH as usize * HEIGHT as usize / 4;
    assert!(
        first <= settled + middle / 100,
        "{first} of {middle} pixels in the middle are bare in the first frame, {settled} once settled"
    );
}

/// A hillshade multiplied over the ground (style-spec#1661's
/// `dev.sargunv.orbis:hillshade-layer-blend`) shades it by its own colour:
/// each pixel is the ground's colour times the hillshade drawn normally
/// over white, W3C multiply's identity, on a flat map, the globe and
/// terrain (lit by the terrain pass, viewport-anchored, which blends it
/// into the drape beneath, or draped, map-anchored, a terrain layer
/// composited apart), on the globe's terrain too, and over another lit
/// hillshade (whose terrain layer it then blends into): each pixel is the
/// map beneath it times the hillshade drawn normally over white. Where the
/// ground doesn't show (the sky, space), the map is as without the
/// hillshade.
#[test]
fn a_multiplied_hillshade_shades_the_ground_by_its_colour() {
    let (dem, _dir) = alps_dem();
    let dem = dem.trim_end_matches(',');
    let at = Geodetic::new(7.9, 46.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let terrain = r#""terrain": {"source": "alps", "exaggeration": 1.2},"#;
    let views: [(&str, &str, At, &str, bool); 6] = [
        ("flat", "", (at, 10.0, 0.0, 0.0), "viewport", false),
        (
            "globe",
            r#""projection": {"type": "globe"},"#,
            (at, 5.0, 0.0, 0.0),
            "viewport",
            false,
        ),
        (
            "terrain, lit",
            terrain,
            (at, 10.5, 20.0, 55.0),
            "viewport",
            false,
        ),
        (
            "terrain, draped",
            terrain,
            (at, 10.5, 20.0, 55.0),
            "map",
            false,
        ),
        (
            "globe terrain",
            r#""projection": {"type": "globe"}, "terrain": {"source": "alps", "exaggeration": 1.2},"#,
            (at, 8.5, 0.0, 45.0),
            "viewport",
            false,
        ),
        (
            "terrain, over a hillshade",
            terrain,
            (at, 10.5, 20.0, 55.0),
            "viewport",
            true,
        ),
    ];
    for (name, extra, view, anchor, under) in views {
        // `under`: another hillshade, lit from the north-east, beneath it.
        let mut draw = |ground: &str, hillshade: Option<&str>, under: bool| {
            let (visibility, blend) = match hillshade {
                Some(blend) => ("visible", blend),
                None => ("none", "normal"),
            };
            let beneath = if under { "visible" } else { "none" };
            let style = format!(
                r##"{{"version": 8, {extra}
                    "sources": {{{dem}}},
                    "layers": [
                      {{"id": "ground", "type": "background", "paint": {{"background-color": "{ground}"}}}},
                      {{"id": "under", "type": "hillshade", "source": "alps",
                        "layout": {{"visibility": "{beneath}"}},
                        "paint": {{"hillshade-illumination-direction": 45,
                          "hillshade-shadow-color": "#204080"}}}},
                      {{"id": "relief", "type": "hillshade", "source": "alps",
                        "layout": {{"visibility": "{visibility}"}},
                        "paint": {{"hillshade-illumination-anchor": "{anchor}",
                          "hillshade-shadow-color": "#3a2a5a", "hillshade-accent-color": "#806a90",
                          "dev.sargunv.orbis:hillshade-layer-blend": "{blend}"}}}}
                    ]}}"##
            );
            let mut map = Map::new(&runtime, options());
            map.set_style_json(&style).unwrap();
            map.jump_to(&camera(view));
            common::settle(&mut map, &mut target);
            target.read_pixels().unwrap()
        };
        let white = draw("#ffffff", None, false);
        let normal = draw("#ffffff", Some("normal"), false);
        let beneath = draw("#66aa33", None, under);
        let multiplied = draw("#66aa33", Some("multiply"), under);
        let (mut ground, mut shaded, mut off) = (0, 0, Vec::new());
        for i in (0..white.len()).step_by(4) {
            let px = |p: &[u8]| [p[i], p[i + 1], p[i + 2]];
            let (got, at) = (px(&multiplied), (i / 4) as u32);
            let (x, y) = (at % WIDTH, at / WIDTH);
            let expected = if px(&white) == [255; 3] {
                ground += 1;
                shaded += usize::from(px(&normal) != [255; 3]);
                let (n, b) = (px(&normal), px(&beneath));
                std::array::from_fn(|c| (f32::from(b[c]) * f32::from(n[c]) / 255.0).round() as u8)
            } else {
                px(&beneath)
            };
            if !got.iter().zip(expected).all(|(&g, e)| g.abs_diff(e) <= 3) {
                off.push(format!("({x}, {y}) is {got:?}, expected {expected:?}"));
            }
        }
        let pixels = white.len() / 4;
        assert!(off.is_empty(), "{name}: {off:#?}");
        assert!(ground > pixels / 4, "{name}: {ground} ground pixels");
        assert!(shaded > ground / 4, "{name}: {shaded} of {ground} shaded");
    }
}

/// A blended hillshade on terrain fades into the fog rather than taking
/// its colour: the ground it blends with is fogged already, so fogged as
/// its own content it would multiply the far ground by the fog's colour
/// again. Toward the horizon, where the fog is thick, the map is as
/// without the hillshade, whether the terrain pass lights it (the fog
/// falls on what it blended into) or it drapes, a terrain layer
/// composited apart (which fades).
#[test]
fn a_blended_hillshade_fades_into_the_fog() {
    // Ridges reaching past the horizon.
    let (dem, _dir) = dem("ridges", (0.0, 40.0, 16.0, 56.0), 7, |lon, lat| {
        1500.0 + 1200.0 * (lon * 9.0).sin() * (lat * 7.0).cos()
    });
    let dem = dem.trim_end_matches(',');
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    for anchor in ["viewport", "map"] {
        let mut draw = |ground: &str, visibility: &str| {
            let style = format!(
                r##"{{"version": 8,
                "terrain": {{"source": "ridges", "exaggeration": 1.2}},
                "sky": {{"sky-color": "#4f8fd9", "horizon-color": "#8899aa", "fog-color": "#8899aa",
                         "fog-ground-blend": 0.2, "horizon-fog-blend": 0.9}},
                "sources": {{{dem}}},
                "layers": [
                  {{"id": "ground", "type": "background", "paint": {{"background-color": "{ground}"}}}},
                  {{"id": "relief", "type": "hillshade", "source": "ridges",
                    "layout": {{"visibility": "{visibility}"}},
                    "paint": {{"hillshade-shadow-color": "#3a2a5a",
                      "hillshade-illumination-anchor": "{anchor}",
                      "dev.sargunv.orbis:hillshade-layer-blend": "multiply"}}}}
                ]}}"##
            );
            let mut map = Map::new(&runtime, options());
            map.set_style_json(&style).unwrap();
            map.jump_to(&camera((Geodetic::new(8.0, 43.0), 7.5, 0.0, 75.0)));
            common::settle(&mut map, &mut target);
            target.read_pixels().unwrap()
        };
        let bare = draw("#66aa33", "none");
        let red = draw("#ff0000", "none");
        let shaded = draw("#66aa33", "visible");
        let row = |y: u32| (y * WIDTH * 4) as usize..((y + 1) * WIDTH * 4) as usize;
        // The horizon: the first row mostly ground, which the background's
        // colour shows through.
        let horizon = (0..HEIGHT)
            .find(|&y| {
                let (a, b) = (&bare[row(y)], &red[row(y)]);
                a.chunks(4).zip(b.chunks(4)).filter(|(a, b)| a != b).count() > WIDTH as usize / 2
            })
            .expect("ground in view");
        let shading = |rows: std::ops::Range<u32>| {
            let n = rows.len() as f64 * f64::from(WIDTH);
            let sum: u64 = (rows.flat_map(&row))
                .map(|i| u64::from(bare[i].abs_diff(shaded[i])))
                .sum();
            sum as f64 / n
        };
        let band = HEIGHT / 20;
        let (far, near) = (
            shading(horizon..horizon + band),
            shading(HEIGHT - band..HEIGHT),
        );
        assert!(horizon < HEIGHT / 2, "the horizon at row {horizon}");
        assert!(
            near > 4.0,
            "{anchor}: the hillshade shades the near ground by {near}"
        );
        assert!(
            far < near / 2.0,
            "{anchor}: the hillshade shades the far ground by {far}, the near by {near}"
        );
    }
}

/// Edits ease alike on terrain and off it: a draped fill hidden, its
/// feature state changed and its paint edited fade frame by frame (drape
/// slots draw them eased each frame they run), and no frame draws nothing
/// while the layer composite a fade goes through compiles.
#[test]
fn draped_edits_ease_as_flat_ones_do() {
    let (lon, lat) = (139.76, 35.68);
    let (hills, _dem) = hills();
    let centre = |pixels: &[u8]| {
        let at = ((HEIGHT / 2 * WIDTH + WIDTH / 2) * 4) as usize;
        [pixels[at], pixels[at + 1], pixels[at + 2]]
    };
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    for terrain in [false, true] {
        let style = format!(
            r##"{{"version": 8, "transition": {{"duration": 600}},
            "sources": {{{}
              "square": {{"type": "geojson", "data": {{"type": "Feature", "id": 1, "properties": {{}},
                "geometry": {{"type": "Polygon", "coordinates": [[[{w}, {s}], [{e}, {s}], [{e}, {n}], [{w}, {n}], [{w}, {s}]]]}}}}}}
            }},
            {}
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {{"id": "square", "type": "fill", "source": "square", "paint": {{"fill-color":
                  ["case", ["boolean", ["feature-state", "hover"], false], "#ff0000", "#00ff00"]}}}}
            ]}}"##,
            hills,
            match terrain {
                true => r#""terrain": {"source": "hills", "exaggeration": 1},"#,
                false => "",
            },
            w = lon - 0.05,
            e = lon + 0.05,
            s = lat - 0.05,
            n = lat + 0.05,
        );
        for commands in [
            r#"[["setLayoutProperty", "square", "visibility", "none"]]"#,
            r#"[["setFeatureState", {"source": "square", "id": 1}, {"hover": true}]]"#,
            r#"[["setPaintProperty", "square", "fill-opacity", 0]]"#,
        ] {
            let mut map = Map::new(&runtime, options());
            map.set_style_json(&style).unwrap();
            map.jump_to(&camera((Geodetic::new(lon, lat), 12.0, 0.0, 30.0)));
            common::settle(&mut map, &mut target);
            let before = centre(&target.read_pixels().unwrap());
            map.edit(&orb_api::Transaction::from_commands(commands).unwrap())
                .unwrap();
            // Frame times run on from well past the settled frames', 20 ms
            // apart whatever the wall clock does: under load a frame took
            // most of the fade, which then seemed to snap.
            let mut seen = Vec::new();
            for frame in 0..40 {
                let time = 1e6 + f64::from(frame) * 20.0;
                map.render(&mut target, time).unwrap();
                let pixels = target.read_pixels().unwrap();
                assert!(
                    pixels.chunks(4).all(|p| p[3] > 0),
                    "terrain {terrain}, {commands}: a frame drew nothing"
                );
                seen.push(centre(&pixels));
                std::thread::sleep(Duration::from_millis(20));
            }
            common::settle(&mut map, &mut target);
            let after = centre(&target.read_pixels().unwrap());
            let c = (0..3)
                .max_by_key(|&c| before[c].abs_diff(after[c]))
                .unwrap();
            let (lo, hi) = (before[c].min(after[c]), before[c].max(after[c]));
            assert!(hi - lo > 200, "{commands}: {before:?} → {after:?}");
            assert!(
                seen.iter().any(|p| p[c] > lo + 20 && p[c] < hi - 20),
                "terrain {terrain}, {commands} snapped: {seen:?}"
            );
        }
    }
}

/// Terrain switched on by an edit rises to what a fresh map with it
/// draws, camera and all; switched off with its DEM source gone, it falls
/// flat as the camera moves, quietly, to what a fresh flat map draws.
#[test]
fn terrain_switched_by_edits_ends_as_a_fresh_map_does() {
    let (hills, _dem) = hills();
    let flat = tokyo("", "");
    let terrain = tokyo(
        r#""terrain": {"source": "hills", "exaggeration": 1.5},"#,
        &hills,
    );
    let at = (STATION, 13.0, 30.0, 60.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, options());
    map.set_style_json(&flat).unwrap();
    map.jump_to(&camera(at));
    common::settle(&mut map, &mut target);
    let warnings = |map: &mut Map| -> Vec<String> {
        std::iter::from_fn(|| map.poll_event())
            .filter_map(|e| match e {
                orb_api::Event::Map(orb_api::MapEvent::Warning { message }) => Some(message),
                _ => None,
            })
            .collect()
    };
    warnings(&mut map);
    let restyle = |json: &str| {
        orb_api::Transaction::from_commands(&format!(r#"[["setStyle", {json}]]"#)).unwrap()
    };
    map.edit(&restyle(&terrain)).unwrap();
    common::settle(&mut map, &mut target);
    let on = (target.read_pixels().unwrap(), map.camera());
    let (fresh, fresh_camera) = fresh_at(&runtime, &mut target, &terrain, at);
    assert_eq!(on.1, fresh_camera, "the camera on the risen terrain");
    assert_same("terrain-on", (on.0, fresh));
    map.edit(&restyle(&flat)).unwrap();
    for step in 0..30 {
        let s = f64::from(step);
        map.jump_to(&camera((STATION, 13.0 + s * 0.03, 30.0 + s, 60.0)));
        map.render(&mut target, 0.0).unwrap();
        std::thread::sleep(Duration::from_millis(10));
    }
    map.jump_to(&camera(at));
    common::settle(&mut map, &mut target);
    let off = target.read_pixels().unwrap();
    assert_eq!(warnings(&mut map), Vec::<String>::new());
    let (fresh, _) = fresh_at(&runtime, &mut target, &flat, at);
    assert_same("terrain-off", (off, fresh));
}

/// A plateau 2,400 m up rising 6 km south of a valley at 300 m, its top
/// red over green ground, seen north at zoom 11.5 and a pitch of 65°: the
/// view's bottom edge falls on the plateau, about 7.3 km south, though at
/// the valley's height it would fall short of it. The cover bounds tiles
/// by the DEM tiles their zoom reads, and near the eye a parent's coarser
/// DEM tile may not have loaded while its children's have: bounded below
/// the plateau as if it stood at the target's height, the parent was
/// culled, with it the children, and nothing asked for the DEM tile that
/// would bring them back. The tiles beyond showed their skirts through
/// the hole, as streaks along the bottom edge.
#[test]
fn ground_near_the_eye_shows_before_its_coarser_dem_tiles_load() {
    let (lon, lat) = (7.9, 46.0);
    let edge = lat - 6.0 / 111.0;
    let (source, _dem) = dem("plateau", (7.6, 45.7, 8.2, 46.3), 12, |_, la| {
        if la < edge { 2400.0 } else { 300.0 }
    });
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    for projection in ["mercator", "globe"] {
        let style = format!(
            r##"{{"version": 8, "projection": {{"type": "{projection}"}},
              "sources": {{{source}
                "top": {{"type": "geojson", "data": {{"type": "Polygon", "coordinates":
                  [[[7.6, 45.7], [8.2, 45.7], [8.2, {edge}], [7.6, {edge}], [7.6, 45.7]]]}}}}}},
              "terrain": {{"source": "plateau", "exaggeration": 1}},
              "layers": [
                {{"id": "ground", "type": "background", "paint": {{"background-color": "#00ff00"}}}},
                {{"id": "top", "type": "fill", "source": "top",
                  "paint": {{"fill-color": "#ff0000", "fill-antialias": false}}}}
              ]}}"##
        );
        let at = (Geodetic::new(lon, lat), 11.5, 0.0, 65.0);
        let (pixels, _) = fresh_at(&runtime, &mut target, &style, at);
        for x in (0..8).map(|i| WIDTH * (2 * i + 1) / 16) {
            let k = (((HEIGHT - 2) * WIDTH + x) * 4) as usize;
            assert_eq!(
                pixels[k..k + 3],
                [255, 0, 0],
                "{projection}: the bottom edge at x {x} misses the plateau"
            );
        }
    }
}
