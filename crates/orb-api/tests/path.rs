//! Every frame is right for its camera: a map brought to a camera through
//! a gesture draws, once loaded and faded, the pixels a fresh map at rest
//! at that camera draws. Nothing the path left behind (resident tiles of
//! other zooms, rows, cull state, warps, paint written for other zooms)
//! may show.

use std::path::{Path, PathBuf};
use std::time::Duration;

use orb_api::{
    CameraTarget, Constraints, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, Target,
};

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

/// Lets a globe's camera look past its horizon.
fn steep() -> Constraints {
    Constraints {
        max_pitch: 179.0,
        ..Constraints::default()
    }
}

/// A map's options for these tests: labels that depend on the camera
/// alone, which MapLibre Native's stabilisers (on by default) don't.
fn options() -> MapOptions {
    MapOptions {
        stable_labels: false,
        ..MapOptions::default()
    }
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

    let mut moved = Map::new(&runtime, options());
    moved.set_constraints(steep());
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
    settle(&mut fresh, target);
    (target.read_pixels().unwrap(), fresh.camera())
}

/// Asserts the two pictures equal. On failure, writes both and their
/// difference to a fresh directory and names it.
fn assert_same(name: &str, pair: (Vec<u8>, Vec<u8>)) {
    assert_within(name, pair, 0);
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
        data = data().display(),
        points = features(0),
        towers = features(1),
    );
    let start = (Geodetic::new(-3.0, 18.0), 4.0, 0.0, 0.0);
    let end = (Geodetic::new(2.3522, 52.0566), 4.3, 180.0, 95.0);
    assert_same("lifted", both_ways(&style, start, end, 20));
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
    // One directory per call: tests run in parallel, and the clock may
    // give two of them the same time, then one's drop would delete the
    // other's tiles under its fresh map.
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "orb-path-dem-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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
/// running across the pixel grid shades as dotted lines.
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

/// Terrain that comes on after the style's first frame, its DEM source
/// behind a TileJSON: a fresh map that never moves drapes the buildings
/// its near drape slots' zoom shows (they show from zoom 13; the view is
/// north of the extract, which lies in its near slots), as a map that
/// moved there after the terrain came on does. Paint written before
/// the terrain was on must not stay.
#[test]
fn a_terrain_that_comes_on_late_drapes_what_a_moved_map_does() {
    let (hills, dem) = hills();
    let tilejson = dem.0.join("tiles.json");
    let tiles = format!("file://{}/{{z}}/{{x}}/{{y}}.png", dem.0.display());
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
    // Drape texels filtered into a pixel, each a step apart: two.
    assert_within("late-terrain", both_ways(&style, start, end, 5), 2);
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
    // The drape differs from a fresh map's by a step in a channel at some
    // pixels along draped lines' edges: their vertices and the terrain's
    // samples round by where in the atlas their slot is, which follows
    // the tiles that came and went.
    assert_within("terrain", both_ways(&style, start, end, 30), 1);
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
    // The hillshade drapes its slopes: they round by where in the atlas
    // their slot is, as the terrain gesture's lines do, a step in a channel
    // at scattered pixels.
    assert_within("alps", both_ways(&alps, start, end, 30), 1);
}

/// A view at rest over the Alps stays put: frames after idle cover the
/// same terrain tiles and read the same DEM tiles, so they pad no DEM
/// borders, write no rows, keep the cull's candidates and run one cull,
/// from the first frame after idle on. Here the DEM tile under one
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
    settle(&mut map, &mut target);
    for frame in 1..=4 {
        let report = map.render(&mut target, 0.0).unwrap().unwrap();
        assert!(report.idle, "frame {frame}");
        assert_eq!(report.frame.prepared, 0, "frame {frame}");
        assert_eq!(report.frame.placements_written, 0, "frame {frame}");
        assert_eq!(report.frame.dispatches, 1, "frame {frame}");
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
    // A step at scattered pixels, as in the alpine gesture.
    assert_within("crest", both_ways(&alps, start, end, 20), 1);
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
    let mut moved = Map::new(&runtime, options());
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
    let mut fresh = Map::new(&runtime, options());
    fresh.set_style_json(&style).unwrap();
    fresh.edit(&hide).unwrap();
    fresh.jump_to(&camera(at(-32.0)));
    settle(&mut fresh, &mut target);
    // The drape rounds by its slots' places as in the terrain gesture.
    assert_within("live-alps", (turned, target.read_pixels().unwrap()), 1);
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
    settle(&mut moved, &mut target);
    moved.jump_to(&camera(at(16.0)));
    moved.render(&mut target, 0.0).unwrap();
    let jumped = centre(&target.read_pixels().unwrap());
    drop(moved);
    let mut fresh = Map::new(&runtime, options());
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
        let mut map = Map::new(&runtime, options());
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
        data = data().display(),
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
    settle(&mut map, &mut target);
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
    settle(&mut map, &mut target);
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
        settle(&mut map, &mut target);
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
    settle(&mut map, &mut target);
    map.jump_to(&camera((at, 13.9, 0.0, 0.0)));
    map.render(&mut target, 0.0).unwrap();
    let first = bare(&target.read_pixels().unwrap());
    settle(&mut map, &mut target);
    let settled = bare(&target.read_pixels().unwrap());
    let middle = WIDTH as usize * HEIGHT as usize / 4;
    assert!(
        first <= settled + middle / 100,
        "{first} of {middle} pixels in the middle are bare in the first frame, {settled} once settled"
    );
}
