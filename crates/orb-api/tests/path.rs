//! Every frame is right for its camera: a map brought to a camera through
//! a gesture draws, once loaded and faded, the pixels a fresh map at rest
//! at that camera draws. Nothing the path left behind (resident tiles of
//! other zooms, rows, cull state, warps, paint written for other zooms)
//! may show.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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
            "pmtiles://./data/",
            &format!("pmtiles://file://{}/", data.display()),
        )
        .replace("\"./data/", &format!("\"file://{}/", data.display()));
    let style = style.replacen('{', &format!("{{{extra}"), 1);
    style.replacen("\"sources\": {", &format!("\"sources\": {{{sources}"), 1)
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
    let started = Instant::now();
    let mut idle = 0;
    loop {
        let report = map.render(target).unwrap();
        idle = if report.is_some_and(|r| r.idle) && !map.needs_frame() {
            idle + 1
        } else {
            0
        };
        if idle == 2 {
            return;
        }
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the map never idled"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// The pixels a map draws at `end` once settled, after moving there frame
/// by frame from `start` (a gesture: one frame per step, nothing waits for
/// data), and the pixels a fresh map draws at rest at `end`.
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
        moved.render(&mut target).unwrap();
        std::thread::sleep(Duration::from_millis(8));
    }
    settle(&mut moved, &mut target);
    let gestured = target.read_pixels().unwrap();
    drop(moved);

    let mut fresh = Map::new(&runtime, MapOptions::default());
    fresh.set_style_json(style).unwrap();
    fresh.jump_to(&camera(end));
    settle(&mut fresh, &mut target);
    (gestured, target.read_pixels().unwrap())
}

/// Asserts the two pictures equal, to one step in a channel: the GPU cull
/// compacts chunks in no fixed order, so overlapping translucent pieces of
/// one layer may blend in either order and round apart, even between two
/// fresh maps. On failure, writes both and their difference to a fresh
/// directory and names it.
fn assert_same(name: &str, (gestured, fresh): (Vec<u8>, Vec<u8>)) {
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
        .filter(|&(_, d)| d > 1)
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
            let close = a.iter().zip(b).all(|(x, y)| x.abs_diff(*y) <= 1);
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

const STATION: Geodetic = Geodetic {
    lon: 139.7671,
    lat: 35.6812,
};

#[test]
fn zooming_in_with_a_turn_and_tilt_draws_what_a_fresh_map_does() {
    let start = (Geodetic::new(139.72, 35.67), 12.6, 0.0, 0.0);
    let end = (STATION, 15.4, 35.0, 50.0);
    assert_same("mercator-in", both_ways(&tokyo("", ""), start, end, 30));
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

/// Terrarium DEM tiles of rolling hills over central Tokyo, zooms 0–11,
/// written to a fresh directory; the source's JSON.
fn hills() -> String {
    let dir = std::env::temp_dir().join(format!(
        "orb-path-dem-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (west, south, east, north) = (139.2, 35.2, 140.3, 36.2);
    const SIZE: u32 = 256;
    for z in 0..=11u8 {
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
                        let metres = 600.0
                            + 500.0 * (lon * 40.0).sin() * (lat * 55.0).cos()
                            + 150.0 * (lon * 170.0 + lat * 90.0).sin();
                        let v = ((metres + 32768.0) * 256.0).round() as u32;
                        pixels.extend([(v >> 16) as u8, (v >> 8) as u8, v as u8, 255]);
                    }
                }
                let path = dir.join(format!("{z}/{x}/{y}.png"));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                write_png(&path, SIZE, SIZE, &pixels);
            }
        }
    }
    format!(
        r#""hills": {{"type": "raster-dem", "encoding": "terrarium", "tileSize": {SIZE},
            "maxzoom": 11, "bounds": [{west}, {south}, {east}, {north}],
            "tiles": ["file://{}/{{z}}/{{x}}/{{y}}.png"]}},"#,
        dir.display()
    )
}

/// Off until drape slots key on content (#80): one pixel at the extract's
/// edge keeps a stain from earlier in the gesture in about one run in six.
#[test]
#[ignore = "#80: a drape slot keeps a stain from earlier in the gesture"]
fn a_terrain_gesture_draws_what_a_fresh_map_does() {
    let style = tokyo(
        r#""terrain": {"source": "hills", "exaggeration": 1.5},"#,
        &hills(),
    );
    let start = (Geodetic::new(139.70, 35.66), 12.0, 0.0, 30.0);
    let end = (STATION, 13.8, 40.0, 60.0);
    assert_same("terrain", both_ways(&style, start, end, 30));
}
