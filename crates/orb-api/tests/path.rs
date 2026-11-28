//! Every frame is right for its camera: a map brought to a camera by any
//! path draws, once loaded and faded, the pixels a fresh map at rest at
//! that camera draws. Nothing the path left behind (resident tiles of
//! other zooms, rows, cull state, warps, paint written for other zooms, an
//! elevation a gesture held) may show.
//!
//! `both_ways` jumps the camera along a line, a frame a step, for what
//! stays resident and drawn along the way. `gestured` hands the map a
//! host's input (drags, a pinch-twist, wheel notches) and lets the glides
//! and eases they start run out, and `flown` flies it, both on a
//! display link's clock: what only moves made by the map itself reach,
//! such as the elevation a gesture holds and puts back at rest. The rest
//! check one part a path can leave behind or redo: drape slots and pages,
//! rows, terrain switched on and off, edits and their eases, labels'
//! clocks and stabilisers, memory, and what a move's first frame shows
//! before its tiles land. What a map at rest draws on terrain is
//! `terrain.rs`'s.

mod common;

use std::time::{Duration, Instant};

use common::dem::{alps, dem, hills};
use common::{At, STATION, camera, extruded, fresh, fresh_at, options, steep, tilted, tokyo};
use glam::DVec2;
use orb_api::{
    Button, FlyOptions, FrameReport, Geodetic, Input, Map, MapOptions, Modifiers, PointerKind,
    Runtime, RuntimeOptions, Target, WheelDelta,
};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 384;

/// Asserts the two pictures of the view these tests draw equal: the
/// moved map's and the fresh map's.
fn assert_same(name: &str, pictures: (Vec<u8>, Vec<u8>)) {
    common::assert_same(name, WIDTH, pictures);
}

/// Tokyo's buildings standing, as extrusions under the v8 `light`.
fn extruded_tokyo() -> String {
    extruded(tokyo("", ""))
}

/// Milliseconds between the frames of the display link these tests stand
/// in for.
const FRAME_MS: f64 = 16.0;

/// A display link's clock for one map: the frames a test draws itself,
/// and the input it hands the map, are a frame apart, so input times, and
/// with them the curves of glides and eases, are the same on every run
/// however long a frame takes to draw, while data still arrives on the
/// wall clock (over terrain, what a gesture grips depends on it); the
/// frames the map times itself while it settles go on from it, as a
/// host's would.
struct Clock {
    /// The last frame's time, and when it was drawn.
    time: f64,
    at: Instant,
}

impl Clock {
    /// Made with its map, before the map's first frame.
    fn new() -> Self {
        Self {
            time: 0.0,
            at: Instant::now(),
        }
    }

    /// Hands the map `inputs` and draws a frame, all at the next frame's
    /// time, then waits up to a frame for data.
    fn frame(&mut self, map: &mut Map, target: &mut Target, inputs: &[Input]) {
        self.time += FRAME_MS;
        self.at = Instant::now();
        for &input in inputs {
            map.input(input, self.time);
        }
        map.render(target, self.time).unwrap();
        map.wait_woken(Duration::from_millis(FRAME_MS as u64));
    }

    /// Draws until the map is idle, timed on the map's own clock, and
    /// catches up with it; the last frame's report.
    fn settle(&mut self, map: &mut Map, target: &mut Target) -> FrameReport {
        let report = map
            .render_until_idle(target, Duration::from_secs(60))
            .expect("the map idles");
        self.time += self.at.elapsed().as_secs_f64() * 1000.0;
        self.at = Instant::now();
        report
    }

    /// Draws frames until the camera stops: glides and eases run out.
    fn until_still(&mut self, map: &mut Map, target: &mut Target) {
        for _ in 0..1000 {
            if !map.is_moving() {
                return;
            }
            self.frame(map, target, &[]);
        }
        panic!("the camera never came to rest");
    }
}

/// What a live map draws once loaded and faded to idle after `drive`
/// moved it from `start` on its clock, and what a fresh map draws at rest
/// at `end`, or at the camera the moved map came to rest at. Labels too:
/// what idle shows depends on the camera and the data alone, not on the
/// order tiles arrived in or what showed on the way.
fn moved_and_fresh(
    style: &str,
    start: At,
    end: Option<At>,
    drive: impl FnOnce(&mut Map, &mut Target, &mut Clock),
) -> (Vec<u8>, Vec<u8>) {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);

    let mut moved = Map::new(&runtime, options());
    let mut clock = Clock::new();
    moved.set_constraints(steep());
    moved.set_style_json(style).unwrap();
    moved.jump_to(&camera(start));
    clock.settle(&mut moved, &mut target);
    let started = moved.camera();
    // The style's pipeline set came with it: what the path shows on the
    // way, and at its end, asks for none.
    let requested = runtime.stats().counters.pipelines_requested;
    drive(&mut moved, &mut target, &mut clock);
    clock.settle(&mut moved, &mut target);
    assert_eq!(
        runtime.stats().counters.pipelines_requested,
        requested,
        "the path's frames asked for pipelines the style's set left out"
    );
    let moved_pixels = target.read_pixels().unwrap();
    let rest = moved.camera();
    assert_ne!(rest, started, "the path never moved the camera");
    drop(moved);
    let (fresh, fresh_rest) = match end {
        Some(end) => fresh_at(&runtime, &mut target, style, end),
        None => fresh(&runtime, &mut target, style, |map| map.jump_to_state(rest)),
    };
    // At rest, the camera too is what a fresh one is: over terrain, it
    // stands on the same ground at the same height.
    assert_eq!(
        rest, fresh_rest,
        "the camera at rest differs from a fresh map's"
    );
    (moved_pixels, fresh)
}

/// What a live map draws at `end` once idle after jumping there from
/// `start` along a line, a frame a step, each step waiting at most a
/// frame for data, and what a fresh map draws at rest at `end`.
fn both_ways(style: &str, start: At, end: At, steps: u32) -> (Vec<u8>, Vec<u8>) {
    moved_and_fresh(style, start, Some(end), |map, target, clock| {
        let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
        for step in 1..=steps {
            let t = f64::from(step) / f64::from(steps);
            let (a, b) = (start.0, end.0);
            map.jump_to(&camera((
                Geodetic::new(lerp(a.lon, b.lon, t), lerp(a.lat, b.lat, t)),
                lerp(start.1, end.1, t),
                lerp(start.2, end.2, t),
                lerp(start.3, end.3, t),
            )));
            clock.frame(map, target, &[]);
        }
    })
}

/// A gesture as a display link brings it: the inputs of each frame.
type Gesture = Vec<Vec<Input>>;

/// The middle of the view, in logical pixels.
const MIDDLE: DVec2 = DVec2::new(WIDTH as f64 / 2.0, HEIGHT as f64 / 2.0);

/// A mouse drag with `button` from `from`, `by` pixels a frame for
/// `frames` frames, let go while it moves: it glides on.
fn drag(button: Button, from: DVec2, by: DVec2, frames: u32) -> Gesture {
    let mut gesture = vec![vec![Input::PointerDown {
        id: 1,
        kind: PointerKind::Mouse,
        button,
        position: from,
        modifiers: Modifiers::default(),
    }]];
    let mut at = from;
    for _ in 0..frames {
        at += by;
        gesture.push(vec![Input::PointerMove {
            id: 1,
            position: at,
        }]);
    }
    gesture.last_mut().unwrap().push(Input::PointerUp {
        id: 1,
        position: at,
    });
    gesture
}

/// Two fingers `radius` pixels either side of `center` spreading by
/// `scale` and turning `turn` degrees clockwise over `frames` frames, let
/// go while they move.
fn pinch(center: DVec2, radius: f64, scale: f64, turn: f64, frames: u32) -> Gesture {
    let finger = |id: u32, f: f64| {
        let angle = turn.to_radians() * f + std::f64::consts::PI * f64::from(id - 2);
        center + radius * scale.powf(f) * DVec2::new(angle.cos(), angle.sin())
    };
    let mut gesture = vec![
        [2, 3]
            .map(|id| Input::PointerDown {
                id,
                kind: PointerKind::Touch,
                button: Button::Primary,
                position: finger(id, 0.0),
                modifiers: Modifiers::default(),
            })
            .to_vec(),
    ];
    for frame in 1..=frames {
        let f = f64::from(frame) / f64::from(frames);
        gesture.push(
            [2, 3]
                .map(|id| Input::PointerMove {
                    id,
                    position: finger(id, f),
                })
                .to_vec(),
        );
    }
    let last = gesture.last_mut().unwrap();
    last.extend([2, 3].map(|id| Input::PointerUp {
        id,
        position: finger(id, 1.0),
    }));
    gesture
}

/// `count` wheel notches at `at`, one every `every` frames, each zooming
/// in by a notch.
fn notches(at: DVec2, count: u32, every: usize) -> Gesture {
    let mut gesture = Vec::new();
    for _ in 0..count {
        gesture.push(vec![Input::Wheel {
            position: at,
            delta: WheelDelta::Notches(-1.0),
            modifiers: Modifiers::default(),
        }]);
        gesture.extend(std::iter::repeat_n(Vec::new(), every - 1));
    }
    gesture
}

/// `frames` frames without input: what a glide or ease does meanwhile.
fn wait(frames: usize) -> Gesture {
    vec![Vec::new(); frames]
}

/// What a hand does to a map in a few seconds: drags it off and lets it
/// glide, pinches it in with a twist, turns and tilts it with the
/// secondary button, and wheels in off the middle, a few frames apart:
/// a press stops the glide before it.
fn handling() -> Gesture {
    [
        drag(Button::Primary, MIDDLE, DVec2::new(-14.0, -8.0), 12),
        wait(12),
        pinch(MIDDLE, 60.0, 1.6, 30.0, 16),
        wait(12),
        drag(Button::Secondary, MIDDLE, DVec2::new(6.0, -5.0), 10),
        wait(8),
        notches(MIDDLE + DVec2::new(80.0, 40.0), 4, 4),
    ]
    .concat()
}

/// What a live map draws once idle after `gesture` from `start` and the
/// glides and eases it started, and what a fresh map draws at rest at the
/// camera the gesture left.
fn gestured(style: &str, start: At, gesture: &[Vec<Input>]) -> (Vec<u8>, Vec<u8>) {
    moved_and_fresh(style, start, None, |map, target, clock| {
        let before = map.camera();
        for inputs in gesture {
            clock.frame(map, target, inputs);
        }
        clock.until_still(map, target);
        let after = map.camera();
        assert_ne!(
            after.resolution, before.resolution,
            "the gesture never zoomed"
        );
        assert_ne!(after.bearing, before.bearing, "the gesture never turned");
        assert_ne!(after.pitch, before.pitch, "the gesture never tilted");
    })
}

/// What a live map draws once idle after flying from `start` to `end`,
/// and what a fresh map draws at rest at `end`.
fn flown(style: &str, start: At, end: At) -> (Vec<u8>, Vec<u8>) {
    moved_and_fresh(style, start, Some(end), |map, target, clock| {
        map.fly_to(&camera(end), &FlyOptions::default());
        clock.until_still(map, target);
    })
}

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

/// A map moved to a display of another pixel ratio draws what a fresh
/// map started there draws: from 1 to 2 its sprites load again at `@2x`,
/// and from 2 to 3 its icons lay out again, filtered as icons drawn at
/// another ratio than their own are.
#[test]
fn a_new_pixel_ratio_draws_what_a_map_started_there_does() {
    let style = tokyo("", "");
    let at = (STATION, 16.0, 0.0, 0.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let fetched = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let seen = fetched.clone();
    runtime.set_request_transform(Some(std::sync::Arc::new(move |url: &str| {
        seen.lock().unwrap().push(url.to_owned());
        Some(orb_api::Request {
            url: url.to_owned(),
            headers: Vec::new(),
        })
    })));
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut moved = Map::new(&runtime, options());
    moved.set_constraints(steep());
    moved.set_style_json(&style).unwrap();
    moved.jump_to(&camera(at));
    common::settle(&mut moved, &mut target);
    // The sprite resources fetched since the record was cleared.
    let sprites = || -> Vec<String> {
        let fetched = fetched.lock().unwrap();
        let sprites = fetched.iter().filter(|url| url.contains("/sprites/ofm"));
        sprites.cloned().collect()
    };
    let one = sprites();
    assert!(
        one.iter().any(|url| url.ends_with("/ofm.png"))
            && !one.iter().any(|url| url.contains("ofm@2x")),
        "a map at 1 loads the 1x sprite: {one:?}"
    );
    for scale in [2.0, 3.0] {
        // Not what the fresh map at the last ratio fetched.
        fetched.lock().unwrap().clear();
        target.resize(WIDTH, HEIGHT, scale);
        common::settle(&mut moved, &mut target);
        let loaded = sprites();
        if scale == 2.0 {
            assert!(
                loaded.iter().any(|url| url.ends_with("/ofm@2x.png")),
                "a map moved to 2 loads the @2x sprite: {loaded:?}"
            );
        } else {
            assert!(
                loaded.is_empty(),
                "a map moved from 2 to {scale} keeps its sprite: {loaded:?}"
            );
        }
        let moved_pixels = target.read_pixels().unwrap();
        let (fresh_pixels, _) = fresh_at(&runtime, &mut target, &style, at);
        assert_same(&format!("ratio-{scale}"), (moved_pixels, fresh_pixels));
    }
}

/// Panning around a circle frame by frame, as the bench's moving camera
/// does, stopping six times to idle and compare with a fresh map there.
#[test]
fn panning_frame_by_frame_draws_what_a_fresh_map_does() {
    let style = tokyo("", "");
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, options());
    let mut clock = Clock::new();
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
    clock.settle(&mut map, &mut target);
    for stop in 1..=6 {
        for f in (stop - 1) * 20 + 1..=stop * 20 {
            map.jump_to(&camera(at(f)));
            clock.frame(&mut map, &mut target, &[]);
        }
        clock.settle(&mut map, &mut target);
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

/// Labels whose filter reads the camera, the far ones of a tilted view
/// dropped by their distance from the centre and their own rank, show
/// after a tilt and turn what a fresh map there shows.
#[test]
fn camera_filtered_labels_after_a_gesture_are_a_fresh_map_s() {
    let style = tokyo("", "").replacen(
        "{\n      \"id\": \"poi_r20\",",
        r#"{"id": "poi-near", "type": "symbol", "source": "openmaptiles",
            "source-layer": "poi", "minzoom": 14,
            "filter": ["all", [">=", ["get", "rank"], 1],
                ["any", ["<", ["pitch"], 20],
                 ["<", ["dev.sargunv.orbis:distance-from-center"], ["/", ["get", "rank"], 30]]]],
            "layout": {"text-field": ["get", "name"], "text-font": ["Noto Sans Italic"],
                       "text-size": 12, "text-allow-overlap": true}},
        {"id": "poi_r20","#,
        1,
    );
    assert!(style.contains("poi-near"));
    let start = (Geodetic::new(139.735, 35.682), 15.0, 0.0, 0.0);
    let end = (STATION, 15.8, 120.0, 60.0);
    let (gestured, fresh) = both_ways(&style, start, end, 30);
    // The filter drops labels there.
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let unfiltered = style.replacen(r#"["<", ["pitch"], 20]"#, "true", 1);
    let (all, _) = fresh_at(&runtime, &mut target, &unfiltered, end);
    assert_ne!(all, fresh, "the filter drops no label");
    assert_same("camera-filter", (gestured, fresh));
}

#[test]
fn zooming_out_draws_what_a_fresh_map_does() {
    let start = (STATION, 16.6, -20.0, 30.0);
    let end = (Geodetic::new(139.75, 35.69), 14.3, 10.0, 0.0);
    assert_same("mercator-out", both_ways(&tokyo("", ""), start, end, 30));
}

/// A map handled by hand, its glides and eases run out, draws what a
/// fresh map at the camera they left draws.
#[test]
fn a_handled_map_draws_what_a_fresh_map_does() {
    let start = (Geodetic::new(139.721, 35.701), 13.0, 0.0, 20.0);
    assert_same("handled", gestured(&tokyo("", ""), start, &handling()));
}

/// The same on the globe, whose drags turn the planet under the camera.
#[test]
fn a_handled_globe_draws_what_a_fresh_globe_does() {
    let style = tokyo(r#""projection": {"type": "globe"},"#, "");
    let start = (Geodetic::new(139.35, 35.92), 9.0, 0.0, 20.0);
    assert_same("handled-globe", gestured(&style, start, &handling()));
}

#[test]
fn a_globe_gesture_draws_what_a_fresh_globe_does() {
    let style = tokyo(r#""projection": {"type": "globe"},"#, "");
    let start = (Geodetic::new(139.6, 35.6), 9.0, 0.0, 0.0);
    let end = (STATION, 12.3, -25.0, 40.0);
    assert_same("globe", both_ways(&style, start, end, 30));
}

/// A globe spun across the world from where it keeps each source's
/// coarsest level loaded to stand in, and zoomed in past where it does,
/// draws what a fresh globe there does, labels too: the coarse tiles it
/// kept (the whole world, the extract's zoom 0) only ever stand in.
#[test]
fn a_globe_spun_past_its_coarse_level_draws_what_a_fresh_globe_does() {
    let style = tokyo(r#""projection": {"type": "globe"},"#, "");
    let start = (Geodetic::new(-60.0, 10.0), 2.0, 0.0, 0.0);
    for (name, zoom) in [("globe-coarse", 3.2), ("globe-past-coarse", 5.0)] {
        let end = (Geodetic::new(139.74, 35.685), zoom, 0.0, 0.0);
        assert_same(name, both_ways(&style, start, end, 30));
    }
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
        common::write_png(&dir.join("held.png"), WIDTH, HEIGHT, &held);
        common::write_png(&dir.join("fresh.png"), WIDTH, HEIGHT, &fresh);
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

/// Terrain that comes on after the style's first frame, its DEM source
/// behind a TileJSON: a fresh map that never moves drapes the buildings
/// its near drape slots' zoom shows (they show from zoom 13; the view is
/// north of the extract, which lies in its near slots), as a map that
/// moved there after the terrain came on does. Paint written before
/// the terrain was on must not stay.
#[test]
fn a_terrain_that_comes_on_late_drapes_what_a_moved_map_does() {
    let (hills, dem) = hills();
    let tilejson = dem.path().join("tiles.json");
    let tiles = format!("file://{}/{{z}}/{{x}}/{{y}}.png", dem.path().display());
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

/// Hills handled by hand: a gesture holds the target's elevation, and at
/// rest the target stands on the ground under the middle of the view, as
/// a fresh map's at that camera does, whatever DEM tiles had landed when
/// the hand let go.
#[test]
fn handled_terrain_draws_what_a_fresh_map_does() {
    let (hills, _dem) = hills();
    let style = tokyo(
        r#""terrain": {"source": "hills", "exaggeration": 1.5},"#,
        &hills,
    );
    let start = (Geodetic::new(139.705, 35.712), 12.0, 0.0, 10.0);
    assert_same("handled-terrain", gestured(&style, start, &handling()));
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

/// Tokyo over the test hills, its buildings standing, with and without
/// their terrain, and the terrain: the 3D layers' depth has a stencil
/// with the terrain and none without. Its background is magenta under a
/// fill of the background's own colour that covers the ground, so that a
/// frame shows no tile where it shows the magenta, or nothing ([`bare`]).
/// Its scratch DEM lives as long as the guard.
fn terrain_turns() -> (String, String, &'static str, tempfile::TempDir) {
    let (hills, dem) = hills();
    let ground = r#""ground": {"type": "geojson", "data": {"type": "Polygon",
        "coordinates": [[[139.2, 35.2], [140.3, 35.2], [140.3, 36.2], [139.2, 36.2], [139.2, 35.2]]]}},"#;
    let style = |terrain: &str| {
        let style = extruded(tokyo(terrain, &format!("{hills}{ground}")));
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
    (with, style(""), terrain, dem)
}

/// Pixels of a frame of [`terrain_turns`]' styles that show no tile: the
/// background's magenta, however the terrain shades it (nothing in
/// Tokyo's style is near it), or nothing at all, where neither the
/// terrain nor the map's layers drew (a terrain waiting for its pipeline
/// leaves the target clear).
fn bare(pixels: &[u8]) -> usize {
    (pixels.chunks(4))
        .filter(|p| p[3] < 255 || u16::from(p[1]) + 96 < u16::from(p[0].min(p[2])))
        .count()
}

/// Where [`terrain_turns`]' maps look: steep enough that the terrain's
/// hills show.
const TERRAIN_TURN_AT: At = (STATION, 14.5, 30.0, 55.0);

/// A map of `style` at [`TERRAIN_TURN_AT`], idle, and the clock its
/// frames are timed by, in milliseconds: settling times its frames on
/// from the last one's.
fn idle_at(runtime: &Runtime, target: &mut Target, style: &str) -> (Map, impl Fn() -> f64 + use<>) {
    let mut map = Map::new(runtime, options());
    map.set_constraints(steep());
    map.set_style_json(style).unwrap();
    map.jump_to(&camera(TERRAIN_TURN_AT));
    let started = std::time::Instant::now();
    let clock = move || started.elapsed().as_secs_f64() * 1000.0;
    map.render(target, clock()).unwrap();
    common::settle(&mut map, target);
    (map, clock)
}

/// Sets the map's terrain to `terrain`, JSON (`null` for none).
fn set_terrain(map: &mut Map, terrain: &str) {
    let tx = format!(r#"[["setTerrain", {terrain}]]"#);
    map.edit(&orb_api::Transaction::from_commands(&tx).unwrap())
        .unwrap();
}

/// Draws `map`'s frames from time `t` (milliseconds), 1/60 s apart
/// however long each takes to draw, until it is idle, and returns them,
/// with last the first idle one, at rest as a fresh map is. A frame whose
/// tiles are still arriving is drawn again at its time once they have, so
/// that what the frames show doesn't depend on how fast this machine
/// loads them: as terrain rises the view reaches ground the start never
/// loaded.
fn frames_until_idle(map: &mut Map, target: &mut Target, mut t: f64, name: &str) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    let started = std::time::Instant::now();
    loop {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "{name}: never idle"
        );
        let report = map.render(target, t).unwrap();
        let Some(report) = report.filter(|r| r.tiles_loading == 0) else {
            std::thread::sleep(Duration::from_millis(4));
            continue;
        };
        t += 1000.0 / 60.0;
        frames.push(target.read_pixels().unwrap());
        if report.idle() {
            return frames;
        }
    }
}

/// Turning terrain on raises it from flat over the style's transition,
/// and turning it off eases it flat and then drops it, with the map's
/// tiles drawn on every frame between (once, removing terrain left the
/// drape no room while the terrain fell, and every frame of the fall drew
/// the labels alone); the frames between differ from both ends (they ease
/// rather than jump), and at rest the map draws what a fresh map with or
/// without the terrain does at that camera.
#[test]
fn turning_terrain_on_and_off_eases_with_every_tile_drawn() {
    let (with, without, terrain, _dem) = terrain_turns();
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let turn = |(start, terrain, end): (&str, &str, &str), name: &str| {
        let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
        let (mut map, clock) = idle_at(&runtime, &mut target, start);
        let mut frames = vec![target.read_pixels().unwrap()];
        set_terrain(&mut map, terrain);
        frames.extend(frames_until_idle(&mut map, &mut target, clock(), name));
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(bare(frame), 0, "{name}: frame {i} shows no tile");
        }
        let (first, last) = (&frames[0], frames.last().unwrap());
        assert_ne!(first, last, "{name} changed nothing");
        assert!(
            frames.iter().any(|frame| frame != first && frame != last),
            "{name}: the map jumped rather than eased"
        );
        let camera = map.camera();
        let (fresh, fresh_camera) = fresh_at(&runtime, &mut target, end, TERRAIN_TURN_AT);
        assert_eq!(
            camera, fresh_camera,
            "{name}: the camera differs from a fresh map's"
        );
        assert_same(name, (last.clone(), fresh));
    };
    turn((&without, terrain, &with), "terrain-on");
    turn((&with, "null", &without), "terrain-off");
}

/// The child's role in
/// [`turning_terrain_on_or_off_waits_for_the_pipelines_it_draws_with`].
const SLOW_CHILD: &str = "ORB_TEST_SLOW_TERRAIN_CHILD";

/// Terrain draws in the map's own passes, whose pipelines differ with and
/// without it: a map takes its terrain, or drops it once it has fallen
/// flat, only once the pipelines it then draws with are made, so that no
/// frame between draws without the ground however long they take.
/// Turning terrain on waits for the terrain's own pipelines, its drape's
/// mips' and its DEM's preparation's (once, the terrain took the map's
/// layers into its drape and then drew nothing while the mips' compiled:
/// frames of labels alone), and rises once it draws, not while it waits;
/// turning it off waits for the flat fills'. Each turn starts from a
/// fresh map, which has made none of the other kind's. In a child
/// process, whose device compiles those slowly (set from the
/// environment).
#[test]
fn turning_terrain_on_or_off_waits_for_the_pipelines_it_draws_with() {
    let test = "turning_terrain_on_or_off_waits_for_the_pipelines_it_draws_with";
    if std::env::var_os(SLOW_CHILD).is_none() {
        // The terrain's and the fills', then the drape's mips' and the
        // DEM's preparation's alone, which the terrain's own then don't
        // hide.
        for slow in ["terrain,fill", "downsample,prepare pad"] {
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", test, "--nocapture"])
                .env(SLOW_CHILD, "1")
                .env("ORB_FAULT_INJECTION", format!("slow:{slow}:300"))
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&child.stderr);
            assert!(child.status.success(), "slow {slow}: {stderr}");
        }
        return;
    }
    let (with, without, terrain, _dem) = terrain_turns();
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let turn = |(start, terrain): (&str, &str), name: &str| {
        let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
        let (mut map, clock) = idle_at(&runtime, &mut target, start);
        let first = target.read_pixels().unwrap();
        set_terrain(&mut map, terrain);
        let frames = frames_until_idle(&mut map, &mut target, clock(), name);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(bare(frame), 0, "{name}: frame {} shows no tile", i + 1);
        }
        assert_ne!(&first, frames.last().unwrap(), "{name} changed nothing");
    };
    turn((&without, terrain), "terrain-on");
    turn((&with, "null"), "terrain-off");
}

/// A map loaded with terrain, turned flat before the flat view's tiles
/// have loaded (its cover wants two tiles the terrain's didn't), draws at
/// rest what a fresh flat map does: once, its antialiased edges ended up
/// to 28 levels off a fresh map's.
#[test]
fn terrain_turned_off_before_the_flat_view_loads_ends_as_a_fresh_map() {
    let (hills, _dem) = hills();
    let with = tokyo(
        r#""terrain": {"source": "hills", "exaggeration": 1.5},"#,
        &hills,
    );
    let without = tokyo("", &hills);
    let off = r#"[["setTerrain", null]]"#;
    edited_ends_as_fresh("terrain-off-loading", (&with, off, &without));
}

/// A globe map turned to Web Mercator at a zoom past the globe's draws at
/// rest what a fresh Mercator map does: its nodes stop drawing with the
/// pipelines' full variants, which stood in for the plane ones while
/// those compiled and round otherwise, once they have.
#[test]
fn a_globe_turned_to_mercator_ends_as_a_fresh_map() {
    let globe = tokyo(r#""projection": {"type": "globe"},"#, "");
    let to = r#"[["setProjection", {"type": "mercator"}]]"#;
    edited_ends_as_fresh("globe-to-mercator", (&globe, to, &tokyo("", "")));
}

/// A map loaded with `style` and edited by the transaction `edit` draws
/// at rest, at a pitched Tokyo camera, what a fresh map of `edited` does.
fn edited_ends_as_fresh(name: &str, (style, edit, edited): (&str, &str, &str)) {
    let at = (STATION, 14.5, 30.0, 55.0);
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, WIDTH, HEIGHT, 1.0);
    let mut map = Map::new(&runtime, options());
    map.set_constraints(steep());
    map.set_style_json(style).unwrap();
    map.jump_to(&camera(at));
    common::settle(&mut map, &mut target);
    map.edit(&orb_api::Transaction::from_commands(edit).unwrap())
        .unwrap();
    common::settle(&mut map, &mut target);
    let rest = map.camera();
    let pixels = target.read_pixels().unwrap();
    drop(map);
    let (fresh, fresh_rest) = fresh_at(&runtime, &mut target, edited, at);
    assert_eq!(
        rest, fresh_rest,
        "{name}: the camera differs from a fresh map's"
    );
    assert_same(name, (pixels, fresh));
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

/// Ridges against the sky over the Alps: a turn and tilt to a pitch of
/// 75° draws the silhouettes a fresh map at rest draws.
#[test]
fn an_alpine_gesture_draws_what_a_fresh_map_does() {
    let start = (Geodetic::new(7.95, 46.0), 11.2, 0.0, 30.0);
    let end = (Geodetic::new(7.85, 45.95), 12.4, 60.0, 75.0);
    let (alps, _dem) = alps();
    assert_same("alps", both_ways(&alps, start, end, 30));
}

/// A flight over the ridges aims at the ground its end stands on as the
/// DEM tiles under it land mid-flight, and ends where a fresh map at its
/// end stands.
#[test]
fn an_alpine_flight_draws_what_a_fresh_map_does() {
    let start = (Geodetic::new(7.6, 45.8), 10.0, 0.0, 0.0);
    let end = (Geodetic::new(7.85, 45.95), 12.4, 60.0, 70.0);
    let (alps, _dem) = alps();
    assert_same("alps-flight", flown(&alps, start, end));
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

/// A moving view's drape pages hold what a fresh map's do once it rests.
/// Its cover takes more tiles and fewer as it turns and tilts, and its
/// slots change sides, and the pages take textures as slots need them and
/// keep those no slot holds until the map settles, copying none.
/// Here the view over the Alps at pitch 80 looking north holds larger
/// pages than looking north-east, and a slow turn from one to the other
/// ends in the pages a fresh map there holds.
#[test]
fn a_turned_view_holds_the_drape_pages_a_fresh_one_does() {
    let (alps, _dem) = alps();
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 1440, 900, 1.0);
    let at = |bearing, pitch| (Geodetic::new(7.9, 46.0), 12.0, bearing, pitch);
    let mut map = Map::new(&runtime, options());
    let mut clock = Clock::new();
    map.set_constraints(tilted());
    map.set_style_json(&alps).unwrap();
    map.jump_to(&camera(at(45.0, 80.0)));
    let rest = clock.settle(&mut map, &mut target).memory.drape_atlas;
    map.jump_to(&camera(at(0.0, 80.0)));
    let grown = clock.settle(&mut map, &mut target).memory.drape_atlas;
    assert!(
        grown > rest,
        "looking north holds no larger pages: the test shows nothing"
    );
    for step in 0..90 {
        map.jump_to(&camera(at(45.0 + f64::from(step) * 0.05, 80.0)));
        clock.frame(&mut map, &mut target, &[]);
    }
    let turned = clock.settle(&mut map, &mut target).memory.drape_atlas;
    let mut fresh = Map::new(&runtime, options());
    fresh.set_constraints(tilted());
    fresh.set_style_json(&alps).unwrap();
    fresh.jump_to(&camera(at(45.0 + 89.0 * 0.05, 80.0)));
    let fresh = fresh
        .render_until_idle(&mut target, Duration::from_secs(60))
        .expect("the map idles")
        .memory
        .drape_atlas;
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
            common::write_png(&path, size[0], size[1], pixels);
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
        let differ = (pixels.chunks(4).zip(image.pixels.chunks(4)))
            .filter(|(a, b)| a != b)
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
    assert_eq!(
        jumped, rest,
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
            // The recorded bundles replayed.
            assert!(!report.frame.bundle_recorded, "step {step}");
        }
    }
}

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
    let mut clock = Clock::new();
    map.set_style_json(&tokyo("", "")).unwrap();
    let start = (Geodetic::new(139.72, 35.67), 12.6, 0.0, 0.0);
    let end = (STATION, 15.4, 35.0, 50.0);
    map.jump_to(&camera(start));
    clock.settle(&mut map, &mut target);
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
        clock.frame(&mut map, &mut target, &[]);
    }
    clock.settle(&mut map, &mut target);
    let shown = |map: &Map| {
        let mut names: Vec<String> = map
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
    // Each step's tiles are in before the next, so the path is the same
    // however fast this machine loads them.
    for step in 0..30 {
        let s = f64::from(step);
        map.jump_to(&camera((STATION, 13.0 + s * 0.03, 30.0 + s, 60.0)));
        let started = Instant::now();
        while map
            .render(&mut target, 0.0)
            .unwrap()
            .is_none_or(|r| r.tiles_loading > 0)
        {
            assert!(started.elapsed() < Duration::from_secs(60), "step {step}");
            map.wait_woken(Duration::from_millis(4));
        }
    }
    map.jump_to(&camera(at));
    common::settle(&mut map, &mut target);
    let off = target.read_pixels().unwrap();
    assert_eq!(warnings(&mut map), Vec::<String>::new());
    let (fresh, _) = fresh_at(&runtime, &mut target, &flat, at);
    assert_same("terrain-off", (off, fresh));
}
