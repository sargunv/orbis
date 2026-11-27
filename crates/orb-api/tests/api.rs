//! The binding scenario through the Rust API: a style, a
//! frame, a query, an edit, a style image, camera moves, a gesture and
//! idle, into a texture target.

mod common;

use std::time::Duration;

use orb_api::{
    Button, CameraEvent, CameraTarget, Constraints, EaseOptions, Error, Event, FeatureTarget,
    Input, Map, MapEvent, MapOptions, Modifiers, Op, PointerKind, Runtime, RuntimeOptions,
    StyleImage, Target, Transaction, Value, json,
};

use common::{answer, now};

/// A red background and a blue box around the origin, with feature ids.
const STYLE: &str = r##"{
  "version": 8,
  "center": [0, 0],
  "zoom": 0,
  "sources": {
    "box": {
      "type": "geojson",
      "data": {
        "type": "Feature",
        "id": 7,
        "properties": {},
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]]
        }
      }
    }
  },
  "layers": [
    {"id": "background", "type": "background", "paint": {"background-color": "#ff0000"}},
    {"id": "box", "type": "fill", "source": "box", "paint": {"fill-color": "#0000ff"}}
  ]
}"##;

const SIZE: u32 = 64;

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i..i + 4].try_into().unwrap()
}

fn commands(json: &str) -> Transaction {
    Transaction::from_commands(json).expect("commands")
}

/// Draws until the map is idle, and returns the events.
fn settle(map: &mut Map, target: &mut Target) -> Vec<Event> {
    common::settle(map, target);
    std::iter::from_fn(|| map.poll_event()).collect()
}

#[test]
fn style_frame_query_edit_gesture_idle() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(STYLE).expect("style");

    let events = settle(&mut map, &mut target);
    let maps: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Event::Map(e) => Some(e),
            _ => None,
        })
        .collect();
    assert!(matches!(maps[0], MapEvent::StyleLoad), "{maps:?}");
    assert!(maps.contains(&&MapEvent::Load), "{maps:?}");
    assert_eq!(maps.last(), Some(&&MapEvent::Idle), "{maps:?}");
    assert!((map.zoom() - 0.0).abs() < 1e-9, "the style's camera");
    assert!(!map.needs_frame());

    let pixels = target.read_pixels().unwrap();
    assert_eq!(pixel(&pixels, SIZE / 2, SIZE / 2), [0, 0, 255, 255]);
    assert_eq!(pixel(&pixels, 1, 1), [255, 0, 0, 255]);

    let hit = map
        .hit_test(glam::DVec2::splat(f64::from(SIZE) / 2.0))
        .expect("the box");
    assert_eq!((hit.layer.as_str(), hit.id.as_deref()), ("box", Some("7")));
    assert!(map.hit_test(glam::DVec2::new(1.0, 1.0)).is_none());

    // Rendered features arrive as an event, and wake the map.
    let center = glam::DVec2::splat(f64::from(SIZE) / 2.0);
    let id = map.query_rendered(center, center);
    common::wake(&map);
    let features = match map.poll_event() {
        Some(Event::Query { id: done, result }) if done == id => result.expect("query"),
        other => panic!("woken for {other:?}"),
    };
    assert!(features.contains(r#""id":"box""#), "{features}");
    assert!(
        features.starts_with('[') && features.contains(r#""id":7"#),
        "{features}"
    );
    settle(&mut map, &mut target);

    // Edits built op by op apply as one: one generation.
    let before = map.generation();
    let paint = |name: &str, value: &str| Op::SetPaint {
        layer: "box".into(),
        name: name.into(),
        value: Some(json::parse(value).unwrap()),
    };
    let mut tx = Transaction::new();
    tx.push(paint("fill-color", "\"#00ff00\""))
        .push(paint("fill-opacity", "1"));
    let generation = map.edit(&tx).expect("edit");
    assert_eq!(generation, before + 1);
    assert!(map.needs_frame());
    let report = map
        .render_until_idle(&mut target, common::DEADLINE)
        .unwrap();
    assert!(report.generation >= generation);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(pixel(&pixels, SIZE / 2, SIZE / 2), [0, 255, 0, 255]);

    // A pattern the style lacks is reported; once added, it draws, and
    // new pixels replace it in place.
    map.edit(&commands(
        r#"[["setPaintProperty", "box", "fill-pattern", "dot"]]"#,
    ))
    .expect("edit");
    let events = settle(&mut map, &mut target);
    assert!(
        events.contains(&Event::Map(MapEvent::StyleImageMissing {
            name: "dot".into()
        })),
        "{events:?}"
    );
    assert!(!map.has_image("dot"));
    let solid = |rgba: [u8; 4]| rgba.repeat(8 * 8);
    let image = StyleImage::with_options(8, 8, solid([255, 255, 0, 255]), r#"{"pixelRatio": 1}"#)
        .expect("image");
    map.add_image("dot", image.clone()).expect("add");
    assert!(map.has_image("dot"));
    assert!(matches!(
        map.add_image("dot", image),
        Err(Error::InvalidArgument(_))
    ));
    settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(pixel(&pixels, SIZE / 2, SIZE / 2), [255, 255, 0, 255]);
    map.update_image("dot", 8, 8, solid([255, 0, 255, 255]))
        .expect("update");
    settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(pixel(&pixels, SIZE / 2, SIZE / 2), [255, 0, 255, 255]);
    map.remove_image("dot").expect("remove");
    assert!(!map.has_image("dot"));
    assert!(map.remove_image("dot").is_err());
    map.edit(&commands(
        r#"[["setPaintProperty", "box", "fill-pattern", null]]"#,
    ))
    .expect("edit");
    settle(&mut map, &mut target);

    // Camera moves carry a roll and an elevation; unclamped, the centre
    // keeps the elevation it is given.
    map.jump_to(&CameraTarget {
        roll: Some(30.0),
        ..CameraTarget::default()
    });
    assert_eq!(map.camera().roll, 30.0);
    map.set_center_clamped_to_ground(false);
    map.set_center_elevation(120.0);
    assert_eq!(map.camera().elevation, 120.0);
    map.ease_to(
        &CameraTarget {
            roll: Some(0.0),
            elevation: Some(0.0),
            ..CameraTarget::default()
        },
        &EaseOptions {
            duration: 50.0,
            ..EaseOptions::default()
        },
    );
    settle(&mut map, &mut target);
    assert_eq!((map.camera().roll, map.camera().elevation), (0.0, 0.0));
    map.set_center_clamped_to_ground(true);

    // A drag moves the camera (and flings it: the moves arrive at once).
    let before = map.camera().target;
    let at = |x| glam::DVec2::new(x, 32.0);
    map.input(
        Input::PointerDown {
            id: 1,
            kind: PointerKind::Mouse,
            button: Button::Primary,
            position: at(20.0),
            modifiers: Modifiers::default(),
        },
        now(),
    );
    for x in [24.0, 30.0, 40.0] {
        map.input(
            Input::PointerMove {
                id: 1,
                position: at(x),
            },
            now(),
        );
    }
    map.input(
        Input::PointerUp {
            id: 1,
            position: at(40.0),
        },
        now(),
    );
    let events = settle(&mut map, &mut target);
    assert!((map.camera().target.lon - before.lon).abs() > 1.0);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Map(MapEvent::Camera(_))))
    );

    // A click is reported, and an idle map draws nothing.
    for input in [
        Input::PointerDown {
            id: 2,
            kind: PointerKind::Mouse,
            button: Button::Primary,
            position: at(32.0),
            modifiers: Modifiers::default(),
        },
        Input::PointerUp {
            id: 2,
            position: at(32.0),
        },
    ] {
        map.input(input, now());
    }
    let events = settle(&mut map, &mut target);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Map(MapEvent::Click { .. }))),
        "{events:?}"
    );
    assert!(!map.needs_frame());
}

/// An animation's frames aren't idle: the map is idle once, at the frame
/// it arrives on, and draws nothing after.
#[test]
fn idle_comes_once_after_an_ease() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(STYLE).expect("style");
    settle(&mut map, &mut target);
    map.ease_to(
        &CameraTarget {
            zoom: Some(1.0),
            ..CameraTarget::default()
        },
        &EaseOptions {
            duration: 400.0,
            ..EaseOptions::default()
        },
    );
    // Frames on a display's 60 Hz clock, whatever the wall clock does:
    // the first starts the ease, which arrives on the frame 400 ms on.
    let (mut frames, mut idles, mut arrived) = (0, 0, None);
    let started = std::time::Instant::now();
    while idles == 0 || map.needs_frame() {
        assert!(started.elapsed() < common::DEADLINE, "the map never idled");
        if map.needs_frame() {
            let time = 1e6 + f64::from(frames) * 1000.0 / 60.0;
            let report = map.render(&mut target, time).unwrap().unwrap();
            frames += 1;
            for event in std::iter::from_fn(|| map.poll_event()) {
                match event {
                    Event::Map(MapEvent::Idle) => {
                        idles += 1;
                        assert!(report.idle(), "idle on a frame with work left");
                    }
                    Event::Map(MapEvent::Camera(CameraEvent::AnimationEnd {
                        interrupted, ..
                    })) => {
                        assert!(!interrupted);
                        arrived = Some(frames);
                    }
                    _ => {}
                }
            }
        } else {
            // Waiting on data (the new zoom's tiles): its wake asks for a frame.
            map.wait_woken(Duration::from_secs(1));
        }
    }
    assert_eq!(
        arrived,
        Some(25),
        "frame 0 at the start, frame 24 at 400 ms"
    );
    assert_eq!(idles, 1);
    assert!((map.zoom() - 1.0).abs() < 1e-9);
}

/// A texture target's viewport is its size in logical pixels, and once
/// resized it reads back at its new size.
#[test]
fn texture_targets_resize_and_read_back() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, 10, 6, 2.0);
    assert_eq!(target.viewport().width, 5.0);
    target.resize(8, 4, 1.0);
    assert_eq!(target.read_pixels().unwrap().len(), 8 * 4 * 4);
}

/// A wake callback dies with its map, even while a fetch the map started
/// is still in flight and the runtime lives on: a host may free the
/// callback's state as soon as the map is gone.
#[test]
fn wake_callbacks_die_with_their_map() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    // Accepts connections and never answers, so the style fetch waits.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/style.json", listener.local_addr().unwrap());
    let mut map = Map::new(&runtime, MapOptions::default());
    map.load_style_url(&url);
    let (connection, _) = listener.accept().unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let probe = calls.clone();
    map.set_wake(move || {
        probe.fetch_add(1, Ordering::SeqCst);
    });
    drop(map);
    assert_eq!(
        Arc::strong_count(&calls),
        1,
        "the map kept its wake callback"
    );
    // The test holds the only reference left, so the fetch's completion
    // has no callback to call, whenever it comes.
    drop(connection);
    drop(runtime);
}

/// A frame borrows the runtime's device once, for the target and the map,
/// and lets it go before the pipeline cache is saved on going idle: frames
/// to idle with a state directory keep the cache in its `pipelines/`
/// where the driver has one.
/// Window targets take this path too.
#[test]
fn frames_to_idle_save_the_pipeline_cache() {
    let dir = common::Scratch::new("api-state");
    let runtime = Runtime::new(&RuntimeOptions {
        state_dir: Some(dir.to_path_buf()),
        ..RuntimeOptions::default()
    })
    .expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(STYLE).expect("style");
    settle(&mut map, &mut target);
    settle(&mut map, &mut target);
    // Drivers with a pipeline cache (Vulkan) keep it there; Metal caches
    // pipelines itself and keeps none, and the directory may never be made.
    if let Ok(entries) = std::fs::read_dir(dir.join("pipelines")) {
        assert!(
            entries.count() > 0,
            "a pipeline cache was started and not written"
        );
    }
    drop((map, target, runtime));
}

/// A camera given at creation is the map's, at any zoom: the style's own
/// (north up and flat) doesn't replace it, and the picture shows it. Two
/// ground bands lie as far north of the target as south: pitched, the far
/// one draws thinner; turned, their edges tilt.
#[test]
fn a_camera_given_keeps_its_pitch_and_bearing() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mercator = *orb_api::TileMatrixSet::web_mercator().projection();
    let (lon, lat) = (139.76_f64, 35.68_f64);
    for zoom in [5.0, 9.0, 15.5] {
        // Eight pixels of ground at this zoom, flat.
        let degrees = 360.0 / (512.0 * 2f64.powf(zoom));
        let (d, w) = (8.0 * degrees * lat.to_radians().cos(), 400.0 * degrees);
        let band = |from: f64, to: f64| {
            format!(
                "[[[{w0}, {from}], [{w1}, {from}], [{w1}, {to}], [{w0}, {to}], [{w0}, {from}]]]",
                w0 = lon - w,
                w1 = lon + w
            )
        };
        let style = format!(
            r##"{{"version": 8, "center": [0, 0], "zoom": 0, "bearing": 0, "pitch": 0,
              "sources": {{"bands": {{"type": "geojson", "data": {{"type": "MultiPolygon",
                "coordinates": [{north}, {south}]}}}}}},
              "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#ff0000"}}}},
                {{"id": "bands", "type": "fill", "source": "bands", "paint": {{"fill-color": "#0000ff"}}}}
              ]}}"##,
            north = band(lat + d, lat + 2.0 * d),
            south = band(lat - 2.0 * d, lat - d),
        );
        for (bearing, pitch) in [(0.0, 0.0), (-30.0, 55.0)] {
            let camera = orb_api::CameraState {
                bearing,
                pitch,
                ..orb_api::CameraState::new(orb_api::Geodetic::new(lon, lat), zoom, &mercator)
            };
            let mut map = Map::new(
                &runtime,
                MapOptions {
                    camera: Some(camera),
                    ..MapOptions::default()
                },
            );
            map.set_style_json(&style).expect("style");
            let image = map.snapshot(SIZE, SIZE, 1.0).expect("snapshot");
            let c = map.camera();
            let at = format!("z{zoom} bearing {bearing} pitch {pitch}");
            assert_eq!((c.bearing, c.pitch), (bearing, pitch), "{at}");
            assert!((map.zoom() - zoom).abs() < 1e-9, "{at}");
            let blue = |x: u32, y: u32| pixel(&image.pixels, x, y)[2] > 200;
            let (north, south) = (
                (0..SIZE / 2).filter(|&y| blue(SIZE / 2, y)).count(),
                (SIZE / 2..SIZE).filter(|&y| blue(SIZE / 2, y)).count(),
            );
            // The north band's top edge, a quarter in from each side.
            let edge = |x: u32| (0..SIZE).find(|&y| blue(x, y)).expect("a band");
            let tilt = edge(SIZE / 4).abs_diff(edge(3 * SIZE / 4));
            if pitch == 0.0 {
                assert!(
                    north.abs_diff(south) <= 1,
                    "{at}: bands {north} and {south} px"
                );
                assert!(tilt <= 1, "{at}: edges {tilt} px apart");
            } else {
                assert!(north + 1 < south, "{at}: bands {north} and {south} px");
                assert!(tilt > 3, "{at}: edges {tilt} px apart");
            }
        }
    }
}

/// Paint reads the camera's pitch and bearing, beyond the spec and
/// without a preview, and follows them as the camera moves: a map moved to
/// a camera draws as one made there.
#[test]
fn paint_reads_the_camera() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let style = r##"{"version": 8, "center": [0, 0], "zoom": 2, "sources": {},
      "layers": [{"id": "background", "type": "background", "paint": {
        "background-color": ["interpolate", ["linear"], ["pitch"], 0, "#ff0000", 60, "#0000ff"],
        "background-opacity": ["step", ["bearing"], 1, 90, 0.5]}}]}"##;
    let at = |pitch: f64, bearing: f64| orb_api::CameraState {
        pitch,
        bearing,
        ..orb_api::CameraState::new(
            orb_api::Geodetic::new(0.0, 0.0),
            2.0,
            orb_api::TileMatrixSet::web_mercator().projection(),
        )
    };
    let map = || {
        let mut map = Map::new(
            &runtime,
            MapOptions {
                camera: Some(at(0.0, 0.0)),
                ..MapOptions::default()
            },
        );
        map.set_style_json(style).expect("style");
        map
    };
    let centre = |map: &mut Map| {
        let image = map.snapshot(SIZE, SIZE, 1.0).expect("snapshot");
        pixel(&image.pixels, SIZE / 2, SIZE / 2)
    };
    let mut moved = map();
    assert_eq!(centre(&mut moved), [255, 0, 0, 255], "flat");
    for (pitch, bearing, expected) in [
        (60.0, 0.0, [0, 0, 255, 255]),
        (30.0, 0.0, [128, 0, 128, 255]),
        (60.0, 120.0, [0, 0, 128, 128]),
    ] {
        moved.jump_to_state(at(pitch, bearing));
        let mut fresh = map();
        fresh.jump_to_state(at(pitch, bearing));
        for (how, map) in [("moved", &mut moved), ("fresh", &mut fresh)] {
            let got = centre(map);
            assert!(
                got.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 1),
                "{how} to pitch {pitch} bearing {bearing}: {got:?}, not {expected:?}"
            );
        }
    }
}

/// `fill-antialias` per feature (beyond the spec) outlines only the
/// features that want one, and by zoom only at the zooms that do.
#[test]
fn fill_antialias_follows_each_feature_and_the_zoom() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let draw = |antialias: &str, zoom: f64| {
        let style = format!(
            r##"{{"version": 8, "center": [0, 0], "zoom": {zoom},
              "sources": {{"b": {{"type": "geojson", "data": {{"type": "FeatureCollection", "features": [
                {{"type": "Feature", "properties": {{"aa": true}}, "geometry": {{"type": "Polygon",
                  "coordinates": [[[-10, -5], [-2, -5], [-2, 5], [-10, 5], [-10, -5]]]}}}},
                {{"type": "Feature", "properties": {{"aa": false}}, "geometry": {{"type": "Polygon",
                  "coordinates": [[[2, -5], [10, -5], [10, 5], [2, 5], [2, -5]]]}}}}]}}}}}},
              "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#000000"}}}},
                {{"id": "fill", "type": "fill", "source": "b",
                  "paint": {{"fill-color": "#ffffff", "fill-outline-color": "#ff0000",
                    "fill-antialias": {antialias}}}}}]}}"##
        );
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&style).expect("style");
        let image = map.snapshot(SIZE, SIZE, 1.0).expect("snapshot");
        // Red outline pixels left and right of the middle.
        let red = |xs: std::ops::Range<u32>| {
            (xs.flat_map(|x| (0..SIZE).map(move |y| (x, y))))
                .filter(|&(x, y)| {
                    let [r, g, ..] = pixel(&image.pixels, x, y);
                    r > 128 && g < 100
                })
                .count()
        };
        [red(0..SIZE / 2), red(SIZE / 2..SIZE)]
    };
    let [left, right] = draw("true", 2.0);
    assert!(left > 0 && right > 0, "both outlined: {left} {right}");
    assert_eq!(draw("false", 2.0), [0, 0]);
    let [left, right] = draw(r#"["get", "aa"]"#, 2.0);
    assert!(left > 0 && right == 0, "the first alone: {left} {right}");
    let step = r#"["step", ["zoom"], false, 2.5, true]"#;
    assert_eq!(draw(step, 2.0), [0, 0]);
    let [left, right] = draw(step, 3.0);
    assert!(left > 0 && right > 0, "outlined from 2.5: {left} {right}");
}

/// Each building takes its own opacity from its data: over black, a
/// building at a quarter draws a quarter as bright as one at full.
#[test]
fn extrusion_opacity_follows_the_feature() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let style = r##"{"version": 8, "center": [0, 0], "zoom": 2, "sources": {"b": {"type": "geojson",
        "data": {"type": "FeatureCollection", "features": [
          {"type": "Feature", "properties": {"o": 1}, "geometry": {"type": "Polygon",
            "coordinates": [[[-10, -5], [-1, -5], [-1, 5], [-10, 5], [-10, -5]]]}},
          {"type": "Feature", "properties": {"o": 0.25}, "geometry": {"type": "Polygon",
            "coordinates": [[[1, -5], [10, -5], [10, 5], [1, 5], [1, -5]]]}}]}}},
      "layers": [
        {"id": "background", "type": "background", "paint": {"background-color": "#000000"}},
        {"id": "b", "type": "fill-extrusion", "source": "b", "paint": {
          "fill-extrusion-color": "#ffffff", "fill-extrusion-height": 1000,
          "fill-extrusion-opacity": ["get", "o"],
          "fill-extrusion-vertical-gradient": ["==", ["get", "o"], 1]}}]}"##;
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).expect("style");
    let image = map.snapshot(SIZE, SIZE, 1.0).expect("snapshot");
    let roof = |x| pixel(&image.pixels, x, SIZE / 2);
    let (full, quarter) = (roof(SIZE / 2 - 12), roof(SIZE / 2 + 12));
    assert!(full[0] > 100, "{full:?}");
    for c in 0..3 {
        let expected = f32::from(full[c]) * 0.25;
        assert!(
            (f32::from(quarter[c]) - expected).abs() <= 2.0,
            "{quarter:?} is not a quarter of {full:?}"
        );
    }
}

/// `fill-extrusion-layer-opacity` fades the layer as one surface, times
/// `fill-extrusion-opacity`: the layer at 0.75 draws as buildings at 0.75
/// do, as do 0.9375 and 0.8 together; with each building's own opacity,
/// each draws at its own times the layer's; at zero nothing draws.
#[test]
fn extrusion_layer_opacity_multiplies_the_buildings() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let snapshot = |paint: &str| {
        let style = format!(
            r##"{{"version": 8, "center": [0, 0], "zoom": 2, "sources": {{"b": {{"type": "geojson",
            "data": {{"type": "FeatureCollection", "features": [
              {{"type": "Feature", "properties": {{"o": 1}}, "geometry": {{"type": "Polygon",
                "coordinates": [[[-10, -5], [-1, -5], [-1, 5], [-10, 5], [-10, -5]]]}}}},
              {{"type": "Feature", "properties": {{"o": 0.25}}, "geometry": {{"type": "Polygon",
                "coordinates": [[[1, -5], [10, -5], [10, 5], [1, 5], [1, -5]]]}}}}]}}}}}},
          "layers": [
            {{"id": "background", "type": "background", "paint": {{"background-color": "#000000"}}}},
            {{"id": "b", "type": "fill-extrusion", "source": "b", "paint": {{
              "fill-extrusion-color": "#ffffff", "fill-extrusion-height": 1000, {paint}}}}}]}}"##
        );
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&style).expect("style");
        map.snapshot(SIZE, SIZE, 1.0).expect("snapshot").pixels
    };
    let close = |a: &[u8], b: &[u8]| a.iter().zip(b).all(|(a, b)| a.abs_diff(*b) <= 1);
    let shared = snapshot(r#""fill-extrusion-opacity": 0.75"#);
    assert!(shared.iter().any(|&c| c > 100), "the buildings draw");
    let layer = snapshot(r#""fill-extrusion-layer-opacity": 0.75"#);
    assert!(close(&shared, &layer), "the layer at 0.75");
    let both = snapshot(r#""fill-extrusion-layer-opacity": 0.9375, "fill-extrusion-opacity": 0.8"#);
    assert!(close(&shared, &both), "0.9375 times 0.8");

    let roofs = |pixels: &[u8]| {
        let roof = |x| pixel(pixels, x, SIZE / 2);
        (roof(SIZE / 2 - 12), roof(SIZE / 2 + 12))
    };
    let per_building = r#""fill-extrusion-opacity": ["get", "o"]"#;
    let (full, _) = roofs(&snapshot(per_building));
    let (half, eighth) = roofs(&snapshot(&format!(
        r#"{per_building}, "fill-extrusion-layer-opacity": 0.5"#
    )));
    assert!(full[0] > 100, "{full:?}");
    for c in 0..3 {
        let full = f32::from(full[c]);
        assert!(
            (f32::from(half[c]) - full * 0.5).abs() <= 2.0,
            "{half:?} of {full}"
        );
        assert!(
            (f32::from(eighth[c]) - full * 0.125).abs() <= 2.0,
            "{eighth:?} of {full}"
        );
    }
    let hidden = snapshot(&format!(
        r#"{per_building}, "fill-extrusion-layer-opacity": 0"#
    ));
    assert!(
        hidden.chunks(4).all(|p| p[..3] == [0, 0, 0]),
        "nothing draws at zero"
    );
}

/// Each feature takes its own translate from its data, beyond the spec:
/// fills and circles draw and pick where their own translate puts them,
/// and a circle takes its own pitch alignment: tilted, one on the map is
/// squashed and one facing the viewport stays round.
#[test]
fn translate_and_circle_pitch_follow_the_feature() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let square = |lon: f64, lat: f64, k: &str| {
        format!(
            r#"{{"type": "Feature", "id": "{k}", "properties": {{"k": "{k}"}}, "geometry": {{"type": "Polygon",
              "coordinates": [[[{a}, {b}], [{c}, {b}], [{c}, {d}], [{a}, {d}], [{a}, {b}]]]}}}}"#,
            a = lon - 5.0,
            c = lon + 5.0,
            b = lat - 5.0,
            d = lat + 5.0
        )
    };
    let point = |lon: f64, lat: f64, k: &str| {
        format!(
            r#"{{"type": "Feature", "id": "{k}", "properties": {{"k": "{k}"}},
              "geometry": {{"type": "Point", "coordinates": [{lon}, {lat}]}}}}"#
        )
    };
    let shifted = r#"["match", ["get", "k"], "b", ["literal", [12, 0]], ["literal", [0, 0]]]"#;
    let style = format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 0, "sources": {{
          "f": {{"type": "geojson", "data": {{"type": "FeatureCollection", "features": [{}, {}]}}}},
          "c": {{"type": "geojson", "data": {{"type": "FeatureCollection", "features": [{}, {}]}}}}}},
          "layers": [
            {{"id": "background", "type": "background", "paint": {{"background-color": "#000000"}}}},
            {{"id": "f", "type": "fill", "source": "f", "paint": {{"fill-color": "#ff0000",
              "fill-translate": {shifted}}}}},
            {{"id": "c", "type": "circle", "source": "c", "paint": {{"circle-color": "#0000ff",
              "circle-radius": 4, "circle-translate": {shifted}}}}}]}}"##,
        square(-12.0, 10.0, "a"),
        square(3.0, 10.0, "b"),
        point(-12.0, -12.0, "a"),
        point(3.0, -12.0, "b"),
    );
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    // The tilt below reaches past the default 60 degrees.
    map.set_constraints(Constraints {
        max_pitch: 85.0,
        ..Constraints::default()
    });
    map.set_style_json(&style).expect("style");
    settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    let at = |x: f64, y: f64| pixel(&pixels, x.round() as u32, y.round() as u32);
    // Zoom 0: 512 / 360 pixels a degree about the centre.
    let x = |lon: f64| 32.0 + lon * 512.0 / 360.0;
    let (fill_y, circle_y) = (32.0 - 14.3, 32.0 + 17.2);
    assert_eq!(at(x(-12.0), fill_y), [255, 0, 0, 255], "a in place");
    assert_eq!(at(x(3.0) + 12.0, fill_y), [255, 0, 0, 255], "b moved");
    assert_eq!(at(x(3.0) - 5.0, fill_y), [0, 0, 0, 255], "b left");
    assert_eq!(at(x(-12.0), circle_y), [0, 0, 255, 255], "circle a");
    assert_eq!(at(x(3.0) + 12.0, circle_y), [0, 0, 255, 255], "circle b");
    assert_eq!(at(x(3.0), circle_y), [0, 0, 0, 255], "circle b left");
    let hit = |px: f64, py: f64| {
        map.hit_test(glam::DVec2::new(px, py))
            .map(|h| (h.layer, h.id.unwrap_or_default()))
    };
    let b = |layer: &str| Some((layer.to_string(), "b".to_string()));
    assert_eq!(hit(x(3.0) + 12.0, fill_y), b("f"));
    assert_eq!(hit(x(3.0) - 5.0, fill_y), None);
    assert_eq!(hit(x(3.0) + 12.0, circle_y), b("c"));
    assert_eq!(hit(x(3.0), circle_y), None);

    // Circle pitch alignment per feature, under a tilt.
    let style = format!(
        r##"{{"version": 8, "sources": {{
          "c": {{"type": "geojson", "data": {{"type": "FeatureCollection", "features": [{}, {}]}}}}}},
          "layers": [
            {{"id": "background", "type": "background", "paint": {{"background-color": "#000000"}}}},
            {{"id": "c", "type": "circle", "source": "c", "paint": {{"circle-color": "#ffffff",
              "circle-radius": 8, "circle-pitch-scale": "viewport",
              "circle-pitch-alignment": ["match", ["get", "k"], "a", "map", "viewport"]}}}}]}}"##,
        point(-3.0, 0.0, "a"),
        point(3.0, 0.0, "b"),
    );
    map.set_style_json(&style).expect("style");
    map.jump_to(&CameraTarget {
        zoom: Some(2.0),
        pitch: Some(70.0),
        ..CameraTarget::default()
    });
    settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    // Each circle's lit width and height in pixels.
    let extent = |left: bool| {
        let lit: Vec<(u32, u32)> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, _)| (x < SIZE / 2) == left)
            .filter(|&(x, y)| pixel(&pixels, x, y)[0] > 128)
            .collect();
        assert!(!lit.is_empty(), "nothing lit, left {left}");
        let span = |v: Vec<u32>| v.iter().max().unwrap() - v.iter().min().unwrap() + 1;
        (
            span(lit.iter().map(|p| p.0).collect()),
            span(lit.iter().map(|p| p.1).collect()),
        )
    };
    let (on_map, facing) = (extent(true), extent(false));
    assert!(
        on_map.1 * 2 < on_map.0,
        "on the map, squashed: {on_map:?} {facing:?}"
    );
    assert!(
        facing.1.abs_diff(facing.0) <= 1,
        "facing, round: {facing:?}"
    );
}

/// The lit runs along row `y` of a `width`-wide target: each one's first
/// column and length.
fn lit_runs(pixels: &[u8], width: u32, y: u32) -> Vec<(u32, u32)> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for x in 0..width {
        if pixel_of(pixels, width, x, y)[0] <= 128 {
            continue;
        }
        match runs.last_mut() {
            Some((start, len)) if *start + *len == x => *len += 1,
            _ => runs.push((x, 1)),
        }
    }
    runs
}

/// A radius from the feature's data or state times a factor of the pitch
/// (beyond the spec) draws, queries and picks at the camera's pitch: rows
/// and the state table hold the feature's part, the node the pitch's.
#[test]
fn pitch_scales_feature_data_and_state() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(
        r##"{"version": 8, "center": [0, 0], "zoom": 2,
            "sources": {"p": {"type": "geojson", "data": {"type": "Feature", "id": 3,
                "properties": {"r": 6}, "geometry": {"type": "Point", "coordinates": [0, 0]}}}},
            "layers": [
                {"id": "background", "type": "background", "paint": {"background-color": "#000000"}},
                {"id": "c", "type": "circle", "source": "p",
                "paint": {
                    "circle-radius": ["*", ["coalesce", ["feature-state", "r"], ["get", "r"]],
                        ["interpolate", ["linear"], ["pitch"], 0, 1, 60, 2]],
                    "circle-pitch-scale": "viewport",
                    "circle-color": "#ffffff"}}]}"##,
    )
    .expect("style");
    let width = |map: &mut Map, target: &mut Target| {
        settle(map, target);
        let pixels = target.read_pixels().unwrap();
        let runs = lit_runs(&pixels, SIZE, SIZE / 2);
        runs.iter().map(|r| r.1).max().unwrap_or(0)
    };
    let center = glam::DVec2::splat(f64::from(SIZE) / 2.0);
    let radius = |map: &mut Map, target: &mut Target| {
        let query = map.query_rendered(center, center);
        let doc = json::parse(&answer(map, target, query).unwrap()).unwrap();
        let layer = doc.as_array().unwrap()[0].get("layer").unwrap();
        let paint = layer.get("paint").unwrap();
        paint.get("circle-radius").unwrap().as_f64().unwrap()
    };
    let flat = width(&mut map, &mut target);
    assert!(flat.abs_diff(12) <= 2, "{flat} px flat");
    assert_eq!(radius(&mut map, &mut target), 6.0);
    map.jump_to(&CameraTarget {
        pitch: Some(60.0),
        ..CameraTarget::default()
    });
    let tilted = width(&mut map, &mut target);
    assert!(tilted.abs_diff(24) <= 2, "{tilted} px at pitch 60");
    assert_eq!(radius(&mut map, &mut target), 12.0);
    let edge = center + glam::DVec2::new(10.0, 0.0);
    assert!(map.hit_test(edge).is_some(), "picks at its tilted radius");
    map.edit(&commands(
        r#"[["setFeatureState", {"source": "p", "id": 3}, {"r": 10}]]"#,
    ))
    .expect("state");
    let state = width(&mut map, &mut target);
    assert!(state.abs_diff(40) <= 2, "{state} px from its state");
    assert_eq!(radius(&mut map, &mut target), 20.0);
}

fn feature(source: &str, layer: Option<&str>, id: Option<&str>) -> FeatureTarget {
    FeatureTarget {
        source: source.into(),
        source_layer: layer.map(Into::into),
        id: id.map(Into::into),
    }
}

/// A feature's state as JSON, keys sorted.
fn state_of(map: &Map, target: &FeatureTarget) -> String {
    let mut state = map.feature_state(target).to_vec();
    state.sort_by(|a, b| a.0.cmp(&b.0));
    Value::Object(state.into()).to_json_string()
}

fn state(json: &str) -> Vec<(String, Value<'static>)> {
    match Value::from_json(&json::parse(json).unwrap()) {
        Value::Object(members) => members.into_owned(),
        other => panic!("{other:?} is no object"),
    }
}

/// The map keeps feature state for any source, before a style too: set
/// merges, reports whether it changed anything, and removal by key, by
/// feature and by source layer takes exactly what it names. The first
/// style starts with none.
#[test]
fn feature_state_is_kept_whatever_the_style() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut map = Map::new(&runtime, MapOptions::default());
    let one = feature("tiles", Some("roads"), Some("1"));
    assert_eq!(state_of(&map, &one), "{}");
    assert!(map.set_feature_state(&one, &state(r#"{"hover": true}"#)));
    assert!(map.set_feature_state(&one, &state(r#"{"n": 3}"#)));
    assert!(
        !map.set_feature_state(&one, &state(r#"{"n": 3}"#)),
        "unchanged"
    );
    assert_eq!(state_of(&map, &one), r#"{"hover":true,"n":3}"#);
    assert!(map.remove_feature_state(&one, Some("hover")));
    assert!(
        !map.remove_feature_state(&one, Some("hover")),
        "already gone"
    );
    assert_eq!(state_of(&map, &one), r#"{"n":3}"#);

    let two = feature("tiles", Some("roads"), Some("2"));
    let water = feature("tiles", Some("water"), Some("1"));
    let bare = feature("tiles", None, Some("1"));
    for target in [&two, &water, &bare] {
        map.set_feature_state(target, &state(r#"{"n": 1}"#));
    }
    assert!(map.remove_feature_state(&feature("tiles", Some("roads"), None), None));
    assert_eq!(state_of(&map, &one), "{}");
    assert_eq!(state_of(&map, &two), "{}");
    assert_eq!(state_of(&map, &water), r#"{"n":1}"#);
    assert_eq!(state_of(&map, &bare), r#"{"n":1}"#);
    // No source layer selects the features without one.
    assert!(map.remove_feature_state(&feature("tiles", None, None), None));
    assert_eq!(state_of(&map, &bare), "{}");
    assert_eq!(state_of(&map, &water), r#"{"n":1}"#);
    // Without an id, a key is ignored: the whole layer's state goes.
    map.set_feature_state(&two, &state(r#"{"n": 1, "m": 2}"#));
    assert!(map.remove_feature_state(&feature("tiles", Some("roads"), None), Some("n")));
    assert_eq!(state_of(&map, &two), "{}");

    map.set_style_json(STYLE).expect("style");
    assert_eq!(state_of(&map, &water), "{}", "a style load clears it");
}

/// State set for a source the style lacks draws once a transaction adds
/// it, a transaction's state reads back through the same store, and a
/// source removed takes its state with it.
#[test]
fn feature_state_follows_the_styles_sources() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(
        r##"{"version": 8, "center": [0, 0], "zoom": 0, "sources": {},
          "layers": [{"id": "background", "type": "background",
            "paint": {"background-color": "#ff0000"}}]}"##,
    )
    .expect("style");
    let seven = feature("box", None, Some("7"));
    map.set_feature_state(&seven, &state(r#"{"on": true}"#));
    let add = r##"[["addSource", "box", {"type": "geojson", "data": {"type": "Feature",
          "id": 7, "properties": {}, "geometry": {"type": "Polygon",
          "coordinates": [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]]}}}],
        ["addLayer", {"id": "box", "type": "fill", "source": "box", "paint": {"fill-color":
          ["case", ["boolean", ["feature-state", "on"], false], "#00ff00", "#0000ff"]}}]]"##;
    let centre = |map: &mut Map, target: &mut Target| {
        settle(map, target);
        pixel(&target.read_pixels().unwrap(), SIZE / 2, SIZE / 2)
    };
    map.edit(&commands(add)).expect("add");
    assert_eq!(
        centre(&mut map, &mut target),
        [0, 255, 0, 255],
        "kept state draws"
    );

    map.edit(&commands(
        r#"[["setFeatureState", {"source": "box", "id": 7}, {"on": false}]]"#,
    ))
    .expect("state");
    assert_eq!(state_of(&map, &seven), r#"{"on":false}"#);
    assert_eq!(centre(&mut map, &mut target), [0, 0, 255, 255]);
    map.set_feature_state(&seven, &state(r#"{"on": true}"#));
    assert_eq!(centre(&mut map, &mut target), [0, 255, 0, 255]);

    map.edit(&commands(
        r#"[["removeLayer", "box"], ["removeSource", "box"]]"#,
    ))
    .expect("remove");
    assert_eq!(state_of(&map, &seven), "{}");
    map.edit(&commands(add)).expect("add again");
    assert_eq!(
        centre(&mut map, &mut target),
        [0, 0, 255, 255],
        "starts with none"
    );

    // A style URL's arrival clears it, though the diff keeps the source.
    map.set_feature_state(&seven, &state(r#"{"on": true}"#));
    assert_eq!(centre(&mut map, &mut target), [0, 255, 0, 255]);
    let dir = common::Scratch::new("api-feature-state");
    let path = dir.join("style.json");
    let style = map.style().expect("style").to_json();
    std::fs::write(&path, style).unwrap();
    map.load_style_url(&format!("file://{}", path.display()));
    let events = settle(&mut map, &mut target);
    assert!(
        (events.iter()).any(|e| matches!(e, Event::Map(MapEvent::StyleLoad))),
        "the style arrived"
    );
    assert_eq!(state_of(&map, &seven), "{}");
    assert_eq!(centre(&mut map, &mut target), [0, 0, 255, 255]);
}

/// A dashed line whose width the camera scales (beyond the spec) sizes
/// its dashes by the width it draws, not by its feature's part: 8 px
/// halved by the pitch's factor dashes as 4 px from the feature's pixels.
#[test]
fn scaled_line_widths_size_dashes_as_drawn() {
    const WIDTH: u32 = 256;
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, WIDTH, SIZE, 1.0);
    let draw = |width: &str, paint: &str, target: &mut Target| {
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&format!(
            r##"{{"version": 8, "center": [0, 0], "zoom": 2,
              "sources": {{"l": {{"type": "geojson", "data": {{"type": "Feature",
                "properties": {{"double": 8, "px": 4}},
                "geometry": {{"type": "LineString", "coordinates": [[-60, 0], [60, 0]]}}}}}}}},
              "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#000000"}}}},
                {{"id": "l", "type": "line", "source": "l",
                  "paint": {{"line-color": "#ffffff", "line-width": {width}, {paint}}}}}]}}"##
        ))
        .expect("style");
        settle(&mut map, target);
        lit_runs(&target.read_pixels().unwrap(), WIDTH, SIZE / 2)
    };
    let scaled =
        r#"["*", ["get", "double"], ["interpolate", ["linear"], ["pitch"], 0, 0.5, 60, 1]]"#;
    let dashes = r#""line-dasharray": [2, 2]"#;
    let want = draw(r#"["get", "px"]"#, dashes, &mut target);
    assert!(want.len() > 4, "{want:?}: dashes");
    assert_eq!(draw(scaled, dashes, &mut target), want, "dashes");
}

/// Buildings larger than an overzoom slice draw whole at z15, z16 and z19
/// below a source's z14 tiles, looking down and tilted, where the view's
/// tiles span several zooms (#143): a long slab, a building whose
/// vertices crowd one end (its centroid's slice would cut it), and a
/// block merged into one multipolygon, no slice holding every part.
#[test]
fn large_buildings_draw_whole_across_overzoom_slices() {
    const VIEW: u32 = 256;
    let rect =
        |x0: f64, y0: f64, x1: f64, y1: f64| vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];
    let slab = rect(0.0040, 0.0100, 0.0170, 0.0108);
    let mut crowded = rect(0.0035, 0.0060, 0.0125, 0.0066);
    crowded.splice(
        2..2,
        (1..20).map(|i| [0.0125, 0.0060 + 0.00003 * f64::from(i)]),
    );
    let block = vec![
        rect(0.0105, 0.0140, 0.0118, 0.0147),
        rect(0.0170, 0.0140, 0.0180, 0.0149),
        rect(0.0020, 0.0140, 0.0027, 0.0147),
        rect(0.0010, 0.0160, 0.0210, 0.0166),
    ];
    let polygon = |ring: &Vec<[f64; 2]>| format!("[{ring:?}]");
    let feature = |kind: &str, coordinates: String| {
        format!(
            r#"{{"type": "Feature", "properties": {{}}, "geometry": {{"type": "{kind}", "coordinates": {coordinates}}}}}"#
        )
    };
    let parts: Vec<String> = block.iter().map(polygon).collect();
    let features = [
        feature("Polygon", polygon(&slab)),
        feature("Polygon", polygon(&crowded)),
        feature("MultiPolygon", format!("[{}]", parts.join(", "))),
    ]
    .join(", ");
    let rings: Vec<&Vec<[f64; 2]>> = [&slab, &crowded].into_iter().chain(&block).collect();
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    for (lon, lat, zoom, bearing, pitch) in [
        (0.011, 0.012, 15.0, 0.0, 0.0),
        (0.011, 0.012, 15.5, 20.0, 60.0),
        (0.008, 0.011, 16.0, 0.0, 0.0),
        (0.008, 0.011, 16.3, -30.0, 60.0),
        (0.0082, 0.0063, 16.0, 0.0, 0.0),
        (0.0082, 0.0063, 16.2, 10.0, 50.0),
        (0.0110, 0.0104, 19.0, 0.0, 0.0),
        (0.0110, 0.0098, 19.0, 18.3, 68.0),
        (0.0082, 0.0063, 19.0, 0.0, 0.0),
        (0.0110, 0.0160, 19.0, 0.0, 60.0),
    ] {
        let style = format!(
            r##"{{"version": 8, "center": [{lon}, {lat}], "zoom": {zoom}, "bearing": {bearing}, "pitch": {pitch},
              "sources": {{"b": {{"type": "geojson", "maxzoom": 14,
                "data": {{"type": "FeatureCollection", "features": [{features}]}}}}}},
              "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#000000"}}}},
                {{"id": "b", "type": "fill-extrusion", "source": "b", "paint": {{
                  "fill-extrusion-color": "#ffffff", "fill-extrusion-height": 1}}}}]}}"##
        );
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&style).expect("style");
        let image = map.snapshot(VIEW, VIEW, 1.0).expect("snapshot");
        let mut seen = 0;
        for ring in &rings {
            let (lo, hi) = ring
                .iter()
                .fold(([f64::MAX; 2], [f64::MIN; 2]), |(lo, hi), p| {
                    (
                        [lo[0].min(p[0]), lo[1].min(p[1])],
                        [hi[0].max(p[0]), hi[1].max(p[1])],
                    )
                });
            for i in 1..120 {
                for j in 1..40 {
                    let x = lo[0] + (hi[0] - lo[0]) * (0.05 + 0.9 * f64::from(i) / 120.0);
                    let y = lo[1] + (hi[1] - lo[1]) * (0.1 + 0.8 * f64::from(j) / 40.0);
                    let Some(p) = map.project(orb_api::Geodetic::new(x, y)) else {
                        continue;
                    };
                    if !(4.0..f64::from(VIEW) - 4.0).contains(&p.x)
                        || !(4.0..f64::from(VIEW) - 4.0).contains(&p.y)
                    {
                        continue;
                    }
                    let at = ((p.y as u32 * VIEW + p.x as u32) * 4) as usize;
                    let px = &image.pixels[at..at + 4];
                    assert!(
                        px[0] > 100,
                        "z{zoom} at {lon},{lat} pitch {pitch}: ({x}, {y}) at {p} is {px:?}, not the roof"
                    );
                    seen += 1;
                }
            }
        }
        assert!(seen > 20, "z{zoom} at {lon},{lat}: {seen} points in view");
    }
}

/// A per-feature extrusion opacity is a property of each building, not the
/// layer: a value that is zero without a feature still draws the buildings
/// it makes opaque, and a building at zero hides nothing behind it.
#[test]
fn extrusions_at_zero_opacity_hide_nothing() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    // A wide low building, and a narrow tall one standing on its roof.
    let style = r##"{"version": 8, "center": [0, 0], "zoom": 2, "sources": {"b": {"type": "geojson",
        "data": {"type": "FeatureCollection", "features": [
          {"type": "Feature", "properties": {"k": "low", "h": 1000}, "geometry": {"type": "Polygon",
            "coordinates": [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]]}},
          {"type": "Feature", "properties": {"k": "tall", "h": 100000}, "geometry": {"type": "Polygon",
            "coordinates": [[[-3, -3], [3, -3], [3, 3], [-3, 3], [-3, -3]]]}}]}}},
      "layers": [
        {"id": "background", "type": "background", "paint": {"background-color": "#000000"}},
        {"id": "b", "type": "fill-extrusion", "source": "b", "paint": {
          "fill-extrusion-color": "#ffffff", "fill-extrusion-height": ["get", "h"],
          "fill-extrusion-opacity": ["match", ["get", "k"], "low", 1, 0]}}]}"##;
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).expect("style");
    let image = map.snapshot(SIZE, SIZE, 1.0).expect("snapshot");
    let centre = pixel(&image.pixels, SIZE / 2, SIZE / 2);
    let beside = pixel(&image.pixels, SIZE / 2 + 24, SIZE / 2);
    assert!(beside[0] > 100, "the low building draws: {beside:?}");
    assert_eq!(centre, beside, "the tall one at zero hides none of it");
}

/// `["elevation"]` in an extrusion's colour is the height above the
/// ground: seen from above, each roof takes the colour at its own height,
/// with stops from the feature's data and then its state. Unlit (light
/// intensity 0), a colour draws 0.03 brighter.
#[test]
fn extrusion_roofs_take_the_colour_at_their_height() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let style = r##"{"version": 8, "center": [0, 0], "zoom": 2, "light": {"intensity": 0},
      "sources": {"b": {"type": "geojson", "data": {"type": "FeatureCollection", "features": [
          {"type": "Feature", "id": 1, "properties": {"h": 1000, "top": "#ffffff"}, "geometry": {"type": "Polygon",
            "coordinates": [[[-10, -5], [-1, -5], [-1, 5], [-10, 5], [-10, -5]]]}},
          {"type": "Feature", "id": 2, "properties": {"h": 3000, "top": "#ffffff"}, "geometry": {"type": "Polygon",
            "coordinates": [[[1, -5], [10, -5], [10, 5], [1, 5], [1, -5]]]}}]}}},
      "layers": [
        {"id": "background", "type": "background", "paint": {"background-color": "#000000"}},
        {"id": "b", "type": "fill-extrusion", "source": "b", "paint": {
          "fill-extrusion-height": ["get", "h"],
          "fill-extrusion-color": ["interpolate", ["linear"], ["elevation"],
            0, "#000000", 4000, ["coalesce", ["feature-state", "top"], ["get", "top"]]]}}]}"##;
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).expect("style");
    let roofs = |map: &mut Map| {
        let image = map.snapshot(SIZE, SIZE, 1.0).expect("snapshot");
        let roof = |x| pixel(&image.pixels, x, SIZE / 2);
        (roof(SIZE / 2 - 12), roof(SIZE / 2 + 12))
    };
    let near = |got: [u8; 4], expected: [f32; 3]| {
        let lit = expected.map(|c| (c + 0.03) * 255.0);
        assert!(
            (0..3).all(|i| (f32::from(got[i]) - lit[i]).abs() <= 2.0),
            "{got:?}, not {lit:?}"
        );
    };
    let (low, high) = roofs(&mut map);
    near(low, [0.25; 3]);
    near(high, [0.75; 3]);
    map.edit(&commands(
        r##"[["setFeatureState", {"source": "b", "id": 2}, {"top": "#ff0000"}]]"##,
    ))
    .expect("state");
    let (low, high) = roofs(&mut map);
    near(low, [0.25; 3]);
    near(high, [0.75, 0.0, 0.0]);
}

/// Walls take the colour at each fragment's height: a building coloured
/// red below 50 m and blue above, seen from the side, shows both, and its
/// roof blue.
#[test]
fn extrusion_walls_take_the_colour_at_each_height() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let style = r##"{"version": 8, "center": [0, 0], "zoom": 14.5, "pitch": 60, "bearing": 30,
      "light": {"intensity": 0},
      "sources": {"b": {"type": "geojson", "data": {"type": "Polygon",
        "coordinates": [[[-0.0002, -0.0002], [0.0002, -0.0002], [0.0002, 0.0002], [-0.0002, 0.0002], [-0.0002, -0.0002]]]}}},
      "layers": [
        {"id": "background", "type": "background", "paint": {"background-color": "#000000"}},
        {"id": "b", "type": "fill-extrusion", "source": "b", "paint": {
          "fill-extrusion-height": 100, "fill-extrusion-vertical-gradient": false,
          "fill-extrusion-color": ["step", ["elevation"], "#ff0000", 50, "#0000ff"]}}]}"##;
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).expect("style");
    let image = map.snapshot(SIZE, SIZE, 1.0).expect("snapshot");
    let count = |f: fn(&[u8]) -> bool| image.pixels.chunks(4).filter(|p| f(p)).count();
    let red = count(|p| p[0] > 200 && p[2] < 40);
    let blue = count(|p| p[2] > 200 && p[0] < 40);
    assert!(red > 40 && blue > 40, "{red} red and {blue} blue pixels");
    // Down the middle, the roof and upper walls come first.
    let top = (0..SIZE)
        .map(|y| pixel(&image.pixels, SIZE / 2, y))
        .find(|p| p[0] > 100 || p[2] > 100);
    assert!(top.is_some_and(|p| p[2] > 200), "the top is blue: {top:?}");
}

/// A stop from the feature's data or state under a zoom curve blends
/// toward the next zoom, though the first stop is the node's: at z2.5, a
/// roof halfway from white to black.
#[test]
fn extrusion_elevation_stops_blend_between_zooms() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    for top in [
        r#"["get", "top"]"#,
        r##"["coalesce", ["feature-state", "top"], "#ffffff"]"##,
    ] {
        let style = format!(
            r##"{{"version": 8, "center": [0, 0], "zoom": 2.5, "light": {{"intensity": 0}},
          "sources": {{"b": {{"type": "geojson", "data": {{"type": "Feature", "id": 1,
            "properties": {{"top": "#ffffff"}}, "geometry": {{"type": "Polygon",
            "coordinates": [[[-10, -5], [10, -5], [10, 5], [-10, 5], [-10, -5]]]}}}}}}}},
          "layers": [
            {{"id": "background", "type": "background", "paint": {{"background-color": "#000000"}}}},
            {{"id": "b", "type": "fill-extrusion", "source": "b", "paint": {{
              "fill-extrusion-height": 4000,
              "fill-extrusion-color": ["interpolate", ["linear"], ["zoom"],
                2, ["interpolate", ["linear"], ["elevation"], 0, "#000000", 4000, {top}],
                3, "#000000"]}}}}]}}"##
        );
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&style).expect("style");
        let image = map.snapshot(SIZE, SIZE, 1.0).expect("snapshot");
        let roof = pixel(&image.pixels, SIZE / 2, SIZE / 2);
        let lit = (0.5 + 0.03) * 255.0;
        assert!(
            (f32::from(roof[0]) - lit).abs() <= 3.0,
            "{top}: {roof:?}, not {lit}"
        );
    }
}

#[test]
fn padding_draws_the_center_where_it_projects() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(STYLE).expect("style");
    settle(&mut map, &mut target);
    // The box is 28 pixels a side: padding at the top moves it down, off
    // the top rows.
    map.jump_to(&CameraTarget {
        padding: Some(orb_api::Padding {
            top: 40.0,
            ..Default::default()
        }),
        ..CameraTarget::default()
    });
    settle(&mut map, &mut target);
    let center = map
        .project(orb_api::Geodetic::new(0.0, 0.0))
        .expect("in view");
    assert!(
        (center - glam::DVec2::new(32.0, 52.0)).length() < 1e-9,
        "{center}"
    );
    let pixels = target.read_pixels().unwrap();
    assert_eq!(pixel(&pixels, 32, 24), [255, 0, 0, 255], "above the box");
    assert_eq!(
        pixel(&pixels, 32, 62),
        [0, 0, 255, 255],
        "the box, moved down"
    );
    let hit = map.hit_test(center).expect("the box");
    assert_eq!(hit.layer, "box");
}

/// A queried feature's `layer` gives the paint and layout the style sets,
/// evaluated for that feature, its state and the camera, as GL JS's
/// `queryRenderedFeatures` does.
#[test]
fn queried_layers_are_evaluated_for_their_feature() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(
        r##"{"version": 8, "center": [0, 0], "zoom": 2,
            "sources": {"p": {"type": "geojson", "data": {"type": "Feature", "id": 3,
                "properties": {"r": 6}, "geometry": {"type": "Point", "coordinates": [0, 0]}}}},
            "layers": [{"id": "c", "type": "circle", "source": "p",
                "layout": {"circle-sort-key": ["get", "r"]},
                "paint": {
                    "circle-radius": ["case", ["boolean", ["feature-state", "big"], false], 20, ["get", "r"]],
                    "circle-color": "#ff0000",
                    "circle-opacity": ["interpolate", ["linear"], ["zoom"], 0, 0, 4, 1]}}]}"##,
    )
    .expect("style");
    settle(&mut map, &mut target);
    let center = glam::DVec2::splat(f64::from(SIZE) / 2.0);
    let layer = |map: &mut Map, target: &mut Target| {
        let query = map.query_rendered(center, center);
        let doc = json::parse(&answer(map, target, query).unwrap()).unwrap();
        doc.as_array().unwrap()[0].get("layer").unwrap().to_string()
    };
    assert_eq!(
        layer(&mut map, &mut target),
        r#"{"id":"c","type":"circle","source":"p","paint":{"circle-radius":6,"circle-color":{"r":1,"g":0,"b":0,"a":1},"circle-opacity":0.5},"layout":{"circle-sort-key":6}}"#
    );
    map.edit(&commands(
        r#"[["setFeatureState", {"source": "p", "id": 3}, {"big": true}]]"#,
    ))
    .expect("state");
    settle(&mut map, &mut target);
    assert!(
        layer(&mut map, &mut target).contains(r#""circle-radius":20,"#),
        "the state's radius"
    );
}

/// A rendered query's filter is evaluated at the feature's tile's zoom,
/// with its properties and geometry, as GL JS and MapLibre Native do.
#[test]
fn rendered_query_filters_evaluate_at_the_tiles_zoom() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(
        r##"{"version": 8, "center": [0, 0], "zoom": 12,
            "sources": {"box": {"type": "geojson", "data": {"type": "Feature", "id": 7,
                "properties": {"kind": "park"}, "geometry": {"type": "Polygon",
                "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]]}}}},
            "layers": [{"id": "box", "type": "fill", "source": "box",
                "paint": {"fill-color": "#0000ff"}}]}"##,
    )
    .expect("style");
    settle(&mut map, &mut target);
    let center = glam::DVec2::splat(f64::from(SIZE) / 2.0);
    let mut query = |filter: &str| {
        let id = map.query_rendered_in(center, center, None, Some(filter));
        answer(&mut map, &mut target, id).unwrap()
    };
    let kept = r#"["all", [">=", ["zoom"], 10], ["==", ["geometry-type"], "Polygon"],
        ["==", ["get", "kind"], "park"]]"#;
    let features = query(kept);
    assert!(features.contains(r#""id":7"#), "{features}");
    assert_eq!(query(r#"["<", ["zoom"], 10]"#), "[]");
    // Legacy syntax, as a layer's filter takes it.
    assert!(query(r#"["==", "kind", "park"]"#).contains(r#""id":7"#));
    assert_eq!(query(r#"["!=", "kind", "park"]"#), "[]");
}

#[test]
fn source_queries_select_by_filter_and_area() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    let points: Vec<String> = (0..6)
        .map(|i| {
            format!(
                r#"{{"type":"Feature","id":{i},"properties":{{"even":{}}},"geometry":{{"type":"Point","coordinates":[{},0]}}}}"#,
                i % 2 == 0,
                i * 10 - 25
            )
        })
        .collect();
    map.set_style_json(&format!(
        r#"{{"version":8,"center":[0,0],"zoom":0,
          "sources":{{"p":{{"type":"geojson","data":{{"type":"FeatureCollection","features":[{}]}}}}}},
          "layers":[{{"id":"p","type":"circle","source":"p"}}]}}"#,
        points.join(",")
    ))
    .expect("style");
    settle(&mut map, &mut target);
    let ids = |json: String| {
        let doc = json::parse(&json).unwrap();
        let mut ids: Vec<i64> = doc
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.get("id").unwrap().as_f64().unwrap() as i64)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    let all = map.query_source("p", None, None);
    assert_eq!(
        ids(answer(&mut map, &mut target, all).unwrap()),
        [0, 1, 2, 3, 4, 5]
    );
    let even = map.query_source("p", None, Some(r#"["get", "even"]"#));
    assert_eq!(ids(answer(&mut map, &mut target, even).unwrap()), [0, 2, 4]);
    // East of the meridian: points at 5, 15 and 25 degrees.
    let east = map.query_source(
        "p",
        None,
        Some(
            r#"["within", {"type":"Polygon","coordinates":[[[0,-10],[30,-10],[30,10],[0,10],[0,-10]]]}]"#,
        ),
    );
    assert_eq!(ids(answer(&mut map, &mut target, east).unwrap()), [3, 4, 5]);
    let missing = map.query_source("nope", None, None);
    assert!(answer(&mut map, &mut target, missing).is_err());
}

#[test]
fn clustered_sources_draw_clusters_and_answer_cluster_queries() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    // Twelve points a few metres apart: one cluster at zoom 0.
    let points: Vec<String> = (0..12)
        .map(|i| {
            format!(
                r#"{{"type":"Feature","id":{i},"properties":{{"n":{i}}},"geometry":{{"type":"Point","coordinates":[{},0]}}}}"#,
                f64::from(i) * 0.0001
            )
        })
        .collect();
    map.set_style_json(&format!(
        r##"{{"version":8,"center":[0,0],"zoom":0,
          "sources":{{"p":{{"type":"geojson","cluster":true,
            "clusterProperties":{{"sum":["+",["get","n"]]}},
            "data":{{"type":"FeatureCollection","features":[{}]}}}}}},
          "layers":[
            {{"id":"background","type":"background","paint":{{"background-color":"#ff0000"}}}},
            {{"id":"clusters","type":"circle","source":"p","filter":["has","point_count"],
              "paint":{{"circle-radius":10,"circle-color":"#0000ff"}}}}]}}"##,
        points.join(",")
    ))
    .expect("style");
    settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(
        pixel(&pixels, SIZE / 2, SIZE / 2),
        [0, 0, 255, 255],
        "the cluster"
    );
    let center = glam::DVec2::splat(f64::from(SIZE) / 2.0);
    let rendered = map.query_rendered(center, center);
    let doc = json::parse(&answer(&mut map, &mut target, rendered).unwrap()).unwrap();
    let cluster = &doc.as_array().unwrap()[0];
    let properties = cluster.get("properties").unwrap();
    assert_eq!(properties.get("point_count").unwrap().as_f64(), Some(12.0));
    assert_eq!(properties.get("sum").unwrap().as_f64(), Some(66.0));
    let id = properties.get("cluster_id").unwrap().as_f64().unwrap() as u64;

    let leaves = map.query_cluster(
        "p",
        orb_api::ClusterQuery::Leaves {
            id,
            limit: 5,
            offset: 10,
        },
    );
    let leaves = json::parse(&answer(&mut map, &mut target, leaves).unwrap()).unwrap();
    assert_eq!(leaves.as_array().unwrap().len(), 2);
    let zoom = map.query_cluster("p", orb_api::ClusterQuery::ExpansionZoom(id));
    let zoom: f64 = answer(&mut map, &mut target, zoom)
        .unwrap()
        .parse()
        .unwrap();
    assert!(zoom >= 1.0, "{zoom}");
    let children = map.query_cluster("p", orb_api::ClusterQuery::Children(id));
    let children = json::parse(&answer(&mut map, &mut target, children).unwrap()).unwrap();
    assert!(!children.as_array().unwrap().is_empty());
    // Not a cluster: an error.
    let wrong = map.query_cluster("p", orb_api::ClusterQuery::Children(3));
    assert!(answer(&mut map, &mut target, wrong).is_err());
}

#[test]
fn canvas_sources_show_the_pixels_the_app_hands_in() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    // No player: a canvas needs none.
    map.set_style_json(
        r##"{"version": 8, "center": [0, 0], "zoom": 2,
            "sources": {"c": {"type": "canvas", "canvas": "drawing", "animate": true,
                "coordinates": [[-120, 70], [120, 70], [120, -70], [-120, -70]]}},
            "layers": [
                {"id": "background", "type": "background", "paint": {"background-color": "#ff0000"}},
                {"id": "c", "type": "raster", "source": "c", "paint": {"raster-fade-duration": 0}}]}"##,
    )
    .expect("style");
    for rgba in [[0, 0, 255, 255], [0, 255, 0, 255]] {
        map.source_frame("c", 4, 4, rgba.repeat(16))
            .expect("a frame");
        let events = settle(&mut map, &mut target);
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Event::Map(MapEvent::Video { .. }))),
            "{events:?}"
        );
        let pixels = target.read_pixels().unwrap();
        assert_eq!(pixel(&pixels, SIZE / 2, SIZE / 2), rgba, "the canvas");
    }
    assert!(map.control_video("c", orb_api::VideoRequest::Play).is_err());
    assert!(map.source_frame("d", 1, 1, vec![0; 4]).is_err());
}

#[test]
fn image_sources_without_a_url_draw_nothing_and_idle() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_viewport(f64::from(SIZE), f64::from(SIZE));
    map.set_style_json(
        r##"{"version": 8, "center": [0, 0], "zoom": 2,
            "sources": {"i": {"type": "image",
                "coordinates": [[-120, 70], [120, 70], [120, -70], [-120, -70]]}},
            "layers": [
                {"id": "background", "type": "background", "paint": {"background-color": "#ff0000"}},
                {"id": "i", "type": "raster", "source": "i"}]}"##,
    )
    .expect("style");
    let events = settle(&mut map, &mut target);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Map(MapEvent::Idle))),
        "{events:?}"
    );
    assert!(
        !events.iter().any(|e| matches!(
            e,
            Event::Map(MapEvent::Error { .. } | MapEvent::Warning { .. })
        )),
        "{events:?}"
    );
    let pixels = target.read_pixels().unwrap();
    assert_eq!(pixel(&pixels, SIZE / 2, SIZE / 2), [255, 0, 0, 255]);
    let still = map.snapshot(SIZE, SIZE, 1.0).expect("a still");
    let i = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
    assert_eq!(still.pixels[i..i + 4], [255, 0, 0, 255]);
}

#[test]
fn geojson_diffs_change_what_queries_find() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(
        r#"{"version":8,"center":[0,0],"zoom":0,
          "sources":{"p":{"type":"geojson","data":{"type":"FeatureCollection","features":[
            {"type":"Feature","id":1,"properties":{"n":1},"geometry":{"type":"Point","coordinates":[0,0]}},
            {"type":"Feature","id":2,"properties":{"n":2},"geometry":{"type":"Point","coordinates":[10,0]}}]}}},
          "layers":[{"id":"p","type":"circle","source":"p"}]}"#,
    )
    .expect("style");
    settle(&mut map, &mut target);
    let mut tx = Transaction::new();
    tx.push(Op::UpdateSourceData {
        id: "p".into(),
        diff: json::parse(
            r#"{"remove":[2],"add":[{"type":"Feature","id":3,"properties":{"n":3},
                 "geometry":{"type":"Point","coordinates":[-10,0]}}],
               "update":[{"id":1,"addOrUpdateProperties":[{"key":"n","value":10}]}]}"#,
        )
        .unwrap(),
    });
    map.edit(&tx).expect("a diff");
    settle(&mut map, &mut target);
    let all = map.query_source("p", None, None);
    let doc = json::parse(&answer(&mut map, &mut target, all).unwrap()).unwrap();
    let mut found: Vec<(i64, i64)> = doc
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            let n = f
                .get("properties")
                .unwrap()
                .get("n")
                .unwrap()
                .as_f64()
                .unwrap();
            (f.get("id").unwrap().as_f64().unwrap() as i64, n as i64)
        })
        .collect();
    found.sort_unstable();
    found.dedup();
    assert_eq!(found, [(1, 10), (3, 3)]);
}

/// Draws frames until the map asks for a tile of an app's source.
fn tile_request(map: &mut Map, target: &mut Target) -> (String, orb_api::TileIndex) {
    common::until(map, target, "tile request", |map, _| {
        std::iter::from_fn(|| map.poll_event()).find_map(|event| match event {
            Event::Map(MapEvent::TileRequest { source, tile }) => Some((source, tile)),
            _ => None,
        })
    })
}

#[test]
fn app_sources_draw_the_tiles_the_app_hands_in() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(
        r##"{"version": 8, "center": [0, 0], "zoom": 0,
            "sources": {"app": {"type": "vector", "tiles": ["orbis-host:"], "maxzoom": 0}},
            "layers": [
                {"id": "background", "type": "background", "paint": {"background-color": "#ff0000"}},
                {"id": "shapes", "type": "fill", "source": "app", "source-layer": "shapes",
                 "paint": {"fill-color": "#0000ff"}}]}"##,
    )
    .expect("style");
    let (source, tile) = tile_request(&mut map, &mut target);
    assert_eq!(
        (source.as_str(), tile),
        ("app", orb_api::TileIndex::new(0, 0, 0))
    );
    let shapes = r#"{"shapes": {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon",
        "coordinates": [[[-20, -20], [20, -20], [20, 20], [-20, 20], [-20, -20]]]}}}"#;
    map.host_tile("app", tile, orb_api::HostTile::GeoJson(shapes.into()))
        .expect("an answer");
    settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(
        pixel(&pixels, SIZE / 2, SIZE / 2),
        [0, 0, 255, 255],
        "the app's shape"
    );

    // New data: asked again; until answered, the old tile stays.
    map.reload_host_tiles("app").expect("a reload");
    let (_, again) = tile_request(&mut map, &mut target);
    assert_eq!(again, tile);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(
        pixel(&pixels, SIZE / 2, SIZE / 2),
        [0, 0, 255, 255],
        "the drawn tile stays"
    );
    map.host_tile("app", tile, orb_api::HostTile::Empty)
        .expect("an answer");
    settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(
        pixel(&pixels, SIZE / 2, SIZE / 2),
        [255, 0, 0, 255],
        "an empty tile"
    );
    assert!(
        map.host_tile("nope", tile, orb_api::HostTile::Empty)
            .is_err()
    );
}

/// A vector tile of one layer, `shapes`, whose one polygon covers the
/// tile, with a property `c` if `marked`.
fn covering_tile(marked: bool) -> Vec<u8> {
    use common::mvt;
    let square = [[0, 0], [4096, 0], [4096, 4096], [0, 4096]];
    let feature = mvt::Feature {
        kind: 3,
        geometry: mvt::path(&square, true),
        tags: if marked { &[0, 0] } else { &[] },
    };
    match marked {
        true => mvt::tile("shapes", &[feature], &["c"], &["x"]),
        false => mvt::tile("shapes", &[feature], &[], &[]),
    }
}

#[test]
fn tiles_whose_data_expires_load_again_while_drawn() {
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // The tile, fresh for a second (which the map takes as its shortest
    // lifetime, a few), then a failure, then marked and fresh for none.
    let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = server.local_addr().unwrap();
    let requests = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&requests);
    std::thread::spawn(move || {
        for stream in server.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut request = [0; 2048];
            let _ = stream.read(&mut request);
            let (status, lifetime, body) = match counted.fetch_add(1, Ordering::SeqCst) {
                0 => (
                    "200 OK",
                    "Cache-Control: max-age=1\r\n",
                    covering_tile(false),
                ),
                1 => ("500 Internal Server Error", "", Vec::new()),
                _ => ("200 OK", "", covering_tile(true)),
            };
            let head = format!(
                "HTTP/1.1 {status}\r\nConnection: close\r\n{lifetime}Content-Length: {}\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    let asked = std::time::Instant::now();
    map.set_style_json(&format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 0,
            "sources": {{"v": {{"type": "vector", "tiles": ["http://{address}/{{z}}/{{x}}/{{y}}.mvt"],
                "maxzoom": 0}}}},
            "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#ff0000"}}}},
                {{"id": "shapes", "type": "fill", "source": "v", "source-layer": "shapes",
                 "paint": {{"fill-color": ["case", ["has", "c"], "#00ff00", "#0000ff"]}}}}]}}"##
    ))
    .expect("style");
    settle(&mut map, &mut target);
    let centre = |target: &Target| pixel(&target.read_pixels().unwrap(), SIZE / 2, SIZE / 2);
    let (old, new) = ([0, 0, 255, 255], [0, 255, 0, 255]);
    assert_eq!(centre(&target), old, "the tile as it came first");
    // Idle while the tile is fresh, until its expiry wakes the map.
    assert!(!map.needs_frame(), "a map waiting for an expiry is idle");
    common::wake(&map);
    assert!(
        asked.elapsed() >= Duration::from_secs(3),
        "woken before the shortest lifetime"
    );
    assert_eq!(
        requests.load(Ordering::SeqCst),
        1,
        "woken before the reload"
    );
    // Each frame draws the old tile until the new one replaces it, the
    // failed reload tried again a little later.
    common::until(&mut map, &mut target, "reloaded tile", |map, target| {
        let shown = centre(target);
        assert!(shown == old || shown == new, "a frame drew {shown:?}");
        (shown == new && !map.needs_frame()).then_some(())
    });
    // The reload came with no lifetime: nothing waits on the clock, so the
    // map loads nothing more.
    let report = map
        .render_until_idle(&mut target, common::DEADLINE)
        .expect("the map idles");
    assert_eq!(report.wake_at, None);
    assert_eq!(requests.load(Ordering::SeqCst), 3);
}

/// A tile that failed for a reason that may pass loads once its backoff
/// ends, on a map idle until then: the map wakes itself for the retry.
#[test]
fn failed_tiles_wake_an_idle_map_for_their_retry() {
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // The server is down, however often the map retries (a first frame
    // to idle slower than the backoff retries before it idles), until
    // it's up; then it answers.
    let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = server.local_addr().unwrap();
    let (up, requests) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicUsize::new(0)),
    );
    let (serving, counted) = (Arc::clone(&up), Arc::clone(&requests));
    std::thread::spawn(move || {
        for stream in server.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut request = [0; 2048];
            let _ = stream.read(&mut request);
            counted.fetch_add(1, Ordering::SeqCst);
            let (status, body) = match serving.load(Ordering::SeqCst) {
                false => ("503 Service Unavailable", Vec::new()),
                true => ("200 OK", covering_tile(false)),
            };
            let head = format!(
                "HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    // The map's clock reads 0 no sooner than this.
    let born = std::time::Instant::now();
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 0,
            "sources": {{"v": {{"type": "vector", "tiles": ["http://{address}/{{z}}/{{x}}/{{y}}.mvt"],
                "maxzoom": 0}}}},
            "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#ff0000"}}}},
                {{"id": "shapes", "type": "fill", "source": "v", "source-layer": "shapes",
                 "paint": {{"fill-color": "#0000ff"}}}}]}}"##
    ))
    .expect("style");
    let report = map
        .render_until_idle(&mut target, common::DEADLINE)
        .expect("the map idles");
    let drawn = std::time::Instant::now();
    let centre = |target: &Target| pixel(&target.read_pixels().unwrap(), SIZE / 2, SIZE / 2);
    assert_eq!(
        centre(&target),
        [255, 0, 0, 255],
        "the failed tile left a hole"
    );
    let failed = requests.load(Ordering::SeqCst);
    assert!(failed >= 1);
    // Idle until the backoff ends: a second after the first failure,
    // doubling with each further one.
    let wake_at = report.wake_at.expect("a wake for the retry");
    let due = born + Duration::from_secs_f64(wake_at / 1000.0);
    assert!(
        due <= drawn + Duration::from_secs(1 << (failed - 1)),
        "due {:?} after {failed} failures",
        due - drawn
    );
    assert!(!map.needs_frame(), "a map waiting for a retry is idle");
    up.store(true, Ordering::SeqCst);
    common::wake(&map);
    let woken = std::time::Instant::now();
    assert!(woken >= due, "woken {:?} early", due - woken);
    assert_eq!(
        requests.load(Ordering::SeqCst),
        failed,
        "retried before the wake"
    );
    // The frame it wakes for asks for the tile again.
    let report = map
        .render_until_idle(&mut target, common::DEADLINE)
        .expect("the map idles");
    assert_eq!(centre(&target), [0, 0, 255, 255], "the retried tile");
    assert_eq!(requests.load(Ordering::SeqCst), failed + 1);
    assert_eq!(report.wake_at, None, "nothing left to retry");
}

/// Still output waits as long as its data keeps arriving, however long
/// that takes in all, and gives up once nothing has arrived for its stall,
/// saying what it waited for.
#[test]
fn stills_wait_while_data_arrives_and_give_up_on_a_stall() {
    use std::io::{Read, Write};
    use std::time::Instant;

    // Answers one tile at a time, each `delay` after it's asked for, or
    // with no delay, never: it holds every request.
    fn serve(delay: Option<Duration>) -> std::net::SocketAddr {
        let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = server.local_addr().unwrap();
        std::thread::spawn(move || {
            let mut held = Vec::new();
            for stream in server.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut request = [0; 2048];
                let _ = stream.read(&mut request);
                let Some(delay) = delay else {
                    held.push(stream);
                    continue;
                };
                std::thread::sleep(delay);
                let body = covering_tile(false);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        address
    }
    // The four tiles of zoom 1, which a 512-pixel view at zoom 1 shows.
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, 512, 512, 1.0);
    let still = |address| {
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&format!(
            r##"{{"version": 8, "center": [0, 0], "zoom": 1,
                "sources": {{"v": {{"type": "vector",
                    "tiles": ["http://{address}/{{z}}/{{x}}/{{y}}.mvt"], "maxzoom": 1}}}},
                "layers": [{{"id": "shapes", "type": "fill", "source": "v",
                    "source-layer": "shapes", "paint": {{"fill-color": "#0000ff"}}}}]}}"##
        ))
        .expect("style");
        map
    };
    let stall = Duration::from_secs(2);
    let mut slow = still(serve(Some(Duration::from_millis(700))));
    let started = Instant::now();
    slow.render_until_idle(&mut target, stall)
        .expect("a still whose tiles keep coming settles");
    assert!(
        started.elapsed() > stall,
        "the tiles came in {:?}, within one stall",
        started.elapsed()
    );
    let mut stalled = still(serve(None));
    let started = Instant::now();
    let error = stalled
        .render_until_idle(&mut target, stall)
        .expect_err("a still whose tiles never come gives up");
    assert!(started.elapsed() >= stall);
    let Error::InvalidState(message) = error else {
        panic!("{error:?}")
    };
    assert!(
        message.contains("stalled") && message.contains("loading 4 tiles"),
        "{message}"
    );
    // A source whose TileJSON never comes has no tiles to count, and is
    // named instead.
    let mut unresolved = Map::new(&runtime, MapOptions::default());
    unresolved
        .set_style_json(&format!(
            r##"{{"version": 8, "center": [0, 0], "zoom": 1,
                "sources": {{"v": {{"type": "vector", "url": "http://{}/v.json"}}}},
                "layers": [{{"id": "shapes", "type": "fill", "source": "v",
                    "source-layer": "shapes"}}]}}"##,
            serve(None)
        ))
        .expect("style");
    let error = unresolved
        .render_until_idle(&mut target, stall)
        .expect_err("a still whose TileJSON never comes gives up");
    assert!(error.message().contains("loading sources v"), "{error:?}");
}

/// A continuous map draws a tile its HTTP cache holds stale at once and
/// revalidates it behind, drawing the server's answer once it comes; a
/// still waits for the server.
#[test]
fn stale_cached_tiles_draw_at_once_while_they_revalidate() {
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Each answer stale on arrival; all but the first held until released.
    let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = server.local_addr().unwrap();
    let requests = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&requests);
    let (release, released) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        for stream in server.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut request = [0; 2048];
            let _ = stream.read(&mut request);
            let n = counted.fetch_add(1, Ordering::SeqCst);
            if n > 0 && released.recv().is_err() {
                return;
            }
            let body = covering_tile(n == 1);
            let head = format!(
                "HTTP/1.1 200 OK\r\nConnection: close\r\nCache-Control: max-age=0\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    let dir = common::Scratch::new("api-stale");
    let runtime = Runtime::new(&RuntimeOptions {
        state_dir: Some(dir.to_path_buf()),
        ..RuntimeOptions::default()
    })
    .expect("runtime");
    let style = format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 0,
            "sources": {{"v": {{"type": "vector", "tiles": ["http://{address}/{{z}}/{{x}}/{{y}}.mvt"],
                "maxzoom": 0}}}},
            "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#ff0000"}}}},
                {{"id": "shapes", "type": "fill", "source": "v", "source-layer": "shapes",
                 "paint": {{"fill-color": ["case", ["has", "c"], "#00ff00", "#0000ff"]}}}}]}}"##
    );
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let centre = |target: &Target| pixel(&target.read_pixels().unwrap(), SIZE / 2, SIZE / 2);
    let (old, new) = ([0, 0, 255, 255], [0, 255, 0, 255]);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).expect("style");
    settle(&mut map, &mut target);
    assert_eq!(centre(&target), old, "the server's first answer");
    drop(map);

    // A new map draws the cached copy while the server holds its answer.
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).expect("style");
    common::until(&mut map, &mut target, "stale copy", |_, target| {
        (centre(target) == old && requests.load(Ordering::SeqCst) >= 2).then_some(())
    });
    assert!(map.needs_frame(), "busy while it revalidates");
    release.send(()).unwrap();
    common::until(&mut map, &mut target, "revalidated tile", |map, target| {
        (centre(target) == new && !map.needs_frame()).then_some(())
    });
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    drop(map);

    // A still draws what the server answers, not the stale copy.
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).expect("style");
    let mut released = false;
    let started = std::time::Instant::now();
    loop {
        if requests.load(Ordering::SeqCst) == 3 && !released {
            release.send(()).unwrap();
            released = true;
        }
        if let Some(image) = map.snapshot_step(SIZE, SIZE, 1.0).unwrap() {
            assert!(released, "a still drew before the server answered");
            assert_eq!(pixel(&image.pixels, SIZE / 2, SIZE / 2), old);
            return;
        }
        assert!(
            started.elapsed() < common::DEADLINE,
            "the still never settled"
        );
        map.wait_woken(Duration::from_millis(16));
    }
}

/// A stale tile the server confirms, with a 304 that gives it no
/// lifetime, stays as drawn and loads no more, as on a map that loaded it
/// fresh from the server.
#[test]
fn stale_cached_tiles_the_server_confirms_settle() {
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // The tile stale on arrival, then confirmed by a bare 304.
    let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = server.local_addr().unwrap();
    let requests = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&requests);
    std::thread::spawn(move || {
        for stream in server.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut request = [0; 2048];
            let read = stream.read(&mut request).unwrap_or(0);
            counted.fetch_add(1, Ordering::SeqCst);
            let request = String::from_utf8_lossy(&request[..read]).to_ascii_lowercase();
            let (head, body) = match request.contains("if-none-match") {
                true => (
                    "HTTP/1.1 304 Not Modified\r\nConnection: close\r\nETag: \"a\"\r\n\r\n"
                        .to_string(),
                    Vec::new(),
                ),
                false => {
                    let body = covering_tile(false);
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nConnection: close\r\nCache-Control: max-age=0\r\nETag: \"a\"\r\nContent-Length: {}\r\n\r\n",
                        body.len()
                    );
                    (head, body)
                }
            };
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    let dir = common::Scratch::new("api-stale-confirmed");
    let runtime = Runtime::new(&RuntimeOptions {
        state_dir: Some(dir.to_path_buf()),
        ..RuntimeOptions::default()
    })
    .expect("runtime");
    let style = format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 0,
            "sources": {{"v": {{"type": "vector", "tiles": ["http://{address}/{{z}}/{{x}}/{{y}}.mvt"],
                "maxzoom": 0}}}},
            "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#ff0000"}}}},
                {{"id": "shapes", "type": "fill", "source": "v", "source-layer": "shapes",
                 "paint": {{"fill-color": "#0000ff"}}}}]}}"##
    );
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).expect("style");
    settle(&mut map, &mut target);
    drop(map);
    assert_eq!(requests.load(Ordering::SeqCst), 1);

    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).expect("style");
    common::until(&mut map, &mut target, "revalidation", |map, _| {
        (requests.load(Ordering::SeqCst) >= 2 && !map.needs_frame()).then_some(())
    });
    // Nothing waits on the clock, no reload's retry: nothing more is asked
    // for.
    let report = map
        .render_until_idle(&mut target, common::DEADLINE)
        .expect("the map idles");
    assert_eq!(report.wake_at, None, "a reload is due");
    let blue = [0, 0, 255, 255];
    assert_eq!(
        pixel(&target.read_pixels().unwrap(), SIZE / 2, SIZE / 2),
        blue
    );
    assert_eq!(requests.load(Ordering::SeqCst), 2, "reloaded again");
}

#[test]
fn an_app_sources_geojson_reaches_layers_without_a_source_layer() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(
        r##"{"version": 8, "center": [0, 0], "zoom": 0,
            "sources": {"app": {"type": "vector", "tiles": ["orbis-host:"], "maxzoom": 0}},
            "layers": [
                {"id": "background", "type": "background", "paint": {"background-color": "#ff0000"}},
                {"id": "shapes", "type": "fill", "source": "app", "paint": {"fill-color": "#0000ff"}}]}"##,
    )
    .expect("a layer without a source-layer on an app's source");
    let (_, tile) = tile_request(&mut map, &mut target);
    // A failed tile draws nothing, and a reload asks for it again.
    map.host_tile("app", tile, orb_api::HostTile::Empty)
        .expect("an answer");
    map.reload_host_tiles("app").expect("a reload");
    let (_, again) = tile_request(&mut map, &mut target);
    assert_eq!(again, tile, "the empty tile asked for again");
    let shapes = r#"{"_geojsonTileLayer": {"type": "Feature", "properties": {"n": 1},
        "geometry": {"type": "Polygon",
        "coordinates": [[[-20, -20], [20, -20], [20, 20], [-20, 20], [-20, -20]]]}}}"#;
    map.host_tile("app", tile, orb_api::HostTile::GeoJson(shapes.into()))
        .expect("an answer");
    settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(
        pixel(&pixels, SIZE / 2, SIZE / 2),
        [0, 0, 255, 255],
        "the app's GeoJSON"
    );
    // A query reads the answer the map keeps.
    let query = map.query_source("app", None, None);
    let found = answer(&mut map, &mut target, query).unwrap();
    assert!(found.contains(r#""n":1"#), "{found}");
}

#[test]
fn a_map_draws_on_the_thread_that_made_its_runtime() {
    // Not the main thread: a runtime and everything made with it belong to
    // the thread that made them, the embedder's render thread.
    let drawn = std::thread::Builder::new()
        .name("render".into())
        .spawn(|| {
            let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
            let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
            let mut map = Map::new(&runtime, MapOptions::default());
            map.set_style_json(STYLE).expect("style");
            settle(&mut map, &mut target);
            let pixels = target.read_pixels().unwrap();
            pixel(&pixels, SIZE / 2, SIZE / 2)
        })
        .unwrap()
        .join()
        .expect("the render thread");
    assert_eq!(drawn, [0, 0, 255, 255]);
}

#[test]
fn request_transforms_rewrite_what_the_map_fetches() {
    let dir = common::Scratch::new("api-requests");
    std::fs::write(dir.join("style.json"), STYLE).unwrap();
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let base = format!("file://{}/", dir.display());
    runtime.set_request_transform(Some(std::sync::Arc::new(move |url: &str| {
        let path = url.strip_prefix("app://")?;
        Some(orb_api::Request {
            url: format!("{base}{path}"),
            headers: vec![("Authorization".into(), "Bearer token".into())],
        })
    })));
    // Files read offline too.
    runtime.set_online(false);
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.load_style_url("app://style.json");
    settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(
        pixel(&pixels, SIZE / 2, SIZE / 2),
        [0, 0, 255, 255],
        "the rewritten style"
    );
    // The transform refuses what it doesn't rewrite.
    map.load_style_url("https://example.com/style.json");
    let refused = common::until(&mut map, &mut target, "error", |map, _| {
        std::iter::from_fn(|| map.poll_event()).find_map(|e| match e {
            Event::Map(MapEvent::Error { message }) => Some(message),
            _ => None,
        })
    });
    assert!(refused.contains("refused"), "{refused}");
}

#[test]
fn the_style_reads_back_with_its_edits() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut map = Map::new(&runtime, MapOptions::default());
    assert!(map.style().is_none());
    map.set_style_json(STYLE).expect("style");
    map.edit(&commands(
        r##"[["setPaintProperty", "box", "fill-color", "#00ff00"]]"##,
    ))
    .expect("edit");
    let style = map.style().expect("loaded");
    assert_eq!(style.layer_ids().collect::<Vec<_>>(), ["background", "box"]);
    assert_eq!(style.source_ids().collect::<Vec<_>>(), ["box"]);
    let paint = style.layer("box").unwrap().get("paint").unwrap();
    assert_eq!(paint.get("fill-color").unwrap().as_str(), Some("#00ff00"));
    assert_eq!(
        style.source("box").unwrap().get("type").unwrap().as_str(),
        Some("geojson")
    );
    assert!(style.layer("nope").is_none());
    assert!(style.to_json().contains("#00ff00"));
}

#[test]
fn fits_before_the_first_frame_use_the_viewport_set() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut map = Map::new(&runtime, MapOptions::default());
    let bounds = orb_api::Bounds {
        west: -10.0,
        south: -10.0,
        east: 10.0,
        north: 10.0,
    };
    let fit = |map: &Map| {
        map.camera_for_bounds(bounds, &orb_api::FitOptions::default())
            .and_then(|t| t.zoom)
            .unwrap()
    };
    // 20 degrees span 2⁹ × 20 / 360 pixels at zoom 0: 64 pixels fit them
    // at zoom log2(64 / 28.4) ≈ 1.17.
    map.set_viewport(64.0, 64.0);
    assert!(
        (fit(&map) - (64.0f64 / (512.0 * 20.0 / 360.0)).log2()).abs() < 0.05,
        "{}",
        fit(&map)
    );
    assert_eq!(map.viewport().width, 64.0);
}

/// Terrarium DEM tiles of the whole world, zooms 0 to 2, `metres` high
/// everywhere, written to a fresh directory.
fn level_dem(metres: f64) -> common::Scratch {
    let dir = common::Scratch::new("api-level-dem");
    const SIZE: u32 = 256;
    let v = ((metres + 32768.0) * 256.0).round() as u32;
    let pixels: Vec<u8> = (0..SIZE * SIZE)
        .flat_map(|_| [(v >> 16) as u8, (v >> 8) as u8, v as u8, 255])
        .collect();
    let mut png = Vec::new();
    let mut encoder = png::Encoder::new(&mut png, SIZE, SIZE);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    (encoder.write_header().unwrap())
        .write_image_data(&pixels)
        .unwrap();
    for z in 0..=2u32 {
        for x in 0..1 << z {
            for y in 0..1 << z {
                let path = dir.join(format!("{z}/{x}/{y}.png"));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, &png).unwrap();
            }
        }
    }
    dir
}

/// #165: ground below sea level runs on to the horizon, where the sky
/// meets it, at a high pitch in the globe's handover to the plane and on
/// the plane. Hidden as if behind the sea-level sphere, or cut off where
/// sea-level ground would end, its last rows left a line of the clear
/// colour; with the eye nearer the sea than the ground is below it, the
/// globe's cover dropped all of it. The sky takes the row the horizon
/// crosses, so no pixel there is left partly clear either.
#[test]
fn ground_below_sea_level_reaches_the_horizon() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    for (projection, metres) in [("globe", -300.0), ("globe", -1000.0), ("mercator", -1000.0)] {
        let dem = level_dem(metres);
        let style = format!(
            r##"{{"version": 8, "projection": {{"type": "{projection}"}},
              "center": [50.6, 40.3], "zoom": 11.7, "pitch": 80, "bearing": 90,
              "terrain": {{"source": "dem"}}, "sky": {{"sky-color": "#7fb3e6",
                "horizon-color": "#e8eef2", "fog-color": "#dfe7ee", "sky-horizon-blend": 0.6,
                "horizon-fog-blend": 0.7, "fog-ground-blend": 0.6}},
              "sources": {{"dem": {{"type": "raster-dem", "encoding": "terrarium",
                "tileSize": 256, "maxzoom": 2, "tiles": ["file://{}/{{z}}/{{x}}/{{y}}.png"]}}}},
              "layers": [{{"id": "background", "type": "background",
                "paint": {{"background-color": "#40a040"}}}}]}}"##,
            dem.display()
        );
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&style).expect("style");
        const WIDTH: u32 = 400;
        let image = map.snapshot(WIDTH, 300, 1.0).expect("snapshot");
        let clear: Vec<(u32, u32, u8)> = (image.pixels.as_chunks::<4>().0.iter().enumerate())
            .filter(|(_, p)| p[3] < 255)
            .map(|(i, p)| (i as u32 % WIDTH, i as u32 / WIDTH, p[3]))
            .collect();
        assert!(
            clear.is_empty(),
            "{projection} at {metres} m: {} pixels not opaque: {:?}",
            clear.len(),
            &clear[..clear.len().min(8)]
        );
    }
}

/// #162: on the sphere at street zooms, a line and a fill across a tile's
/// edge draw where they do on the plane, without a jog at the edge. The
/// points past a tile's edge (a stroke's side, a polygon's buffer) go on
/// from it by a step taken in f32, which as a difference of positions
/// millions of metres long was off by up to a metre, several pixels here.
#[test]
fn the_sphere_draws_across_tile_edges_as_the_plane_does() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    // A corner of four zoom-18 tiles near Tokyo, where GeoJSON's tiles end.
    let n = f64::from(1u32 << 18);
    let col = ((139.74 + 180.0) / 360.0 * n).round();
    let mercator_y = 35.685f64.to_radians().tan().asinh() / std::f64::consts::PI;
    let row = (n / 2.0 * (1.0 - mercator_y)).round();
    let lon = col / n * 360.0 - 180.0;
    let lat = (std::f64::consts::PI * (1.0 - 2.0 * row / n))
        .sinh()
        .atan()
        .to_degrees();
    let d = 0.0002;
    let draw = |projection: &str| {
        let style = format!(
            r##"{{"version": 8, "projection": {{"type": "{projection}"}},
              "center": [{lon}, {lat}], "zoom": 20, "bearing": 20,
              "sources": {{
                "line": {{"type": "geojson", "data": {{"type": "LineString",
                  "coordinates": [[{w}, {s}], [{e}, {n}]]}}}},
                "fill": {{"type": "geojson", "data": {{"type": "Polygon",
                  "coordinates": [[[{w}, {inside}], [{e}, {inside}], [{e}, {n}], [{w}, {inside}]]]}}}}}},
              "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
                {{"id": "fill", "type": "fill", "source": "fill", "paint": {{"fill-color": "#0000ff"}}}},
                {{"id": "line", "type": "line", "source": "line",
                  "paint": {{"line-color": "#000000", "line-width": 6}}}}]}}"##,
            w = lon - d,
            e = lon + d,
            s = lat - d,
            n = lat + d,
            inside = lat - d / 4.0,
        );
        let mut map = Map::new(&runtime, MapOptions::default());
        map.set_style_json(&style).expect("style");
        map.snapshot(256, 256, 1.0).expect("snapshot").pixels
    };
    let (plane, sphere) = (draw("mercator"), draw("vertical-perspective"));
    let (plane, sphere) = (plane.as_chunks::<4>().0, sphere.as_chunks::<4>().0);
    let apart: Vec<usize> = (plane.iter().zip(sphere))
        .enumerate()
        .filter(|(_, (a, b))| a.iter().zip(b.iter()).any(|(a, b)| a.abs_diff(*b) > 48))
        .map(|(i, _)| i)
        .collect();
    // A jog moves the stroke and the fill's edge along the tile's edge:
    // runs of pixels, hundreds at #162's metre. A lone pixel apart is a
    // rasterizer tie on an edge both draw alike: on RADV the fill's
    // outline pixels around one agree to a level (their positions to
    // under 1/100 px), and the sphere's f32 rounds the tie the other way
    // (#172).
    let lone = |&i: &usize| {
        let (x, y) = ((i % 256) as i64, (i / 256) as i64);
        !apart.iter().any(|&j| {
            let (u, v) = ((j % 256) as i64, (j / 256) as i64);
            j != i && (u - x).abs() <= 1 && (v - y).abs() <= 1
        })
    };
    let at = |i: usize| (i % 256, i / 256);
    assert!(
        apart.len() <= 4 && apart.iter().all(lone),
        "{} pixels apart: {:?}",
        apart.len(),
        apart.iter().take(8).map(|&i| at(i)).collect::<Vec<_>>()
    );
}

fn pixel_of(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let at = ((y * width + x) * 4) as usize;
    pixels[at..at + 4].try_into().unwrap()
}

/// A glyph range that failed for a reason that may pass (its server
/// down) is fetched again when failed resources are retried, as a still
/// does first, and its labels lay out with it: the map then draws as one
/// that had the range from the start.
#[test]
fn failed_glyph_ranges_are_fetched_again() {
    use std::io::{Read, Write};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // The server is down, however often the map retries, until it's up;
    // then it answers.
    let range = std::fs::read(common::data().join("glyphs/Noto Sans Regular/0-255.pbf")).unwrap();
    let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = server.local_addr().unwrap();
    let (up, answered) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicUsize::new(0)),
    );
    let (serving, counted) = (Arc::clone(&up), Arc::clone(&answered));
    std::thread::spawn(move || {
        for stream in server.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut request = [0; 2048];
            let _ = stream.read(&mut request);
            let (status, body) = match serving.load(Ordering::SeqCst) {
                false => ("503 Service Unavailable", &[][..]),
                true => {
                    counted.fetch_add(1, Ordering::SeqCst);
                    ("200 OK", &range[..])
                }
            };
            let head = format!(
                "HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body);
        }
    });
    let style = format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 2,
        "glyphs": "http://{address}/{{fontstack}}/{{range}}.pbf",
        "sources": {{"p": {{"type": "geojson", "data": {{"type": "Feature", "properties": {{}},
            "geometry": {{"type": "Point", "coordinates": [0, 0]}}}}}}}},
        "layers": [
          {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "t", "type": "symbol", "source": "p",
           "layout": {{"text-field": "Orbis", "text-font": ["Noto Sans Regular"], "text-size": 30}}}}]}}"##
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let still = |map: &mut Map| map.snapshot(200, 100, 1.0).expect("snapshot").pixels;
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    let without = still(&mut map);
    up.store(true, Ordering::SeqCst);
    let retried = still(&mut map);
    assert_eq!(answered.load(Ordering::SeqCst), 1, "fetched again once");
    let mut fresh = Map::new(&runtime, MapOptions::default());
    fresh.set_style_json(&style).unwrap();
    let expected = still(&mut fresh);
    assert!(
        without != expected,
        "without its range the label draws otherwise"
    );
    assert!(
        retried == expected,
        "the retried range draws as a fresh map's"
    );
}

/// Two symbol layers that lay out alike (a label drawn twice, in two
/// colours) show exactly the same labels: one bucket, placed for both by
/// the first, after a pan and after the second leaves the group and joins
/// it again.
#[test]
fn layers_that_lay_out_alike_show_the_same_labels() {
    let points: Vec<String> = (0..24)
        .map(|i| {
            let (lon, lat) = (f64::from(i % 6) * 3.0 - 7.5, f64::from(i / 6) * 2.0 - 3.0);
            format!(
                r#"{{"type":"Feature","id":{i},"properties":{{"name":"Label {i}"}},
                  "geometry":{{"type":"Point","coordinates":[{lon},{lat}]}}}}"#
            )
        })
        .collect();
    let layout =
        r#"{"text-field": ["get", "name"], "text-font": ["Noto Sans Regular"], "text-size": 14}"#;
    let style = format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 3,
        "glyphs": "file://{}/glyphs/{{fontstack}}/{{range}}.pbf",
        "sources": {{"p": {{"type": "geojson",
            "data": {{"type": "FeatureCollection", "features": [{}]}}}}}},
        "layers": [
          {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "a", "type": "symbol", "source": "p", "layout": {layout},
            "paint": {{"text-color": "#ff0000"}}}},
          {{"id": "b", "type": "symbol", "source": "p", "layout": {layout},
            "paint": {{"text-color": "#0000ff", "text-translate": [1, 1]}}}}]}}"##,
        common::data().display(),
        points.join(",")
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, 256, 256, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).expect("style");
    // Per layer, the features its labels show.
    let mut shown = |map: &mut Map| {
        settle(map, &mut target);
        let all = map.query_rendered(glam::DVec2::ZERO, glam::DVec2::splat(256.0));
        let doc = json::parse(&answer(map, &mut target, all).unwrap()).unwrap();
        let mut by_layer = [Vec::new(), Vec::new()];
        for feature in doc.as_array().unwrap() {
            let layer = feature.get("layer").and_then(|l| l.get("id"));
            let at = match layer.and_then(|l| l.as_str()) {
                Some("a") => 0,
                Some("b") => 1,
                other => panic!("{other:?}"),
            };
            by_layer[at].push(feature.get("id").unwrap().as_f64().unwrap() as u32);
        }
        for ids in &mut by_layer {
            ids.sort_unstable();
        }
        by_layer
    };
    let same = |[a, b]: [Vec<u32>; 2], when: &str| {
        assert!(
            !a.is_empty() && a.len() < points.len(),
            "{when}: some labels collide"
        );
        assert_eq!(a, b, "{when}");
    };
    same(shown(&mut map), "at first");
    map.jump_to(&CameraTarget {
        center: Some(orb_api::Geodetic::new(4.0, 1.0)),
        ..CameraTarget::default()
    });
    let panned = shown(&mut map);
    same(panned.clone(), "after a pan");
    // Laid out otherwise, each layer places its own labels: `b`'s all
    // show and leave `a`'s as they were; laid out alike again, `b` shows
    // `a`'s.
    let apart = r#"[["setLayoutProperty", "b", "text-allow-overlap", true],
        ["setLayoutProperty", "b", "text-ignore-placement", true]]"#;
    map.edit(&commands(apart)).expect("edit");
    let [a, b] = shown(&mut map);
    assert_eq!(a, panned[0], "apart, `a` places as before");
    assert_eq!(b.len(), points.len(), "apart, every label of `b` shows");
    let together = r#"[["setLayoutProperty", "b", "text-allow-overlap", null],
        ["setLayoutProperty", "b", "text-ignore-placement", null]]"#;
    map.edit(&commands(together)).expect("edit");
    same(shown(&mut map), "grouped again");
    // Hiding a layer leaves the group as it was: `b`'s labels, hidden,
    // don't collide with `a`'s.
    let visibility = |v: &str| format!(r#"[["setLayoutProperty", "b", "visibility", "{v}"]]"#);
    map.edit(&commands(&visibility("none"))).expect("edit");
    assert_eq!(shown(&mut map)[0], panned[0], "with `b` hidden");
    map.edit(&commands(&visibility("visible"))).expect("edit");
    same(shown(&mut map), "shown again");
    // The other layer leaves the group; `a` places as before.
    map.edit(&commands(r#"[["removeLayer", "b"]]"#))
        .expect("edit");
    let [a, b] = shown(&mut map);
    assert_eq!(a, panned[0], "with `b` gone, `a` places as before");
    assert!(b.is_empty());
}

/// A map drawn upside down draws the picture it draws the right way up,
/// mirrored top to bottom: a pitched, turned box shows the walls and roof
/// it shows upright, not the faces culled away from the camera. Turned
/// back, it draws upright again.
#[test]
fn a_flipped_map_draws_its_upright_picture_upside_down() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let style = r##"{"version": 8, "center": [0, 0], "zoom": 15, "pitch": 60, "bearing": 30,
      "sources": {"box": {"type": "geojson", "data": {"type": "Polygon",
        "coordinates": [[[-0.001, -0.001], [0.001, -0.001], [0.001, 0.001], [-0.001, 0.001], [-0.001, -0.001]]]}}},
      "layers": [
        {"id": "background", "type": "background", "paint": {"background-color": "#ffffff"}},
        {"id": "box", "type": "fill-extrusion", "source": "box",
         "paint": {"fill-extrusion-color": "#d02060", "fill-extrusion-height": 200}}
      ]}"##;
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).expect("style");
    let (width, height) = (96, 64);
    let mut draw = |flipped: bool| {
        map.set_flip_y(flipped);
        map.snapshot(width, height, 1.0).expect("snapshot").pixels
    };
    let upright = draw(false);
    let upside_down = draw(true);
    let turned_over = |pixels: &[u8]| -> Vec<u8> {
        let row = width as usize * 4;
        pixels.chunks_exact(row).rev().flatten().copied().collect()
    };
    assert_ne!(
        upright,
        turned_over(&upright),
        "the box looks the same upside down"
    );
    assert!(
        upright == turned_over(&upside_down),
        "upside down isn't the upright picture turned over"
    );
    assert!(draw(false) == upright, "turned back, the map isn't upright");
}
