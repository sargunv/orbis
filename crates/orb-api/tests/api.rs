//! The binding scenario through the Rust API: a style, a
//! frame, a query, an edit, a style image, camera moves, a gesture and
//! idle, into a texture target.

mod common;

use std::time::Duration;

use orb_api::{
    Button, CameraEvent, CameraTarget, EaseOptions, Error, Event, Input, Map, MapEvent, MapOptions,
    Modifiers, Op, PointerKind, Runtime, RuntimeOptions, StyleImage, Target, Transaction, json,
};

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

/// Milliseconds on the test's clock, which frames and input share.
fn now() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f64()
        * 1000.0
}

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
    let started = std::time::Instant::now();
    let features = loop {
        if let Some(Event::Query { id: done, result }) = map.poll_event() {
            assert_eq!(done, id);
            break result.expect("query");
        }
        assert!(started.elapsed().as_secs() < 10, "no query result");
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    assert!(features.contains(r#""id":"box""#), "{features}");
    assert!(
        features.starts_with('[') && features.contains(r#""id":7"#),
        "{features}"
    );
    assert!(map.needs_frame(), "the result woke the map");
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
        .render_until_idle(&mut target, Duration::from_secs(30))
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
    assert!(!map.center_clamped_to_ground());
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
        if map.needs_frame() {
            let time = 1e6 + f64::from(frames) * 1000.0 / 60.0;
            let report = map.render(&mut target, time).unwrap().unwrap();
            frames += 1;
            for event in std::iter::from_fn(|| map.poll_event()) {
                match event {
                    Event::Map(MapEvent::Idle) => {
                        idles += 1;
                        assert!(report.idle, "idle on a frame with work left");
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
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(started.elapsed().as_secs() < 30, "the map never idled");
    }
    assert_eq!(
        arrived,
        Some(25),
        "frame 0 at the start, frame 24 at 400 ms"
    );
    assert_eq!(idles, 1);
    assert!((map.zoom() - 1.0).abs() < 1e-9);
}

#[test]
fn texture_targets_report_size_and_read_back() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mut target = Target::texture(&runtime, 10, 6, 2.0);
    assert_eq!(target.size(), (10, 6));
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

    // The fetch fails now, and its completion finds no callback to call.
    drop(connection);
    drop(listener);
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    drop(runtime);
}

/// A frame borrows the runtime's device once, for the target and the map,
/// and lets it go before the pipeline cache is saved on going idle: frames
/// to idle with a state directory keep the cache in its `pipelines/`
/// where the driver has one.
/// Window targets take this path too.
#[test]
fn frames_to_idle_save_the_pipeline_cache() {
    let dir = std::env::temp_dir().join(format!("orb-api-state-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let runtime = Runtime::new(&RuntimeOptions {
        state_dir: Some(dir.clone()),
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
    let _ = std::fs::remove_dir_all(dir);
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

/// style-spec#1889: behind a globe, the backdrop with stars on it, which
/// turn with the globe. Off, the style's sky draws as it would without
/// them: nothing behind the globe from orbit.
#[test]
fn stars_and_backdrop_draw_behind_the_globe_with_their_preview() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let style = r##"{"version": 8, "projection": {"type": "globe"}, "center": [0, 0], "zoom": 0,
      "sky": {"dev.sargunv.orbis:backdrop-color": "#000020", "dev.sargunv.orbis:star-opacity": 1, "atmosphere-blend": 0},
      "sources": {}, "layers": [{"id": "background", "type": "background",
        "paint": {"background-color": "#808080"}}]}"##;
    let draw = |previews, lon| {
        let mut map = Map::new(
            &runtime,
            MapOptions {
                previews,
                ..MapOptions::default()
            },
        );
        map.set_style_json(style).expect("style");
        map.jump_to(&CameraTarget {
            center: Some(orb_api::Geodetic::new(lon, 0.0)),
            ..CameraTarget::default()
        });
        map.snapshot(WIDE, WIDE, 1.0).expect("snapshot").pixels
    };
    // A corner, off the globe, and the centre.
    const WIDE: u32 = 256;
    let at = |pixels: &[u8], x: u32, y: u32| -> [u8; 4] {
        let i = ((y * WIDE + x) * 4) as usize;
        pixels[i..i + 4].try_into().unwrap()
    };
    let corners = |pixels: &[u8]| -> Vec<[u8; 4]> {
        (0..40)
            .flat_map(|y| (0..40).map(move |x| (x, y)))
            .map(|(x, y)| at(pixels, x, y))
            .collect()
    };
    let on = draw(orb_api::Previews::ALL, 0.0);
    let sky = corners(&on);
    let backdrop = sky.iter().filter(|&&p| p == [0, 0, 0x20, 255]).count();
    assert!(backdrop > sky.len() / 2, "{sky:?}");
    assert!(sky.iter().any(|p| p[0] > 60), "no stars: {sky:?}");
    assert_eq!(at(&on, WIDE / 2, WIDE / 2), [128, 128, 128, 255]);
    // The same view draws the same stars; a turned globe, others.
    assert_eq!(corners(&draw(orb_api::Previews::ALL, 0.0)), sky);
    assert_ne!(corners(&draw(orb_api::Previews::ALL, 40.0)), sky);
    let off = draw(orb_api::Previews::NONE, 0.0);
    assert!(corners(&off).iter().all(|p| p[3] == 0));
}

/// In linear light the globe's scattering adds to the sunlit surface
/// exactly, tinting it, rather than lifting it toward white as GL JS's
/// gamma-encoded addition does.
#[test]
fn linear_light_adds_the_scattering_to_the_surface() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let style = |space: &str| {
        format!(
            r##"{{"version": 8, "projection": {{"type": "globe"}}, "center": [0, 0], "zoom": 0,
              "metadata": {{"dev.sargunv.orbis:color-space": "{space}"}},
              "sky": {{"atmosphere-blend": 1}}, "light": {{"anchor": "map", "position": [1.15, 270, 60]}},
              "sources": {{}}, "layers": [{{"id": "background", "type": "background",
                "paint": {{"background-color": "#808080"}}}}]}}"##
        )
    };
    let draw = |space: &str| {
        let mut map = Map::new(
            &runtime,
            MapOptions {
                previews: orb_api::Previews::ALL,
                ..MapOptions::default()
            },
        );
        map.set_style_json(&style(space)).expect("style");
        map.snapshot(WIDE, WIDE, 1.0).expect("snapshot").pixels
    };
    const WIDE: u32 = 512;
    // Blue across the middle row, over the grey surface.
    let lift = |pixels: &[u8], x: u32| {
        let i = ((WIDE / 2 * WIDE + x) * 4) as usize;
        pixels[i + 2].saturating_sub(128)
    };
    let linear = draw("linear");
    let gl = draw("srgb");
    // Somewhere across the middle row the sun is up: GL JS's model lifts it
    // most there.
    let x = (WIDE * 3 / 8..WIDE * 5 / 8)
        .max_by_key(|&x| lift(&gl, x))
        .unwrap();
    let (gl, linear) = (lift(&gl, x), lift(&linear, x));
    assert!(gl > 20, "GL JS's model lifts the face by {gl}");
    assert!(
        linear > 0 && linear * 2 < gl,
        "{linear} against GL JS's {gl}"
    );
}

/// `dev.sargunv.orbis:ground-lighting` darkens a globe's night side and
/// moves nothing: the scattering is lit by the same sun with it off or on,
/// so the day side draws the same, and the globe's outermost pixels on
/// the night side are as dark as the ground just inside them.
#[test]
fn ground_lighting_darkens_the_night_side_to_its_limb() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    // The sun 30° up over null island, to its west: night east of 30° E.
    let style = |lighting: f32| {
        format!(
            r##"{{"version": 8, "projection": {{"type": "globe"}}, "center": [0, 0], "zoom": 0,
              "sky": {{"atmosphere-blend": 1}},
              "light": {{"anchor": "map", "position": [1.15, 270, 60],
                "dev.sargunv.orbis:ground-lighting": {lighting},
                "dev.sargunv.orbis:ambient-intensity": 0.25}},
              "sources": {{}}, "layers": [{{"id": "background", "type": "background",
                "paint": {{"background-color": "#808080"}}}}]}}"##
        )
    };
    let draw = |lighting| {
        let mut map = Map::new(
            &runtime,
            MapOptions {
                previews: orb_api::Previews::ALL,
                ..MapOptions::default()
            },
        );
        map.set_style_json(&style(lighting)).expect("style");
        map.snapshot(WIDE, WIDE, 1.0).expect("snapshot").pixels
    };
    const WIDE: u32 = 256;
    let at = |pixels: &[u8], x: u32| pixel_of(pixels, WIDE, x, WIDE / 2);
    let (off, on) = (draw(0.0), draw(1.0));
    // The day side, left of centre, haze and all.
    for x in [WIDE * 3 / 8, WIDE / 2 - 8] {
        let (a, b) = (at(&off, x), at(&on, x));
        assert!(
            a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 1),
            "day side at {x}: {a:?} off, {b:?} on"
        );
    }
    // The night side's limb: the rightmost pixel the globe covers, against
    // the ground a few pixels inside it.
    let edge = (0..WIDE).rev().find(|&x| at(&on, x)[3] == 255).unwrap();
    let (rim, inside) = (at(&on, edge), at(&on, edge - 4));
    assert!(inside[0] < 100, "the night side is lit: {inside:?}");
    assert!(
        rim[..3].iter().zip(inside).all(|(r, i)| *r <= i + 4),
        "the limb at {edge} is {rim:?}, the ground inside it {inside:?}"
    );
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

/// `dev.sargunv.orbis:datetime` puts the night side where it is at that
/// moment: at midnight UTC null island is dark and the antimeridian lit,
/// at noon the other way round, wherever `position` says the sun is; a
/// viewport-anchored light doesn't read it.
#[test]
fn a_datetime_puts_the_night_side_where_it_is() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let draw = |anchor: &str, time: &str, lon: f64| {
        let style = format!(
            r##"{{"version": 8, "projection": {{"type": "globe"}}, "center": [{lon}, 0], "zoom": 1,
              "light": {{"anchor": "{anchor}", "position": [1.15, 0, 0],
                "dev.sargunv.orbis:ground-lighting": 1, "dev.sargunv.orbis:ambient-intensity": 0.25,
                "dev.sargunv.orbis:datetime": "{time}"}},
              "sources": {{}}, "layers": [{{"id": "background", "type": "background",
                "paint": {{"background-color": "#808080"}}}}]}}"##
        );
        let mut map = Map::new(
            &runtime,
            MapOptions {
                previews: orb_api::Previews::ALL,
                ..MapOptions::default()
            },
        );
        map.set_style_json(&style).expect("style");
        let image = map.snapshot(WIDE, WIDE, 1.0).expect("snapshot");
        pixel_of(&image.pixels, WIDE, WIDE / 2, WIDE / 2)[0]
    };
    const WIDE: u32 = 128;
    let (day, night) = (0x80, 0x80 / 2);
    for (time, lit, dark) in [
        ("2026-06-21T00:00:00Z", 180.0, 0.0),
        ("2026-06-21T12:00:00Z", 0.0, 180.0),
        // The same moment, written in Tokyo's time.
        ("2026-06-21T21:00:00+09:00", 0.0, 180.0),
    ] {
        assert_eq!(draw("map", time, lit), day, "{time} at {lit}");
        assert!(draw("map", time, dark) < night, "{time} at {dark}");
    }
    // A viewport light stands over the target, overhead here, at any time.
    assert_eq!(draw("viewport", "2026-06-21T00:00:00Z", 0.0), day);
}

/// `dev.sargunv.orbis:color-space`: a translucent fill mixes with what's
/// beneath in linear light, opaque colours draw as they are, and without
/// the preview the key warns and the map blends as GL JS does. A map
/// switched between the spaces draws as a fresh one in each.
#[test]
fn linear_light_mixes_translucent_layers() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let style = |space: &str| {
        format!(
            r##"{{"version": 8, "center": [-6, 0], "zoom": 2,
              "metadata": {{"dev.sargunv.orbis:color-space": "{space}"}},
              "sources": {{"b": {{"type": "geojson", "data": {{"type": "Feature", "properties": {{}},
                "geometry": {{"type": "Polygon",
                  "coordinates": [[[-12, -6], [-6, -6], [-6, 6], [-12, 6], [-12, -6]]]}}}}}}}},
              "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
                {{"id": "dark", "type": "fill", "source": "b", "paint": {{"fill-color": "#203040"}}}},
                {{"id": "half", "type": "fill", "source": "b",
                  "paint": {{"fill-color": "#ff0000", "fill-opacity": 0.5, "fill-translate": [16, 0]}}}}]}}"##
        )
    };
    let map = |previews| {
        Map::new(
            &runtime,
            MapOptions {
                previews,
                ..MapOptions::default()
            },
        )
    };
    let draw = |map: &mut Map| {
        let image = map.snapshot(SIZE, SIZE, 1.0).expect("snapshot");
        // The dark fill alone; the red over it; the red over white.
        let at = |x| pixel(&image.pixels, x, SIZE / 2);
        [at(6), at(24), at(40)]
    };
    let close =
        |got: [u8; 4], expected: [u8; 4]| got.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 2);
    // Half red over white: linear light (1, 0.5, 0.5) encodes to 188.
    let linear = [
        [0x20, 0x30, 0x40, 255],
        [189, 32, 44, 255],
        [255, 188, 188, 255],
    ];
    let srgb = [
        [0x20, 0x30, 0x40, 255],
        [144, 24, 32, 255],
        [255, 128, 128, 255],
    ];
    let mut fresh = map(orb_api::Previews::ALL);
    fresh.set_style_json(&style("linear")).expect("style");
    let got = draw(&mut fresh);
    for (got, expected) in got.into_iter().zip(linear) {
        assert!(close(got, expected), "linear: {got:?}, not {expected:?}");
    }
    let mut off = map(orb_api::Previews::NONE);
    off.set_style_json(&style("linear")).expect("style");
    let got = draw(&mut off);
    for (got, expected) in got.into_iter().zip(srgb) {
        assert!(
            close(got, expected),
            "without the preview: {got:?}, not {expected:?}"
        );
    }
    // Switched each way, a map draws as a fresh one.
    let mut switched = map(orb_api::Previews::ALL);
    for (space, expected) in [("srgb", srgb), ("linear", linear), ("srgb", srgb)] {
        switched.set_style_json(&style(space)).expect("style");
        let got = draw(&mut switched);
        for (got, expected) in got.into_iter().zip(expected) {
            assert!(
                close(got, expected),
                "switched to {space}: {got:?}, not {expected:?}"
            );
        }
    }
    // No device was lost.
    while let Some(event) = switched.poll_event() {
        assert!(
            !matches!(event, Event::Map(MapEvent::Restored)),
            "a switch isn't a restore"
        );
    }
}

/// Each building takes its own opacity from its data, beyond the spec:
/// over black, a building at a quarter draws a quarter as bright as one at
/// full.
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

/// maplibre/maplibre#555's ambient light: with `light`'s ambient keys set,
/// a face the sun doesn't reach takes the ambient light rather than GL
/// JS's `1 − intensity` of the light's colour, and `intensity` is the
/// sun's alone; unset, extrusions draw as GL JS shades them.
#[test]
fn extrusions_take_the_ambient_light() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let draw = |ambient: &str, previews| {
        let style = format!(
            r##"{{"version": 8, "center": [0, 0], "zoom": 2, "sources": {{"b": {{"type": "geojson",
              "data": {{"type": "Polygon", "coordinates": [[[-10, -5], [10, -5], [10, 5], [-10, 5], [-10, -5]]]}}}}}},
              "light": {{"anchor": "map", "position": [1, 0, 90], "intensity": 0.4 {ambient}}},
              "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#000000"}}}},
                {{"id": "b", "type": "fill-extrusion", "source": "b", "paint": {{
                  "fill-extrusion-color": "#ffffff", "fill-extrusion-height": 1000}}}}]}}"##
        );
        let mut map = Map::new(
            &runtime,
            MapOptions {
                previews,
                ..MapOptions::default()
            },
        );
        map.set_style_json(&style).expect("style");
        let image = map.snapshot(SIZE, SIZE, 1.0).expect("snapshot");
        pixel(&image.pixels, SIZE / 2, SIZE / 2)
    };
    // The sun on the horizon: the roof has the ambient light alone, GL
    // JS's 0.6 of white, 1.03 times (its lift) is 158.
    let gl = draw("", orb_api::Previews::ALL);
    assert!(gl[..3].iter().all(|&c| c.abs_diff(158) <= 1), "{gl:?}");
    // The style's: 0.5 of red.
    let ambient = r##", "dev.sargunv.orbis:ambient-color": "#ff0000", "dev.sargunv.orbis:ambient-intensity": 0.5"##;
    let red = draw(ambient, orb_api::Previews::ALL);
    assert!(
        red[0].abs_diff(131) <= 1 && red[1] == 0 && red[2] == 0,
        "{red:?}"
    );
    // Without the previews the keys are ignored.
    assert_eq!(draw(ambient, orb_api::Previews::NONE), gl);
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

/// style-spec#1881: `["dev.sargunv.orbis:meters-per-pixel"]` is the ground resolution at the
/// camera's target, so a circle 200 km across on the ground draws twice as
/// wide at 60° north as on the equator, at the same zoom, and picks as
/// wide; a map moved there draws as one made there. Off, the radius is the
/// default.
#[test]
fn meters_per_pixel_follows_the_target_with_its_preview() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let point = |lat: f64| {
        format!(
            r#"{{"type": "Feature", "properties": {{}}, "geometry": {{"type": "Point", "coordinates": [0, {lat}]}}}}"#
        )
    };
    let style = format!(
        r##"{{"version": 8, "sources": {{"c": {{"type": "geojson",
          "data": {{"type": "FeatureCollection", "features": [{}, {}]}}}}}},
          "layers": [
            {{"id": "background", "type": "background", "paint": {{"background-color": "#000000"}}}},
            {{"id": "c", "type": "circle", "source": "c", "paint": {{"circle-color": "#ffffff",
              "circle-pitch-alignment": "map",
              "circle-radius": ["/", 200000, ["dev.sargunv.orbis:meters-per-pixel"]]}}}}]}}"##,
        point(0.0),
        point(60.0),
    );
    let mut target = Target::texture(&runtime, SIZE, SIZE, 1.0);
    let at = |lat: f64| CameraTarget {
        center: Some(orb_api::Geodetic::new(0.0, lat)),
        zoom: Some(2.0),
        ..CameraTarget::default()
    };
    let width = |map: &mut Map, target: &mut Target| {
        settle(map, target);
        let pixels = target.read_pixels().unwrap();
        let lit: Vec<u32> = (0..SIZE)
            .filter(|&x| pixel(&pixels, x, SIZE / 2)[0] > 128)
            .collect();
        lit.last().map_or(0, |last| last - lit[0] + 1)
    };
    let map = |previews| {
        let mut map = Map::new(
            &runtime,
            MapOptions {
                previews,
                ..MapOptions::default()
            },
        );
        map.set_style_json(&style).expect("style");
        map
    };
    let mut moved = map(orb_api::Previews::ALL);
    moved.jump_to(&at(0.0));
    let equator = width(&mut moved, &mut target);
    // 200 km over 19568 m a pixel at zoom 2, twice: 20 px.
    assert!(equator.abs_diff(20) <= 2, "{equator} px on the equator");
    moved.jump_to(&at(60.0));
    let north = width(&mut moved, &mut target);
    assert!(
        north.abs_diff(2 * equator) <= 2,
        "{north} px at 60°, {equator} on the equator"
    );
    let edge = f64::from(SIZE) / 2.0 + f64::from(north) / 2.0 - 2.0;
    let hit = moved.hit_test(glam::DVec2::new(edge, f64::from(SIZE) / 2.0));
    assert_eq!(hit.map(|h| h.layer).as_deref(), Some("c"), "picks as wide");
    let mut fresh = map(orb_api::Previews::ALL);
    fresh.jump_to(&at(60.0));
    assert_eq!(width(&mut fresh, &mut target), north, "fresh at 60°");

    let mut off = map(orb_api::Previews::NONE);
    off.jump_to(&at(60.0));
    let default = width(&mut off, &mut target);
    off.set_style_json(&style.replace(
        r#"["/", 200000, ["dev.sargunv.orbis:meters-per-pixel"]]"#,
        "5",
    ))
    .expect("style");
    assert_eq!(
        default,
        width(&mut off, &mut target),
        "the default radius, 5"
    );
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
    let mut map = Map::new(
        &runtime,
        MapOptions {
            previews: orb_api::Previews::ALL,
            ..MapOptions::default()
        },
    );
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
        let doc = json::parse(&query_answer(map, target, query)).unwrap();
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

/// The answer to query `id`, waiting for it.
fn answer(map: &mut Map, id: u64) -> String {
    let started = std::time::Instant::now();
    loop {
        match map.poll_event() {
            Some(Event::Query { id: done, result }) if done == id => {
                return result.expect("query");
            }
            Some(_) => {}
            None => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
        assert!(started.elapsed().as_secs() < 10, "no answer to query {id}");
    }
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
    assert_eq!(ids(answer(&mut map, all)), [0, 1, 2, 3, 4, 5]);
    let even = map.query_source("p", None, Some(r#"["get", "even"]"#));
    assert_eq!(ids(answer(&mut map, even)), [0, 2, 4]);
    // East of the meridian: points at 5, 15 and 25 degrees.
    let east = map.query_source(
        "p",
        None,
        Some(
            r#"["within", {"type":"Polygon","coordinates":[[[0,-10],[30,-10],[30,10],[0,10],[0,-10]]]}]"#,
        ),
    );
    assert_eq!(ids(answer(&mut map, east)), [3, 4, 5]);
    let missing = map.query_source("nope", None, None);
    let started = std::time::Instant::now();
    loop {
        if let Some(Event::Query { id, result }) = map.poll_event() {
            assert_eq!(id, missing);
            assert!(result.is_err());
            break;
        }
        assert!(started.elapsed().as_secs() < 10);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
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
    let doc = json::parse(&answer(&mut map, rendered)).unwrap();
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
    let leaves = json::parse(&answer(&mut map, leaves)).unwrap();
    assert_eq!(leaves.as_array().unwrap().len(), 2);
    let zoom = map.query_cluster("p", orb_api::ClusterQuery::ExpansionZoom(id));
    let zoom: f64 = answer(&mut map, zoom).parse().unwrap();
    assert!(zoom >= 1.0, "{zoom}");
    let children = map.query_cluster("p", orb_api::ClusterQuery::Children(id));
    let children = json::parse(&answer(&mut map, children)).unwrap();
    assert!(!children.as_array().unwrap().is_empty());
    // Not a cluster: an error.
    let wrong = map.query_cluster("p", orb_api::ClusterQuery::Children(3));
    let started = std::time::Instant::now();
    loop {
        if let Some(Event::Query { id, result }) = map.poll_event() {
            assert_eq!(id, wrong);
            assert!(result.is_err());
            break;
        }
        assert!(started.elapsed().as_secs() < 10);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
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
    let doc = json::parse(&answer(&mut map, all)).unwrap();
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
    let started = std::time::Instant::now();
    loop {
        map.render(target, now()).unwrap();
        while let Some(event) = map.poll_event() {
            if let Event::Map(MapEvent::TileRequest { source, tile }) = event {
                return (source, tile);
            }
        }
        assert!(started.elapsed().as_secs() < 10, "no tile request");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
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

/// The answer of `query` once it arrives.
fn query_answer(map: &mut Map, target: &mut Target, query: u64) -> String {
    let started = std::time::Instant::now();
    loop {
        map.render(target, now()).unwrap();
        while let Some(event) = map.poll_event() {
            if let Event::Query { id, result } = event
                && id == query
            {
                return result.expect("a query answer");
            }
        }
        assert!(started.elapsed().as_secs() < 10, "no query answer");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
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
    let found = query_answer(&mut map, &mut target, query);
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
    let dir = std::env::temp_dir().join(format!("orb-api-requests-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
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
    assert!(!runtime.online());
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
    let started = std::time::Instant::now();
    let refused = loop {
        map.render(&mut target, now()).unwrap();
        if let Some(message) = std::iter::from_fn(|| map.poll_event()).find_map(|e| match e {
            Event::Map(MapEvent::Error { message }) => Some(message),
            _ => None,
        }) {
            break message;
        }
        assert!(started.elapsed().as_secs() < 10, "no error");
        std::thread::sleep(std::time::Duration::from_millis(2));
    };
    assert!(refused.contains("refused"), "{refused}");
    std::fs::remove_dir_all(dir).unwrap();
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

/// A building `height` metres tall and 44 m square at the equator seen
/// from above, its light from the south 45° up, on white: the pixel
/// `metres` north of its middle (south for negative), with `light` and
/// `paint` added to its style.
fn beside_a_building(height: f64, light: &str, paint: &str, metres: &[f64]) -> Vec<[u8; 4]> {
    beside_a_building_on(None, height, light, paint, metres)
}

/// [`beside_a_building`] on `terrain`, if any (see [`a_building`]).
fn beside_a_building_on(
    terrain: Option<&dyn Fn(f64) -> f64>,
    height: f64,
    light: &str,
    paint: &str,
    metres: &[f64],
) -> Vec<[u8; 4]> {
    a_building(terrain, height, (light, paint), 0.0, |map, pixels| {
        (metres.iter())
            .map(|&m| {
                let p = map.project(north_of_building(m)).expect("in view");
                pixel_of(pixels, BUILDING_VIEW, p.x as u32, p.y as u32)
            })
            .collect()
    })
}

/// The side of [`a_building`]'s pictures, in pixels.
const BUILDING_VIEW: u32 = 256;

/// Where [`a_building`] stands: within one tile at every zoom, as the
/// origin isn't.
const BUILDING: (f64, f64) = (0.001, 0.001);

/// The point `metres` north of [`a_building`]'s middle.
fn north_of_building(metres: f64) -> orb_api::Geodetic {
    orb_api::Geodetic::new(BUILDING.0, BUILDING.1 + metres / 110_574.0)
}

/// `read` of a [`BUILDING_VIEW`]-pixel square picture of a building
/// `height` metres tall and 44 m square near the equator ([`BUILDING`]),
/// in the middle of the picture, its light from the
/// south 45° up, on white, with `light` and `paint` added to its style,
/// seen from the south at `pitch` degrees, and of its map. On `terrain`,
/// if any, its ground stands `terrain(north)` metres high, `north` metres
/// north of the building's middle.
fn a_building<T>(
    terrain: Option<&dyn Fn(f64) -> f64>,
    height: f64,
    (light, paint): (&str, &str),
    pitch: f64,
    read: impl FnOnce(&Map, &[u8]) -> T,
) -> T {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let dem = terrain.map(equator_dem);
    let (dem_source, terrain) = dem.as_ref().map_or((String::new(), String::new()), |d| {
        (
            format!(
                r#", "dem": {{"type": "raster-dem", "encoding": "terrarium", "tileSize": 256,
                  "maxzoom": {DEM_ZOOM}, "tiles": ["file://{}/{{z}}/{{x}}/{{y}}.png"]}}"#,
                d.0.display()
            ),
            r#""terrain": {"source": "dem"},"#.to_owned(),
        )
    });
    let (x, y) = BUILDING;
    let [w, s, e, n] = [x - 0.0002, y - 0.0002, x + 0.0002, y + 0.0002];
    let style = format!(
        r##"{{"version": 8, "center": [{x}, {y}], "zoom": 17, "pitch": {pitch}, {terrain}
          "light": {{"anchor": "map", "position": [1.15, 180, 45], {light}}},
          "sources": {{"b": {{"type": "geojson", "data": {{"type": "Polygon", "coordinates":
            [[[{w}, {s}], [{e}, {s}], [{e}, {n}], [{w}, {n}], [{w}, {s}]]]}}}}{dem_source}}},
          "layers": [
            {{"id": "background", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
            {{"id": "b", "type": "fill-extrusion", "source": "b", "paint": {{
              "fill-extrusion-color": "#ffffff", "fill-extrusion-height": {height} {paint}}}}}]}}"##
    );
    let mut map = Map::new(
        &runtime,
        MapOptions {
            previews: orb_api::Previews::ALL,
            ..MapOptions::default()
        },
    );
    map.set_style_json(&style).expect("style");
    let image = map
        .snapshot(BUILDING_VIEW, BUILDING_VIEW, 1.0)
        .expect("snapshot");
    read(&map, &image.pixels)
}

/// The deepest zoom of [`equator_dem`]'s tiles.
const DEM_ZOOM: u8 = 15;

/// A directory of DEM tiles, removed when the test is done with it.
struct DemDir(std::path::PathBuf);

impl Drop for DemDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Terrarium DEM tiles, zooms 0 to [`DEM_ZOOM`], of the ground within
/// 0.01° of the origin standing `metres(north)` high, `north` metres
/// north of [`BUILDING`], written to a fresh directory.
fn equator_dem(metres: &dyn Fn(f64) -> f64) -> DemDir {
    // One directory per call: tests run in parallel.
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "orb-api-dem-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    const SIZE: u32 = 256;
    for z in 0..=DEM_ZOOM {
        let n = f64::from(1u32 << z);
        let cell = |v: f64| ((v + 180.0) / 360.0 * n).floor() as u32;
        let (lo, hi) = (cell(-0.01), cell(0.01));
        for x in lo..=hi {
            for y in lo..=hi {
                let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
                for py in 0..SIZE {
                    for _ in 0..SIZE {
                        let fy = (f64::from(y) + (f64::from(py) + 0.5) / f64::from(SIZE)) / n;
                        let lat = (std::f64::consts::PI * (1.0 - 2.0 * fy))
                            .sinh()
                            .atan()
                            .to_degrees();
                        let north = (lat - BUILDING.1) * 110_574.0;
                        let v = ((metres(north) + 32768.0) * 256.0).round() as u32;
                        pixels.extend([(v >> 16) as u8, (v >> 8) as u8, v as u8, 255]);
                    }
                }
                let path = dir.join(format!("{z}/{x}/{y}.png"));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                let file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
                let mut encoder = png::Encoder::new(file, SIZE, SIZE);
                encoder.set_color(png::ColorType::Rgba);
                encoder.set_depth(png::BitDepth::Eight);
                (encoder.write_header().unwrap())
                    .write_image_data(&pixels)
                    .unwrap();
            }
        }
    }
    DemDir(dir)
}

/// Terrarium DEM tiles of the whole world, zooms 0 to 2, `metres` high
/// everywhere, written to a fresh directory.
fn level_dem(metres: f64) -> DemDir {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "orb-api-level-dem-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
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
    DemDir(dir)
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
            dem.0.display()
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

/// style-spec#1797: a building's shadow falls away from the light, as far
/// as its height times the tangent of the light's angle from the zenith,
/// darkening the ground by the shadow intensity; overlapping wall and roof
/// shadows darken it once. Off, there is none.
#[test]
fn shadows_fall_away_from_the_light() {
    // North of the building 10 m and 25 m (its shadow reaches 30 m), 40 m
    // (past it), and 10 m south.
    let at = [32.0, 47.0, 62.0, -32.0];
    let off = beside_a_building(30.0, r#""intensity": 0.5"#, "", &at);
    assert!(off.iter().all(|p| p[0] == 255), "{off:?}");
    let on = beside_a_building(
        30.0,
        r#""intensity": 0.5, "dev.sargunv.orbis:shadow-intensity": 0.6"#,
        "",
        &at,
    );
    for p in &on[..2] {
        assert!((p[0] as i32 - 102).abs() <= 2, "shadowed: {on:?}");
    }
    assert_eq!(on[2][0], 255, "past the shadow: {on:?}");
    assert_eq!(on[3][0], 255, "on the lit side: {on:?}");
}

/// A part raised off the ground lets light out by its underside, which has
/// no faces: its shadow there comes from the walls facing the light, swept
/// from its base to its top.
#[test]
fn a_raised_part_shadows_beneath_its_lit_side() {
    // A slab from 50 m to 60 m up, its shadow 28 m to 82 m north of its
    // centre: the near part of it (33 m, 36 m) falls from the light's side,
    // which the roof's shadow (38 m on) misses; then within it, and 10 m
    // south of the slab.
    let at = [33.0, 36.0, 60.0, -32.0];
    let on = beside_a_building(
        60.0,
        r#""intensity": 0.5, "dev.sargunv.orbis:shadow-intensity": 0.6"#,
        r#", "fill-extrusion-base": 50"#,
        &at,
    );
    for p in &on[..3] {
        assert!((p[0] as i32 - 102).abs() <= 2, "shadowed: {on:?}");
    }
    assert_eq!(on[3][0], 255, "on the lit side: {on:?}");
}

/// style-spec#1796: the ground darkens along the walls by the ground
/// intensity at their foot, fading out over the ground radius; the walls
/// darken toward the ground over theirs (a wall, from above, shows at the
/// building's edge in perspective).
#[test]
fn ambient_occlusion_darkens_the_ground_along_the_walls() {
    // South of the building 1 m, 4 m and 12 m out, with a radius of 8 m.
    let at = [-23.0, -26.0, -34.0];
    let paint = r#", "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-intensity": 0.8,
        "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-radius": 8"#;
    let on = beside_a_building(2.0, r#""intensity": 0.5"#, paint, &at);
    assert!(on[0][0] < 80, "at the foot: {on:?}");
    assert!(on[0][0] < on[1][0] && on[1][0] < 255, "fading out: {on:?}");
    assert_eq!(on[2][0], 255, "past the radius: {on:?}");
}

/// On terrain a shadow falls on the ground where the light meets it: on
/// a plateau as on flat ground, and on ground rising away from the light,
/// shorter. 30 m up, the roof's north edge 22 m north of the middle
/// stands 19 m over ground rising one in two, and lands as high on it,
/// 12.7 m further: 34.7 m north rather than 52 m.
#[test]
fn shadows_fall_on_the_terrain() {
    let light = r#""intensity": 0.5, "dev.sargunv.orbis:shadow-intensity": 0.6"#;
    let at = [32.0, 47.0, 62.0, -32.0];
    let plateau = beside_a_building_on(Some(&|_| 300.0), 30.0, light, "", &at);
    for p in &plateau[..2] {
        assert!((p[0] as i32 - 102).abs() <= 2, "shadowed: {plateau:?}");
    }
    assert_eq!(plateau[2][0], 255, "past the shadow: {plateau:?}");
    assert_eq!(plateau[3][0], 255, "on the lit side: {plateau:?}");
    let at = [30.0, 40.0, 47.0];
    let rising = beside_a_building_on(Some(&|north| 300.0 + north / 2.0), 30.0, light, "", &at);
    assert!(
        (rising[0][0] as i32 - 102).abs() <= 2,
        "shadowed: {rising:?}"
    );
    assert_eq!(rising[1][0], 255, "past the shadow: {rising:?}");
    assert_eq!(rising[2][0], 255, "past the shadow: {rising:?}");
}

/// On ground falling away from the light as steeply as the light falls, a
/// shadow would reach on until the ground rose again: it falls at most
/// twice as far as on flat ground. The roof's north edge, 22 m north of
/// the middle, 30 m up, lands 60 m on, 82 m north.
#[test]
fn shadows_down_a_steep_slope_reach_twice_as_far() {
    let light = r#""intensity": 0.5, "dev.sargunv.orbis:shadow-intensity": 0.6"#;
    let at = [75.0, 90.0];
    let falling = beside_a_building_on(Some(&|north| 300.0 - north), 30.0, light, "", &at);
    assert!(
        (falling[0][0] as i32 - 102).abs() <= 2,
        "shadowed: {falling:?}"
    );
    assert_eq!(falling[1][0], 255, "past the shadow: {falling:?}");
}

/// On terrain the ground darkens along the walls as on flat ground: at
/// their foot by the ground intensity, fading out over the radius, on a
/// plateau and on a slope. North of the building, where the slope rises
/// past its roof: south of it, the roof standing over the falling ground
/// would hide the foot from above.
#[test]
fn ambient_occlusion_darkens_the_terrain_along_the_walls() {
    let paint = r#", "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-intensity": 0.8,
        "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-radius": 8"#;
    let at = [23.0, 26.0, 34.0];
    let grounds: [&dyn Fn(f64) -> f64; 2] = [&|_| 300.0, &|north| 300.0 + north * 0.3];
    for ground in grounds {
        let on = beside_a_building_on(Some(ground), 2.0, r#""intensity": 0.5"#, paint, &at);
        assert!(on[0][0] < 80, "at the foot: {on:?}");
        assert!(on[0][0] < on[1][0] && on[1][0] < 255, "fading out: {on:?}");
        assert_eq!(on[2][0], 255, "past the radius: {on:?}");
    }
}

/// On terrain a wall darkens toward the ground at its foot, wherever that
/// is: over the wall radius above the ground under it, not under the
/// building's middle. On ground rising north one in three, the south wall
/// stands 6.6 m lower than the middle, and darkens over as many rows as
/// it does on a plateau.
#[test]
fn walls_darken_toward_the_terrain_they_stand_on() {
    let paint = r#", "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity": 0.8,
        "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-radius": 6"#;
    let light = r#""intensity": 0.5"#;
    // Rows of the south wall's middle, from its foot up, darker with the
    // occlusion than without.
    let darkened = |ground: &dyn Fn(f64) -> f64| {
        let foot = north_of_building(-22.1);
        let column = |paint| {
            a_building(Some(ground), 30.0, (light, paint), 60.0, |map, pixels| {
                let p = map.project(foot).expect("in view");
                (0..p.y as u32)
                    .rev()
                    .map(|y| pixel_of(pixels, BUILDING_VIEW, p.x as u32, y)[0])
                    .collect::<Vec<_>>()
            })
        };
        let (off, on) = (column(""), column(paint));
        let rows = off
            .iter()
            .zip(&on)
            .filter(|(a, b)| u16::from(**b) + 2 < u16::from(**a))
            .count();
        assert!(rows > 0, "no occlusion: {off:?} {on:?}");
        rows
    };
    let plateau = darkened(&|_| 300.0);
    let slope = darkened(&|north| 300.0 + north * 0.3);
    assert!(
        slope.abs_diff(plateau) <= 2,
        "{slope} rows on the slope, {plateau} on the plateau"
    );
}

/// style-spec#1047: a fill's corners round off by the distance along their
/// edges, at most a fifth of each, and its outline follows the arc.
#[test]
fn fill_corners_round_off_with_their_outline() {
    const VIEW: u32 = 256;
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let style = |layout: &str| {
        format!(
            r##"{{"version": 8, "center": [0, 0], "zoom": 17,
              "sources": {{"f": {{"type": "geojson", "data": {{"type": "Polygon", "coordinates":
                [[[-0.0002, -0.0002], [0.0002, -0.0002], [0.0002, 0.0002], [-0.0002, 0.0002], [-0.0002, -0.0002]]]}}}}}},
              "layers": [
                {{"id": "background", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
                {{"id": "f", "type": "fill", "source": "f", "layout": {{{layout}}},
                  "paint": {{"fill-color": "#000000", "fill-outline-color": "#ff0000"}}}}]}}"##
        )
    };
    let corner = |layout: &str| {
        let mut map = Map::new(
            &runtime,
            MapOptions {
                previews: orb_api::Previews::ALL,
                ..MapOptions::default()
            },
        );
        map.set_style_json(&style(layout)).expect("style");
        let image = map.snapshot(VIEW, VIEW, 1.0).expect("snapshot");
        // Just inside the north-east corner, and inside the arc beside it.
        [(0.000_19, 0.000_19), (0.000_19, 0.000_1)].map(|(lon, lat)| {
            let p = map.project(orb_api::Geodetic::new(lon, lat)).unwrap();
            pixel_of(&image.pixels, VIEW, p.x as u32, p.y as u32)
        })
    };
    assert_eq!(corner("")[0], [0, 0, 0, 255]);
    let [cut, kept] = corner(r#""dev.sargunv.orbis:fill-rounded-corner-distance": 8"#);
    assert_eq!(cut, [255, 255, 255, 255], "the corner is gone");
    assert_eq!(kept, [0, 0, 0, 255], "the edge stays");
    // The outline runs round the arc: red pixels about its middle, 3.3 m in
    // from the corner along the diagonal.
    let mut map = Map::new(
        &runtime,
        MapOptions {
            previews: orb_api::Previews::ALL,
            ..MapOptions::default()
        },
    );
    map.set_style_json(&style(
        r#""dev.sargunv.orbis:fill-rounded-corner-distance": 8"#,
    ))
    .expect("style");
    let image = map.snapshot(VIEW, VIEW, 1.0).expect("snapshot");
    let arc = 0.0002 - 3.3 / 110_574.0 / 2f64.sqrt();
    let mid = map.project(orb_api::Geodetic::new(arc, arc)).unwrap();
    let red = (-4..=4)
        .flat_map(|y| (-4..=4).map(move |x| (x, y)))
        .map(|(x, y)| {
            let at = (mid.x as i32 + x, mid.y as i32 + y);
            pixel_of(&image.pixels, VIEW, at.0 as u32, at.1 as u32)
        })
        .filter(|p| p[0] > 150 && p[1] < 60)
        .count();
    assert!(red >= 4, "{red} outline pixels about the arc");
}
