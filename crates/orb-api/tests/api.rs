//! The binding scenario (plan §14.1) through the Rust API: a style, a
//! frame, a query, an edit, a style image, camera moves, a gesture and
//! idle, into a texture target.

use std::time::Duration;

use orb_api::{
    Button, CameraEvent, CameraTarget, EaseOptions, Error, Event, Input, Map, MapEvent, MapOptions,
    Modifiers, PointerKind, Runtime, RuntimeOptions, StyleImage, Target,
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

/// Draws until the map is idle, and returns the events.
fn settle(map: &mut Map, target: &mut Target) -> Vec<Event> {
    map.render_until_idle(target, Duration::from_secs(30))
        .expect("the map idles");
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

    let generation = map
        .edit(r##"[["setPaintProperty", "box", "fill-color", "#00ff00"]]"##)
        .expect("edit");
    assert!(map.needs_frame());
    let report = map
        .render_until_idle(&mut target, Duration::from_secs(30))
        .unwrap();
    assert!(report.generation >= generation);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(pixel(&pixels, SIZE / 2, SIZE / 2), [0, 255, 0, 255]);

    // A pattern the style lacks is reported; once added, it draws, and
    // new pixels replace it in place.
    map.edit(r#"[["setPaintProperty", "box", "fill-pattern", "dot"]]"#)
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
    map.edit(r##"[["setPaintProperty", "box", "fill-pattern", null]]"##)
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
        events.iter().any(|e| matches!(
            e,
            Event::Map(MapEvent::Click {
                button: Button::Primary,
                ..
            })
        )),
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
/// Window targets on this path: the Swift smoke test's `CAMetalLayer`.
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
    // Drivers with a pipeline cache (Vulkan, GLES) keep it there; Metal
    // caches pipelines itself and keeps none.
    if let Ok(entries) = std::fs::read_dir(dir.join("pipelines")) {
        assert!(
            entries.count() > 0,
            "a pipeline cache was started and not written"
        );
    }
    drop((map, target, runtime));
    std::fs::remove_dir_all(dir).unwrap();
}

/// A camera given at creation is the map's, at any zoom: the style's own
/// (north up and flat) doesn't replace it, and the picture shows it. Two
/// ground bands lie as far north of the target as south: pitched, the far
/// one draws thinner; turned, their edges tilt.
#[test]
fn a_camera_given_keeps_its_pitch_and_bearing() {
    let runtime = Runtime::new(&RuntimeOptions::default()).expect("runtime");
    let mercator = *orb_api::TileMatrixSet::by_id("WebMercatorQuad")
        .unwrap()
        .projection();
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
