//! The binding scenario (plan §14.1) through the Rust API: a style, a
//! frame, a query, an edit, a gesture and idle, into a texture target.

use orb_api::{
    Button, Event, Input, Map, MapEvent, MapOptions, Modifiers, PointerKind, Runtime,
    RuntimeOptions, Target,
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

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i..i + 4].try_into().unwrap()
}

/// Draws frames until the map needs none, and returns the events.
fn settle(map: &mut Map, target: &mut Target) -> Vec<Event> {
    let started = std::time::Instant::now();
    while map.needs_frame() {
        map.render(target).expect("render");
        assert!(started.elapsed().as_secs() < 30, "the map never idled");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
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
    assert!(matches!(maps[0], MapEvent::StyleLoad { .. }), "{maps:?}");
    assert!(maps.contains(&&MapEvent::Load), "{maps:?}");
    assert_eq!(maps.last(), Some(&&MapEvent::Idle), "{maps:?}");
    assert!(
        (map.camera().zoom() - 0.0).abs() < 1e-9,
        "the style's camera"
    );
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
    let mut shown = 0;
    while map.needs_frame() {
        let report = map.render(&mut target).unwrap().unwrap();
        shown = shown.max(report.generation);
    }
    assert!(shown >= generation);
    let pixels = target.read_pixels().unwrap();
    assert_eq!(pixel(&pixels, SIZE / 2, SIZE / 2), [0, 255, 0, 255]);

    // A drag moves the camera (and flings it: the moves arrive at once).
    let before = map.camera().target;
    let at = |x| glam::DVec2::new(x, 32.0);
    map.input(Input::PointerDown {
        id: 1,
        kind: PointerKind::Mouse,
        button: Button::Primary,
        position: at(20.0),
        modifiers: Modifiers::default(),
    });
    for x in [24.0, 30.0, 40.0] {
        map.input(Input::PointerMove {
            id: 1,
            position: at(x),
        });
    }
    map.input(Input::PointerUp {
        id: 1,
        position: at(40.0),
    });
    let events = settle(&mut map, &mut target);
    assert!((map.camera().target.lon - before.lon).abs() > 1.0);
    assert!(events.iter().any(|e| matches!(e, Event::Camera(_))));

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
        map.input(input);
    }
    let events = settle(&mut map, &mut target);
    assert!(
        events.iter().any(|e| matches!(
            e,
            Event::Click {
                button: Button::Primary,
                ..
            }
        )),
        "{events:?}"
    );
    assert!(!map.needs_frame());
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
