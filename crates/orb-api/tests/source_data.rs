//! A GeoJSON source's data replaced every frame (`setSourceData`, GL JS's
//! `setData`) never leaves a frame without its features: the drawn tiles
//! stay until the new data's replace them.

mod common;

use orb_api::{Map, MapOptions, Op, Runtime, RuntimeOptions, Target, Transaction, json};

/// GL JS `setData` on source `p`.
fn source_data(data: &str) -> Transaction {
    let mut tx = Transaction::new();
    tx.push(Op::SetSourceData {
        id: "p".into(),
        data: json::parse(data).expect("data"),
    });
    tx
}

fn point(lon: f64) -> String {
    format!(
        r#"{{"type":"Feature","properties":{{}},"geometry":{{"type":"Point","coordinates":[{lon},0]}}}}"#
    )
}

/// The red circle's pixel count and mean column.
fn circle(target: &Target) -> (usize, f64) {
    let pixels = target.read_pixels().unwrap();
    let width = target.size().0 as usize;
    let red: Vec<usize> = pixels
        .chunks(4)
        .enumerate()
        .filter(|(_, p)| p[0] > 200 && p[1] < 60 && p[2] < 60)
        .map(|(i, _)| i % width)
        .collect();
    let mean = red.iter().sum::<usize>() as f64 / red.len().max(1) as f64;
    (red.len(), mean)
}

#[test]
fn data_replaced_every_frame_never_blanks() {
    let style = format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 2,
            "sources": {{"p": {{"type": "geojson", "data": {}}}}},
            "layers": [
              {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {{"id": "c", "type": "circle", "source": "p",
                "paint": {{"circle-radius": 10, "circle-color": "#ff0000"}}}}]}}"##,
        point(0.0)
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 256, 256, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    map.snapshot(256, 256, 1.0).unwrap();
    // A continuous frame or two after the snapshot, until the circle shows.
    map.render(&mut target, 0.0).unwrap();
    common::until_at(&mut map, &mut target, 0.0, "circle", |_, target| {
        (circle(target).0 > 0).then_some(())
    });
    let (start, _) = circle(&target);
    assert!(start > 200, "{start} red pixels before the edits");
    let mut last = 0.0;
    for frame in 1..=60 {
        last = f64::from(frame) * 0.25;
        map.edit(&source_data(&point(last))).unwrap();
        // A frame's time: the jobs lay the data out between frames.
        std::thread::sleep(std::time::Duration::from_millis(16));
        map.render(&mut target, 0.0).unwrap();
        let (count, _) = circle(&target);
        assert!(count > start / 2, "frame {frame}: {count} red pixels");
    }
    // Settled, it shows the last data: 15° east at zoom 2 is about 85 px.
    let image = map.snapshot(256, 256, 1.0).unwrap();
    let red: Vec<usize> = image
        .pixels
        .chunks(4)
        .enumerate()
        .filter(|(_, p)| p[0] > 200 && p[1] < 60 && p[2] < 60)
        .map(|(i, _)| i % 256)
        .collect();
    let mean = red.iter().sum::<usize>() as f64 / red.len() as f64;
    let expected = 128.0 + last / 360.0 * 2048.0;
    assert!(
        (mean - expected).abs() < 2.0,
        "circle at {mean}, data at {expected}"
    );
}

/// A rendered query answers for what's drawn: right after new data,
/// before its tiles replace the drawn ones, a query at the old feature
/// gets the old feature, not the new data's at its index.
#[test]
fn queries_answer_from_the_drawn_data() {
    let feature = |lon: f64, name: &str| {
        format!(
            r#"{{"type":"Feature","properties":{{"name":"{name}"}},"geometry":{{"type":"Point","coordinates":[{lon},0]}}}}"#
        )
    };
    let style = format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 2,
            "sources": {{"p": {{"type": "geojson", "data": {}}}}},
            "layers": [{{"id": "c", "type": "circle", "source": "p", "paint": {{"circle-radius": 10}}}}]}}"##,
        feature(0.0, "a")
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 256, 256, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    map.render_until_idle(&mut target, common::DEADLINE)
        .unwrap();
    let centre = glam::DVec2::splat(128.0);
    assert!(map.hit_test(centre).is_some());
    let (wake, woken) = std::sync::mpsc::channel();
    map.set_wake(move || {
        let _ = wake.send(());
    });
    map.edit(&source_data(&feature(40.0, "b"))).unwrap();
    // The map was idle: its next wake is the jobs', with what they laid
    // out from the new data, which no frame draws yet.
    woken
        .recv_timeout(common::DEADLINE)
        .expect("the jobs' wake");
    let id = map.query_rendered(centre, centre);
    let features = loop {
        let answer = std::iter::from_fn(|| map.poll_event()).find_map(|event| match event {
            orb_api::Event::Query { id: got, result } if got == id => Some(result),
            _ => None,
        });
        if let Some(answer) = answer {
            break answer.unwrap();
        }
        woken
            .recv_timeout(common::DEADLINE)
            .expect("the answer's wake");
    };
    assert!(features.contains(r#""name":"a""#), "{features}");
}

/// New data lays a tile out again, and its new labels show at once, as GL
/// JS and MapLibre Native show a reloaded tile's (`justReloaded`): labels
/// fade in only on a tile's first load.
#[test]
fn new_data_shows_its_labels_at_once() {
    let glyphs = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../web/demo/data/glyphs")
        .canonicalize()
        .unwrap();
    let feature = |name: &str| {
        format!(
            r#"{{"type":"Feature","properties":{{"name":"{name}"}},"geometry":{{"type":"Point","coordinates":[0,0]}}}}"#
        )
    };
    let style = format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 3,
        "glyphs": "file://{}/{{fontstack}}/{{range}}.pbf",
        "sources": {{"p": {{"type": "geojson", "data": {}}}}},
        "layers": [{{"id": "t", "type": "symbol", "source": "p",
           "layout": {{"text-field": ["get", "name"], "text-font": ["Noto Sans Regular"]}}}}]}}"##,
        glyphs.display(),
        feature("before")
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 256, 256, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    map.render_until_idle(&mut target, common::DEADLINE)
        .unwrap();
    // Frames from here on are all at one time: nothing fades.
    let time = 1e9;
    map.render(&mut target, time).unwrap();
    assert_eq!(map.shown_labels().get("before"), Some(&1.0));
    map.edit(&source_data(&feature("after"))).unwrap();
    // Fading in from nothing, the label would stay hidden at this time.
    common::until_at(&mut map, &mut target, time, "new label", |map, _| {
        (map.shown_labels().get("after") == Some(&1.0)).then_some(())
    });
    let shown = map.shown_labels();
    assert_eq!(shown.get("after"), Some(&1.0), "{shown:?}");
    assert!(!shown.contains_key("before"), "{shown:?}");
}
