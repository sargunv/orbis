//! A GeoJSON source's data replaced every frame (`setSourceData`, GL JS's
//! `setData`) never leaves a frame without its features: the drawn tiles
//! stay until the new data's replace them.

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
    for _ in 0..100 {
        map.render(&mut target, 0.0).unwrap();
        if circle(&target).0 > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
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

/// A rendered query answers for what's drawn (#89): right after new data,
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
    map.snapshot(256, 256, 1.0).unwrap();
    for _ in 0..50 {
        map.render(&mut target, 0.0).unwrap();
        if map.hit_test(glam::DVec2::splat(128.0)).is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    map.edit(&source_data(&feature(40.0, "b"))).unwrap();
    // The jobs have the new data; the frame still draws the old.
    std::thread::sleep(std::time::Duration::from_millis(50));
    let centre = glam::DVec2::splat(128.0);
    let id = map.query_rendered(centre, centre);
    let started = std::time::Instant::now();
    let features = loop {
        if let Some(orb_api::Event::Query { id: got, result }) = map.poll_event()
            && got == id
        {
            break result.unwrap();
        }
        assert!(started.elapsed().as_secs() < 10, "no answer");
        std::thread::sleep(std::time::Duration::from_millis(1));
    };
    assert!(features.contains(r#""name":"a""#), "{features}");
}
