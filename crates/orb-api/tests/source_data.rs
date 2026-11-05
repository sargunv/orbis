//! A GeoJSON source's data replaced every frame (`setSourceData`, GL JS's
//! `setData`) never leaves a frame without its features: the drawn tiles
//! stay until the new data's replace them.

use orb_api::{Map, MapOptions, Runtime, RuntimeOptions, Target};

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
        map.render(&mut target).unwrap();
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
        map.edit(&format!(r#"[["setSourceData", "p", {}]]"#, point(last)))
            .unwrap();
        // A frame's time: the jobs lay the data out between frames.
        std::thread::sleep(std::time::Duration::from_millis(16));
        map.render(&mut target).unwrap();
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
