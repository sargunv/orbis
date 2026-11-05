//! Hits land where features draw: picking projects as the renderer does,
//! on the globe too.

use std::time::Duration;

use glam::DVec2;
use orb_api::{Map, MapOptions, Runtime, RuntimeOptions, Target};

fn settle(map: &mut Map, target: &mut Target) {
    map.render_until_idle(target, Duration::from_secs(30))
        .expect("the map idles");
}

/// The centre of the red pixels.
fn red_centre(target: &Target) -> DVec2 {
    let pixels = target.read_pixels().unwrap();
    let width = target.size().0 as usize;
    let (mut sum, mut n) = (DVec2::ZERO, 0.0);
    for (i, p) in pixels.chunks(4).enumerate() {
        if p[0] > 200 && p[1] < 60 && p[2] < 60 {
            sum += DVec2::new((i % width) as f64 + 0.5, (i / width) as f64 + 0.5);
            n += 1.0;
        }
    }
    assert!(n > 50.0, "{n} red pixels");
    sum / n
}

#[test]
fn hits_land_where_features_draw_on_the_globe() {
    let style = r##"{"version": 8, "center": [20, 10], "zoom": 1.2, "pitch": 30,
        "projection": {"type": "globe"},
        "sources": {"p": {"type": "geojson", "data": {"type": "Feature", "id": 3, "properties": {},
            "geometry": {"type": "Point", "coordinates": [70, 40]}}}},
        "layers": [
          {"id": "bg", "type": "background", "paint": {"background-color": "#ffffff"}},
          {"id": "c", "type": "circle", "source": "p",
           "paint": {"circle-radius": 9, "circle-color": "#ff0000", "circle-pitch-alignment": "viewport"}}]}"##;
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 400, 300, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).unwrap();
    settle(&mut map, &mut target);
    let drawn = red_centre(&target);
    let hit = map.hit_test(drawn).expect("the circle where it draws");
    assert_eq!((hit.layer.as_str(), hit.id.as_deref()), ("c", Some("3")));
    // Off the circle, nothing.
    assert!(map.hit_test(drawn + DVec2::new(30.0, 0.0)).is_none());
}

/// Terrarium DEM tiles of a 3000 m plateau, 16 pixels a side, zooms 0–8,
/// in a fresh directory; the source's JSON.
fn plateau() -> String {
    let dir = std::env::temp_dir().join(format!("orb-picking-dem-{}", std::process::id()));
    let v = ((3000.0f64 + 32768.0) * 256.0) as u32;
    let texel = [(v >> 16) as u8, (v >> 8) as u8, v as u8, 255];
    let pixels: Vec<u8> = std::iter::repeat_n(texel, 16 * 16).flatten().collect();
    for z in 0..=8u32 {
        // The tiles around (10°, 5°) at this zoom.
        let n = f64::from(1u32 << z);
        let (x, y) = (
            (190.0 / 360.0 * n) as u32,
            ((1.0 - (5f64.to_radians().tan().asinh()) / std::f64::consts::PI) / 2.0 * n) as u32,
        );
        for col in x.saturating_sub(2)..=(x + 2).min((1 << z) - 1) {
            for row in y.saturating_sub(2)..=(y + 2).min((1 << z) - 1) {
                let path = dir.join(format!("{z}/{col}/{row}.png"));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                let mut png = Vec::new();
                let mut encoder = png::Encoder::new(&mut png, 16, 16);
                encoder.set_color(png::ColorType::Rgba);
                encoder.set_depth(png::BitDepth::Eight);
                encoder
                    .write_header()
                    .unwrap()
                    .write_image_data(&pixels)
                    .unwrap();
                std::fs::write(path, png).unwrap();
            }
        }
    }
    format!(
        r#"{{"type": "raster-dem", "encoding": "terrarium", "tileSize": 16, "maxzoom": 8,
            "tiles": ["file://{}/{{z}}/{{x}}/{{y}}.png"]}}"#,
        dir.display()
    )
}

#[test]
fn hits_land_where_features_draw_on_terrain() {
    let style = format!(
        r##"{{"version": 8, "center": [10, 5], "zoom": 7, "pitch": 60,
        "terrain": {{"source": "dem", "exaggeration": 2}},
        "sources": {{"dem": {}, "p": {{"type": "geojson", "data": {{"type": "Feature", "id": 4, "properties": {{}},
            "geometry": {{"type": "Point", "coordinates": [10.05, 5.05]}}}}}}}},
        "layers": [
          {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "c", "type": "circle", "source": "p",
           "paint": {{"circle-radius": 8, "circle-color": "#ff0000"}}}}]}}"##,
        plateau()
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 400, 300, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    settle(&mut map, &mut target);
    let drawn = red_centre(&target);
    let hit = map
        .hit_test(drawn)
        .expect("the circle where it draws, on the plateau");
    assert_eq!((hit.layer.as_str(), hit.id.as_deref()), ("c", Some("4")));
    assert!(map.hit_test(drawn + DVec2::new(0.0, 30.0)).is_none());
}
