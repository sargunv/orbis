//! Hits land where features draw: picking projects as the renderer does,
//! on the globe too.

mod common;

use glam::DVec2;
use orb_api::{Map, MapOptions, Runtime, RuntimeOptions, Target};

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
    common::settle(&mut map, &mut target);
    let drawn = red_centre(&target);
    let hit = map.hit_test(drawn).expect("the circle where it draws");
    assert_eq!((hit.layer.as_str(), hit.id.as_deref()), ("c", Some("3")));
    // Off the circle, nothing.
    assert!(map.hit_test(drawn + DVec2::new(30.0, 0.0)).is_none());
}

/// Terrarium DEM tiles of a 3000 m plateau, 16 pixels a side, zooms 0–8,
/// in a fresh directory; the source's JSON, and the directory to keep
/// while it's read.
fn plateau() -> (String, common::Scratch) {
    let dir = common::Scratch::new("picking-dem");
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
    let source = format!(
        r#"{{"type": "raster-dem", "encoding": "terrarium", "tileSize": 16, "maxzoom": 8,
            "tiles": ["file://{}/{{z}}/{{x}}/{{y}}.png"]}}"#,
        dir.display()
    );
    (source, dir)
}

#[test]
fn hits_land_where_features_draw_on_terrain() {
    let (dem, _tiles) = plateau();
    let style = format!(
        r##"{{"version": 8, "center": [10, 5], "zoom": 7, "pitch": 60,
        "terrain": {{"source": "dem", "exaggeration": 2}},
        "sources": {{"dem": {}, "p": {{"type": "geojson", "data": {{"type": "Feature", "id": 4, "properties": {{}},
            "geometry": {{"type": "Point", "coordinates": [10.05, 5.05]}}}}}}}},
        "layers": [
          {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "c", "type": "circle", "source": "p",
           "paint": {{"circle-radius": 8, "circle-color": "#ff0000"}}}}]}}"##,
        dem
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 400, 300, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    common::settle(&mut map, &mut target);
    let drawn = red_centre(&target);
    let hit = map
        .hit_test(drawn)
        .expect("the circle where it draws, on the plateau");
    assert_eq!((hit.layer.as_str(), hit.id.as_deref()), ("c", Some("4")));
    assert!(map.hit_test(drawn + DVec2::new(0.0, 30.0)).is_none());
}

#[test]
fn labels_pick_where_their_variable_anchor_puts_them() {
    let glyphs = common::data().join("glyphs");
    let style = format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 3,
        "glyphs": "file://{}/{{fontstack}}/{{range}}.pbf",
        "sources": {{"p": {{"type": "geojson", "data": {{"type": "Feature", "id": 5, "properties": {{}},
            "geometry": {{"type": "Point", "coordinates": [0, 0]}}}}}}}},
        "layers": [
          {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "t", "type": "symbol", "source": "p",
           "layout": {{"text-field": "WWWWWWWWWW", "text-font": ["Noto Sans Regular"], "text-size": 20,
             "text-variable-anchor": ["left"]}}}}]}}"##,
        glyphs.display()
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 400, 300, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    common::settle(&mut map, &mut target);
    // Anchored on its left, the text runs east of the point (200, 150):
    // well past half its width.
    let hit = map
        .hit_test(DVec2::new(330.0, 150.0))
        .expect("the shifted label");
    assert_eq!((hit.layer.as_str(), hit.id.as_deref()), ("t", Some("5")));
    assert!(
        map.hit_test(DVec2::new(70.0, 150.0)).is_none(),
        "where it isn't"
    );
}

/// Every pixel a tall building draws picks it, across the edge of the
/// tiles that hold it, under pitch.
#[test]
fn buildings_pick_wherever_they_draw() {
    // Straddling the z13 tile edge at 0° longitude, a quarter tile wide.
    let style = r##"{"version": 8, "center": [0.0, 0.01], "zoom": 13.5, "pitch": 55, "bearing": 20,
        "sources": {"b": {"type": "geojson", "maxzoom": 13, "data": {"type": "Feature", "id": 6, "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[[-0.006, 0.004], [0.004, 0.004], [0.004, 0.012], [-0.006, 0.012], [-0.006, 0.004]]]}}}},
        "layers": [
          {"id": "bg", "type": "background", "paint": {"background-color": "#ffffff"}},
          {"id": "x", "type": "fill-extrusion", "source": "b",
           "paint": {"fill-extrusion-color": "#ff0000", "fill-extrusion-height": 400,
             "fill-extrusion-vertical-gradient": false}}]}"##;
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 300, 300, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(style).unwrap();
    common::settle(&mut map, &mut target);
    let pixels = target.read_pixels().unwrap();
    let (mut drawn, mut missed) = (0, 0);
    for (i, p) in pixels.chunks(4).enumerate() {
        // Lit walls darken; any strongly red pixel is the building.
        if p[0] > 120 && p[1] < 60 && p[2] < 60 {
            drawn += 1;
            let at = DVec2::new((i % 300) as f64 + 0.5, (i / 300) as f64 + 0.5);
            if map.hit_test(at).is_none_or(|h| h.layer != "x") {
                missed += 1;
            }
        }
    }
    assert!(drawn > 2000, "{drawn} pixels drawn");
    // Edge pixels round either way.
    assert!(
        missed * 100 < drawn,
        "{missed} of {drawn} drawn pixels miss"
    );
}

/// A symbol layer of points `[id, lon, lat]` with `layout`, 20 px labels from the demo's glyphs, drawn into a 400 × 300 map at zoom 3.
fn labels(points: &[(u32, f64, f64)], layout: &str) -> (Map, Target) {
    painted_labels(points, layout, "")
}

/// [`labels`] with `paint`.
fn painted_labels(points: &[(u32, f64, f64)], layout: &str, paint: &str) -> (Map, Target) {
    let glyphs = common::data().join("glyphs");
    let features: Vec<String> = points
        .iter()
        .map(|(id, lon, lat)| {
            format!(
                r#"{{"type": "Feature", "id": {id}, "properties": {{"k": {id}}},
                  "geometry": {{"type": "Point", "coordinates": [{lon}, {lat}]}}}}"#
            )
        })
        .collect();
    let style = format!(
        r##"{{"version": 8, "center": [0, 0], "zoom": 3,
        "glyphs": "file://{}/{{fontstack}}/{{range}}.pbf",
        "sources": {{"p": {{"type": "geojson", "data": {{"type": "FeatureCollection", "features": [{}]}}}}}},
        "layers": [
          {{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
          {{"id": "t", "type": "symbol", "source": "p", "paint": {{{paint}}},
           "layout": {{"text-font": ["Noto Sans Regular"], "text-size": 20, {layout}}}}}]}}"##,
        glyphs.display(),
        features.join(","),
    );
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let mut target = Target::texture(&runtime, 400, 300, 1.0);
    let mut map = Map::new(&runtime, MapOptions::default());
    map.set_style_json(&style).unwrap();
    common::settle(&mut map, &mut target);
    (map, target)
}

/// The id of the label at `(x, y)`, if one shows there.
fn label_at(map: &mut Map, x: f64, y: f64) -> Option<String> {
    map.hit_test(DVec2::new(x, y)).and_then(|h| h.id)
}

/// Beyond the spec, each feature takes its own `text-variable-anchor`:
/// one label runs east of its point, the other west of its own.
#[test]
fn labels_take_their_own_variable_anchors() {
    let anchors = r#""text-field": "WWWW", "text-variable-anchor": ["match", ["get", "k"], 1, ["literal", ["left"]], ["literal", ["right"]]]"#;
    // Zoom 3: 4096 px about the world; 10° is 114 px.
    let (mut map, _target) = labels(&[(1, 0.0, 6.0), (2, 0.0, -6.0)], anchors);
    let (east, west) = (200.0 + 50.0, 200.0 - 50.0);
    let (north, south) = (150.0 - 68.0, 150.0 + 68.0);
    assert_eq!(label_at(&mut map, east, north).as_deref(), Some("1"));
    assert_eq!(label_at(&mut map, west, north), None);
    assert_eq!(label_at(&mut map, west, south).as_deref(), Some("2"));
    assert_eq!(label_at(&mut map, east, south), None);
}

/// Beyond the spec, each feature takes its own overlap and
/// `text-line-height`: of two labels that collide, the one allowed to
/// overlap and ignoring placement shows beside the other, which it
/// doesn't block, and a two-line label set
/// three ems apart stands taller than one set at 1.2.
#[test]
fn labels_take_their_own_overlap_and_line_height() {
    // 2° apart, about 23 px: the labels collide.
    let points = [(1, -1.0, 0.0), (2, 1.0, 0.0)];
    let (mut map, _target) = labels(&points, r#""text-field": "WWWW""#);
    let shown = |map: &mut Map| {
        (140..260)
            .step_by(4)
            .filter_map(|x| label_at(map, f64::from(x), 150.0))
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert_eq!(shown(&mut map).len(), 1, "one hides the other");
    let allow = r#""text-field": "WWWW", "text-allow-overlap": ["==", ["get", "k"], 2],
        "text-ignore-placement": ["==", ["get", "k"], 2]"#;
    let (mut map, _target) = labels(&points, allow);
    assert_eq!(shown(&mut map).len(), 2, "the second shows over the first");

    let lines = r#""text-field": "WW WW", "text-max-width": 1, "text-allow-overlap": true,
        "text-line-height": ["match", ["get", "k"], 1, 3, 1.2]"#;
    let (mut map, _target) = labels(&[(1, -8.0, 0.0), (2, 8.0, 0.0)], lines);
    // Each label's two lines centre on its point (x 109 and 291 px).
    let tall = |map: &mut Map, x: f64| {
        (0..150u32)
            .filter(|&dy| label_at(map, x, 150.0 - f64::from(dy)).is_some())
            .max()
            .unwrap_or(0)
    };
    let (three, normal) = (tall(&mut map, 109.0), tall(&mut map, 291.0));
    assert!(
        three > normal + 15,
        "three ems apart, {three} px above the point; 1.2, {normal}"
    );
}

/// Beyond the spec, overlap follows the camera at each placement: two
/// colliding labels both show once the map tilts past 30°, and one hides
/// again when it levels, as a fresh map at each camera places them. It
/// does so too where the layer reads another setting per feature.
#[test]
fn label_overlap_follows_the_camera() {
    let points = [(1, -1.0, 0.0), (2, 1.0, 0.0)];
    let tilted = r#""text-field": "WWWW", "text-allow-overlap": [">", ["pitch"], 30],
        "text-ignore-placement": [">", ["pitch"], 30]"#;
    let shown = |map: &mut Map| {
        (140..260)
            .step_by(4)
            .filter_map(|x| label_at(map, f64::from(x), 150.0))
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    };
    let tilt = |map: &mut Map, target: &mut Target, pitch: f64| {
        map.jump_to(&orb_api::CameraTarget {
            pitch: Some(pitch),
            ..orb_api::CameraTarget::default()
        });
        common::settle(map, target);
    };
    let per_feature = format!(r#"{tilted}, "text-optional": ["==", ["get", "k"], 2]"#);
    for layout in [tilted, &per_feature] {
        let (mut map, mut target) = labels(&points, layout);
        assert_eq!(shown(&mut map), 1, "level, one hides the other: {layout}");
        tilt(&mut map, &mut target, 45.0);
        assert_eq!(shown(&mut map), 2, "tilted, both show: {layout}");
        tilt(&mut map, &mut target, 0.0);
        assert_eq!(shown(&mut map), 1, "level again: {layout}");
    }
}

/// Labels pick where their `text-translate` draws them, and beyond the
/// spec each takes its own: of two labels at one point, the one moved
/// 40 px down shows beside the other rather than hiding, and draws and
/// picks there.
#[test]
fn labels_take_their_own_translate() {
    let field = r#""text-field": "WWWW""#;
    let ids = |map: &mut Map| {
        (100..240)
            .step_by(4)
            .filter_map(|y| label_at(map, 200.0, f64::from(y)).map(|id| (id, y)))
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    // Dark pixels in `rows` near the middle.
    let ink = |target: &Target, rows: std::ops::Range<usize>| {
        let pixels = target.read_pixels().unwrap();
        let width = target.size().0 as usize;
        rows.flat_map(|y| (160..240).map(move |x| (y * width + x) * 4))
            .filter(|&i| pixels[i] < 128)
            .count()
    };
    let (mut map, _target) = labels(&[(1, 0.0, 0.0), (2, 0.0, 0.0)], field);
    assert_eq!(ids(&mut map).len(), 1, "at one point, one hides the other");

    let layer = r#""text-translate": [0, 40]"#;
    let (mut map, target) = painted_labels(&[(1, 0.0, 0.0)], field, layer);
    assert_eq!(label_at(&mut map, 200.0, 190.0).as_deref(), Some("1"));
    assert_eq!(label_at(&mut map, 200.0, 150.0), None);
    assert!(ink(&target, 180..200) > 20 && ink(&target, 140..160) == 0);

    let own = r#"["match", ["get", "k"], 2, ["literal", [0, 40]], ["literal", [0, 0]]]"#;
    let paint = format!(r#""text-translate": {own}"#);
    let (mut map, target) = painted_labels(&[(1, 0.0, 0.0), (2, 0.0, 0.0)], field, &paint);
    assert_eq!(label_at(&mut map, 200.0, 150.0).as_deref(), Some("1"));
    assert_eq!(label_at(&mut map, 200.0, 190.0).as_deref(), Some("2"));
    assert!(ink(&target, 180..200) > 20 && ink(&target, 140..160) > 20);

    // An edit to and from per-feature translates places the labels again.
    let (mut map, mut target) = labels(&[(1, 0.0, 0.0), (2, 0.0, 0.0)], field);
    let set = |map: &mut Map, target: &mut Target, value: &str| {
        let tx = format!(r#"[["setPaintProperty", "t", "text-translate", {value}]]"#);
        map.edit(&orb_api::Transaction::from_commands(&tx).unwrap())
            .unwrap();
        common::settle(map, target);
    };
    set(&mut map, &mut target, own);
    assert_eq!(ids(&mut map).len(), 2, "each its own way, both show");
    set(&mut map, &mut target, "[0, 0]");
    assert_eq!(ids(&mut map).len(), 1, "together again, one hides");
}

/// The centre of the pixels where a label picks, sampled every 2 px, and
/// of its dark pixels.
fn picked_and_drawn(map: &mut Map, target: &Target) -> (DVec2, DVec2) {
    let (mut picked, mut n) = (DVec2::ZERO, 0.0);
    for y in (0..300).step_by(2) {
        for x in (0..400).step_by(2) {
            let at = DVec2::new(f64::from(x), f64::from(y));
            if label_at(map, at.x, at.y).is_some() {
                picked += at;
                n += 1.0;
            }
        }
    }
    let pixels = target.read_pixels().unwrap();
    let (mut drawn, mut m) = (DVec2::ZERO, 0.0);
    for (i, p) in pixels.chunks(4).enumerate() {
        if p[0] < 128 {
            drawn += DVec2::new((i % 400) as f64 + 0.5, (i / 400) as f64 + 0.5);
            m += 1.0;
        }
    }
    assert!(n > 0.0 && m > 0.0, "{n} picked, {m} drawn");
    (picked / n, drawn / m)
}

/// Beyond the spec, a label's own translate blends between zooms and
/// turns with the map: at zoom 3.5, halfway to 200 px south, it draws and
/// picks 100 px from its point whatever the bearing and pitch.
#[test]
fn labels_own_translates_blend_by_zoom_and_turn_with_the_map() {
    let own = r#"["interpolate", ["linear"], ["zoom"], 3, ["literal", [0, 0]],
        4, ["match", ["get", "k"], 1, ["literal", [0, 200]], ["literal", [0, 0]]]]"#;
    let paint = format!(r#""text-translate": {own}"#);
    let (mut map, mut target) = painted_labels(&[(1, 0.0, 0.0)], r#""text-field": "WWWW""#, &paint);
    for (bearing, pitch) in [(0.0, 0.0), (45.0, 0.0), (45.0, 50.0), (300.0, 30.0)] {
        map.jump_to(&orb_api::CameraTarget {
            zoom: Some(3.5),
            bearing: Some(bearing),
            pitch: Some(pitch),
            ..orb_api::CameraTarget::default()
        });
        common::settle(&mut map, &mut target);
        let (picked, drawn) = picked_and_drawn(&mut map, &target);
        let at = format!("bearing {bearing}, pitch {pitch}");
        assert!(
            picked.distance(drawn) < 3.0,
            "{at}: picks at {picked}, draws at {drawn}"
        );
        if pitch == 0.0 {
            let away = drawn.distance(DVec2::new(200.0, 150.0));
            assert!((away - 100.0).abs() < 3.0, "{at}: {away} px from its point");
        }
    }
}

/// A label at the centre of a tilted map draws the same whatever the
/// bearing, and picks across the same width: the perspective ratio its
/// boxes scale by isn't foreshortened as a step east is once the bearing
/// turns east up the screen.
#[test]
fn tilted_labels_pick_across_their_width_at_any_bearing() {
    let (mut map, mut target) = labels(&[(1, 0.0, 0.0)], r#""text-field": "WWWW""#);
    let mut spans = Vec::new();
    for bearing in [0.0, 90.0, 135.0] {
        map.jump_to(&orb_api::CameraTarget {
            bearing: Some(bearing),
            pitch: Some(60.0),
            ..orb_api::CameraTarget::default()
        });
        common::settle(&mut map, &mut target);
        let pixels = target.read_pixels().unwrap();
        let dark: Vec<usize> = (0..400)
            .filter(|&x| (140..160).any(|y| pixels[(y * 400 + x) * 4] < 128))
            .collect();
        let picked: Vec<usize> = (0..400)
            .filter(|&x| label_at(&mut map, x as f64, 150.0).is_some())
            .collect();
        let span = |v: &[usize]| [v[0], v[v.len() - 1]];
        spans.push((bearing, span(&dark), span(&picked)));
    }
    let (_, drawn, picked) = spans[0];
    for &(bearing, d, p) in &spans[1..] {
        assert_eq!(d, drawn, "bearing {bearing} draws as bearing 0 does");
        assert!(
            (0..2).all(|i| p[i].abs_diff(picked[i]) <= 1),
            "bearing {bearing} picks across {p:?}, bearing 0 across {picked:?}"
        );
    }
}

/// Zoom-and-data sizes pick at the size they draw at between whole zooms,
/// below and above a GeoJSON source's maxzoom, as MapLibre Native's
/// `Query.QueryRenderedFeaturesCompositePaintZoom` (maplibre-native#4703)
/// asks: the size is 22 pixels at the camera's zoom and 4 at the tile's.
#[test]
fn composite_sizes_pick_at_the_cameras_zoom() {
    // (property, pixels below the feature that pick, and that don't)
    let cases = [
        ("line-width", 8.0, 14.0),
        ("line-gap-width", 14.0, 20.0),
        ("circle-radius", 15.0, 26.0),
        ("circle-stroke-width", 24.0, 32.0),
    ];
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    for zoom in [16.5f64, 20.5] {
        for (property, inside, outside) in cases {
            let trace = format!("{property} at zoom {zoom}");
            let circle = property.starts_with("circle-");
            let geometry = if circle {
                r#"{"type": "Point", "coordinates": [1.003, 1]}"#
            } else {
                r#"{"type": "LineString", "coordinates": [[0.993, 1], [1.013, 1]]}"#
            };
            let paint = match property {
                "circle-stroke-width" => r#""circle-radius": 6,"#,
                "line-gap-width" => r#""line-width": 6,"#,
                _ => "",
            };
            let (lower, upper) = (zoom.floor(), zoom.floor() + 1.0);
            let style = format!(
                r#"{{"version": 8, "center": [1.003, 1], "zoom": {zoom},
                "sources": {{"s": {{"type": "geojson", "data": {{"type": "Feature",
                    "properties": {{"small": 4, "large": 40}}, "geometry": {geometry}}}}}}},
                "layers": [{{"id": "l", "type": "{}", "source": "s", "paint": {{{paint}
                    "{property}": ["interpolate", ["linear"], ["zoom"],
                        {lower}, ["get", "small"], {upper}, ["get", "large"]]}}}}]}}"#,
                if circle { "circle" } else { "line" },
            );
            let mut target = Target::texture(&runtime, 256, 256, 1.0);
            let mut map = Map::new(&runtime, MapOptions::default());
            map.set_style_json(&style).unwrap();
            common::settle(&mut map, &mut target);
            let centre = map
                .project(orb_api::Geodetic::new(1.003, 1.0))
                .expect("the feature in view");
            let pixels = target.read_pixels().unwrap();
            let alpha = |p: DVec2| pixels[4 * (p.y as usize * 256 + p.x as usize) + 3];
            let (inside, outside) = (
                centre + DVec2::new(0.0, inside),
                centre + DVec2::new(0.0, outside),
            );
            assert!(alpha(inside) > 0, "{trace}: drawn");
            assert_eq!(alpha(outside), 0, "{trace}: not drawn");
            assert!(map.hit_test(inside).is_some(), "{trace}: picks");
            assert!(map.hit_test(outside).is_none(), "{trace}: doesn't pick");
        }
    }
}
