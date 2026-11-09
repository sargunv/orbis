//! Frame-sequence tests of style edits on the Tokyo extract (M1 exit
//! criterion 3): per S6 class, what an edit costs in frames, jobs and
//! uploads, and that its pixels equal a fresh load of the edited style.

mod common;

use std::time::{Duration, Instant};

use orb_runtime::{FrameReport, JobCounts, Transaction};

use crate::common::{Offscreen, Options};

fn tokyo() -> Offscreen {
    let options = Options {
        style: None,
        encoding: "mvt".into(),
        at: Some((orb_crs::Geodetic::new(139.74, 35.685), 15.0, 0.0, 0.0)),
    };
    let (mut offscreen, _, _) = Offscreen::load(&options).unwrap();
    settle(&mut offscreen);
    offscreen
}

/// Renders until idle. Returns the frames drawn and the last report.
fn settle(offscreen: &mut Offscreen) -> (u32, FrameReport) {
    let started = Instant::now();
    let mut frames = 0;
    // Two idle frames in a row: the second draws what the first's messages
    // installed.
    let mut was_idle = false;
    loop {
        let report = offscreen.frame();
        frames += 1;
        let warnings = crate::common::take_warnings(&mut offscreen.map);
        assert!(warnings.is_empty(), "{warnings:?}");
        if report.idle && was_idle {
            return (frames, report);
        }
        was_idle = report.idle;
        assert!(started.elapsed().as_secs() < 60, "edit did not settle");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn edit(offscreen: &mut Offscreen, commands: &str) -> u64 {
    offscreen
        .map
        .edit(&Transaction::from_commands(commands).unwrap())
        .unwrap()
}

fn delta(before: JobCounts, after: JobCounts) -> JobCounts {
    JobCounts {
        loads: after.loads - before.loads,
        relayouts: after.relayouts - before.relayouts,
        columns: after.columns - before.columns,
        meshes_uploaded: after.meshes_uploaded - before.meshes_uploaded,
        rows_uploaded: after.rows_uploaded - before.rows_uploaded,
        state_rows: after.state_rows - before.state_rows,
    }
}

/// Applies an edit and renders until it shows in full. Returns the frames
/// that took, the first frame's report and the jobs and uploads it caused.
fn measure(offscreen: &mut Offscreen, commands: &str) -> (u32, FrameReport, JobCounts) {
    let before = offscreen.map.job_counts();
    let generation = edit(offscreen, commands);
    let first = offscreen.frame();
    let (frames, last) = if first.idle {
        (1, first)
    } else {
        let (frames, last) = settle(offscreen);
        (frames + 1, last)
    };
    assert!(last.generation >= generation);
    (frames, first, delta(before, offscreen.map.job_counts()))
}

/// The edited map's pixels equal those of its style loaded fresh, with the
/// same camera.
fn assert_matches_fresh(offscreen: &mut Offscreen) {
    let json = offscreen.map.style_doc().unwrap().to_json();
    let camera = *offscreen.map.camera();
    let mut fresh = Offscreen::load(&Options {
        style: None,
        encoding: "mvt".into(),
        at: Some((
            camera.target,
            offscreen.map.zoom(),
            camera.bearing,
            camera.pitch,
        )),
    })
    .unwrap()
    .0;
    fresh.map.jump_to_state(camera);
    fresh.map.set_style_json(&json).unwrap();
    settle(&mut fresh);
    settle(offscreen);
    let (a, b) = (offscreen.pixels(), fresh.pixels());
    let differing = crate::common::differing_pixels(&a, &b);
    if differing > 0 {
        // Both pictures, for the hunt.
        let dir = std::env::temp_dir().join(format!("orb-edits-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, pixels) in [("edited", &a), ("fresh", &b)] {
            let path = dir.join(format!("{name}.png"));
            crate::common::write_png(path.to_str().unwrap(), pixels).unwrap();
        }
        panic!(
            "{differing} edited and fresh pixels differ; see {}",
            dir.display()
        );
    }
}

#[test]
fn uniform_paint_edits_cost_one_frame_and_no_jobs() {
    let mut map = tokyo();
    // Without transitions, so each edit shows in full at once.
    edit(&mut map, r#"[["setTransition", {"duration": 0}]]"#);
    settle(&mut map);
    for commands in [
        r##"[["setPaintProperty", "water", "fill-color", "#ff0000"]]"##,
        r##"[["setPaintProperty", "road_minor", "line-width",
            ["interpolate", ["linear"], ["zoom"], 10, 1, 18, 12]]]"##,
        r##"[["setLayoutProperty", "park", "visibility", "none"]]"##,
        r##"[["setPaintProperty", "background", "background-color", "#202020"]]"##,
    ] {
        let (frames, first, jobs) = measure(&mut map, commands);
        assert_eq!(frames, 1, "{commands}");
        // Visibility changes cull candidates, not uniforms.
        assert!(first.frame.paints_written > 0 || commands.contains("visibility"));
        assert_eq!(jobs, JobCounts::default(), "{commands}");
    }
    assert_matches_fresh(&mut map);
}

#[test]
fn binding_flips_evaluate_rows_without_relayout() {
    let mut map = tokyo();
    let flip = r##"[["setPaintProperty", "road_minor", "line-color",
        ["match", ["get", "class"], "minor", "#ff0000", "#0000ff"]]]"##;
    let (frames, _, jobs) = measure(&mut map, flip);
    eprintln!("constant → data-driven: {frames} frames, {jobs:?}");
    assert_eq!(
        (jobs.loads, jobs.relayouts, jobs.meshes_uploaded),
        (0, 0, 0)
    );
    assert!(jobs.columns > 0 && jobs.rows_uploaded > 0, "{jobs:?}");
    assert_matches_fresh(&mut map);
    // Back to a constant: the rows are left unread.
    let back = r##"[["setPaintProperty", "road_minor", "line-color", "#00ff00"]]"##;
    let (frames, _, jobs) = measure(&mut map, back);
    assert_eq!((frames, jobs), (1, JobCounts::default()));
    assert_matches_fresh(&mut map);
}

#[test]
fn layout_edits_lay_out_one_layer_from_cached_tiles() {
    let mut map = tokyo();
    let filter = r##"[["setFilter", "road_minor", ["==", ["get", "class"], "service"]]]"##;
    let (frames, _, jobs) = measure(&mut map, filter);
    eprintln!("filter: {frames} frames, {jobs:?}");
    assert_eq!((jobs.loads, jobs.columns), (0, 0));
    assert!(jobs.relayouts > 0, "{jobs:?}");
    assert_matches_fresh(&mut map);
    let cap = r##"[["setLayoutProperty", "road_minor", "line-cap", "butt"]]"##;
    let (_, _, jobs) = measure(&mut map, cap);
    assert_eq!(jobs.loads, 0);
    assert_matches_fresh(&mut map);
}

#[test]
fn layer_and_source_edits_keep_the_other_layers() {
    let mut map = tokyo();
    let layers = r##"[
        ["addLayer", {"id": "outline", "type": "line", "source": "openmaptiles",
            "source-layer": "water", "paint": {"line-color": "#000", "line-width": 2}}, "park"],
        ["moveLayer", "water", "road_minor"],
        ["removeLayer", "landuse_residential"]
    ]"##;
    let (frames, _, jobs) = measure(&mut map, layers);
    eprintln!("layers: {frames} frames, {jobs:?}");
    assert_eq!((jobs.loads, jobs.columns), (0, 0), "{jobs:?}");
    assert!(
        jobs.relayouts > 0,
        "only the new layer is laid out: {jobs:?}"
    );
    assert_matches_fresh(&mut map);
    let source = r##"[
        ["addSource", "pin", {"type": "geojson", "data": {"type": "Point", "coordinates": [139.74, 35.685]}}],
        ["addLayer", {"id": "pin", "type": "circle", "source": "pin", "paint": {"circle-radius": 20}}]
    ]"##;
    let (_, _, jobs) = measure(&mut map, source);
    eprintln!("source: {jobs:?}");
    assert_eq!((jobs.relayouts, jobs.columns), (0, 0));
    assert!(
        jobs.loads > 0 && jobs.loads < 8,
        "only the new source loads: {jobs:?}"
    );
    assert!(
        jobs.meshes_uploaded <= jobs.loads,
        "one mesh per new tile: {jobs:?}"
    );
    assert_matches_fresh(&mut map);
}

/// A GeoJSON grid of points with generated ids, drawn with a state lookup.
fn grid(n: usize) -> String {
    let features: Vec<String> = (0..n * n)
        .map(|i| {
            let (x, y) = ((i % n) as f64, (i / n) as f64);
            format!(
                r##"{{"type":"Feature","properties":{{}},"geometry":{{"type":"Point","coordinates":[{},{}]}}}}"##,
                139.735 + x * 0.0004,
                35.682 + y * 0.0003
            )
        })
        .collect();
    format!(
        r##"[["addSource", "grid", {{"type": "geojson", "generateId": true,
            "data": {{"type": "FeatureCollection", "features": [{}]}}}}],
        ["addLayer", {{"id": "grid", "type": "circle", "source": "grid", "paint": {{
            "circle-radius": 4,
            "circle-color": ["case", ["boolean", ["feature-state", "hover"], false], "#ff0000", "#0000ff"]}}}}]]"##,
        features.join(",")
    )
}

#[test]
fn feature_state_writes_one_row() {
    let mut map = tokyo();
    let n = 20;
    measure(&mut map, &grid(n));
    let set = |id: usize, hover: bool| {
        format!(
            r##"[["setFeatureState", {{"source": "grid", "id": {id}}}, {{"hover": {hover}}}]]"##
        )
    };
    let (frames, first, jobs) = measure(&mut map, &set(7, true));
    assert_eq!(frames, 1);
    assert_eq!(first.frame.paints_written, 0, "no uniform changes");
    assert_eq!(
        jobs,
        JobCounts {
            state_rows: 1,
            ..JobCounts::default()
        }
    );
    // Every feature hovered draws as the hover colour does.
    let all: Vec<String> = (0..n * n)
        .map(|id| {
            format!(r##"["setFeatureState", {{"source": "grid", "id": {id}}}, {{"hover": true}}]"##)
        })
        .collect();
    measure(&mut map, &format!("[{}]", all.join(",")));
    let hovered = map.pixels();
    edit(
        &mut map,
        r##"[["setPaintProperty", "grid", "circle-color", "#ff0000"]]"##,
    );
    settle(&mut map);
    assert_eq!(map.pixels(), hovered);
    assert_matches_fresh(&mut map);
}

#[test]
fn paint_reads_feature_state_and_data_together() {
    let mut map = tokyo();
    let n = 10;
    let features: Vec<String> = (0..n * n)
        .map(|i| {
            let (x, y) = ((i % n) as f64, (i / n) as f64);
            format!(
                r##"{{"type":"Feature","properties":{{"kind":"{}"}},"geometry":{{"type":"Point","coordinates":[{},{}]}}}}"##,
                ["a", "b"][i % 2],
                139.735 + x * 0.0004,
                35.682 + y * 0.0003
            )
        })
        .collect();
    let by_kind = r##"["match", ["get", "kind"], "a", "#ff0000", "#00ff00"]"##;
    let otherwise = r##"["match", ["get", "kind"], "a", "#0000ff", "#ffff00"]"##;
    measure(
        &mut map,
        &format!(
            r##"[["addSource", "grid", {{"type": "geojson", "generateId": true,
                "data": {{"type": "FeatureCollection", "features": [{}]}}}}],
            ["addLayer", {{"id": "grid", "type": "circle", "source": "grid", "paint": {{
                "circle-radius": 4,
                "circle-color": ["case", ["boolean", ["feature-state", "hover"], false],
                    {by_kind}, {otherwise}]}}}}]]"##,
            features.join(",")
        ),
    );
    let all: Vec<String> = (0..n * n)
        .map(|id| {
            format!(r##"["setFeatureState", {{"source": "grid", "id": {id}}}, {{"hover": true}}]"##)
        })
        .collect();
    let plain = map.pixels();
    let before = map.map.job_counts();
    edit(&mut map, &format!("[{}]", all.join(",")));
    // The very next frame draws the state: the scene evaluates the
    // features' captured data, with no job.
    assert!(map.frame().idle);
    assert_eq!(
        delta(before, map.map.job_counts()),
        JobCounts {
            state_rows: (n * n) as u64,
            ..JobCounts::default()
        }
    );
    // Hovered features draw their data's colour, as the colour without
    // the state branch does.
    let hovered = map.pixels();
    assert_ne!(hovered, plain);
    edit(
        &mut map,
        &format!(r##"[["setPaintProperty", "grid", "circle-color", {by_kind}]]"##),
    );
    settle(&mut map);
    assert_eq!(map.pixels(), hovered);
}

/// The CPU cost of a feature-state write on the scene thread: the edit
/// call, and the next frame's extra time over a frame without one.
/// `cargo test --release -p orb-demo feature_state_write_cost -- --ignored --nocapture`
#[test]
#[ignore = "a measurement"]
fn feature_state_write_cost() {
    let mut map = tokyo();
    measure(&mut map, &grid(40));
    let txs: Vec<Transaction> = (0..2)
        .map(|hover| {
            Transaction::from_commands(&format!(
                r##"[["setFeatureState", {{"source": "grid", "id": 123}}, {{"hover": {}}}]]"##,
                hover == 0
            ))
            .unwrap()
        })
        .collect();
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (mut edits, mut with, mut without) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..400 {
        let started = Instant::now();
        map.frame();
        without.push(started.elapsed().as_secs_f64() * 1e3);
        let started = Instant::now();
        map.map.edit(&txs[i % 2]).unwrap();
        edits.push(started.elapsed().as_secs_f64() * 1e3);
        let started = Instant::now();
        map.frame();
        with.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let (edit, with, without) = (median(edits), median(with), median(without));
    eprintln!(
        "feature-state write: edit {edit:.4} ms + frame {with:.4} − {without:.4} ms = {:.4} ms",
        edit + (with - without)
    );
    // A uniform paint edit, for comparison: it re-parses one layer of the
    // 66-layer style and publishes a new snapshot.
    let paints: Vec<Transaction> = ["#ff0000", "#00ff00"]
        .iter()
        .map(|color| {
            Transaction::from_commands(&format!(
                r##"[["setPaintProperty", "water", "fill-color", "{color}"]]"##
            ))
            .unwrap()
        })
        .collect();
    let (mut edits, mut with) = (Vec::new(), Vec::new());
    for i in 0..200 {
        let started = Instant::now();
        map.map.edit(&paints[i % 2]).unwrap();
        edits.push(started.elapsed().as_secs_f64() * 1e3);
        let started = Instant::now();
        map.frame();
        with.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let (edit, with) = (median(edits), median(with));
    eprintln!(
        "paint edit: edit {edit:.4} ms, its frame {with:.4} ms (camera-only {without:.4} ms)"
    );
}

#[test]
fn extrusion_nodes_survive_renumbering() {
    let mut map = tokyo();
    map.map.jump_to_state(orb_runtime::CameraState {
        pitch: 45.0,
        ..*map.map.camera()
    });
    settle(&mut map);
    let add = r##"[["addLayer", {"id": "buildings", "type": "fill-extrusion",
        "source": "openmaptiles", "source-layer": "building",
        "paint": {"fill-extrusion-height": 30, "fill-extrusion-color": "#a0a0a0"}}]]"##;
    let (_, _, jobs) = measure(&mut map, add);
    assert_eq!(jobs.loads, 0);
    assert_matches_fresh(&mut map);
    // Moving layers below it renumbers the extrusion node, which keeps its
    // meshes and redraws its targets.
    let (_, _, jobs) = measure(&mut map, r#"[["moveLayer", "buildings", "water"]]"#);
    assert_eq!(jobs, JobCounts::default());
    assert_matches_fresh(&mut map);
    let (_, _, jobs) = measure(&mut map, r#"[["removeLayer", "background"]]"#);
    assert_eq!(jobs, JobCounts::default());
    assert_matches_fresh(&mut map);
}

#[test]
fn hits_follow_layer_order_visibility_and_camera() {
    let mut map = tokyo();
    map.map
        .set_style_json(include_str!("../../../web/demo/data/playground.json"))
        .unwrap();
    map.map.jump_to_state(orb_runtime::CameraState::new(
        orb_crs::Geodetic::new(139.7655, 35.6835),
        15.0,
        &crate::common::web_mercator(),
    ));
    settle(&mut map);
    let center = glam::DVec2::new(512., 384.);
    let hit = map.map.hit_test(center).expect("center station");
    assert_eq!(
        (hit.layer.as_str(), hit.id.as_deref()),
        ("stations", Some("31"))
    );
    assert_eq!(hit.source, "district");
    assert_eq!(hit.source_layer, None);
    // The last frame answers for its camera.
    crate::common::move_camera(&mut map.map, |c| c.bearing = 30.0);
    map.frame();
    assert!(map.map.hit_test(center).is_some());
    crate::common::move_camera(&mut map.map, |c| c.bearing = 0.0);
    settle(&mut map);
    let tx =
        Transaction::from_commands(r#"[["setLayoutProperty","stations","visibility","none"]]"#)
            .unwrap();
    map.map.edit(&tx).unwrap();
    assert_eq!(map.map.hit_test(center), Some(hit)); // published frame remains queryable
    let frame = map.frame();
    let route = map.map.hit_test(center).expect("route under station");
    assert_eq!(route.layer, "route");
    assert_eq!(route.generation, frame.generation);
    crate::common::move_camera(&mut map.map, |c| c.target = orb_crs::Geodetic::new(0., 0.));
    assert_eq!(map.map.hit_test(center), Some(route));
    settle(&mut map);
    assert_eq!(map.map.hit_test(center), None);
    assert_eq!(map.map.hit_test(glam::DVec2::splat(f64::NAN)), None);
}

/// A style of one GeoJSON point at the camera's target (or `east` degrees
/// east of it) drawn as a circle with the given paint, after an unused
/// source that comes first in the document.
fn point_style(east: f64, circle_paint: &str) -> String {
    let (lon, lat) = (139.74 + east, 35.685);
    format!(
        r#"{{"version": 8,
            "sources": {{
                "unused": {{"type": "geojson", "data": {{"type": "FeatureCollection", "features": []}}}},
                "points": {{"type": "geojson", "data": {{"type": "Feature", "id": 7,
                    "properties": {{"r": 100, "s": 20}},
                    "geometry": {{"type": "Point", "coordinates": [{lon}, {lat}]}}}}}}
            }},
            "layers": [
                {{"id": "paper", "type": "background", "paint": {{"background-color": "white"}}}},
                {{"id": "dots", "type": "circle", "source": "points", "paint": {circle_paint}}}
            ]}}"#
    )
}

fn load_points(style: &str) -> Offscreen {
    let mut map = tokyo();
    map.map.set_style_json(style).unwrap();
    map.map.jump_to_state(orb_runtime::CameraState::new(
        orb_crs::Geodetic::new(139.74, 35.685),
        15.0,
        &crate::common::web_mercator(),
    ));
    settle(&mut map);
    map
}

#[test]
fn a_radius_made_per_feature_widens_culling() {
    // About 24 px beyond the right edge of the 1024 px viewport at z15: a
    // 5 px circle there is culled; a 100 px one reaches into view.
    let mut map = load_points(&point_style(
        0.0115,
        r#"{"circle-radius": 5, "circle-color": "red"}"#,
    ));
    let (_, _, jobs) = measure(
        &mut map,
        r#"[["setPaintProperty", "dots", "circle-radius", ["get", "r"]]]"#,
    );
    assert_eq!(jobs.relayouts, 0);
    assert_matches_fresh(&mut map);
}

#[test]
fn hits_name_the_source_their_layer_draws() {
    let map = load_points(&point_style(0.0, r#"{"circle-radius": 8}"#));
    let hit = map.map.hit_test(glam::DVec2::new(512., 384.)).expect("dot");
    assert_eq!(
        (hit.layer.as_str(), hit.source.as_str()),
        ("dots", "points")
    );
}

#[test]
fn circle_hits_cover_the_stroke_however_each_part_is_bound() {
    // 15 px from the centre: outside a 5 px radius, inside its 20 px stroke.
    let probe = glam::DVec2::new(512. + 15., 384.);
    let map = load_points(&point_style(
        0.0,
        r#"{"circle-radius": 5, "circle-stroke-width": ["get", "s"]}"#,
    ));
    assert!(map.map.hit_test(probe).is_some(), "per-feature stroke");

    let mut map = load_points(&point_style(
        0.0,
        r#"{"circle-radius": ["/", ["get", "r"], 20], "circle-stroke-width": 0}"#,
    ));
    assert!(map.map.hit_test(probe).is_none());
    measure(
        &mut map,
        r#"[["setPaintProperty", "dots", "circle-stroke-width", 20]]"#,
    );
    assert!(map.map.hit_test(probe).is_some(), "uniform stroke edit");
}

/// A projection switch shows on the next frame and
/// fetches nothing again. Resident vector tiles lay out again for the
/// globe's curvature from cached data; only tiles the new surface's cover
/// alone wants load. (Pixels aren't compared with a fresh load: the two
/// covers differ, and the pyramid keeps loaded descendants standing in
/// for new ideal tiles, which a fresh load doesn't have.)
#[test]
fn projection_switches_load_nothing() {
    let mut map = tokyo();
    let surface = map.map.surface();
    let mut camera = *map.map.camera();
    camera.set_zoom(&surface.projection, 9.0);
    map.map.jump_to_state(camera);
    settle(&mut map);
    for (commands, globe) in [
        (r#"[["setProjection", {"type": "globe"}]]"#, true),
        (r#"[["setProjection", {"type": "mercator"}]]"#, false),
    ] {
        let cover = |map: &Offscreen| {
            let tms = orb_crs::TileMatrixSet::by_id("WebMercatorQuad").unwrap();
            let view = map.map.view(map.viewport);
            let tiles: std::collections::HashSet<_> = view
                .cover(&tms, 512.0, 14)
                .iter()
                .map(|t| t.index)
                .collect();
            tiles
        };
        let before = cover(&map);
        let started = Instant::now();
        let (frames, first, jobs) = measure(&mut map, commands);
        let only_new = cover(&map).difference(&before).count() as u32;
        eprintln!(
            "{commands}: {frames} frames to settle, {:.1} ms, {jobs:?}",
            started.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(map.map.surface().globe.is_some(), globe);
        assert!(
            first.tiles_drawn > 0,
            "the first frame draws the new surface"
        );
        assert!(
            jobs.loads <= only_new,
            "{commands}: {jobs:?}, {only_new} new tiles"
        );
        assert!(
            jobs.relayouts > 0,
            "{commands}: resident tiles lay out again"
        );
    }
}

/// A scratch directory of this test's own, removed when it drops.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("orb-edits-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The Tokyo style with its URLs absolute, to write elsewhere.
fn tokyo_json() -> String {
    let path = std::path::Path::new(crate::common::DEMO_DATA)
        .join("tokyo-mvt.json")
        .canonicalize()
        .unwrap();
    let json = std::fs::read_to_string(&path).unwrap();
    orb_source::endpoint::resolve_style_urls(&json, &format!("file://{}", path.display()))
}

/// Another sprite in a style edit loads the sprite and lays the symbol
/// layers out again from cached tiles, loading no tile; images the new
/// sprite lacks leave, as a fresh load of the edited style shows.
#[test]
fn another_sprite_loads_only_the_sprite() {
    let scratch = Scratch::new("sprite");
    // Half the extract's sprite, the same sheet.
    for name in ["ofm.json", "ofm@2x.json"] {
        let path = std::path::Path::new(crate::common::DEMO_DATA)
            .join("sprites")
            .join(name);
        let text = std::fs::read_to_string(path).unwrap();
        let json = orb_expr::json::parse(&text).unwrap();
        let members = json.as_object().unwrap();
        let half: Vec<String> = (members.iter().take(members.len() / 2))
            .map(|m| format!("{:?}:{}", m.key, m.value))
            .collect();
        let path = scratch.0.join(name.replace("ofm", "half"));
        std::fs::write(path, format!("{{{}}}", half.join(","))).unwrap();
    }
    for name in ["ofm.png", "ofm@2x.png"] {
        let from = std::path::Path::new(crate::common::DEMO_DATA)
            .join("sprites")
            .join(name);
        if from.exists() {
            std::fs::copy(from, scratch.0.join(name.replace("ofm", "half"))).unwrap();
        }
    }
    let mut map = tokyo();
    edit(&mut map, r#"[["setTransition", {"duration": 0}]]"#);
    settle(&mut map);
    let sprite = format!("file://{}/half", scratch.0.display());
    let (_, _, jobs) = measure(&mut map, &format!(r#"[["setSprite", {sprite:?}]]"#));
    assert_eq!(jobs.loads, 0, "{jobs:?}");
    assert!(jobs.relayouts > 0, "symbols lay out again: {jobs:?}");
    assert_matches_fresh(&mut map);
}

/// A style URL loaded over a style applies as the diff from it: its
/// source keeps its tiles, paint eases by the transition, and the map
/// then draws what a fresh load of the new style draws.
#[test]
fn a_style_url_over_a_style_diffs() {
    let scratch = Scratch::new("url");
    let next = tokyo_json().replace("rgb(158,189,255)", "rgb(40,60,200)");
    assert_ne!(next, tokyo_json());
    let path = scratch.0.join("next.json");
    std::fs::write(&path, next).unwrap();
    let mut map = tokyo();
    let before = map.map.job_counts();
    map.map
        .load_style_url(&format!("file://{}", path.display()));
    let started = Instant::now();
    let mut loaded = false;
    let mut eased = false;
    while !loaded || map.map.transitioning() {
        map.frame();
        loaded |= map
            .map
            .take_events()
            .iter()
            .any(|e| matches!(e, orb_runtime::MapEvent::StyleLoad));
        eased |= map.map.transitioning();
        assert!(started.elapsed().as_secs() < 30, "the style did not arrive");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(eased, "the water's colour eases");
    settle(&mut map);
    let jobs = delta(before, map.map.job_counts());
    assert_eq!(jobs.loads, 0, "{jobs:?}");
    assert_matches_fresh(&mut map);
}

/// A camera move started after a projection edit runs on the new surface:
/// the edit's surface reaches the camera at once, not at the next frame,
/// where it would stop the move.
#[test]
fn a_move_after_a_projection_edit_runs() {
    let mut map = tokyo();
    edit(&mut map, r#"[["setProjection", {"type": "globe"}]]"#);
    let target = orb_runtime::CameraTarget {
        zoom: Some(map.map.zoom() - 4.0),
        ..Default::default()
    };
    map.map.ease_to(
        &target,
        orb_runtime::EaseOptions {
            duration: 200.0,
            ..Default::default()
        },
    );
    map.frame();
    assert!(map.map.is_moving(), "the move survives the first frame");
    settle(&mut map);
    assert!((map.map.zoom() - target.zoom.unwrap()).abs() < 1e-6);
}
