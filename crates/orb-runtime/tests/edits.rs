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
    // The change fades over frames, in the shaders: one row, written once.
    let (frames, first, jobs) = measure(&mut map, &set(7, true));
    assert!(frames > 1, "it fades");
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
    edit(&mut map, r#"[["setTransition", {"duration": 0}]]"#);
    edit(&mut map, &format!("[{}]", all.join(",")));
    // Without a fade, the very next frame draws the state: the scene
    // evaluates the features' captured data, with no job.
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

/// A feature-state change fades by the property's transition: mid-fade a
/// hovered feature draws between its colours, and once the fade ends, in
/// and back out, the map draws what a map that never faded draws.
#[test]
fn feature_state_changes_fade() {
    let mut map = tokyo();
    measure(&mut map, &grid(4));
    edit(
        &mut map,
        r##"[["setPaintProperty", "grid", "circle-color-transition", {"duration": 400}]]"##,
    );
    settle(&mut map);
    let plain = map.pixels();
    let hover = |on: bool| {
        let all: Vec<String> = (0..16)
            .map(|id| {
                format!(
                    r##"["setFeatureState", {{"source": "grid", "id": {id}}}, {{"hover": {on}}}]"##
                )
            })
            .collect();
        format!("[{}]", all.join(","))
    };
    edit(&mut map, &hover(true));
    // The red channel of each frame until the fade ends, per pixel.
    let started = Instant::now();
    let mut frames = Vec::new();
    loop {
        let report = map.frame();
        frames.push(map.pixels());
        if report.idle {
            break;
        }
        assert!(started.elapsed().as_secs() < 10, "the fade never ended");
        std::thread::sleep(Duration::from_millis(5));
    }
    settle(&mut map);
    let hovered = map.pixels();
    // A circle's middle: blue without hover, red with it, and between
    // the two in some frame of the fade.
    let middle = (0..plain.len() / 4)
        .max_by_key(|&i| hovered[4 * i].saturating_sub(plain[4 * i]))
        .unwrap();
    let (from, to) = (plain[4 * middle], hovered[4 * middle]);
    assert!(to > from + 200, "{from} → {to}");
    assert!(
        frames
            .iter()
            .any(|f| f[4 * middle] > from + 20 && f[4 * middle] < to - 20),
        "the circle's red went {:?}",
        frames.iter().map(|f| f[4 * middle]).collect::<Vec<_>>()
    );
    // Back out, interrupted halfway by a second change.
    edit(&mut map, &hover(false));
    map.frame();
    std::thread::sleep(Duration::from_millis(150));
    map.frame();
    edit(&mut map, &hover(true));
    settle(&mut map);
    assert_eq!(map.pixels(), hovered);
    edit(&mut map, &hover(false));
    settle(&mut map);
    assert_eq!(map.pixels(), plain);
}

/// A paint change eases by the transition in force at the next frame, as
/// GL JS and Native resolve it at their next update: one set to zero when
/// the value changed and made longer before that frame still eases.
#[test]
fn a_transition_set_after_a_change_eases_it() {
    let mut map = tokyo();
    edit(
        &mut map,
        r##"[["setPaintProperty", "background", "background-color", "#000000"]]"##,
    );
    settle(&mut map);
    let black = map.pixels();
    edit(
        &mut map,
        r##"[["setPaintProperty", "background", "background-color-transition", {"duration": 0}],
            ["setPaintProperty", "background", "background-color", "#ffffff"]]"##,
    );
    edit(
        &mut map,
        r##"[["setPaintProperty", "background", "background-color-transition", {"duration": 400}]]"##,
    );
    let frames = frames_until_idle(&mut map);
    let white = frames.last().unwrap().clone();
    assert_ne!(black, white);
    assert!(some_frame_between(&black, &white, &frames));
}

/// Frames until the map idles after an edit, each frame's pixels.
fn frames_until_idle(map: &mut Offscreen) -> Vec<Vec<u8>> {
    let started = Instant::now();
    let mut frames = Vec::new();
    loop {
        let report = map.frame();
        frames.push(map.pixels());
        if report.idle {
            return frames;
        }
        assert!(started.elapsed().as_secs() < 10, "the edit never settled");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Whether, at the pixel that differs most between `a` and `b`, some
/// frame draws neither.
fn some_frame_between(a: &[u8], b: &[u8], frames: &[Vec<u8>]) -> bool {
    let at = (0..a.len()).max_by_key(|&i| a[i].abs_diff(b[i])).unwrap();
    let (lo, hi) = (a[at].min(b[at]), a[at].max(b[at]));
    assert!(hi - lo > 60, "{lo}..{hi}");
    frames
        .iter()
        .any(|f| f[at] > lo.saturating_add(10) && f[at] < hi.saturating_sub(10))
}

/// A layer hidden or shown fades out or in, then draws as a fresh map.
#[test]
fn visibility_edits_fade() {
    for layer in ["water", "road_minor", "poi_r1"] {
        let mut map = tokyo();
        let shown = map.pixels();
        let hide = format!(r#"[["setLayoutProperty", "{layer}", "visibility", "none"]]"#);
        edit(&mut map, &hide);
        let frames = frames_until_idle(&mut map);
        settle(&mut map);
        let hidden = map.pixels();
        assert!(
            some_frame_between(&shown, &hidden, &frames),
            "{layer} faded out"
        );
        assert_matches_fresh(&mut map);
        let show = format!(r#"[["setLayoutProperty", "{layer}", "visibility", "visible"]]"#);
        edit(&mut map, &show);
        let frames = frames_until_idle(&mut map);
        assert!(
            some_frame_between(&shown, &hidden, &frames),
            "{layer} faded in"
        );
        settle(&mut map);
        assert_eq!(map.pixels(), shown);
    }
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
            let tms = orb_crs::TileMatrixSet::web_mercator();
            let view = map.map.view(map.viewport);
            let tiles: std::collections::HashSet<_> =
                view.cover(tms, 512.0, 14).iter().map(|t| t.index).collect();
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
    let mut doc = orb_style::v8::read(&std::fs::read_to_string(&path).unwrap()).unwrap();
    orb_source::endpoint::resolve_style_urls(&mut doc, &format!("file://{}", path.display()));
    doc.to_string()
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

/// Image tiles take the mesh of the surface they draw on, however the map
/// got there: a style URL that leaves an Equal Earth low-zoom plane, with
/// a globe edit on its `styleload`, regrids the resident tiles for the
/// sphere. Their finer meshes grow the scene's pools; a tile left without
/// room drew nothing (or, warped, its quads without their patches).
#[test]
fn image_tiles_regrid_for_a_globe_after_leaving_a_plane() {
    let scratch = Scratch::new("regrid");
    let tile = scratch.0.join("tile.png");
    let mut png = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png, 256, 256);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_image_data(&[64u8, 128, 255, 255].repeat(256 * 256))
            .unwrap();
    }
    std::fs::write(&tile, png).unwrap();
    let style = |projection: &str| {
        format!(
            r##"{{"version": 8, {projection}
                "sources": {{"imagery": {{"type": "raster", "tiles": ["file://{}"], "tileSize": 256}}}},
                "layers": [{{"id": "bg", "type": "background", "paint": {{"background-color": "#f00"}}}},
                    {{"id": "raster", "type": "raster", "source": "imagery",
                      "paint": {{"raster-fade-duration": 0}}}}]}}"##,
            tile.display()
        )
    };
    let [plane, flat] = ["plane", "flat"].map(|name| scratch.0.join(format!("{name}.json")));
    std::fs::write(
        &plane,
        style(r#""projection": {"type": "dev.sargunv.orbis:equal-earth"},"#),
    )
    .unwrap();
    std::fs::write(&flat, style("")).unwrap();
    let mut map = Offscreen::new(
        &Options {
            style: Some(format!("file://{}", plane.display())),
            encoding: "mvt".into(),
            at: Some((orb_crs::Geodetic::new(0.0, 20.0), 1.5, 0.0, 0.0)),
        },
        Default::default(),
    )
    .unwrap();
    map.map.set_previews(orb_style::Previews::ALL);
    settle(&mut map);
    assert!(map.map.surface().globe.is_some_and(|g| g.plane.is_some()));
    map.map
        .load_style_url(&format!("file://{}", flat.display()));
    let started = Instant::now();
    loop {
        map.frame();
        let events = map.map.take_events();
        let warnings: Vec<_> = (events.iter())
            .filter(|e| matches!(e, orb_runtime::MapEvent::Warning { .. }))
            .collect();
        assert!(warnings.is_empty(), "{warnings:?}");
        if (events.iter()).any(|e| matches!(e, orb_runtime::MapEvent::StyleLoad)) {
            break;
        }
        assert!(started.elapsed().as_secs() < 30, "the style did not arrive");
        std::thread::sleep(Duration::from_millis(1));
    }
    edit(&mut map, r#"[["setProjection", {"type": "globe"}]]"#);
    assert_matches_fresh(&mut map);
}

/// A sky and light edit eases by their transitions, frame by frame, and
/// once it ends the map draws what a fresh load of the edited style draws.
#[test]
fn sky_and_light_edits_ease_then_match_a_fresh_map() {
    let mut map = tokyo();
    let mut camera = *map.map.camera();
    camera.pitch = 70.0;
    map.map.jump_to_state(camera);
    edit(&mut map, r##"[["setSky", {"sky-color": "#102040"}]]"##);
    settle(&mut map);
    edit(
        &mut map,
        r##"[["setSky", {"sky-color": "#ff8000", "sky-color-transition": {"duration": 200}}],
            ["setLight", {"color": "#ffd0a0", "intensity": 0.8}]]"##,
    );
    let mut eased = 0;
    while map.map.transitioning() || eased == 0 {
        map.frame();
        eased += 1;
        assert!(eased < 10_000, "the edit never finished easing");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(eased > 2, "the sky eased over {eased} frames");
    assert_matches_fresh(&mut map);
}

/// Layers added to a loaded, pitched view get the scene room they need,
/// as in a fresh load of the edited style: an edit's upload once grew the
/// pools by one doubling and never the chunk table, so heavy layers added
/// over a loaded view, as a style editor adds them, went undrawn. Taking
/// them out again gives the room back.
#[test]
fn layers_added_to_a_loaded_pitched_view_get_scene_room() {
    let scratch = Scratch::new("room");
    let full: serde_json::Value = serde_json::from_str(&tokyo_json()).unwrap();
    let kept = ["background", "water", "label_city"];
    let added = || {
        full["layers"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|l| !kept.contains(&l["id"].as_str().unwrap()))
    };
    let mut reduced = full.clone();
    reduced["layers"]
        .as_array_mut()
        .unwrap()
        .retain(|l| kept.contains(&l["id"].as_str().unwrap()));
    let path = scratch.0.join("reduced.json");
    std::fs::write(&path, reduced.to_string()).unwrap();
    // First sizes far below what the full style needs, which only the
    // edit finds out.
    let capacity = orb_runtime::SceneCapacity {
        vertex_words: 1 << 12,
        index_words: 1 << 12,
        prop_words: 1 << 12,
        compacted_words: 1 << 12,
        chunks: 64,
        candidates: 64,
        ..orb_runtime::SceneCapacity::growable()
    };
    let options = Options {
        style: Some(format!("file://{}", path.display())),
        encoding: "mvt".into(),
        at: Some((orb_crs::Geodetic::new(139.765, 35.68), 15.5, -20.0, 60.0)),
    };
    let (mut map, _, _) = Offscreen::load_with(&options, capacity).unwrap();
    let (_, loaded) = settle(&mut map);
    let buildings = serde_json::json!({"id": "buildings-3d", "type": "fill-extrusion",
        "source": "openmaptiles", "source-layer": "building",
        "paint": {"fill-extrusion-height": ["coalesce", ["get", "render_height"], 20],
            "fill-extrusion-color": "#a0a0a0"}});
    let commands: Vec<serde_json::Value> = [serde_json::json!(["setTransition", {"duration": 0}])]
        .into_iter()
        .chain(added().map(|l| serde_json::json!(["addLayer", l, "label_city"])))
        .chain([serde_json::json!(["addLayer", buildings, "label_city"])])
        .collect();
    let (_, _, jobs) = measure(&mut map, &serde_json::Value::from(commands).to_string());
    assert_eq!(jobs.loads, 0, "{jobs:?}");
    let (_, edited) = settle(&mut map);
    let words = |r: &FrameReport| r.memory.scene.vertices.allocated;
    assert!(
        words(&edited) > 4 * words(&loaded),
        "the edit must need the pools grown: {} bytes loaded, {} edited",
        words(&loaded),
        words(&edited)
    );
    // At rest the pools have room for the next view (#176): none is more
    // than three quarters full below its ceiling. Loaded, the property
    // pool would be 98% full.
    for (name, r) in [("loaded", &loaded), ("edited", &edited)] {
        let s = r.memory.scene;
        for pool in [s.vertices, s.indices, s.props] {
            assert!(
                pool.used * 4 <= pool.allocated * 3 || pool.allocated == pool.ceiling,
                "{name} at rest: {pool:?}"
            );
        }
    }
    assert_matches_fresh(&mut map);
    let remove: Vec<serde_json::Value> = added()
        .map(|l| serde_json::json!(["removeLayer", l["id"]]))
        .chain([serde_json::json!(["removeLayer", "buildings-3d"])])
        .collect();
    measure(&mut map, &serde_json::Value::from(remove).to_string());
    let (_, removed) = settle(&mut map);
    for (name, r) in [
        ("loaded", &loaded),
        ("edited", &edited),
        ("removed", &removed),
    ] {
        eprintln!("{name}: {:?}", r.memory.scene);
    }
    let compacted = |r: &FrameReport| r.memory.scene.compacted.allocated;
    assert!(
        words(&removed) * 4 <= words(&edited) && compacted(&removed) * 4 <= compacted(&edited),
        "the pools give back what the layers took: {:?}, then {:?}",
        edited.memory.scene,
        removed.memory.scene
    );
}
