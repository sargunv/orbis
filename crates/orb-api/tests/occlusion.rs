//! Labels behind 3D content (the scene's `symbol-occlusion`): a building or
//! a model standing between the eye and a label hides it; without
//! occlusion the label draws over them, as GL JS draws labels.

use orb_api::{CameraState, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, TileMatrixSet};

/// Metres north → degrees of latitude near the equator.
fn north(m: f64) -> f64 {
    m / 110_574.0
}

/// A red label at the origin, and `blocker` (sources `b`, a 120 m square
/// 100 m south of it, and `p`, a point there) between it and an eye to the
/// south.
fn scene(occlusion: bool, blocker: &str) -> String {
    let dir = env!("CARGO_MANIFEST_DIR");
    let data = format!("{dir}/../../web/demo/data");
    let half = 60.0 / 111_320.0;
    let [s, n] = [north(-160.0), north(-40.0)];
    let at = north(-100.0);
    format!(
        r##"{{
          "symbol-occlusion": {occlusion},
          "models": {{"pagoda": "file://{data}/models/pagoda.glb"}},
          "style": {{
            "version": 8,
            "glyphs": "file://{data}/glyphs/{{fontstack}}/{{range}}.pbf",
            "sources": {{
              "label": {{"type": "geojson", "data": {{"type": "Point", "coordinates": [0, 0]}}}},
              "b": {{"type": "geojson", "data": {{"type": "Polygon",
                "coordinates": [[[-{half}, {s}], [{half}, {s}], [{half}, {n}], [-{half}, {n}], [-{half}, {s}]]]}}}},
              "p": {{"type": "geojson", "data": {{"type": "Point", "coordinates": [0, {at}]}}}}
            }},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {blocker},
              {{"id": "label", "type": "symbol", "source": "label",
                "layout": {{"text-field": "XXXX", "text-font": ["Noto Sans Regular"], "text-size": 24}},
                "paint": {{"text-color": "#ff0000"}}}}
            ]
          }}
        }}"##
    )
}

const BUILDING: &str = r##"{"id": "building", "type": "fill-extrusion", "source": "b",
  "paint": {"fill-extrusion-color": "#888888", "fill-extrusion-height": 300}}"##;
/// 300 m tall, 138 m across.
const PAGODA: &str = r##"{"id": "pagoda", "type": "model", "source": "p",
  "layout": {"model-id": "pagoda"}, "paint": {"model-scale": [6, 6, 6]}}"##;

/// The red pixels of `scene` seen from the south at zoom 16, pitch 60.
fn red_pixels(scene: &str) -> usize {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
    let mut camera = CameraState::new(Geodetic::new(0.0, 0.0), 16.0, tms.projection());
    camera.pitch = 60.0;
    let mut map = Map::new(
        &runtime,
        MapOptions {
            camera: Some(camera),
            ..MapOptions::default()
        },
    );
    map.set_style_json(scene).unwrap();
    let image = map.snapshot(256, 256, 1.0).unwrap();
    image
        .pixels
        .chunks(4)
        .filter(|p| p[0] > 200 && p[1] < 80 && p[2] < 80)
        .count()
}

fn hidden_behind(blocker: &str) {
    let shown = red_pixels(&scene(false, blocker));
    let hidden = red_pixels(&scene(true, blocker));
    assert!(
        shown > 50,
        "without occlusion the label shows: {shown} pixels"
    );
    assert_eq!(hidden, 0, "the label is hidden behind it");
}

/// With nothing standing in front of it, an occluded scene's label shows.
#[test]
fn labels_in_the_open_show() {
    let flat = BUILDING.replace(
        "\"fill-extrusion-height\": 300",
        "\"fill-extrusion-height\": 0",
    );
    let shown = red_pixels(&scene(true, &flat));
    assert!(shown > 50, "the label shows: {shown} pixels");
}

#[test]
fn labels_hide_behind_buildings() {
    hidden_behind(BUILDING);
}

#[test]
fn labels_hide_behind_models() {
    hidden_behind(PAGODA);
}
