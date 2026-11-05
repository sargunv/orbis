//! Model layers (decision 0047) through the embedding API.

use std::path::PathBuf;

use orb_api::{CameraState, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, Transaction};

/// The demo's torii, seen from above at zoom 19, its lintel across the
/// view's centre, coloured red by feature state `hot` and blue without.
fn torii_scene() -> String {
    let torii = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../web/demo/data/models/torii.glb")
        .canonicalize()
        .unwrap();
    format!(
        r##"{{
          "style": {{"version": 8,
            "sources": {{"g": {{"type": "geojson", "data": {{"type": "Feature", "id": 1,
              "properties": {{}}, "geometry": {{"type": "Point", "coordinates": [0, 0]}}}}}}}},
            "layers": [{{"id": "bg", "type": "background", "paint": {{"background-color": "#ffffff"}}}}]}},
          "models": {{"torii": "file://{}"}},
          "layers+": [{{"id": "gates", "type": "model", "source": "g",
            "layout": {{"model-id": "torii"}},
            "paint": {{
              "model-color": ["case", ["boolean", ["feature-state", "hot"], false], "#ff0000", "#0000ff"],
              "model-color-mix-intensity": 1,
              "model-emissive-strength": 1}}}}]
        }}"##,
        torii.display()
    )
}

#[test]
fn model_paint_reads_feature_state() {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let projection = *orb_api::TileMatrixSet::by_id("WebMercatorQuad")
        .unwrap()
        .projection();
    let mut map = Map::new(
        &runtime,
        MapOptions {
            camera: Some(CameraState::new(Geodetic::new(0.0, 0.0), 19.0, &projection)),
            ..MapOptions::default()
        },
    );
    map.set_style_json(&torii_scene()).unwrap();
    // 256 px high puts the eye 57 m up, above the 23 m gate.
    let centre = |map: &mut Map| {
        let image = map.snapshot(256, 256, 1.0).unwrap();
        let at = ((128 * 256 + 128) * 4) as usize;
        image.pixels[at..at + 4].to_vec()
    };
    assert_eq!(centre(&mut map), [0, 0, 255, 255]);
    let hot = r#"[["setFeatureState", {"source": "g", "id": 1}, {"hot": true}]]"#;
    map.edit(&Transaction::from_commands(hot).unwrap()).unwrap();
    assert_eq!(centre(&mut map), [255, 0, 0, 255]);
}
