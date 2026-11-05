//! Shadows from the scene's directional light (decision 0046): a building
//! lit from the east darkens the ground west of it, away from the light,
//! and not east of it.

use orb_api::{CameraState, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, TileMatrixSet};

/// A 100 m building 40 m across at the origin on a white ground, lit from
/// the east 45° up.
fn scene(cast: bool) -> String {
    let half = 0.00018;
    format!(
        r##"{{
          "style": {{
            "version": 8,
            "sources": {{"b": {{"type": "geojson", "data": {{"type": "Polygon",
              "coordinates": [[[-{half}, -{half}], [{half}, -{half}], [{half}, {half}], [-{half}, {half}], [-{half}, -{half}]]]}}}}}},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {{"id": "building", "type": "fill-extrusion", "source": "b",
                "paint": {{"fill-extrusion-color": "#888888", "fill-extrusion-height": 100}}}}
            ]
          }},
          "lights": [
            {{"type": "ambient", "properties": {{"intensity": 0.5}}}},
            {{"type": "directional", "properties": {{"direction": [90, 45], "intensity": 0.5, "cast-shadows": {cast}}}}}
          ]
        }}"##
    )
}

/// The ground's brightness 50 m west and east of the building's centre:
/// past its roof, which stands nearer the camera, and inside its shadow,
/// 100 m long.
fn ground(cast: bool) -> [u8; 2] {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
    let mut map = Map::new(
        &runtime,
        MapOptions {
            camera: Some(CameraState::new(
                Geodetic::new(0.0, 0.0),
                17.0,
                tms.projection(),
            )),
            ..MapOptions::default()
        },
    );
    map.set_style_json(&scene(cast)).unwrap();
    let image = map.snapshot(256, 256, 1.0).unwrap();
    let away = 50.0 / 111_320.0;
    [-away, away].map(|lon| {
        let p = map.project(Geodetic::new(lon, 0.0)).unwrap();
        let [x, y] = [p.x, p.y].map(|v| v as u32);
        assert!(x < image.width && y < image.height, "{p} is in the view");
        image.pixels[(y * image.width + x) as usize * 4]
    })
}

#[test]
fn shadows_fall_away_from_the_light() {
    let [west, east] = ground(true);
    assert!(east > 240, "the ground toward the light is lit: {east}");
    assert!(
        west < 160,
        "the ground away from the light is in shadow: {west}"
    );
    // Without casting, both sides are lit alike.
    let [west, east] = ground(false);
    assert_eq!(west, east);
}
