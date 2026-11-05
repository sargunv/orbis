//! Shadows from the scene's directional light (decision 0046): a building
//! or a model lit from the east darkens the ground west of it, away from
//! the light, and not east of it; a building's shadow darkens a model.

use orb_api::{CameraState, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, TileMatrixSet};

/// Metres east → degrees of longitude at the equator.
fn east(m: f64) -> f64 {
    m / 111_320.0
}

/// A scene on a white ground lit from the east 45° up, with `layers`
/// (sources `b`, a 100 m building 40 m across at `building` metres east of
/// the origin, and `p`, a point at the origin) and the demo's 50 m pagoda.
fn scene(cast: bool, building: f64, layers: &str) -> String {
    let half = east(20.0);
    let [w, e] = [building - 20.0, building + 20.0].map(east);
    let models = concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/data/models");
    format!(
        r##"{{
          "style": {{
            "version": 8,
            "sources": {{
              "b": {{"type": "geojson", "data": {{"type": "Polygon",
                "coordinates": [[[{w}, -{half}], [{e}, -{half}], [{e}, {half}], [{w}, {half}], [{w}, -{half}]]]}}}},
              "p": {{"type": "geojson", "data": {{"type": "Point", "coordinates": [0, 0]}}}}
            }},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "#ffffff"}}}},
              {layers}
            ]
          }},
          "models": {{"pagoda": "file://{models}/pagoda.glb"}},
          "lights": [
            {{"type": "ambient", "properties": {{"intensity": 0.5}}}},
            {{"type": "directional", "properties": {{"direction": [90, 45], "intensity": 0.5, "cast-shadows": {cast}}}}}
          ]
        }}"##
    )
}

const BUILDING: &str = r##"{"id": "building", "type": "fill-extrusion", "source": "b",
  "paint": {"fill-extrusion-color": "#888888", "fill-extrusion-height": 100}}"##;
/// White, for its lighting to show.
const PAGODA: &str = r##"{"id": "pagoda", "type": "model", "source": "p",
  "layout": {"model-id": "pagoda"},
  "paint": {"model-color": "#ffffff", "model-color-mix-intensity": 1}}"##;

/// The red of `scene` from above at zoom 17 at each of `at`, metres east
/// of the origin.
fn red<const N: usize>(scene: &str, at: [f64; N]) -> [u8; N] {
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
    map.set_style_json(scene).unwrap();
    let image = map.snapshot(256, 256, 1.0).unwrap();
    at.map(|m| {
        let p = map.project(Geodetic::new(east(m), 0.0)).unwrap();
        let [x, y] = [p.x, p.y].map(|v| v as u32);
        assert!(x < image.width && y < image.height, "{p} is in the view");
        image.pixels[(y * image.width + x) as usize * 4]
    })
}

/// West of a caster at the origin is in shadow; east of it is lit; and
/// without casting both sides are lit alike. The ground is sampled 50 m
/// from the building's centre and 30 m from the pagoda's: past their
/// roofs, which stand nearer the camera, and inside their shadows.
///
/// The scene's lights shade the white ground (decision 0048): both lights
/// give it 0.5 + 0.5 cos 45° = 0.85 (238 in sRGB), the ambient alone 0.5
/// (188).
fn falls_away(layer: &str, away: f64) {
    let [west, east] = red(&scene(true, 0.0, layer), [-away, away]);
    assert!(east > 230, "the ground toward the light is lit: {east}");
    assert!(
        west < 195,
        "the ground away from the light is in shadow: {west}"
    );
    let [west, east] = red(&scene(false, 0.0, layer), [-away, away]);
    assert_eq!(west, east);
}

#[test]
fn shadows_fall_away_from_the_light() {
    falls_away(BUILDING, 50.0);
}

#[test]
fn models_cast_shadows() {
    falls_away(PAGODA, 30.0);
}

/// The pagoda's top roof, seen from straight above beside its spire, is
/// darker in the shadow of a building 40 m east of it than without
/// shadows.
#[test]
fn models_receive_shadows() {
    let layers = format!("{BUILDING}, {PAGODA}");
    let [lit] = red(&scene(false, 40.0, &layers), [-4.0]);
    let [shadowed] = red(&scene(true, 40.0, &layers), [-4.0]);
    // In full shadow the roof keeps the ambient light alone: 0.8 of
    // white (its metal takes the rest) at 0.5, 170 in sRGB.
    assert!(
        shadowed.abs_diff(170) <= 4 && lit > shadowed + 20,
        "the roof in shadow ({shadowed}) keeps the ambient light, darker than lit ({lit})"
    );
}

/// A building out of view toward a low light still casts into it: its
/// tile stays in the cover (#92). The 200 m building 500 m east, under a
/// light 20° up, shades 550 m west of it, past the view's centre, while
/// the view at zoom 17 spans about 150 m.
#[test]
fn casters_out_of_view_cast_into_it() {
    let low = |cast| {
        scene(cast, 520.0, BUILDING)
            .replace("[90, 45]", "[90, 70]")
            .replace(
                "\"fill-extrusion-height\": 100",
                "\"fill-extrusion-height\": 200",
            )
    };
    let [lit] = red(&low(false), [0.0]);
    let [shadowed] = red(&low(true), [0.0]);
    assert_eq!(lit, 255, "the ground is white without shadows");
    assert!(
        shadowed < 200,
        "the ground is in the building's shadow: {shadowed}"
    );
}
