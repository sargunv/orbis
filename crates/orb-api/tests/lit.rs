//! The lit map under a sun: each point sees the sun from
//! where it stands, so a globe has a day side and a night side whatever
//! the camera, and night lights show only where the sun is down.

use orb_api::{CameraState, Geodetic, Map, MapOptions, Runtime, RuntimeOptions, TileMatrixSet};

const SIZE: u32 = 256;

/// A scene of `layers` over `ground` on the globe, under a dim sky and the
/// sun at `time`.
fn scene(time: &str, ground: &str, layers: &str, materials: &str) -> String {
    format!(
        r##"{{
          "style": {{
            "version": 8,
            "projection": {{"type": "globe"}},
            "sources": {{
              "box": {{"type": "geojson", "data": {{"type": "Polygon",
                "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]]}}}}
            }},
            "layers": [
              {{"id": "ground", "type": "background", "paint": {{"background-color": "{ground}"}}}}
              {layers}
            ]
          }},
          "lights": [
            {{"type": "ambient", "properties": {{"intensity": 0.1}}}},
            {{"type": "sun", "properties": {{"time": "{time}", "intensity": 1}}}}
          ],
          "layers": {{{materials}}}
        }}"##
    )
}

/// `scene` seen from over `at` at `zoom`, and the map, to project points.
fn render(scene: &str, at: Geodetic, zoom: f64) -> (Map, Vec<u8>) {
    let runtime = Runtime::new(&RuntimeOptions::default()).unwrap();
    let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
    let mut map = Map::new(
        &runtime,
        MapOptions {
            camera: Some(CameraState::new(at, zoom, tms.projection())),
            ..MapOptions::default()
        },
    );
    map.set_style_json(scene).unwrap();
    let image = map.snapshot(SIZE, SIZE, 1.0).unwrap();
    (map, image.pixels)
}

/// The red at `point` in a rendering.
fn red(map: &Map, pixels: &[u8], point: Geodetic) -> u8 {
    let p = map.project(point).unwrap();
    let [x, y] = [p.x, p.y].map(|v| v as u32);
    assert!(x < SIZE && y < SIZE, "{p} is in the view");
    pixels[(y * SIZE + x) as usize * 4]
}

/// At the March equinox's noon in Greenwich the sun stands over 0°, 0°:
/// seen from over 90° E, the globe's west half is day and its east half
/// night, and a point shades the same seen from either side of it.
#[test]
fn a_globe_has_a_day_side_and_a_night_side() {
    let noon = scene("2024-03-20T12:00:00Z", "#ffffff", "", "");
    let (map, pixels) = render(&noon, Geodetic::new(90.0, 0.0), 0.5);
    let day = red(&map, &pixels, Geodetic::new(50.0, 0.0));
    let night = red(&map, &pixels, Geodetic::new(130.0, 0.0));
    // The sun 40° up at 50° E lights white ground to 0.1 + 0.64; past the
    // terminator only the sky's 0.1 remains.
    assert!(day > 200, "the day side is lit: {day}");
    assert!(night < 100, "the night side is dark: {night}");
    // The light at 50° E doesn't depend on where the camera looks from.
    let (map, pixels) = render(&noon, Geodetic::new(20.0, 0.0), 0.5);
    let again = red(&map, &pixels, Geodetic::new(50.0, 0.0));
    assert!(
        again.abs_diff(day) <= 4,
        "50° E shades alike from 90° E ({day}) and 20° E ({again})"
    );
}

/// A night light adds its colour where the sun is down and nothing by day.
#[test]
fn night_lights_show_at_night() {
    let lights = r##", {"id": "lights", "type": "fill", "source": "box",
        "paint": {"fill-color": "#ff0000"}}"##;
    // A grey ground that shows its own colour whatever the sun.
    let materials = r#""ground": {"emissive-strength": 1}, "lights": {"night-light": 1}"#;
    let centre = |time: &str| {
        let scene = scene(time, "#808080", lights, materials);
        let (map, pixels) = render(&scene, Geodetic::new(0.0, 0.0), 6.0);
        let p = map.project(Geodetic::new(0.0, 0.0)).unwrap();
        let at = ((p.y as u32) * SIZE + p.x as u32) as usize * 4;
        [pixels[at], pixels[at + 1], pixels[at + 2]]
    };
    assert_eq!(centre("2024-03-20T12:00:00Z"), [128; 3], "nothing by day");
    assert_eq!(
        centre("2024-03-20T00:00:00Z"),
        [255, 128, 128],
        "red light added at midnight"
    );
}
