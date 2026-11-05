//! The 4,000-camera Web Mercator pin against the pinned GL JS transform
//! (`generate-pins.mjs`): each camera's screen points unproject where GL JS
//! puts them and project back, and every tenth camera covers the same tiles.

use glam::DVec2;
use orb_camera::{CameraState, View, Viewport};
use orb_crs::TileMatrixSet;
use orb_geo::Geodetic;

/// The screen points each camera unprojects, as fractions of its viewport:
/// all in the lower half, below any horizon.
const FRACTIONS: [[f64; 2]; 5] = [[0.5, 0.5], [0.1, 0.9], [0.9, 0.95], [0.25, 0.6], [0.8, 0.7]];

#[test]
fn cameras_match_pinned_gl_js() {
    let golden: serde_json::Value = serde_json::from_str(include_str!("data/pins.json")).unwrap();
    let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
    let projection = tms.projection();
    let cameras = golden["cameras"].as_array().unwrap();
    assert_eq!(cameras.len(), 4000);
    let (mut worst_px, mut worst_deg, mut covers) = (0.0f64, 0.0f64, 0);
    let mut failures = Vec::new();
    for (i, case) in cameras.iter().enumerate() {
        let n = |key: &str| case[key].as_f64().unwrap();
        let mut camera = CameraState::new(Geodetic::new(n("lon"), n("lat")), n("zoom"), projection);
        camera.pitch = n("pitch");
        camera.bearing = n("bearing");
        camera.fov = n("fov");
        let viewport = Viewport {
            width: n("width"),
            height: n("height"),
        };
        let view = View::new(&camera, viewport, projection);
        let points = case["points"].as_array().unwrap();
        assert_eq!(points.len(), FRACTIONS.len());
        for (point, [u, v]) in points.iter().zip(FRACTIONS) {
            let [lon, lat] = [0, 1].map(|k| point[k].as_f64().unwrap());
            let screen = DVec2::new(u * viewport.width, v * viewport.height);
            // GL JS's location projects to its screen point (longitudes past
            // ±180 are in other world copies, where Web Mercator's x goes on)...
            let world = projection.forward(Geodetic::new(lon, lat));
            // Off the world's top or bottom (past ±85.0511°), the location
            // has no Web Mercator position to project from.
            let on_world = lat.abs() < 85.051_128;
            let px = if on_world {
                view.project(world).unwrap().distance(screen)
            } else {
                0.0
            };
            // ...and its screen point unprojects to its location.
            let ground = projection.inverse(view.unproject(screen).unwrap());
            let dlon = ((ground.lon - lon + 540.0).rem_euclid(360.0) - 180.0).abs();
            let deg = dlon.max((ground.lat - lat).abs());
            worst_px = worst_px.max(px);
            // Compare angles at the camera's scale: a pixel's worth.
            let deg_per_px = 360.0 / (512.0 * n("zoom").exp2());
            worst_deg = worst_deg.max(deg / deg_per_px);
            if px > 1e-2 || deg > 1e-2 * deg_per_px {
                failures.push(format!(
                    "camera {i}: point ({}, {}) off by {px:.2e} px, {:.2e} px unprojected",
                    screen.x,
                    screen.y,
                    deg / deg_per_px
                ));
            }
        }
        if let Some(tiles) = case.get("tiles") {
            covers += 1;
            let mut actual: Vec<[i64; 4]> = view
                .cover(&tms, 512.0, 22)
                .iter()
                .map(|t| {
                    [
                        i64::from(t.wrap),
                        i64::from(t.index.z),
                        i64::from(t.index.col),
                        i64::from(t.index.row),
                    ]
                })
                .collect();
            actual.sort();
            let expected: Vec<[i64; 4]> = serde_json::from_value(tiles.clone()).unwrap();
            if actual != expected {
                let extra = actual.iter().filter(|t| !expected.contains(t)).count();
                let missing = expected.iter().filter(|t| !actual.contains(t)).count();
                failures.push(format!("camera {i}: cover +{extra} −{missing} tiles"));
            }
        }
    }
    println!(
        "4,000 cameras: worst {worst_px:.2e} px projected, {worst_deg:.2e} px unprojected; {covers} covers"
    );
    assert!(
        failures.is_empty(),
        "{} differences:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
