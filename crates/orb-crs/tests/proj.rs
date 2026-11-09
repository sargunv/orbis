//! Every family against PROJ: forward within 0.1 mm, inverse within
//! a nano-degree, over sample positions in each code's area of use
//! (`proj_reference.py` writes the reference).

use orb_crs::Geodetic;
use orb_crs::{Descriptor, Projection};
use serde_json::Value;

#[test]
fn families_agree_with_proj() {
    let reference: Value =
        serde_json::from_str(include_str!("proj_reference.json")).expect("reference parses");
    let mut checked = 0;
    for case in reference["cases"].as_array().unwrap() {
        let d = case["descriptor"].as_object().unwrap();
        let descriptor = Descriptor {
            kind: d["type"].as_str().unwrap().to_string(),
            params: d
                .iter()
                .filter(|(k, _)| *k != "type")
                .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
                .collect(),
        };
        let projection =
            Projection::new(&descriptor).unwrap_or_else(|e| panic!("{}: {e}", descriptor.kind));
        // Geographic planes are in degrees; everything else in metres.
        let tolerance = if descriptor.kind == "EPSG:4326" {
            1e-9
        } else {
            1e-4
        };
        for point in case["points"].as_array().unwrap() {
            let [lon, lat, x, y] = [0, 1, 2, 3].map(|i| point[i].as_f64().unwrap());
            let g = Geodetic::new(lon, lat);
            let p = projection.forward(g);
            assert!(
                (p.x - x).abs() < tolerance && (p.y - y).abs() < tolerance,
                "{}: forward {g:?} = {p}, PROJ ({x}, {y})",
                descriptor.kind
            );
            let back = projection.inverse(glam::DVec2::new(x, y));
            let dlon = ((back.lon - lon + 540.0).rem_euclid(360.0) - 180.0).abs();
            assert!(
                dlon * lat.to_radians().cos() < 1e-9 && (back.lat - lat).abs() < 1e-9,
                "{}: inverse ({x}, {y}) = {back:?}, PROJ ({lon}, {lat})",
                descriptor.kind
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 43 * 25);
}
