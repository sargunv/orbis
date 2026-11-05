//! The scene's lights for a frame (decision 0026): the scene document's
//! lights folded into what shaders take, the sun placed for the camera's
//! target.

use orb_geo::Geodetic;
use orb_render::FrameLights;
use orb_v8::{Direction, Lights, TranslateAnchor};

/// The lights at `zoom` for a camera turned `bearing` degrees over `target`.
pub(crate) fn frame_lights(
    lights: &Lights,
    zoom: f64,
    bearing: f64,
    target: Geodetic,
) -> FrameLights {
    let rgb = |c: orb_v8::Color| {
        let [r, g, b, _] = c.premultiplied();
        [r, g, b]
    };
    let mut frame = FrameLights {
        ambient: [0.0; 3],
        ambient_level: 0.0,
        directional: [0.0; 3],
        directional_level: 0.0,
        position: [0.0; 3],
        terrain: lights.terrain,
    };
    if let Some(a) = &lights.ambient {
        frame.ambient = rgb(a.color.evaluate(zoom));
        frame.ambient_level = a.level.evaluate(zoom);
    }
    if let Some(d) = &lights.directional {
        let [radius, azimuth, polar] = match d.direction {
            Direction::Position(ref p) => p.evaluate(zoom).map(f64::from),
            Direction::Sun { time } => {
                let (azimuth, altitude) = sun_position(time, target);
                // The sun fades out over its last two degrees.
                [
                    (altitude / 2.0).clamp(0.0, 1.0),
                    azimuth,
                    90.0 - altitude.max(0.0),
                ]
            }
        };
        // GL JS's `sphericalToCartesian`.
        let (a, p) = ((azimuth + 90.0).to_radians(), polar.to_radians());
        let [x, y, z] = [a.cos() * p.sin(), a.sin() * p.sin(), p.cos()].map(|v| radius * v);
        let (s, c) = match d.anchor.evaluate(zoom) {
            TranslateAnchor::Viewport => bearing.to_radians().sin_cos(),
            TranslateAnchor::Map => (0.0, 1.0),
        };
        frame.position = [x * c - y * s, x * s + y * c, z].map(|v| v as f32);
        frame.directional = rgb(d.color.evaluate(zoom));
        frame.directional_level = d.level.evaluate(zoom);
    }
    frame
}

/// A globe view's sphere for its sky and scattering, with the sun the
/// style's directional light gives: GL JS's `getSunPos` for a positioned
/// light (in the view for a viewport anchor, fixed to the globe for a map
/// one), the real sun for a scene's `sun`, overhead without one.
pub(crate) fn globe_sky(
    globe: &orb_camera::GlobeView,
    view: &orb_camera::View,
    camera: &orb_camera::CameraState,
    lights: Option<&Lights>,
    zoom: f64,
) -> orb_render::GlobeSky {
    use glam::{DMat3, DVec3};
    let target = camera.target;
    let frame = orb_crs::sphere::frame(target);
    let directional = lights.and_then(|l| l.directional.as_ref());
    let sun = match directional.map(|d| (&d.direction, d.anchor.evaluate(zoom))) {
        Some((Direction::Position(p), anchor)) => {
            let [_, azimuth, polar] = p.evaluate(zoom).map(f64::from);
            let (a, p) = ((azimuth + 90.0).to_radians(), polar.to_radians());
            let v = -DVec3::new(a.cos() * p.sin(), a.sin() * p.sin(), p.cos());
            match anchor {
                // View axes back through the view's pitch and bearing to
                // the target's east, north and up.
                TranslateAnchor::Viewport => {
                    frame
                        * DMat3::from_rotation_z(-camera.bearing.to_radians())
                        * DMat3::from_rotation_x(camera.pitch.to_radians())
                        * v
                }
                TranslateAnchor::Map => {
                    frame
                        * DMat3::from_rotation_x(target.lat.to_radians())
                        * DMat3::from_rotation_y(-target.lon.to_radians())
                        * v
                }
            }
        }
        Some((Direction::Sun { time }, _)) => {
            let (azimuth, altitude) = sun_position(*time, target);
            let (a, h) = (azimuth.to_radians(), altitude.to_radians());
            frame * DVec3::new(a.sin() * h.cos(), a.cos() * h.cos(), h.sin())
        }
        None => frame * DVec3::Z,
    };
    let eye = globe.target + globe.eye;
    orb_render::GlobeSky {
        inverse_view_proj: globe.view_proj.inverse().as_mat4().to_cols_array_2d(),
        eye: eye.as_vec3().to_array(),
        radius: orb_crs::sphere::RADIUS as f32,
        amount: globe.amount as f32,
        eye_distance_px: view.eye_distance_px() as f32,
        sun: sun.normalize().as_vec3().to_array(),
    }
}

/// Where the sun stands at `time` (Unix seconds) over `at`: azimuth in
/// degrees clockwise from north and altitude in degrees above the horizon,
/// by the Astronomical Almanac's low-precision formulas (about 0.01° from
/// 1950 to 2050), without refraction.
pub(crate) fn sun_position(time: f64, at: Geodetic) -> (f64, f64) {
    // Days since J2000.0.
    let n = time / 86_400.0 + 2_440_587.5 - 2_451_545.0;
    let mean_longitude = (280.460 + 0.985_647_4 * n).rem_euclid(360.0);
    let anomaly = (357.528 + 0.985_600_3 * n).rem_euclid(360.0).to_radians();
    let ecliptic =
        (mean_longitude + 1.915 * anomaly.sin() + 0.020 * (2.0 * anomaly).sin()).to_radians();
    let obliquity = (23.439 - 0.000_000_4 * n).to_radians();
    let right_ascension = (obliquity.cos() * ecliptic.sin()).atan2(ecliptic.cos());
    let declination = (obliquity.sin() * ecliptic.sin()).asin();
    let sidereal = (280.460_618_37 + 360.985_647_366_29 * n).rem_euclid(360.0);
    let hour = (sidereal + at.lon).to_radians() - right_ascension;
    let lat = at.lat.to_radians();
    let altitude =
        (lat.sin() * declination.sin() + lat.cos() * declination.cos() * hour.cos()).asin();
    let azimuth = (-hour.sin()).atan2(declination.tan() * lat.cos() - lat.sin() * hour.cos());
    (
        azimuth.to_degrees().rem_euclid(360.0),
        altitude.to_degrees(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(time: &str, lon: f64, lat: f64) -> (f64, f64) {
        sun_position(orb_v8::parse_time(time).unwrap(), Geodetic::new(lon, lat))
    }

    #[test]
    fn the_sun_stands_where_almanacs_put_it() {
        // Solar noon at Greenwich on the June solstice: due south, at
        // 90° − 51.48° + 23.44°.
        let (azimuth, altitude) = at("2024-06-21T12:02:00Z", 0.0, 51.48);
        assert!((azimuth - 180.0).abs() < 1.0, "{azimuth}");
        assert!((altitude - 61.96).abs() < 0.2, "{altitude}");
        // The March equinox's noon on the equator is overhead.
        let (_, altitude) = at("2024-03-20T12:07:00Z", 0.0, 0.0);
        assert!(altitude > 89.0, "{altitude}");
        // Morning in Zermatt on the solstice: in the east-northeast, low.
        let (azimuth, altitude) = at("2024-06-21T04:30:00Z", 7.75, 46.02);
        assert!((50.0..80.0).contains(&azimuth), "{azimuth}");
        assert!((5.0..15.0).contains(&altitude), "{altitude}");
        // Midnight there is below the horizon.
        assert!(at("2024-06-21T23:00:00Z", 7.75, 46.02).1 < 0.0);
    }

    #[test]
    fn lights_lower_into_the_frame() {
        let target = Geodetic::new(7.75, 46.02);
        // v8's default light: ambient 1 − 0.5, directional 2 · 0.5, at
        // [1.15, 210, 30], turning with the map.
        let v8 = frame_lights(&Lights::default(), 0.0, 0.0, target);
        assert_eq!((v8.ambient_level, v8.directional_level), (0.5, 1.0));
        assert!(!v8.terrain);
        let turned = frame_lights(&Lights::default(), 0.0, 90.0, target);
        assert!((v8.position[2] - turned.position[2]).abs() < 1e-6);
        assert!(
            (v8.position[0] - turned.position[1]).abs() < 1e-5,
            "{v8:?} {turned:?}"
        );
        // A scene's sun below the horizon gives no directional light.
        let night = orb_v8::parse_scene(
            r#"{"style": "s.json", "lights": [{"type": "sun", "properties": {"time": "2024-06-21T23:00:00Z"}}]}"#,
        )
        .unwrap();
        let frame = frame_lights(&night.lights.unwrap(), 0.0, 0.0, target);
        assert_eq!(frame.position, [0.0; 3]);
        assert!(frame.terrain);
    }
}
