//! The scene's lights for a frame: the style's lights folded into what
//! shaders take, the sun placed for the camera's target.

use orb_crs::Geodetic;
use orb_render::{FrameLights, FrameShadow};
use orb_style::{Direction, Lights, TranslateAnchor};

/// The lights at `zoom` for a camera turned `bearing` degrees over `target`,
/// with shadows falling between `heights` (world units) if they cast them.
pub(crate) fn frame_lights(
    lights: &Lights,
    zoom: f64,
    bearing: f64,
    target: Geodetic,
    heights: [f32; 2],
) -> FrameLights {
    let rgb = |c: orb_style::Color| {
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
        shadow: None,
    };
    if let Some(a) = &lights.ambient {
        frame.ambient = rgb(a.color.evaluate(zoom));
        frame.ambient_level = a.level.evaluate(zoom);
    }
    if let Some(d) = &lights.directional {
        let [radius, azimuth, polar] = match d.direction {
            Direction::Position(ref p) => p.evaluate(zoom).map(f64::from),
            // Where the sun stands from the target, below its horizon
            // too: shaders fade it out over its last two degrees above each
            // point's own horizon, so a globe's day side stays lit.
            Direction::Sun { time } => {
                let (azimuth, altitude) = sun_position(time, target);
                [1.0, azimuth, 90.0 - altitude]
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
        frame.shadow = d.shadow.map(|intensity| FrameShadow { intensity, heights });
    }
    frame
}

/// The atmosphere of `scene` at `zoom` and how `view` of `camera` sees it:
/// the sky above the surface's horizon (a globe's limb), fog on terrain,
/// scattering over a globe.
pub(crate) fn atmosphere(
    scene: Option<&orb_style::Scene>,
    view: &orb_camera::View,
    camera: &orb_camera::CameraState,
    zoom: f64,
) -> (
    Option<orb_render::AtmospherePaint>,
    orb_render::AtmosphereView,
) {
    let lights = scene.map(|s| &s.lights);
    let paint = scene
        .and_then(|s| s.atmosphere.as_ref())
        .map(|a| orb_render::AtmospherePaint {
            sky_color: a.sky_color.evaluate(zoom).premultiplied(),
            horizon_color: a.horizon_color.evaluate(zoom).premultiplied(),
            fog_color: a.fog_color.evaluate(zoom).premultiplied(),
            sky_horizon_blend: a.sky_horizon_blend.evaluate(zoom),
            horizon_fog_blend: a.horizon_fog_blend.evaluate(zoom),
            fog_ground_blend: a.fog_ground_blend.evaluate(zoom),
            atmosphere_blend: a.atmosphere_blend.evaluate(zoom),
        });
    let (fog_near, fog_far) = view.fog_range_px();
    let (horizon, horizon_normal) = view.sky_horizon_line();
    // GL JS's `calculateFogBlendOpacity`: terrain fog fades in from 60° to
    // 70° of pitch.
    let fog_opacity = ((camera.pitch - 60.0) / 10.0).clamp(0.0, 1.0) as f32;
    let at = orb_render::AtmosphereView {
        horizon: horizon.as_vec2().to_array(),
        horizon_normal: horizon_normal.as_vec2().to_array(),
        height: view.viewport().height as f32,
        fog_range: [fog_near as f32, fog_far as f32],
        fog_opacity,
        ray_inverse: ray_inverse(view, camera.target),
        // The sky over a sphere; a low-zoom plane's is a plane's.
        globe: (view.globe().filter(|g| g.plane.is_none()))
            .map(|g| globe_sky(g, view, camera, lights, zoom)),
    };
    (paint, at)
}

/// A globe view's sphere for its sky and scattering, with the sun the
/// style's directional light gives: GL JS's `getSunPos` for a positioned
/// light (in the view for a viewport anchor, fixed to the globe for a map
/// one), the real sun for a scene's `sun`, overhead without one.
fn globe_sky(
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
        eye: eye.as_vec3().to_array(),
        radius: orb_crs::sphere::RADIUS as f32,
        amount: globe.amount as f32,
        eye_distance_px: view.eye_distance_px() as f32,
        sun: sun.normalize().as_vec3().to_array(),
    }
}

/// The view ray's matrix for the atmosphere: clip → camera-relative space
/// on Earth-fixed axes. On a flat surface, camera-relative world space is
/// east, north and up at the target (projected coordinates are eastings
/// and northings), turned onto the Earth's axes there.
fn ray_inverse(view: &orb_camera::View, target: Geodetic) -> [[f32; 4]; 4] {
    let inverse = match view.globe().filter(|g| g.plane.is_none()) {
        Some(globe) => globe.view_proj.inverse(),
        None => glam::DMat4::from_mat3(orb_crs::sphere::frame(target)) * view.view_proj().inverse(),
    };
    inverse.as_mat4().to_cols_array_2d()
}

/// Greenwich mean sidereal time at `time` (Unix seconds), in degrees.
fn sidereal_degrees(time: f64) -> f64 {
    let n = time / 86_400.0 + 2_440_587.5 - 2_451_545.0;
    (280.460_618_37 + 360.985_647_366_29 * n).rem_euclid(360.0)
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
    let sidereal = sidereal_degrees(time);
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

    #[test]
    fn lights_lower_into_the_frame() {
        let target = Geodetic::new(7.75, 46.02);
        // v8's default light: ambient 1 − 0.5, directional 2 · 0.5, at
        // [1.15, 210, 30], turning with the map.
        let v8 = frame_lights(&orb_v8::default_lights(), 0.0, 0.0, target, [0.0; 2]);
        assert_eq!((v8.ambient_level, v8.directional_level), (0.5, 1.0));
        assert!(!v8.terrain);
        let turned = frame_lights(&orb_v8::default_lights(), 0.0, 90.0, target, [0.0; 2]);
        assert!((v8.position[2] - turned.position[2]).abs() < 1e-6);
        assert!(
            (v8.position[0] - turned.position[1]).abs() < 1e-5,
            "{v8:?} {turned:?}"
        );
    }
}
