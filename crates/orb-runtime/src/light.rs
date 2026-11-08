//! v8's `light` for a frame, as shaders take it, and the atmosphere.

use orb_crs::Geodetic;
use orb_render::FrameLight;
use orb_style::{Light, TranslateAnchor};

/// The light at `zoom` for a camera turned `bearing` degrees.
pub(crate) fn frame_light(light: &Light, zoom: f64, bearing: f64) -> FrameLight {
    let [r, g, b, _] = light.color.evaluate(zoom).premultiplied();
    let [radius, azimuth, polar] = light.position.evaluate(zoom).map(f64::from);
    // GL JS's `sphericalToCartesian`.
    let (a, p) = ((azimuth + 90.0).to_radians(), polar.to_radians());
    let [x, y, z] = [a.cos() * p.sin(), a.sin() * p.sin(), p.cos()].map(|v| radius * v);
    let (s, c) = match light.anchor.evaluate(zoom) {
        TranslateAnchor::Viewport => bearing.to_radians().sin_cos(),
        TranslateAnchor::Map => (0.0, 1.0),
    };
    FrameLight {
        color: [r, g, b],
        intensity: light.intensity.evaluate(zoom),
        position: [x * c - y * s, x * s + y * c, z].map(|v| v as f32),
    }
}

/// The atmosphere of `style` at `zoom` and how `view` of `camera` sees it:
/// the sky above the surface's horizon (a globe's limb), fog on terrain,
/// scattering over a globe.
pub(crate) fn atmosphere(
    style: Option<&orb_style::Style>,
    view: &orb_camera::View,
    camera: &orb_camera::CameraState,
    zoom: f64,
) -> (
    Option<orb_render::AtmospherePaint>,
    orb_render::AtmosphereView,
) {
    let light = style.map(|s| &s.light);
    let paint = style
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
            .map(|g| globe_sky(g, view, camera, light, zoom)),
    };
    (paint, at)
}

/// A globe view's sphere for its sky and scattering, with the sun where
/// v8's `light` is: GL JS's `getSunPos` (in the view for a viewport
/// anchor, fixed to the globe for a map one), overhead without one.
fn globe_sky(
    globe: &orb_camera::GlobeView,
    view: &orb_camera::View,
    camera: &orb_camera::CameraState,
    light: Option<&Light>,
    zoom: f64,
) -> orb_render::GlobeSky {
    use glam::{DMat3, DVec3};
    let target = camera.target;
    let frame = orb_crs::sphere::frame(target);
    let sun = match light {
        Some(light) => {
            let [_, azimuth, polar] = light.position.evaluate(zoom).map(f64::from);
            let (a, p) = ((azimuth + 90.0).to_radians(), polar.to_radians());
            let v = -DVec3::new(a.cos() * p.sin(), a.sin() * p.sin(), p.cos());
            match light.anchor.evaluate(zoom) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_light_turns_with_a_viewport_anchor() {
        // v8's default light: intensity 0.5 at [1.15, 210, 30], turning
        // with the map.
        let light = orb_style::v8::default_light();
        let still = frame_light(&light, 0.0, 0.0);
        assert_eq!(still.intensity, 0.5);
        let turned = frame_light(&light, 0.0, 90.0);
        assert!((still.position[2] - turned.position[2]).abs() < 1e-6);
        assert!(
            (still.position[0] - turned.position[1]).abs() < 1e-5,
            "{still:?} {turned:?}"
        );
    }
}
