//! v8's `light` for a frame, as shaders take it, and how the view sees
//! the atmosphere.

use orb_crs::Geodetic;
use orb_render::FrameLight;
use orb_style::TranslateAnchor;

use crate::root::LightValues;

/// The light, evaluated, for a camera turned `bearing` degrees.
pub(crate) fn frame_light(light: &LightValues, bearing: f64) -> FrameLight {
    let [r, g, b, _] = light.color;
    let [radius, azimuth, polar] = light.position.map(f64::from);
    // GL JS's `sphericalToCartesian`.
    let (a, p) = ((azimuth + 90.0).to_radians(), polar.to_radians());
    let [x, y, z] = [a.cos() * p.sin(), a.sin() * p.sin(), p.cos()].map(|v| radius * v);
    let (s, c) = match light.anchor {
        TranslateAnchor::Viewport => bearing.to_radians().sin_cos(),
        TranslateAnchor::Map => (0.0, 1.0),
    };
    FrameLight {
        color: [r, g, b],
        intensity: light.intensity,
        position: [x * c - y * s, x * s + y * c, z].map(|v| v as f32),
        shadow_intensity: light.shadow_intensity,
    }
}

/// How `view` of `camera` sees the atmosphere: the sky above the surface's
/// horizon (a globe's limb), fog on terrain standing `terrain_presence`
/// (0 to 1), scattering over a globe lit by `light`.
pub(crate) fn atmosphere_view(
    light: &LightValues,
    terrain_presence: f64,
    view: &orb_camera::View,
    camera: &orb_camera::CameraState,
) -> orb_render::AtmosphereView {
    let (fog_near, fog_far) = view.fog_range_px();
    let (horizon, horizon_normal) = view.sky_horizon_line();
    // GL JS's `calculateFogBlendOpacity`: terrain fog fades in from 60° to
    // 70° of pitch. It fades in with zoom too as a globe hands over: none
    // on the whole low-zoom surface, all of it on the plane. (GL JS turns
    // it off while its globe shows at all, so it snaps on where the
    // handover ends.)
    let pitched = ((camera.pitch - 60.0) / 10.0).clamp(0.0, 1.0);
    let handed_over = 1.0 - view.globe().map_or(0.0, |g| g.amount);
    let fog_opacity = (pitched * handed_over * terrain_presence) as f32;
    orb_render::AtmosphereView {
        horizon: horizon.as_vec2().to_array(),
        horizon_normal: horizon_normal.as_vec2().to_array(),
        height: view.viewport().height as f32,
        fog_range: [fog_near as f32, fog_far as f32],
        fog_opacity,
        ray_inverse: ray_inverse(view, camera.target),
        // The sky over a sphere; a low-zoom plane's is a plane's.
        globe: (view.globe().filter(|g| g.plane.is_none()))
            .map(|g| globe_sky(g, view, camera, light)),
    }
}

/// A globe view's sphere for its sky and scattering, with the sun where
/// v8's `light` is: GL JS's `getSunPos` (in the view for a viewport
/// anchor, fixed to the globe for a map one).
fn globe_sky(
    globe: &orb_camera::GlobeView,
    view: &orb_camera::View,
    camera: &orb_camera::CameraState,
    light: &LightValues,
) -> orb_render::GlobeSky {
    use glam::{DMat3, DVec3};
    let target = camera.target;
    let frame = orb_crs::sphere::frame(target);
    let [_, azimuth, polar] = light.position.map(f64::from);
    let (a, p) = ((azimuth + 90.0).to_radians(), polar.to_radians());
    let v = -DVec3::new(a.cos() * p.sin(), a.sin() * p.sin(), p.cos());
    let sun = match light.anchor {
        // View axes back through the view's pitch and bearing to the
        // target's east, north and up.
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
        let light = LightValues::of(&orb_style::v8::default_light(), 0.0);
        let still = frame_light(&light, 0.0);
        assert_eq!(still.intensity, 0.5);
        let turned = frame_light(&light, 90.0);
        assert!((still.position[2] - turned.position[2]).abs() < 1e-6);
        assert!(
            (still.position[0] - turned.position[1]).abs() < 1e-5,
            "{still:?} {turned:?}"
        );
    }

    /// Terrain fog fades in with zoom through the globe's handover to Web
    /// Mercator rather than snapping on where it ends (#154).
    #[test]
    fn fog_fades_in_through_the_globe_handover() {
        let light = LightValues::of(&orb_style::v8::default_light(), 0.0);
        let tms = orb_crs::TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let mercator = *tms.projection();
        let surface = orb_crs::Surface {
            projection: mercator,
            globe: Some(orb_crs::Globe::V8),
        };
        let viewport = orb_camera::Viewport {
            width: 800.0,
            height: 600.0,
        };
        let fog = |zoom: f64| {
            let target = Geodetic::new(7.75, 46.02);
            let mut camera = orb_camera::CameraState::new(target, zoom, &mercator);
            camera.pitch = 75.0;
            let view = orb_camera::View::new(&camera, viewport, &surface);
            atmosphere_view(&light, 1.0, &view, &camera).fog_opacity
        };
        let band = [10.5, 11.0, 11.25, 11.5, 11.75, 12.0, 12.5].map(fog);
        assert_eq!((band[0], band[1]), (0.0, 0.0));
        assert_eq!(band[2..5], [0.25, 0.5, 0.75], "{band:?}");
        assert_eq!((band[5], band[6]), (1.0, 1.0));
    }
}
