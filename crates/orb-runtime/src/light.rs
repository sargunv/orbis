//! v8's `light` for a frame, as shaders take it, and how the view sees
//! the atmosphere.

use orb_crs::Geodetic;
use orb_render::FrameLight;
use orb_style::TranslateAnchor;

use crate::root::LightValues;

/// The light, evaluated, for `camera`.
pub(crate) fn frame_light(light: &LightValues, camera: &orb_camera::CameraState) -> FrameLight {
    let [r, g, b, _] = light.color;
    let [radius, azimuth, polar] = light.position.map(f64::from);
    // GL JS's `sphericalToCartesian`.
    let (a, p) = ((azimuth + 90.0).to_radians(), polar.to_radians());
    let [mut x, mut y, mut z] = [a.cos() * p.sin(), a.sin() * p.sin(), p.cos()].map(|v| radius * v);
    let (s, c) = match light.anchor {
        TranslateAnchor::Viewport => camera.bearing.to_radians().sin_cos(),
        TranslateAnchor::Map => (0.0, 1.0),
    };
    if light.anchor == TranslateAnchor::Map && light.timed {
        // The sun as it stands over the target, in the same axes (west,
        // north, up), its direct light fading out as it sets.
        let sun = sun_direction(light);
        let local = orb_crs::sphere::frame(camera.target).transpose() * sun;
        let half = (f64::from(light.twilight) / 2.0).to_radians().sin();
        let day = if half > 0.0 {
            smoothstep(-half, half, local.z)
        } else {
            f64::from(u8::from(local.z >= 0.0))
        };
        [x, y, z] = [-local.x, local.y, local.z].map(|v| radius * day * v);
    }
    FrameLight {
        color: [r, g, b],
        intensity: light.intensity,
        position: [x * c - y * s, x * s + y * c, z].map(|v| v as f32),
        shadow_intensity: light.shadow_intensity,
        ambient: [0, 1, 2].map(|k| light.ambient_color[k] * light.ambient_intensity),
        ambient_amount: light.ambient,
        ground: None,
    }
}

/// How `view` of `camera` sees the atmosphere: the sky above the surface's
/// horizon (a globe's limb), fog on terrain standing `terrain_presence`
/// (0 to 1) and drawn at `exaggeration`, scattering over a globe lit by
/// `light`.
pub(crate) fn atmosphere_view(
    light: &LightValues,
    terrain_presence: f64,
    exaggeration: f64,
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
        ground: ground_light(light, camera, exaggeration),
    }
}

/// The highest ground on Earth, metres.
const HIGHEST_GROUND: f64 = 8849.0;

/// How the light shades the ground (`dev.sargunv.orbis:ground-lighting`),
/// when it does, with terrain drawn at `exaggeration` (0 without).
fn ground_light(
    light: &LightValues,
    camera: &orb_camera::CameraState,
    exaggeration: f64,
) -> Option<orb_render::GroundLight> {
    if light.ground_lighting <= 0.0 {
        return None;
    }
    let [r, g, b, a] = light.ambient_color;
    let i = light.ambient_intensity;
    let up = orb_crs::sphere::up(camera.target);
    Some(orb_render::GroundLight {
        amount: light.ground_lighting.min(1.0),
        day: light.color,
        night: [r * i, g * i, b * i, a],
        twilight: light.twilight,
        sun: toward_light(camera, light).as_vec3().to_array(),
        up: up.as_vec3().to_array(),
        // As high as Everest, exaggerated.
        relief: (exaggeration * HIGHEST_GROUND / orb_crs::sphere::RADIUS) as f32,
    })
}

/// A globe view's sphere for its sky and scattering, lit by the sun the
/// extrusions and the ground take (`toward_light`), whether or not the
/// light shades the ground, so that shading the ground darkens its night
/// side and moves nothing. (GL JS's atmosphere turns the light's position
/// half a turn about the east axis, which puts the default sun behind the
/// globe; see `docs/style-support.md`.)
fn globe_sky(
    globe: &orb_camera::GlobeView,
    view: &orb_camera::View,
    camera: &orb_camera::CameraState,
    light: &LightValues,
) -> orb_render::GlobeSky {
    let eye = globe.target + globe.eye;
    orb_render::GlobeSky {
        eye: eye.as_vec3().to_array(),
        radius: orb_crs::sphere::RADIUS as f32,
        amount: globe.amount as f32,
        eye_distance_px: view.eye_distance_px() as f32,
        sun: toward_light(camera, light).as_vec3().to_array(),
    }
}

/// Toward the light, unit, on Earth-fixed axes, as v8 places it: the
/// polar angle from the zenith and the azimuth clockwise from the top of
/// the viewport (a viewport anchor) or from north (a map one). A
/// viewport's light stands over the camera's target, turning with the
/// bearing as the extrusions' does (tilting the view doesn't move it); a
/// map's is fixed to the globe, standing so over null island (0°, 0°), so
/// the night side turns with the Earth.
fn toward_light(camera: &orb_camera::CameraState, light: &LightValues) -> glam::DVec3 {
    match light.anchor {
        TranslateAnchor::Viewport => {
            let [_, azimuth, polar] = light.position;
            over(
                camera.target,
                f64::from(azimuth) + camera.bearing,
                polar.into(),
            )
        }
        TranslateAnchor::Map => sun_direction(light),
    }
}

/// Toward a light at `azimuth` (clockwise from north) and `polar` degrees
/// from the zenith over `at`, unit, on Earth-fixed axes.
fn over(at: Geodetic, azimuth: f64, polar: f64) -> glam::DVec3 {
    let (a, p) = (azimuth.to_radians(), polar.to_radians());
    let local = glam::DVec3::new(p.sin() * a.sin(), p.sin() * a.cos(), p.cos());
    (orb_crs::sphere::frame(at) * local).normalize()
}

/// Toward the map-anchored sun, unit, on Earth-fixed axes.
fn sun_direction(light: &LightValues) -> glam::DVec3 {
    let sun = glam::DVec3::from(light.sun.map(f64::from));
    sun.try_normalize().unwrap_or(glam::DVec3::Z)
}

/// Toward a map-anchored sun on Earth-fixed axes: overhead where it is at
/// `dev.sargunv.orbis:datetime`, else as `position` places it over 0°, 0°.
pub(crate) fn map_sun(light: &orb_style::Light, zoom: f64) -> [f32; 3] {
    let sun = match light.datetime {
        Some(ms) => orb_crs::sphere::up(subsolar(ms)),
        None => {
            let [_, azimuth, polar] = light.position.evaluate(zoom);
            over(Geodetic::new(0.0, 0.0), azimuth.into(), polar.into())
        }
    };
    sun.as_vec3().to_array()
}

/// Where the sun is overhead at `ms` milliseconds since 1970 UTC: the
/// Astronomical Almanac's low-precision solar coordinates (about 0.01°
/// from 1950 to 2050) and the Greenwich mean sidereal time.
fn subsolar(ms: f64) -> Geodetic {
    // Days since J2000.0, 2000-01-01T12:00Z.
    let d = ms / 86_400_000.0 - 10_957.5;
    let mean_longitude = 280.460 + 0.985_647_4 * d;
    let anomaly = (357.528 + 0.985_600_3 * d).to_radians();
    let longitude =
        (mean_longitude + 1.915 * anomaly.sin() + 0.020 * (2.0 * anomaly).sin()).to_radians();
    let obliquity = (23.439 - 0.000_000_4 * d).to_radians();
    let right_ascension = (obliquity.cos() * longitude.sin()).atan2(longitude.cos());
    let declination = (obliquity.sin() * longitude.sin()).asin();
    let sidereal = 280.460_618_37 + 360.985_647_366_29 * d;
    let lon = (right_ascension.to_degrees() - sidereal).rem_euclid(360.0);
    Geodetic::new(
        if lon > 180.0 { lon - 360.0 } else { lon },
        declination.to_degrees(),
    )
}

fn smoothstep(low: f64, high: f64, x: f64) -> f64 {
    let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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

    /// The sun overhead at a solstice's and an equinox's moments, against
    /// published ephemerides.
    #[test]
    fn the_sun_stands_where_it_does_at_a_time() {
        // 2026-06-21T12:00Z: declination 23.43°, solar noon at Greenwich
        // 1.6 minutes later (the equation of time), so 0.4° east.
        let noon = subsolar(1_782_043_200_000.0);
        assert!((noon.lat - 23.43).abs() < 0.02, "{noon:?}");
        assert!((noon.lon - 0.4).abs() < 0.1, "{noon:?}");
        // 2026-03-20T14:46Z, the equinox: on the equator, the mean sun's
        // 41.5° W less the equation of time's 7.4 minutes, 1.85°.
        let equinox = subsolar(1_774_017_960_000.0);
        assert!(equinox.lat.abs() < 0.02, "{equinox:?}");
        assert!((equinox.lon - -39.65).abs() < 0.1, "{equinox:?}");
    }

    /// With a time, a map-anchored light's extrusions take the sun as it
    /// stands over the target, and none of its direct light once it has set.
    #[test]
    fn a_timed_sun_lights_the_extrusions_as_it_stands_over_the_target() {
        let style = |time: &str| {
            orb_style::v8::parse_with(
                &format!(
                    r#"{{"version": 8, "sources": {{}}, "layers": [],
                        "light": {{"anchor": "map", "dev.sargunv.orbis:datetime": "{time}"}}}}"#
                ),
                orb_style::Previews::ALL,
                &[],
            )
            .unwrap()
        };
        let tms = orb_crs::TileMatrixSet::web_mercator();
        let camera =
            |lon| orb_camera::CameraState::new(Geodetic::new(lon, 0.0), 2.0, tms.projection());
        let light =
            |time, lon| frame_light(&LightValues::of(&style(time).light, 0.0), &camera(lon));
        // Noon at Greenwich in June: high in the north, as GL JS's axes put it.
        let [x, y, z] = light("2026-06-21T12:00:00Z", 0.0).position;
        let radius = 1.15;
        assert!(
            (z / radius - 23.43f32.to_radians().cos()).abs() < 0.01,
            "{z}"
        );
        assert!(y > 0.0 && x.abs() < 0.01, "{x} {y}");
        // Six hours later the sun sets there: low in the west...
        let [x, _, z] = light("2026-06-21T17:00:00Z", 0.0).position;
        assert!(x > 0.5 && z > 0.0 && z < 0.4, "{x} {z}");
        // ...and stands over 90° W.
        let [x, _, z] = light("2026-06-21T18:00:00Z", -90.0).position;
        assert!(z > 1.0 && x.abs() < 0.1, "{x} {z}");
        // Midnight: no direct light, and so no shadows.
        let night = light("2026-06-21T00:00:00Z", 0.0);
        assert_eq!(night.position, [0.0; 3]);
        assert_eq!(night.shadow().1, 0.0);
    }

    #[test]
    fn the_light_turns_with_a_viewport_anchor() {
        // v8's default light: intensity 0.5 at [1.15, 210, 30], turning
        // with the map.
        let light = LightValues::of(&orb_style::v8::default_light(), 0.0);
        let tms = orb_crs::TileMatrixSet::web_mercator();
        let mut camera =
            orb_camera::CameraState::new(Geodetic::new(0.0, 0.0), 2.0, tms.projection());
        let still = frame_light(&light, &camera);
        assert_eq!(still.intensity, 0.5);
        camera.bearing = 90.0;
        let turned = frame_light(&light, &camera);
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
        let tms = orb_crs::TileMatrixSet::web_mercator();
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
            atmosphere_view(&light, 1.0, 1.0, &view, &camera).fog_opacity
        };
        let band = [10.5, 11.0, 11.25, 11.5, 11.75, 12.0, 12.5].map(fog);
        assert_eq!((band[0], band[1]), (0.0, 0.0));
        assert_eq!(band[2..5], [0.25, 0.5, 0.75], "{band:?}");
        assert_eq!((band[5], band[6]), (1.0, 1.0));
    }

    /// The haze is lit by the ground's sun whether or not the light shades
    /// the ground, so easing ground lighting in moves nothing.
    #[test]
    fn the_haze_and_the_ground_share_one_sun() {
        let tms = orb_crs::TileMatrixSet::web_mercator();
        let mercator = *tms.projection();
        let surface = orb_crs::Surface {
            projection: mercator,
            globe: Some(orb_crs::Globe::V8),
        };
        let viewport = orb_camera::Viewport {
            width: 800.0,
            height: 600.0,
        };
        let camera = orb_camera::CameraState::new(Geodetic::new(10.0, 30.0), 1.5, &mercator);
        let view = orb_camera::View::new(&camera, viewport, &surface);
        let globe = view.globe().unwrap();
        for anchor in [TranslateAnchor::Viewport, TranslateAnchor::Map] {
            let mut light = LightValues::of(&orb_style::v8::default_light(), 0.0);
            light.anchor = anchor;
            light.ground_lighting = 1.0;
            let ground = glam::Vec3::from(ground_light(&light, &camera, 0.0).unwrap().sun);
            for amount in [0.0, 0.5, 1.0] {
                light.ground_lighting = amount;
                let haze = glam::Vec3::from(globe_sky(globe, &view, &camera, &light).sun);
                assert!(
                    haze.angle_between(ground) < 1e-6,
                    "{anchor:?} {amount}: {haze} {ground}"
                );
            }
        }
        // v8's default viewport light stands over the target, 60° up: the
        // haze's sun is on the eye's side of the globe, not behind it.
        let light = LightValues::of(&orb_style::v8::default_light(), 0.0);
        let sun = glam::DVec3::from(globe_sky(globe, &view, &camera, &light).sun.map(f64::from));
        let eye = globe.target + globe.eye;
        assert!(sun.dot(eye.normalize()) > 0.5, "{sun} {eye}");
    }
}
