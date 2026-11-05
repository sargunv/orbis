//! The free camera (GL JS's `FreeCameraOptions`): a camera given by its
//! eye, where it is and which way it looks, rather than by the point it
//! orbits.

use glam::DVec2;
use orb_crs::{Geodetic, Surface};

use crate::{CameraState, View, Viewport};

/// A camera's eye and orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FreeCamera {
    /// The point under the eye.
    pub position: Geodetic,
    /// The eye's height above sea level, metres.
    pub altitude: f64,
    /// Degrees clockwise from north the eye faces.
    pub bearing: f64,
    /// Degrees from looking straight down.
    pub pitch: f64,
    /// Degrees about the view axis.
    pub roll: f64,
}

/// How far ahead a camera looking at or above the horizon puts its
/// target, metres (GL JS's).
const LEVEL_DISTANCE_M: f64 = 10_000.0;

impl CameraState {
    /// This camera's eye in `viewport` on `surface`'s plane.
    pub fn free(&self, viewport: Viewport, surface: &Surface) -> FreeCamera {
        let view = View::new(self, viewport, &Surface::plane(surface.projection));
        let eye = view.eye();
        FreeCamera {
            position: surface
                .projection
                .inverse(view.camera_world() + eye.truncate())
                .wrapped(),
            altitude: eye.z / view.height_scale(),
            bearing: self.bearing,
            pitch: self.pitch,
            roll: self.roll,
        }
    }

    /// The camera with `free`'s eye, keeping this one's field of view and
    /// padding: it orbits the sea-level point the eye looks at, or one
    /// 10 km ahead when it looks at or above the horizon (GL JS's
    /// `calculateCameraOptionsFromCameraLngLatAltRotation`).
    pub fn from_free(&self, free: &FreeCamera, viewport: Viewport, surface: &Surface) -> Self {
        let projection = &surface.projection;
        let eye = projection.forward(free.position.wrapped());
        let pitch = free.pitch.to_radians();
        let (distance_m, elevation) = if free.altitude > 0.0 && pitch.cos() >= 0.1 {
            (free.altitude / pitch.cos(), 0.0)
        } else {
            let d = LEVEL_DISTANCE_M;
            (d, free.altitude - d * pitch.cos())
        };
        let bearing = free.bearing.to_radians();
        let ahead = DVec2::new(bearing.sin(), bearing.cos());
        let eye_distance_px = 0.5 * viewport.height / (self.fov.to_radians() / 2.0).tan();
        let mut camera = Self {
            bearing: free.bearing,
            pitch: free.pitch,
            roll: free.roll,
            elevation,
            ..*self
        };
        // World units per metre are the target's: two steps settle them.
        let mut scale = projection.style_height_scale(free.position.wrapped());
        for _ in 0..3 {
            let distance = distance_m * scale;
            camera.set_target_world(
                projection,
                eye + ahead * distance * pitch.sin(),
                distance / eye_distance_px,
            );
            scale = projection.style_height_scale(camera.target);
        }
        camera
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Padding;

    #[test]
    fn a_free_camera_round_trips() {
        let mercator = *orb_crs::TileMatrixSet::by_id("WebMercatorQuad")
            .unwrap()
            .projection();
        let surface = Surface::plane(mercator);
        let viewport = Viewport {
            width: 800.0,
            height: 600.0,
        };
        for (bearing, pitch, roll) in [(0.0, 0.0, 0.0), (30.0, 45.0, 0.0), (-120.0, 70.0, 10.0)] {
            let camera = CameraState {
                bearing,
                pitch,
                roll,
                padding: Padding {
                    top: 100.0,
                    ..Padding::default()
                },
                ..CameraState::new(Geodetic::new(11.4, 47.3), 13.0, &mercator)
            };
            let free = camera.free(viewport, &surface);
            // The eye stands behind the target, away from where it looks.
            if pitch > 0.0 {
                let back = (free.position.lon - 11.4, free.position.lat - 47.3);
                let facing = bearing.to_radians();
                assert!(
                    back.0 * facing.sin() + back.1 * facing.cos() < 0.0,
                    "{free:?}"
                );
            }
            let again = camera.from_free(&free, viewport, &surface);
            assert!((again.target.lon - 11.4).abs() < 1e-9, "{again:?}");
            assert!((again.target.lat - 47.3).abs() < 1e-9, "{again:?}");
            assert!((again.zoom(&mercator) - 13.0).abs() < 1e-9, "{again:?}");
            assert_eq!(again.padding, camera.padding);
        }
        // Looking at the horizon: the target is 10 km ahead, at the eye's
        // height.
        let level = CameraState::new(Geodetic::new(0.0, 0.0), 10.0, &mercator).from_free(
            &FreeCamera {
                position: Geodetic::new(0.0, 0.0),
                altitude: 500.0,
                bearing: 90.0,
                pitch: 90.0,
                roll: 0.0,
            },
            viewport,
            &surface,
        );
        assert!((level.elevation - 500.0).abs() < 1e-6, "{level:?}");
        assert!(
            level.target.lon > 0.08 && level.target.lon < 0.1,
            "{level:?}"
        );
    }
}
