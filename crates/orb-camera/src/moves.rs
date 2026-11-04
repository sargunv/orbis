//! Camera moves in viewport pixels, which gestures and animations are built
//! from, and the limits every move is kept to.

use glam::DVec2;
use orb_crs::{Projection, WorldRect};
use orb_geo::Geodetic;

use crate::{CameraState, View, Viewport};

/// A longitude and latitude box, in degrees. `west` may exceed `east` only
/// by wrapping: boxes that cross the antimeridian give `east` above 180.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl Bounds {
    /// The smallest box holding every point, or `None` for no points.
    pub fn around(points: impl IntoIterator<Item = Geodetic>) -> Option<Self> {
        points.into_iter().fold(None, |bounds, p| {
            Some(match bounds {
                None => Self {
                    west: p.lon,
                    south: p.lat,
                    east: p.lon,
                    north: p.lat,
                },
                Some(b) => Self {
                    west: b.west.min(p.lon),
                    south: b.south.min(p.lat),
                    east: b.east.max(p.lon),
                    north: b.north.max(p.lat),
                },
            })
        })
    }

    pub fn corners(&self) -> [Geodetic; 4] {
        [
            Geodetic::new(self.west, self.north),
            Geodetic::new(self.east, self.north),
            Geodetic::new(self.east, self.south),
            Geodetic::new(self.west, self.south),
        ]
    }

    fn world(&self, projection: &Projection) -> WorldRect {
        let sw = projection.forward(Geodetic::new(self.west, self.south));
        let ne = projection.forward(Geodetic::new(self.east, self.north));
        WorldRect {
            min: sw.min(ne),
            max: sw.max(ne),
        }
    }
}

/// What a camera may show: GL JS's `minZoom`, `maxZoom`, `minPitch`,
/// `maxPitch` and `maxBounds`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constraints {
    pub min_zoom: f64,
    pub max_zoom: f64,
    /// Degrees from looking straight down.
    pub min_pitch: f64,
    pub max_pitch: f64,
    /// Keep the viewport inside this box, zooming in if it is smaller.
    pub bounds: Option<Bounds>,
}

impl Default for Constraints {
    /// GL JS's defaults, except that pitch reaches 85 degrees, the steepest
    /// view the flat surface's horizon supports (GL JS stops at 60).
    fn default() -> Self {
        Self {
            min_zoom: -2.0,
            max_zoom: 22.0,
            min_pitch: 0.0,
            max_pitch: 85.0,
            bounds: None,
        }
    }
}

impl Constraints {
    /// No limits beyond the world's own edge, which every camera keeps to.
    pub const WORLD: Self = Self {
        min_zoom: f64::NEG_INFINITY,
        max_zoom: f64::INFINITY,
        min_pitch: 0.0,
        max_pitch: 90.0,
        bounds: None,
    };
}

impl CameraState {
    /// World units per logical pixel at the target. At a given zoom it is
    /// the same everywhere on the surface, so moves keep it.
    pub fn world_per_px(&self, projection: &Projection) -> f64 {
        self.resolution * projection.scale_factor(self.target.wrapped())
    }

    /// The target's position in world space.
    pub fn target_world(&self, projection: &Projection) -> DVec2 {
        projection.forward(self.target.wrapped())
    }

    /// Puts the target at a world position with the given scale.
    pub fn set_target_world(&mut self, projection: &Projection, world: DVec2, world_per_px: f64) {
        self.target = projection.inverse(world).wrapped();
        self.resolution = world_per_px / projection.scale_factor(self.target);
    }

    /// Moves the camera so the ground under `from` (logical pixels) ends up
    /// under `to`. Nothing moves when either is above the horizon.
    pub fn pan(&mut self, projection: &Projection, viewport: Viewport, from: DVec2, to: DVec2) {
        let view = View::new(self, viewport, projection);
        let (Some(a), Some(b)) = (view.unproject(from), view.unproject(to)) else {
            return;
        };
        let world = view.camera_world() - (b - a);
        self.set_target_world(projection, world, view.world_per_px());
    }

    /// Sets the zoom, keeping the ground under `at` in place.
    pub fn zoom_around(
        &mut self,
        projection: &Projection,
        viewport: Viewport,
        at: DVec2,
        zoom: f64,
    ) {
        self.holding(projection, viewport, at, |camera| camera.set_zoom(zoom));
    }

    /// Sets the bearing, keeping the ground under `at` in place.
    pub fn rotate_around(
        &mut self,
        projection: &Projection,
        viewport: Viewport,
        at: DVec2,
        bearing: f64,
    ) {
        self.holding(projection, viewport, at, |camera| {
            camera.bearing = wrap_bearing(bearing);
        });
    }

    /// Applies `change`, then moves the target so the ground that was under
    /// `at` is under it again.
    pub(crate) fn holding(
        &mut self,
        projection: &Projection,
        viewport: Viewport,
        at: DVec2,
        change: impl FnOnce(&mut Self),
    ) {
        let center = DVec2::new(viewport.width, viewport.height) / 2.0;
        if at.distance_squared(center) < 1e-4 {
            // The centre is the target already. A projection round trip
            // could move it by an ULP and change the ideal cover at a zoom
            // boundary.
            change(self);
            return;
        }
        let before = View::new(self, viewport, projection).unproject(at);
        change(self);
        let view = View::new(self, viewport, projection);
        if let (Some(before), Some(after)) = (before, view.unproject(at)) {
            let world = view.camera_world() + (before - after);
            self.set_target_world(projection, world, view.world_per_px());
        }
    }

    /// Keeps the camera to `constraints`: zoom and pitch within their
    /// ranges, and the viewport within the bounds or, without them, within
    /// the world's top and bottom edge, zooming in where the box is smaller
    /// than the viewport (GL JS `getConstrained`; like it, ignoring pitch
    /// and bearing). Leaves a camera that already complies untouched.
    pub fn constrain(
        &mut self,
        constraints: &Constraints,
        projection: &Projection,
        viewport: Viewport,
    ) {
        let zoom = self.zoom();
        let clamped = zoom.clamp(
            constraints.min_zoom,
            constraints.max_zoom.max(constraints.min_zoom),
        );
        if clamped != zoom {
            self.set_zoom(clamped);
        }
        self.pitch = self.pitch.clamp(
            constraints.min_pitch,
            constraints.max_pitch.max(constraints.min_pitch),
        );
        let (rect, both_axes) = match constraints.bounds {
            Some(bounds) => (bounds.world(projection), true),
            None => (
                projection.world_bounds(),
                projection.wrap_period().is_none(),
            ),
        };
        self.keep_within(projection, viewport, rect, both_axes);
    }

    fn keep_within(
        &mut self,
        projection: &Projection,
        viewport: Viewport,
        rect: WorldRect,
        both_axes: bool,
    ) {
        let world_per_px = self.world_per_px(projection);
        let size = DVec2::new(viewport.width, viewport.height);
        let mut fit = size.y * world_per_px / rect.size().y;
        if both_axes {
            fit = fit.max(size.x * world_per_px / rect.size().x);
        }
        let scaled = if fit > 1.0 {
            world_per_px / fit
        } else {
            world_per_px
        };
        let half = size * scaled / 2.0;
        let center = self.target_world(projection);
        let mut world = center;
        world.y = world.y.clamp(
            rect.min.y + half.y,
            (rect.max.y - half.y).max(rect.min.y + half.y),
        );
        if both_axes {
            world.x = world.x.clamp(
                rect.min.x + half.x,
                (rect.max.x - half.x).max(rect.min.x + half.x),
            );
        }
        // A target past the world's edge projects onto it, so it moves too.
        if world != center || scaled != world_per_px {
            self.set_target_world(projection, world, scaled);
        }
    }
}

/// A bearing in GL JS's range, (-180, 180].
pub fn wrap_bearing(bearing: f64) -> f64 {
    let wrapped = (bearing + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped == -180.0 { 180.0 } else { wrapped }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_crs::TileMatrixSet;

    fn mercator() -> Projection {
        *TileMatrixSet::by_id("WebMercatorQuad")
            .unwrap()
            .projection()
    }

    const VIEWPORT: Viewport = Viewport {
        width: 800.0,
        height: 600.0,
    };

    #[test]
    fn zooming_around_a_point_keeps_its_ground() {
        let p = mercator();
        let mut camera = CameraState::new(Geodetic::new(139.7, 35.6), 12.0);
        camera.pitch = 40.0;
        camera.bearing = 30.0;
        let at = DVec2::new(100.0, 450.0);
        let before = View::new(&camera, VIEWPORT, &p).unproject(at).unwrap();
        camera.zoom_around(&p, VIEWPORT, at, 13.5);
        let after = View::new(&camera, VIEWPORT, &p).unproject(at).unwrap();
        assert!((camera.zoom() - 13.5).abs() < 1e-9);
        assert!(before.distance(after) < 1e-6, "{before} {after}");
        camera.rotate_around(&p, VIEWPORT, at, 75.0);
        let rotated = View::new(&camera, VIEWPORT, &p).unproject(at).unwrap();
        assert!(before.distance(rotated) < 1e-6);
        assert_eq!(camera.bearing, 75.0);
    }

    #[test]
    fn panning_moves_ground_with_the_pointer() {
        let p = mercator();
        let mut camera = CameraState::new(Geodetic::new(-122.3, 47.6), 10.0);
        camera.pitch = 50.0;
        let from = DVec2::new(400.0, 400.0);
        let to = DVec2::new(470.0, 350.0);
        let ground = View::new(&camera, VIEWPORT, &p).unproject(from).unwrap();
        camera.pan(&p, VIEWPORT, from, to);
        let now = View::new(&camera, VIEWPORT, &p).project(ground).unwrap();
        assert!(now.distance(to) < 1e-6, "{now}");
        assert!((camera.zoom() - 10.0).abs() < 1e-9, "a pan doesn't zoom");
    }

    #[test]
    fn bounds_keep_the_viewport_inside() {
        let p = mercator();
        let bounds = Bounds {
            west: 139.0,
            south: 35.0,
            east: 140.0,
            north: 36.0,
        };
        let constraints = Constraints {
            bounds: Some(bounds),
            ..Constraints::default()
        };
        // Far outside and too far out: zooms in until the box fills the
        // viewport, and moves inside.
        let mut camera = CameraState::new(Geodetic::new(0.0, 0.0), 3.0);
        camera.constrain(&constraints, &p, VIEWPORT);
        let view = View::new(&camera, VIEWPORT, &p);
        let rect = bounds.world(&p);
        for corner in [DVec2::ZERO, DVec2::new(VIEWPORT.width, VIEWPORT.height)] {
            let world = view.unproject(corner).unwrap();
            assert!(world.x >= rect.min.x - 1e-3 && world.x <= rect.max.x + 1e-3);
            assert!(world.y >= rect.min.y - 1e-3 && world.y <= rect.max.y + 1e-3);
        }
        // A view inside is left alone.
        let mut inside = CameraState::new(Geodetic::new(139.5, 35.5), 12.0);
        let before = inside;
        inside.constrain(&constraints, &p, VIEWPORT);
        assert_eq!(inside, before);
        // Zoom and pitch ranges.
        let mut steep = CameraState::new(Geodetic::new(139.5, 35.5), 30.0);
        steep.pitch = 89.0;
        steep.constrain(&Constraints::default(), &p, VIEWPORT);
        assert!((steep.zoom() - 22.0).abs() < 1e-9);
        assert_eq!(steep.pitch, 85.0);
    }

    #[test]
    fn bearings_wrap_like_gl_js() {
        assert_eq!(wrap_bearing(190.0), -170.0);
        assert_eq!(wrap_bearing(-180.0), 180.0);
        assert_eq!(wrap_bearing(540.0), 180.0);
        assert_eq!(wrap_bearing(-30.0), -30.0);
    }
}
