//! Camera moves in viewport pixels, which gestures and animations are built
//! from, and the limits every move is kept to.

use glam::DVec2;
use orb_crs::Geodetic;
use orb_crs::{Projection, WorldRect, sphere};

use crate::{CameraState, STYLE_ZOOM_TILE_PX, View, Viewport};
use orb_crs::Surface;

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
    pub fn corners(&self) -> [Geodetic; 4] {
        [
            Geodetic::new(self.west, self.north),
            Geodetic::new(self.east, self.north),
            Geodetic::new(self.east, self.south),
            Geodetic::new(self.west, self.south),
        ]
    }

    /// The box's outline in world space: its corners on a wrapping
    /// (cylindrical) plane, where its edges are straight, else points along
    /// its edges, which curve.
    pub(crate) fn outline(&self, projection: &Projection) -> Vec<DVec2> {
        if projection.wrap_period().is_some() {
            return self.corners().map(|c| projection.forward(c)).to_vec();
        }
        const STEPS: u32 = 32;
        let [nw, ne, se, sw] = self.corners();
        [(nw, ne), (ne, se), (se, sw), (sw, nw)]
            .into_iter()
            .flat_map(|(a, b)| {
                (0..STEPS).map(move |i| {
                    let t = f64::from(i) / f64::from(STEPS);
                    Geodetic::new(a.lon + (b.lon - a.lon) * t, a.lat + (b.lat - a.lat) * t)
                })
            })
            .filter(|g| projection.contains(*g))
            .map(|g| projection.forward(g))
            .collect()
    }

    fn world(&self, projection: &Projection) -> WorldRect {
        WorldRect::around(self.outline(projection))
    }
}

/// What a camera may show: GL JS's `minZoom`, `maxZoom`, `minPitch`,
/// `maxPitch` and `maxBounds`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Constraints {
    /// Style zoom limits (see [`CameraState::zoom`]): on every plane the
    /// style zoom stays put as the view pans.
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
    /// World units per logical pixel at the target: the view's scale,
    /// which moves keep (a pan changes ground resolution only as the
    /// plane's own scale factor does).
    pub fn world_per_px(&self, projection: &Projection) -> f64 {
        self.resolution * projection.scale_factor(self.target.wrapped())
    }

    /// Whether moves keep the planet's size on screen rather than the
    /// style zoom (GL JS's globe `getZoomAdjustment`): on the sphere, whose
    /// radius on screen is its own over the ground resolution while the
    /// style zoom also moves with the target's latitude.
    pub(crate) fn on_sphere(&self, surface: &Surface) -> bool {
        surface.globe.is_some_and(|g| g.plane.is_none())
            && surface.globe_amount(self.zoom(&surface.projection)) > 0.0
    }

    /// The style zoom the ground resolution would have on the equator (GL
    /// JS's globe zoom normalized by `getZoomAdjustment`): what sets the
    /// sphere's size on screen.
    pub(crate) fn globe_zoom(&self) -> f64 {
        (crate::equator_circumference_px_per_m(Geodetic::new(0.0, 0.0)) / self.resolution).log2()
    }

    pub(crate) fn set_globe_zoom(&mut self, zoom: f64) {
        self.resolution =
            crate::equator_circumference_px_per_m(Geodetic::new(0.0, 0.0)) / zoom.exp2();
    }

    /// The least style zoom `constraints` allow at the target. On the
    /// sphere the least zoom bounds its size as it is at the equator (GL
    /// JS's globe `defaultConstrain`), so the smallest globe is as small
    /// over a pole, where its style zoom is lower.
    pub(crate) fn least_zoom(&self, constraints: &Constraints, surface: &Surface) -> f64 {
        if constraints.bounds.is_none() && self.on_sphere(surface) {
            constraints.min_zoom + self.zoom(&surface.projection) - self.globe_zoom()
        } else {
            constraints.min_zoom
        }
    }

    /// Keeps the style zoom within limits, keeping the target.
    pub(crate) fn limit_zoom(&mut self, min: f64, max: f64, projection: &Projection) {
        let zoom = self.zoom(projection);
        let clamped = zoom.clamp(min, max.max(min));
        if clamped != zoom {
            self.set_zoom(projection, clamped);
        }
    }

    /// The target's position in world space.
    pub(crate) fn target_world(&self, projection: &Projection) -> DVec2 {
        projection.forward(self.target.wrapped())
    }

    /// Puts the target at a world position with the given scale.
    pub(crate) fn set_target_world(
        &mut self,
        projection: &Projection,
        world: DVec2,
        world_per_px: f64,
    ) {
        self.target = projection.inverse(world).wrapped();
        self.resolution = world_per_px / projection.scale_factor(self.target);
    }

    /// Moves the camera so the ground under `from` (logical pixels) ends up
    /// under `to`. Nothing moves when either is above the horizon.
    pub fn pan(&mut self, surface: &Surface, viewport: Viewport, from: DVec2, to: DVec2) {
        let projection = &surface.projection;
        let view = View::new(self, viewport, surface);
        if view.globe().is_some() {
            if let Some(ground) = view.low_ground(from, view.elevation()) {
                self.put_under(surface, viewport, ground, to);
            }
            return;
        }
        let (Some(a), Some(b)) = (view.unproject(from), view.unproject(to)) else {
            return;
        };
        let world = view.camera_world() - (b - a);
        self.set_target_world(projection, world, view.world_per_px());
    }

    /// Sets the zoom, keeping the ground under `at` in place.
    pub fn zoom_around(&mut self, surface: &Surface, viewport: Viewport, at: DVec2, zoom: f64) {
        let projection = &surface.projection;
        self.holding(surface, viewport, at, |camera| {
            camera.set_zoom(projection, zoom)
        });
    }

    /// Sets the bearing, keeping the ground under `at` in place.
    pub fn rotate_around(
        &mut self,
        surface: &Surface,
        viewport: Viewport,
        at: DVec2,
        bearing: f64,
    ) {
        self.holding(surface, viewport, at, |camera| {
            camera.bearing = wrap_bearing(bearing);
        });
    }

    /// Applies `change`, then moves the target so the ground that was under
    /// `at` is under it again.
    pub(crate) fn holding(
        &mut self,
        surface: &Surface,
        viewport: Viewport,
        at: DVec2,
        change: impl FnOnce(&mut Self),
    ) {
        let projection = &surface.projection;
        let center = self.center(viewport);
        if at.distance_squared(center) < 1e-4 {
            // The centre is the target already. A projection round trip
            // could move it by an ULP and change the ideal cover at a zoom
            // boundary.
            change(self);
            return;
        }
        let view = View::new(self, viewport, surface);
        if view.globe().is_some() {
            let ground = view.low_ground(at, view.elevation());
            change(self);
            if let Some(ground) = ground {
                self.put_under(surface, viewport, ground, at);
            }
            return;
        }
        let before = view.unproject(at);
        change(self);
        let view = View::new(self, viewport, surface);
        if let (Some(before), Some(after)) = (before, view.unproject(at)) {
            let world = view.camera_world() + (before - after);
            self.set_target_world(projection, world, view.world_per_px());
        }
    }

    /// On a low-zoom surface, moves the camera so `ground` shows at `px`,
    /// keeping the bearing, and the sphere's size on screen (its ground
    /// resolution; GL JS's `getZoomAdjustment`) or a plane's zoom: on a
    /// globe, the arc from what shows there to `ground` turns the target
    /// (GL JS's `versorSetLocationAtPoint`), on a plane it moves by their
    /// distance on it; then Newton steps on the target's longitude and
    /// latitude land it exactly. Nothing moves where `px` is off the
    /// surface.
    pub(crate) fn put_under(
        &mut self,
        surface: &Surface,
        viewport: Viewport,
        ground: Geodetic,
        px: DVec2,
    ) {
        let projection = &surface.projection;
        let zoom = self.zoom(projection);
        let view = View::new(self, viewport, surface);
        let Some(now) = view.low_ground(px, view.elevation()) else {
            return;
        };
        let sphere = view.low_plane().is_none();
        let guess = match view.low_plane() {
            Some(low) => {
                let plane = low.projection();
                let moved = low.camera_world() + plane.forward(ground) - plane.forward(now);
                plane.inverse(moved)
            }
            None => {
                let (now, ground) = (sphere::ecef(now, 0.0), sphere::ecef(ground, 0.0));
                let turn = glam::DQuat::from_rotation_arc(now.normalize(), ground.normalize());
                sphere::geodetic(turn * sphere::up(self.target.wrapped()))
            }
        };
        let at = |target: Geodetic| {
            let mut camera = CameraState { target, ..*self };
            if !sphere {
                camera.set_zoom(projection, zoom);
            }
            camera
        };
        let shown = |camera: &CameraState| {
            let view = View::new(camera, viewport, surface);
            view.globe()?;
            view.project(projection.forward(ground))
        };
        let mut best = at(guess);
        for _ in 0..4 {
            let Some(p) = shown(&best) else {
                break;
            };
            let miss = p - px;
            if miss.length() < 1e-6 {
                break;
            }
            // The pixel's motion per degree of longitude and latitude.
            let h = 1e-6;
            let t = best.target;
            let (Some(dx), Some(dy)) = (
                shown(&at(Geodetic::new(t.lon + h, t.lat))),
                shown(&at(Geodetic::new(t.lon, t.lat + h))),
            ) else {
                break;
            };
            let jacobian = glam::DMat2::from_cols((dx - p) / h, (dy - p) / h);
            if jacobian.determinant().abs() < 1e-12 {
                break;
            }
            let step = jacobian.inverse() * miss;
            best = at(Geodetic::new(t.lon - step.x, t.lat - step.y));
        }
        if shown(&best).is_some() {
            let g = best.target;
            *self = at(Geodetic::new(g.lon, g.lat.clamp(-90.0, 90.0)).wrapped());
        }
    }

    /// Keeps the camera to `constraints`: zoom and pitch within their
    /// ranges, and the viewport within the bounds or, without them, within
    /// the world's top and bottom edge, zooming in where the box is smaller
    /// than the viewport (GL JS `getConstrained`; like it, ignoring pitch).
    /// Unlike GL JS, bounds hold the rotated viewport's extent, so turning
    /// the map never shows past them. Leaves a camera that already complies
    /// untouched.
    pub(crate) fn constrain(
        &mut self,
        constraints: &Constraints,
        surface: &Surface,
        viewport: Viewport,
    ) {
        let projection = &surface.projection;
        let sphere = constraints.bounds.is_none() && self.on_sphere(surface);
        let least = self.least_zoom(constraints, surface);
        self.limit_zoom(least, constraints.max_zoom, projection);
        self.pitch = self.pitch.clamp(
            constraints.min_pitch,
            constraints.max_pitch.max(constraints.min_pitch),
        );
        if sphere && projection.wrap_period().is_some() {
            // Past the plane's edge the sphere has no data: the target
            // reaches a pole only while that cap fits the viewport.
            let reach = self.max_latitude(projection, viewport);
            self.target.lat = self.target.lat.clamp(-reach, reach);
            return;
        }
        if constraints.bounds.is_none() && surface.globe_amount(self.zoom(projection)) > 0.0 {
            // A globe shows the whole world at any size: only the target
            // keeps to it.
            let world = projection.world_bounds();
            let at = self.target_world(projection);
            let mut kept = at;
            kept.y = kept.y.clamp(world.min.y, world.max.y);
            if projection.wrap_period().is_none() {
                kept.x = kept.x.clamp(world.min.x, world.max.x);
            }
            if kept != at {
                let world_per_px = self.world_per_px(projection);
                self.set_target_world(projection, kept, world_per_px);
            }
            return;
        }
        let (rect, both_axes, extent) = match constraints.bounds {
            Some(bounds) => {
                let (sin, cos) = self.bearing.to_radians().sin_cos();
                let (sin, cos) = (sin.abs(), cos.abs());
                let (w, h) = (viewport.width, viewport.height);
                let extent = DVec2::new(w * cos + h * sin, w * sin + h * cos);
                (bounds.world(projection), true, extent)
            }
            None => (
                projection.world_bounds(),
                projection.wrap_period().is_none(),
                DVec2::new(viewport.width, viewport.height),
            ),
        };
        self.keep_within(projection, extent, rect, both_axes);
    }

    /// How far from the equator the target may go on the sphere, in
    /// degrees (GL JS's `_getMaxLatitude`): to a pole while the arc from
    /// the plane's top edge to it is at most half the viewport's least side
    /// on screen, back to the edge by the time it is four times that, and
    /// the edge beyond.
    fn max_latitude(&self, projection: &Projection, viewport: Viewport) -> f64 {
        let world = projection.world_bounds();
        let edge = projection
            .inverse(DVec2::new(world.center().x, world.max.y))
            .lat
            .abs()
            .min(90.0);
        let cap = (90.0 - edge).to_radians();
        if cap <= 0.0 {
            return 90.0;
        }
        // That arc on screen with the target at the edge.
        let radius = STYLE_ZOOM_TILE_PX * self.zoom(projection).exp2()
            / (2.0 * std::f64::consts::PI * edge.to_radians().cos());
        let area = radius * cap;
        let reach = viewport.width.min(viewport.height) / 2.0;
        let fade = ((area - reach) / (reach * (POLE_FADE - 1.0))).clamp(0.0, 1.0);
        let center = area.min(reach) * (1.0 - fade);
        edge + (cap * center / area).to_degrees()
    }

    /// Keeps a viewport `size` pixels across, axis-aligned in world space,
    /// inside `rect`.
    fn keep_within(
        &mut self,
        projection: &Projection,
        size: DVec2,
        rect: WorldRect,
        both_axes: bool,
    ) {
        let world_per_px = self.world_per_px(projection);
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

/// How many times the reach (half the viewport's least side) the polar cap
/// has grown to on screen when the target is back at the plane's edge (GL
/// JS's `POLE_AREA_FADE_SCALE`).
const POLE_FADE: f64 = 4.0;

/// A bearing in GL JS's range, (-180, 180].
pub(crate) fn wrap_bearing(bearing: f64) -> f64 {
    let wrapped = (bearing + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped == -180.0 { 180.0 } else { wrapped }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_crs::TileMatrixSet;

    fn mercator() -> Projection {
        *TileMatrixSet::web_mercator().projection()
    }

    const VIEWPORT: Viewport = Viewport {
        width: 800.0,
        height: 600.0,
    };

    #[test]
    fn zooming_around_a_point_keeps_its_ground() {
        let p = mercator();
        let mut camera = CameraState::new(Geodetic::new(139.7, 35.6), 12.0, &mercator());
        camera.pitch = 40.0;
        camera.bearing = 30.0;
        let at = DVec2::new(100.0, 450.0);
        let before = View::new(&camera, VIEWPORT, &Surface::from(&p))
            .unproject(at)
            .unwrap();
        camera.zoom_around(&Surface::from(&p), VIEWPORT, at, 13.5);
        let after = View::new(&camera, VIEWPORT, &Surface::from(&p))
            .unproject(at)
            .unwrap();
        assert!((camera.zoom(&mercator()) - 13.5).abs() < 1e-9);
        assert!(before.distance(after) < 1e-6, "{before} {after}");
        camera.rotate_around(&Surface::from(&p), VIEWPORT, at, 75.0);
        let rotated = View::new(&camera, VIEWPORT, &Surface::from(&p))
            .unproject(at)
            .unwrap();
        assert!(before.distance(rotated) < 1e-6);
        assert_eq!(camera.bearing, 75.0);
    }

    #[test]
    fn panning_moves_ground_with_the_pointer() {
        let p = mercator();
        let mut camera = CameraState::new(Geodetic::new(-122.3, 47.6), 10.0, &mercator());
        camera.pitch = 50.0;
        let from = DVec2::new(400.0, 400.0);
        let to = DVec2::new(470.0, 350.0);
        let ground = View::new(&camera, VIEWPORT, &Surface::from(&p))
            .unproject(from)
            .unwrap();
        camera.pan(&Surface::from(&p), VIEWPORT, from, to);
        let now = View::new(&camera, VIEWPORT, &Surface::from(&p))
            .project(ground)
            .unwrap();
        assert!(now.distance(to) < 1e-6, "{now}");
        assert!(
            (camera.zoom(&mercator()) - 10.0).abs() < 1e-9,
            "a pan doesn't zoom"
        );
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
        let mut camera = CameraState::new(Geodetic::new(0.0, 0.0), 3.0, &mercator());
        camera.constrain(&constraints, &Surface::from(&p), VIEWPORT);
        let view = View::new(&camera, VIEWPORT, &Surface::from(&p));
        let rect = bounds.world(&p);
        for corner in [DVec2::ZERO, DVec2::new(VIEWPORT.width, VIEWPORT.height)] {
            let world = view.unproject(corner).unwrap();
            assert!(world.x >= rect.min.x - 1e-3 && world.x <= rect.max.x + 1e-3);
            assert!(world.y >= rect.min.y - 1e-3 && world.y <= rect.max.y + 1e-3);
        }
        // Turned, every corner of the viewport stays inside too.
        let mut turned = CameraState::new(Geodetic::new(139.02, 35.02), 9.5, &mercator());
        turned.bearing = 35.0;
        turned.constrain(&constraints, &Surface::from(&p), VIEWPORT);
        let view = View::new(&turned, VIEWPORT, &Surface::from(&p));
        for (x, y) in [
            (0.0, 0.0),
            (VIEWPORT.width, 0.0),
            (0.0, VIEWPORT.height),
            (VIEWPORT.width, VIEWPORT.height),
        ] {
            let world = view.unproject(DVec2::new(x, y)).unwrap();
            assert!(
                world.x >= rect.min.x - 1e-3 && world.x <= rect.max.x + 1e-3,
                "{world}"
            );
            assert!(
                world.y >= rect.min.y - 1e-3 && world.y <= rect.max.y + 1e-3,
                "{world}"
            );
        }
        // A view inside is left alone.
        let mut inside = CameraState::new(Geodetic::new(139.5, 35.5), 12.0, &mercator());
        let before = inside;
        inside.constrain(&constraints, &Surface::from(&p), VIEWPORT);
        assert_eq!(inside, before);
        // Zoom and pitch ranges.
        let mut steep = CameraState::new(Geodetic::new(139.5, 35.5), 30.0, &mercator());
        steep.pitch = 89.0;
        steep.constrain(&Constraints::default(), &Surface::from(&p), VIEWPORT);
        assert!((steep.zoom(&mercator()) - 22.0).abs() < 1e-9);
        assert_eq!(steep.pitch, 85.0);
    }

    /// Moves on planes other than Web Mercator: a polar plane (where north
    /// turns around the pole) and a transverse Mercator one, each with its
    /// tile matrix set's extent.
    #[test]
    fn moves_keep_their_ground_on_other_planes() {
        for (id, target) in [
            ("UPSArcticWGS84Quad", Geodetic::new(-40.0, 78.0)),
            ("NZTM2000Quad", Geodetic::new(174.8, -41.3)),
        ] {
            let p = *TileMatrixSet::by_id(id).unwrap().projection();
            let mut camera = CameraState::new(target, 6.0, &p);
            camera.pitch = 35.0;
            camera.bearing = -20.0;
            let (from, to) = (DVec2::new(300.0, 420.0), DVec2::new(520.0, 260.0));
            let ground = View::new(&camera, VIEWPORT, &Surface::from(&p))
                .unproject(from)
                .unwrap();
            camera.pan(&Surface::from(&p), VIEWPORT, from, to);
            let now = View::new(&camera, VIEWPORT, &Surface::from(&p))
                .project(ground)
                .unwrap();
            assert!(now.distance(to) < 1e-6, "{id}: pan {now}");
            let at = DVec2::new(150.0, 500.0);
            let before = View::new(&camera, VIEWPORT, &Surface::from(&p))
                .unproject(at)
                .unwrap();
            camera.zoom_around(&Surface::from(&p), VIEWPORT, at, 9.0);
            camera.rotate_around(&Surface::from(&p), VIEWPORT, at, 60.0);
            let after = View::new(&camera, VIEWPORT, &Surface::from(&p))
                .unproject(at)
                .unwrap();
            assert!(before.distance(after) < 1e-6, "{id}: {before} {after}");
            // The world's edge holds on both axes: a plane that doesn't wrap
            // keeps the viewport inside its extent.
            let mut far = CameraState::new(target, 1.0, &p);
            far.constrain(&Constraints::WORLD, &Surface::from(&p), VIEWPORT);
            let view = View::new(&far, VIEWPORT, &Surface::from(&p));
            let world = p.world_bounds();
            for corner in [DVec2::ZERO, DVec2::new(VIEWPORT.width, VIEWPORT.height)] {
                let w = view.unproject(corner).unwrap();
                assert!(
                    w.cmpge(world.min - 1e-3).all() && w.cmple(world.max + 1e-3).all(),
                    "{id}"
                );
            }
        }
    }

    /// A box around a pole is a disc on a polar plane: its outline, not its
    /// corners, bounds it.
    #[test]
    fn polar_boxes_bound_their_outline() {
        let p = *TileMatrixSet::by_id("UPSArcticWGS84Quad")
            .unwrap()
            .projection();
        let cap = Bounds {
            west: -180.0,
            south: 70.0,
            east: 180.0,
            north: 90.0,
        };
        let rect = cap.world(&p);
        let pole = p.forward(Geodetic::new(0.0, 90.0));
        let edge = p.forward(Geodetic::new(90.0, 70.0)) - pole;
        assert!((rect.center() - pole).length() < 1.0);
        assert!((rect.size().x / 2.0 - edge.length()).abs() < 1.0);
        let camera = CameraState::new(Geodetic::new(0.0, 60.0), 2.0, &p);
        let fit = camera
            .for_bounds(
                cap,
                &crate::FitOptions::default(),
                &Surface::from(&p),
                VIEWPORT,
            )
            .unwrap();
        let center = fit.center.unwrap();
        assert!(center.lat > 89.99, "{center:?}");
    }

    #[test]
    fn bearings_wrap_like_gl_js() {
        assert_eq!(wrap_bearing(190.0), -170.0);
        assert_eq!(wrap_bearing(-180.0), 180.0);
        assert_eq!(wrap_bearing(540.0), 180.0);
        assert_eq!(wrap_bearing(-30.0), -30.0);
    }

    fn globe_surface() -> Surface {
        Surface {
            projection: mercator(),
            globe: Some(orb_crs::Globe::V8),
        }
    }

    /// The sphere's radius on screen, along the viewport's middle row.
    fn limb(camera: &CameraState, viewport: Viewport) -> f64 {
        let view = View::new(camera, viewport, &globe_surface());
        let center = camera.center(viewport);
        let (mut on, mut off) = (0.0, viewport.width);
        while off - on > 1.0 / 64.0 {
            let x = (on + off) / 2.0;
            match view.unproject(center + DVec2::new(x, 0.0)) {
                Some(_) => on = x,
                None => off = x,
            }
        }
        on
    }

    /// Dragging the globe north keeps its size on screen, as GL JS's zoom
    /// adjustment does, where keeping the style zoom grows it by a third
    /// from the equator to 46°N.
    #[test]
    fn globe_drags_keep_its_size() {
        let surface = globe_surface();
        let viewport = Viewport {
            width: 800.0,
            height: 800.0,
        };
        let mut camera = CameraState::new(Geodetic::new(0.0, 0.0), 2.0, &mercator());
        let radius = limb(&camera, viewport);
        for _ in 0..10 {
            camera.pan(
                &surface,
                viewport,
                DVec2::new(400.0, 400.0),
                DVec2::new(400.0, 430.0),
            );
            assert!((limb(&camera, viewport) - radius).abs() < 1.0);
        }
        assert!(camera.target.lat > 40.0, "{:?}", camera.target);
        // The style zoom falls as the parallels shorten.
        let adjusted = 2.0 + camera.target.lat.to_radians().cos().log2();
        assert!((camera.zoom(&mercator()) - adjusted).abs() < 1e-9);
    }

    /// GL JS's `_getMaxLatitude`: a small globe centres on a pole, and the
    /// reach falls back to Web Mercator's edge as the cap past it outgrows
    /// the viewport.
    #[test]
    fn small_globes_reach_the_poles() {
        let surface = globe_surface();
        let edge = orb_crs::MERCATOR_MAX_LATITUDE;
        let reach = |zoom: f64| {
            let mut camera = CameraState::new(Geodetic::new(30.0, 90.0), zoom, &mercator());
            camera.constrain(&Constraints::default(), &surface, VIEWPORT);
            camera.target.lat
        };
        // The cap is 82 px across per 2^zoom; the reach is 300 px.
        assert_eq!(reach(1.0), 90.0);
        assert_eq!(reach(1.8), 90.0);
        let mut last = 90.0;
        for zoom in [2.0, 2.5, 3.0, 3.5] {
            let lat = reach(zoom);
            assert!(lat < last && lat > edge, "z{zoom}: {lat}");
            last = lat;
        }
        assert!((reach(4.0) - edge).abs() < 1e-6);
        assert!((reach(8.0) - edge).abs() < 1e-6);
        // A pole-centred view keeps its pole and draws.
        let camera = CameraState::new(Geodetic::new(0.0, -90.0), 1.0, &mercator());
        let view = View::new(&camera, VIEWPORT, &surface);
        let tms = orb_crs::TileMatrixSet::web_mercator();
        assert!(!view.cover(tms, 512.0, 22).is_empty());
        assert!(view.unproject(DVec2::new(400.0, 300.0)).is_some());
    }

    /// On the sphere the least zoom bounds its size as at the equator, so a
    /// globe as small as the limit allows is as small over a pole.
    #[test]
    fn the_least_zoom_bounds_the_globes_size() {
        let surface = globe_surface();
        let constraints = Constraints {
            min_zoom: 1.0,
            ..Constraints::default()
        };
        let mut equator = CameraState::new(Geodetic::new(0.0, 0.0), 0.0, &mercator());
        let mut north = CameraState::new(Geodetic::new(0.0, 80.0), 0.0, &mercator());
        north.set_globe_zoom(0.0);
        equator.constrain(&constraints, &surface, VIEWPORT);
        north.constrain(&constraints, &surface, VIEWPORT);
        assert!((equator.resolution / north.resolution - 1.0).abs() < 1e-12);
        assert!(north.zoom(&mercator()) < 1.0);
    }

    /// On the globe and on a low-zoom plane, a drag keeps the ground it
    /// grabbed under the pointer, near the poles and across the
    /// antimeridian too, and keeps the sphere's size or the plane's zoom.
    #[test]
    fn low_zoom_drags_hold_the_grabbed_ground() {
        let equal_earth = orb_crs::Globe {
            handover: Some([6.0, 7.0]),
            plane: Some(orb_crs::Projection::from_descriptor("equal-earth").unwrap()),
        };
        for (globe, cameras) in [
            (
                orb_crs::Globe::V8,
                &[
                    (10.0, 30.0, 2.0, 0.0, 0.0),
                    (179.0, -20.0, 4.0, 40.0, 30.0),
                    (-30.0, 82.0, 3.0, -120.0, 50.0),
                    (100.0, 10.0, 9.0, 10.0, 60.0),
                ][..],
            ),
            (
                equal_earth,
                &[
                    (-87.6, 41.9, 5.0, 0.0, 0.0),
                    (150.0, -30.0, 3.0, 30.0, 40.0),
                    (-87.6, 41.9, 6.5, -20.0, 20.0),
                ][..],
            ),
        ] {
            let surface = Surface {
                projection: mercator(),
                globe: Some(globe),
            };
            for &(lon, lat, zoom, bearing, pitch) in cameras {
                let mut camera = CameraState::new(Geodetic::new(lon, lat), zoom, &mercator());
                camera.bearing = bearing;
                camera.pitch = pitch;
                let at = DVec2::new(520.0, 420.0);
                for to in [DVec2::new(560.0, 380.0), DVec2::new(400.0, 470.0)] {
                    let view = View::new(&camera, VIEWPORT, &surface);
                    let ground = view.low_ground(at, view.elevation()).unwrap();
                    let mut moved = camera;
                    moved.pan(&surface, VIEWPORT, at, to);
                    let view = View::new(&moved, VIEWPORT, &surface);
                    let p = view.project(mercator().forward(ground)).unwrap();
                    assert!(p.distance(to) < 0.05, "{lon},{lat} z{zoom}: {p} for {to}");
                    // The sphere keeps its size, a plane its zoom.
                    if globe.plane.is_none() {
                        assert_eq!(moved.resolution, camera.resolution);
                    } else {
                        assert!((moved.zoom(&mercator()) - zoom).abs() < 1e-9);
                    }
                    assert_eq!(moved.bearing, bearing);
                }
                // Zooming around a point keeps it in place.
                let view = View::new(&camera, VIEWPORT, &surface);
                let ground = view.low_ground(at, view.elevation()).unwrap();
                let mut zoomed = camera;
                zoomed.zoom_around(&surface, VIEWPORT, at, zoom + 0.3);
                let view = View::new(&zoomed, VIEWPORT, &surface);
                let p = view.project(mercator().forward(ground)).unwrap();
                assert!(p.distance(at) < 0.05, "{lon},{lat}: zoom around {p}");
            }
        }
    }
}
