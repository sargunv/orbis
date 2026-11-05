//! The terrain a camera stands on, as the camera sees it: elevations to
//! cull and choose detail by, to put the target on the ground and to find
//! the ground under a pixel.

use glam::{DVec2, DVec3};
use orb_crs::{Projection, WorldRect};

use crate::{CameraState, View, Viewport};
use orb_crs::Surface;

/// Elevations of a map's terrain, in metres with its exaggeration: the
/// surface as drawn, wherever DEM data has loaded. Positions are world
/// coordinates of the map's projection, in any world copy.
pub trait Elevation: Send + Sync + std::fmt::Debug {
    /// The elevation at a world position, or `None` where no DEM data has
    /// loaded yet.
    fn at(&self, world: DVec2) -> Option<f64>;

    /// The lowest and highest elevation over a tile's world rectangle at
    /// zoom `z`, or `None` where unknown.
    fn range(&self, rect: WorldRect, z: u8) -> Option<(f64, f64)>;
}

/// How close, in metres, the eye and its near plane may come to the
/// terrain: a little slack over zero, so the near plane never cuts a
/// slope. GL JS keeps only the eye out of the terrain; a wider margin
/// lifts low, steeply pitched cameras above where the same camera stands
/// in GL JS, which moves their near plane and their fog.
const CLEARANCE_M: f64 = 2.0;

impl View {
    /// The terrain point under a logical pixel, as `(world, elevation in
    /// metres)`, found by walking the view ray (a CPU DEM ray walk,
    /// never a GPU readback). Where no DEM data has loaded the
    /// ground is at sea level. `None` above the horizon.
    pub fn unproject_terrain(&self, px: DVec2, terrain: &dyn Elevation) -> Option<(DVec2, f64)> {
        let (near, far) = self.ray(px, &self.view_proj);
        let ground = |p: DVec3| {
            terrain
                .at(p.truncate() + self.camera_world())
                .unwrap_or(0.0)
                * self.height_scale()
        };
        let above = |p: DVec3| p.z - ground(p);
        // Steps of about four pixels at the target's scale, as GL JS's
        // pick takes, then at most 512 of them spread geometrically.
        let length = near.distance(far);
        let step = 4.0 * self.world_per_px();
        let samples = (length / step).ceil().clamp(1.0, 512.0) as usize;
        let at = |i: usize| {
            let f = i as f64 / samples as f64;
            // Geometric spacing keeps the near steps fine on long rays.
            let t = if samples < 512 { f } else { f * f };
            near + (far - near) * t
        };
        let mut previous = near;
        if above(previous) < 0.0 {
            return None;
        }
        for i in 1..=samples {
            let p = at(i);
            if above(p) <= 0.0 {
                // Bisect between the last point above and this one.
                let (mut lo, mut hi) = (previous, p);
                for _ in 0..24 {
                    let mid = (lo + hi) / 2.0;
                    if above(mid) > 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let hit = (lo + hi) / 2.0;
                return Some((
                    hit.truncate() + self.camera_world(),
                    hit.z / self.height_scale(),
                ));
            }
            previous = p;
        }
        None
    }
}

impl View {
    /// How far the eye and its near plane stand above `terrain`, in
    /// metres: the least height over the ground of the eye and a grid of
    /// points across the near plane, where the terrain would show cut open.
    /// `None` where no DEM data has loaded under any of them.
    pub fn clearance(&self, terrain: &dyn Elevation) -> Option<f64> {
        self.near_points()
            .filter_map(|p| {
                let ground = terrain.at(p.truncate() + self.camera_world())?;
                Some(p.z / self.height_scale() - ground)
            })
            .min_by(f64::total_cmp)
    }

    /// The ground [`clearance`](Self::clearance) reads, in world
    /// coordinates: the terrain must have DEM data there too, though it
    /// lies outside the view.
    pub fn near_footprint(&self) -> WorldRect {
        let mut points = self
            .near_points()
            .map(|p| p.truncate() + self.camera_world());
        let first = points.next().expect("the eye");
        points.fold(
            WorldRect {
                min: first,
                max: first,
            },
            |r, p| WorldRect {
                min: r.min.min(p),
                max: r.max.max(p),
            },
        )
    }

    /// The eye and a grid across the near plane, camera-relative.
    fn near_points(&self) -> impl Iterator<Item = DVec3> + '_ {
        const GRID: usize = 5;
        let size = DVec2::new(self.viewport().width, self.viewport().height);
        let near = (0..GRID * GRID).map(move |k| {
            let cell = DVec2::new((k % GRID) as f64, (k / GRID) as f64);
            self.ray(cell / (GRID - 1) as f64 * size, &self.view_proj).0
        });
        std::iter::once(self.eye()).chain(near)
    }
}

impl View {
    /// The view of `camera` over `terrain`: its far plane reaches the
    /// lowest elevation of the DEM tile under the target, as GL JS's
    /// `minElevationForCurrentTile` makes it.
    pub fn over(
        camera: &CameraState,
        viewport: Viewport,
        surface: &Surface,
        terrain: &dyn Elevation,
    ) -> Self {
        let projection = &surface.projection;
        let world = camera.target_world(projection);
        let point = WorldRect {
            min: world,
            max: world,
        };
        let z = camera.zoom(projection).floor().clamp(0.0, 24.0) as u8;
        let lowest = terrain
            .range(point, z)
            .map_or(camera.elevation, |(lo, _)| lo);
        Self::on_ground(camera, viewport, surface, lowest)
    }
}

impl CameraState {
    /// Puts the target on the terrain under it, GL JS's
    /// `clampCenterToGround`: where its DEM data is still loading the
    /// target keeps its elevation, so the camera moves once, when the data
    /// lands (GL JS #8549). Returns whether it changed.
    pub fn settle_on(&mut self, terrain: &dyn Elevation, projection: &Projection) -> bool {
        match terrain.at(self.target_world(projection)) {
            Some(elevation) if elevation != self.elevation => {
                self.elevation = elevation;
                true
            }
            _ => false,
        }
    }

    /// Moves the target to the terrain along the view, keeping the eye
    /// where it is (GL JS's `recalculateZoomAndCenter`): the picture
    /// doesn't move while the target, its elevation and the zoom do. Ends
    /// gestures and animations that held the elevation. Returns whether it
    /// changed.
    pub fn recenter_on(
        &mut self,
        terrain: &dyn Elevation,
        surface: &Surface,
        viewport: Viewport,
    ) -> bool {
        let projection = &surface.projection;
        let view = View::over(self, viewport, surface, terrain);
        let Some((world, elevation)) = view.unproject_terrain(view.center(), terrain) else {
            return false;
        };
        let hit = (world - view.camera_world()).extend(elevation * view.height_scale());
        let distance = view.eye().distance(hit);
        let eye_distance_px = 0.5 * viewport.height / (self.fov.to_radians() / 2.0).tan();
        let before = *self;
        self.set_target_world(projection, world, distance / eye_distance_px);
        self.elevation = elevation;
        *self != before
    }

    /// Raises the camera until the eye and its near plane clear the
    /// terrain by a few tens of metres ([`View::clearance`]). Raising moves
    /// them straight up, over the same ground, so one step suffices, and
    /// the lift follows the ground continuously as the camera moves.
    /// Returns whether it moved.
    pub fn keep_above(
        &mut self,
        terrain: &dyn Elevation,
        surface: &Surface,
        viewport: Viewport,
    ) -> bool {
        let view = View::new(self, viewport, surface);
        let Some(clearance) = view.clearance(terrain) else {
            return false;
        };
        let short = CLEARANCE_M - clearance;
        if short <= 0.0 {
            return false;
        }
        self.elevation += short;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Camera, CameraTarget};
    use orb_crs::Geodetic;
    use orb_crs::TileMatrixSet;
    use std::sync::Arc;

    /// A ridge rising eastward, loaded only west of `loaded_until`.
    #[derive(Debug)]
    struct Ridge {
        loaded_until: f64,
    }

    impl Elevation for Ridge {
        fn at(&self, world: DVec2) -> Option<f64> {
            (world.x < self.loaded_until).then(|| 1000.0 + world.x.max(0.0) * 1e-3)
        }
        fn range(&self, _: WorldRect, _: u8) -> Option<(f64, f64)> {
            Some((0.0, 5000.0))
        }
    }

    fn setup() -> (Projection, Viewport, CameraState) {
        let projection = *TileMatrixSet::by_id("WebMercatorQuad")
            .unwrap()
            .projection();
        let viewport = Viewport {
            width: 800.0,
            height: 600.0,
        };
        let mut camera = CameraState::new(Geodetic::new(0.0, 0.0), 13.0, &projection);
        camera.pitch = 60.0;
        (projection, viewport, camera)
    }

    #[test]
    fn the_ray_walk_finds_the_drawn_surface() {
        let (projection, viewport, mut camera) = setup();
        let ridge = Ridge {
            loaded_until: f64::INFINITY,
        };
        assert!(camera.settle_on(&ridge, &projection));
        assert_eq!(camera.elevation, 1000.0);
        let view = View::new(&camera, viewport, &Surface::from(&projection));
        let center = DVec2::new(400.0, 300.0);
        let (world, elevation) = view.unproject_terrain(center, &ridge).unwrap();
        assert!(world.distance(view.camera_world()) < 0.01, "{world}");
        assert!((elevation - 1000.0).abs() < 0.01);
        // Off centre, the hit projects back to its pixel.
        let px = DVec2::new(200.0, 500.0);
        let (world, elevation) = view.unproject_terrain(px, &ridge).unwrap();
        assert!((ridge.at(world).unwrap() - elevation).abs() < 0.01);
        let back = view
            .project_3d(world, elevation * view.height_scale())
            .unwrap();
        assert!(back.distance(px) < 0.01, "{back}");
        // Sky has no ground.
        camera.pitch = 85.0;
        let view = View::new(&camera, viewport, &Surface::from(&projection));
        assert!(
            view.unproject_terrain(DVec2::new(400.0, 0.0), &ridge)
                .is_none()
        );
    }

    #[test]
    fn recentering_keeps_the_eye_in_place() {
        let (projection, viewport, mut camera) = setup();
        let ridge = Ridge {
            loaded_until: f64::INFINITY,
        };
        camera.elevation = 200.0;
        let eye = |c: &CameraState| {
            let view = View::new(c, viewport, &Surface::from(&projection));
            view.camera_world().extend(0.0) + view.eye()
        };
        let before = eye(&camera);
        assert!(camera.recenter_on(&ridge, &Surface::from(&projection), viewport));
        assert!((camera.elevation - 1000.0).abs() < 1.0);
        assert!(before.distance(eye(&camera)) < 1e-3 * before.length().max(1.0));
        assert!(camera.zoom(&projection) > 13.0, "the target came closer");
    }

    #[test]
    fn the_eye_stays_above_the_terrain() {
        let (projection, viewport, mut camera) = setup();
        let ridge = Ridge {
            loaded_until: f64::INFINITY,
        };
        camera.set_zoom(&projection, 20.0);
        camera.pitch = 80.0;
        camera.elevation = 0.0;
        assert!(camera.keep_above(&ridge, &Surface::from(&projection), viewport));
        let view = View::new(&camera, viewport, &Surface::from(&projection));
        let eye_m = view.eye().z / view.height_scale();
        assert!(eye_m >= 1000.0 + CLEARANCE_M - 1e-6);
    }

    /// A cone 3000 m high with 45° slopes, its summit at the origin.
    #[derive(Debug)]
    struct Peak;

    impl Elevation for Peak {
        fn at(&self, world: DVec2) -> Option<f64> {
            // World units are metres on the equator.
            Some((3000.0 - world.length()).max(0.0))
        }
        fn range(&self, _: WorldRect, _: u8) -> Option<(f64, f64)> {
            Some((0.0, 3000.0))
        }
    }

    /// Follows a store's camera over the peak: after each step the eye and
    /// its near plane clear the slope by the margin, and during a move the
    /// camera's elevation changes no faster than the slope under the
    /// moving eye and near plane allows (no jumps).
    struct Watch {
        projection: Projection,
        viewport: Viewport,
        /// Elevation, footprint centre and eye of the last step.
        last: Option<(f64, DVec2, DVec3)>,
        steps: usize,
    }

    impl Watch {
        fn check(&mut self, store: &Camera, what: &str) {
            let (view, clearance) = self.clearance(store, what);
            let footprint = view.near_footprint();
            let (elevation, at) = (store.state().elevation, footprint.center());
            if let Some((before, was, _)) = self.last {
                // 45° slopes: the ground under the samples moves at most as
                // far as they do, which is at most the footprint's centre
                // shift plus its extent.
                let moved = at.distance(was) + footprint.size().length();
                assert!(
                    (elevation - before).abs() <= moved + 1.0,
                    "{what}, step {}: elevation {before} -> {elevation} over {moved} m ({clearance} m clear)",
                    self.steps
                );
            }
            self.last = Some((elevation, at, Self::eye(&view)));
        }

        /// A gesture ended: the target moved onto the terrain along the
        /// view, the eye stayed where it was.
        fn released(&mut self, store: &Camera, what: &str) {
            let (view, _) = self.clearance(store, what);
            let (_, _, eye) = self.last.expect("a step before");
            let now = Self::eye(&view);
            assert!(
                eye.distance(now) < 1.0,
                "{what}: the eye moved {eye} -> {now}"
            );
            self.last = Some((store.state().elevation, view.near_footprint().center(), now));
        }

        fn clearance(&mut self, store: &Camera, what: &str) -> (View, f64) {
            let view = View::new(
                store.state(),
                self.viewport,
                &Surface::from(&self.projection),
            );
            let clearance = view.clearance(&Peak).unwrap();
            assert!(
                clearance >= CLEARANCE_M - 1e-6,
                "{what}, step {}: {clearance} m",
                self.steps
            );
            self.steps += 1;
            (view, clearance)
        }

        /// The eye in world units, its height in metres.
        fn eye(view: &View) -> DVec3 {
            (view.eye().truncate() + view.camera_world()).extend(view.eye().z / view.height_scale())
        }
    }

    #[test]
    fn gestures_and_flights_into_a_slope_keep_the_eye_above_it() {
        let (projection, viewport, _) = setup();
        // 4 km west of the summit, low, looking east at the slope.
        let mut state = CameraState::new(
            projection.inverse(DVec2::new(-4000.0, 0.0)),
            15.0,
            &projection,
        );
        state.pitch = 75.0;
        state.bearing = 90.0;
        let mut store = Camera::new(state, Surface::from(projection), viewport);
        store.set_terrain(Some(Arc::new(Peak)));
        let mut watch = Watch {
            projection,
            viewport,
            last: None,
            steps: 0,
        };
        watch.check(&store, "start");
        let center = DVec2::new(400.0, 300.0);
        // Drag the slope toward the eye, through the summit and beyond.
        store.hold();
        for _ in 0..400 {
            store.pan(center, center + DVec2::new(0.0, 20.0));
            watch.check(&store, "pan");
        }
        store.release();
        watch.released(&store, "pan end");
        // Zoom into a point low on the screen, on the slope.
        store.hold();
        for _ in 0..100 {
            store.zoom_by(DVec2::new(400.0, 500.0), 0.05);
            watch.check(&store, "zoom");
        }
        // Turn and tilt toward the horizon there.
        for _ in 0..90 {
            store.rotate_by(center, 4.0);
            store.pitch_by(0.1);
            watch.check(&store, "rotate");
        }
        store.release();
        watch.released(&store, "gesture end");
        // Fly low across the summit to its far side.
        let mut now = 0.0;
        store.fly_to(
            &CameraTarget {
                center: Some(projection.inverse(DVec2::new(3500.0, 200.0))),
                zoom: Some(16.5),
                pitch: Some(80.0),
                bearing: Some(270.0),
                ..CameraTarget::default()
            },
            &crate::FlyOptions {
                duration: Some(4000.0),
                ..crate::FlyOptions::default()
            },
            Some(now),
        );
        while store.is_animating() {
            now += 16.0;
            store.frame(now);
            watch.check(&store, "flight");
        }
        assert!(watch.steps > 800);
    }

    #[test]
    fn dem_landing_mid_flight_bends_the_path() {
        let (projection, viewport, state) = setup();
        let mut store = Camera::new(state, Surface::from(projection), viewport);
        let destination = Geodetic::new(0.05, 0.0);
        store.fly_to(
            &CameraTarget {
                center: Some(destination),
                ..CameraTarget::default()
            },
            &crate::FlyOptions {
                duration: Some(2000.0),
                ..crate::FlyOptions::default()
            },
            Some(0.0),
        );
        let (mut now, mut last, mut largest) = (0.0, 0.0, 0.0f64);
        while store.is_animating() {
            now += 16.0;
            if now == 1008.0 {
                store.set_terrain(Some(Arc::new(Ridge {
                    loaded_until: f64::INFINITY,
                })));
            }
            store.frame(now);
            let elevation = store.state().elevation;
            largest = largest.max((elevation - last).abs());
            last = elevation;
        }
        // Halfway, 1000 m of ground land under a camera at sea level: the
        // rest of the flight climbs to it, about 1000 m over 62 frames
        // with the easing's steepest frames above the mean.
        assert!(largest < 60.0, "{largest} m in one frame");
        let ground = Ridge {
            loaded_until: f64::INFINITY,
        }
        .at(projection.forward(destination))
        .unwrap();
        assert!((store.state().elevation - ground).abs() < 1e-6);
    }

    #[test]
    fn a_jump_keeps_its_elevation_until_the_dem_lands() {
        let (projection, viewport, state) = setup();
        let mut store = Camera::new(state, Surface::from(projection), viewport);
        store.set_terrain(Some(Arc::new(Ridge { loaded_until: 1.0 })));
        assert_eq!(store.state().elevation, 1000.0);
        // East of the loaded data the target keeps 1000 m (GL JS #8549).
        store.jump_to(&CameraTarget {
            center: Some(Geodetic::new(10.0, 0.0)),
            ..CameraTarget::default()
        });
        assert_eq!(store.state().elevation, 1000.0);
        // The data lands: it moves once.
        store.set_terrain(Some(Arc::new(Ridge {
            loaded_until: f64::INFINITY,
        })));
        let landed = store.state().elevation;
        assert!(landed > 1000.0);
        store.set_terrain(None);
        assert_eq!(store.state().elevation, 0.0);
    }

    #[test]
    fn an_unclamped_target_keeps_the_elevation_it_is_given() {
        let (projection, viewport, state) = setup();
        let mut store = Camera::new(state, Surface::from(projection), viewport);
        store.set_terrain(Some(Arc::new(Ridge {
            loaded_until: f64::INFINITY,
        })));
        let ground = store.state().elevation;
        // Clamped, the terrain takes the target back.
        store.set_center_elevation(ground + 500.0);
        assert_eq!(store.state().elevation, ground);
        store.set_center_clamped_to_ground(false);
        store.set_center_elevation(ground + 500.0);
        assert_eq!(store.state().elevation, ground + 500.0);
        store.jump_to(&CameraTarget {
            center: Some(Geodetic::new(10.0, 0.0)),
            ..CameraTarget::default()
        });
        assert_eq!(store.state().elevation, ground + 500.0);
        // A move with an elevation of its own keeps it, unclamped.
        let mut now = 0.0;
        store.ease_to(
            &CameraTarget {
                center: Some(Geodetic::new(5.0, 0.0)),
                elevation: Some(ground + 800.0),
                ..CameraTarget::default()
            },
            crate::EaseOptions::default(),
            Some(now),
        );
        while store.is_animating() {
            now += 16.0;
            store.frame(now);
        }
        assert_eq!(store.state().elevation, ground + 800.0);
        // Clamped again, it stands on the ground under it.
        store.set_center_clamped_to_ground(true);
        assert_ne!(store.state().elevation, ground + 500.0);
    }
}
