//! The terrain a camera stands on, as the camera sees it: elevations to
//! cull and choose detail by, to put the target on the ground and to find
//! the ground under a pixel.

use glam::{DVec2, DVec3};
use orb_crs::{Projection, WorldRect};

use crate::{CameraState, View, Viewport};

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

/// How close, in metres, the eye may come to the terrain under it.
const CLEARANCE_M: f64 = 50.0;

impl View {
    /// The terrain point under a logical pixel, as `(world, elevation in
    /// metres)`, found by walking the view ray (plan §9.4: a CPU DEM ray
    /// walk, never a GPU readback). Where no DEM data has loaded the
    /// ground is at sea level. `None` above the horizon.
    pub fn unproject_terrain(&self, px: DVec2, terrain: &dyn Elevation) -> Option<(DVec2, f64)> {
        let (near, far) = self.ray(px);
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
    /// The view of `camera` over `terrain`: its far plane reaches the
    /// lowest elevation of the DEM tile under the target, as GL JS's
    /// `minElevationForCurrentTile` makes it.
    pub fn over(
        camera: &CameraState,
        viewport: Viewport,
        projection: &Projection,
        terrain: &dyn Elevation,
    ) -> Self {
        let world = camera.target_world(projection);
        let point = WorldRect {
            min: world,
            max: world,
        };
        let z = camera.zoom().floor().clamp(0.0, 24.0) as u8;
        let lowest = terrain
            .range(point, z)
            .map_or(camera.elevation, |(lo, _)| lo);
        Self::on_ground(camera, viewport, projection, lowest)
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
        projection: &Projection,
        viewport: Viewport,
    ) -> bool {
        let view = View::over(self, viewport, projection, terrain);
        let center = DVec2::new(viewport.width, viewport.height) / 2.0;
        let Some((world, elevation)) = view.unproject_terrain(center, terrain) else {
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

    /// Raises the target until the eye clears the terrain under it by a
    /// few tens of metres. Returns whether it moved.
    pub fn keep_above(
        &mut self,
        terrain: &dyn Elevation,
        projection: &Projection,
        viewport: Viewport,
    ) -> bool {
        let view = View::new(self, viewport, projection);
        let eye = view.eye();
        let Some(ground) = terrain.at(eye.truncate() + view.camera_world()) else {
            return false;
        };
        let eye_m = eye.z / view.height_scale();
        let short = ground + CLEARANCE_M - eye_m;
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
    use orb_crs::TileMatrixSet;
    use orb_geo::Geodetic;
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
        let mut camera = CameraState::new(Geodetic::new(0.0, 0.0), 13.0);
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
        let view = View::new(&camera, viewport, &projection);
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
        let view = View::new(&camera, viewport, &projection);
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
            let view = View::new(c, viewport, &projection);
            view.camera_world().extend(0.0) + view.eye()
        };
        let before = eye(&camera);
        assert!(camera.recenter_on(&ridge, &projection, viewport));
        assert!((camera.elevation - 1000.0).abs() < 1.0);
        assert!(before.distance(eye(&camera)) < 1e-3 * before.length().max(1.0));
        assert!(camera.zoom() > 13.0, "the target came closer");
    }

    #[test]
    fn the_eye_stays_above_the_terrain() {
        let (projection, viewport, mut camera) = setup();
        let ridge = Ridge {
            loaded_until: f64::INFINITY,
        };
        camera.set_zoom(20.0);
        camera.pitch = 80.0;
        camera.elevation = 0.0;
        assert!(camera.keep_above(&ridge, &projection, viewport));
        let view = View::new(&camera, viewport, &projection);
        let eye_m = view.eye().z / view.height_scale();
        assert!(eye_m >= 1000.0 + CLEARANCE_M - 1e-6);
    }

    #[test]
    fn a_jump_keeps_its_elevation_until_the_dem_lands() {
        let (projection, viewport, state) = setup();
        let mut store = Camera::new(state, projection, viewport);
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
}
