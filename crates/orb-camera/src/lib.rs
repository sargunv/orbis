//! Camera state and the view transforms derived from it.
//!
//! [`CameraState`] is the single source of truth: a target, a ground
//! resolution and an orientation. A [`View`] combines it with a viewport and a
//! projection into the matrices and tile cover a frame needs. World positions
//! handed to the GPU are relative to the camera's world position so they stay
//! precise in f32 at any zoom.

#![forbid(unsafe_code)]

mod animate;
mod fit;
mod moves;
mod store;
mod terrain;

pub use animate::{CameraTarget, EaseOptions, Easing, FlyOptions, Glide, Track};
pub use fit::{FitOptions, Padding};
pub use moves::{Bounds, Constraints, wrap_bearing};
pub use store::{AnimationId, Camera, CameraEvent, Change, Phase};
pub use terrain::Elevation;

use std::f64::consts::{FRAC_PI_2, PI};

use glam::{DMat4, DVec2, DVec3, DVec4};
use orb_crs::{
    CoverDecision, CoveredTile, Footprint, Projection, TileIndex, TileMatrixSet, WorldRect,
};
use orb_geo::{Geodetic, wgs84};

/// Where the camera looks and how.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraState {
    /// The geodetic point at the centre of the viewport.
    pub target: Geodetic,
    /// Ground metres per logical pixel at the target.
    pub resolution: f64,
    /// Clockwise rotation of the map from north-up, in degrees.
    pub bearing: f64,
    /// Tilt from looking straight down, in degrees.
    pub pitch: f64,
    /// Vertical field of view, in degrees.
    pub fov: f64,
    /// The target's height above the surface in metres (GL JS's center
    /// elevation): the camera orbits a point this high. On terrain it is
    /// the ground's elevation under the target.
    pub elevation: f64,
}

/// Size of one tile side at its own zoom in the style zoom convention, in
/// logical pixels.
const STYLE_ZOOM_TILE_PX: f64 = 512.0;

/// Finite horizon policy for a flat surface. Keeping a small angle below
/// horizontal bounds the far range and the LOD integral, as in GL JS.
const MAX_HORIZON_PITCH: f64 = 89.25;

impl CameraState {
    /// GL JS's default vertical field of view.
    pub const DEFAULT_FOV: f64 = 36.869_897_645_844_02;

    /// A camera looking straight down at `target` with the given style zoom.
    pub fn new(target: Geodetic, zoom: f64) -> Self {
        Self {
            target,
            resolution: resolution_at_zoom(target, zoom),
            bearing: 0.0,
            pitch: 0.0,
            fov: Self::DEFAULT_FOV,
            elevation: 0.0,
        }
    }

    /// The style zoom: v8 stops, `minzoom` and `maxzoom` use it on every
    /// surface, matching GL JS.
    pub fn zoom(&self) -> f64 {
        (equator_circumference_px_per_m(self.target) / self.resolution).log2()
    }

    pub fn set_zoom(&mut self, zoom: f64) {
        self.resolution = resolution_at_zoom(self.target, zoom);
    }
}

/// Ground resolution at `target` for a style zoom.
fn resolution_at_zoom(target: Geodetic, zoom: f64) -> f64 {
    equator_circumference_px_per_m(target) / zoom.exp2()
}

/// Style zoom's scale at zoom 0: the length of the parallel through `target`
/// divided by one tile's width.
fn equator_circumference_px_per_m(target: Geodetic) -> f64 {
    2.0 * PI * wgs84::SEMI_MAJOR_M * target.lat.to_radians().cos() / STYLE_ZOOM_TILE_PX
}

/// A viewport in logical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub width: f64,
    pub height: f64,
}

/// A camera resolved against a viewport and a projection for one frame.
#[derive(Clone, Debug)]
pub struct View {
    viewport: Viewport,
    pitch: f64,
    fov: f64,
    camera_world: DVec2,
    world_per_px: f64,
    /// Camera-relative world (metres, z up) → clip space, depth in [0, 1].
    view_proj: DMat4,
    /// The eye in camera-relative world space.
    eye: DVec3,
    /// The target's height in world units: camera-relative world space
    /// keeps the surface datum at z = 0.
    target_z: f64,
    /// Distance from the eye to sea level along the view, in pixels: GL
    /// JS's fog matrix starts here.
    sea_level_px: f64,
    /// Distance from the eye to the target, in pixels.
    eye_distance_px: f64,
    /// Distance from the eye to the farthest visible ground point, in pixels.
    far_px: f64,
    wrap_period: Option<f64>,
    frustum_min: DVec3,
    frustum_max: DVec3,
    height_scale: f64,
}

impl View {
    pub fn new(camera: &CameraState, viewport: Viewport, projection: &Projection) -> Self {
        Self::on_ground(camera, viewport, projection, camera.elevation)
    }

    /// A view over ground whose lowest elevation near the target is
    /// `lowest` metres (GL JS's `minElevationForCurrentTile`, from the
    /// terrain's DEM tile under the target): the far plane reaches down to
    /// it.
    pub fn on_ground(
        camera: &CameraState,
        viewport: Viewport,
        projection: &Projection,
        lowest: f64,
    ) -> Self {
        let target = camera.target.wrapped();
        let camera_world = projection.forward(target);
        let world_per_px = camera.resolution * projection.scale_factor(target);

        let fov = camera.fov.to_radians();
        let pitch = camera.pitch.to_radians();
        let half_fov = fov / 2.0;
        let eye_distance_px = 0.5 * viewport.height / half_fov.tan();
        let height_scale = projection.style_height_scale(target);
        let target_z = camera.elevation * height_scale;
        // GL JS's `_calculateNearFarZ`: the far range reaches sea level,
        // or the lowest of the target and a plane 100 style metres below
        // the eye when that is lower. Omitting the plane below the eye
        // clips high-zoom ground too early.
        let px_per_m = height_scale / world_per_px;
        let limited = pitch.min(MAX_HORIZON_PITCH.to_radians());
        let sea_level_px = (eye_distance_px / 2.0)
            .max(eye_distance_px + camera.elevation * px_per_m / limited.cos());
        let altitude = camera.elevation + eye_distance_px * pitch.cos() / px_per_m;
        let lowest = camera.elevation.min(lowest).min(altitude - 100.0);
        let lowest_plane_px = if lowest < 0.0 {
            sea_level_px - lowest * px_per_m / limited.cos()
        } else {
            sea_level_px
        };

        // Distance to the ground point at the top edge of the viewport, after
        // GL JS's transform.
        let ground_angle = FRAC_PI_2 + pitch;
        let top_half_surface = half_fov.sin() * lowest_plane_px
            / (PI - ground_angle - half_fov).clamp(0.01, PI - 0.01).sin();
        let horizon_angle = ((FRAC_PI_2 - pitch).tan() * 0.85)
            .min((MAX_HORIZON_PITCH.to_radians() - pitch).tan())
            .atan();
        let horizon_fov = horizon_angle.max((90.0 - MAX_HORIZON_PITCH).to_radians());
        let horizon_surface = horizon_fov.sin() * lowest_plane_px
            / (PI - ground_angle - horizon_fov)
                .clamp(0.01, PI - 0.01)
                .sin();
        let far_px = ((FRAC_PI_2 - pitch).cos() * top_half_surface.min(horizon_surface)
            + lowest_plane_px)
            * 1.01;
        let near_px = viewport.height / 50.0;

        let to_px = DMat4::from_scale(DVec3::splat(1.0 / world_per_px));
        let view = DMat4::from_translation(DVec3::new(0.0, 0.0, -eye_distance_px))
            * DMat4::from_rotation_x(-pitch)
            * DMat4::from_rotation_z(camera.bearing.to_radians())
            * to_px
            * DMat4::from_translation(DVec3::new(0.0, 0.0, -target_z));
        let proj = glam::dcamera::rh::proj::directx::perspective(
            fov,
            viewport.width / viewport.height,
            near_px,
            far_px,
        );
        let eye = view.inverse().transform_point3(DVec3::ZERO);

        let view_proj = proj * view;
        let inverse = view_proj.inverse();
        let mut frustum_min = DVec3::splat(f64::INFINITY);
        let mut frustum_max = DVec3::splat(f64::NEG_INFINITY);
        for i in 0..8 {
            let corner = inverse.project_point3(DVec3::new(
                if i & 1 == 0 { -1.0 } else { 1.0 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { 0.0 } else { 1.0 },
            ));
            frustum_min = frustum_min.min(corner);
            frustum_max = frustum_max.max(corner);
        }

        Self {
            viewport,
            pitch: camera.pitch,
            fov: camera.fov,
            camera_world,
            world_per_px,
            view_proj,
            eye,
            target_z,
            sea_level_px,
            eye_distance_px,
            far_px,
            wrap_period: projection.wrap_period(),
            frustum_min,
            frustum_max,
            height_scale,
        }
    }

    /// Top logical pixel row where the ground plane enters the finite frustum.
    pub fn ground_horizon(&self) -> f64 {
        if self.pitch <= 0.0 {
            return 0.0;
        }
        let pitch = self.pitch.to_radians();
        let ndc_y = (1.0 - self.eye_distance_px / self.far_px)
            / pitch.tan()
            / (self.fov.to_radians() / 2.0).tan();
        ((1.0 - ndc_y) * 0.5 * self.viewport.height).max(0.0)
    }

    /// Top logical row of the apparent horizon, including the reference's
    /// finite planar horizon adjustment. Sky starts here and extends upward.
    pub fn sky_horizon(&self) -> f64 {
        let pitch = self.pitch.to_radians();
        let offset = self.eye_distance_px
            * ((FRAC_PI_2 - pitch).tan() * 0.85)
                .min((MAX_HORIZON_PITCH.to_radians() - pitch).tan());
        self.viewport.height / 2.0 - offset
    }

    pub fn viewport(&self) -> Viewport {
        self.viewport
    }

    /// The camera target's world position: the origin of camera-relative
    /// world space.
    pub fn camera_world(&self) -> DVec2 {
        self.camera_world
    }

    /// World units per logical pixel at the target.
    pub fn world_per_px(&self) -> f64 {
        self.world_per_px
    }

    /// World units per metre of v8 feature height at the target.
    pub fn height_scale(&self) -> f64 {
        self.height_scale
    }

    /// Camera-relative world → clip space, with depth in [0, 1].
    pub fn view_proj(&self) -> DMat4 {
        self.view_proj
    }

    /// Projects a world position to logical pixels, top-left origin, or
    /// `None` if it is behind the eye.
    pub fn project(&self, world: DVec2) -> Option<DVec2> {
        self.project_3d(world, self.target_z)
    }

    /// Projects a world position `z` world units above the datum.
    pub fn project_3d(&self, world: DVec2, z: f64) -> Option<DVec2> {
        let clip = self.view_proj * (world - self.camera_world).extend(z).extend(1.0);
        (clip.w > 0.0).then(|| {
            let ndc = clip.truncate().truncate() / clip.w;
            DVec2::new(
                (ndc.x + 1.0) * 0.5 * self.viewport.width,
                (1.0 - ndc.y) * 0.5 * self.viewport.height,
            )
        })
    }

    /// The world position under a point in logical pixels on the plane
    /// through the target, level with it (the ground without terrain; on
    /// terrain the plane gestures drag, as GL JS's do), or `None` above
    /// its horizon.
    pub fn unproject(&self, px: DVec2) -> Option<DVec2> {
        let (near, far) = self.ray(px);
        let t = (near.z - self.target_z) / (near.z - far.z);
        (far.z < near.z && t >= 0.0)
            .then(|| (near + (far - near) * t).truncate() + self.camera_world)
    }

    /// The near and far ends of the view ray through a point in logical
    /// pixels, in camera-relative world space.
    fn ray(&self, px: DVec2) -> (DVec3, DVec3) {
        let ndc = DVec2::new(
            px.x / self.viewport.width * 2.0 - 1.0,
            1.0 - px.y / self.viewport.height * 2.0,
        );
        let inverse = self.view_proj.inverse();
        (
            inverse.project_point3(ndc.extend(0.0)),
            inverse.project_point3(ndc.extend(1.0)),
        )
    }

    /// The target's height in world units.
    pub fn target_z(&self) -> f64 {
        self.target_z
    }

    /// GL JS's fog depth range in pixels along the view: from where the
    /// view meets sea level to the far plane.
    pub fn fog_range_px(&self) -> (f64, f64) {
        (self.sea_level_px, self.far_px)
    }

    /// The eye's position in camera-relative world space.
    pub fn eye(&self) -> DVec3 {
        self.eye
    }

    /// The tiles of `tms` that cover the view, choosing for each the zoom at
    /// which GL JS's default level-of-detail policy selects `tile_px` tiles.
    pub fn cover(&self, tms: &TileMatrixSet, tile_px: f64, max_zoom: u8) -> Vec<CoveredTile> {
        self.cover_volume(tms, tile_px, max_zoom, 0.0, None)
    }

    /// Covers ground and elevated content. `height` is the maximum content
    /// height in world units. As the bottom edge approaches the horizon the
    /// culling volume grows toward this height (at least 500 style metres),
    /// retaining tiles whose roofs can enter the view above their ground.
    /// On `terrain`, tiles are culled by the elevations under them where
    /// known, and detail varies with distance at any pitch, as GL JS's
    /// cover does with terrain.
    pub fn cover_volume(
        &self,
        tms: &TileMatrixSet,
        tile_px: f64,
        max_zoom: u8,
        height: f64,
        terrain: Option<&dyn Elevation>,
    ) -> Vec<CoveredTile> {
        let center_zoom = (tms.tile_span(0) / (self.world_per_px * tile_px)).log2();
        self.cover_tiles(tms, max_zoom, height, terrain, |index, offset| {
            let bounds = tms.tile_bounds(index).translated(DVec2::new(offset, 0.0));
            Some((bounds, center_zoom))
        })
    }

    /// Covers the view with tiles of a set on another plane (the
    /// `RasterWarp` plan), from where each tile lands on this one.
    pub fn cover_warped(
        &self,
        tms: &TileMatrixSet,
        tile_px: f64,
        max_zoom: u8,
        mut footprint: impl FnMut(TileIndex) -> Option<Footprint>,
    ) -> Vec<CoveredTile> {
        let scale = self.world_per_px * tile_px;
        self.cover_tiles(tms, max_zoom, 0.0, None, |index, offset| {
            let f = footprint(index)?;
            // The zoom at which this tile, at its own scale, would span
            // `tile_px` pixels at the camera.
            let center = f64::from(index.z) + (f.span / scale).log2();
            Some((f.bounds.translated(DVec2::new(offset, 0.0)), center))
        })
    }

    /// The cover's traversal: `place` gives each tile's bounds in its copy
    /// of the world (by x offset) and the zoom whose tiles would span the
    /// source's tile size at the target's scale, or `None` to skip it.
    fn cover_tiles(
        &self,
        tms: &TileMatrixSet,
        max_zoom: u8,
        height: f64,
        terrain: Option<&dyn Elevation>,
        mut place: impl FnMut(TileIndex, f64) -> Option<(WorldRect, f64)>,
    ) -> Vec<CoveredTile> {
        // GL JS grows the culling volume as the bottom edge approaches the
        // horizon. Heights here stay in world units, just like tile bounds.
        let proximity =
            ((15.0 - (MAX_HORIZON_PITCH - self.pitch - self.fov / 2.0)) / 15.0).clamp(0.0, 1.0);
        let culling_z = self.target_z + proximity * height.max(500.0 * self.height_scale);
        let flat_range = (culling_z.min(0.0), culling_z.max(0.0));
        let reach = (self.far_px * self.world_per_px).max(self.viewport.width * self.world_per_px);
        // Match the reference's bounded repeated-world policy: at most
        // three copies on either side, with the same generic DFS in each.
        let copies = self
            .wrap_period
            .map_or(0, |period| ((reach / period).ceil() as i32).min(3));
        // GL JS holds a constant zoom through 60 degrees (or the
        // FOV-dependent threshold) without terrain. Above it, and on
        // terrain, use its default tile LOD.
        let threshold = (78.5 - self.fov / 2.0).clamp(0.0, 60.0);
        let lod = (terrain.is_some() || self.pitch > threshold).then(|| {
            let horizon = MAX_HORIZON_PITCH.to_radians();
            let fov = self.fov.to_radians();
            let behavior =
                2.0 * ((9.314 - 1.0) / ((horizon - fov).cos() / horizon.cos()).log2() - 1.0);
            let integral = |lo: f64, hi: f64| {
                (0..10)
                    .map(|i| {
                        let x = lo + (f64::from(i) + 0.5) / 10.0 * (hi - lo);
                        (hi - lo) / 10.0 * x.cos().powf(behavior - 1.0)
                    })
                    .sum::<f64>()
            };
            let pitch = self.pitch.to_radians();
            let hi = horizon.min(pitch + fov / 2.0);
            let count = integral(hi.min(pitch - fov / 2.0), hi);
            let count0 = 2.0 * integral(0.0, fov / 2.0);
            (
                behavior,
                (fov / 2.0).cos().max(0.5),
                (count / count0 / 3.0).max(1.0).log2() / 2.0,
            )
        });
        // The eye's height above the target, GL JS's `distanceToTileZ`.
        let eye_height = (self.eye.z - self.target_z).abs();
        tms.cover_with(
            -copies..=copies,
            self.wrap_period,
            max_zoom,
            |index, offset| {
                let Some((bounds, center_zoom)) = place(index, offset) else {
                    return CoverDecision::Skip;
                };
                let (lo, hi) = terrain
                    .and_then(|t| t.range(bounds, index.z))
                    .map_or(flat_range, |(lo, hi)| {
                        (lo * self.height_scale, hi * self.height_scale)
                    });
                let rel = bounds.translated(-self.camera_world);
                if !self.intersects_frustum(&rel, lo, hi) {
                    return CoverDecision::Skip;
                }
                let desired = if let Some((behavior, fov_scale, penalty)) = lod {
                    let distance_2d = rel.clamp(self.eye.truncate()).distance(self.eye.truncate());
                    let distance = distance_2d.hypot(eye_height).max(1.0e-9);
                    let tile_pitch = distance_2d.atan2(eye_height);
                    let center_distance = self.eye_distance_px * self.world_per_px;
                    center_zoom
                        + (center_distance / distance / fov_scale).log2()
                        + behavior * tile_pitch.cos().log2() / 2.0
                        - penalty
                } else {
                    center_zoom
                };
                if f64::from(index.z) < desired.floor().max(0.0) {
                    CoverDecision::Refine
                } else {
                    CoverDecision::Accept
                }
            },
        )
    }

    /// Conservative test of a camera-relative ground rectangle against the
    /// view frustum.
    fn intersects_frustum(&self, rect: &WorldRect, lo: f64, hi: f64) -> bool {
        // The frustum planes alone can accept disjoint boxes near a corner.
        // Include the box's three axes, as GL JS's AABB test does.
        if rect.min.x > self.frustum_max.x
            || rect.min.y > self.frustum_max.y
            || lo > self.frustum_max.z
            || rect.max.x < self.frustum_min.x
            || rect.max.y < self.frustum_min.y
            || hi < self.frustum_min.z
        {
            return false;
        }
        let corners: [_; 8] = std::array::from_fn(|i| {
            let x = if i & 1 == 0 { rect.min.x } else { rect.max.x };
            let y = if i & 2 == 0 { rect.min.y } else { rect.max.y };
            let z = if i & 4 == 0 { lo } else { hi };
            self.view_proj * DVec4::new(x, y, z, 1.0)
        });
        let all = |f: &dyn Fn(&DVec4) -> bool| corners.iter().all(f);
        !(all(&|c| c.x > c.w)
            || all(&|c| c.x < -c.w)
            || all(&|c| c.y > c.w)
            || all(&|c| c.y < -c.w)
            || all(&|c| c.z > c.w)
            || all(&|c| c.z < 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_crs::TileIndex;

    #[test]
    fn mercator_cameras_keep_to_the_world_vertically() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let p = tms.projection();
        let world = Constraints::WORLD;
        let viewport = Viewport {
            width: 256.0,
            height: 256.0,
        };
        // Near the pole, the target moves south until the top edge is the
        // world's (GL JS `getConstrained`).
        let mut camera = CameraState::new(Geodetic::new(0.0, 85.0), 4.0);
        camera.constrain(&world, p, viewport);
        assert!((camera.zoom() - 4.0).abs() < 1e-9);
        assert!(camera.target.lat < 85.0 && camera.target.lat > 84.0);
        // A world shorter than the viewport zooms in to fill it.
        let tall = Viewport {
            width: 256.0,
            height: 1024.0,
        };
        let mut camera = CameraState::new(Geodetic::new(10.0, 30.0), 0.0);
        camera.constrain(&world, p, tall);
        assert!((camera.zoom() - 1.0).abs() < 1e-9);
        assert!(camera.target.lat.abs() < 1e-9);
        // A view inside the world is left alone.
        let mut camera = CameraState::new(Geodetic::new(10.0, 30.0), 6.0);
        let before = camera;
        camera.constrain(&world, p, viewport);
        assert_eq!(camera, before);
    }

    fn setup(zoom: f64, pitch: f64) -> (View, TileMatrixSet) {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let mut camera = CameraState::new(Geodetic::new(0.0, 0.0), zoom);
        camera.pitch = pitch;
        let viewport = Viewport {
            width: 1000.0,
            height: 500.0,
        };
        (View::new(&camera, viewport, tms.projection()), tms)
    }

    #[test]
    fn elevated_cover_keeps_roofs_whose_ground_is_below_the_view() {
        let (view, tms) = setup(14.0, 80.0);
        let ground = view.cover(&tms, 512.0, 14);
        assert!(
            ground.iter().any(|tile| {
                let rect = tms.tile_bounds(tile.index).translated(-view.camera_world());
                !view.intersects_frustum(&rect, 0.0, 0.0)
                    && view.intersects_frustum(&rect, 0.0, 500.0)
            }),
            "the assumed feature height retains tiles whose roofs enter the view"
        );
        assert!(view.ground_horizon() > 0.0);
        assert!(view.ground_horizon() < view.viewport().height / 2.0);
    }

    #[test]
    fn high_pitch_views_have_finite_frames_and_bounded_covers() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        for zoom in [0.0, 14.48, 24.0] {
            for pitch in [60.0, 80.0, 85.0] {
                let mut camera = CameraState::new(Geodetic::new(-74.0113, 40.7075), zoom);
                camera.pitch = pitch;
                camera.bearing = 90.0;
                let view = View::new(
                    &camera,
                    Viewport {
                        width: 1024.0,
                        height: 768.0,
                    },
                    tms.projection(),
                );
                assert!(view.view_proj().is_finite());
                assert!(view.far_px > view.eye_distance_px);
                assert!(view.ground_horizon().is_finite());
                assert!(view.sky_horizon().is_finite());
                let cover = view.cover(&tms, 512.0, 24);
                assert!(!cover.is_empty());
                assert!(
                    cover.len() < 512,
                    "z{zoom} pitch{pitch}: {} tiles",
                    cover.len()
                );
                assert!(cover.iter().all(|t| t.wrap.abs() <= 3));
                let world = view.unproject(DVec2::new(512.0, 767.0)).unwrap();
                assert!(world.is_finite());
                assert!(
                    view.project(world)
                        .unwrap()
                        .distance(DVec2::new(512.0, 767.0))
                        < 1e-4
                );
            }
        }
    }

    #[test]
    fn zoom_round_trips() {
        let mut camera = CameraState::new(Geodetic::new(10.0, 60.0), 12.3);
        assert!((camera.zoom() - 12.3).abs() < 1e-12);
        camera.set_zoom(4.0);
        assert!((camera.zoom() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn world_spans_512_px_at_zoom_0_on_the_equator() {
        let (view, tms) = setup(0.0, 0.0);
        let width = tms.projection().world_bounds().size().x;
        assert!((width / view.world_per_px() - 512.0).abs() < 1e-9);
    }

    #[test]
    fn project_and_unproject_agree() {
        let (view, _) = setup(3.5, 45.0);
        for px in [
            DVec2::new(500.0, 250.0),
            DVec2::new(10.0, 490.0),
            DVec2::new(990.0, 100.0),
        ] {
            let world = view.unproject(px).unwrap();
            let back = view.project(world).unwrap();
            assert!(back.distance(px) < 1e-6, "{px} -> {back}");
        }
        let center = view.project(view.camera_world()).unwrap();
        assert!(center.distance(DVec2::new(500.0, 250.0)) < 1e-9);
    }

    #[test]
    fn covers_at_the_floor_of_the_zoom() {
        let (view, tms) = setup(2.2, 0.0);
        let tiles = view.cover(&tms, 512.0, 22);
        // A 1000×500 viewport at zoom 2.2 on the equator sees the middle two
        // columns and rows of the 4×4 matrix.
        let mut indices: Vec<_> = tiles.iter().map(|t| (t.wrap, t.index)).collect();
        indices.sort();
        let mut expected =
            [(1, 1), (2, 1), (1, 2), (2, 2)].map(|(col, row)| (0, TileIndex::new(2, col, row)));
        expected.sort();
        assert_eq!(indices, expected);
    }

    #[test]
    fn sixty_degree_cover_keeps_the_nominal_zoom() {
        let (view, tms) = setup(10.0, 60.0);
        assert!(view.cover(&tms, 512.0, 22).iter().all(|t| t.index.z == 10));
    }

    #[test]
    fn pitched_cover_coarsens_with_distance() {
        let (view, tms) = setup(10.0, 70.0);
        let tiles = view.cover(&tms, 512.0, 22);
        let min = tiles.iter().map(|t| t.index.z).min().unwrap();
        let max = tiles.iter().map(|t| t.index.z).max().unwrap();
        assert!(max >= 10);
        assert!(min < max, "zooms {min}..={max}");
    }

    #[test]
    fn low_zoom_covers_wrapped_copies() {
        let (view, tms) = setup(0.0, 0.0);
        let tiles = view.cover(&tms, 512.0, 22);
        assert!(tiles.iter().any(|t| t.wrap == -1));
        assert!(tiles.iter().any(|t| t.wrap == 1));
    }
}
