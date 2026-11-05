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
    CoverDecision, CoveredTile, Footprint, Projection, Surface, TileIndex, TileMatrixSet,
    WorldRect, sphere,
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
pub(crate) const STYLE_ZOOM_TILE_PX: f64 = 512.0;

/// Finite horizon policy for a flat surface. Keeping a small angle below
/// horizontal bounds the far range and the LOD integral, as in GL JS.
const MAX_HORIZON_PITCH: f64 = 89.25;

impl CameraState {
    /// GL JS's default vertical field of view.
    pub const DEFAULT_FOV: f64 = 36.869_897_645_844_02;

    /// A camera looking straight down at `target` with the given style zoom
    /// on `projection`'s plane.
    pub fn new(target: Geodetic, zoom: f64, projection: &Projection) -> Self {
        let mut camera = Self {
            target,
            resolution: 1.0,
            bearing: 0.0,
            pitch: 0.0,
            fov: Self::DEFAULT_FOV,
            elevation: 0.0,
        };
        camera.set_zoom(projection, zoom);
        camera
    }

    /// The view's style zoom on `projection`'s plane (decision 0022), which
    /// v8 stops, `minzoom` and `maxzoom` evaluate at. In Web Mercator (and
    /// on the v8 globe) it is GL JS's: ground resolution at the target
    /// against the parallel's length. On other planes it is the zoom at
    /// which Web Mercator's ground resolution at the plane's reference
    /// latitude is the view's: one number for the whole view, which panning
    /// keeps.
    pub fn zoom(&self, projection: &Projection) -> f64 {
        match projection.zoom_at_scale(self.world_per_px(projection)) {
            Some(zoom) => zoom,
            None => (equator_circumference_px_per_m(self.target) / self.resolution).log2(),
        }
    }

    /// Sets the style zoom on `projection`'s plane, keeping the target.
    pub fn set_zoom(&mut self, projection: &Projection, zoom: f64) {
        self.resolution = match projection.scale_at_zoom(zoom) {
            Some(world_per_px) => world_per_px / projection.scale_factor(self.target.wrapped()),
            None => equator_circumference_px_per_m(self.target) / zoom.exp2(),
        };
    }
}

/// Style zoom's scale at zoom 0: the length of the parallel through `target`
/// divided by one tile's width.
fn equator_circumference_px_per_m(target: Geodetic) -> f64 {
    2.0 * PI * wgs84::SEMI_MAJOR_M * orb_geo::style_parallel(target.lat) / STYLE_ZOOM_TILE_PX
}

/// A viewport in logical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub width: f64,
    pub height: f64,
}

/// A view's globe: its matrices in camera-relative globe space, ECEF
/// metres (`orb_crs::sphere`) less the target's position.
#[derive(Clone, Debug)]
pub struct GlobeView {
    /// How much of a globe the view is, in (0, 1].
    pub amount: f64,
    /// The target's position on (or above) the sphere: the origin of
    /// camera-relative globe space.
    pub target: DVec3,
    /// Camera-relative globe space → clip space.
    pub view_proj: DMat4,
    /// The eye, camera-relative.
    pub eye: DVec3,
    /// Metres per logical pixel at the target.
    pub metres_per_px: f64,
}

impl GlobeView {
    /// Whether a camera-relative point on the sphere faces the eye.
    pub fn faces_eye(&self, p: DVec3) -> bool {
        let eye = self.target + self.eye;
        (p + self.target).dot(eye) >= sphere::RADIUS * sphere::RADIUS
    }

    /// The angle from the point under the eye to the horizon, in radians.
    pub fn horizon_angle(&self) -> f64 {
        (sphere::RADIUS / (self.target + self.eye).length())
            .min(1.0)
            .acos()
    }
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
    projection: Projection,
    globe: Option<GlobeView>,
}

impl View {
    pub fn new(camera: &CameraState, viewport: Viewport, surface: &Surface) -> Self {
        Self::on_ground(camera, viewport, surface, camera.elevation)
    }

    /// A view over ground whose lowest elevation near the target is
    /// `lowest` metres (GL JS's `minElevationForCurrentTile`, from the
    /// terrain's DEM tile under the target): the far plane reaches down to
    /// it.
    pub fn on_ground(
        camera: &CameraState,
        viewport: Viewport,
        surface: &Surface,
        lowest: f64,
    ) -> Self {
        let projection = &surface.projection;
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

        let amount = surface.globe_amount(camera.zoom(projection));
        let globe = (amount > 0.0).then(|| {
            // The plane's view, turned from the target's east, north and up
            // to the sphere's axes, in metres.
            let local = DMat4::from_mat3(sphere::frame(target).transpose());
            let view = DMat4::from_translation(DVec3::new(0.0, 0.0, -eye_distance_px))
                * DMat4::from_rotation_x(-pitch)
                * DMat4::from_rotation_z(camera.bearing.to_radians())
                * DMat4::from_scale(DVec3::splat(1.0 / camera.resolution))
                * local;
            let radius_px = sphere::RADIUS / camera.resolution;
            let proj = glam::dcamera::rh::proj::directx::perspective(
                fov,
                viewport.width / viewport.height,
                near_px,
                eye_distance_px + 2.0 * radius_px,
            );
            GlobeView {
                amount,
                target: sphere::ecef(target, camera.elevation),
                view_proj: proj * view,
                eye: view.inverse().transform_point3(DVec3::ZERO),
                metres_per_px: camera.resolution,
            }
        });

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
            projection: *projection,
            globe,
        }
    }

    /// The view's globe, when it is one at its zoom.
    pub fn globe(&self) -> Option<&GlobeView> {
        self.globe.as_ref()
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

    /// Projects a world position `z` world units above the datum. On a
    /// globe, positions beyond the horizon are `None` too.
    pub fn project_3d(&self, world: DVec2, z: f64) -> Option<DVec2> {
        let clip = match &self.globe {
            Some(globe) => {
                let g = self.projection.inverse(world);
                let p = sphere::ecef(g, z / self.height_scale) - globe.target;
                if !globe.faces_eye(sphere::ecef(g, 0.0) - globe.target) {
                    return None;
                }
                globe.view_proj * p.extend(1.0)
            }
            None => self.view_proj * (world - self.camera_world).extend(z).extend(1.0),
        };
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
        self.unproject_at(px, self.target_z)
    }

    /// The world position under a point in logical pixels on the level
    /// plane `z` world units above the datum, or `None` above its horizon.
    pub fn unproject_at(&self, px: DVec2, z: f64) -> Option<DVec2> {
        if self.globe.is_some() {
            let world = self
                .projection
                .forward(sphere::geodetic(self.globe_ground(px, z)?));
            // The copy of a wrapping world nearest the target.
            return Some(match self.wrap_period {
                Some(period) => DVec2::new(
                    world.x - ((world.x - self.camera_world.x) / period).round() * period,
                    world.y,
                ),
                None => world,
            });
        }
        let (near, far) = self.ray(px, &self.view_proj);
        let t = (near.z - z) / (near.z - far.z);
        (far.z < near.z && t >= 0.0)
            .then(|| (near + (far - near) * t).truncate() + self.camera_world)
    }

    /// On a globe, the ECEF position under a point in logical pixels on the
    /// sphere `z` world units above it: the view ray's nearest meeting with
    /// it, or `None` where the ray misses.
    pub(crate) fn globe_ground(&self, px: DVec2, z: f64) -> Option<DVec3> {
        let globe = self.globe.as_ref()?;
        let (near, far) = self.ray(px, &globe.view_proj);
        let (origin, dir) = (near + globe.target, (far - near).normalize());
        let radius = sphere::RADIUS + z / self.height_scale;
        let b = origin.dot(dir);
        let disc = b * b - (origin.length_squared() - radius * radius);
        if disc < 0.0 {
            return None;
        }
        let t = -b - disc.sqrt();
        (t >= 0.0).then(|| origin + dir * t)
    }

    /// The near and far ends of the view ray through a point in logical
    /// pixels, in camera-relative world space.
    fn ray(&self, px: DVec2, view_proj: &DMat4) -> (DVec3, DVec3) {
        let ndc = DVec2::new(
            px.x / self.viewport.width * 2.0 - 1.0,
            1.0 - px.y / self.viewport.height * 2.0,
        );
        let inverse = view_proj.inverse();
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

    /// The eye's distance from the target, logical pixels (GL JS's
    /// `cameraToCenterDistance`).
    pub fn eye_distance_px(&self) -> f64 {
        self.eye_distance_px
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
        if let Some(globe) = &self.globe {
            return Self::cover_globe(globe, tms, tile_px, max_zoom);
        }
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
        // Any set's tiles wrap the globe directly.
        if let Some(globe) = &self.globe {
            return Self::cover_globe(globe, tms, tile_px, max_zoom);
        }
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

    /// The tiles of `tms` that cover a globe view: those on the eye's side
    /// of the horizon and in the frustum, each at the zoom whose tiles span
    /// `tile_px` pixels at its distance (so tiles coarsen toward the
    /// horizon and, being smaller on the sphere, toward the poles).
    fn cover_globe(
        globe: &GlobeView,
        tms: &TileMatrixSet,
        tile_px: f64,
        max_zoom: u8,
    ) -> Vec<CoveredTile> {
        let eye = globe.target + globe.eye;
        let eye_dir = eye.normalize();
        let horizon = globe.horizon_angle();
        let scale = globe.metres_per_px * tile_px;
        let center_distance = globe.eye.length();
        let projection = tms.projection();
        tms.cover_with(0..=0, None, max_zoom, |index, _| {
            let b = tms.tile_bounds(index);
            // A grid of the tile's points on the sphere, row by row: 5×5
            // for the widest tiles, 3×3 below.
            let steps = if index.z < 3 { 4 } else { 2 };
            let n = steps + 1;
            let mut points = [DVec3::NAN; 25];
            for (k, point) in points.iter_mut().enumerate().take(n * n) {
                let uv = DVec2::new((k % n) as f64, (k / n) as f64) / steps as f64;
                let world = DVec2::new(b.min.x + uv.x * b.size().x, b.max.y - uv.y * b.size().y);
                let g = projection.inverse(world);
                if g.lon.is_finite() && g.lat.is_finite() {
                    *point = sphere::ecef(g, 0.0);
                }
            }
            let points = &points[..n * n];
            let center = points[points.len() / 2];
            if !center.is_finite() {
                return CoverDecision::Skip;
            }
            // Its cap: the widest angle from its centre to a point, plus
            // the widest gap between neighbouring points.
            let finite = || points.iter().copied().filter(|p| p.is_finite());
            let pair = |a: usize, b: usize| {
                let (a, b) = (points[a], points[b]);
                (a.is_finite() && b.is_finite()).then(|| a.angle_between(b))
            };
            let mut gap: f64 = 0.0;
            for k in 0..points.len() {
                if k % n != steps {
                    gap = gap.max(pair(k, k + 1).unwrap_or(0.0));
                }
                if k + n < points.len() {
                    gap = gap.max(pair(k, k + n).unwrap_or(0.0));
                }
            }
            let radius = finite()
                .map(|p| center.angle_between(p))
                .fold(0.0, f64::max)
                + gap;
            if center.angle_between(eye_dir) - radius > horizon {
                return CoverDecision::Skip;
            }
            // Its box, bulged by the sagitta between points.
            let bulge = sphere::RADIUS * (1.0 - (gap / 2.0).min(FRAC_PI_2).cos());
            let (mut lo, mut hi) = (DVec3::INFINITY, DVec3::NEG_INFINITY);
            let mut nearest = f64::INFINITY;
            for p in finite() {
                lo = lo.min(p - globe.target);
                hi = hi.max(p - globe.target);
                nearest = nearest.min(p.distance(eye));
            }
            if !box_in_frustum(&globe.view_proj, lo - bulge, hi + bulge) {
                return CoverDecision::Skip;
            }
            // Its side at the centre, along its middle row and column.
            let mid = steps / 2;
            let length = |at: &dyn Fn(usize) -> usize| {
                (0..steps)
                    .map(|i| (points[at(i)], points[at(i + 1)]))
                    .filter(|(a, b)| a.is_finite() && b.is_finite())
                    .map(|(a, b)| a.distance(b))
                    .sum::<f64>()
            };
            let side = length(&|i| mid * n + i).max(length(&|j| j * n + mid));
            // The nearest point lies within half a gap of a sample.
            let distance = (nearest - sphere::RADIUS * gap / 2.0).max(center_distance * 1e-3);
            let desired =
                f64::from(index.z) + (side / scale).log2() + (center_distance / distance).log2();
            if f64::from(index.z) < desired.floor().max(0.0) {
                CoverDecision::Refine
            } else {
                CoverDecision::Accept
            }
        })
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
        box_in_frustum(&self.view_proj, rect.min.extend(lo), rect.max.extend(hi))
    }
}

/// Whether a box may meet the frustum of `view_proj`: no clip plane has
/// all eight corners outside.
fn box_in_frustum(view_proj: &DMat4, lo: DVec3, hi: DVec3) -> bool {
    let corners: [_; 8] = std::array::from_fn(|i| {
        let x = if i & 1 == 0 { lo.x } else { hi.x };
        let y = if i & 2 == 0 { lo.y } else { hi.y };
        let z = if i & 4 == 0 { lo.z } else { hi.z };
        *view_proj * DVec4::new(x, y, z, 1.0)
    });
    let all = |f: &dyn Fn(&DVec4) -> bool| corners.iter().all(f);
    !(all(&|c| c.x > c.w)
        || all(&|c| c.x < -c.w)
        || all(&|c| c.y > c.w)
        || all(&|c| c.y < -c.w)
        || all(&|c| c.z > c.w)
        || all(&|c| c.z < 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_crs::Globe;

    fn mercator() -> Projection {
        *TileMatrixSet::by_id("WebMercatorQuad")
            .unwrap()
            .projection()
    }
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
        let mut camera = CameraState::new(Geodetic::new(0.0, 85.0), 4.0, &mercator());
        camera.constrain(&world, &Surface::from(p), viewport);
        assert!((camera.zoom(&mercator()) - 4.0).abs() < 1e-9);
        assert!(camera.target.lat < 85.0 && camera.target.lat > 84.0);
        // A world shorter than the viewport zooms in to fill it.
        let tall = Viewport {
            width: 256.0,
            height: 1024.0,
        };
        let mut camera = CameraState::new(Geodetic::new(10.0, 30.0), 0.0, &mercator());
        camera.constrain(&world, &Surface::from(p), tall);
        assert!((camera.zoom(&mercator()) - 1.0).abs() < 1e-9);
        assert!(camera.target.lat.abs() < 1e-9);
        // A view inside the world is left alone.
        let mut camera = CameraState::new(Geodetic::new(10.0, 30.0), 6.0, &mercator());
        let before = camera;
        camera.constrain(&world, &Surface::from(p), viewport);
        assert_eq!(camera, before);
    }

    fn setup(zoom: f64, pitch: f64) -> (View, TileMatrixSet) {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let mut camera = CameraState::new(Geodetic::new(0.0, 0.0), zoom, &mercator());
        camera.pitch = pitch;
        let viewport = Viewport {
            width: 1000.0,
            height: 500.0,
        };
        (
            View::new(&camera, viewport, &Surface::from(tms.projection())),
            tms,
        )
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
                let mut camera =
                    CameraState::new(Geodetic::new(-74.0113, 40.7075), zoom, &mercator());
                camera.pitch = pitch;
                camera.bearing = 90.0;
                let view = View::new(
                    &camera,
                    Viewport {
                        width: 1024.0,
                        height: 768.0,
                    },
                    &Surface::from(tms.projection()),
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

    /// Web Mercator's style zoom is GL JS's, bit for bit: ground resolution
    /// at the target against the parallel's length.
    #[test]
    fn web_mercator_zoom_is_unchanged() {
        for (lat, zoom) in [(0.0, 3.0), (35.68, 15.2), (-60.0, 7.7), (84.9, 1.1)] {
            let target = Geodetic::new(12.0, lat);
            let camera = CameraState::new(target, zoom, &mercator());
            let parallel = 2.0 * PI * wgs84::SEMI_MAJOR_M * f64::to_radians(lat).cos() / 512.0;
            assert_eq!(camera.resolution, parallel / zoom.exp2());
            assert_eq!(
                camera.zoom(&mercator()),
                (parallel / camera.resolution).log2()
            );
        }
    }

    /// On other planes style zoom is the view's scale against Web Mercator's
    /// at the plane's reference latitude: NZTM2000's natural origin on the
    /// equator, EPSG:3413's true-scale latitude of 70°N.
    #[test]
    fn plane_zoom_is_measured_at_the_reference() {
        let nztm = Projection::from_descriptor("EPSG:2193").unwrap();
        let camera = CameraState::new(Geodetic::new(174.8, -41.3), 10.0, &nztm);
        let metres_per_px = camera.world_per_px(&nztm) / 0.9996;
        let equator = 2.0 * PI * wgs84::SEMI_MAJOR_M / 512.0 / 1024.0;
        assert!((metres_per_px / equator - 1.0).abs() < 1e-6);
        let polar = Projection::from_descriptor("EPSG:3413").unwrap();
        assert_eq!(polar.zoom_reference().unwrap().lat.round(), 70.0);
        // The same scale anywhere on the plane is the same zoom.
        let a = CameraState::new(Geodetic::new(-45.0, 60.0), 5.0, &polar);
        let mut b = a;
        b.set_target_world(
            &polar,
            polar.forward(Geodetic::new(100.0, 89.0)),
            a.world_per_px(&polar),
        );
        assert!((b.zoom(&polar) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn zoom_round_trips() {
        let mut camera = CameraState::new(Geodetic::new(10.0, 60.0), 12.3, &mercator());
        assert!((camera.zoom(&mercator()) - 12.3).abs() < 1e-12);
        camera.set_zoom(&mercator(), 4.0);
        assert!((camera.zoom(&mercator()) - 4.0).abs() < 1e-12);
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

    fn globe_view(target: Geodetic, zoom: f64, pitch: f64, bearing: f64) -> View {
        let mut camera = CameraState::new(target, zoom, &mercator());
        camera.pitch = pitch;
        camera.bearing = bearing;
        let surface = Surface {
            projection: mercator(),
            globe: Some(Globe::V8),
        };
        View::new(
            &camera,
            Viewport {
                width: 1000.0,
                height: 700.0,
            },
            &surface,
        )
    }

    #[test]
    fn globe_hands_over_to_the_plane() {
        assert_eq!(Globe::V8.amount(3.0), 1.0);
        assert_eq!(Globe::V8.amount(11.5), 0.5);
        assert_eq!(Globe::V8.amount(12.0), 0.0);
        assert!(
            globe_view(Geodetic::new(0.0, 0.0), 12.0, 0.0, 0.0)
                .globe()
                .is_none()
        );
        assert!(
            globe_view(Geodetic::new(0.0, 0.0), 11.9, 0.0, 0.0)
                .globe()
                .is_some()
        );
    }

    #[test]
    fn globe_projects_and_unprojects() {
        for (target, zoom, pitch, bearing) in [
            (Geodetic::new(0.0, 0.0), 1.0, 0.0, 0.0),
            (Geodetic::new(139.7, 35.7), 3.3, 40.0, 30.0),
            (Geodetic::new(-170.0, -60.0), 6.0, 60.0, -100.0),
            (Geodetic::new(20.0, 84.0), 2.0, 20.0, 180.0),
        ] {
            let view = globe_view(target, zoom, pitch, bearing);
            let center = DVec2::new(500.0, 350.0);
            let p = view.project(view.camera_world()).unwrap();
            assert!(p.distance(center) < 1e-6, "{target:?}: {p}");
            // Points on the globe: within half its radius of the centre.
            let radius_px = sphere::RADIUS / view.globe().unwrap().metres_per_px;
            let reach = (radius_px / 2.0).min(300.0);
            for offset in [
                DVec2::new(0.0, 1.0),
                DVec2::new(-1.0, 0.5),
                DVec2::new(0.9, 0.1),
            ] {
                let px = center + offset * reach;
                let world = view.unproject(px).unwrap();
                // Past ±85.0511°, Web Mercator holds no position to return.
                if world.y.abs() >= mercator().world_bounds().max.y {
                    continue;
                }
                let back = view.project(world).unwrap();
                assert!(back.distance(px) < 1e-5, "{target:?} {px}: {back}");
            }
            // The far side of the globe is hidden.
            let antipode = Geodetic::new(target.lon + 180.0, -target.lat);
            assert!(view.project(mercator().forward(antipode)).is_none());
        }
        // Space around a small globe unprojects to nothing.
        let view = globe_view(Geodetic::new(0.0, 0.0), 0.5, 0.0, 0.0);
        assert!(view.unproject(DVec2::new(5.0, 5.0)).is_none());
    }

    /// At the target, the globe shows the plane's scale, as GL JS's does.
    #[test]
    fn globe_keeps_the_planes_scale_at_the_target() {
        for (lat, zoom) in [(0.0, 4.0), (60.0, 7.0), (-35.0, 10.5)] {
            let target = Geodetic::new(30.0, lat);
            let globe = globe_view(target, zoom, 0.0, 0.0);
            let camera = CameraState::new(target, zoom, &mercator());
            let plane = View::new(&camera, globe.viewport(), &Surface::plane(mercator()));
            let step = DVec2::new(1.0, 1.0) * plane.world_per_px() * 20.0;
            let world = plane.camera_world() + step;
            let (a, b) = (plane.project(world).unwrap(), globe.project(world).unwrap());
            assert!(a.distance(b) < 0.05, "lat {lat}: {a} vs {b}");
        }
    }

    #[test]
    fn globe_covers_every_visible_point() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        for (target, zoom, pitch) in [
            (Geodetic::new(0.0, 0.0), 0.5, 0.0),
            (Geodetic::new(10.0, 50.0), 2.5, 50.0),
            (Geodetic::new(-120.0, 75.0), 4.0, 70.0),
            (Geodetic::new(150.0, -30.0), 8.0, 0.0),
            (Geodetic::new(5.0, 5.0), 11.5, 60.0),
        ] {
            let view = globe_view(target, zoom, pitch, 25.0);
            let cover = view.cover(&tms, 512.0, 22);
            assert!(
                !cover.is_empty() && cover.len() < 400,
                "{} tiles",
                cover.len()
            );
            let mut seen = 0;
            for i in 0..=40 {
                for j in 0..=40 {
                    let px = DVec2::new(f64::from(i) * 25.0, f64::from(j) * 17.5);
                    let Some(world) = view.unproject(px) else {
                        continue;
                    };
                    seen += 1;
                    let period = tms.tile_span(0);
                    let x = world.x
                        - ((world.x - tms.tile_bounds(TileIndex::new(0, 0, 0)).center().x)
                            / period)
                            .round()
                            * period;
                    let world = DVec2::new(x, world.y);
                    assert!(
                        cover.iter().any(|t| {
                            let b = tms.tile_bounds(t.index);
                            b.min.cmple(world).all() && world.cmple(b.max).all()
                        }) || world.y.abs() >= tms.projection().world_bounds().max.y,
                        "z{zoom} {px}: no tile"
                    );
                }
            }
            assert!(seen > 50, "{seen} points on the globe");
        }
    }
}
