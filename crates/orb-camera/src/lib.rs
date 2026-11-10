//! Camera state and the view transforms derived from it.
//!
//! [`CameraState`] is the single source of truth: a target, a ground
//! resolution and an orientation. A [`View`] combines it with a viewport and a
//! projection into the matrices and tile cover a frame needs. World positions
//! handed to the GPU are relative to the camera's world position so they stay
//! precise in f32 at any zoom.
//!
//! The [`input`] module turns pointer, touch, wheel and keyboard input into
//! moves of the [`Camera`] store.

#![forbid(unsafe_code)]

mod animate;
mod fit;
mod free;
pub mod input;
mod lod;
mod momentum;
mod moves;
mod store;
mod terrain;

pub use animate::{CameraTarget, EaseOptions, Easing, FlyOptions};
pub use fit::{FitOptions, Padding};
pub use free::FreeCamera;
pub use lod::{LOD_EASE, TileLod, TileLodMode};
pub use momentum::{
    AnchoredMomentum, DecayModel, GestureMomentum, MomentumDecay, MomentumSnap, MomentumVelocity,
    PanMomentum,
};
pub use moves::{Bounds, Constraints};
pub use store::{AnimationId, Camera, CameraEvent, Change, Phase};
pub use terrain::Elevation;

use std::f64::consts::{FRAC_PI_2, PI};

use glam::{DMat3, DMat4, DVec2, DVec3, DVec4};
use orb_crs::{
    CoverDecision, CoveredTile, Footprint, Projection, Surface, TileIndex, TileMatrixSet,
    WorldRect, sphere,
};
use orb_crs::{Geodetic, wgs84};

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
    /// Rotation about the view axis, in degrees: GL JS's roll, turning the
    /// picture counter-clockwise.
    pub roll: f64,
    /// Vertical field of view, in degrees.
    pub fov: f64,
    /// The target's height above the surface in metres (GL JS's center
    /// elevation): the camera orbits a point this high. On terrain it is
    /// the ground's elevation under the target.
    pub elevation: f64,
    /// Space the map keeps clear at the viewport's edges (GL JS's
    /// `padding`): the target and the vanishing point sit at the middle
    /// of what is left.
    pub padding: Padding,
}

/// Size of one tile side at its own zoom in the style zoom convention, in
/// logical pixels.
pub(crate) const STYLE_ZOOM_TILE_PX: f64 = 512.0;

/// Finite horizon policy for a flat surface. Keeping a small angle below
/// horizontal bounds the far range and the LOD integral, as in GL JS.
const MAX_HORIZON_PITCH: f64 = 89.25;

/// How far up toward a flat surface's true horizon the ground is drawn, as
/// a share of the target's distance below it on screen (MapLibre Native's
/// bound on the far plane): the farthest ground lies about 100 times as
/// deep as the target, and the sky starts there.
const HORIZON_SHARE: f64 = 0.99;

/// The sine of the lowest pitch the sky's horizon is placed for, which
/// keeps it finite looking straight down.
fn lowest_pitch_sin() -> f64 {
    (90.0 - MAX_HORIZON_PITCH).to_radians().sin()
}

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
            roll: 0.0,
            fov: Self::DEFAULT_FOV,
            elevation: 0.0,
            padding: Padding::default(),
        };
        camera.set_zoom(projection, zoom);
        camera
    }

    /// Where the target shows in `viewport`: the middle of its padded
    /// area, in logical pixels.
    pub fn center(&self, viewport: Viewport) -> DVec2 {
        let p = self.padding;
        DVec2::new(
            (viewport.width + p.left - p.right) / 2.0,
            (viewport.height + p.top - p.bottom) / 2.0,
        )
    }

    /// The view's style zoom on `projection`'s plane, which
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
    2.0 * PI * wgs84::SEMI_MAJOR_M * orb_crs::style_parallel(target.lat) / STYLE_ZOOM_TILE_PX
}

/// A viewport in logical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub width: f64,
    pub height: f64,
}

/// A view's low-zoom surface ([`orb_crs::Globe`]): its matrices in
/// camera-relative globe space, ECEF metres (`orb_crs::sphere`) less the
/// target's position, or on a low-zoom plane that plane's world units (z
/// up) less the target's.
#[derive(Clone, Debug)]
pub struct GlobeView {
    /// How much of the low-zoom surface the view is, in (0, 1].
    pub amount: f64,
    /// The target's position on (or above) the surface: the origin of
    /// camera-relative globe space.
    pub target: DVec3,
    /// Camera-relative globe space → clip space.
    pub view_proj: DMat4,
    /// The eye, camera-relative.
    pub eye: DVec3,
    /// Metres (on a plane, its world units) per logical pixel at the
    /// target.
    pub metres_per_px: f64,
    /// On a low-zoom plane, the camera's view of it.
    pub plane: Option<Box<View>>,
}

impl GlobeView {
    /// Whether a camera-relative point on the surface faces the eye: on the
    /// sphere, in front of its horizon; on a plane, always.
    pub(crate) fn faces_eye(&self, p: DVec3) -> bool {
        if self.plane.is_some() {
            return true;
        }
        let eye = self.target + self.eye;
        (p + self.target).dot(eye) >= sphere::RADIUS * sphere::RADIUS
    }

    /// The angle from the point under the eye to the horizon, in radians.
    pub(crate) fn horizon_angle(&self) -> f64 {
        (sphere::RADIUS / (self.target + self.eye).length())
            .min(1.0)
            .acos()
    }
}

/// A camera resolved against a viewport and a projection for one frame.
#[derive(Clone, Debug)]
pub struct View {
    viewport: Viewport,
    /// Where the target shows, logical pixels: the padded area's middle.
    center: DVec2,
    pitch: f64,
    /// Degrees about the view axis.
    roll: f64,
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
    /// How far above the target's place the ground ends and the sky
    /// starts, logical pixels, without roll.
    horizon_px: f64,
    height_scale: f64,
    projection: Projection,
    globe: Option<GlobeView>,
    /// How its covers' tiles coarsen with distance.
    lod: TileLod,
}

impl View {
    pub fn new(camera: &CameraState, viewport: Viewport, surface: &Surface) -> Self {
        Self::on_ground(camera, viewport, surface, camera.elevation)
    }

    /// A view over ground whose lowest elevation near the target is
    /// `lowest` metres (GL JS's `minElevationForCurrentTile`, from the
    /// terrain's DEM tile under the target): the far plane reaches down to
    /// it.
    pub(crate) fn on_ground(
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

        // MapLibre Native's far plane (`TransformState::getProjMatrix`): the
        // depth of the ground point at the top edge, out to where the ground
        // shows `HORIZON_SHARE` of the way up to the true horizon. GL JS
        // stops at 85% of the angle to it and fills the rest with sky; the
        // ground is drawn on instead, and the sky starts where it ends.
        // The view's highest corner above its axis, in pixels at the
        // target's depth: padding moves the target off the middle, and a
        // rolled view's vertical extent spans some of its width (Native's
        // `tanFovAboveCenter`, here exact for any padding and roll).
        let (roll_sin, roll_cos) = camera.roll.to_radians().sin_cos();
        let center = camera.center(viewport);
        let offset = center - DVec2::new(viewport.width, viewport.height) / 2.0;
        let above_center = viewport.height / 2.0 * roll_cos.abs()
            + offset.y * roll_cos
            + viewport.width / 2.0 * roll_sin.abs()
            + offset.x * roll_sin;
        let top = (above_center / eye_distance_px * limited.tan()).clamp(0.0, HORIZON_SHARE);
        // The deepest ground drawn.
        let mut deepest = lowest_plane_px / (1.0 - HORIZON_SHARE);
        let altitude_px = altitude * px_per_m;
        // A surface that is a sphere at low zoom ends its plane's ground
        // at the sphere's horizon, as far along the ground as the eye sees
        // over the sphere, so the two meet through the handover and the
        // ground reaches no farther past it.
        if surface.globe.is_some_and(|g| g.plane.is_none()) {
            let altitude = altitude.max(1.0);
            let r = sphere::RADIUS;
            let ahead = ((r + altitude).powi(2) - r * r).sqrt() * px_per_m;
            let limb = ahead * pitch.sin() + altitude * px_per_m * pitch.cos();
            deepest = deepest.min(limb.max(lowest_plane_px));
        }
        let far_px = (lowest_plane_px / (1.0 - top)).min(deepest) * 1.01;
        // Where the ground ends on screen and the sky starts, above the
        // target's place: the ground point that deep below the eye.
        let horizon_px = eye_distance_px * (pitch.cos() - altitude_px / deepest)
            / pitch.sin().max(lowest_pitch_sin());
        let near_px = viewport.height / 50.0;

        let to_px = DMat4::from_scale(DVec3::splat(1.0 / world_per_px));
        let roll = DMat4::from_rotation_z(camera.roll.to_radians());
        let view = DMat4::from_translation(DVec3::new(0.0, 0.0, -eye_distance_px))
            * roll
            * DMat4::from_rotation_x(-pitch)
            * DMat4::from_rotation_z(camera.bearing.to_radians())
            * to_px
            * DMat4::from_translation(DVec3::new(0.0, 0.0, -target_z));
        // The eye looks at the target; the picture shifts it to the padded
        // area's middle (GL JS's `centerOffset`).
        let shift = DMat4::from_translation(DVec3::new(
            2.0 * offset.x / viewport.width,
            -2.0 * offset.y / viewport.height,
            0.0,
        ));
        let proj = shift
            * glam::dcamera::rh::proj::directx::perspective(
                fov,
                viewport.width / viewport.height,
                near_px,
                far_px,
            );
        let eye = view.inverse().transform_point3(DVec3::ZERO);

        let view_proj = proj * view;

        let amount = surface.globe_amount(camera.zoom(projection));
        let low_plane = surface.globe.and_then(|g| g.plane);
        let globe = (amount > 0.0).then(|| {
            if let Some(plane) = low_plane {
                let low = Self::on_ground(camera, viewport, &Surface::plane(plane), lowest);
                return GlobeView {
                    amount,
                    target: low.camera_world.extend(0.0),
                    view_proj: low.view_proj,
                    eye: low.eye,
                    metres_per_px: low.world_per_px,
                    plane: Some(Box::new(low)),
                };
            }
            // The plane's view, turned from the target's east, north and up
            // to the sphere's axes, in metres.
            let local = DMat4::from_mat3(sphere::frame(target).transpose());
            let view = DMat4::from_translation(DVec3::new(0.0, 0.0, -eye_distance_px))
                * roll
                * DMat4::from_rotation_x(-pitch)
                * DMat4::from_rotation_z(camera.bearing.to_radians())
                * DMat4::from_scale(DVec3::splat(1.0 / camera.resolution))
                * local;
            let radius_px = sphere::RADIUS / camera.resolution;
            let proj = shift
                * glam::dcamera::rh::proj::directx::perspective(
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
                plane: None,
            }
        });

        Self {
            viewport,
            center,
            pitch: camera.pitch,
            roll: camera.roll,
            fov: camera.fov,
            camera_world,
            world_per_px,
            view_proj,
            eye,
            target_z,
            sea_level_px,
            eye_distance_px,
            far_px,
            horizon_px,
            height_scale,
            projection: *projection,
            globe,
            lod: TileLod::default(),
        }
    }

    /// This view with its covers' tiles coarsening by `lod`
    /// ([`TileLod::default`] until set), its low-zoom surface's too.
    pub fn with_tile_lod(mut self, lod: TileLod) -> Self {
        self.lod = lod.sanitized();
        if let Some(low) = self.globe.as_mut().and_then(|g| g.plane.as_mut()) {
            low.lod = self.lod;
        }
        self
    }

    /// How its covers' tiles coarsen with distance.
    pub fn tile_lod(&self) -> TileLod {
        self.lod
    }

    /// The view's low-zoom surface, when it shows at its zoom.
    pub fn globe(&self) -> Option<&GlobeView> {
        self.globe.as_ref()
    }

    /// The camera's view of the low-zoom plane, when one shows at its zoom:
    /// what the view's cover, picking and labels see.
    pub fn low_plane(&self) -> Option<&View> {
        self.globe.as_ref()?.plane.as_deref()
    }

    /// The projection the view's world positions are on.
    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    /// A world position's place on the low-zoom plane, and a height on it.
    fn to_low(&self, low: &View, world: DVec2, z: f64) -> (DVec2, f64) {
        let g = self.projection.inverse(world).wrapped();
        (
            low.projection.forward(g),
            z / self.height_scale * low.height_scale,
        )
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
        ((1.0 - ndc_y) * 0.5 * self.viewport.height + self.center.y - self.viewport.height / 2.0)
            .max(0.0)
    }

    /// Top logical row of the apparent horizon without roll, for tests.
    #[cfg(test)]
    pub(crate) fn sky_horizon(&self) -> f64 {
        self.center.y - self.horizon_px
    }

    /// The apparent horizon as a line: a point on it and its unit normal
    /// toward the sky, logical pixels with y down. Roll turns it about the
    /// target's place (GL JS's `u_horizon`, `u_horizon_normal`).
    pub fn sky_horizon_line(&self) -> (DVec2, DVec2) {
        let (sin, cos) = self.roll.to_radians().sin_cos();
        let h = self.horizon_px;
        (
            DVec2::new(self.center.x - h * sin, self.center.y - h * cos),
            DVec2::new(-sin, -cos),
        )
    }

    pub fn viewport(&self) -> Viewport {
        self.viewport
    }

    /// Where the target shows, logical pixels: the padded area's middle.
    pub fn center(&self) -> DVec2 {
        self.center
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
        self.project_depth(world, z).map(|(px, _)| px)
    }

    /// [`project_3d`](Self::project_3d) with the position's depth: its clip
    /// z over w, 0 at the near plane and 1 at the far one.
    pub fn project_depth(&self, world: DVec2, z: f64) -> Option<(DVec2, f64)> {
        if let Some(low) = self.low_plane() {
            let (world, z) = self.to_low(low, world, z);
            return low.project_depth(world, z);
        }
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
            let ndc = clip.truncate() / clip.w;
            let px = DVec2::new(
                (ndc.x + 1.0) * 0.5 * self.viewport.width,
                (1.0 - ndc.y) * 0.5 * self.viewport.height,
            );
            (px, ndc.z)
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
            let world = self.projection.forward(self.low_ground(px, z)?);
            // The copy of a wrapping world nearest the target.
            return Some(self.projection.nearest_copy(world, self.camera_world));
        }
        let (near, far) = self.ray(px, &self.view_proj);
        let t = (near.z - z) / (near.z - far.z);
        (far.z < near.z && t >= 0.0)
            .then(|| (near + (far - near) * t).truncate() + self.camera_world)
    }

    /// On a low-zoom surface, the point under a point in logical pixels on
    /// it, `z` world units up, or `None` where the view ray misses it.
    pub(crate) fn low_ground(&self, px: DVec2, z: f64) -> Option<Geodetic> {
        match self.low_plane() {
            Some(low) => {
                let z = z / self.height_scale * low.height_scale;
                Some(low.projection.inverse(low.unproject_at(px, z)?))
            }
            None => self.globe_ground(px, z).map(sphere::geodetic),
        }
    }

    /// On a globe, the ECEF position under a point in logical pixels on the
    /// sphere `z` world units above it: the view ray's nearest meeting with
    /// it, or `None` where the ray misses.
    pub(crate) fn globe_ground(&self, px: DVec2, z: f64) -> Option<DVec3> {
        let globe = self.globe.as_ref().filter(|g| g.plane.is_none())?;
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
    pub(crate) fn target_z(&self) -> f64 {
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

    /// The target's clip w: how far it lies in front of the eye along the
    /// view, in clip units (GL JS's `cameraToCenterDistance`, which sizes
    /// labels by their distance against it). On terrain the target stands
    /// above the datum, the origin of camera-relative space.
    pub fn camera_to_center(&self) -> f64 {
        (self.view_proj * DVec4::new(0.0, 0.0, self.target_z, 1.0)).w
    }

    /// The eye's position in camera-relative world space.
    pub fn eye(&self) -> DVec3 {
        self.eye
    }

    /// The tiles of `tms` that cover the view: of the zoom whose tiles span
    /// `tile_px` pixels at the target, coarsening with distance past the
    /// [`TileLod`]'s pitch threshold.
    pub fn cover(&self, tms: &TileMatrixSet, tile_px: f64, max_zoom: u8) -> Vec<CoveredTile> {
        self.cover_volume(tms, tile_px, max_zoom, 0.0, None, 0.0)
    }

    /// Covers ground and content standing on it up to `height` (world
    /// units). As the bottom edge approaches the horizon the culling volume
    /// grows toward that height (at least 500 style metres), retaining
    /// tiles whose roofs can enter the view above their ground.
    /// On `terrain`, tiles are culled by the elevations under them where
    /// known, and take the terrain's level of detail
    /// ([`cover_terrain`](Self::cover_terrain)): draped content shows no
    /// more detail than its terrain tile's drape slot holds.
    pub fn cover_volume(
        &self,
        tms: &TileMatrixSet,
        tile_px: f64,
        max_zoom: u8,
        height: f64,
        terrain: Option<&dyn Elevation>,
        margin_px: f64,
    ) -> Vec<CoveredTile> {
        let detail = match terrain {
            Some(_) => Detail::Terrain,
            None => Detail::Lod(self.lod),
        };
        self.cover_detailed(tms, (tile_px, max_zoom), height, terrain, margin_px, detail)
    }

    /// The terrain's own cover: its tiles of `tms`, culled by the
    /// elevations under them where known, coarsening with distance at any
    /// pitch by GL JS's terrain level of detail. Each terrain tile takes a
    /// drape slot per stack, so the terrain keeps GL JS's fewer, coarser
    /// far tiles rather than the map's [`TileLod`]. Native draws no
    /// terrain; its default detail wants 43 to 63 terrain tiles at pitches
    /// 53 to 85 over the Alps and Japan where GL JS's wants 26 to 31, past
    /// the 32 the drape atlas holds with a viewport-lit hillshade. And
    /// where the ground falls away from the target, as straight down onto a
    /// crest, one zoom across the view (Native's below its pitch
    /// threshold) covers far more ground than the view shows.
    pub fn cover_terrain(
        &self,
        tms: &TileMatrixSet,
        tile_px: f64,
        max_zoom: u8,
        terrain: &dyn Elevation,
    ) -> Vec<CoveredTile> {
        let detail = Detail::Terrain;
        self.cover_detailed(tms, (tile_px, max_zoom), 0.0, Some(terrain), 0.0, detail)
    }

    fn cover_detailed(
        &self,
        tms: &TileMatrixSet,
        (tile_px, max_zoom): (f64, u8),
        height: f64,
        terrain: Option<&dyn Elevation>,
        margin_px: f64,
        detail: Detail,
    ) -> Vec<CoveredTile> {
        if let Some(low) = self.low_plane() {
            return low.cover_low(tms, tile_px, max_zoom);
        }
        if let Some(globe) = &self.globe {
            let metres = height * globe.metres_per_px / self.world_per_px;
            return self.cover_globe(globe, tms, (tile_px, max_zoom), metres, detail);
        }
        let center_zoom = (tms.tile_span(0) / (self.world_per_px * tile_px)).log2();
        self.cover_tiles(
            tms,
            max_zoom,
            (height, terrain),
            margin_px,
            detail,
            |index, offset| {
                let bounds = tms.tile_area(index).translated(DVec2::new(offset, 0.0));
                Some((bounds, center_zoom))
            },
        )
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
        if let Some(low) = self.low_plane() {
            return low.cover_low(tms, tile_px, max_zoom);
        }
        // Any set's tiles wrap the globe directly.
        if let Some(globe) = &self.globe {
            let detail = Detail::Lod(self.lod);
            return self.cover_globe(globe, tms, (tile_px, max_zoom), 0.0, detail);
        }
        let scale = self.world_per_px * tile_px;
        let detail = Detail::Lod(self.lod);
        self.cover_tiles(tms, max_zoom, (0.0, None), 0.0, detail, |index, offset| {
            let f = footprint(index)?;
            // The zoom at which this tile, at its own scale, would span
            // `tile_px` pixels at the camera.
            let center = f64::from(index.z) + (f.span / scale).log2();
            Some((f.bounds.translated(DVec2::new(offset, 0.0)), center))
        })
    }

    /// Covers this view of a low-zoom plane with tiles of a set on another,
    /// from where each lands on it. Hosts that cover often keep the
    /// footprints and call [`cover_warped`](Self::cover_warped) on
    /// [`low_plane`](Self::low_plane) instead.
    fn cover_low(&self, tms: &TileMatrixSet, tile_px: f64, max_zoom: u8) -> Vec<CoveredTile> {
        let warp = orb_crs::Warp::new(tms.clone(), self.projection);
        self.cover_warped(tms, tile_px, max_zoom, |tile| warp.footprint(tile))
    }

    /// The cover's traversal: `place` gives each tile's bounds in its copy
    /// of the world (by x offset) and the zoom whose tiles would span the
    /// source's tile size at the target's scale, or `None` to skip it.
    fn cover_tiles(
        &self,
        tms: &TileMatrixSet,
        max_zoom: u8,
        (height, terrain): (f64, Option<&dyn Elevation>),
        margin_px: f64,
        detail: Detail,
        mut place: impl FnMut(TileIndex, f64) -> Option<(WorldRect, f64)>,
    ) -> Vec<CoveredTile> {
        let frustum = Frustum::new(
            self.view_proj,
            DVec2::new(
                2.0 * margin_px / self.viewport.width,
                2.0 * margin_px / self.viewport.height,
            ),
        );
        // GL JS grows the culling volume as the bottom edge approaches the
        // horizon. Heights here stay in world units, just like tile bounds.
        let proximity =
            ((15.0 - (MAX_HORIZON_PITCH - self.pitch - self.fov / 2.0)) / 15.0).clamp(0.0, 1.0);
        let raised = proximity * height.max(500.0 * self.height_scale);
        let culling_z = self.target_z + raised;
        let flat_range = (culling_z.min(0.0), culling_z.max(0.0));
        let reach = (self.far_px * self.world_per_px).max(self.viewport.width * self.world_per_px);
        // Match the reference's bounded repeated-world policy: at most
        // three copies on either side, with the same generic DFS in each.
        let wrap_period = self.projection.wrap_period();
        let copies = wrap_period.map_or(0, |period| ((reach / period).ceil() as i32).min(3));
        let lod = match detail {
            Detail::Lod(lod) => lod,
            Detail::Terrain => TileLod::default(),
        };
        let weight = lod.weight(self.pitch);
        let eye = self.eye.truncate();
        // The eye's distance from the target: the centre's tiles show at
        // the covering zoom.
        let center_distance = self.eye_distance_px * self.world_per_px;
        let terrain_lod = matches!(detail, Detail::Terrain).then(|| self.terrain_lod());
        tms.cover_with(-copies..=copies, wrap_period, max_zoom, |index, offset| {
            let Some((bounds, center_zoom)) = place(index, offset) else {
                return CoverDecision::Skip;
            };
            let ground = terrain
                .and_then(|t| t.range(bounds, index.z))
                .map(|(lo, hi)| (lo * self.height_scale, hi * self.height_scale));
            // Elevated content stands on the ground under the tile.
            let (lo, hi) = ground.map_or(flat_range, |(lo, hi)| (lo, hi + raised));
            let rel = bounds.translated(-self.camera_world);
            if !frustum.meets(&rel, lo, hi) {
                return CoverDecision::Skip;
            }
            let z = f64::from(index.z);
            if let Some(terrain_lod) = &terrain_lod {
                let desired = center_zoom + terrain_lod.at(rel.clamp(eye));
                return if z < tile_zoom(desired) {
                    CoverDecision::Refine
                } else {
                    CoverDecision::Accept
                };
            }
            // Native's ideal zoom: no deeper than the source's tiles.
            let covering = tile_zoom(center_zoom + lod.zoom_shift).min(f64::from(max_zoom));
            let refine = match lod.mode {
                _ if weight == 0.0 => z < covering,
                TileLodMode::Default => {
                    // Its distance from the target along the farther axis,
                    // in tiles of the covering zoom.
                    let span = rel.size().max_element() * (z - covering).exp2();
                    let gap = (rel.min.max(-rel.max)).max(DVec2::ZERO).max_element();
                    z < covering && lod.splits(covering - z, gap / span, weight)
                }
                TileLodMode::Distance => {
                    // Its nearest point, and the eye's height over it.
                    let height = (self.eye.z - lo.min(self.eye.z)).max(1e-9);
                    let distance = rel.clamp(eye).distance(eye).hypot(height);
                    let levels = lod.levels(distance / center_distance, height / distance, weight);
                    z < tile_zoom(covering + levels)
                }
            };
            if refine {
                CoverDecision::Refine
            } else {
                CoverDecision::Accept
            }
        })
    }

    /// GL JS's terrain level of detail over this view: its distance terms
    /// at any pitch, from the eye's ground position and height.
    fn terrain_lod(&self) -> TerrainLod {
        let (behavior, fov_scale, penalty) = self.gl_js_terms();
        let center_distance = self.eye_distance_px * self.world_per_px;
        TerrainLod {
            eye: self.eye.truncate(),
            // The eye's height above the target, GL JS's `distanceToTileZ`.
            height: (self.eye.z - self.target_z).abs(),
            offset: (center_distance / fov_scale).log2() - penalty,
            behavior: behavior / 2.0,
        }
    }

    /// GL JS's default tile LOD terms: its distance behaviour, FOV scale and
    /// tile-count penalty.
    fn gl_js_terms(&self) -> (f64, f64, f64) {
        let horizon = MAX_HORIZON_PITCH.to_radians();
        let fov = self.fov.to_radians();
        let behavior = 2.0 * ((9.314 - 1.0) / ((horizon - fov).cos() / horizon.cos()).log2() - 1.0);
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
    }

    /// The tiles of `tms` that cover a globe view: those on the eye's side
    /// of the horizon and in the frustum. While the covering zoom (the
    /// target's tile zoom) is 4 or less, every tile takes it, as GL JS's
    /// globe gives them. Past that each tile takes the zoom whose tiles
    /// span `tile_px` pixels at the target's distance where it lies (being
    /// smaller on the sphere toward the poles), coarsening with distance
    /// by the [`TileLod`] as on the plane: distances along the sphere from
    /// the target, or from the eye with each tile's angle from straight
    /// down.
    ///
    /// Content standing up to `height` metres over a tile (extrusions,
    /// lifted labels) keeps it covered while that content can show: past
    /// the ground's horizon by the angle it peeks over it, and above the
    /// view's bottom edge, as GL JS's `expandCullingToContentElevation`.
    fn cover_globe(
        &self,
        globe: &GlobeView,
        tms: &TileMatrixSet,
        (tile_px, max_zoom): (f64, u8),
        height: f64,
        detail: Detail,
    ) -> Vec<CoveredTile> {
        let eye = globe.target + globe.eye;
        let eye_dir = eye.normalize();
        let height = height.max(0.0);
        let horizon = globe.horizon_angle() + (sphere::RADIUS / (sphere::RADIUS + height)).acos();
        // Points that can face the eye, or hold content that can, lie in
        // front of the horizon's plane.
        let horizon_plane = sphere::RADIUS * horizon.cos();
        let center_distance = globe.eye.length();
        let projection = tms.projection();
        // The zoom whose tiles span `tile_px` pixels where the set's scale
        // factor is `scale`, at the target's distance.
        let zoom_at =
            |scale: f64| (tms.tile_span(0) / (globe.metres_per_px * scale * tile_px)).log2();
        let covering = zoom_at(projection.scale_factor(sphere::geodetic(globe.target)));
        // (A set whose plane doesn't reach the target has no covering zoom.)
        let fixed = covering.is_finite() && tile_zoom(covering) <= GLOBE_FIXED_ZOOM;
        let covering = tile_zoom(covering);
        let (lod, weight) = match detail {
            Detail::Lod(lod) => (lod, lod.weight(self.pitch)),
            // GL JS's terms, at any pitch.
            Detail::Terrain => (TileLod::default(), 1.0),
        };
        let (behavior, fov_scale, penalty) = self.gl_js_terms();
        // The points under the eye and the target on the set's plane, for
        // tiles' nearest points.
        let under = projection.forward(sphere::geodetic(eye_dir));
        let target_dir = globe.target.normalize();
        let target = projection.forward(sphere::geodetic(target_dir));
        let period = projection.wrap_period();
        let frustum = GlobeFrustum::new(globe, height);
        tms.cover_with(0..=0, None, max_zoom, |index, _| {
            let b = tms.tile_area(index);
            // A grid of the tile's points on the sphere, row by row: 5×5
            // for the widest tiles, 3×3 below.
            let steps = if index.z < 3 { 4 } else { 2 };
            let n = steps + 1;
            let mut points = [DVec3::NAN; 27];
            for (k, point) in points.iter_mut().enumerate().take(n * n) {
                let uv = DVec2::new((k % n) as f64, (k / n) as f64) / steps as f64;
                let world = tms.tile_point(index, uv);
                let g = projection.inverse(world);
                if g.lon.is_finite() && g.lat.is_finite() {
                    *point = sphere::ecef(g, 0.0);
                }
            }
            let center = points[n * n / 2];
            if !center.is_finite() {
                return CoverDecision::Skip;
            }
            // A tile on the world's edge also draws the cap out to the
            // pole past it.
            let mut count = n * n;
            for (reaches, z) in sphere::poles(tms, index).into_iter().zip([1.0, -1.0]) {
                if reaches {
                    points[count] = DVec3::new(0.0, 0.0, z * sphere::RADIUS);
                    count += 1;
                }
            }
            let grid = &points[..n * n];
            let points = &points[..count];
            let finite = || points.iter().copied().filter(|p| p.is_finite());
            // The widest gap between neighbouring points: the sphere bulges
            // past them by at most its sagitta.
            let pair = |a: usize, b: usize| {
                let (a, b) = (grid[a], grid[b]);
                (a.is_finite() && b.is_finite()).then(|| a.angle_between(b))
            };
            let mut gap: f64 = 0.0;
            for k in 0..grid.len() {
                if k % n != steps {
                    gap = gap.max(pair(k, k + 1).unwrap_or(0.0));
                }
                if k + n < grid.len() {
                    gap = gap.max(pair(k, k + n).unwrap_or(0.0));
                }
            }
            // Its cap: the widest angle from its centre to a point, plus
            // the gap.
            let radius = finite()
                .map(|p| center.angle_between(p))
                .fold(0.0, f64::max)
                + gap;
            if center.angle_between(eye_dir) - radius > horizon {
                return CoverDecision::Skip;
            }
            // Its box in the east, north and up frame at its centre, padded
            // by the sagitta: far tighter than a box on the sphere's axes,
            // which a tile tilted to them fills only diagonally.
            let bulge = sphere::RADIUS * (1.0 - (gap / 2.0).min(FRAC_PI_2).cos());
            let up = center / sphere::RADIUS;
            let east = DVec3::Z.cross(up).try_normalize().unwrap_or(DVec3::Y);
            let frame = DMat3::from_cols(east, up.cross(east), up);
            let to_local = frame.transpose();
            let (mut lo, mut hi) = (DVec3::INFINITY, DVec3::NEG_INFINITY);
            for p in finite() {
                let local = to_local * (p - center);
                lo = lo.min(local);
                hi = hi.max(local);
            }
            let (lo, hi) = (lo - bulge, hi + bulge + DVec3::Z * height);
            let corners = box_corners(lo, hi).map(|c| center + frame * c);
            // Outside if the box is wholly behind the horizon's plane or
            // outside a plane of the frustum, or if the frustum, cut off
            // at the horizon, is wholly outside a face of the box or a
            // plane through the sphere's centre and an edge's ends (GL JS's
            // east and west planes, for any set's edges).
            if corners.iter().all(|c| c.dot(eye_dir) < horizon_plane)
                || !corners_in_frustum(
                    &globe.view_proj,
                    &corners.map(|c| c - globe.target),
                    DVec2::ZERO,
                )
            {
                return CoverDecision::Skip;
            }
            let faces = (0..3).any(|axis| {
                let a = frame.col(axis);
                let at = a.dot(center);
                frustum.beyond(a, at + lo[axis]) || frustum.beyond(-a, -(at + hi[axis]))
            });
            let last = n * n - 1;
            let edges = [
                (0, steps),
                (last - steps, last),
                (0, last - steps),
                (steps, last),
            ];
            let sides = edges.into_iter().any(|(a, b)| {
                // Ends nearly the same or nearly opposite pin no plane:
                // the cross product's length over the chord's (times the
                // radius) is the cosine of half the angle between them.
                let (a, b) = (grid[a], grid[b]);
                let (cross, chord) = (a.cross(b), a.distance(b));
                if !(chord > 1e-9 * sphere::RADIUS
                    && cross.length() > 1e-3 * chord * sphere::RADIUS)
                {
                    return false;
                }
                let normal = cross.normalize();
                let normal = normal * normal.dot(center).signum();
                let offset = finite()
                    .map(|p| normal.dot(p))
                    .fold(f64::INFINITY, f64::min);
                frustum.beyond(normal, offset - bulge)
            });
            if faces || sides {
                return CoverDecision::Skip;
            }
            if fixed {
                return if f64::from(index.z) < covering {
                    CoverDecision::Refine
                } else {
                    CoverDecision::Accept
                };
            }
            // Its point nearest `p` on the set's plane, clamped into it (the
            // nearest where the plane keeps the sphere's shapes).
            let nearest_to = |p: DVec2| {
                let shift = period.map_or(0.0, |w| ((b.center().x - p.x) / w).round() * w);
                let inside = projection.inverse((p + DVec2::new(shift, 0.0)).clamp(b.min, b.max));
                if inside.lon.is_finite() && inside.lat.is_finite() {
                    inside
                } else {
                    sphere::geodetic(center)
                }
            };
            let z = f64::from(index.z);
            let gl_js = matches!(detail, Detail::Terrain);
            if !gl_js && (lod.mode == TileLodMode::Default || weight == 0.0) {
                // Its covering zoom where it comes nearest the target, and
                // its distance from it along the sphere in tiles of that
                // zoom there.
                let near = nearest_to(target);
                let here = zoom_at(projection.scale_factor(near)) + lod.zoom_shift;
                let zoom = tile_zoom(here).min(f64::from(max_zoom));
                let refine = z < zoom && {
                    let span = globe.metres_per_px * tile_px * (here - zoom).exp2();
                    let arc = sphere::ecef(near, 0.0).angle_between(target_dir) * sphere::RADIUS;
                    lod.splits(zoom - z, arc / span, weight)
                };
                return if refine {
                    CoverDecision::Refine
                } else {
                    CoverDecision::Accept
                };
            }
            // Its nearest point to the eye: the point under the eye in it,
            // or else one of its grid or a pole past its edge.
            let inside = nearest_to(under);
            let nearest = finite()
                .chain([sphere::ecef(inside, 0.0)])
                .min_by(|a, b| a.distance_squared(eye).total_cmp(&b.distance_squared(eye)))
                .unwrap_or(center);
            let distance = nearest.distance(eye).max(center_distance * 1e-3);
            // The tile's scale taken in the tile (even when the nearest
            // point is a pole past it), and its tilt, as GL JS's globe
            // takes it: its nearest point's angle from straight down at the
            // eye. (Its angle to the ground there grows on toward the
            // horizon, which would coarsen far tiles far more.)
            let tile_pitch = (eye - nearest).angle_between(eye);
            let levels = if gl_js {
                // GL JS's distance, behaviour, FOV scale and penalty.
                (center_distance / distance / fov_scale).log2()
                    + behavior * tile_pitch.cos().max(1e-6).log2() / 2.0
                    - penalty
            } else {
                lod.levels(distance / center_distance, tile_pitch.cos(), weight)
            };
            let desired = zoom_at(projection.scale_factor(inside)) + lod.zoom_shift + levels;
            if z < tile_zoom(desired) {
                CoverDecision::Refine
            } else {
                CoverDecision::Accept
            }
        })
    }
}

/// The covering zoom up to which GL JS's globe gives every tile that zoom;
/// past it, tiles take zooms by distance.
const GLOBE_FIXED_ZOOM: f64 = 4.0;

/// A globe view's frustum cut off where it passes the horizon, as GL JS
/// fits its globe frustum's far plane to the horizon: its eight corners
/// in sphere space.
struct GlobeFrustum {
    corners: [DVec3; 8],
}

impl GlobeFrustum {
    /// The view's frustum, its far end pushed out for content standing up
    /// to `height` metres, which shows past the ground's horizon.
    fn new(globe: &GlobeView, height: f64) -> Self {
        let inverse = globe.view_proj.inverse();
        let ndc = box_corners(DVec3::new(-1.0, -1.0, 0.0), DVec3::ONE);
        let mut corners = ndc.map(|c| inverse.project_point3(c) + globe.target);
        if globe.plane.is_some() {
            return Self { corners };
        }
        let eye = globe.target + globe.eye;
        let up = eye.normalize();
        let plane = sphere::RADIUS * sphere::RADIUS / eye.length();
        let forward = -globe.eye.normalize();
        let depth = |p: DVec3| (p - eye).dot(forward);
        // The farthest depth at which the edges meet the horizon's plane
        // (or their full length where they don't)...
        let rays = (0..4)
            .map(|i| {
                let (near, along) = (corners[i], corners[i + 4] - corners[i]);
                let t = (plane - near.dot(up)) / along.dot(up);
                if along.dot(up) < 0.0 && t >= 0.0 {
                    depth(near + along * t.min(1.0))
                } else {
                    depth(corners[i + 4])
                }
            })
            .fold(0.0, f64::max);
        // ...but no farther than the farthest visible point of the sphere:
        // on the horizon, or where the sphere faces straight back along
        // the view if it can be seen there.
        let ring = (plane * up - eye).dot(forward)
            + (sphere::RADIUS * sphere::RADIUS - plane * plane)
                .max(0.0)
                .sqrt()
                * (forward - up * forward.dot(up)).length();
        let back = forward * sphere::RADIUS;
        let visible = if back.dot(eye) >= sphere::RADIUS * sphere::RADIUS {
            ring.max(depth(back))
        } else {
            ring
        };
        // Content standing on the sphere shows farther along a ray that
        // grazes it, by up to the tangent's length from it to the
        // content's top.
        let shell = sphere::RADIUS + height;
        let far = rays.min(visible) + (shell * shell - sphere::RADIUS * sphere::RADIUS).sqrt();
        for i in 0..4 {
            let (near, along) = (corners[i], corners[i + 4] - corners[i]);
            let s = ((far - depth(near)) / along.dot(forward)).clamp(0.0, 1.0);
            corners[i + 4] = near + along * s;
        }
        Self { corners }
    }

    /// Whether all of it is on the outer side of a plane: every corner
    /// `p` has `normal · p < offset`.
    fn beyond(&self, normal: DVec3, offset: f64) -> bool {
        self.corners.iter().all(|c| normal.dot(*c) < offset)
    }
}

/// A view's frustum, widened by a margin in NDC per unit w (a reach in
/// pixels past the viewport's edges), for conservative tests of
/// camera-relative boxes.
struct Frustum {
    view_proj: DMat4,
    margin: DVec2,
    /// Its axis-aligned bounds in camera-relative world space.
    min: DVec3,
    max: DVec3,
}

impl Frustum {
    fn new(view_proj: DMat4, margin: DVec2) -> Self {
        let inverse = view_proj.inverse();
        let (mut min, mut max) = (DVec3::INFINITY, DVec3::NEG_INFINITY);
        for i in 0..8 {
            let corner = inverse.project_point3(DVec3::new(
                if i & 1 == 0 { -1.0 } else { 1.0 } * (1.0 + margin.x),
                if i & 2 == 0 { -1.0 } else { 1.0 } * (1.0 + margin.y),
                if i & 4 == 0 { 0.0 } else { 1.0 },
            ));
            min = min.min(corner);
            max = max.max(corner);
        }
        Self {
            view_proj,
            margin,
            min,
            max,
        }
    }

    /// Whether a ground rectangle raised between `lo` and `hi` may meet it.
    fn meets(&self, rect: &WorldRect, lo: f64, hi: f64) -> bool {
        // The frustum planes alone can accept disjoint boxes near a corner.
        // Include the box's three axes, as GL JS's AABB test does.
        if rect.min.x > self.max.x
            || rect.min.y > self.max.y
            || lo > self.max.z
            || rect.max.x < self.min.x
            || rect.max.y < self.min.y
            || hi < self.min.z
        {
            return false;
        }
        box_in_frustum(
            &self.view_proj,
            rect.min.extend(lo),
            rect.max.extend(hi),
            self.margin,
        )
    }
}

/// Whether a box may meet the frustum of `view_proj` widened by `margin`
/// (NDC per unit w): no clip plane has all eight corners outside.
fn box_in_frustum(view_proj: &DMat4, lo: DVec3, hi: DVec3, margin: DVec2) -> bool {
    corners_in_frustum(view_proj, &box_corners(lo, hi), margin)
}

/// A box's corners: bit 0 of the index picks x's end, bit 1 y's, bit 2 z's.
fn box_corners(lo: DVec3, hi: DVec3) -> [DVec3; 8] {
    std::array::from_fn(|i| {
        DVec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        )
    })
}

/// Whether a convex volume with these corners may meet the frustum of
/// `view_proj` widened by `margin` (NDC per unit w): no clip plane has
/// them all outside.
fn corners_in_frustum(view_proj: &DMat4, corners: &[DVec3; 8], margin: DVec2) -> bool {
    let clip = corners.map(|c| *view_proj * c.extend(1.0));
    let all = |f: &dyn Fn(&DVec4) -> bool| clip.iter().all(f);
    let (mx, my) = (1.0 + margin.x, 1.0 + margin.y);
    !(all(&|c| c.x > c.w * mx)
        || all(&|c| c.x < -c.w * mx)
        || all(&|c| c.y > c.w * my)
        || all(&|c| c.y < -c.w * my)
        || all(&|c| c.z > c.w)
        || all(&|c| c.z < 0.0))
}

/// How a cover picks its tiles' zooms.
#[derive(Clone, Copy, Debug)]
enum Detail {
    /// The map's tile level of detail.
    Lod(TileLod),
    /// GL JS's terrain level of detail ([`View::cover_terrain`]).
    Terrain,
}

/// GL JS's terrain level of detail: a tile whose nearest ground point is
/// `p` (camera-relative world) takes the zoom whose tiles span the
/// source's tile size at the target, plus [`at`](Self::at)`(p)`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TerrainLod {
    /// The eye's ground position, camera-relative world.
    eye: DVec2,
    /// The eye's height above the target, world units.
    height: f64,
    /// `offset − log2(distance) + behavior · log2(height / distance)`.
    offset: f64,
    behavior: f64,
}

impl TerrainLod {
    /// The zoom levels added to the cover's centre zoom at `p`.
    fn at(&self, p: DVec2) -> f64 {
        let distance = p.distance(self.eye).hypot(self.height).max(1.0e-9);
        self.offset - distance.log2() + self.behavior * (self.height / distance).log2()
    }
}

/// The whole tile zoom a cover's desired zoom asks for. A whole zoom
/// computed through scales and trigonometry can land an ulp short of
/// itself (1.9999999999999998 at zoom 2, latitude 57°); it means that zoom.
fn tile_zoom(desired: f64) -> f64 {
    (desired + 1e-9).floor().max(0.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn whole_zooms_cover_with_their_own_tiles_at_any_latitude() {
        let tms = orb_crs::TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        for lat in [0.0, 57.0, 80.0] {
            let camera = CameraState::new(Geodetic::new(0.0, lat), 2.0, &mercator());
            let surface = Surface {
                projection: mercator(),
                globe: None,
            };
            let viewport = Viewport {
                width: 900.0,
                height: 512.0,
            };
            let view = View::new(&camera, viewport, &surface);
            let zooms: Vec<u8> = view
                .cover(&tms, 512.0, 18)
                .iter()
                .map(|c| c.index.z)
                .collect();
            assert!(zooms.iter().all(|&z| z == 2), "latitude {lat}: {zooms:?}");
        }
    }

    use super::*;
    use orb_crs::Globe;

    fn mercator() -> Projection {
        *TileMatrixSet::by_id("WebMercatorQuad")
            .unwrap()
            .projection()
    }
    use orb_crs::TileIndex;

    #[test]
    fn padding_moves_the_target_and_the_horizon_to_the_padded_middle() {
        let viewport = Viewport {
            width: 800.0,
            height: 600.0,
        };
        let surface = Surface::plane(mercator());
        let plain = CameraState {
            pitch: 60.0,
            ..CameraState::new(Geodetic::new(139.7, 35.7), 12.0, &mercator())
        };
        let padded = CameraState {
            padding: Padding {
                top: 200.0,
                left: 100.0,
                ..Padding::default()
            },
            ..plain
        };
        let (a, b) = (
            View::new(&plain, viewport, &surface),
            View::new(&padded, viewport, &surface),
        );
        let target = mercator().forward(plain.target);
        let moved = DVec2::new(50.0, 100.0);
        assert!((b.center() - DVec2::new(450.0, 400.0)).length() < 1e-9);
        assert!((b.project(target).unwrap() - b.center()).length() < 1e-6);
        // The same picture, shifted.
        let ahead = target + DVec2::new(0.0, 2000.0);
        let shifted = a.project(ahead).unwrap() + moved;
        assert!((b.project(ahead).unwrap() - shifted).length() < 1e-6);
        assert!((b.unproject(shifted).unwrap() - ahead).length() < 1e-3);
        assert!((b.sky_horizon() - a.sky_horizon() - moved.y).abs() < 1e-9);
        // More shows above the target, so the far plane reaches farther and
        // the ground with it.
        assert!(b.ground_horizon() < a.ground_horizon() + moved.y);
        assert_eq!(a.eye(), b.eye());
    }

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

    /// A margin keeps the tiles whose points reach into the view from past
    /// its edge.
    #[test]
    fn cover_margins_reach_past_the_edge() {
        let (view, tms) = setup(3.0, 0.0);
        let plain = view.cover_volume(&tms, 512.0, 14, 0.0, None, 0.0);
        let padded = view.cover_volume(&tms, 512.0, 14, 0.0, None, 300.0);
        assert!(padded.len() > plain.len());
        assert!(plain.iter().all(|t| padded.contains(t)));
    }

    #[test]
    fn elevated_cover_keeps_roofs_whose_ground_is_below_the_view() {
        let (view, tms) = setup(14.0, 80.0);
        let ground = view.cover(&tms, 512.0, 14);
        let frustum = Frustum::new(view.view_proj(), DVec2::ZERO);
        assert!(
            ground.iter().any(|tile| {
                let rect = tms.tile_bounds(tile.index).translated(-view.camera_world());
                !frustum.meets(&rect, 0.0, 0.0) && frustum.meets(&rect, 0.0, 500.0)
            }),
            "the assumed feature height retains tiles whose roofs enter the view"
        );
        assert!(view.ground_horizon() > 0.0);
        assert!(view.ground_horizon() < view.viewport().height / 2.0);
    }

    /// Past the pitch where the top edge reaches the horizon, the ground
    /// runs out to MapLibre Native's far plane, 100 times as deep as the
    /// target, within a percent of the true horizon, and the sky starts
    /// where it ends.
    #[test]
    fn high_pitch_ground_reaches_natives_far_plane_and_meets_the_sky() {
        let viewport = Viewport {
            width: 1200.0,
            height: 700.0,
        };
        for pitch in [75.0, 80.0, 85.0] {
            let mut camera = CameraState::new(Geodetic::new(-74.0, 40.7), 14.0, &mercator());
            camera.pitch = pitch;
            let view = View::new(&camera, viewport, &Surface::plane(mercator()));
            let depth = view.far_px / 1.01 / view.eye_distance_px;
            assert!((depth - 100.0).abs() < 1e-6, "{pitch}: {depth}");
            assert!((view.ground_horizon() - view.sky_horizon()).abs() < 0.5);
            let below_true = view.eye_distance_px / pitch.to_radians().tan();
            let below_sky = view.center().y - view.sky_horizon();
            assert!((below_sky / below_true - HORIZON_SHARE).abs() < 1e-9);
        }
    }

    /// On the v8 globe the plane's ground stops where the sphere's does,
    /// at its horizon from the eye's altitude, short of Native's far plane:
    /// the two surfaces meet through the handover, and the sky meets both.
    #[test]
    fn globe_planes_end_their_ground_at_the_spheres_horizon() {
        let viewport = Viewport {
            width: 1200.0,
            height: 700.0,
        };
        let globe = Surface {
            projection: mercator(),
            globe: Some(Globe::V8),
        };
        for zoom in [12.0, 14.0] {
            let mut camera = CameraState::new(Geodetic::new(-74.0, 40.7), zoom, &mercator());
            camera.pitch = 80.0;
            let view = View::new(&camera, viewport, &globe);
            let flat = View::new(&camera, viewport, &Surface::plane(mercator()));
            assert!(view.far_px < flat.far_px / 2.0, "{zoom}");
            assert!((view.ground_horizon() - view.sky_horizon()).abs() < 0.5);
            // The ground's far end is the sphere's horizon: the ground point
            // straight ahead as far as the eye sees over the sphere shows
            // there.
            let altitude = view.eye.z / view.height_scale;
            let r = sphere::RADIUS;
            let reach = ((r + altitude).powi(2) - r * r).sqrt() * view.height_scale;
            let under_eye = view.eye.truncate();
            let ahead = under_eye - under_eye.normalize() * reach;
            let end = view.project_3d(ahead + view.camera_world, 0.0).unwrap();
            assert!((end.y - view.sky_horizon()).abs() < 0.5, "{zoom}: {end}");
        }
    }

    /// Below the pitch where Native's bound takes over, the far plane
    /// reaches the ground at every corner of the view, however padding
    /// moves the target and roll turns the view.
    #[test]
    fn the_far_plane_holds_the_ground_at_every_corner() {
        let viewport = Viewport {
            width: 1200.0,
            height: 700.0,
        };
        let surface = Surface::plane(mercator());
        // Top, right, bottom and left padding, and roll.
        for (top, right, bottom, left, roll) in [
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 350.0, 0.0, 0.0),
            (300.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 400.0, 30.0),
            (300.0, 0.0, 0.0, 0.0, 180.0),
            (0.0, 0.0, 300.0, 0.0, 150.0),
            (0.0, 500.0, 0.0, 0.0, -90.0),
        ] {
            let padding = Padding {
                top,
                right,
                bottom,
                left,
            };
            for pitch in [45.0, 55.0, 60.0, 65.0] {
                let mut camera = CameraState::new(Geodetic::new(-74.0, 40.7), 14.0, &mercator());
                camera.pitch = pitch;
                camera.roll = roll;
                camera.padding = padding;
                let view = View::new(&camera, viewport, &surface);
                for corner in [
                    DVec2::ZERO,
                    DVec2::new(viewport.width, 0.0),
                    DVec2::new(0.0, viewport.height),
                    DVec2::new(viewport.width, viewport.height),
                ] {
                    let (near, far) = view.ray(corner, &view.view_proj);
                    if near.z <= far.z {
                        continue;
                    }
                    // The corner looks at the ground: its depth there, when
                    // nearer than Native's bound, is within the far plane.
                    let ground = near + (far - near) * (near.z / (near.z - far.z));
                    let depth = (view.view_proj * ground.extend(1.0)).w;
                    if depth < view.sea_level_px / (1.0 - HORIZON_SHARE) {
                        assert!(
                            depth < view.far_px,
                            "{padding:?} roll {roll} pitch {pitch} {corner}: {depth} past {}",
                            view.far_px
                        );
                    }
                }
            }
        }
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
    fn detail_changes_continuously_with_pitch() {
        // Up to GL JS's threshold, one zoom, as GL JS.
        let (view, tms) = setup(10.0, 60.0);
        assert!(view.cover(&tms, 512.0, 22).iter().all(|t| t.index.z == 10));
        // Across its switch at 60°, a tenth of a degree changes little (at
        // a zoom between whole ones, which any change tips).
        let zooms = |pitch| {
            let (view, tms) = setup(10.3, pitch);
            let tiles = view.cover(&tms, 512.0, 22);
            tiles.iter().map(|t| f64::from(t.index.z)).sum::<f64>() / tiles.len() as f64
        };
        assert!((zooms(59.9) - zooms(60.1)).abs() < 0.05);
        // And where the ease ends.
        assert!((zooms(64.9) - zooms(65.1)).abs() < 0.05);
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

    /// A low-zoom plane shows through zoom 6 and hands over to Mercator by
    /// zoom 7; while it shows, positions go through it both ways.
    #[test]
    fn low_planes_hand_over_to_mercator() {
        let equal_earth = Projection::from_descriptor("equal-earth").unwrap();
        let surface = Surface {
            projection: mercator(),
            globe: Some(Globe {
                handover: Some([6.0, 7.0]),
                plane: Some(equal_earth),
            }),
        };
        let chicago = Geodetic::new(-87.6357, 41.88155);
        let view = |zoom: f64| {
            let mut camera = CameraState::new(chicago, zoom, &mercator());
            camera.pitch = 30.0;
            let viewport = Viewport {
                width: 1024.0,
                height: 768.0,
            };
            View::new(&camera, viewport, &surface)
        };
        assert_eq!(view(5.0).globe().map(|g| g.amount), Some(1.0));
        assert!((view(6.5).globe().unwrap().amount - 0.5).abs() < 1e-9);
        assert!(view(7.0).globe().is_none());
        for zoom in [2.0, 5.0, 6.5] {
            let view = view(zoom);
            assert!(view.low_plane().is_some());
            for px in [DVec2::new(512.0, 384.0), DVec2::new(100.0, 600.0)] {
                let world = view.unproject(px).unwrap();
                let back = view.project(world).unwrap();
                assert!(back.distance(px) < 1e-6, "z{zoom} {px} -> {back}");
            }
            // The target sits in the middle, on the low plane.
            let middle = view.project(mercator().forward(chicago)).unwrap();
            assert!(middle.distance(DVec2::new(512.0, 384.0)) < 1e-6);
            assert!(
                !view
                    .cover(&TileMatrixSet::by_id("WebMercatorQuad").unwrap(), 512.0, 14)
                    .is_empty()
            );
        }
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

    /// Content lifted 500 km over the ground (a `symbol-height-offset`)
    /// keeps tiles past the ground's horizon covered, as it shows over it.
    #[test]
    fn globe_cover_keeps_tiles_under_lifted_content() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        let view = globe_view(Geodetic::new(0.0, 0.0), 4.5, 80.0, 0.0);
        let globe = view.globe().unwrap();
        let lift = 500e3 * view.world_per_px() / globe.metres_per_px;
        let plain = view.cover_volume(&tms, 512.0, 22, 0.0, None, 0.0);
        let lifted = view.cover_volume(&tms, 512.0, 22, lift, None, 0.0);
        assert!(plain.iter().all(|t| lifted.contains(t)));
        let eye = (globe.target + globe.eye).normalize();
        let past: Vec<_> = (lifted.iter())
            .filter(|t| !plain.contains(t))
            .map(|t| {
                let world = tms.tile_bounds(t.index).center();
                sphere::ecef(tms.projection().inverse(world), 0.0).angle_between(eye)
            })
            .collect();
        assert!(!past.is_empty());
        // Some of the added tiles lie past the ground's horizon.
        assert!(past.iter().any(|&a| a > globe.horizon_angle()), "{past:?}");
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
