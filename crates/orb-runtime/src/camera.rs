//! The map's camera and gestures: the map is the camera's
//! only writer. The app's moves, the style's camera and input all go
//! through its `orb_camera::Camera`; each frame applies arrived data, then
//! advances gestures and animations at the frame's timestamp, then draws.

use glam::DVec2;
use orb_camera::input::{GestureOptions, Input, InteractEvent};
use orb_camera::{
    AnimationId, Bounds, CameraState, CameraTarget, Constraints, EaseOptions, FitOptions,
    FlyOptions, FreeCamera, GestureMomentum, Skew, Viewport,
};
use orb_crs::Geodetic;

use crate::{Map, MapEvent};

impl Map {
    /// The camera as it is now: it moves during gestures and animations.
    pub fn camera(&self) -> &CameraState {
        self.camera.state()
    }

    /// Moves the camera at once (GL JS `jumpTo`). Like every move, it
    /// takes only the properties it sets from the animations running.
    pub fn jump_to(&mut self, target: &CameraTarget) {
        self.camera.jump_to(target);
    }

    /// Moves the camera to exactly `state` at once, as
    /// [`jump_to`](Self::jump_to) does: for hosts that hold exact cameras.
    pub fn jump_to_state(&mut self, state: CameraState) {
        self.camera.jump_to_state(state);
    }

    /// Animates the camera (GL JS `easeTo`) from the next frame on.
    pub fn ease_to(&mut self, target: &CameraTarget, options: EaseOptions) -> AnimationId {
        self.camera.ease_to(target, options, None)
    }

    /// Flies along GL JS's zoom-out curve (GL JS `flyTo`) from the next
    /// frame on.
    pub fn fly_to(&mut self, target: &CameraTarget, options: &FlyOptions) -> AnimationId {
        self.camera.fly_to(target, options, None)
    }

    /// Continues a gesture the host recognized itself with `momentum`
    /// from the next frame on (mln-ffi's end-phase camera delta). `None`
    /// when nothing moves or settles over time.
    pub fn start_momentum(&mut self, momentum: &GestureMomentum) -> Option<AnimationId> {
        self.camera.start_momentum(momentum, None)
    }

    /// Stops every running animation where it is, as interrupted.
    pub fn stop(&mut self) {
        self.camera.stop();
    }

    /// Stops the animation `id` where it is, as interrupted, leaving the
    /// others running.
    pub fn stop_animation(&mut self, id: AnimationId) {
        self.camera.stop_animation(id);
    }

    /// The camera that shows `bounds` in the viewport (GL JS
    /// `cameraForBounds`), or `None` when the padding leaves no room.
    pub fn camera_for_bounds(&self, bounds: Bounds, options: &FitOptions) -> Option<CameraTarget> {
        self.camera.camera_for_bounds(bounds, options)
    }

    /// Zoom and pitch ranges and bounds the camera keeps to, whoever moves
    /// it.
    pub fn set_constraints(&mut self, constraints: Constraints) {
        self.camera.set_constraints(constraints);
    }

    /// Whether the camera's target stands on the terrain at rest (GL JS
    /// `centerClampedToGround`, on by default). Off, it keeps the elevation
    /// it is given, and the camera orbits a point in the air.
    pub fn set_center_clamped_to_ground(&mut self, clamped: bool) {
        self.camera.set_center_clamped_to_ground(clamped);
    }

    /// Draws the map in MapLibre Native's axonometric projection, heights
    /// leaning by `skew` and padding ignored, or in perspective (`None`).
    pub fn set_axonometric(&mut self, skew: Option<Skew>) {
        self.camera.set_axonometric(skew);
        (self.services.wake)();
    }

    pub fn center_clamped_to_ground(&self) -> bool {
        self.camera.center_clamped_to_ground()
    }

    /// Puts the target this many metres above sea level at once (GL JS
    /// `setCenterElevation`); while clamped, the terrain takes it back.
    pub fn set_center_elevation(&mut self, elevation: f64) {
        self.camera.set_center_elevation(elevation);
    }

    /// Hands the map a pointer, wheel or key input at `time`, the event's
    /// timestamp on the clock frame times use (milliseconds), in logical
    /// pixels from the viewport's top-left corner. Returns whether the map
    /// used it, so hosts can stop the platform's default.
    pub fn input(&mut self, input: Input, time: f64) -> bool {
        self.input.handle(&mut self.camera, input, time)
    }

    /// What gestures may move, how releases continue, and the platform's
    /// thresholds; changing it cancels a gesture under way.
    pub fn set_gestures(&mut self, options: GestureOptions) {
        self.input.set_options(&mut self.camera, options);
    }

    pub fn gestures(&self) -> &GestureOptions {
        self.input.options()
    }

    /// The box a Shift-drag zooms to while it is drawn, corner to corner
    /// in logical pixels, for the host to show.
    pub fn box_zoom(&self) -> Option<[DVec2; 2]> {
        self.input.box_zoom()
    }

    /// Whether an animation, a glide or a gesture moves the camera, input
    /// holds it, or a move's end waits for the next frame to report it: a
    /// map that says it rests has sent `moveend`.
    pub fn is_moving(&self) -> bool {
        self.camera.is_moving() || self.input.is_active()
    }

    /// Whether an animation or a glide moves the camera on its own
    /// (MapLibre Native's `inTransition`): unlike [`is_moving`](Self::is_moving),
    /// not input holding it, nor a move's end waiting for a frame to report it.
    pub fn is_animating(&self) -> bool {
        self.camera.is_animating()
    }

    /// The logical pixel showing `point` now, on the terrain if there is
    /// one, or `None` behind the eye.
    pub fn project(&self, point: Geodetic) -> Option<DVec2> {
        self.camera.project(point)
    }

    /// The ground point under a logical pixel now, on the terrain if there
    /// is one, or `None` above the horizon.
    pub fn unproject(&self, point: DVec2) -> Option<Geodetic> {
        self.camera.unproject(point)
    }

    /// [`project`](Self::project) for `camera` in this map's viewport, on
    /// its surface and terrain.
    pub fn project_with(&self, camera: &CameraState, point: Geodetic) -> Option<DVec2> {
        self.camera.project_with(camera, point)
    }

    /// [`unproject`](Self::unproject) for `camera` in this map's viewport.
    pub fn unproject_with(&self, camera: &CameraState, point: DVec2) -> Option<Geodetic> {
        self.camera.unproject_with(camera, point)
    }

    /// Where a jump to `target` would put the camera, without moving it.
    pub fn camera_for(&self, target: &CameraTarget) -> CameraState {
        self.camera.state_for(target)
    }

    /// The camera's eye and orientation now (GL JS `getFreeCameraOptions`).
    pub fn free_camera(&self) -> FreeCamera {
        self.camera.free()
    }

    /// Moves the eye to `free` at once (GL JS `setFreeCameraOptions`),
    /// ending any animation.
    pub fn jump_to_free(&mut self, free: &FreeCamera) {
        self.camera.jump_to_free(free);
    }

    /// The viewport the camera and gestures use: the last frame's, in
    /// logical pixels.
    pub fn viewport(&self) -> Viewport {
        self.camera.viewport()
    }

    /// Sets the viewport before a frame does (each frame sets its
    /// target's): fits, projections and constraints use it at once.
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.camera.set_viewport(viewport);
    }

    /// Stands the camera on the DEM data that has arrived since it last
    /// did. Returns whether that moved it.
    pub(crate) fn stand_on_terrain(&mut self) -> bool {
        let terrain = self.elevation_revision();
        if terrain == self.terrain_seen {
            return false;
        }
        self.terrain_seen = terrain;
        let before = self.camera.revision();
        self.camera.set_terrain(self.elevation());
        self.camera.revision() != before
    }

    /// Brings the camera to this frame: it stands on the surface and
    /// terrain that have arrived and the frame's viewport, then gestures
    /// and animations advance to `time`. Returns whether it moves on.
    pub(crate) fn advance_camera(&mut self, viewport: Viewport, time: f64) -> bool {
        self.camera.set_surface(self.surface.view_surface());
        self.stand_on_terrain();
        if self.camera.viewport() != viewport {
            self.camera.set_viewport(viewport);
        }
        let gesture = self.input.frame(&mut self.camera, time);
        let animating = self.camera.frame(time);
        // A camera that moved since the last frame makes the next rest a
        // new idle.
        if self.camera.revision() != self.camera_seen {
            self.camera_seen = self.camera.revision();
            self.events.wake();
        }
        for event in self.camera.take_events() {
            self.events.push(MapEvent::Camera(event));
        }
        for event in self.input.take_events() {
            self.events.push(match event {
                InteractEvent::Click { position } => MapEvent::Click { position },
                InteractEvent::DoubleClick { position } => MapEvent::DoubleClick { position },
                InteractEvent::LongClick { position } => MapEvent::LongClick { position },
            });
        }
        gesture || animating
    }
}
