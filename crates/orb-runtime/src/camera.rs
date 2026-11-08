//! The map's camera and gestures: the map is the camera's
//! only writer. The app's moves, the style's camera and input all go
//! through its `orb_camera::Camera`; each frame applies arrived data, then
//! advances gestures and animations at the frame's timestamp, then draws.

use glam::DVec2;
use orb_camera::input::{Input, InteractEvent};
use orb_camera::{
    AnimationId, Bounds, CameraState, CameraTarget, Constraints, EaseOptions, FitOptions,
    FlyOptions, FreeCamera, Viewport,
};
use orb_crs::Geodetic;

use crate::{Map, MapEvent};

impl Map {
    /// The camera as it is now: it moves during gestures and animations.
    pub fn camera(&self) -> &CameraState {
        self.camera.state()
    }

    /// Moves the camera at once (GL JS `jumpTo`), ending any animation.
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

    /// Stops the running animation where it is, as interrupted.
    pub fn stop(&mut self) {
        self.camera.stop();
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

    /// Whether an animation, a glide or a gesture moves the camera, or
    /// input holds it.
    pub fn is_moving(&self) -> bool {
        self.camera.is_animating() || self.camera.is_held() || self.input.is_active()
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

    /// Brings the camera to this frame: it stands on the surface and
    /// terrain that have arrived and the frame's viewport, then gestures
    /// and animations advance to `time`. Returns whether it moves on.
    pub(crate) fn advance_camera(&mut self, viewport: Viewport, time: f64) -> bool {
        self.camera.set_surface(self.surface.view_surface());
        let terrain = self.elevation_revision();
        if terrain != self.terrain_seen {
            self.terrain_seen = terrain;
            self.camera.set_terrain(self.elevation());
        }
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
                InteractEvent::Click { position, button } => MapEvent::Click { position, button },
                InteractEvent::DoubleClick { position } => MapEvent::DoubleClick { position },
            });
        }
        gesture || animating
    }
}
