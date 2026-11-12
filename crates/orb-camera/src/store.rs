//! The camera store: the single writer of a map's camera. Gestures, the
//! app's animations and jumps all move it; the host reads it once a frame
//! and draws what it shows.

use std::sync::Arc;

use glam::DVec2;
use orb_crs::Geodetic;
use orb_crs::Projection;

use crate::Skew;
use crate::animate::{CameraTarget, EaseOptions, FlyOptions, Glide, Motion};
use crate::fit::FitOptions;
use crate::momentum::{GestureMomentum, Momentum, MomentumVelocity, Stage};
use crate::moves::{Bounds, Constraints, wrap_bearing};
use crate::{CameraState, Elevation, FreeCamera, Surface, View, Viewport};

/// Identifies an animation in [`CameraEvent::AnimationEnd`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AnimationId(pub u64);

/// What changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Change {
    /// Anything about the camera: GL JS's `move`.
    Move,
    Zoom,
    Rotate,
    Pitch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Phase {
    Start,
    /// Once per frame while it changes.
    Progress,
    End,
}

/// The camera's events, GL JS's `movestart` … `pitchend` as reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraEvent {
    Changed {
        change: Change,
        phase: Phase,
        /// The move came from user input (GL JS events with an
        /// `originalEvent`).
        user: bool,
    },
    /// An animation stopped: it arrived, or another move or user input
    /// interrupted it. `velocity` is how fast an interrupted momentum still
    /// moved, which the next gesture can add to its own release; zero for
    /// every other end.
    AnimationEnd {
        id: AnimationId,
        interrupted: bool,
        velocity: MomentumVelocity,
    },
}

const CHANGES: [Change; 4] = [Change::Move, Change::Zoom, Change::Rotate, Change::Pitch];

/// The camera properties an animation moves. A later move takes the ones
/// it sets from the animations running, which go on moving the rest, as
/// MapLibre Native's transitions do; one left with none ends interrupted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Fields(u8);

impl Fields {
    pub(crate) const NONE: Self = Self(0);
    pub(crate) const CENTER: Self = Self(1);
    pub(crate) const ZOOM: Self = Self(1 << 1);
    pub(crate) const BEARING: Self = Self(1 << 2);
    pub(crate) const PITCH: Self = Self(1 << 3);
    const ROLL: Self = Self(1 << 4);
    const FOV: Self = Self(1 << 5);
    const ELEVATION: Self = Self(1 << 6);
    const PADDING: Self = Self(1 << 7);
    const ALL: Self = Self(u8::MAX);

    pub(crate) fn has(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub(crate) fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub(crate) fn is_empty(self) -> bool {
        self.0 == 0
    }

    fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// What differs between two cameras.
    fn between(a: &CameraState, b: &CameraState) -> Self {
        let set = |on: bool, field: Self| if on { field } else { Self::NONE };
        set(a.target != b.target, Self::CENTER)
            | set(a.resolution != b.resolution, Self::ZOOM)
            | set(a.bearing != b.bearing, Self::BEARING)
            | set(a.pitch != b.pitch, Self::PITCH)
            | set(a.roll != b.roll, Self::ROLL)
            | set(a.fov != b.fov, Self::FOV)
            | set(a.elevation != b.elevation, Self::ELEVATION)
            | set(a.padding != b.padding, Self::PADDING)
    }

    /// What a move to `target` sets. The center moves with a center or an
    /// offset, and so does its height when it lands on the terrain
    /// (`aims`).
    fn of(target: &CameraTarget, aims: bool) -> Self {
        let set = |on: bool, field: Self| if on { field } else { Self::NONE };
        let center = target.center.is_some() || target.offset != DVec2::ZERO;
        set(center, Self::CENTER)
            | set(target.zoom.is_some(), Self::ZOOM)
            | set(target.bearing.is_some(), Self::BEARING)
            | set(target.pitch.is_some(), Self::PITCH)
            | set(target.roll.is_some(), Self::ROLL)
            | set(target.fov.is_some(), Self::FOV)
            | set(
                target.elevation.is_some() || (center && aims),
                Self::ELEVATION,
            )
            | set(target.padding.is_some(), Self::PADDING)
    }
}

impl std::ops::BitOr for Fields {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOrAssign for Fields {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

/// Copies the `fields` of `from` onto `next`. A zoom without its center
/// keeps its style zoom wherever the center is; a center without its zoom
/// keeps the scale, as a pan does.
fn take(next: &mut CameraState, from: &CameraState, fields: Fields, surface: &Surface) {
    let projection = &surface.projection;
    match (fields.has(Fields::CENTER), fields.has(Fields::ZOOM)) {
        (true, true) => {
            next.target = from.target;
            next.resolution = from.resolution;
        }
        (true, false) if next.target != from.target => {
            let world_per_px = next.world_per_px(projection);
            next.target = from.target;
            if !next.on_sphere(surface) {
                next.resolution = world_per_px / projection.scale_factor(next.target);
            }
        }
        (false, true) if next.target == from.target => next.resolution = from.resolution,
        (false, true) => next.set_zoom(projection, from.zoom(projection)),
        _ => {}
    }
    if fields.has(Fields::BEARING) {
        next.bearing = from.bearing;
    }
    if fields.has(Fields::PITCH) {
        next.pitch = from.pitch;
    }
    if fields.has(Fields::ROLL) {
        next.roll = from.roll;
    }
    if fields.has(Fields::FOV) {
        next.fov = from.fov;
    }
    if fields.has(Fields::ELEVATION) {
        next.elevation = from.elevation;
    }
    if fields.has(Fields::PADDING) {
        next.padding = from.padding;
    }
}

#[derive(Debug)]
struct Running {
    id: AnimationId,
    motion: Motion,
    /// The properties it still moves.
    fields: Fields,
    /// Properties that go together: a later move that takes one takes
    /// them all (a flight's center and zoom, or whatever turns about a
    /// point).
    coupled: Fields,
    /// When it started: `None` until the first frame after the call.
    started: Option<f64>,
    /// Milliseconds in at the last frame.
    elapsed: f64,
    user: bool,
    /// The move ends on the terrain under its destination: it wasn't
    /// given an elevation of its own, or the target is clamped.
    aims: bool,
}

/// A map's camera, its limits and its animations.
#[derive(Debug)]
pub struct Camera {
    state: CameraState,
    surface: Surface,
    viewport: Viewport,
    constraints: Constraints,
    /// The animations running, oldest first: each frame applies them in
    /// turn, each to the properties it moves.
    running: Vec<Running>,
    next_id: u64,
    /// User input is moving the camera.
    held: bool,
    /// The state the last frame's events reported.
    reported: CameraState,
    /// Which changes have started and not ended, in [`CHANGES`] order.
    active: [bool; 4],
    user: bool,
    events: Vec<CameraEvent>,
    modified: bool,
    revision: u64,
    /// The map's terrain, which the target stands on.
    terrain: Option<Arc<dyn Elevation>>,
    /// A gesture held the target's elevation: putting it back on the
    /// terrain keeps the eye in place.
    elevation_held: bool,
    /// The target stands on the terrain (GL JS `centerClampedToGround`).
    clamped: bool,
    /// Drags and their glides on the sphere turn the bearing with the
    /// planet.
    globe_free_bearing: bool,
}

impl Camera {
    pub fn new(state: CameraState, surface: Surface, viewport: Viewport) -> Self {
        Self {
            state,
            surface,
            viewport,
            constraints: Constraints::default(),
            running: Vec::new(),
            next_id: 0,
            held: false,
            reported: state,
            active: [false; 4],
            user: false,
            events: Vec::new(),
            modified: false,
            revision: 0,
            terrain: None,
            elevation_held: false,
            clamped: true,
            globe_free_bearing: false,
        }
    }

    /// Stands the target on `terrain`, or on the surface without it. Call
    /// it again when the terrain's DEM data changes (`Map::elevation`).
    pub fn set_terrain(&mut self, terrain: Option<Arc<dyn Elevation>>) {
        if terrain.is_none() && self.clamped {
            self.state.elevation = 0.0;
        }
        self.terrain = terrain;
        self.write(self.state);
    }

    /// Whether the target stands on the terrain at rest (GL JS
    /// `centerClampedToGround`, on by default). Off, it keeps the elevation
    /// [`set_center_elevation`](Self::set_center_elevation) gives it, and
    /// the camera orbits a point that high.
    pub fn set_center_clamped_to_ground(&mut self, clamped: bool) {
        self.clamped = clamped;
        self.write(self.state);
    }

    pub fn center_clamped_to_ground(&self) -> bool {
        self.clamped
    }

    /// Draws the map in MapLibre Native's axonometric projection, leaning
    /// heights by `skew`, or in perspective (`None`). Moves and animations
    /// keep it.
    pub fn set_axonometric(&mut self, skew: Option<Skew>) {
        if self.state.axonometric != skew {
            self.state.axonometric = skew;
            self.revision += 1;
        }
    }

    /// Puts the target this many metres above the surface at once (GL JS
    /// `setCenterElevation`). While the target is clamped to the ground,
    /// the terrain takes it back.
    pub fn set_center_elevation(&mut self, elevation: f64) {
        self.stop();
        self.modified = true;
        self.user = false;
        self.write(CameraState {
            elevation,
            ..self.state
        });
    }

    pub fn state(&self) -> &CameraState {
        &self.state
    }

    /// Counts changes to the state, so hosts copy it only when it moved.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The app or the user has moved the camera since it was created. A
    /// style's camera applies only to an unmodified one, as in GL JS.
    pub fn modified(&self) -> bool {
        self.modified
    }

    pub fn viewport(&self) -> Viewport {
        self.viewport
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    /// The surface's plane, which style zoom is measured on.
    pub fn projection(&self) -> &Projection {
        &self.surface.projection
    }

    /// The view's style zoom on the camera's plane (see
    /// [`CameraState::zoom`]).
    pub fn zoom(&self) -> f64 {
        self.state.zoom(&self.surface.projection)
    }

    /// Moves to a new surface (a style loaded, or the map's projection
    /// changed): the target and ground resolution stay, running
    /// animations stop, and the camera keeps to the new world.
    pub fn set_surface(&mut self, surface: Surface) {
        if surface == self.surface {
            return;
        }
        self.surface = surface;
        self.stop();
        self.write(self.state);
    }

    /// Resizes the viewport, keeping the camera to its constraints.
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport = viewport;
        self.write(self.state);
    }

    pub fn set_constraints(&mut self, constraints: Constraints) {
        self.constraints = constraints;
        self.write(self.state);
    }

    /// Whether drags and their glides on the sphere turn the bearing with
    /// the planet, so the ground under the pointer stays under it and a
    /// drag carries on over a pole, rather than hold the bearing as GL JS
    /// does (the default; MapLibre Native's globe free bearing).
    pub fn set_globe_free_bearing(&mut self, free: bool) {
        self.globe_free_bearing = free;
    }

    pub fn globe_free_bearing(&self) -> bool {
        self.globe_free_bearing
    }

    pub fn view(&self) -> View {
        self.view_of(&self.state)
    }

    /// The view `camera` would have in this one's viewport, on its surface
    /// and terrain.
    pub(crate) fn view_of(&self, camera: &CameraState) -> View {
        match self.terrain.as_deref() {
            Some(terrain) => View::over(camera, self.viewport, &self.surface, terrain),
            None => View::new(camera, self.viewport, &self.surface),
        }
    }

    /// The logical pixel showing `point`, on the terrain if there is one,
    /// or `None` behind the eye.
    pub fn project(&self, point: Geodetic) -> Option<DVec2> {
        self.project_with(&self.state, point)
    }

    /// [`project`](Self::project) for `camera` in this one's viewport.
    pub fn project_with(&self, camera: &CameraState, point: Geodetic) -> Option<DVec2> {
        let view = self.view_of(camera);
        let world = self.surface.projection.forward(point);
        // The copy of a wrapping world nearest the view's centre.
        let world = self
            .surface
            .projection
            .nearest_copy(world, view.camera_world());
        match self.terrain.as_deref() {
            Some(terrain) => view.project_3d(world, terrain.at(world).unwrap_or_default()),
            None => view.project(world),
        }
    }

    /// The ground point under a logical pixel, on the terrain if there is
    /// one (a CPU ray walk over the elevations this camera has: on the web
    /// ui role, the approximate elevation pyramid), or `None` above the
    /// horizon.
    pub fn unproject(&self, px: DVec2) -> Option<Geodetic> {
        self.unproject_with(&self.state, px)
    }

    /// [`unproject`](Self::unproject) for `camera` in this one's viewport.
    pub fn unproject_with(&self, camera: &CameraState, px: DVec2) -> Option<Geodetic> {
        let view = self.view_of(camera);
        let world = match self.terrain.as_deref() {
            Some(terrain) => view.unproject_terrain(px, terrain).map(|(world, _)| world),
            None => view.unproject(px),
        };
        world.map(|world| self.surface.projection.inverse(world))
    }

    /// Where a jump to `target` would put the camera, without moving it.
    pub fn state_for(&self, target: &CameraTarget) -> CameraState {
        self.resolve(target).0
    }

    /// The camera's eye and orientation now.
    pub fn free(&self) -> FreeCamera {
        self.state.free(self.viewport, &self.surface)
    }

    /// Moves the eye to `free` at once (GL JS `setFreeCameraOptions`),
    /// keeping the field of view and padding.
    pub fn jump_to_free(&mut self, free: &FreeCamera) {
        let state = self.state.with_free(free, self.viewport, &self.surface);
        self.jump_to_state(state);
    }

    /// Moves to `target` at once (GL JS `jumpTo`). Animations go on
    /// moving the properties it doesn't set, as in MapLibre Native.
    pub fn jump_to(&mut self, target: &CameraTarget) {
        let (to, around) = self.resolve(target);
        self.yield_fields(Self::claims(target, around.is_some(), self.aims(target)).0);
        self.modified = true;
        self.user = false;
        self.write(to);
    }

    /// Moves to `state` at once, as [`jump_to`](Self::jump_to) does, for
    /// hosts that hold exact cameras: fixtures, benches, still output.
    pub fn jump_to_state(&mut self, state: CameraState) {
        self.stop();
        self.modified = true;
        self.user = false;
        self.write(state);
    }

    /// Animates to `target` (GL JS `easeTo`), starting at `start` on the
    /// host clock in milliseconds, or with `None` at the next
    /// [`frame`](Self::frame)'s time.
    pub fn ease_to(
        &mut self,
        target: &CameraTarget,
        options: EaseOptions,
        start: Option<f64>,
    ) -> AnimationId {
        self.ease(target, options, start, false)
    }

    /// [`ease_to`](Self::ease_to) for input handlers: its events say the
    /// user moved the camera.
    pub(crate) fn user_ease(
        &mut self,
        target: &CameraTarget,
        options: EaseOptions,
        now: f64,
    ) -> AnimationId {
        self.ease(target, options, Some(now), true)
    }

    fn ease(
        &mut self,
        target: &CameraTarget,
        options: EaseOptions,
        start: Option<f64>,
        user: bool,
    ) -> AnimationId {
        let (to, around) = self.resolve(target);
        let aims = self.aims(target);
        let (fields, coupled) = match user {
            // Input takes the whole camera.
            true => (Fields::ALL, Fields::ALL),
            false => Self::claims(target, around.is_some(), aims),
        };
        self.yield_fields(fields);
        let motion = Motion::ease(self.state, to, around, options, &self.surface);
        self.start(motion, fields, coupled, start, user, aims)
    }

    /// The properties a move to `target` takes, and those of them that go
    /// together: whatever turns about a point goes with the center.
    fn claims(target: &CameraTarget, around: bool, aims: bool) -> (Fields, Fields) {
        let fields = Fields::of(target, aims);
        match around {
            true => (fields | Fields::CENTER, fields | Fields::CENTER),
            false => (fields, Fields::NONE),
        }
    }

    /// Takes `fields` from the animations running: the ones left moving
    /// nothing end interrupted.
    fn yield_fields(&mut self, fields: Fields) {
        if fields.is_empty() {
            return;
        }
        let mut i = 0;
        while i < self.running.len() {
            let running = &mut self.running[i];
            let mut taken = fields;
            if running.coupled.has(taken) {
                taken |= running.coupled;
            }
            running.fields = running.fields.without(taken);
            if let Motion::Momentum(momentum) = &mut running.motion {
                running.fields = momentum.retain(running.fields);
            }
            if running.fields.is_empty() {
                let running = self.running.remove(i);
                self.end(running, true);
            } else {
                i += 1;
            }
        }
    }

    /// Flies to `target` along a zoom-out curve (GL JS `flyTo`), starting
    /// as [`ease_to`](Self::ease_to) does.
    pub fn fly_to(
        &mut self,
        target: &CameraTarget,
        options: &FlyOptions,
        start: Option<f64>,
    ) -> AnimationId {
        let (to, _) = self.resolve(target);
        let aims = self.aims(target);
        // A flight's center and zoom move along one curve, and the
        // height it lands at with them.
        let path = match aims {
            true => Fields::CENTER | Fields::ZOOM | Fields::ELEVATION,
            false => Fields::CENTER | Fields::ZOOM,
        };
        let fields = Fields::of(target, aims) | path;
        self.yield_fields(fields);
        let motion = Motion::fly(
            self.state,
            to,
            options,
            self.constraints.min_zoom,
            &self.surface,
            self.viewport,
        );
        self.start(motion, fields, path, start, false, aims)
    }

    /// The camera that shows `bounds` (GL JS `cameraForBounds`), for
    /// [`jump_to`](Self::jump_to), [`ease_to`](Self::ease_to) or
    /// [`fly_to`](Self::fly_to); `None` if the padding leaves no room.
    pub fn camera_for_bounds(&self, bounds: Bounds, options: &FitOptions) -> Option<CameraTarget> {
        let mut target = self
            .state
            .for_bounds(bounds, options, &self.surface, self.viewport)?;
        if let (Some(center), Some(zoom)) = (target.center, target.zoom) {
            let mut at = CameraState {
                target: center,
                ..self.state
            };
            at.set_zoom(&self.surface.projection, zoom);
            at.limit_zoom(
                f64::NEG_INFINITY,
                self.constraints.max_zoom,
                &self.surface.projection,
            );
            target.zoom = Some(at.zoom(&self.surface.projection));
        }
        Some(target)
    }

    /// Continues a user's move with a glide (smooth wheel zoom). A
    /// glide started during another adds what that one had left, and
    /// starts at the speed it had. Its zoom stops at the zoom range, as GL
    /// JS's wheel target does: levels past a limit would hold the zoom
    /// there once the scroll turned back, until as many came back.
    pub(crate) fn glide(&mut self, mut glide: Glide, now: f64) -> AnimationId {
        if let Some(rest) = self
            .running
            .iter()
            .find_map(|r| r.motion.glide_rest(r.elapsed))
        {
            glide.continue_from(&rest);
        }
        if let Some(track) = &mut glide.zoom {
            let zoom = self.state.zoom(&self.surface.projection);
            let least = self.state.least_zoom(&self.constraints, &self.surface);
            let most = self.constraints.max_zoom.max(least);
            // A zoom outside the range may still move toward it.
            track.amount = track
                .amount
                .clamp((least - zoom).min(0.0), (most - zoom).max(0.0));
        }
        self.stop();
        let motion = Motion::glide(glide);
        self.start(motion, Fields::ALL, Fields::ALL, Some(now), true, false)
    }

    /// Continues a gesture with `momentum` from host time `start`
    /// (milliseconds, the release's), or from the next frame without one.
    /// It takes the properties it moves from the animations running, and
    /// a later move that takes one of them stops the parts that need it.
    /// Returns `None` when nothing moves and nothing settles over time;
    /// snaps that settle at once without motion apply here.
    pub fn start_momentum(
        &mut self,
        momentum: &GestureMomentum,
        start: Option<f64>,
    ) -> Option<AnimationId> {
        let mut next = self.state;
        let stage = Stage {
            surface: &self.surface,
            viewport: self.viewport,
            constraints: &self.constraints,
            free_bearing: self.globe_free_bearing,
        };
        let planned = Momentum::new(momentum, &mut next, &stage);
        let fields = match &planned {
            Some(planned) => planned.fields(),
            None => Fields::between(&self.state, &next),
        };
        self.yield_fields(fields);
        if next != self.state {
            self.user_write(next);
        }
        let motion = Motion::Momentum(Box::new(planned?));
        Some(self.start(motion, fields, Fields::NONE, start, true, false))
    }

    /// Stops every running animation where it is, as interrupted.
    pub fn stop(&mut self) {
        self.interrupt();
    }

    /// Stops the animation `id`, if it runs, as interrupted: the others go
    /// on.
    pub fn stop_animation(&mut self, id: AnimationId) {
        if let Some(i) = self.running.iter().position(|r| r.id == id) {
            let running = self.running.remove(i);
            self.end(running, true);
        }
    }

    /// Stops every running animation as interrupted, returning how fast a
    /// momentum among them still moved.
    fn interrupt(&mut self) -> Option<MomentumVelocity> {
        let mut velocity = None;
        for running in std::mem::take(&mut self.running) {
            velocity = velocity.or(self.end(running, true));
        }
        velocity
    }

    /// Queues the end of an animation taken from the running ones, and
    /// returns how fast it still moved if it is a momentum. A momentum
    /// that settles after a later move stopped part of it reports itself
    /// interrupted too.
    fn end(&mut self, running: Running, interrupted: bool) -> Option<MomentumVelocity> {
        let (velocity, cut) = match &running.motion {
            Motion::Momentum(momentum) => (Some(momentum.velocity()), momentum.replaced()),
            _ => (None, false),
        };
        self.events.push(CameraEvent::AnimationEnd {
            id: running.id,
            interrupted: interrupted || cut,
            velocity: match interrupted {
                true => velocity.unwrap_or_default(),
                false => MomentumVelocity::default(),
            },
        });
        velocity
    }

    pub fn is_animating(&self) -> bool {
        !self.running.is_empty()
    }

    /// User input holds the camera: movement continues, frame to frame,
    /// until [`release`](Self::release). Stops any animation, and returns
    /// how fast a momentum it stopped still moved.
    pub fn hold(&mut self) -> Option<MomentumVelocity> {
        let velocity = self.interrupt();
        self.held = true;
        self.user = true;
        velocity
    }

    pub fn release(&mut self) {
        self.held = false;
        self.write(self.state);
    }

    pub fn is_held(&self) -> bool {
        self.held
    }

    /// Whether the camera moves: animating, held, or with a change whose
    /// end the next frame reports, as a release leaves one. From a
    /// change's start event to its end event, as GL JS's `isMoving`.
    pub fn is_moving(&self) -> bool {
        self.is_animating() || self.held || self.active.contains(&true)
    }

    /// Moves the ground under `from` to `to` (logical pixels).
    pub fn pan(&mut self, from: DVec2, to: DVec2) {
        let next = self.panned(from, to);
        self.user_write(next);
    }

    /// The camera [`pan`](Self::pan) moves to, before the constraints.
    pub fn panned(&self, from: DVec2, to: DVec2) -> CameraState {
        let mut next = self.state;
        let free = self.globe_free_bearing;
        next.pan_with(&self.surface, self.viewport, from, to, free);
        next
    }

    /// Zooms by `delta` levels around a pixel.
    pub fn zoom_by(&mut self, at: DVec2, delta: f64) {
        let mut next = self.state;
        let least = next.least_zoom(&self.constraints, &self.surface);
        let zoom = (next.zoom(&self.surface.projection) + delta)
            .clamp(least, self.constraints.max_zoom.max(least));
        next.zoom_around(&self.surface, self.viewport, at, zoom);
        self.user_write(next);
    }

    /// Turns the map clockwise by `delta` degrees around a pixel.
    pub(crate) fn rotate_by(&mut self, at: DVec2, delta: f64) {
        let mut next = self.state;
        next.rotate_around(&self.surface, self.viewport, at, next.bearing + delta);
        self.user_write(next);
    }

    /// Tilts by `delta` degrees around the viewport's middle.
    pub(crate) fn pitch_by(&mut self, delta: f64) {
        let mut next = self.state;
        next.pitch += delta;
        self.user_write(next);
    }

    /// Advances the running animations to `now`, then queues this frame's
    /// events. Returns whether the camera is still moving on its own, so
    /// the host keeps drawing frames.
    pub fn frame(&mut self, now: f64) -> bool {
        if self.running.is_empty() {
            self.report();
            return false;
        }
        let mut next = self.state;
        let mut user = false;
        let stage = Stage {
            surface: &self.surface,
            viewport: self.viewport,
            constraints: &self.constraints,
            free_bearing: self.globe_free_bearing,
        };
        for running in &mut self.running {
            // An app's move ends on the terrain under its destination,
            // known once its DEM data lands; a gesture holds the
            // elevation.
            if let Some(terrain) = self
                .terrain
                .as_deref()
                .filter(|_| !running.user && running.aims)
                && let Some(ground) = running.motion.destination().and_then(|w| terrain.at(w))
            {
                running.motion.aim_elevation(ground.0, running.elapsed);
            }
            let elapsed = now - *running.started.get_or_insert(now);
            running.elapsed = elapsed;
            user |= running.user;
            match &mut running.motion {
                Motion::Momentum(momentum) => momentum.apply(&mut next, elapsed, &stage),
                // A glide moves the camera on from where it is, and owns
                // all of it.
                motion @ Motion::Glide { .. } => {
                    motion.apply(&mut next, elapsed, stage.surface, stage.viewport);
                }
                motion => {
                    let mut at = next;
                    motion.apply(&mut at, elapsed, stage.surface, stage.viewport);
                    take(&mut next, &at, running.fields, stage.surface);
                }
            }
        }
        self.user = user;
        // Written while the moves still run, so their elevation stays
        // their own.
        self.write(next);
        let mut ended = false;
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].elapsed >= self.running[i].motion.duration() {
                let running = self.running.remove(i);
                self.end(running, false);
                ended = true;
            } else {
                i += 1;
            }
        }
        if ended && self.running.is_empty() {
            // At rest: on the terrain.
            self.write(self.state);
        }
        self.report();
        !self.running.is_empty()
    }

    /// The events queued by frames since the last call.
    pub fn take_events(&mut self) -> Vec<CameraEvent> {
        std::mem::take(&mut self.events)
    }

    fn report(&mut self) {
        let (a, b) = (self.reported, self.state);
        let changed = [
            a != b,
            a.zoom(&self.surface.projection) != b.zoom(&self.surface.projection),
            a.bearing != b.bearing,
            a.pitch != b.pitch,
        ];
        let user = self.user;
        for (i, change) in CHANGES.into_iter().enumerate() {
            if changed[i] && !self.active[i] {
                self.active[i] = true;
                self.events.push(CameraEvent::Changed {
                    change,
                    phase: Phase::Start,
                    user,
                });
            }
        }
        for (i, change) in CHANGES.into_iter().enumerate() {
            if changed[i] {
                self.events.push(CameraEvent::Changed {
                    change,
                    phase: Phase::Progress,
                    user,
                });
            }
        }
        if self.running.is_empty() && !self.held {
            // GL JS ends zoom, rotate and pitch before the move.
            for i in [1, 2, 3, 0] {
                if std::mem::take(&mut self.active[i]) {
                    self.events.push(CameraEvent::Changed {
                        change: CHANGES[i],
                        phase: Phase::End,
                        user,
                    });
                }
            }
        }
        self.reported = self.state;
    }

    fn start(
        &mut self,
        motion: Motion,
        fields: Fields,
        coupled: Fields,
        start: Option<f64>,
        user: bool,
        aims: bool,
    ) -> AnimationId {
        self.next_id += 1;
        let id = AnimationId(self.next_id);
        self.modified = true;
        self.user = user;
        self.running.push(Running {
            id,
            motion,
            fields,
            coupled,
            started: start,
            elapsed: 0.0,
            user,
            aims,
        });
        id
    }

    fn user_write(&mut self, next: CameraState) {
        self.modified = true;
        self.user = true;
        self.write(next);
    }

    /// Writes a constrained state. Moves outside the limits are clamped,
    /// whoever makes them.
    fn write(&mut self, mut next: CameraState) {
        // Only `set_axonometric` changes the projection.
        next.axonometric = self.state.axonometric;
        next.bearing = wrap_bearing(next.bearing);
        next.constrain(&self.constraints, &self.surface, self.viewport);
        if let Some(terrain) = self.terrain.as_deref().filter(|_| self.clamped) {
            // Gestures hold the target's elevation, as GL JS's do; at rest
            // it stands on the terrain.
            let gesture = self.held || self.running.iter().any(|r| r.user);
            if gesture {
                self.elevation_held = true;
            } else if self.running.is_empty() {
                if std::mem::take(&mut self.elevation_held) {
                    next.recenter_on(terrain, &self.surface, self.viewport);
                } else {
                    next.settle_on(terrain, &self.surface.projection);
                }
            }
        }
        if let Some(terrain) = self.terrain.as_deref() {
            next.keep_above(terrain, &self.surface, self.viewport);
        }
        if next != self.state {
            self.state = next;
            self.revision += 1;
        }
    }

    /// Whether a move to `target` ends on the terrain under it.
    fn aims(&self, target: &CameraTarget) -> bool {
        self.clamped || target.elevation.is_none()
    }

    /// The end state of a move to `target`, and the pixel and ground point
    /// a zoom or rotation keeps in place.
    fn resolve(&self, target: &CameraTarget) -> (CameraState, Option<(Geodetic, DVec2)>) {
        let (surface, viewport) = (&self.surface, self.viewport);
        let projection = &surface.projection;
        let mut to = self.state;
        if let Some(fov) = target.fov {
            to.fov = fov.clamp(0.01, 179.0);
        }
        if let Some(padding) = target.padding {
            to.padding = padding;
        }
        let center = to.center(viewport);
        // Limits bound the scale at the destination.
        let zoom = target.zoom.map(|z| {
            let mut at = CameraState {
                target: target.center.unwrap_or(self.state.target),
                ..self.state
            };
            at.set_zoom(projection, z);
            let least = at.least_zoom(&self.constraints, surface);
            at.limit_zoom(least, self.constraints.max_zoom, projection);
            at.zoom(projection)
        });
        let mut around = None;
        match (target.center, target.around) {
            (Some(c), _) => {
                to.target = c;
                if let Some(zoom) = zoom {
                    to.set_zoom(projection, zoom);
                } else if !self.state.on_sphere(surface) {
                    // Keep the scale on screen, as a pan does (on the
                    // sphere, its size: the ground resolution).
                    to.resolution =
                        self.state.world_per_px(projection) / projection.scale_factor(c);
                }
            }
            (None, Some(ground)) => {
                let at = self.project(ground).unwrap_or(center);
                around = Some((ground, at));
                to.holding(surface, viewport, at, |c| {
                    if let Some(zoom) = zoom {
                        c.set_zoom(projection, zoom);
                    }
                });
            }
            (None, None) => {
                if let Some(zoom) = zoom {
                    to.set_zoom(projection, zoom);
                }
            }
        }
        if let Some(bearing) = target.bearing {
            match around {
                Some((_, at)) => to.rotate_around(surface, viewport, at, bearing),
                None => to.bearing = wrap_bearing(bearing),
            }
        }
        if let Some(pitch) = target.pitch {
            to.pitch = pitch;
        }
        if let Some(roll) = target.roll {
            to.roll = wrap_bearing(roll);
        }
        if let Some(elevation) = target.elevation {
            to.elevation = elevation;
        }
        if target.offset != DVec2::ZERO {
            to.pan(surface, viewport, center, center + target.offset);
        }
        to.bearing = wrap_bearing(to.bearing);
        to.constrain(&self.constraints, surface, viewport);
        if let Some(terrain) = self.terrain.as_deref().filter(|_| self.clamped) {
            to.settle_on(terrain, projection);
        }
        (to, around)
    }
}

#[cfg(test)]
mod tests {
    fn mercator() -> orb_crs::Projection {
        *orb_crs::TileMatrixSet::web_mercator().projection()
    }

    use super::*;
    use crate::animate::{Easing, Track};
    use orb_crs::TileMatrixSet;

    fn camera() -> Camera {
        let projection = *TileMatrixSet::web_mercator().projection();
        Camera::new(
            CameraState::new(Geodetic::new(139.7, 35.68), 10.0, &mercator()),
            Surface::from(projection),
            Viewport {
                width: 800.0,
                height: 600.0,
            },
        )
    }

    fn changes(events: &[CameraEvent]) -> Vec<(Change, Phase)> {
        events
            .iter()
            .filter_map(|e| match e {
                CameraEvent::Changed { change, phase, .. } => Some((*change, *phase)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_jump_starts_moves_and_ends_in_one_frame() {
        let mut camera = camera();
        camera.jump_to(&CameraTarget {
            zoom: Some(12.0),
            ..CameraTarget::default()
        });
        assert!(!camera.frame(0.0));
        assert_eq!(
            changes(&camera.take_events()),
            [
                (Change::Move, Phase::Start),
                (Change::Zoom, Phase::Start),
                (Change::Move, Phase::Progress),
                (Change::Zoom, Phase::Progress),
                (Change::Zoom, Phase::End),
                (Change::Move, Phase::End),
            ]
        );
        // Nothing changes, nothing is reported.
        assert!(!camera.frame(16.0));
        assert!(camera.take_events().is_empty());
    }

    #[test]
    fn eases_roll_the_short_way_and_take_an_elevation() {
        let mut camera = camera();
        camera.jump_to(&CameraTarget {
            roll: Some(170.0),
            ..CameraTarget::default()
        });
        assert_eq!(camera.state().roll, 170.0);
        camera.ease_to(
            &CameraTarget {
                roll: Some(-170.0),
                elevation: Some(300.0),
                ..CameraTarget::default()
            },
            EaseOptions {
                duration: 100.0,
                easing: Easing::Linear,
            },
            Some(0.0),
        );
        camera.frame(50.0);
        // Halfway across the seam, not back through 0°.
        assert!((camera.state().roll.abs() - 180.0).abs() < 1e-9);
        assert!((camera.state().elevation - 150.0).abs() < 1e-9);
        camera.frame(100.0);
        assert_eq!(camera.state().roll, -170.0);
        // Without terrain nothing takes the elevation back.
        assert_eq!(camera.state().elevation, 300.0);
    }

    /// On the globe the least zoom bounds its size at any latitude, for
    /// wheel zooms and eases too: the least globe, dragged north, zooms
    /// in by the step asked for rather than jumping to the least style
    /// zoom, and zooming or easing out there stops at the same size.
    #[test]
    fn globe_zooms_keep_the_least_size() {
        let surface = Surface {
            projection: mercator(),
            globe: Some(orb_crs::Globe::V8),
        };
        let viewport = Viewport {
            width: 800.0,
            height: 600.0,
        };
        let start = CameraState::new(Geodetic::new(0.0, 0.0), 2.0, &mercator());
        let mut camera = Camera::new(start, surface, viewport);
        camera.set_constraints(Constraints {
            min_zoom: 1.0,
            ..Constraints::default()
        });
        let middle = DVec2::new(400.0, 300.0);
        camera.zoom_by(middle, -10.0);
        assert!((camera.state().globe_zoom() - 1.0).abs() < 1e-9);
        for _ in 0..10 {
            camera.pan(middle, middle + DVec2::new(0.0, 20.0));
        }
        assert!(camera.state().target.lat > 65.0);
        assert!((camera.state().globe_zoom() - 1.0).abs() < 1e-9);
        camera.zoom_by(middle, 0.1);
        assert!((camera.state().globe_zoom() - 1.1).abs() < 1e-3);
        camera.zoom_by(middle, -1.0);
        assert!((camera.state().globe_zoom() - 1.0).abs() < 1e-9);
        camera.ease_to(
            &CameraTarget {
                zoom: Some(-5.0),
                ..CameraTarget::default()
            },
            EaseOptions {
                duration: 0.0,
                easing: Easing::Linear,
            },
            Some(0.0),
        );
        camera.frame(0.0);
        assert!((camera.state().globe_zoom() - 1.0).abs() < 1e-9);
    }

    /// On the globe, eases and flights that don't set a zoom keep its size
    /// all the way (GL JS's zoom adjustment), and one to a pole goes there
    /// over the sphere, latitude moving steadily.
    #[test]
    fn globe_eases_keep_its_size() {
        let surface = Surface {
            projection: mercator(),
            globe: Some(orb_crs::Globe::V8),
        };
        let viewport = Viewport {
            width: 800.0,
            height: 800.0,
        };
        let start = CameraState::new(Geodetic::new(0.0, 0.0), 2.0, &mercator());
        for fly in [false, true] {
            let mut camera = Camera::new(start, surface, viewport);
            let target = CameraTarget {
                center: Some(Geodetic::new(60.0, 50.0)),
                ..CameraTarget::default()
            };
            let options = EaseOptions {
                duration: 100.0,
                easing: Easing::Linear,
            };
            match fly {
                false => camera.ease_to(&target, options, Some(0.0)),
                true => camera.fly_to(
                    &target,
                    &FlyOptions {
                        duration: Some(100.0),
                        ..FlyOptions::default()
                    },
                    Some(0.0),
                ),
            };
            let mut t = 0.0;
            while camera.frame(t) {
                if !fly {
                    assert_eq!(camera.state().resolution, start.resolution);
                }
                t += 10.0;
            }
            let end = camera.state();
            assert!((end.resolution / start.resolution - 1.0).abs() < 1e-12);
            assert!((end.target.lat - 50.0).abs() < 1e-9);
        }
        let mut camera = Camera::new(start, surface, viewport);
        camera.ease_to(
            &CameraTarget {
                center: Some(Geodetic::new(0.0, 90.0)),
                zoom: Some(1.0),
                ..CameraTarget::default()
            },
            EaseOptions {
                duration: 100.0,
                easing: Easing::Linear,
            },
            Some(0.0),
        );
        camera.frame(50.0);
        assert!((camera.state().target.lat - 45.0).abs() < 1e-9);
        camera.frame(100.0);
        assert_eq!(camera.state().target.lat, 90.0);
    }

    #[test]
    fn eases_arrive_and_report_every_frame() {
        let mut camera = camera();
        let to = Geodetic::new(139.8, 35.6);
        let id = camera.ease_to(
            &CameraTarget {
                center: Some(to),
                zoom: Some(11.0),
                bearing: Some(90.0),
                ..CameraTarget::default()
            },
            EaseOptions::default(),
            Some(1000.0),
        );
        let mut frames = 0;
        let mut t = 1000.0;
        while camera.frame(t) {
            frames += 1;
            t += 16.0;
        }
        assert!((30..=33).contains(&frames), "{frames}");
        let state = camera.state();
        assert!((state.target.lon - to.lon).abs() < 1e-9);
        assert!((state.zoom(&mercator()) - 11.0).abs() < 1e-9);
        assert_eq!(state.bearing, 90.0);
        let events = camera.take_events();
        assert!(events.contains(&CameraEvent::AnimationEnd {
            id,
            interrupted: false,
            velocity: Default::default(),
        }));
        let progress = changes(&events)
            .iter()
            .filter(|c| **c == (Change::Move, Phase::Progress))
            .count();
        assert_eq!(progress, frames + 1);
        assert_eq!(changes(&events).last(), Some(&(Change::Move, Phase::End)));
    }

    #[test]
    fn an_ease_without_a_start_starts_at_the_next_frame() {
        let mut camera = camera();
        let start = *camera.state();
        let target = CameraTarget {
            zoom: Some(11.0),
            ..CameraTarget::default()
        };
        let options = EaseOptions {
            duration: 100.0,
            easing: crate::Easing::Linear,
        };
        camera.ease_to(&target, options, None);
        // Called long after the last frame: the first frame is its start.
        assert!(camera.frame(5000.0));
        let zoom = camera.state().zoom(&mercator());
        assert!((zoom - start.zoom(&mercator())).abs() < 1e-9, "{zoom}");
        assert!(camera.frame(5050.0));
        let halfway = camera.state().zoom(&mercator());
        assert!((halfway - (start.zoom(&mercator()) + 11.0) / 2.0).abs() < 1e-9);
        assert!(!camera.frame(5100.0));
    }

    #[test]
    fn user_input_interrupts_an_animation() {
        let mut camera = camera();
        let id = camera.fly_to(
            &CameraTarget {
                center: Some(Geodetic::new(-74.0, 40.7)),
                zoom: Some(12.0),
                ..CameraTarget::default()
            },
            &FlyOptions::default(),
            Some(0.0),
        );
        assert!(camera.frame(100.0));
        camera.hold();
        camera.pan(DVec2::new(400.0, 300.0), DVec2::new(420.0, 300.0));
        assert!(!camera.frame(116.0));
        let events = camera.take_events();
        assert!(events.contains(&CameraEvent::AnimationEnd {
            id,
            interrupted: true,
            velocity: Default::default(),
        }));
        // Held: the move hasn't ended.
        assert!(!changes(&events).contains(&(Change::Move, Phase::End)));
        camera.release();
        // Until the frame that reports its end, the move goes on.
        assert!(camera.is_moving());
        camera.frame(132.0);
        assert_eq!(
            changes(&camera.take_events()),
            [(Change::Zoom, Phase::End), (Change::Move, Phase::End)]
        );
        assert!(!camera.is_moving());
    }

    fn ends(events: &[CameraEvent]) -> Vec<(AnimationId, bool)> {
        events
            .iter()
            .filter_map(|e| match e {
                CameraEvent::AnimationEnd {
                    id, interrupted, ..
                } => Some((*id, *interrupted)),
                _ => None,
            })
            .collect()
    }

    /// A move takes the properties it sets from the animations running,
    /// which go on moving the rest; one left with nothing ends
    /// interrupted, as MapLibre Native's transitions do.
    #[test]
    fn moves_take_only_the_properties_they_set() {
        let mut camera = camera();
        let linear = EaseOptions {
            duration: 100.0,
            easing: Easing::Linear,
        };
        let to = Geodetic::new(139.8, 35.6);
        let ease = camera.ease_to(
            &CameraTarget {
                center: Some(to),
                bearing: Some(90.0),
                ..CameraTarget::default()
            },
            linear,
            Some(0.0),
        );
        let zoom = camera.ease_to(
            &CameraTarget {
                zoom: Some(12.0),
                ..CameraTarget::default()
            },
            linear,
            Some(0.0),
        );
        camera.frame(50.0);
        assert!((camera.state().bearing - 45.0).abs() < 1e-9);
        assert!((camera.zoom() - 11.0).abs() < 1e-9);
        // A jump of the bearing leaves the center and the zoom moving.
        camera.jump_to(&CameraTarget {
            bearing: Some(10.0),
            ..CameraTarget::default()
        });
        assert!(ends(&camera.take_events()).is_empty());
        camera.frame(75.0);
        assert_eq!(camera.state().bearing, 10.0);
        assert!((camera.zoom() - 11.5).abs() < 1e-9);
        // A zoom ease takes the zoom's whole animation.
        let again = camera.ease_to(
            &CameraTarget {
                zoom: Some(10.0),
                ..CameraTarget::default()
            },
            linear,
            Some(75.0),
        );
        assert_eq!(ends(&camera.take_events()), [(zoom, true)]);
        // A jump of the center ends the ease left with nothing else.
        camera.jump_to(&CameraTarget {
            center: Some(to),
            ..CameraTarget::default()
        });
        assert_eq!(ends(&camera.take_events()), [(ease, true)]);
        assert!(!camera.frame(175.0));
        assert_eq!(ends(&camera.take_events()), [(again, false)]);
        assert_eq!(camera.state().target, to);
        assert!((camera.zoom() - 10.0).abs() < 1e-9);
    }

    /// A flight's center and zoom go together, and stopping one animation
    /// leaves the others.
    #[test]
    fn flights_yield_their_path_whole() {
        let mut camera = camera();
        let flight = camera.fly_to(
            &CameraTarget {
                center: Some(Geodetic::new(-74.0, 40.7)),
                ..CameraTarget::default()
            },
            &FlyOptions::default(),
            Some(0.0),
        );
        let pitch = camera.ease_to(
            &CameraTarget {
                pitch: Some(40.0),
                ..CameraTarget::default()
            },
            EaseOptions::default(),
            Some(0.0),
        );
        camera.frame(10.0);
        camera.jump_to(&CameraTarget {
            zoom: Some(3.0),
            ..CameraTarget::default()
        });
        assert_eq!(ends(&camera.take_events()), [(flight, true)]);
        assert!(camera.is_animating());
        camera.stop_animation(pitch);
        assert_eq!(ends(&camera.take_events()), [(pitch, true)]);
        assert!(!camera.is_animating());
    }

    /// A move that takes what a momentum part needs stops that part; the
    /// momentum ends interrupted once its last part does.
    #[test]
    fn moves_stop_the_momentum_parts_they_take() {
        let mut camera = camera();
        let friction = crate::MomentumDecay {
            model: crate::DecayModel::Friction {
                half_life: 600_000.0,
                stop_speed: 1.0,
            },
            ..crate::MomentumDecay::PAN
        };
        let id = camera
            .start_momentum(
                &GestureMomentum {
                    pan: Some(crate::PanMomentum {
                        decay: friction,
                        ..crate::PanMomentum::new(DVec2::new(1000.0, 0.0))
                    }),
                    zoom: Some(crate::AnchoredMomentum {
                        decay: friction,
                        ..crate::AnchoredMomentum::zoom(Some(1.0))
                    }),
                    ..GestureMomentum::default()
                },
                Some(0.0),
            )
            .expect("it moves");
        camera.frame(10.0);
        camera.jump_to(&CameraTarget {
            zoom: Some(3.0),
            ..CameraTarget::default()
        });
        assert!(ends(&camera.take_events()).is_empty());
        let zoom = camera.zoom();
        let lon = camera.state().target.lon;
        camera.frame(20.0);
        assert_eq!(camera.zoom(), zoom);
        assert!(camera.state().target.lon < lon, "the pan goes on");
        camera.jump_to(&CameraTarget {
            center: Some(Geodetic::new(0.0, 0.0)),
            ..CameraTarget::default()
        });
        assert_eq!(ends(&camera.take_events()), [(id, true)]);
        assert!(!camera.is_animating());
    }

    #[test]
    fn flights_zoom_out_and_arrive() {
        let mut camera = camera();
        let to = Geodetic::new(-74.0, 40.7);
        camera.fly_to(
            &CameraTarget {
                center: Some(to),
                zoom: Some(10.0),
                ..CameraTarget::default()
            },
            &FlyOptions::default(),
            Some(0.0),
        );
        let mut lowest = f64::INFINITY;
        let mut t = 0.0;
        while camera.frame(t) {
            lowest = lowest.min(camera.state().zoom(&mercator()));
            t += 16.0;
        }
        assert!(lowest < 4.0, "{lowest}");
        assert!((camera.state().target.lon - to.lon).abs() < 1e-9);
        assert!(t > 2000.0 && t < 20_000.0, "{t} ms");
    }

    #[test]
    fn flights_that_would_never_end_jump() {
        let to = Geodetic::new(-74.0, 40.7);
        let target = CameraTarget {
            center: Some(to),
            zoom: Some(10.0),
            ..CameraTarget::default()
        };
        let speeds = [0.0, f64::NAN, 1e-300, -1.0];
        let options = speeds
            .map(|speed| FlyOptions {
                speed,
                ..FlyOptions::default()
            })
            .into_iter()
            .chain([
                FlyOptions {
                    screen_speed: Some(0.0),
                    ..FlyOptions::default()
                },
                FlyOptions {
                    duration: Some(f64::INFINITY),
                    ..FlyOptions::default()
                },
            ]);
        for options in options {
            let mut camera = camera();
            camera.fly_to(&target, &options, Some(0.0));
            assert!(!camera.frame(0.0), "{options:?} lands at once");
            assert!((camera.state().target.lon - to.lon).abs() < 1e-9);
            assert!((camera.state().zoom(&mercator()) - 10.0).abs() < 1e-9);
        }
    }

    #[test]
    fn glides_continue_and_compose() {
        let mut camera = camera();
        let start = camera.state().zoom(&mercator());
        let track = Track {
            amount: 1.0,
            duration: 200.0,
            easing: Easing::DEFAULT,
        };
        camera.glide(
            Glide {
                zoom: Some(track),
                around: Some(DVec2::new(100.0, 100.0)),
                ..Glide::default()
            },
            0.0,
        );
        camera.frame(100.0);
        // A second wheel step adds to what's left.
        camera.glide(
            Glide {
                zoom: Some(track),
                around: Some(DVec2::new(100.0, 100.0)),
                ..Glide::default()
            },
            100.0,
        );
        let mut t = 100.0;
        while camera.frame(t) {
            t += 16.0;
        }
        assert!((camera.state().zoom(&mercator()) - (start + 2.0)).abs() < 1e-6);
    }

    #[test]
    fn a_retargeted_glide_keeps_its_speed() {
        let track = Track {
            amount: 1.0,
            duration: 200.0,
            easing: Easing::DEFAULT,
        };
        let glide = Glide {
            zoom: Some(track),
            around: Some(DVec2::new(100.0, 100.0)),
            ..Glide::default()
        };
        // A second wheel notch 20, 50 or 80 ms into the first: restarting
        // `ease` from rest would slow the zoom to its start slope, 0.4×
        // its average speed (2.21 levels per 200 ms down to 0.64 at 50 ms).
        for at in [20.0, 50.0, 80.0] {
            let mut camera = camera();
            let zoom = |camera: &Camera| camera.state().zoom(&mercator());
            let from = zoom(&camera);
            camera.glide(glide, 0.0);
            // Levels per 0.1 ms on either side.
            camera.frame(at - 0.1);
            let before = zoom(&camera);
            camera.frame(at);
            let speed = zoom(&camera) - before;
            camera.glide(glide, at);
            let start = zoom(&camera);
            camera.frame(at + 0.1);
            let after = zoom(&camera) - start;
            assert!(
                (after / speed - 1.0).abs() < 0.02,
                "{at} ms in: {speed} then {after} levels per 0.1 ms"
            );
            let mut t = at + 0.1;
            while camera.frame(t) {
                t += 16.0;
            }
            assert!(
                (zoom(&camera) - (from + 2.0)).abs() < 1e-6,
                "and zooms both"
            );
        }
    }

    #[test]
    fn ease_around_a_point_keeps_it_still() {
        let mut camera = camera();
        let at = DVec2::new(200.0, 150.0);
        let ground = camera.unproject(at).unwrap();
        camera.ease_to(
            &CameraTarget {
                zoom: Some(12.0),
                around: Some(ground),
                ..CameraTarget::default()
            },
            EaseOptions::default(),
            Some(0.0),
        );
        let mut t = 0.0;
        while camera.frame(t) {
            let p = camera.project(ground).unwrap();
            assert!(p.distance(at) < 1e-6, "{p}");
            t += 16.0;
        }
        assert!((camera.state().zoom(&mercator()) - 12.0).abs() < 1e-9);
    }

    /// Dragging across the pole of a polar plane keeps the map's scale and
    /// its style zoom: world units per pixel and the zoom
    /// stay put, past a minimum zoom the old latitude-bound zoom would
    /// have crossed, and ground resolution at the target moves only with
    /// the plane's own scale factor (a rigid drag can't hold both on a
    /// projection whose scale varies).
    #[test]
    fn panning_across_the_pole_keeps_scale_and_zoom() {
        let projection = orb_crs::Projection::from_descriptor("EPSG:3413").unwrap();
        let viewport = Viewport {
            width: 1024.0,
            height: 768.0,
        };
        let start = CameraState::new(Geodetic::new(-45.0, 72.0), 2.0, &projection);
        let mut camera = Camera::new(start, Surface::from(projection), viewport);
        camera.set_constraints(Constraints {
            min_zoom: 1.9,
            ..Constraints::default()
        });
        let world_per_px = camera.state().world_per_px(&projection);
        let zoom = camera.state().zoom(&projection);
        assert!((zoom - 2.0).abs() < 1e-9);
        let mut crossed = false;
        // The pole is straight up the screen from 72°N on -45°: drag down
        // past it and on to 72°N on the far side.
        for _ in 0..40 {
            camera.pan(DVec2::new(512.0, 300.0), DVec2::new(512.0, 340.0));
            let state = camera.state();
            crossed |= state.target.lat > 89.0;
            let ratio = state.world_per_px(&projection) / world_per_px;
            assert!(
                (ratio - 1.0).abs() < 1e-9,
                "scale changed by {ratio} at {state:?}"
            );
            assert!((state.zoom(&projection) - zoom).abs() < 1e-9);
            let k = projection.scale_factor(state.target);
            let k0 = projection.scale_factor(start.target);
            assert!((state.resolution * k / (start.resolution * k0) - 1.0).abs() < 1e-9);
        }
        assert!(crossed && camera.state().target.lon > 100.0);
        // Zoom limits bound the same zoom.
        camera.zoom_by(DVec2::new(512.0, 384.0), -5.0);
        assert!((camera.state().zoom(&projection) - 1.9).abs() < 1e-9);
    }

    #[test]
    fn fits_bounds_within_constraints() {
        let mut camera = camera();
        camera.set_constraints(Constraints {
            max_zoom: 14.0,
            ..Constraints::default()
        });
        let tiny = Bounds {
            west: 139.7,
            south: 35.68,
            east: 139.7001,
            north: 35.6801,
        };
        let target = camera
            .camera_for_bounds(tiny, &FitOptions::default())
            .unwrap();
        assert_eq!(target.zoom, Some(14.0));
    }
}
