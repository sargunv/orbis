//! The camera store: the single writer of a map's camera. Gestures, the
//! app's animations and jumps all move it; the host reads it once a frame
//! and draws what it shows.

use std::sync::Arc;

use glam::DVec2;
use orb_crs::Projection;
use orb_geo::Geodetic;

use crate::animate::{CameraTarget, EaseOptions, FlyOptions, Glide, Motion};
use crate::fit::FitOptions;
use crate::moves::{Bounds, Constraints, wrap_bearing};
use crate::{CameraState, Elevation, View, Viewport};

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
    /// interrupted it.
    AnimationEnd { id: AnimationId, interrupted: bool },
}

const CHANGES: [Change; 4] = [Change::Move, Change::Zoom, Change::Rotate, Change::Pitch];

#[derive(Debug)]
struct Running {
    id: AnimationId,
    motion: Motion,
    started: f64,
    /// Milliseconds in at the last frame.
    elapsed: f64,
    user: bool,
}

/// A map's camera, its limits and its animations.
#[derive(Debug)]
pub struct Camera {
    state: CameraState,
    projection: Projection,
    viewport: Viewport,
    constraints: Constraints,
    running: Option<Running>,
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
}

impl Camera {
    pub fn new(state: CameraState, projection: Projection, viewport: Viewport) -> Self {
        Self {
            state,
            projection,
            viewport,
            constraints: Constraints::default(),
            running: None,
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
        }
    }

    /// Stands the target on `terrain`, or on the surface without it. Call
    /// it again when the terrain's DEM data changes (`Map::elevation`).
    pub fn set_terrain(&mut self, terrain: Option<Arc<dyn Elevation>>) {
        if terrain.is_none() {
            self.state.elevation = 0.0;
        }
        self.terrain = terrain;
        self.write(self.state);
    }

    /// The terrain the camera stands on.
    pub fn terrain(&self) -> Option<&dyn Elevation> {
        self.terrain.as_deref()
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

    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    /// The view's style zoom on the camera's plane (see
    /// [`CameraState::zoom`]).
    pub fn zoom(&self) -> f64 {
        self.state.zoom(&self.projection)
    }

    /// Moves on the surface's new plane (a scene document loaded, or the
    /// map's projection changed): the target and ground resolution stay,
    /// running animations stop, and the camera keeps to the new world.
    pub fn set_projection(&mut self, projection: Projection) {
        if projection == self.projection {
            return;
        }
        self.projection = projection;
        self.running = None;
        self.write(self.state);
    }

    /// Resizes the viewport, keeping the camera to its constraints.
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport = viewport;
        self.write(self.state);
    }

    pub fn constraints(&self) -> &Constraints {
        &self.constraints
    }

    pub fn set_constraints(&mut self, constraints: Constraints) {
        self.constraints = constraints;
        self.write(self.state);
    }

    pub fn view(&self) -> View {
        match self.terrain.as_deref() {
            Some(terrain) => View::over(&self.state, self.viewport, &self.projection, terrain),
            None => View::new(&self.state, self.viewport, &self.projection),
        }
    }

    /// The logical pixel showing `point`, on the terrain if there is one,
    /// or `None` behind the eye.
    pub fn project(&self, point: Geodetic) -> Option<DVec2> {
        let view = self.view();
        let world = self.projection.forward(point);
        // The copy of a wrapping world nearest the view's centre.
        let world = match self.projection.wrap_period() {
            Some(period) => DVec2::new(
                world.x - ((world.x - view.camera_world().x) / period).round() * period,
                world.y,
            ),
            None => world,
        };
        match self.terrain.as_deref() {
            Some(terrain) => {
                let metres = terrain.at(world).unwrap_or(0.0);
                view.project_3d(world, metres * view.height_scale())
            }
            None => view.project(world),
        }
    }

    /// The ground point under a logical pixel, on the terrain if there is
    /// one (a CPU ray walk over the elevations this camera has: on the web
    /// ui role, the approximate elevation pyramid), or `None` above the
    /// horizon.
    pub fn unproject(&self, px: DVec2) -> Option<Geodetic> {
        let view = self.view();
        let world = match self.terrain.as_deref() {
            Some(terrain) => view.unproject_terrain(px, terrain).map(|(world, _)| world),
            None => view.unproject(px),
        };
        world.map(|world| self.projection.inverse(world))
    }

    /// Moves to `target` at once (GL JS `jumpTo`).
    pub fn jump_to(&mut self, target: &CameraTarget) {
        self.stop();
        let (to, _) = self.resolve(target);
        self.modified = true;
        self.user = false;
        self.write(to);
    }

    /// Animates to `target` (GL JS `easeTo`). `now` is the host clock in
    /// milliseconds.
    pub fn ease_to(
        &mut self,
        target: &CameraTarget,
        options: EaseOptions,
        now: f64,
    ) -> AnimationId {
        self.stop();
        let (to, around) = self.resolve(target);
        let motion = Motion::ease(self.state, to, around, options, &self.projection);
        self.start(motion, now, false)
    }

    /// [`ease_to`](Self::ease_to) for input handlers: its events say the
    /// user moved the camera.
    pub fn user_ease(
        &mut self,
        target: &CameraTarget,
        options: EaseOptions,
        now: f64,
    ) -> AnimationId {
        self.stop();
        let (to, around) = self.resolve(target);
        let motion = Motion::ease(self.state, to, around, options, &self.projection);
        self.start(motion, now, true)
    }

    /// Flies to `target` along a zoom-out curve (GL JS `flyTo`).
    pub fn fly_to(&mut self, target: &CameraTarget, options: &FlyOptions, now: f64) -> AnimationId {
        self.stop();
        let (to, _) = self.resolve(target);
        let motion = Motion::fly(
            self.state,
            to,
            options,
            self.constraints.min_zoom,
            &self.projection,
            self.viewport,
        );
        self.start(motion, now, false)
    }

    /// The camera that shows `bounds` (GL JS `cameraForBounds`), for
    /// [`jump_to`](Self::jump_to), [`ease_to`](Self::ease_to) or
    /// [`fly_to`](Self::fly_to); `None` if the padding leaves no room.
    pub fn camera_for_bounds(&self, bounds: Bounds, options: &FitOptions) -> Option<CameraTarget> {
        let mut target = self
            .state
            .for_bounds(bounds, options, &self.projection, self.viewport)?;
        if let (Some(center), Some(zoom)) = (target.center, target.zoom) {
            let mut at = CameraState {
                target: center,
                ..self.state
            };
            at.set_zoom(&self.projection, zoom);
            at.limit_zoom(
                f64::NEG_INFINITY,
                self.constraints.max_zoom,
                &self.projection,
            );
            target.zoom = Some(at.zoom(&self.projection));
        }
        Some(target)
    }

    /// Continues a user's move with a glide (inertia, smooth wheel zoom). A
    /// glide started during another adds what that one had left.
    pub fn glide(&mut self, mut glide: Glide, now: f64) -> AnimationId {
        if let Some(rest) = self
            .running
            .as_ref()
            .and_then(|r| r.motion.remaining_glide())
        {
            let merge = |a: &mut Option<crate::animate::Track<f64>>,
                         b: Option<crate::animate::Track<f64>>| {
                if let (Some(a), Some(b)) = (a.as_mut(), b) {
                    a.amount += b.amount;
                }
            };
            merge(&mut glide.zoom, rest.zoom);
            merge(&mut glide.bearing, rest.bearing);
            merge(&mut glide.pitch, rest.pitch);
            if let (Some(a), Some(b)) = (glide.pan.as_mut(), rest.pan) {
                a.amount += b.amount;
            }
        }
        self.stop();
        self.start(Motion::glide(glide), now, true)
    }

    /// Stops the running animation, if any, as interrupted.
    pub fn stop(&mut self) {
        if let Some(running) = self.running.take() {
            self.events.push(CameraEvent::AnimationEnd {
                id: running.id,
                interrupted: true,
            });
        }
    }

    pub fn is_animating(&self) -> bool {
        self.running.is_some()
    }

    /// User input holds the camera: movement continues, frame to frame,
    /// until [`release`](Self::release). Stops any animation.
    pub fn hold(&mut self) {
        self.stop();
        self.held = true;
        self.user = true;
    }

    pub fn release(&mut self) {
        self.held = false;
        self.write(self.state);
    }

    pub fn is_held(&self) -> bool {
        self.held
    }

    /// Moves the ground under `from` to `to` (logical pixels).
    pub fn pan(&mut self, from: DVec2, to: DVec2) {
        let mut next = self.state;
        next.pan(&self.projection, self.viewport, from, to);
        self.user_write(next);
    }

    /// Zooms by `delta` levels around a pixel.
    pub fn zoom_by(&mut self, at: DVec2, delta: f64) {
        let mut next = self.state;
        let zoom = (next.zoom(&self.projection) + delta)
            .clamp(self.constraints.min_zoom, self.constraints.max_zoom);
        next.zoom_around(&self.projection, self.viewport, at, zoom);
        self.user_write(next);
    }

    /// Turns the map clockwise by `delta` degrees around a pixel.
    pub fn rotate_by(&mut self, at: DVec2, delta: f64) {
        let mut next = self.state;
        next.rotate_around(&self.projection, self.viewport, at, next.bearing + delta);
        self.user_write(next);
    }

    /// Tilts by `delta` degrees around the viewport's middle.
    pub fn pitch_by(&mut self, delta: f64) {
        let mut next = self.state;
        next.pitch += delta;
        self.user_write(next);
    }

    /// Advances the running animation to `now`, then queues this frame's
    /// events. Returns whether the camera is still moving on its own, so
    /// the host keeps drawing frames.
    pub fn frame(&mut self, now: f64) -> bool {
        let Some(running) = self.running.as_mut() else {
            self.report();
            return false;
        };
        // An app's move ends on the terrain under its destination, known
        // once its DEM data lands; a gesture holds the elevation.
        if let Some(terrain) = self.terrain.as_deref().filter(|_| !running.user)
            && let Some(ground) = running.motion.destination().and_then(|w| terrain.at(w))
        {
            running.motion.aim_elevation(ground, running.elapsed);
        }
        let mut next = self.state;
        let elapsed = now - running.started;
        running.elapsed = elapsed;
        running
            .motion
            .apply(&mut next, elapsed, &self.projection, self.viewport);
        self.user = running.user;
        let done = elapsed >= running.motion.duration();
        // Written while the move still runs, so its elevation stays its own.
        self.write(next);
        if done {
            let running = self.running.take().expect("running");
            self.events.push(CameraEvent::AnimationEnd {
                id: running.id,
                interrupted: false,
            });
            // At rest: on the terrain.
            self.write(self.state);
        }
        self.report();
        self.running.is_some()
    }

    /// The events queued by frames since the last call.
    pub fn take_events(&mut self) -> Vec<CameraEvent> {
        std::mem::take(&mut self.events)
    }

    fn report(&mut self) {
        let (a, b) = (self.reported, self.state);
        let changed = [
            a != b,
            a.zoom(&self.projection) != b.zoom(&self.projection),
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
        if self.running.is_none() && !self.held {
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

    fn start(&mut self, motion: Motion, now: f64, user: bool) -> AnimationId {
        self.next_id += 1;
        let id = AnimationId(self.next_id);
        self.modified = true;
        self.user = user;
        self.running = Some(Running {
            id,
            motion,
            started: now,
            elapsed: 0.0,
            user,
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
        next.bearing = wrap_bearing(next.bearing);
        next.constrain(&self.constraints, &self.projection, self.viewport);
        if let Some(terrain) = self.terrain.as_deref() {
            // Gestures hold the target's elevation, as GL JS's do; at rest
            // it stands on the terrain.
            let gesture = self.held || self.running.as_ref().is_some_and(|r| r.user);
            if gesture {
                self.elevation_held = true;
            } else if self.running.is_none() {
                if std::mem::take(&mut self.elevation_held) {
                    next.recenter_on(terrain, &self.projection, self.viewport);
                } else {
                    next.settle_on(terrain, &self.projection);
                }
            }
            next.keep_above(terrain, &self.projection, self.viewport);
        }
        if next != self.state {
            self.state = next;
            self.revision += 1;
        }
    }

    /// The end state of a move to `target`, and the pixel and ground point
    /// a zoom or rotation keeps in place.
    fn resolve(&self, target: &CameraTarget) -> (CameraState, Option<(Geodetic, DVec2)>) {
        let (projection, viewport) = (&self.projection, self.viewport);
        let center = DVec2::new(viewport.width, viewport.height) / 2.0;
        let mut to = self.state;
        // Limits bound the scale at the destination.
        let zoom = target.zoom.map(|z| {
            let mut at = CameraState {
                target: target.center.unwrap_or(self.state.target),
                ..self.state
            };
            at.set_zoom(projection, z);
            at.limit_zoom(
                self.constraints.min_zoom,
                self.constraints.max_zoom,
                projection,
            );
            at.zoom(projection)
        });
        let mut around = None;
        match (target.center, target.around) {
            (Some(c), _) => {
                to.target = c;
                if let Some(zoom) = zoom {
                    to.set_zoom(projection, zoom);
                } else {
                    // Keep the scale on screen, as a pan does.
                    to.resolution =
                        self.state.world_per_px(projection) / projection.scale_factor(c);
                }
            }
            (None, Some(ground)) => {
                let at = self.project(ground).unwrap_or(center);
                around = Some((ground, at));
                to.holding(projection, viewport, at, |c| {
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
                Some((_, at)) => to.rotate_around(projection, viewport, at, bearing),
                None => to.bearing = wrap_bearing(bearing),
            }
        }
        if let Some(pitch) = target.pitch {
            to.pitch = pitch;
        }
        if target.offset != DVec2::ZERO {
            to.pan(projection, viewport, center, center + target.offset);
        }
        to.bearing = wrap_bearing(to.bearing);
        to.constrain(&self.constraints, projection, viewport);
        if let Some(terrain) = self.terrain.as_deref() {
            to.settle_on(terrain, projection);
        }
        (to, around)
    }
}

#[cfg(test)]
mod tests {
    fn mercator() -> orb_crs::Projection {
        *orb_crs::TileMatrixSet::by_id("WebMercatorQuad")
            .unwrap()
            .projection()
    }

    use super::*;
    use crate::animate::{Easing, Track};
    use orb_crs::TileMatrixSet;

    fn camera() -> Camera {
        let projection = *TileMatrixSet::by_id("WebMercatorQuad")
            .unwrap()
            .projection();
        Camera::new(
            CameraState::new(Geodetic::new(139.7, 35.68), 10.0, &mercator()),
            projection,
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
            1000.0,
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
            interrupted: false
        }));
        let progress = changes(&events)
            .iter()
            .filter(|c| **c == (Change::Move, Phase::Progress))
            .count();
        assert_eq!(progress, frames + 1);
        assert_eq!(changes(&events).last(), Some(&(Change::Move, Phase::End)));
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
            0.0,
        );
        assert!(camera.frame(100.0));
        camera.hold();
        camera.pan(DVec2::new(400.0, 300.0), DVec2::new(420.0, 300.0));
        assert!(!camera.frame(116.0));
        let events = camera.take_events();
        assert!(events.contains(&CameraEvent::AnimationEnd {
            id,
            interrupted: true
        }));
        // Held: the move hasn't ended.
        assert!(!changes(&events).contains(&(Change::Move, Phase::End)));
        camera.release();
        camera.frame(132.0);
        assert_eq!(
            changes(&camera.take_events()),
            [(Change::Zoom, Phase::End), (Change::Move, Phase::End)]
        );
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
            0.0,
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
            0.0,
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
    /// its style zoom (decision 0022): world units per pixel and the zoom
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
        let mut camera = Camera::new(start, projection, viewport);
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
