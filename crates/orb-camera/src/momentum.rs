//! Gesture momentum: camera motion that continues a gesture after its last
//! contact lifts, shaped after maplibre-native-ffi's `mln_gesture_momentum`
//! (and its MapLibre Native patch), whose C API Orbis will take on. Each
//! part (pan, zoom, bearing, pitch) starts at its release speed, slows by
//! its own [`MomentumDecay`], and may settle on a [`MomentumSnap`]. Steps
//! apply to the camera as it stands, so bounds and limits that hold a part
//! in place stop it.

use glam::DVec2;

use crate::animate::Easing;
use crate::moves::{Constraints, wrap_bearing};
use crate::store::Fields;
use crate::{CameraState, Surface, Viewport};

/// How a part's speed falls from its release speed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecayModel {
    /// The speed halves every `half_life` milliseconds, until it drops to
    /// `stop_speed` (positive), in the part's unit per second.
    Friction { half_life: f64, stop_speed: f64 },
    /// The part moves `distance` over `duration` milliseconds along
    /// `easing`, as the caller plans it, for example from a platform's own
    /// fling physics. Android's scroll fling follows (0.175, 0.5, 0.35, 1).
    Ease {
        distance: f64,
        duration: f64,
        easing: Easing,
    },
}

/// How one part of a momentum slows from its release speed, in the part's
/// unit: logical pixels for pan, zoom levels for zoom, degrees for bearing
/// and pitch (`mln_momentum_decay`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentumDecay {
    pub model: DecayModel,
    /// A slower release doesn't move.
    pub min_speed: f64,
    /// For friction, a faster release counts as this speed.
    pub max_speed: f64,
    /// The farthest the part moves. A friction release that would move
    /// farther starts slower, so it ends sooner; an eased glide moves at
    /// most this far over its duration.
    pub max_distance: f64,
}

/// mln-ffi's half-life for every part's friction.
const HALF_LIFE: f64 = 140.0;
/// mln-ffi's default snap duration.
const SNAP_DURATION: f64 = 300.0;
/// Pans step no farther than a drag event moves the content, so a tilted
/// map's content travels evenly on screen; a frame that would need more
/// steps takes longer ones.
const PAN_STEP: f64 = 16.0;
const MAX_PAN_STEPS: f64 = 100.0;

impl MomentumDecay {
    /// Friction with mln-ffi's half-life, stopping at `stop_speed`, with no
    /// limits.
    pub const fn friction(stop_speed: f64) -> Self {
        Self {
            model: DecayModel::Friction {
                half_life: HALF_LIFE,
                stop_speed,
            },
            min_speed: 0.0,
            max_speed: f64::INFINITY,
            max_distance: f64::INFINITY,
        }
    }

    /// mln-ffi's pan default: stops at 10 px/s, needs a release of at
    /// least 200 px/s, and counts one faster than 10 000 px/s as 10 000.
    pub const PAN: Self = Self {
        min_speed: 200.0,
        max_speed: 10_000.0,
        ..Self::friction(10.0)
    };
    /// mln-ffi's zoom default: stops at 0.01 levels/s and moves at most 2.5
    /// levels.
    pub const ZOOM: Self = Self {
        max_distance: 2.5,
        ..Self::friction(0.01)
    };
    /// mln-ffi's bearing default: stops at 1°/s and needs at least 60°/s.
    pub const BEARING: Self = Self {
        min_speed: 60.0,
        ..Self::friction(1.0)
    };
    /// mln-ffi's pitch default: stops at 1°/s and needs at least 5°/s.
    pub const PITCH: Self = Self {
        min_speed: 5.0,
        ..Self::friction(1.0)
    };
}

/// Where a part settles once its glide ends, or at once when it doesn't
/// move: on the nearest of the targets every `spacing` from `offset`, when
/// it is within `tolerance` of one (`mln_momentum_snap`). Bearing targets
/// wrap around the circle, and the bearing takes the shorter way there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentumSnap {
    /// Positive.
    pub spacing: f64,
    pub offset: f64,
    pub tolerance: f64,
    /// Milliseconds; zero settles at once.
    pub duration: f64,
}

impl MomentumSnap {
    /// mln-ffi's zoom snap: the nearest whole level.
    pub const ZOOM: Self = Self {
        spacing: 1.0,
        offset: 0.0,
        tolerance: 0.5,
        duration: SNAP_DURATION,
    };
    /// mln-ffi's bearing snap: north, within 7°.
    pub const BEARING: Self = Self {
        spacing: 360.0,
        offset: 0.0,
        tolerance: 7.0,
        duration: SNAP_DURATION,
    };
    /// mln-ffi's pitch snap: flat, within 7°.
    pub const PITCH: Self = Self {
        spacing: 90.0,
        offset: 0.0,
        tolerance: 7.0,
        duration: SNAP_DURATION,
    };

    /// The change that settles `value`, or zero when it stays.
    fn delta(&self, value: f64, circular: bool) -> f64 {
        if self.spacing.is_nan() || self.spacing <= 0.0 || !value.is_finite() {
            return 0.0;
        }
        let delta = if circular {
            let turn = (value - self.offset).rem_euclid(360.0);
            let below = (turn / self.spacing).floor() * self.spacing;
            let above = (below + self.spacing).min(360.0);
            if turn - below <= above - turn {
                below - turn
            } else {
                above - turn
            }
        } else {
            // `f64::round` rounds halfway away from zero, as C++'s does.
            self.offset + ((value - self.offset) / self.spacing).round() * self.spacing - value
        };
        if delta.abs() <= self.tolerance {
            delta
        } else {
            0.0
        }
    }
}

/// How a gesture's pan keeps moving (`mln_pan_momentum`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PanMomentum {
    /// How fast the map's content moved on screen at release, in logical
    /// pixels per second.
    pub velocity: DVec2,
    pub decay: MomentumDecay,
}

impl PanMomentum {
    /// A pan at `velocity` with mln-ffi's default decay.
    pub fn new(velocity: DVec2) -> Self {
        Self {
            velocity,
            decay: MomentumDecay::PAN,
        }
    }
}

/// How a gesture's zoom, bearing or pitch keeps changing
/// (`mln_anchored_momentum`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchoredMomentum {
    /// How fast the value changed at release: zoom levels per second, or
    /// degrees per second of bearing (clockwise) or pitch. The part moves
    /// only when this is set.
    pub velocity: Option<f64>,
    /// The logical pixel the change keeps in place; the view's center
    /// without one.
    pub anchor: Option<DVec2>,
    pub decay: MomentumDecay,
    /// Where the part settles; without one it stays where the gesture and
    /// its glide leave it.
    pub snap: Option<MomentumSnap>,
}

impl AnchoredMomentum {
    fn new(velocity: Option<f64>, decay: MomentumDecay) -> Self {
        Self {
            velocity,
            anchor: None,
            decay,
            snap: None,
        }
    }

    /// A zoom part with mln-ffi's default decay.
    pub fn zoom(velocity: Option<f64>) -> Self {
        Self::new(velocity, MomentumDecay::ZOOM)
    }

    /// A bearing part with mln-ffi's default decay.
    pub fn bearing(velocity: Option<f64>) -> Self {
        Self::new(velocity, MomentumDecay::BEARING)
    }

    /// A pitch part with mln-ffi's default decay.
    pub fn pitch(velocity: Option<f64>) -> Self {
        Self::new(velocity, MomentumDecay::PITCH)
    }

    #[must_use]
    pub fn with_anchor(self, anchor: Option<DVec2>) -> Self {
        Self { anchor, ..self }
    }

    #[must_use]
    pub fn with_snap(self, snap: Option<MomentumSnap>) -> Self {
        Self { snap, ..self }
    }
}

/// How a gesture keeps moving the camera after the user lets go
/// (`mln_gesture_momentum`). Absent parts don't move.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GestureMomentum {
    pub pan: Option<PanMomentum>,
    pub zoom: Option<AnchoredMomentum>,
    pub bearing: Option<AnchoredMomentum>,
    pub pitch: Option<AnchoredMomentum>,
}

/// How fast each part of a momentum moves, in its unit per second
/// (`mln_camera_velocity`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MomentumVelocity {
    /// Logical pixels per second the map's content moves on screen.
    pub pan: DVec2,
    pub zoom: f64,
    pub bearing: f64,
    pub pitch: f64,
}

/// How far one part moves, over how long, and along which curve; times in
/// seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Plan {
    pub distance: f64,
    pub duration: f64,
    /// For friction, the decay rate per second.
    rate: f64,
    easing: Option<Easing>,
}

impl Plan {
    /// Plans a release at `speed` (not negative), or nothing when it
    /// doesn't move.
    pub(crate) fn new(speed: f64, decay: &MomentumDecay) -> Option<Self> {
        if speed.is_nan() || speed <= 0.0 || speed < decay.min_speed {
            return None;
        }
        let speed = speed.min(decay.max_speed);
        let (distance, duration, rate, easing) = match decay.model {
            DecayModel::Friction {
                half_life,
                stop_speed,
            } => {
                let half_life = half_life / 1000.0;
                if half_life.is_nan()
                    || half_life <= 0.0
                    || stop_speed.is_nan()
                    || stop_speed <= 0.0
                {
                    return None;
                }
                let rate = std::f64::consts::LN_2 / half_life;
                let speed = speed.min(stop_speed + rate * decay.max_distance);
                if speed <= stop_speed {
                    return None;
                }
                let seconds = (speed / stop_speed).ln() / rate;
                ((speed - stop_speed) / rate, seconds, rate, None)
            }
            DecayModel::Ease {
                distance,
                duration,
                easing,
            } => (
                distance.min(decay.max_distance),
                duration / 1000.0,
                0.0,
                Some(easing),
            ),
        };
        (distance > 0.0 && duration > 0.0 && duration.is_finite()).then_some(Self {
            distance,
            duration,
            rate,
            easing,
        })
    }

    /// The share of the distance covered after `elapsed` seconds.
    pub(crate) fn progress(&self, elapsed: f64) -> f64 {
        if elapsed >= self.duration {
            return 1.0;
        }
        if elapsed <= 0.0 {
            return 0.0;
        }
        match self.easing {
            None => (-self.rate * elapsed).exp_m1() / (-self.rate * self.duration).exp_m1(),
            Some(easing) => easing.at(elapsed / self.duration),
        }
    }

    /// The share of the distance covered per second after `elapsed`
    /// seconds.
    pub(crate) fn speed(&self, elapsed: f64) -> f64 {
        if elapsed >= self.duration {
            return 0.0;
        }
        let t = elapsed.max(0.0);
        match self.easing {
            None => self.rate * (-self.rate * t).exp() / -(-self.rate * self.duration).exp_m1(),
            Some(easing) => easing.slope(t / self.duration) / self.duration,
        }
    }
}

/// One scalar part of a running momentum.
#[derive(Clone, Copy, Debug)]
struct Scalar {
    glide: Option<Plan>,
    /// The signed distance.
    total: f64,
    applied: f64,
    moving: bool,
    anchor: Option<DVec2>,
    snap: Option<MomentumSnap>,
    snap_pending: bool,
    /// The settling's change while it runs, when it started (seconds), and
    /// its share applied.
    settling: Option<(f64, f64, f64)>,
}

impl Scalar {
    fn active(&self) -> bool {
        self.moving || self.snap_pending || self.settling.is_some()
    }
}

#[derive(Clone, Copy, Debug)]
struct Pan {
    glide: Plan,
    total: DVec2,
    applied: f64,
    moving: bool,
}

const ZOOM: usize = 0;
const BEARING: usize = 1;
const PITCH: usize = 2;

/// A momentum under way, which the camera store runs frame by frame.
#[derive(Clone, Debug)]
pub(crate) struct Momentum {
    pan: Option<Pan>,
    parts: [Option<Scalar>; 3],
    /// Seconds in at the last frame.
    elapsed: f64,
    /// A later move took a property a moving or settling part needed,
    /// which stopped that part.
    replaced: bool,
}

/// The property each scalar part moves.
const PART_FIELDS: [Fields; 3] = [Fields::ZOOM, Fields::BEARING, Fields::PITCH];

/// The camera's limits and frame, which each step keeps to.
pub(crate) struct Stage<'a> {
    pub surface: &'a Surface,
    pub viewport: Viewport,
    pub constraints: &'a Constraints,
    /// Pans on the sphere turn the bearing with the planet.
    pub free_bearing: bool,
}

impl Stage<'_> {
    fn value(&self, camera: &CameraState, part: usize) -> f64 {
        match part {
            ZOOM => camera.zoom(&self.surface.projection),
            BEARING => camera.bearing,
            _ => camera.pitch,
        }
    }

    fn settle(&self, camera: &mut CameraState) {
        camera.bearing = wrap_bearing(camera.bearing);
        camera.constrain(self.constraints, self.surface, self.viewport);
    }

    /// Moves the content `offset` pixels on screen, in drag-sized steps.
    fn pan(&self, camera: &mut CameraState, offset: DVec2) {
        let distance = offset.length();
        if distance.is_nan() || distance <= 0.0 {
            return;
        }
        let steps = (distance / PAN_STEP).ceil().min(MAX_PAN_STEPS);
        let step = offset / steps;
        for _ in 0..steps as usize {
            let center = camera.center(self.viewport);
            let to = center + step;
            camera.pan_with(self.surface, self.viewport, center, to, self.free_bearing);
            self.settle(camera);
        }
    }

    fn step(&self, camera: &mut CameraState, part: usize, amount: f64, anchor: Option<DVec2>) {
        let (surface, viewport) = (self.surface, self.viewport);
        let at = anchor.unwrap_or_else(|| camera.center(viewport));
        match part {
            ZOOM => {
                let least = camera.least_zoom(self.constraints, surface);
                let zoom = (camera.zoom(&surface.projection) + amount)
                    .clamp(least, self.constraints.max_zoom.max(least));
                camera.zoom_around(surface, viewport, at, zoom);
            }
            BEARING => camera.rotate_around(surface, viewport, at, camera.bearing + amount),
            _ => {
                let (min, max) = (
                    self.constraints.min_pitch,
                    self.constraints.max_pitch.max(self.constraints.min_pitch),
                );
                camera.holding(surface, viewport, at, |c| {
                    c.pitch = (c.pitch + amount).clamp(min, max);
                });
            }
        }
        self.settle(camera);
    }
}

impl Momentum {
    /// Plans `momentum` from `camera`, or `None` when nothing moves and
    /// nothing settles. Snaps that settle at once without motion apply to
    /// `camera` here.
    pub(crate) fn new(
        momentum: &GestureMomentum,
        camera: &mut CameraState,
        stage: &Stage,
    ) -> Option<Self> {
        let pan = momentum.pan.and_then(|pan| {
            let speed = pan.velocity.length();
            let glide = Plan::new(speed, &pan.decay)?;
            Some(Pan {
                glide,
                total: pan.velocity * (glide.distance / speed),
                applied: 0.0,
                moving: true,
            })
        });
        let mut parts = [momentum.zoom, momentum.bearing, momentum.pitch].map(|part| {
            part.map(|part| {
                let velocity = part.velocity.filter(|v| v.is_finite());
                let glide = velocity.and_then(|v| Plan::new(v.abs(), &part.decay));
                Scalar {
                    glide,
                    total: glide.map_or(0.0, |g| g.distance.copysign(velocity.unwrap_or(0.0))),
                    applied: 0.0,
                    moving: glide.is_some(),
                    anchor: part.anchor,
                    snap: part.snap,
                    snap_pending: false,
                    settling: None,
                }
            })
        });
        for (i, part) in parts.iter_mut().enumerate() {
            if let Some(part) = part {
                let value = stage.value(camera, i);
                part.snap_pending = part
                    .snap
                    .is_some_and(|s| part.moving || s.delta(value, i == BEARING) != 0.0);
            }
        }
        let moving = pan.is_some() || parts.iter().flatten().any(|p| p.moving);
        let settling: Vec<_> = parts.iter().flatten().filter(|p| p.snap_pending).collect();
        if !moving
            && settling
                .iter()
                .all(|p| p.snap.is_some_and(|s| s.duration <= 0.0))
        {
            // Nothing moves, and what settles does so at once.
            for (i, part) in parts.iter().enumerate() {
                if let Some(part) = part.filter(|p| p.snap_pending) {
                    let snap = part.snap.expect("pending snaps have one");
                    let delta = snap.delta(stage.value(camera, i), i == BEARING);
                    stage.step(camera, i, delta, part.anchor);
                }
            }
            return None;
        }
        Some(Self {
            pan,
            parts,
            elapsed: 0.0,
            replaced: false,
        })
    }

    /// The properties a part needs: its own, and the center when it turns
    /// about an anchor.
    fn part_fields(i: usize, part: &Scalar) -> Fields {
        match part.anchor {
            Some(_) => PART_FIELDS[i] | Fields::CENTER,
            None => PART_FIELDS[i],
        }
    }

    /// The properties the parts still moving or settling need.
    pub(crate) fn fields(&self) -> Fields {
        let mut fields = Fields::NONE;
        if self.pan.is_some_and(|p| p.moving) {
            fields |= Fields::CENTER;
        }
        for (i, part) in self.parts.iter().enumerate() {
            if let Some(part) = part.filter(Scalar::active) {
                fields |= Self::part_fields(i, &part);
            }
        }
        fields
    }

    /// Stops the parts that need a property not in `owned`, which a later
    /// move took, and returns what the rest still need.
    pub(crate) fn retain(&mut self, owned: Fields) -> Fields {
        if let Some(pan) = self.pan.as_mut().filter(|p| p.moving)
            && !owned.contains(Fields::CENTER)
        {
            pan.moving = false;
            self.replaced = true;
        }
        for (i, slot) in self.parts.iter_mut().enumerate() {
            let Some(part) = slot.as_mut().filter(|p| p.active()) else {
                continue;
            };
            if !owned.contains(Self::part_fields(i, part)) {
                self.replaced |= part.moving || part.settling.is_some();
                part.moving = false;
                part.snap_pending = false;
                part.settling = None;
            }
        }
        self.fields()
    }

    /// Whether a later move stopped a part that was moving or settling.
    pub(crate) fn replaced(&self) -> bool {
        self.replaced
    }

    /// Whether any part still moves or settles.
    pub(crate) fn active(&self) -> bool {
        self.pan.is_some_and(|p| p.moving) || self.parts.iter().flatten().any(Scalar::active)
    }

    /// How fast the parts still moving moved at the last frame.
    pub(crate) fn velocity(&self) -> MomentumVelocity {
        let t = self.elapsed;
        let speed = |part: &Option<Scalar>| match part {
            Some(Scalar {
                glide: Some(glide),
                moving: true,
                total,
                ..
            }) => total * glide.speed(t),
            _ => 0.0,
        };
        MomentumVelocity {
            pan: match self.pan {
                Some(pan) if pan.moving => pan.total * pan.glide.speed(t),
                _ => DVec2::ZERO,
            },
            zoom: speed(&self.parts[ZOOM]),
            bearing: speed(&self.parts[BEARING]),
            pitch: speed(&self.parts[PITCH]),
        }
    }

    /// Moves `camera` on to `elapsed` milliseconds in.
    pub(crate) fn apply(&mut self, camera: &mut CameraState, elapsed: f64, stage: &Stage) {
        let t = elapsed / 1000.0;
        self.elapsed = t;
        if let Some(pan) = self.pan.as_mut().filter(|p| p.moving) {
            let progress = pan.glide.progress(t);
            let fraction = progress - pan.applied;
            pan.applied = progress;
            pan.moving = progress < 1.0;
            if fraction != 0.0 {
                let before = camera.target;
                stage.pan(camera, pan.total * fraction);
                // Bounds that hold the pan stop it.
                if camera.target == before {
                    pan.moving = false;
                }
            }
        }
        for (i, part) in self.parts.iter_mut().enumerate() {
            let Some(part) = part else { continue };
            if let Some(glide) = part.glide.filter(|_| part.moving) {
                let progress = glide.progress(t);
                let fraction = progress - part.applied;
                part.applied = progress;
                part.moving = progress < 1.0;
                if fraction != 0.0 {
                    let before = stage.value(camera, i);
                    stage.step(camera, i, part.total * fraction, part.anchor);
                    // Limits that hold the part stop it; bearing has none.
                    if i != BEARING && stage.value(camera, i) == before {
                        part.moving = false;
                    }
                }
            }
            // It settles once it stops, which bounds and limits can make
            // early.
            if part.snap_pending && !part.moving {
                part.snap_pending = false;
                let snap = part.snap.expect("pending snaps have one");
                let delta = snap.delta(stage.value(camera, i), i == BEARING);
                if delta != 0.0 {
                    part.settling = Some((delta, t, 0.0));
                }
            }
            if let Some((delta, start, applied)) = part.settling {
                let duration = part.snap.map_or(0.0, |s| s.duration) / 1000.0;
                let k = if duration > 0.0 {
                    ((t - start) / duration).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let eased = if k >= 1.0 { 1.0 } else { Easing::SNAP.at(k) };
                if eased != applied {
                    stage.step(camera, i, delta * (eased - applied), part.anchor);
                }
                part.settling = (k < 1.0).then_some((delta, start, eased));
            }
        }
    }
}

#[cfg(test)]
mod tests;
