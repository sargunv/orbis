//! Gestures: pointer, touch, wheel and keyboard input turned into
//! camera moves. Host-independent: hosts feed [`Input`]s with their
//! timestamps and call [`Interaction::frame`] once a display frame; every
//! move goes through the [`Camera`] store's public moves, which report the
//! camera events, and every release continues with the store's
//! [`GestureMomentum`].
//!
//! The gesture set is MapLibre Compose's: drag to pan; right- or
//! Ctrl-drag to rotate and tilt; Shift-drag to zoom to a box; wheel and
//! trackpad zoom; double-click or double-tap to zoom in, Shift-double-click
//! or a two-finger tap to zoom out; double-tap-and-drag to zoom; two-finger
//! pinch, twist and vertical tilt; the arrow, `+`/`=` and `-` keys. Each
//! gesture's rates and thresholds are GL JS's. Releases are Compose's: the
//! velocity of the last 100 ms ([`velocity`]), a glide with mln-ffi's
//! default decays, a glide the gesture caught added to one going the same
//! way, and the bearing settling on north within 7°.

mod rates;
mod velocity;

use rates::*;
use velocity::VelocityTracker;

use crate::animate::{Glide, Track};
use crate::{
    AnchoredMomentum, Camera, CameraTarget, EaseOptions, Easing, GestureMomentum, MomentumDecay,
    MomentumSnap, MomentumVelocity, PanMomentum,
};
use glam::DVec2;

/// What kind of device a pointer is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointerKind {
    Mouse,
    Touch,
    Pen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Button {
    Primary,
    Secondary,
    Middle,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    pub meta: bool,
}

/// The keys the map responds to; hosts map platform key codes to them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Left,
    Right,
    Up,
    Down,
    ZoomIn,
    ZoomOut,
    /// Cancels a box zoom.
    Escape,
}

/// The unit of a wheel delta, as DOM `deltaMode` gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WheelUnit {
    Pixel,
    Line,
    Page,
}

/// Input in logical pixels from the viewport's top-left corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Input {
    PointerDown {
        id: u32,
        kind: PointerKind,
        button: Button,
        position: DVec2,
        modifiers: Modifiers,
    },
    PointerMove {
        id: u32,
        position: DVec2,
    },
    PointerUp {
        id: u32,
        position: DVec2,
    },
    /// The platform took the pointer away (a system gesture, lost
    /// capture): no click and no momentum.
    PointerCancel {
        id: u32,
    },
    /// Positive `delta.y` scrolls down, which zooms out.
    Wheel {
        position: DVec2,
        delta: DVec2,
        unit: WheelUnit,
        modifiers: Modifiers,
    },
    Key {
        key: Key,
        modifiers: Modifiers,
    },
    /// The view lost focus: every pointer is released.
    Blur,
}

/// What a gesture means beyond the camera's own events.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InteractEvent {
    /// A press and release that didn't move the map: a click or a tap.
    Click {
        position: DVec2,
        button: Button,
    },
    DoubleClick {
        position: DVec2,
    },
}

#[derive(Clone, Copy, Debug)]
struct Pointer {
    id: u32,
    kind: PointerKind,
    position: DVec2,
    down: DVec2,
    down_at: f64,
}

#[derive(Clone, Copy, Debug)]
struct Pinch {
    last: [DVec2; 2],
    start_distance: f64,
    start_vector: DVec2,
    min_diameter: f64,
    zooming: bool,
    rotating: bool,
    /// Whether the fingers move up or down together: undecided at first.
    pitch: Option<bool>,
    single_since: Option<f64>,
    panning: bool,
    /// Started within the tap interval and not moved: a two-finger tap so
    /// far.
    tap: bool,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Idle,
    /// Pressed and not moved past the click tolerance.
    Pressed {
        rotate: bool,
        button: Button,
        shift: bool,
    },
    Pan,
    /// Shift-drag: the box from `origin` to the pointer is where the map
    /// zooms on release.
    BoxZoom {
        origin: DVec2,
    },
    /// Mouse rotation and tilt.
    Rotate,
    Pinch(Pinch),
    /// The second press of a double tap, held: dragging zooms.
    TapDrag {
        origin: DVec2,
        moved: bool,
    },
    /// Buttons or pointers the map doesn't use are down.
    Ignored,
}

/// One applied gesture step, for the release velocity.
#[derive(Clone, Copy, Debug, Default)]
struct Step {
    pan: DVec2,
    zoom: f64,
    bearing: f64,
    pitch: f64,
    around: Option<DVec2>,
}

/// Per second, at release.
#[derive(Clone, Copy, Debug, Default)]
struct Release {
    pan: Option<DVec2>,
    zoom: Option<f64>,
    bearing: Option<f64>,
    pitch: Option<f64>,
    zoom_anchor: Option<DVec2>,
    bearing_anchor: Option<DVec2>,
}

/// What a gesture has moved so far, and how fast.
#[derive(Debug, Default)]
struct Tracking {
    /// Totals since the gesture began: pixels, levels, degrees.
    pan: DVec2,
    zoom: f64,
    bearing: f64,
    pitch: f64,
    /// Pan x and y, zoom, bearing, pitch.
    trackers: [VelocityTracker; 5],
    zoom_anchor: Option<DVec2>,
    bearing_anchor: Option<DVec2>,
    /// The gesture turned the map, so the bearing may settle.
    rotated: bool,
    /// How fast a momentum this gesture caught still moved.
    boost: Option<MomentumVelocity>,
    /// A pinch's release, staged when it continued as a one-finger pan,
    /// and when.
    staged: Option<(f64, Release)>,
}

impl Tracking {
    fn reset(&mut self, now: f64, boost: Option<MomentumVelocity>) {
        *self = Self {
            boost,
            ..Self::default()
        };
        self.record(now, Step::default());
    }

    /// Adds a step's samples: to each part it moves, and to each part the
    /// gesture moves already, which stands still otherwise.
    fn record(&mut self, now: f64, step: Step) {
        self.pan += step.pan;
        self.zoom += step.zoom;
        self.bearing += step.bearing;
        self.pitch += step.pitch;
        if step.zoom != 0.0 {
            self.zoom_anchor = step.around;
        }
        if step.bearing != 0.0 {
            self.bearing_anchor = step.around;
            self.rotated = true;
        }
        let values = [self.pan.x, self.pan.y, self.zoom, self.bearing, self.pitch];
        let moved = [
            step.pan != DVec2::ZERO,
            step.pan != DVec2::ZERO,
            step.zoom != 0.0,
            step.bearing != 0.0,
            step.pitch != 0.0,
        ];
        let start = self.trackers.iter().all(VelocityTracker::is_empty);
        for ((tracker, value), moved) in self.trackers.iter_mut().zip(values).zip(moved) {
            if moved || start || !tracker.is_empty() {
                tracker.add(now, value);
            }
        }
    }

    /// The release at `now`.
    fn release(&mut self, now: f64) -> Release {
        let [x, y, zoom, bearing, pitch] = self.trackers.each_ref().map(|t| t.velocity(now));
        let some = |v: f64| (v != 0.0 && v.is_finite()).then_some(v);
        let mut release = Release {
            pan: (x != 0.0 || y != 0.0).then_some(DVec2::new(x, y)),
            zoom: some(zoom),
            bearing: some(bearing),
            pitch: some(pitch),
            zoom_anchor: self.zoom_anchor,
            bearing_anchor: self.bearing_anchor,
        };
        // A pinch that ended moments ago keeps what it had, as Compose's
        // contact changes do; a pan carried on longer than the velocity
        // horizon has left it behind.
        if let Some((at, staged)) = self.staged.take()
            && now - at <= 100.0
        {
            if release.zoom.is_none() {
                release.zoom = staged.zoom;
                release.zoom_anchor = staged.zoom_anchor;
            }
            if release.bearing.is_none() {
                release.bearing = staged.bearing;
                release.bearing_anchor = staged.bearing_anchor;
            }
            if release.pitch.is_none() {
                release.pitch = staged.pitch;
            }
        }
        release
    }

    /// A pinch continues as a one-finger pan: its zoom, turn and tilt end
    /// here, and its pan carries on.
    fn stage(&mut self, now: f64) {
        let release = self.release(now);
        self.staged = Some((now, release));
        for tracker in &mut self.trackers[2..] {
            tracker.reset();
        }
    }

    /// The momentum a release continues with: mln-ffi's default decays,
    /// a caught momentum added to each part going its way, and the bearing
    /// settling on north if the gesture turned the map (Compose's
    /// `resolveGestureMomentum`). `None` when nothing moves or settles.
    fn momentum(&self, release: Release, glide: bool) -> Option<GestureMomentum> {
        let release = if glide { release } else { Release::default() };
        let boost = self.boost.unwrap_or_default();
        let pan = release.pan.and_then(|mut v| {
            if v.is_nan() || v.length() < MomentumDecay::PAN.min_speed {
                return None;
            }
            if v.dot(boost.pan) > 0.0 {
                v += boost.pan;
            }
            Some(PanMomentum::new(v))
        });
        let speed = |v: Option<f64>, decay: MomentumDecay, boost: f64| {
            v.filter(|v| v.abs() >= decay.min_speed).map(|v| {
                if v.signum() == boost.signum() {
                    v + boost
                } else {
                    v
                }
            })
        };
        let zoom = speed(release.zoom, MomentumDecay::ZOOM, boost.zoom)
            .map(|v| AnchoredMomentum::zoom(Some(v)).with_anchor(release.zoom_anchor));
        let bearing = speed(release.bearing, MomentumDecay::BEARING, boost.bearing);
        let bearing = (bearing.is_some() || self.rotated).then(|| {
            AnchoredMomentum::bearing(bearing)
                .with_anchor(release.bearing_anchor)
                .with_snap(self.rotated.then_some(MomentumSnap::BEARING))
        });
        let pitch = speed(release.pitch, MomentumDecay::PITCH, boost.pitch)
            .map(|v| AnchoredMomentum::pitch(Some(v)));
        let momentum = GestureMomentum {
            pan,
            zoom,
            bearing,
            pitch,
        };
        (momentum != GestureMomentum::default()).then_some(momentum)
    }
}

/// How a gesture ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// The platform took it away: nothing continues.
    Cancel,
    /// Released without a fling (a tap, a box): only snaps.
    Settle,
    /// Released: it glides on.
    Glide,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WheelKind {
    Wheel,
    Trackpad,
}

/// GL JS's mouse-wheel step, which trackpads never produce exactly.
const WHEEL_STEP: f64 = 4.000_244_140_625;
/// Trackpad zoom stays one move until this long after its last event.
const TRACKPAD_END: f64 = 200.0;

/// The gesture recognizers for one map.
#[derive(Debug)]
pub struct Interaction {
    pointers: Vec<Pointer>,
    mode: Mode,
    tracking: Tracking,
    /// The last tap, for double taps: where and when.
    last_tap: Option<(DVec2, f64, PointerKind)>,
    wheel_kind: Option<WheelKind>,
    last_wheel: f64,
    /// Trackpad zoom holds the camera until then.
    wheel_hold_until: Option<f64>,
    events: Vec<InteractEvent>,
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            pointers: Vec::new(),
            mode: Mode::Idle,
            tracking: Tracking::default(),
            last_tap: None,
            wheel_kind: None,
            last_wheel: f64::NEG_INFINITY,
            wheel_hold_until: None,
            events: Vec::new(),
        }
    }
}

impl Interaction {
    /// A gesture is under way: hosts keep pointer capture and scrolling off.
    pub fn is_active(&self) -> bool {
        !matches!(self.mode, Mode::Idle) || self.wheel_hold_until.is_some()
    }

    /// The box a Shift-drag zooms to, corner to corner in logical pixels,
    /// while it is drawn: hosts show it.
    pub fn box_zoom(&self) -> Option<[DVec2; 2]> {
        match self.mode {
            Mode::BoxZoom { origin } => Some([origin, self.pointers.first()?.position]),
            _ => None,
        }
    }

    /// Clicks and double clicks recognized since the last call.
    pub fn take_events(&mut self) -> Vec<InteractEvent> {
        std::mem::take(&mut self.events)
    }

    /// Ends gestures that time out. Call once a display frame with the
    /// time; returns whether it needs more frames to do so.
    pub fn frame(&mut self, camera: &mut Camera, now: f64) -> bool {
        if let Some(until) = self.wheel_hold_until {
            if now >= until {
                self.wheel_hold_until = None;
                if matches!(self.mode, Mode::Idle) {
                    camera.release();
                }
            } else {
                return true;
            }
        }
        false
    }

    /// Handles one input at host time `now` (milliseconds). Returns whether
    /// the map used it, so hosts can stop the platform's default.
    pub fn handle(&mut self, camera: &mut Camera, input: Input, now: f64) -> bool {
        match input {
            Input::PointerDown {
                id,
                kind,
                button,
                position,
                modifiers,
            } => self.down(camera, id, kind, button, position, modifiers, now),
            Input::PointerMove { id, position } => self.moved(camera, id, position, now),
            Input::PointerUp { id, position } => self.up(camera, id, Some(position), now),
            Input::PointerCancel { id } => self.up(camera, id, None, now),
            Input::Wheel {
                position,
                delta,
                unit,
                modifiers,
            } => self.wheel(camera, position, delta, unit, modifiers, now),
            Input::Key { key, modifiers } => self.key(camera, key, modifiers, now),
            Input::Blur => {
                self.pointers.clear();
                self.mode = Mode::Idle;
                camera.release();
                true
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn down(
        &mut self,
        camera: &mut Camera,
        id: u32,
        kind: PointerKind,
        button: Button,
        position: DVec2,
        modifiers: Modifiers,
        now: f64,
    ) -> bool {
        if self.pointers.iter().any(|p| p.id == id) {
            return false;
        }
        let touch = kind == PointerKind::Touch;
        if !touch && (!self.pointers.is_empty() || button == Button::Middle) {
            return false;
        }
        self.pointers.push(Pointer {
            id,
            kind,
            position,
            down: position,
            down_at: now,
        });
        let caught = camera.hold();
        match (self.mode, self.pointers.len()) {
            (Mode::Idle, 1) => {
                self.mode = if touch && self.follows_tap(kind, position, now) {
                    Mode::TapDrag {
                        origin: position,
                        moved: false,
                    }
                } else {
                    Mode::Pressed {
                        rotate: button == Button::Secondary || (modifiers.ctrl && !touch),
                        button,
                        shift: modifiers.shift && !touch,
                    }
                };
                self.tracking.reset(now, caught);
            }
            (Mode::Pressed { .. } | Mode::Pan | Mode::TapDrag { .. }, 2) if touch => {
                let [a, b] = [self.pointers[0].position, self.pointers[1].position];
                let vector = b - a;
                self.mode = Mode::Pinch(Pinch {
                    last: [a, b],
                    start_distance: vector.length(),
                    start_vector: vector,
                    min_diameter: vector.length(),
                    zooming: false,
                    rotating: false,
                    // Fingers above one another can't tilt.
                    pitch: (vector.y.abs() > vector.x.abs()).then_some(false),
                    single_since: None,
                    panning: false,
                    tap: now - self.pointers[0].down_at < TAP_INTERVAL
                        && matches!(self.mode, Mode::Pressed { .. }),
                });
                self.tracking.record(now, Step::default());
            }
            _ => self.mode = Mode::Ignored,
        }
        true
    }

    fn moved(&mut self, camera: &mut Camera, id: u32, position: DVec2, now: f64) -> bool {
        let Some(index) = self.pointers.iter().position(|p| p.id == id) else {
            return false;
        };
        let last = self.pointers[index].position;
        self.pointers[index].position = position;
        match &mut self.mode {
            Mode::Pressed {
                rotate,
                shift,
                button,
            } => {
                let down = self.pointers[index].down;
                if position.distance(down) > CLICK_TOLERANCE {
                    self.mode = match (*rotate, *shift && *button == Button::Primary) {
                        (true, _) => Mode::Rotate,
                        (false, true) => Mode::BoxZoom { origin: down },
                        (false, false) => Mode::Pan,
                    };
                    self.step(camera, down, position, now);
                }
            }
            Mode::Pan | Mode::Rotate => self.step(camera, last, position, now),
            Mode::BoxZoom { .. } => {}
            Mode::TapDrag { origin, moved } => {
                let (origin, dy) = (*origin, position.y - last.y);
                *moved = *moved || position.distance(origin) > CLICK_TOLERANCE;
                if *moved {
                    let zoom = dy / 128.0;
                    camera.zoom_by(origin, zoom);
                    self.tracking.record(
                        now,
                        Step {
                            zoom,
                            around: Some(origin),
                            ..Step::default()
                        },
                    );
                }
            }
            Mode::Pinch(_) if self.pointers.len() < 2 => {
                // The other finger of a two-finger tap lifted, and this one
                // moves on: a pan.
                self.mode = Mode::Pan;
                self.step(camera, last, position, now);
            }
            Mode::Pinch(pinch) => {
                if index > 1 {
                    return true;
                }
                let mut pinch = *pinch;
                let points = [self.pointers[0].position, self.pointers[1].position];
                self.pinch_step(camera, &mut pinch, points, now);
                self.mode = Mode::Pinch(pinch);
            }
            Mode::Idle | Mode::Ignored => {}
        }
        true
    }

    /// One pan or mouse-rotate step from `from` to `to`.
    fn step(&mut self, camera: &mut Camera, from: DVec2, to: DVec2, now: f64) {
        let delta = to - from;
        match self.mode {
            Mode::Rotate => {
                let bearing = delta.x * DRAG_ROTATE;
                let pitch = -delta.y * DRAG_PITCH;
                let center = center(camera);
                camera.rotate_by(center, bearing);
                camera.pitch_by(pitch);
                self.tracking.record(
                    now,
                    Step {
                        bearing,
                        pitch,
                        ..Step::default()
                    },
                );
            }
            Mode::BoxZoom { .. } => {}
            _ => {
                camera.pan(from, to);
                self.tracking.record(
                    now,
                    Step {
                        pan: delta,
                        ..Step::default()
                    },
                );
            }
        }
    }

    fn pinch_step(&mut self, camera: &mut Camera, pinch: &mut Pinch, points: [DVec2; 2], now: f64) {
        let [a, b] = points;
        let [la, lb] = pinch.last;
        let (va, vb) = (a - la, b - lb);
        let vector = b - a;
        if pinch.tap
            && (a.distance(self.pointers[0].down) > TAP_TOLERANCE
                || b.distance(self.pointers[1].down) > TAP_TOLERANCE)
        {
            pinch.tap = false;
        }
        // Two fingers moving up or down together tilt, and do nothing else
        // (GL JS `TwoFingersTouchPitchHandler`).
        if pinch.pitch.is_none() {
            let (moved_a, moved_b) = (va.length() >= 2.0, vb.length() >= 2.0);
            if moved_a && moved_b {
                let vertical = |v: DVec2| v.y.abs() > v.x.abs();
                pinch.pitch = Some(vertical(va) && vertical(vb) && (va.y > 0.0) == (vb.y > 0.0));
            } else if moved_a || moved_b {
                let since = *pinch.single_since.get_or_insert(now);
                if now - since >= 100.0 {
                    pinch.pitch = Some(false);
                }
            }
            if pinch.pitch.is_none() {
                return;
            }
        }
        if pinch.pitch == Some(true) {
            let pitch = (va.y + vb.y) / 2.0 * -DRAG_PITCH;
            camera.pitch_by(pitch);
            pinch.last = points;
            pinch.tap = false;
            self.tracking.record(
                now,
                Step {
                    pitch,
                    ..Step::default()
                },
            );
            return;
        }
        let (mid, last_mid) = ((a + b) / 2.0, (la + lb) / 2.0);
        let distance = vector.length();
        let last_distance = (lb - la).length();
        pinch.min_diameter = pinch.min_diameter.min(distance);
        if !pinch.zooming && (distance / pinch.start_distance).log2().abs() >= PINCH_ZOOM_THRESHOLD
        {
            pinch.zooming = true;
        }
        if !pinch.rotating {
            let threshold = ROTATE_THRESHOLD / (std::f64::consts::PI * pinch.min_diameter) * 360.0;
            if angle(pinch.start_vector, vector).abs() >= threshold {
                pinch.rotating = true;
            }
        }
        if !pinch.panning
            && (mid - (self.pointers[0].down + self.pointers[1].down) / 2.0).length()
                > CLICK_TOLERANCE
        {
            pinch.panning = true;
        }
        let pan = if pinch.panning {
            mid - last_mid
        } else {
            DVec2::ZERO
        };
        let zoom = if pinch.zooming && last_distance > 0.0 {
            (distance / last_distance).log2()
        } else {
            0.0
        };
        // Fingers turning clockwise turn the map clockwise.
        let bearing = if pinch.rotating {
            -angle(lb - la, vector)
        } else {
            0.0
        };
        if pan != DVec2::ZERO {
            camera.pan(last_mid, mid);
        }
        if zoom != 0.0 {
            camera.zoom_by(mid, zoom);
        }
        if bearing != 0.0 {
            camera.rotate_by(mid, bearing);
        }
        pinch.last = points;
        if pan != DVec2::ZERO || zoom != 0.0 || bearing != 0.0 {
            pinch.tap = false;
            self.tracking.record(
                now,
                Step {
                    pan,
                    zoom,
                    bearing,
                    around: Some(mid),
                    ..Step::default()
                },
            );
        }
    }

    fn up(&mut self, camera: &mut Camera, id: u32, position: Option<DVec2>, now: f64) -> bool {
        let Some(index) = self.pointers.iter().position(|p| p.id == id) else {
            return false;
        };
        let pointer = self.pointers.remove(index);
        let cancelled = position.is_none();
        match self.mode {
            Mode::Pinch(pinch) => {
                if !self.pointers.is_empty() {
                    // Lifting one finger continues as a one-finger pan;
                    // lifting the second ends a two-finger tap.
                    let tap = pinch.tap && now - pointer.down_at < TAP_INTERVAL;
                    self.mode = if tap {
                        Mode::Pinch(pinch)
                    } else {
                        self.tracking.stage(now);
                        Mode::Pan
                    };
                    return true;
                }
                if pinch.tap && !cancelled {
                    let mid = (pinch.last[0] + pinch.last[1]) / 2.0;
                    self.zoom_step(camera, mid, -1.0, now);
                }
                let end = match (cancelled, pinch.tap) {
                    (true, _) => End::Cancel,
                    (false, true) => End::Settle,
                    (false, false) => End::Glide,
                };
                self.finish(camera, now, end);
            }
            Mode::Pan if !self.pointers.is_empty() => {}
            Mode::Pressed { button, shift, .. } => {
                if self.pointers.is_empty() {
                    if !cancelled {
                        self.tap(camera, pointer, button, shift, now);
                    }
                    self.finish(camera, now, End::Settle);
                }
            }
            Mode::BoxZoom { origin } => {
                if !cancelled {
                    self.zoom_to_box(camera, origin, position.unwrap_or(pointer.position), now);
                }
                self.finish(camera, now, End::Cancel);
            }
            Mode::TapDrag { origin, moved } => {
                if self.pointers.is_empty() {
                    self.last_tap = None;
                    if !moved && !cancelled {
                        self.events
                            .push(InteractEvent::DoubleClick { position: origin });
                        self.zoom_step(camera, origin, 1.0, now);
                    }
                    let end = match (cancelled, moved) {
                        (true, _) => End::Cancel,
                        (false, true) => End::Glide,
                        (false, false) => End::Settle,
                    };
                    self.finish(camera, now, end);
                }
            }
            Mode::Pan | Mode::Rotate => {
                self.finish(
                    camera,
                    now,
                    if cancelled { End::Cancel } else { End::Glide },
                );
            }
            Mode::Ignored | Mode::Idle => {
                if self.pointers.is_empty() {
                    self.finish(camera, now, End::Cancel);
                }
            }
        }
        true
    }

    /// A press released without moving: a click, perhaps the second of a
    /// double click.
    fn tap(
        &mut self,
        camera: &mut Camera,
        pointer: Pointer,
        button: Button,
        shift: bool,
        now: f64,
    ) {
        let position = pointer.position;
        self.events.push(InteractEvent::Click { position, button });
        if button != Button::Primary || now - pointer.down_at > TAP_INTERVAL {
            return;
        }
        if self.follows_tap(pointer.kind, position, now) {
            self.last_tap = None;
            self.events.push(InteractEvent::DoubleClick { position });
            // Shift-double-click zooms out.
            self.zoom_step(camera, position, if shift { -1.0 } else { 1.0 }, now);
        } else {
            self.last_tap = Some((position, now, pointer.kind));
        }
    }

    /// Ends the gesture with every pointer up: a release glides on with
    /// the speed its last moves had, and settles.
    fn finish(&mut self, camera: &mut Camera, now: f64, end: End) {
        self.mode = Mode::Idle;
        if end != End::Cancel {
            let release = self.tracking.release(now);
            if let Some(momentum) = self.tracking.momentum(release, end == End::Glide) {
                camera.start_momentum(&momentum, Some(now));
            }
        }
        if self.wheel_hold_until.is_none() {
            camera.release();
        }
    }

    /// Whether a press of `kind` at `position` is the second of a double tap.
    fn follows_tap(&self, kind: PointerKind, position: DVec2, now: f64) -> bool {
        self.last_tap.is_some_and(|(at, time, k)| {
            k == kind && now - time < TAP_INTERVAL && at.distance(position) < TAP_TOLERANCE
        })
    }

    /// Zooms so the box from `a` to `b` fills the view, centred on its
    /// middle (GL JS `BoxZoomHandler`); a box without area does nothing.
    fn zoom_to_box(&mut self, camera: &mut Camera, a: DVec2, b: DVec2, now: f64) {
        let size = (b - a).abs();
        let viewport = camera.viewport();
        if size.x < 1.0 || size.y < 1.0 {
            return;
        }
        let Some(center) = camera.unproject((a + b) / 2.0) else {
            return;
        };
        let zoom = (viewport.width / size.x)
            .min(viewport.height / size.y)
            .log2();
        camera.user_ease(
            &CameraTarget {
                center: Some(center),
                zoom: Some(camera.zoom() + zoom),
                ..CameraTarget::default()
            },
            EaseOptions::default(),
            now,
        );
    }

    /// Double-click, double-tap and two-finger-tap zoom: one level around
    /// a point, eased (GL JS, 300 ms).
    fn zoom_step(&mut self, camera: &mut Camera, at: DVec2, delta: f64, now: f64) {
        let Some(around) = camera.unproject(at) else {
            return;
        };
        camera.user_ease(
            &CameraTarget {
                zoom: Some(camera.zoom() + delta),
                around: Some(around),
                ..CameraTarget::default()
            },
            EaseOptions {
                duration: 300.0,
                easing: Easing::DEFAULT,
            },
            now,
        );
    }

    /// GL JS's `ScrollZoomHandler`: mouse wheels zoom smoothly to a target;
    /// trackpads zoom as they scroll.
    fn wheel(
        &mut self,
        camera: &mut Camera,
        position: DVec2,
        delta: DVec2,
        unit: WheelUnit,
        modifiers: Modifiers,
        now: f64,
    ) -> bool {
        let mut value = match unit {
            WheelUnit::Pixel => delta.y,
            WheelUnit::Line => delta.y * 40.0,
            WheelUnit::Page => delta.y * camera.viewport().height,
        };
        if value == 0.0 {
            return true;
        }
        let since = now - self.last_wheel;
        self.last_wheel = now;
        if value % WHEEL_STEP == 0.0 {
            self.wheel_kind = Some(WheelKind::Wheel);
        } else if value.abs() < 4.0 || modifiers.ctrl {
            // Pinching a trackpad sends Ctrl-wheel events.
            self.wheel_kind = Some(WheelKind::Trackpad);
        } else if since > 400.0 {
            // A new scroll of unknown kind: GL JS waits 40 ms to see; take
            // it as a wheel, which its timeout decides for a lone event.
            self.wheel_kind = None;
        } else if self.wheel_kind.is_none() {
            self.wheel_kind = Some(if (since * value).abs() < 200.0 {
                WheelKind::Trackpad
            } else {
                WheelKind::Wheel
            });
        }
        if modifiers.shift {
            value /= 4.0;
        }
        let kind = self.wheel_kind.unwrap_or(WheelKind::Wheel);
        let rate = if kind == WheelKind::Wheel && value.abs() > WHEEL_STEP {
            WHEEL_ZOOM_RATE
        } else {
            TRACKPAD_ZOOM_RATE
        };
        let mut scale = 2.0 / (1.0 + (-(value * rate).abs()).exp());
        if value > 0.0 {
            scale = 1.0 / scale;
        }
        let zoom = scale.log2();
        match kind {
            WheelKind::Trackpad => {
                if !camera.is_held() {
                    camera.hold();
                }
                camera.zoom_by(position, zoom);
                self.wheel_hold_until = Some(now + TRACKPAD_END);
            }
            WheelKind::Wheel => {
                camera.glide(
                    Glide {
                        zoom: Some(Track {
                            amount: zoom,
                            duration: 200.0,
                            easing: Easing::DEFAULT,
                        }),
                        around: Some(position),
                        ..Glide::default()
                    },
                    now,
                );
            }
        }
        true
    }

    /// GL JS's `KeyboardHandler`: arrows pan, Shift-arrows rotate and
    /// tilt, `+` and `-` zoom; each an eased step.
    fn key(&mut self, camera: &mut Camera, key: Key, modifiers: Modifiers, now: f64) -> bool {
        if key == Key::Escape {
            let boxing = matches!(self.mode, Mode::BoxZoom { .. });
            if boxing {
                self.mode = Mode::Ignored;
            }
            return boxing;
        }
        let state = *camera.state();
        let mut target = CameraTarget {
            center: Some(state.target),
            ..CameraTarget::default()
        };
        let shift = modifiers.shift;
        match (key, shift) {
            (Key::ZoomIn | Key::ZoomOut, _) => {
                let step = if shift { 2.0 } else { 1.0 };
                let sign = if key == Key::ZoomIn { 1.0 } else { -1.0 };
                target.zoom = Some(camera.zoom() + sign * step);
            }
            (Key::Left, true) => target.bearing = Some(state.bearing - KEY_BEARING),
            (Key::Right, true) => target.bearing = Some(state.bearing + KEY_BEARING),
            (Key::Up, true) => target.pitch = Some(state.pitch + KEY_PITCH),
            (Key::Down, true) => target.pitch = Some(state.pitch - KEY_PITCH),
            (Key::Left, false) => target.offset = DVec2::new(KEY_PAN, 0.0),
            (Key::Right, false) => target.offset = DVec2::new(-KEY_PAN, 0.0),
            (Key::Up, false) => target.offset = DVec2::new(0.0, KEY_PAN),
            (Key::Down, false) => target.offset = DVec2::new(0.0, -KEY_PAN),
            (Key::Escape, _) => unreachable!("handled above"),
        }
        camera.user_ease(
            &target,
            EaseOptions {
                duration: 300.0,
                easing: Easing::EASE_OUT,
            },
            now,
        );
        true
    }
}

fn center(camera: &Camera) -> DVec2 {
    camera.state().center(camera.viewport())
}

/// The clockwise angle from `a` to `b` on screen, in degrees.
fn angle(a: DVec2, b: DVec2) -> f64 {
    (a.x * b.y - a.y * b.x).atan2(a.dot(b)).to_degrees()
}

#[cfg(test)]
mod tests;
