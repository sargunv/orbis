//! Gestures: pointer, touch, wheel and keyboard input turned into
//! camera moves, with inertia and feel profiles. Host-independent: hosts
//! feed [`Input`]s with their timestamps and call [`Interaction::frame`]
//! once a display frame; every move goes through the
//! [`Camera`](orb_camera::Camera) store, which reports the camera events.
//!
//! GL JS's handlers are the reference for what each gesture does and at
//! what rate: drag to pan, right- or Ctrl-drag to rotate and tilt, wheel
//! and trackpad zoom, double-click and double-tap zoom, two-finger tap to
//! zoom out, tap-and-drag zoom, two-finger pinch, twist and vertical tilt,
//! and the arrow and `+`/`-` keys.

#![forbid(unsafe_code)]

mod feel;

pub use feel::{Feel, Inertia};

use std::collections::VecDeque;

use glam::DVec2;
use orb_camera::{Camera, CameraTarget, EaseOptions, Easing, Glide, Track};

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
    /// capture): no click and no inertia.
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
    },
    Pan,
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

/// One applied gesture step, for inertia.
#[derive(Clone, Copy, Debug, Default)]
struct Sample {
    time: f64,
    pan: DVec2,
    zoom: f64,
    bearing: f64,
    pitch: f64,
    around: Option<DVec2>,
}

impl Sample {
    fn at(time: f64) -> Self {
        Self {
            time,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WheelKind {
    Wheel,
    Trackpad,
}

/// GL JS's mouse-wheel step, which trackpads never produce exactly.
const WHEEL_STEP: f64 = 4.000_244_140_625;
/// GL JS's inertia sample buffer and velocity window, in milliseconds.
const BUFFER_CUTOFF: f64 = 160.0;
const VELOCITY_WINDOW: f64 = 60.0;
/// Trackpad zoom stays one move until this long after its last event.
const TRACKPAD_END: f64 = 200.0;

/// The gesture recognizers for one map.
#[derive(Debug)]
pub struct Interaction {
    feel: Feel,
    pointers: Vec<Pointer>,
    mode: Mode,
    samples: VecDeque<Sample>,
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
        Self::new(Feel::default())
    }
}

impl Interaction {
    pub fn new(feel: Feel) -> Self {
        Self {
            feel,
            pointers: Vec::new(),
            mode: Mode::Idle,
            samples: VecDeque::new(),
            last_tap: None,
            wheel_kind: None,
            last_wheel: f64::NEG_INFINITY,
            wheel_hold_until: None,
            events: Vec::new(),
        }
    }

    pub fn set_feel(&mut self, feel: Feel) {
        self.feel = feel;
    }

    /// A gesture is under way: hosts keep pointer capture and scrolling off.
    pub fn is_active(&self) -> bool {
        !matches!(self.mode, Mode::Idle) || self.wheel_hold_until.is_some()
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
                self.samples.clear();
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
        camera.hold();
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
                    }
                };
                self.samples.clear();
                self.record(Sample::at(now));
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
                    tap: now - self.pointers[0].down_at < self.feel.tap_interval
                        && matches!(self.mode, Mode::Pressed { .. }),
                });
                self.record(Sample::at(now));
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
            Mode::Pressed { rotate, .. } => {
                if position.distance(self.pointers[index].down) > self.feel.click_tolerance {
                    let rotate = *rotate;
                    self.mode = if rotate { Mode::Rotate } else { Mode::Pan };
                    let down = self.pointers[index].down;
                    self.step(camera, down, position, now);
                }
            }
            Mode::Pan | Mode::Rotate => self.step(camera, last, position, now),
            Mode::TapDrag { origin, moved } => {
                let (origin, dy) = (*origin, position.y - last.y);
                *moved = *moved || position.distance(origin) > self.feel.click_tolerance;
                if *moved {
                    let zoom = dy / 128.0;
                    camera.zoom_by(origin, zoom);
                    self.record(Sample {
                        time: now,
                        zoom,
                        around: Some(origin),
                        ..Sample::default()
                    });
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
                let bearing = delta.x * self.feel.drag_rotate;
                let pitch = -delta.y * self.feel.drag_pitch;
                let center = center(camera);
                camera.rotate_by(center, bearing);
                camera.pitch_by(pitch);
                self.record(Sample {
                    time: now,
                    bearing,
                    pitch,
                    ..Sample::default()
                });
            }
            _ => {
                camera.pan(from, to);
                self.record(Sample {
                    time: now,
                    pan: delta,
                    ..Sample::default()
                });
            }
        }
    }

    fn pinch_step(&mut self, camera: &mut Camera, pinch: &mut Pinch, points: [DVec2; 2], now: f64) {
        let [a, b] = points;
        let [la, lb] = pinch.last;
        let (va, vb) = (a - la, b - lb);
        let vector = b - a;
        let feel = &self.feel;
        if pinch.tap
            && (a.distance(self.pointers[0].down) > feel.tap_tolerance
                || b.distance(self.pointers[1].down) > feel.tap_tolerance)
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
            let pitch = (va.y + vb.y) / 2.0 * -feel.drag_pitch;
            camera.pitch_by(pitch);
            pinch.last = points;
            pinch.tap = false;
            self.record(Sample {
                time: now,
                pitch,
                ..Sample::default()
            });
            return;
        }
        let (mid, last_mid) = ((a + b) / 2.0, (la + lb) / 2.0);
        let distance = vector.length();
        let last_distance = (lb - la).length();
        pinch.min_diameter = pinch.min_diameter.min(distance);
        if !pinch.zooming
            && (distance / pinch.start_distance).log2().abs() >= feel.pinch_zoom_threshold
        {
            pinch.zooming = true;
        }
        if !pinch.rotating {
            let threshold =
                feel.rotate_threshold / (std::f64::consts::PI * pinch.min_diameter) * 360.0;
            if angle(pinch.start_vector, vector).abs() >= threshold {
                pinch.rotating = true;
            }
        }
        if !pinch.panning
            && (mid - (self.pointers[0].down + self.pointers[1].down) / 2.0).length()
                > feel.click_tolerance
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
            self.record(Sample {
                time: now,
                pan,
                zoom,
                bearing,
                around: Some(mid),
                ..Sample::default()
            });
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
                    let tap = pinch.tap && now - pointer.down_at < self.feel.tap_interval;
                    self.mode = if tap {
                        Mode::Pinch(pinch)
                    } else {
                        self.record(Sample::at(now));
                        Mode::Pan
                    };
                    return true;
                }
                if pinch.tap && !cancelled {
                    let mid = (pinch.last[0] + pinch.last[1]) / 2.0;
                    self.zoom_step(camera, mid, -1.0, now);
                }
                self.finish(camera, now, !cancelled && !pinch.tap);
            }
            Mode::Pan if !self.pointers.is_empty() => {}
            Mode::Pressed { button, .. } => {
                if self.pointers.is_empty() {
                    if !cancelled {
                        self.tap(camera, pointer, button, now);
                    }
                    self.finish(camera, now, false);
                }
            }
            Mode::TapDrag { origin, moved } => {
                if self.pointers.is_empty() {
                    self.last_tap = None;
                    if !moved && !cancelled {
                        self.events
                            .push(InteractEvent::DoubleClick { position: origin });
                        self.zoom_step(camera, origin, 1.0, now);
                    }
                    self.finish(camera, now, moved && !cancelled);
                }
            }
            Mode::Pan | Mode::Rotate => self.finish(camera, now, !cancelled),
            Mode::Ignored | Mode::Idle => {
                if self.pointers.is_empty() {
                    self.finish(camera, now, false);
                }
            }
        }
        true
    }

    /// A press released without moving: a click, perhaps the second of a
    /// double click.
    fn tap(&mut self, camera: &mut Camera, pointer: Pointer, button: Button, now: f64) {
        let position = pointer.position;
        self.events.push(InteractEvent::Click { position, button });
        if button != Button::Primary || now - pointer.down_at > self.feel.tap_interval {
            return;
        }
        if self.follows_tap(pointer.kind, position, now) {
            self.last_tap = None;
            self.events.push(InteractEvent::DoubleClick { position });
            self.zoom_step(camera, position, 1.0, now);
        } else {
            self.last_tap = Some((position, now, pointer.kind));
        }
    }

    /// Ends the gesture with every pointer up, gliding on with what its
    /// last moves had of speed.
    fn finish(&mut self, camera: &mut Camera, now: f64, inertia: bool) {
        self.mode = Mode::Idle;
        if inertia && let Some(glide) = self.inertia(camera, now) {
            camera.glide(glide, now);
        }
        self.samples.clear();
        if self.wheel_hold_until.is_none() {
            camera.release();
        }
    }

    /// Whether a press of `kind` at `position` is the second of a double tap.
    fn follows_tap(&self, kind: PointerKind, position: DVec2, now: f64) -> bool {
        self.last_tap.is_some_and(|(at, time, k)| {
            k == kind
                && now - time < self.feel.tap_interval
                && at.distance(position) < self.feel.tap_tolerance
        })
    }

    fn record(&mut self, sample: Sample) {
        self.prune(sample.time);
        self.samples.push_back(sample);
    }

    /// Drops samples older than the inertia buffer at `now`.
    fn prune(&mut self, now: f64) {
        while self
            .samples
            .front()
            .is_some_and(|s| now - s.time > BUFFER_CUTOFF)
        {
            self.samples.pop_front();
        }
    }

    /// GL JS's `HandlerInertia._onMoveEnd`: speed over the last moves up
    /// to the release, so a gesture held still before release glides less.
    fn inertia(&mut self, camera: &Camera, now: f64) -> Option<Glide> {
        self.prune(now);
        let samples: Vec<_> = self.samples.iter().copied().collect();
        let mut first = samples.len().saturating_sub(2);
        while first > 0 && samples[first - 1].time >= now - VELOCITY_WINDOW {
            first -= 1;
        }
        let window = &samples[first..];
        if window.len() < 2 {
            return None;
        }
        let duration = now - window[0].time;
        let mut total = Sample::default();
        for s in window {
            total.around = s.around.or(total.around);
        }
        for s in &window[1..] {
            total.pan += s.pan;
            total.zoom += s.zoom;
            total.bearing += s.bearing;
            total.pitch += s.pitch;
        }
        let feel = &self.feel;
        let mut glide = Glide {
            around: total.around,
            ..Glide::default()
        };
        if total.pan.length() > 0.0 {
            let (amount, duration, easing) = feel.pan_inertia.ease(total.pan.length(), duration);
            // Never glide past the horizon, where the pan would turn back
            // (GL JS `handlePanInertia`).
            let view = camera.view();
            let horizon = (camera.viewport().height / 2.0 - view.sky_horizon()).abs();
            let length = amount.abs().min(horizon * 0.75);
            glide.pan = Some(Track {
                amount: total.pan.normalize() * length,
                duration,
                easing,
            });
        }
        let track = |inertia: &Inertia, amount: f64| {
            (amount != 0.0).then(|| {
                let (amount, duration, easing) = inertia.ease(amount, duration);
                Track {
                    amount,
                    duration,
                    easing,
                }
            })
        };
        glide.zoom = track(&feel.zoom_inertia, total.zoom);
        glide.bearing = track(&feel.bearing_inertia, total.bearing).map(|mut t| {
            t.amount = t.amount.clamp(-179.0, 179.0);
            t
        });
        glide.pitch = track(&feel.pitch_inertia, total.pitch);
        (glide.pan.is_some()
            || glide.zoom.is_some()
            || glide.bearing.is_some()
            || glide.pitch.is_some())
        .then_some(glide)
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
            self.feel.wheel_zoom_rate
        } else {
            self.feel.trackpad_zoom_rate
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
        let feel = &self.feel;
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
            (Key::Left, true) => target.bearing = Some(state.bearing - feel.key_bearing),
            (Key::Right, true) => target.bearing = Some(state.bearing + feel.key_bearing),
            (Key::Up, true) => target.pitch = Some(state.pitch + feel.key_pitch),
            (Key::Down, true) => target.pitch = Some(state.pitch - feel.key_pitch),
            (Key::Left, false) => target.offset = DVec2::new(feel.key_pan, 0.0),
            (Key::Right, false) => target.offset = DVec2::new(-feel.key_pan, 0.0),
            (Key::Up, false) => target.offset = DVec2::new(0.0, feel.key_pan),
            (Key::Down, false) => target.offset = DVec2::new(0.0, -feel.key_pan),
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
