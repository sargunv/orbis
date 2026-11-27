//! Gestures: pointer, touch, wheel and keyboard input turned into
//! camera moves. Host-independent: hosts feed [`Input`]s with their
//! timestamps and call [`Interaction::frame`] when it asks for time
//! ([`Interaction::next_tick`]). The recognizer keeps no camera: it says
//! what to do as [`Move`]s ([`Interaction::take_moves`]), which the
//! [`Camera`] store resolves against itself as it applies them
//! ([`Camera::apply`]), reporting the camera events, so its camera may live
//! elsewhere, as a C API map's does on the runtime's thread. Every release
//! continues with the store's [`GestureMomentum`].
//!
//! The gesture set is MapLibre Compose's: drag to pan; right- or
//! Ctrl-drag to rotate and tilt; Shift-drag to zoom to a box; wheel and
//! trackpad zoom; double-click or double-tap to zoom in, Shift-double-click
//! or a two-finger tap to zoom out; double-tap-and-drag to zoom; two-finger
//! pinch, twist and vertical tilt; the arrow, `+`/`=` and `-` keys. Each
//! gesture's rates and thresholds are mln-rs's `mln_gestures` (a port of
//! this module, whose README says where each comes from), so the two feel
//! alike. A release goes on at the average velocity of its last 100 ms
//! ([`velocity`]), as mln-ffi's demo and GL JS take it; the rest of it is
//! Compose's: a glide with mln-ffi's default decays, a glide the gesture
//! caught added to one going the same way, and the bearing settling on
//! north within 7°. Hosts say whether a scroll is a wheel's notches or a
//! trackpad's pixels ([`WheelDelta`]). A touch held still
//! is a long click, as a right click is. Hosts whose platform recognizes
//! trackpad pinches itself (macOS, Safari, Windows precision touchpads,
//! libinput) hand them over as [`Input::Pinch`], which zooms, turns and
//! glides as a touch pinch does.
//!
//! [`GestureOptions`] turns each movement on or off, with its momentum,
//! settling and notches, and takes the platform's thresholds. Notches
//! report where input moves the zoom, bearing or pitch to or across a
//! value, for haptic feedback; the host samples the camera for them
//! ([`Interaction::sample`]).

mod moves;
mod notches;
mod options;
mod rates;
mod velocity;

pub use moves::Move;
use moves::{Plan, Step};
use notches::NotchWatch;
pub use options::{Axis, Fling, GestureOptions, Movement, Notch, Thresholds};
use rates::*;
use velocity::VelocityTracker;

use crate::momentum::Delta;
#[cfg(doc)]
use crate::{Camera, GestureMomentum};
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

/// How far a scroll went, as its host classifies it. Positive scrolls
/// down, as the DOM's `deltaY` does, which zooms out.
///
/// The host knows best which device scrolled (macOS's
/// `hasPreciseScrollingDeltas`, the DOM's `deltaMode`, the platform's wheel
/// units), so the gestures don't guess from the deltas' sizes and timing,
/// which GL JS does and which read a slow trackpad start as a wheel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WheelDelta {
    /// Detents of a mouse wheel, fractional for high-resolution wheels:
    /// each zooms by a fixed amount, eased.
    Notches(f64),
    /// Continuous scrolling, as trackpads and precise wheels send, in
    /// logical pixels: zooms at once, as the platform already smoothed it.
    Pixels(f64),
}

/// Where a platform-recognized gesture is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GesturePhase {
    Begin,
    Update,
    /// The fingers lifted: it glides on.
    End,
    /// The platform took it away: nothing continues.
    Cancel,
}

/// What a one-pointer drag or a double tap does instead of its default.
/// A press reads them as it goes down, and keeps them until it lifts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Modes {
    /// Sideways turns the map and up and down tilts it.
    pub rotate: bool,
    /// Draws a box the map zooms to when the pointer lifts.
    pub box_zoom: bool,
    /// A double tap zooms out instead of in.
    pub zoom_out: bool,
}

/// Input in logical pixels from the viewport's top-left corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Input {
    /// A pointer went down: a secondary button rotates and its click is a
    /// long click, Ctrl rotates and Shift zooms to a box or out, as
    /// [`Input::Press`]'s modes.
    PointerDown {
        id: u32,
        kind: PointerKind,
        button: Button,
        position: DVec2,
        modifiers: Modifiers,
    },
    /// A pointer went down with the modes its host chose for it, as the C
    /// API's hosts choose them from buttons and held keys.
    Press {
        id: u32,
        kind: PointerKind,
        position: DVec2,
        modes: Modes,
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
    /// A scroll at `position`. With Ctrl, which browsers add to a
    /// trackpad pinch, pixels zoom at the pinch rate.
    Wheel {
        position: DVec2,
        delta: WheelDelta,
        modifiers: Modifiers,
    },
    /// A pinch the platform recognized (macOS magnify and rotate, Safari's
    /// `gesture*` events, libinput's pinch): each update scales by `scale`
    /// (above 1 spreads, zooming in) and turns the fingers `rotation`
    /// degrees clockwise since the last, around `center`.
    Pinch {
        center: DVec2,
        scale: f64,
        rotation: f64,
        phase: GesturePhase,
    },
    Key {
        key: Key,
        modifiers: Modifiers,
    },
    /// Camera movement from a wheel, a dial, keys or buttons, which the
    /// host measured in camera units: how far the map's content moves on
    /// screen, levels of zoom and degrees clockwise of bearing and of
    /// pitch, zooms and turns keeping `position` in place. A discrete one,
    /// such as a wheel's notch, eases, and a quick run of them merges into
    /// one ease that speeds up; a continuous one, such as a trackpad's
    /// scroll, moves at once, and holds the camera until a while after the
    /// last.
    Delta {
        position: DVec2,
        discrete: bool,
        pan: DVec2,
        zoom: f64,
        bearing: f64,
        pitch: f64,
    },
    /// The view lost focus: every pointer is released.
    Blur,
}

/// What a gesture means beyond the camera's own events, with the pointer
/// that made it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InteractEvent {
    /// A primary press and release that didn't move the map: a click or
    /// a tap.
    Click { id: u32, position: DVec2 },
    /// The second click of a double click, which zooms as
    /// [`GestureOptions::double_tap_zoom`] says.
    DoubleClick { id: u32, position: DVec2 },
    /// A touch or pen held still, or a right click: the place's context
    /// (Compose's `longClick`, GL JS's `contextmenu`).
    LongClick { id: u32, position: DVec2 },
    /// Two fingers tapped together, which zooms as
    /// [`GestureOptions::two_finger_tap_zoom`] says.
    TwoFingerTap { position: DVec2 },
    /// Input moved a part of the camera to or across one of its
    /// [`Movement::notches`], at `value` (a bearing from 0 up to 360°), of
    /// the notch's `kind`. A part reports at most one notch every 50 ms.
    Notch { axis: Axis, value: f64, kind: u32 },
}

/// What a host reads of its camera for notches
/// ([`Interaction::sample`]); a [`Camera`] store gives it
/// ([`Camera::reading`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reading {
    pub zoom: f64,
    /// Degrees clockwise from north.
    pub bearing: f64,
    pub pitch: f64,
    /// Whether input still moves the camera: holds it, or a glide, snap or
    /// eased step it started runs.
    pub moving: bool,
}

/// When the recognizer next needs [`Interaction::frame`] and
/// [`Interaction::sample`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tick {
    /// Every display frame: input moves the camera, which may reach a
    /// notch.
    Frame,
    /// At this time on the inputs' clock: when a press becomes a long
    /// press, or a continuous delta's hold ends.
    At(f64),
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
        modes: Modes,
        /// A secondary button: its click is a long click.
        context: bool,
        /// A touch or pen held long enough: a long click, not a click.
        long: bool,
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
    /// A pinch the platform recognized.
    Platform,
    /// Buttons or pointers the map doesn't use are down.
    Ignored,
}

/// One applied gesture step, for the release velocity.
#[derive(Clone, Copy, Debug, Default)]
struct Sample {
    pan: DVec2,
    zoom: f64,
    bearing: f64,
    pitch: f64,
    around: Option<DVec2>,
}

/// Per second, at release.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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
    /// Whether the gesture changed the zoom, bearing and pitch, which may
    /// settle.
    changed: [bool; 3],
    /// A pinch's release, staged when it continued as a one-finger pan,
    /// and when.
    staged: Option<(f64, Release)>,
}

impl Tracking {
    fn reset(&mut self, now: f64) {
        *self = Self::default();
        self.record(now, Sample::default());
    }

    /// Adds a step's samples: to each part it moves, and to each part the
    /// gesture moves already, which stands still otherwise.
    fn record(&mut self, now: f64, step: Sample) {
        self.pan += step.pan;
        self.zoom += step.zoom;
        self.bearing += step.bearing;
        self.pitch += step.pitch;
        if step.zoom != 0.0 {
            self.zoom_anchor = step.around;
            self.changed[0] = true;
        }
        if step.bearing != 0.0 {
            self.bearing_anchor = step.around;
            self.changed[1] = true;
        }
        if step.pitch != 0.0 {
            self.changed[2] = true;
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

/// The gesture recognizers for one map.
#[derive(Debug)]
pub struct Interaction {
    pointers: Vec<Pointer>,
    mode: Mode,
    tracking: Tracking,
    /// The last tap, for double taps: where and when.
    last_tap: Option<(DVec2, f64, PointerKind)>,
    /// A continuous scroll's zoom holds the camera until then.
    wheel_hold_until: Option<f64>,
    events: Vec<InteractEvent>,
    /// The camera moves made and not yet taken.
    moves: Vec<Move>,
    options: GestureOptions,
    /// The notches input moving the camera may reach, from its first move
    /// until the host's [`Reading`] says input moves it no more.
    notches: Option<NotchWatch>,
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            pointers: Vec::new(),
            mode: Mode::Idle,
            tracking: Tracking::default(),
            last_tap: None,
            wheel_hold_until: None,
            events: Vec::new(),
            moves: Vec::new(),
            options: GestureOptions::default(),
            notches: None,
        }
    }
}

impl Interaction {
    pub fn options(&self) -> &GestureOptions {
        &self.options
    }

    /// Sets what gestures may move. A change cancels the gesture under
    /// way, with no click and no momentum, and reports no more notches of
    /// what input started before; pointers still down, or a platform pinch
    /// under way, do nothing until they end.
    pub fn set_options(&mut self, options: GestureOptions) {
        if options == self.options {
            return;
        }
        self.options = options;
        self.last_tap = None;
        self.notches = None;
        if self.is_active() {
            self.mode = if self.pointers.is_empty() && !matches!(self.mode, Mode::Platform) {
                Mode::Idle
            } else {
                Mode::Ignored
            };
            self.wheel_hold_until = None;
            self.camera(Step::Release);
        }
    }

    /// The camera moves made since the last call, in order, for the camera
    /// to [apply](crate::Camera::apply).
    pub fn take_moves(&mut self) -> Vec<Move> {
        std::mem::take(&mut self.moves)
    }

    fn camera(&mut self, step: Step) {
        match step {
            // What the platform took away, and a box zoom's ease, report no
            // notches.
            Step::Cancel | Step::ZoomToBox(_) => self.notches = None,
            Step::Hold { .. }
            | Step::Scroll
            | Step::Delta(_)
            | Step::TapZoom { .. }
            | Step::Key { .. }
                if self.notches.is_none() && self.options.has_notches() =>
            {
                self.notches = Some(NotchWatch::default());
            }
            _ => {}
        }
        self.moves.push(Move(step));
    }

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

    /// Clicks, double clicks and notches recognized since the last call.
    pub fn take_events(&mut self) -> Vec<InteractEvent> {
        std::mem::take(&mut self.events)
    }

    /// When a press held still becomes a long press, while one may.
    fn long_press_at(&self) -> Option<f64> {
        match (self.mode, &self.pointers[..]) {
            (
                Mode::Pressed {
                    context: false,
                    long: false,
                    ..
                },
                [pointer],
            ) if pointer.kind != PointerKind::Mouse => {
                Some(pointer.down_at + self.options.thresholds.long_press)
            }
            _ => None,
        }
    }

    /// When the recognizer next needs [`frame`](Self::frame) and
    /// [`sample`](Self::sample): every display frame while input moving the
    /// camera may reach a notch, else when a press becomes a long press or
    /// a continuous delta's hold ends; `None` until the next input.
    pub fn next_tick(&self) -> Option<Tick> {
        // A press held still moves nothing that could reach a notch.
        let still = matches!(
            self.mode,
            Mode::Pressed { .. } | Mode::TapDrag { moved: false, .. }
        );
        if self.notches.is_some() && !still {
            return Some(Tick::Frame);
        }
        let at = match (self.long_press_at(), self.wheel_hold_until) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        at.map(Tick::At)
    }

    /// Checks the camera, as `reading` finds it at `now`, for the notches
    /// input moving it reached since the last sample, which
    /// [`take_events`](Self::take_events) reports. Hosts call it once a
    /// display frame with the camera the frame shows, and may call it
    /// before applying an input's moves too, so a gesture's notches start
    /// from where the camera stood, and after, so an input that leaves the
    /// camera still ends the watch. A notch the camera stands at as input
    /// starts moving it reports once it leaves and comes back.
    pub fn sample(&mut self, reading: Reading, now: f64) {
        let Some(watch) = &mut self.notches else {
            return;
        };
        let values = [reading.zoom, reading.bearing, reading.pitch];
        watch.sample(values, &self.options, now, &mut self.events);
        // Input moves the camera no more: the gesture, its glide and its
        // snap, or its eased step, ended.
        if !reading.moving && !self.is_active() {
            self.notches = None;
        }
    }

    /// Whether input moving the camera may reach a notch, so
    /// [`sample`](Self::sample) has something to check: a host that reads
    /// its camera at a cost can skip the reading while not.
    pub fn watches_notches(&self) -> bool {
        self.notches.is_some()
    }

    /// Ends gestures that time out and recognizes long presses. Call at
    /// [`next_tick`](Self::next_tick), or once a display frame, with the
    /// time; returns whether it needs more time to do so.
    pub fn frame(&mut self, now: f64) -> bool {
        let mut waiting = false;
        if let Some(at) = self.long_press_at() {
            if now < at {
                waiting = true;
            } else {
                self.long_press();
            }
        }
        if let Some(until) = self.wheel_hold_until {
            if now >= until {
                self.wheel_hold_until = None;
                if matches!(self.mode, Mode::Idle) {
                    self.camera(Step::Release);
                }
            } else {
                waiting = true;
            }
        }
        waiting
    }

    /// The press under way is a long press.
    fn long_press(&mut self) {
        if let Mode::Pressed { long, .. } = &mut self.mode
            && let [pointer] = self.pointers[..]
        {
            *long = true;
            self.last_tap = None;
            self.events.push(InteractEvent::LongClick {
                id: pointer.id,
                position: pointer.position,
            });
        }
    }

    /// Handles one input at host time `now` (milliseconds). Returns whether
    /// the map used it, so hosts can stop the platform's default.
    pub fn handle(&mut self, input: Input, now: f64) -> bool {
        match input {
            Input::PointerDown {
                id,
                kind,
                button,
                position,
                modifiers,
            } => {
                let touch = kind == PointerKind::Touch;
                if !touch && button == Button::Middle {
                    return false;
                }
                let modes = Modes {
                    rotate: button == Button::Secondary || (modifiers.ctrl && !touch),
                    box_zoom: modifiers.shift && !touch && button == Button::Primary,
                    zoom_out: modifiers.shift && !touch,
                };
                self.down(id, kind, position, modes, button != Button::Primary, now)
            }
            Input::Press {
                id,
                kind,
                position,
                modes,
            } => self.down(id, kind, position, modes, false, now),
            Input::PointerMove { id, position } => self.moved(id, position, now),
            Input::PointerUp { id, position } => self.up(id, Some(position), now),
            Input::PointerCancel { id } => self.up(id, None, now),
            Input::Wheel {
                position,
                delta,
                modifiers,
            } => self.wheel(position, delta, modifiers, now),
            Input::Pinch {
                center,
                scale,
                rotation,
                phase,
            } => self.platform_pinch(center, scale, rotation, phase, now),
            Input::Key { key, modifiers } => self.key(key, modifiers),
            Input::Delta {
                position,
                discrete,
                pan,
                zoom,
                bearing,
                pitch,
            } => self.delta(position, discrete, pan, [zoom, bearing, pitch], now),
            Input::Blur => {
                self.pointers.clear();
                self.mode = Mode::Idle;
                self.camera(Step::Release);
                true
            }
        }
    }

    /// A pointer down with `modes`; `context` for a secondary button.
    fn down(
        &mut self,
        id: u32,
        kind: PointerKind,
        position: DVec2,
        modes: Modes,
        context: bool,
        now: f64,
    ) -> bool {
        if self.pointers.iter().any(|p| p.id == id) {
            return false;
        }
        let touch = kind == PointerKind::Touch;
        if !touch && !self.pointers.is_empty() {
            return false;
        }
        if matches!(self.mode, Mode::Platform) {
            // A touch during a platform pinch takes over from it.
            self.finish(now, End::Cancel);
        }
        let zoom_enabled = self.options.zoom.enabled;
        let pitch_enabled = self.options.pitch.enabled;
        let tap_interval = self.options.thresholds.tap_interval;
        self.pointers.push(Pointer {
            id,
            kind,
            position,
            down: position,
            down_at: now,
        });
        let first = matches!(self.mode, Mode::Idle) && self.pointers.len() == 1;
        self.camera(Step::Hold { first });
        match (self.mode, self.pointers.len()) {
            (Mode::Idle, 1) => {
                self.mode = if touch && zoom_enabled && self.follows_tap(kind, position, now) {
                    Mode::TapDrag {
                        origin: position,
                        moved: false,
                    }
                } else {
                    Mode::Pressed {
                        modes: Modes {
                            box_zoom: modes.box_zoom && zoom_enabled,
                            ..modes
                        },
                        context,
                        long: false,
                    }
                };
                self.tracking.reset(now);
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
                    pitch: (vector.y.abs() > vector.x.abs() || !pitch_enabled).then_some(false),
                    single_since: None,
                    panning: false,
                    tap: now - self.pointers[0].down_at < tap_interval
                        && matches!(self.mode, Mode::Pressed { .. }),
                });
                self.tracking.record(now, Sample::default());
            }
            _ => self.mode = Mode::Ignored,
        }
        true
    }

    fn moved(&mut self, id: u32, position: DVec2, now: f64) -> bool {
        let Some(index) = self.pointers.iter().position(|p| p.id == id) else {
            return false;
        };
        let last = self.pointers[index].position;
        self.pointers[index].position = position;
        match &mut self.mode {
            Mode::Pressed { modes, .. } => {
                let down = self.pointers[index].down;
                if position.distance(down) > self.options.thresholds.click_tolerance {
                    self.mode = match (modes.rotate, modes.box_zoom) {
                        (true, _) => Mode::Rotate,
                        (false, true) => Mode::BoxZoom { origin: down },
                        (false, false) => Mode::Pan,
                    };
                    self.step(down, position, now);
                }
            }
            Mode::Pan | Mode::Rotate => self.step(last, position, now),
            Mode::BoxZoom { .. } => {}
            Mode::TapDrag { origin, moved } => {
                let (origin, dy) = (*origin, position.y - last.y);
                *moved =
                    *moved || position.distance(origin) > self.options.thresholds.click_tolerance;
                if *moved {
                    let zoom = dy / 128.0;
                    self.camera(Step::ZoomBy {
                        around: origin,
                        levels: zoom,
                    });
                    self.tracking.record(
                        now,
                        Sample {
                            zoom,
                            around: Some(origin),
                            ..Sample::default()
                        },
                    );
                }
            }
            Mode::Pinch(_) if self.pointers.len() < 2 => {
                // The other finger of a two-finger tap lifted, and this one
                // moves on: a pan.
                self.mode = Mode::Pan;
                self.step(last, position, now);
            }
            Mode::Pinch(pinch) => {
                if index > 1 {
                    return true;
                }
                let mut pinch = *pinch;
                let points = [self.pointers[0].position, self.pointers[1].position];
                self.pinch_step(&mut pinch, points, now);
                self.mode = Mode::Pinch(pinch);
            }
            Mode::Idle | Mode::Ignored | Mode::Platform => {}
        }
        true
    }

    /// One pan or mouse-rotate step from `from` to `to`.
    fn step(&mut self, from: DVec2, to: DVec2, now: f64) {
        let delta = to - from;
        match self.mode {
            Mode::Rotate => {
                let on = |movement: &Movement, v: f64| if movement.enabled { v } else { 0.0 };
                let bearing = on(&self.options.rotate, delta.x * DRAG_ROTATE);
                let pitch = on(&self.options.pitch, -delta.y * DRAG_PITCH);
                if bearing != 0.0 {
                    self.camera(Step::RotateBy {
                        around: None,
                        degrees: bearing,
                    });
                }
                if pitch != 0.0 {
                    self.camera(Step::PitchBy(pitch));
                }
                self.tracking.record(
                    now,
                    Sample {
                        bearing,
                        pitch,
                        ..Sample::default()
                    },
                );
            }
            Mode::BoxZoom { .. } => {}
            _ if !self.options.pan.enabled => {}
            _ => {
                self.camera(Step::Pan { from, to });
                self.tracking.record(
                    now,
                    Sample {
                        pan: delta,
                        ..Sample::default()
                    },
                );
            }
        }
    }

    fn pinch_step(&mut self, pinch: &mut Pinch, points: [DVec2; 2], now: f64) {
        let [a, b] = points;
        let [la, lb] = pinch.last;
        let (va, vb) = (a - la, b - lb);
        let vector = b - a;
        let options = &self.options;
        let tolerance = options.thresholds.tap_tolerance;
        if pinch.tap
            && (a.distance(self.pointers[0].down) > tolerance
                || b.distance(self.pointers[1].down) > tolerance)
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
            self.camera(Step::PitchBy(pitch));
            pinch.last = points;
            pinch.tap = false;
            self.tracking.record(
                now,
                Sample {
                    pitch,
                    ..Sample::default()
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
                > options.thresholds.click_tolerance
        {
            pinch.panning = true;
        }
        let pan = if pinch.panning && options.pan.enabled {
            mid - last_mid
        } else {
            DVec2::ZERO
        };
        let zoom = if pinch.zooming && last_distance > 0.0 && options.zoom.enabled {
            (distance / last_distance).log2()
        } else {
            0.0
        };
        // Fingers turning clockwise turn the map clockwise.
        let bearing = if pinch.rotating && options.rotate.enabled {
            -angle(lb - la, vector)
        } else {
            0.0
        };
        if pan != DVec2::ZERO {
            self.camera(Step::Pan {
                from: last_mid,
                to: mid,
            });
        }
        if zoom != 0.0 {
            self.camera(Step::ZoomBy {
                around: mid,
                levels: zoom,
            });
        }
        if bearing != 0.0 {
            self.camera(Step::RotateBy {
                around: Some(mid),
                degrees: bearing,
            });
        }
        pinch.last = points;
        if pan != DVec2::ZERO || zoom != 0.0 || bearing != 0.0 {
            pinch.tap = false;
            self.tracking.record(
                now,
                Sample {
                    pan,
                    zoom,
                    bearing,
                    around: Some(mid),
                    ..Sample::default()
                },
            );
        }
    }

    fn up(&mut self, id: u32, position: Option<DVec2>, now: f64) -> bool {
        let Some(index) = self.pointers.iter().position(|p| p.id == id) else {
            return false;
        };
        let pointer = self.pointers.remove(index);
        let cancelled = position.is_none();
        if cancelled && self.pointers.is_empty() && !matches!(self.mode, Mode::Idle | Mode::Ignored)
        {
            self.camera(Step::Cancel);
        }
        match self.mode {
            Mode::Pinch(pinch) => {
                if !self.pointers.is_empty() {
                    // Lifting one finger continues as a one-finger pan;
                    // lifting the second ends a two-finger tap.
                    let tap =
                        pinch.tap && now - pointer.down_at < self.options.thresholds.tap_interval;
                    self.mode = if tap {
                        Mode::Pinch(pinch)
                    } else {
                        self.tracking.stage(now);
                        Mode::Pan
                    };
                    return true;
                }
                if pinch.tap && !cancelled {
                    let position = (pinch.last[0] + pinch.last[1]) / 2.0;
                    self.events.push(InteractEvent::TwoFingerTap { position });
                    self.zoom_step(position, self.options.two_finger_tap_zoom);
                }
                let end = match (cancelled, pinch.tap) {
                    (true, _) => End::Cancel,
                    (false, true) => End::Settle,
                    (false, false) => End::Glide,
                };
                self.finish(now, end);
            }
            Mode::Pan if !self.pointers.is_empty() => {}
            Mode::Pressed {
                modes,
                context,
                long,
            } => {
                if self.pointers.is_empty() {
                    // A press held long enough with no frame to time it is
                    // a long press as it lifts.
                    let held = !context
                        && pointer.kind != PointerKind::Mouse
                        && now - pointer.down_at >= self.options.thresholds.long_press;
                    match (cancelled || long, held) {
                        (true, _) => {}
                        (false, true) => {
                            self.last_tap = None;
                            self.events.push(InteractEvent::LongClick {
                                id: pointer.id,
                                position: pointer.position,
                            });
                        }
                        (false, false) => self.tap(pointer, modes, context, now),
                    }
                    self.finish(now, End::Settle);
                }
            }
            Mode::BoxZoom { origin } => {
                if !cancelled {
                    self.camera(Step::ZoomToBox([
                        origin,
                        position.unwrap_or(pointer.position),
                    ]));
                }
                self.finish(now, End::Cancel);
            }
            Mode::TapDrag { origin, moved } => {
                if self.pointers.is_empty() {
                    self.last_tap = None;
                    if !moved && !cancelled {
                        self.events.push(InteractEvent::DoubleClick {
                            id: pointer.id,
                            position: origin,
                        });
                        self.zoom_step(origin, self.options.double_tap_zoom);
                    }
                    let end = match (cancelled, moved) {
                        (true, _) => End::Cancel,
                        (false, true) => End::Glide,
                        (false, false) => End::Settle,
                    };
                    self.finish(now, end);
                }
            }
            Mode::Pan | Mode::Rotate => {
                self.finish(now, if cancelled { End::Cancel } else { End::Glide });
            }
            Mode::Ignored | Mode::Idle | Mode::Platform => {
                if self.pointers.is_empty() {
                    self.finish(now, End::Cancel);
                }
            }
        }
        true
    }

    /// A press released without moving: a click, perhaps the second of a
    /// double click.
    fn tap(&mut self, pointer: Pointer, modes: Modes, context: bool, now: f64) {
        let (id, position) = (pointer.id, pointer.position);
        if context {
            // A right click is a long click (Compose's) or a context menu
            // (GL JS's).
            self.events.push(InteractEvent::LongClick { id, position });
            return;
        }
        self.events.push(InteractEvent::Click { id, position });
        if now - pointer.down_at > self.options.thresholds.tap_interval {
            return;
        }
        if self.follows_tap(pointer.kind, position, now) {
            self.last_tap = None;
            self.events
                .push(InteractEvent::DoubleClick { id, position });
            let zoom = self.options.double_tap_zoom;
            self.zoom_step(position, if modes.zoom_out { -zoom } else { zoom });
        } else {
            self.last_tap = Some((position, now, pointer.kind));
        }
    }

    /// Ends the gesture with every pointer up: a release glides on with
    /// the speed its last moves had, and settles.
    fn finish(&mut self, now: f64, end: End) {
        self.mode = Mode::Idle;
        if end != End::Cancel {
            let plan = Plan {
                release: self.tracking.release(now),
                glide: end == End::Glide,
                options: self.options.clone(),
                changed: self.tracking.changed,
            };
            self.camera(Step::Momentum(Box::new(plan)));
        }
        if self.wheel_hold_until.is_none() {
            self.camera(Step::Release);
        }
    }

    /// Whether a press of `kind` at `position` is the second of a double tap.
    fn follows_tap(&self, kind: PointerKind, position: DVec2, now: f64) -> bool {
        let thresholds = &self.options.thresholds;
        self.last_tap.is_some_and(|(at, time, k)| {
            k == kind
                && now - time < thresholds.tap_interval
                && at.distance(position) < thresholds.tap_tolerance
        })
    }

    /// Double-click, double-tap and two-finger-tap zoom: `levels` around a
    /// point, eased (GL JS, 300 ms); none when zero.
    fn zoom_step(&mut self, at: DVec2, levels: f64) {
        if self.options.zoom.enabled && levels != 0.0 {
            self.camera(Step::TapZoom { at, levels });
        }
    }

    /// A scroll zooms around where the pointer is: a wheel's notches by
    /// a fixed amount each, eased (a notch while earlier ones ease adds to
    /// what they have left, from the speed they had), and continuous
    /// pixels by their distance at once, faster with Ctrl, which browsers
    /// add to a trackpad pinch (mln-rs's `mln_gestures`). A scroll the map
    /// doesn't zoom with isn't used, so the page scrolls.
    fn wheel(
        &mut self,
        position: DVec2,
        delta: WheelDelta,
        modifiers: Modifiers,
        now: f64,
    ) -> bool {
        if !self.options.zoom.enabled {
            return false;
        }
        // A platform pinch already zooms: a trackpad pinch some browsers
        // also send as Ctrl-wheel events (Safari's `gesture*` events
        // beside them, if it does) would zoom twice.
        if modifiers.ctrl && matches!(self.mode, Mode::Platform) {
            return true;
        }
        match delta {
            WheelDelta::Pixels(pixels) => {
                let rate = match modifiers.ctrl {
                    true => PINCH_ZOOM_RATE,
                    false => SCROLL_ZOOM_RATE,
                };
                let zoom = (-pixels * rate).clamp(-MAX_SCROLL_ZOOM, MAX_SCROLL_ZOOM);
                self.delta(position, false, DVec2::ZERO, [zoom, 0.0, 0.0], now);
            }
            WheelDelta::Notches(notches) => {
                let zoom = -notches * NOTCH_ZOOM;
                self.delta(position, true, DVec2::ZERO, [zoom, 0.0, 0.0], now);
            }
        }
        true
    }

    /// A delta in camera units ([`Input::Delta`]): each part the options
    /// let move, eased if `discrete`, else at once, holding the camera
    /// until a while after. Returns whether any part moved.
    fn delta(
        &mut self,
        position: DVec2,
        discrete: bool,
        pan: DVec2,
        [zoom, bearing, pitch]: [f64; 3],
        now: f64,
    ) -> bool {
        let options = &self.options;
        let on = |movement: &Movement, v: f64| {
            (movement.enabled && v.is_finite() && v != 0.0).then_some(v)
        };
        let pan = Some(pan).filter(|p| options.pan.enabled && p.is_finite() && *p != DVec2::ZERO);
        let (zoom, bearing, pitch) = (
            on(&options.zoom, zoom),
            on(&options.rotate, bearing),
            on(&options.pitch, pitch),
        );
        if pan.is_none() && zoom.is_none() && bearing.is_none() && pitch.is_none() {
            return false;
        }
        if discrete {
            self.camera(Step::Delta(Box::new(Delta {
                pan: pan.unwrap_or_default(),
                zoom: zoom.unwrap_or_default(),
                bearing: bearing.unwrap_or_default(),
                pitch: pitch.unwrap_or_default(),
                anchor: position,
            })));
            return true;
        }
        self.camera(Step::Scroll);
        if let Some(pan) = pan {
            self.camera(Step::Pan {
                from: position,
                to: position + pan,
            });
        }
        if let Some(levels) = zoom {
            self.camera(Step::ZoomBy {
                around: position,
                levels,
            });
        }
        if let Some(degrees) = bearing {
            self.camera(Step::RotateBy {
                around: Some(position),
                degrees,
            });
        }
        if let Some(degrees) = pitch {
            self.camera(Step::PitchBy(degrees));
        }
        self.wheel_hold_until = Some(now + SCROLL_IDLE);
        true
    }

    /// GL JS's `KeyboardHandler`: arrows pan, Shift-arrows rotate and
    /// tilt, `+` and `-` zoom; each an eased step.
    fn key(&mut self, key: Key, modifiers: Modifiers) -> bool {
        if key == Key::Escape {
            let boxing = matches!(self.mode, Mode::BoxZoom { .. });
            if boxing {
                self.mode = Mode::Ignored;
            }
            return boxing;
        }
        let shift = modifiers.shift;
        let movement = match (key, shift) {
            (Key::ZoomIn | Key::ZoomOut, _) => &self.options.zoom,
            (Key::Left | Key::Right, true) => &self.options.rotate,
            (Key::Up | Key::Down, true) => &self.options.pitch,
            _ => &self.options.pan,
        };
        if !movement.enabled {
            return false;
        }
        let (mut zoom, mut bearing, mut pitch, mut offset) = (0.0, 0.0, 0.0, DVec2::ZERO);
        match (key, shift) {
            (Key::ZoomIn | Key::ZoomOut, _) => {
                let step = if shift { 2.0 } else { 1.0 };
                let sign = if key == Key::ZoomIn { 1.0 } else { -1.0 };
                zoom = sign * step;
            }
            (Key::Left, true) => bearing = -KEY_BEARING,
            (Key::Right, true) => bearing = KEY_BEARING,
            (Key::Up, true) => pitch = KEY_PITCH,
            (Key::Down, true) => pitch = -KEY_PITCH,
            (Key::Left, false) => offset = DVec2::new(KEY_PAN, 0.0),
            (Key::Right, false) => offset = DVec2::new(-KEY_PAN, 0.0),
            (Key::Up, false) => offset = DVec2::new(0.0, KEY_PAN),
            (Key::Down, false) => offset = DVec2::new(0.0, -KEY_PAN),
            (Key::Escape, _) => unreachable!("handled above"),
        }
        self.camera(Step::Key {
            zoom,
            bearing,
            pitch,
            offset,
        });
        true
    }

    /// A pinch the platform recognized: it zooms and turns around its
    /// center as a touch pinch does, and glides on when it ends.
    fn platform_pinch(
        &mut self,
        center: DVec2,
        scale: f64,
        rotation: f64,
        phase: GesturePhase,
        now: f64,
    ) -> bool {
        let pinching = matches!(self.mode, Mode::Platform);
        match phase {
            GesturePhase::End | GesturePhase::Cancel => {
                if !pinching {
                    // The end of one the options cancelled.
                    if matches!(self.mode, Mode::Ignored) && self.pointers.is_empty() {
                        self.mode = Mode::Idle;
                    }
                    return false;
                }
                let end = if phase == GesturePhase::End {
                    End::Glide
                } else {
                    self.camera(Step::Cancel);
                    End::Cancel
                };
                self.finish(now, end);
                return true;
            }
            // Pointers down have the map; an update without a begin
            // begins.
            GesturePhase::Begin | GesturePhase::Update if !pinching => {
                if !matches!(self.mode, Mode::Idle) {
                    return false;
                }
                self.camera(Step::Hold { first: true });
                self.tracking.reset(now);
                self.mode = Mode::Platform;
                if phase == GesturePhase::Begin {
                    return true;
                }
            }
            GesturePhase::Begin | GesturePhase::Update => {}
        }
        let zoom = if self.options.zoom.enabled && scale > 0.0 && scale.is_finite() {
            scale.log2()
        } else {
            0.0
        };
        // Fingers turning clockwise turn the map clockwise.
        let bearing = if self.options.rotate.enabled && rotation.is_finite() {
            -rotation
        } else {
            0.0
        };
        if zoom != 0.0 {
            self.camera(Step::ZoomBy {
                around: center,
                levels: zoom,
            });
        }
        if bearing != 0.0 {
            self.camera(Step::RotateBy {
                around: Some(center),
                degrees: bearing,
            });
        }
        if zoom != 0.0 || bearing != 0.0 {
            self.tracking.record(
                now,
                Sample {
                    zoom,
                    bearing,
                    around: Some(center),
                    ..Sample::default()
                },
            );
        }
        true
    }
}

/// The clockwise angle from `a` to `b` on screen, in degrees.
fn angle(a: DVec2, b: DVec2) -> f64 {
    (a.x * b.y - a.y * b.x).atan2(a.dot(b)).to_degrees()
}

#[cfg(test)]
mod tests;
