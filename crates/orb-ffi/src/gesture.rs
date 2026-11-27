// `gesture.h`: gesture recognizers that move a map's camera.
//
// Each recognizer is orb-camera's `Interaction`, the recognizer Orbis's own
// hosts use, behind a lock on the caller's side: its calls run on the
// caller's thread and answer at once, and the camera moves it makes go to
// the map as one command an input, which applies them to the map's camera
// on the runtime's thread after the commands sent before.
//
// The header's inputs map onto Orbis's: a press's modes, with those of the
// host's held keys, are Orbis's press modes; deltas, platform pinches and
// cancels are Orbis's own; and taps, double taps, two-finger taps and long
// presses come back as its events. What differs is Orbis's: a mouse makes
// no long press (a host shows a context menu itself), so a mouse held past
// 500 ms still taps as it lifts, as a browser's click does; a drag and a
// double tap read their modes as the pointer goes down; options that change
// cancel the gesture under way, letting go of the camera without a glide,
// and end its notches. The recognizer samples the camera for notches from
// the map's snapshot, as the runtime's thread last published it, on each
// input and tick; until the snapshot shows the moves it sent, it takes the
// camera as still moving. In a flipped viewport the host's positions are
// the picture's, upside down: the recognizer takes them turned the right way up, as the
// map's own, a pinch's turn on the screen with them (the delta's bearing
// is the camera's, so it stays), and its taps and box come back turned
// over again.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use glam::DVec2;
use orb_api::{
    Axis, DecayModel, Fling, GestureOptions, GesturePhase, Input, InteractEvent, Interaction,
    Modes, MomentumDecay, MomentumSnap, Movement, Notch, PointerKind, Reading, Tick,
};

use crate::base::{Failure, Result, boundary, fresh_out, lock, put, sized};
use crate::handles::{self, Object};
use crate::map::{Drawn, MapHandle};
use crate::projection::Screen;
use crate::sys::*;

/// A recognizer and the map whose camera it moves.
pub(crate) struct Recognizer {
    map: Arc<MapHandle>,
    state: Mutex<State>,
}

struct State {
    gestures: Interaction,
    /// The modes the host's held keys select, for every press.
    held: Modes,
    /// Events a result had no room for, which the next one reports first.
    pending: VecDeque<mln_gesture_event>,
    /// Each part's glide and snap as the host set them, kept while it
    /// turns them off.
    kept: Kept,
    /// The number of the last command that sent the map moves.
    sent: u64,
}

/// How each part glides (pan, zoom, bearing, pitch) and settles (zoom,
/// bearing, pitch) when on: a part whose glide or snap is off keeps what
/// the host set for it, as the header keeps every setting it isn't sent.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Kept {
    flings: [Fling; 4],
    snaps: [MomentumSnap; 3],
}

impl Kept {
    /// What `options`' parts glide and settle by, or their defaults where
    /// they don't.
    fn of(o: &GestureOptions) -> Self {
        Self {
            flings: [
                (o.pan.momentum, MomentumDecay::PAN),
                (o.zoom.momentum, MomentumDecay::ZOOM),
                (o.rotate.momentum, MomentumDecay::BEARING),
                (o.pitch.momentum, MomentumDecay::PITCH),
            ]
            .map(|(fling, decay)| fling.unwrap_or(Fling::Decay(decay))),
            snaps: [
                (o.zoom.snap, MomentumSnap::ZOOM),
                (o.rotate.snap, MomentumSnap::BEARING),
                (o.pitch.snap, MomentumSnap::PITCH),
            ]
            .map(|(snap, default)| snap.unwrap_or(default)),
        }
    }
}

/// A new recognizer's options: Orbis's, with the header's long press and
/// notch at north.
fn defaults() -> GestureOptions {
    let mut options = GestureOptions::default();
    options.thresholds.long_press = 500.0;
    options.rotate.notches = vec![Notch::NORTH];
    options
}

// MARK: - Options

fn known(fields: u32, all: u32, what: &str) -> Result {
    match fields & !all {
        0 => Ok(()),
        _ => Err(Failure::argument(format!("{what} has unknown bits"))),
    }
}

/// `options` over `current` and what it `kept`, checked.
///
/// # Safety
///
/// Each part's `notches` is null or points to `notch_count` notches.
unsafe fn options(
    o: &mln_gesture_options,
    current: &GestureOptions,
    mut kept: Kept,
) -> Result<(GestureOptions, Kept)> {
    known(
        o.fields,
        MLN_GESTURE_OPTION_DOUBLE_TAP_ZOOM | MLN_GESTURE_OPTION_TWO_FINGER_TAP_ZOOM,
        "mln_gesture_options.fields",
    )?;
    let mut next = current.clone();
    let finite = |v: f64, what: &str| match v.is_finite() {
        true => Ok(v),
        false => Err(Failure::argument(format!("{what} must be finite"))),
    };
    if o.fields & MLN_GESTURE_OPTION_DOUBLE_TAP_ZOOM != 0 {
        next.double_tap_zoom = finite(o.double_tap_zoom, "double_tap_zoom")?;
    }
    if o.fields & MLN_GESTURE_OPTION_TWO_FINGER_TAP_ZOOM != 0 {
        next.two_finger_tap_zoom = finite(o.two_finger_tap_zoom, "two_finger_tap_zoom")?;
    }
    let pan = &o.pan;
    known(
        pan.fields,
        MLN_GESTURE_PAN_OPTION_ENABLED
            | MLN_GESTURE_PAN_OPTION_GLIDES
            | MLN_GESTURE_PAN_OPTION_MOMENTUM,
        "mln_gesture_pan_options.fields",
    )?;
    let has = |bit| pan.fields & bit != 0;
    if has(MLN_GESTURE_PAN_OPTION_ENABLED) {
        next.pan.enabled = pan.enabled;
    }
    let [pan_fling, zoom_fling, bearing_fling, pitch_fling] = &mut kept.flings;
    let [zoom_snap, bearing_snap, pitch_snap] = &mut kept.snaps;
    next.pan.momentum = glide(
        next.pan.momentum,
        pan_fling,
        has(MLN_GESTURE_PAN_OPTION_GLIDES).then_some(pan.glides),
        has(MLN_GESTURE_PAN_OPTION_MOMENTUM).then_some(&pan.momentum),
        MomentumDecay::PAN,
        "pan",
    )?;
    let parts = [
        (
            &o.zoom,
            &mut next.zoom,
            zoom_fling,
            zoom_snap,
            MomentumDecay::ZOOM,
            "zoom",
        ),
        (
            &o.bearing,
            &mut next.rotate,
            bearing_fling,
            bearing_snap,
            MomentumDecay::BEARING,
            "bearing",
        ),
        (
            &o.pitch,
            &mut next.pitch,
            pitch_fling,
            pitch_snap,
            MomentumDecay::PITCH,
            "pitch",
        ),
    ];
    for (options, part, fling, snap, decay, name) in parts {
        // SAFETY: the caller's contract, which covers each part's notches.
        unsafe { axis(options, part, fling, snap, decay, name) }?;
    }
    Ok((next, kept))
}

/// # Safety
///
/// `o.notches` is null or points to `o.notch_count` notches.
unsafe fn axis(
    o: &mln_gesture_axis_options,
    part: &mut Movement,
    kept_fling: &mut Fling,
    kept_snap: &mut MomentumSnap,
    default_decay: MomentumDecay,
    name: &str,
) -> Result {
    known(
        o.fields,
        MLN_GESTURE_AXIS_OPTION_ENABLED
            | MLN_GESTURE_AXIS_OPTION_GLIDES
            | MLN_GESTURE_AXIS_OPTION_MOMENTUM
            | MLN_GESTURE_AXIS_OPTION_SNAPS
            | MLN_GESTURE_AXIS_OPTION_SNAP
            | MLN_GESTURE_AXIS_OPTION_NOTCHES,
        "mln_gesture_axis_options.fields",
    )?;
    let has = |bit| o.fields & bit != 0;
    if has(MLN_GESTURE_AXIS_OPTION_ENABLED) {
        part.enabled = o.enabled;
    }
    part.momentum = glide(
        part.momentum,
        kept_fling,
        has(MLN_GESTURE_AXIS_OPTION_GLIDES).then_some(o.glides),
        has(MLN_GESTURE_AXIS_OPTION_MOMENTUM).then_some(&o.momentum),
        default_decay,
        name,
    )?;
    if has(MLN_GESTURE_AXIS_OPTION_SNAP) {
        *kept_snap = snap(&o.snap, *kept_snap, name)?;
    }
    let snaps = match has(MLN_GESTURE_AXIS_OPTION_SNAPS) {
        true => o.snaps,
        false => part.snap.is_some(),
    };
    part.snap = snaps.then_some(*kept_snap);
    if has(MLN_GESTURE_AXIS_OPTION_NOTCHES) {
        let notches = match o.notch_count {
            0 => &[][..],
            _ if o.notches.is_null() => {
                return Err(Failure::argument(format!("the notches of {name} are null")));
            }
            // SAFETY: non-null, and the caller's contract.
            count => unsafe { std::slice::from_raw_parts(o.notches, count) },
        };
        part.notches = notches
            .iter()
            .map(|n| notch(n, name))
            .collect::<Result<_>>()?;
    }
    Ok(())
}

/// `n`, checked.
fn notch(n: &mln_gesture_notch, part: &str) -> Result<Notch> {
    if !(n.spacing.is_finite() && n.spacing > 0.0 && n.offset.is_finite()) {
        return Err(Failure::argument(format!(
            "a notch of {part} needs a positive, finite spacing and a finite offset"
        )));
    }
    Ok(Notch {
        spacing: n.spacing,
        offset: n.offset,
        kind: n.kind,
    })
}

/// A part's glide: `decay` over the one it `kept`, which it glides by
/// while `glides` (else as it did) keeps it on.
fn glide(
    current: Option<Fling>,
    kept: &mut Fling,
    glides: Option<bool>,
    decay: Option<&mln_momentum_decay>,
    default: MomentumDecay,
    part: &str,
) -> Result<Option<Fling>> {
    if let Some(d) = decay {
        *kept = fling(d, *kept, default, part)?;
    }
    Ok(glides.unwrap_or(current.is_some()).then_some(*kept))
}

/// `d` over `base`. A new model starts from its defaults: friction from
/// the part's, `default`, and a curve from Android's fling.
fn fling(d: &mln_momentum_decay, base: Fling, default: MomentumDecay, part: &str) -> Result<Fling> {
    let fail = |problem: &str| Failure::argument(format!("the momentum of {part} {problem}"));
    let friction_bits = MLN_MOMENTUM_DECAY_HALF_LIFE_MS | MLN_MOMENTUM_DECAY_STOP_SPEED;
    let curve_bits = MLN_MOMENTUM_DECAY_REFERENCE_SPEED
        | MLN_MOMENTUM_DECAY_DURATION_MS
        | MLN_MOMENTUM_DECAY_DURATION_EXPONENT
        | MLN_MOMENTUM_DECAY_EASING;
    let limits = MLN_MOMENTUM_DECAY_MIN_SPEED
        | MLN_MOMENTUM_DECAY_MAX_SPEED
        | MLN_MOMENTUM_DECAY_MAX_DISTANCE;
    known(
        d.fields,
        MLN_MOMENTUM_DECAY_MODEL | friction_bits | curve_bits | limits,
        "mln_momentum_decay.fields",
    )?;
    let has = |bit| d.fields & bit != 0;
    let curve = match (has(MLN_MOMENTUM_DECAY_MODEL), d.model) {
        (false, _) => matches!(base, Fling::Curve { .. }),
        (true, MLN_MOMENTUM_DECAY_MODEL_FRICTION) => false,
        (true, MLN_MOMENTUM_DECAY_MODEL_CURVE) => true,
        (true, _) => return Err(fail("names no decay model")),
    };
    let mut fling = match (curve, base) {
        (false, Fling::Decay(_)) | (true, Fling::Curve { .. }) => base,
        (false, Fling::Curve { .. }) => Fling::Decay(default),
        (true, Fling::Decay(_)) => Fling::ANDROID,
    };
    if d.fields & if curve { friction_bits } else { curve_bits } != 0 {
        return Err(fail("sets settings of the other decay model"));
    }
    let positive = |v: f64, what: &str| match v.is_finite() && v > 0.0 {
        true => Ok(v),
        false => Err(fail(&format!("needs a positive, finite {what}"))),
    };
    // The maximums may be infinite.
    let most = |v: f64, what: &str| match v > 0.0 {
        true => Ok(v),
        false => Err(fail(&format!("needs a positive {what}"))),
    };
    let (min_speed, max_speed, max_distance) = match &mut fling {
        Fling::Decay(decay) => {
            if let DecayModel::Friction {
                half_life,
                stop_speed,
            } = &mut decay.model
            {
                if has(MLN_MOMENTUM_DECAY_HALF_LIFE_MS) {
                    *half_life = positive(d.half_life_ms, "half-life")?;
                }
                if has(MLN_MOMENTUM_DECAY_STOP_SPEED) {
                    *stop_speed = positive(d.stop_speed, "stop speed")?;
                }
            }
            (
                &mut decay.min_speed,
                &mut decay.max_speed,
                &mut decay.max_distance,
            )
        }
        Fling::Curve {
            reference_speed,
            duration,
            exponent,
            easing,
            min_speed,
            max_speed,
            max_distance,
        } => {
            if has(MLN_MOMENTUM_DECAY_REFERENCE_SPEED) {
                *reference_speed = positive(d.reference_speed, "reference speed")?;
            }
            if has(MLN_MOMENTUM_DECAY_DURATION_MS) {
                *duration = positive(d.duration_ms, "duration")?;
            }
            if has(MLN_MOMENTUM_DECAY_DURATION_EXPONENT) {
                let e = d.duration_exponent;
                if !(e.is_finite() && e > -1.0) {
                    return Err(fail("needs a finite duration exponent above -1"));
                }
                *exponent = e;
            }
            if has(MLN_MOMENTUM_DECAY_EASING) {
                let e = d.easing;
                let unit = |v: f64| (0.0..=1.0).contains(&v);
                if !(unit(e.x1)
                    && unit(e.y1)
                    && unit(e.x2)
                    && unit(e.y2)
                    && e.x1 > 0.0
                    && e.y1 > 0.0)
                {
                    return Err(fail(
                        "needs an easing whose control points are within [0, 1], x1 and y1 above 0",
                    ));
                }
                *easing = [e.x1, e.y1, e.x2, e.y2];
            }
            (min_speed, max_speed, max_distance)
        }
    };
    if has(MLN_MOMENTUM_DECAY_MIN_SPEED) {
        if !(d.min_speed.is_finite() && d.min_speed >= 0.0) {
            return Err(fail("needs a finite minimum speed that isn't negative"));
        }
        *min_speed = d.min_speed;
    }
    if has(MLN_MOMENTUM_DECAY_MAX_SPEED) {
        *max_speed = most(d.max_speed, "maximum speed")?;
    }
    if has(MLN_MOMENTUM_DECAY_MAX_DISTANCE) {
        *max_distance = most(d.max_distance, "maximum distance")?;
    }
    Ok(fling)
}

/// `s` over `base`, checked.
fn snap(s: &mln_momentum_snap, base: MomentumSnap, part: &str) -> Result<MomentumSnap> {
    known(
        s.fields,
        MLN_MOMENTUM_SNAP_SPACING
            | MLN_MOMENTUM_SNAP_OFFSET
            | MLN_MOMENTUM_SNAP_TOLERANCE
            | MLN_MOMENTUM_SNAP_DURATION_MS,
        "mln_momentum_snap.fields",
    )?;
    let fail = |problem: &str| Failure::argument(format!("the snap of {part} {problem}"));
    let has = |bit| s.fields & bit != 0;
    let mut out = base;
    if has(MLN_MOMENTUM_SNAP_SPACING) {
        if !(s.spacing.is_finite() && s.spacing > 0.0) {
            return Err(fail("needs a positive, finite spacing"));
        }
        out.spacing = s.spacing;
    }
    if has(MLN_MOMENTUM_SNAP_OFFSET) {
        if !s.offset.is_finite() {
            return Err(fail("needs a finite offset"));
        }
        out.offset = s.offset;
    }
    if has(MLN_MOMENTUM_SNAP_TOLERANCE) {
        if !(s.tolerance.is_finite() && s.tolerance >= 0.0) {
            return Err(fail("needs a finite tolerance that isn't negative"));
        }
        out.tolerance = s.tolerance;
    }
    if has(MLN_MOMENTUM_SNAP_DURATION_MS) {
        if !(s.duration_ms.is_finite() && s.duration_ms >= 0.0) {
            return Err(fail("needs a finite duration that isn't negative"));
        }
        out.duration = s.duration_ms;
    }
    Ok(out)
}

// MARK: - Inputs and results

fn point(p: mln_screen_point, what: &str) -> Result<DVec2> {
    match p.x.is_finite() && p.y.is_finite() {
        true => Ok(DVec2::new(p.x, p.y)),
        false => Err(Failure::argument(format!("{what} must be finite"))),
    }
}

fn modes(bits: mln_gesture_modes) -> Result<Modes> {
    known(
        bits,
        MLN_GESTURE_MODE_ROTATE | MLN_GESTURE_MODE_BOX_ZOOM | MLN_GESTURE_MODE_ZOOM_OUT,
        "the gesture modes",
    )?;
    Ok(Modes {
        rotate: bits & MLN_GESTURE_MODE_ROTATE != 0,
        box_zoom: bits & MLN_GESTURE_MODE_BOX_ZOOM != 0,
        zoom_out: bits & MLN_GESTURE_MODE_ZOOM_OUT != 0,
    })
}

/// What an input is to Orbis's recognizer: `None` for held keys' modes,
/// which `held` takes.
fn input(i: &mln_gesture_input, held: &mut Modes, screen: Screen) -> Result<Option<Input>> {
    let point = |p, what| point(p, what).map(|p| screen.point(p));
    // SAFETY: the union's variant is the one `type_` names, each plain C data.
    unsafe {
        Ok(Some(match i.type_ {
            MLN_GESTURE_INPUT_POINTER_DOWN => {
                let p = i.data.pointer_down;
                let kind = match p.kind {
                    MLN_POINTER_KIND_MOUSE => PointerKind::Mouse,
                    MLN_POINTER_KIND_TOUCH => PointerKind::Touch,
                    MLN_POINTER_KIND_PEN => PointerKind::Pen,
                    _ => return Err(Failure::argument("the pointer kind is unknown")),
                };
                let own = modes(p.modes)?;
                Input::Press {
                    id: p.id,
                    kind,
                    position: point(p.position, "the pointer's position")?,
                    modes: Modes {
                        rotate: own.rotate || held.rotate,
                        box_zoom: own.box_zoom || held.box_zoom,
                        zoom_out: own.zoom_out || held.zoom_out,
                    },
                }
            }
            MLN_GESTURE_INPUT_POINTER_MOVE => Input::PointerMove {
                id: i.data.pointer_move.id,
                position: point(i.data.pointer_move.position, "the pointer's position")?,
            },
            MLN_GESTURE_INPUT_POINTER_UP => Input::PointerUp {
                id: i.data.pointer_up.id,
                position: point(i.data.pointer_up.position, "the pointer's position")?,
            },
            MLN_GESTURE_INPUT_POINTER_CANCEL => Input::PointerCancel {
                id: i.data.pointer_cancel.id,
            },
            MLN_GESTURE_INPUT_MODES => {
                *held = modes(i.data.modes)?;
                return Ok(None);
            }
            MLN_GESTURE_INPUT_DELTA => {
                let d = i.data.delta;
                if !(d.zoom.is_finite() && d.bearing.is_finite() && d.pitch.is_finite()) {
                    return Err(Failure::argument("the delta's values must be finite"));
                }
                Input::Delta {
                    position: point(d.position, "the delta's position")?,
                    discrete: d.discrete,
                    pan: screen.offset(self::point(d.pan, "the delta's pan")?),
                    zoom: d.zoom,
                    bearing: d.bearing,
                    pitch: d.pitch,
                }
            }
            MLN_GESTURE_INPUT_PINCH => {
                let p = i.data.pinch;
                let phase = match p.phase {
                    MLN_PINCH_PHASE_BEGIN => GesturePhase::Begin,
                    MLN_PINCH_PHASE_UPDATE => GesturePhase::Update,
                    MLN_PINCH_PHASE_END => GesturePhase::End,
                    MLN_PINCH_PHASE_CANCEL => GesturePhase::Cancel,
                    _ => return Err(Failure::argument("the pinch phase is unknown")),
                };
                if !(p.scale.is_finite() && p.scale > 0.0 && p.rotation.is_finite()) {
                    return Err(Failure::argument(
                        "the pinch's scale must be positive and its rotation finite",
                    ));
                }
                Input::Pinch {
                    center: point(p.center, "the pinch's center")?,
                    scale: p.scale,
                    rotation: screen.turn(p.rotation),
                    phase,
                }
            }
            _ => return Err(Failure::argument("the gesture input's type is unknown")),
        }))
    }
}

fn tap_event(kind: mln_gesture_event_type, id: u32, at: DVec2) -> mln_gesture_event {
    let tap = mln_gesture_tap {
        id,
        position: mln_screen_point { x: at.x, y: at.y },
    };
    mln_gesture_event {
        type_: kind,
        data: mln_gesture_event_data { tap },
    }
}

/// An event of the recognizer's as the host takes it, its position on
/// `screen`.
fn event(e: InteractEvent, screen: Screen) -> mln_gesture_event {
    let tap_event = |kind, id, at| tap_event(kind, id, screen.point(at));
    match e {
        InteractEvent::Click { id, position } => tap_event(MLN_GESTURE_EVENT_TAP, id, position),
        InteractEvent::DoubleClick { id, position } => {
            tap_event(MLN_GESTURE_EVENT_DOUBLE_TAP, id, position)
        }
        InteractEvent::LongClick { id, position } => {
            tap_event(MLN_GESTURE_EVENT_LONG_PRESS, id, position)
        }
        InteractEvent::TwoFingerTap { position } => {
            tap_event(MLN_GESTURE_EVENT_TWO_FINGER_TAP, 0, position)
        }
        InteractEvent::Notch { axis, value, kind } => mln_gesture_event {
            type_: MLN_GESTURE_EVENT_NOTCH,
            data: mln_gesture_event_data {
                notch: mln_gesture_notch_event {
                    axis: match axis {
                        Axis::Zoom => MLN_GESTURE_AXIS_ZOOM,
                        Axis::Bearing => MLN_GESTURE_AXIS_BEARING,
                        Axis::Pitch => MLN_GESTURE_AXIS_PITCH,
                    },
                    kind,
                    value,
                },
            },
        },
    }
}

impl Recognizer {
    /// The screen of the map as last published: positions turn over in a
    /// flipped viewport.
    fn screen(&self) -> Screen {
        let view = self.map.snapshot().map(|s| s.view);
        view.map_or_else(Screen::default, |v| {
            Screen::of(&v.viewport, f64::from(v.extent.height))
        })
    }

    /// Sends the moves the last call made to the map, as one command.
    fn send(&self, state: &mut State) {
        let moves = state.gestures.take_moves();
        if !moves.is_empty() {
            // A map released or not made takes no moves, as the header says.
            let _ = self.map.change(None, move |entry| {
                entry.apply_gesture(moves);
                Ok(Drawn::New)
            });
            // This command's number, or a later one's.
            state.sent = self.map.posted();
        }
    }

    /// Samples the camera for notches at `now`, as the map's snapshot
    /// shows it: input still moves the camera until the snapshot shows the
    /// moves sent last.
    fn sample(&self, state: &mut State, now: f64) {
        if !state.gestures.watches_notches() {
            return;
        }
        let shown = self.map.shown();
        let Some(snapshot) = self.map.snapshot() else {
            return;
        };
        let camera = snapshot.view.camera;
        let reading = Reading {
            zoom: camera.zoom,
            bearing: camera.bearing,
            pitch: camera.pitch,
            // A released map moves no more, whatever it last published.
            moving: shown != u64::MAX && (snapshot.gesture_in_progress || shown < state.sent),
        };
        state.gestures.sample(reading, now);
    }

    /// Sends the moves the last call made, samples the camera for notches,
    /// and fills `result` with what the recognizer reported.
    fn finish(&self, state: &mut State, handled: bool, now: f64, result: &mut mln_gesture_result) {
        self.send(state);
        self.sample(state, now);
        let screen = self.screen();
        let events = state.gestures.take_events().into_iter();
        let events = events.map(|e| event(e, screen));
        state.pending.extend(events);
        result.handled = handled;
        (result.next_tick, result.next_tick_ms) = match state.gestures.next_tick() {
            Some(Tick::Frame) => (MLN_GESTURE_NEXT_TICK_FRAME, 0.0),
            Some(Tick::At(at)) => (MLN_GESTURE_NEXT_TICK_AT, at),
            None => (MLN_GESTURE_NEXT_TICK_NONE, 0.0),
        };
        let corners = state.gestures.box_zoom();
        result.box_zoom_active = corners.is_some();
        let [a, b] = corners.map_or_default(|c| c.map(|c| screen.point(c)));
        let (min, max) = (a.min(b), a.max(b));
        result.box_zoom = mln_screen_box {
            min: mln_screen_point { x: min.x, y: min.y },
            max: mln_screen_point { x: max.x, y: max.y },
        };
        let count = state.pending.len().min(result.events.len());
        for (slot, e) in result.events.iter_mut().zip(state.pending.drain(..count)) {
            *slot = e;
        }
        result.event_count = count;
    }
}

/// The caller's result struct, checked, with its size kept.
///
/// # Safety
///
/// `out` is null or points to a sized, writable `mln_gesture_result`.
unsafe fn result<'a>(out: *mut mln_gesture_result) -> Result<&'a mut mln_gesture_result> {
    // SAFETY: the caller's contract.
    unsafe { sized(out.cast_const(), "out_result")? };
    // SAFETY: checked above.
    Ok(unsafe { &mut *out })
}

fn time(ms: f64) -> Result<f64> {
    match ms.is_finite() {
        true => Ok(ms),
        false => Err(Failure::argument("time_ms must be finite")),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gesture_recognizer_create(
    map: mln_map,
    options: *const mln_gesture_options,
    out_recognizer: *mut mln_gesture_recognizer,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let map = handles::map(map.0)?;
            let mut gestures = Interaction::default();
            let defaults = defaults();
            let kept = Kept::of(&defaults);
            let (options, kept) = match options.is_null() {
                true => (defaults, kept),
                false => self::options(sized(options, "options")?, &defaults, kept)?,
            };
            gestures.set_options(options);
            fresh_out(out_recognizer, "out_recognizer")?;
            let recognizer = Arc::new(Recognizer {
                map,
                state: Mutex::new(State {
                    gestures,
                    held: Modes::default(),
                    pending: VecDeque::new(),
                    kept,
                    sent: 0,
                }),
            });
            let handle = handles::insert(Object::Gestures(recognizer));
            put(
                out_recognizer,
                mln_gesture_recognizer(handle),
                "out_recognizer",
            )
        })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mln_gesture_recognizer_release(recognizer: mln_gesture_recognizer) {
    let Some(Object::Gestures(recognizer)) = handles::remove(recognizer.0) else {
        return;
    };
    // A gesture under way lets go of the camera, as when the view loses
    // focus; a glide it started keeps going.
    crate::base::survive("mln_gesture_recognizer_release", || {
        let mut state = lock(&recognizer.state);
        if state.gestures.is_active() {
            state.gestures.handle(Input::Blur, 0.0);
            recognizer.send(&mut state);
        }
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gesture_recognizer_set_options(
    recognizer: mln_gesture_recognizer,
    options: *const mln_gesture_options,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let recognizer = handles::gestures(recognizer.0)?;
            let o = sized(options, "options")?;
            let mut state = lock(&recognizer.state);
            let (next, kept) = self::options(o, state.gestures.options(), state.kept)?;
            state.gestures.set_options(next);
            state.kept = kept;
            // A change cancels the gesture under way, letting go of the
            // camera.
            recognizer.send(&mut state);
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gesture_recognizer_handle(
    recognizer: mln_gesture_recognizer,
    input: *const mln_gesture_input,
    time_ms: f64,
    out_result: *mut mln_gesture_result,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let recognizer = handles::gestures(recognizer.0)?;
            let i = sized(input, "input")?;
            let now = time(time_ms)?;
            let result = result(out_result)?;
            let mut state = lock(&recognizer.state);
            let state = &mut *state;
            let handled = match self::input(i, &mut state.held, recognizer.screen())? {
                Some(input) => state.gestures.handle(input, now),
                None => false,
            };
            recognizer.finish(state, handled, now, result);
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn mln_gesture_recognizer_tick(
    recognizer: mln_gesture_recognizer,
    time_ms: f64,
    out_result: *mut mln_gesture_result,
    out_diagnostic: *mut mln_diagnostic,
) -> mln_status {
    // SAFETY: the caller's pointers.
    unsafe {
        boundary(out_diagnostic, || {
            let recognizer = handles::gestures(recognizer.0)?;
            let now = time(time_ms)?;
            let result = result(out_result)?;
            let mut state = lock(&recognizer.state);
            state.gestures.frame(now);
            recognizer.finish(&mut state, false, now, result);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::delivery::testing::{Report, completion};
    use crate::map::{mln_map_get_snapshot, mln_map_release, mln_runtime_create_map};
    use crate::runtime::{mln_runtime_create, mln_runtime_release};

    const WAIT: Duration = Duration::from_secs(10);

    fn snapshot(map: mln_map) -> mln_map_snapshot {
        // SAFETY: plain C data, sized for the call.
        let mut s: mln_map_snapshot = unsafe { std::mem::zeroed() };
        s.size = std::mem::size_of::<mln_map_snapshot>() as u32;
        // SAFETY: a live map and a sized out struct.
        let status = unsafe { mln_map_get_snapshot(map, &mut s, std::ptr::null_mut()) };
        assert_eq!(status, MLN_STATUS_OK);
        s
    }

    /// Waits for the map's snapshot to satisfy `done`, and returns it.
    fn until(map: mln_map, done: impl Fn(&mln_map_snapshot) -> bool) -> mln_map_snapshot {
        let start = Instant::now();
        loop {
            let s = snapshot(map);
            if done(&s) {
                return s;
            }
            assert!(start.elapsed() < WAIT, "the map never got there");
            std::thread::yield_now();
        }
    }

    fn pointer(r: mln_gesture_recognizer, kind: mln_gesture_input_type, x: f64, t: f64) {
        assert!(press(r, kind, x, t, 0).handled);
    }

    fn empty_result() -> mln_gesture_result {
        // SAFETY: plain C data.
        let mut result: mln_gesture_result = unsafe { std::mem::zeroed() };
        result.size = std::mem::size_of::<mln_gesture_result>() as u32;
        result
    }

    /// A mouse's input at `x`, pressing with `modes`.
    fn press(
        r: mln_gesture_recognizer,
        kind: mln_gesture_input_type,
        x: f64,
        t: f64,
        modes: mln_gesture_modes,
    ) -> mln_gesture_result {
        // SAFETY: plain C data.
        let mut input: mln_gesture_input = unsafe { std::mem::zeroed() };
        input.size = std::mem::size_of::<mln_gesture_input>() as u32;
        input.type_ = kind;
        let position = mln_screen_point { x, y: 300.0 };
        input.data = match kind {
            MLN_GESTURE_INPUT_POINTER_DOWN => mln_gesture_input_data {
                pointer_down: mln_gesture_pointer_down {
                    id: 1,
                    kind: MLN_POINTER_KIND_MOUSE,
                    position,
                    modes,
                },
            },
            _ => mln_gesture_input_data {
                pointer_move: mln_gesture_pointer { id: 1, position },
            },
        };
        let mut result = empty_result();
        // SAFETY: a live recognizer, a sized input and result.
        let status = unsafe {
            mln_gesture_recognizer_handle(r, &input, t, &mut result, std::ptr::null_mut())
        };
        assert_eq!(status, MLN_STATUS_OK);
        result
    }

    fn tick(r: mln_gesture_recognizer, t: f64) -> mln_gesture_result {
        let mut result = empty_result();
        // SAFETY: a live recognizer and a sized result.
        let status =
            unsafe { mln_gesture_recognizer_tick(r, t, &mut result, std::ptr::null_mut()) };
        assert_eq!(status, MLN_STATUS_OK);
        result
    }

    /// The notches a result reported: axis, value and kind.
    fn notches(result: &mln_gesture_result) -> Vec<(mln_gesture_axis, f64, u32)> {
        let events = &result.events[..result.event_count];
        let notches = events.iter().filter(|e| e.type_ == MLN_GESTURE_EVENT_NOTCH);
        // SAFETY: a notch event's data is its notch.
        let notches = notches.map(|e| unsafe { e.data.notch });
        notches.map(|n| (n.axis, n.value, n.kind)).collect()
    }

    /// A runtime and an 800 × 600 map on it, with a recognizer of the
    /// default options.
    fn recognizer() -> (mln_runtime, mln_map, mln_gesture_recognizer) {
        let mut runtime = mln_runtime(0);
        // SAFETY: default options, a fresh out handle.
        let status =
            unsafe { mln_runtime_create(std::ptr::null(), &mut runtime, std::ptr::null_mut()) };
        assert_eq!(status, MLN_STATUS_OK);
        // SAFETY: plain C data.
        let mut options: mln_map_options = unsafe { std::mem::zeroed() };
        options.size = std::mem::size_of::<mln_map_options>() as u32;
        options.initial_extent = mln_logical_extent {
            width: 800,
            height: 600,
            scale_factor: 1.0,
        };
        let (c, made) = completion();
        let mut map = mln_map(0);
        // SAFETY: a live runtime, sized options, a fresh out handle.
        let status = unsafe {
            mln_runtime_create_map(runtime, &options, &mut map, &c, std::ptr::null_mut())
        };
        assert_eq!(status, MLN_STATUS_OK);
        assert_eq!(
            made.recv_timeout(WAIT).unwrap(),
            Report::Result(MLN_STATUS_OK, String::new())
        );
        let mut recognizer = mln_gesture_recognizer(0);
        // SAFETY: a live map, default options, a fresh out handle.
        let status = unsafe {
            mln_gesture_recognizer_create(
                map,
                std::ptr::null(),
                &mut recognizer,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(status, MLN_STATUS_OK);
        (runtime, map, recognizer)
    }

    fn release(runtime: mln_runtime, map: mln_map, recognizer: mln_gesture_recognizer) {
        mln_gesture_recognizer_release(recognizer);
        let (c, released) = completion();
        // SAFETY: a live map, a valid completion.
        assert_eq!(
            unsafe { mln_map_release(map, &c, std::ptr::null_mut()) },
            MLN_STATUS_OK
        );
        assert_eq!(
            released.recv_timeout(WAIT).unwrap(),
            Report::Result(MLN_STATUS_OK, String::new())
        );
        let (c, released) = completion();
        // SAFETY: a live runtime without maps, a valid completion.
        assert_eq!(
            unsafe { mln_runtime_release(runtime, &c, std::ptr::null_mut()) },
            MLN_STATUS_OK
        );
        assert_eq!(
            released.recv_timeout(WAIT).unwrap(),
            Report::Result(MLN_STATUS_OK, String::new())
        );
    }

    fn gesture_options() -> mln_gesture_options {
        // SAFETY: plain C data.
        let mut o: mln_gesture_options = unsafe { std::mem::zeroed() };
        o.size = std::mem::size_of::<mln_gesture_options>() as u32;
        o
    }

    /// A part whose glide and snap are off stays off when sent only their
    /// settings, and glides and settles by them once turned back on.
    #[test]
    fn a_glide_turned_off_keeps_its_settings_until_it_is_on() {
        let defaults = defaults();
        let mut off = gesture_options();
        off.zoom.fields = MLN_GESTURE_AXIS_OPTION_GLIDES | MLN_GESTURE_AXIS_OPTION_SNAPS;
        // SAFETY: no notches.
        let (now, kept) = unsafe { options(&off, &defaults, Kept::of(&defaults)) }.unwrap();
        assert_eq!((now.zoom.momentum, now.zoom.snap), (None, None));

        let mut settings = gesture_options();
        settings.zoom.fields = MLN_GESTURE_AXIS_OPTION_MOMENTUM | MLN_GESTURE_AXIS_OPTION_SNAP;
        settings.zoom.momentum.fields = MLN_MOMENTUM_DECAY_HALF_LIFE_MS;
        settings.zoom.momentum.half_life_ms = 300.0;
        settings.zoom.snap.fields = MLN_MOMENTUM_SNAP_TOLERANCE;
        settings.zoom.snap.tolerance = 0.25;
        // SAFETY: no notches.
        let (now, kept) = unsafe { options(&settings, &now, kept) }.unwrap();
        assert_eq!((now.zoom.momentum, now.zoom.snap), (None, None));

        let mut on = gesture_options();
        on.zoom.fields = MLN_GESTURE_AXIS_OPTION_GLIDES | MLN_GESTURE_AXIS_OPTION_SNAPS;
        on.zoom.glides = true;
        on.zoom.snaps = true;
        // SAFETY: no notches.
        let (now, _) = unsafe { options(&on, &now, kept) }.unwrap();
        let Some(Fling::Decay(decay)) = now.zoom.momentum else {
            panic!("the zoom glides by friction: {:?}", now.zoom.momentum);
        };
        assert!(matches!(
            decay.model,
            DecayModel::Friction { half_life, .. } if half_life == 300.0
        ));
        assert_eq!(now.zoom.snap.map(|s| s.tolerance), Some(0.25));
    }

    /// A drag through a recognizer moves the map's camera as a gesture,
    /// and its release glides on before the gesture ends.
    #[test]
    fn a_flung_drag_moves_the_map_and_glides_on() {
        let (runtime, map, recognizer) = recognizer();
        let before = snapshot(map);
        let start = before.view.camera.center.longitude;

        pointer(recognizer, MLN_GESTURE_INPUT_POINTER_DOWN, 100.0, 0.0);
        for step in 1..=6 {
            let x = 100.0 + 5.0 * f64::from(step);
            pointer(
                recognizer,
                MLN_GESTURE_INPUT_POINTER_MOVE,
                x,
                16.0 * f64::from(step),
            );
        }
        // Each input sends its moves as one command, which grows the map's
        // generation, as its steps on the runtime's thread may too.
        let held = until(map, |s| s.generation >= before.generation + 7);
        assert!(held.gesture_in_progress, "the drag is a gesture");
        let dragged = held.view.camera.center.longitude;
        pointer(recognizer, MLN_GESTURE_INPUT_POINTER_UP, 130.0, 100.0);
        let rest = until(map, |s| !s.gesture_in_progress);
        // The content moved right, so the center went west, and on past
        // where the drag left it, by less than a turn.
        let end = rest.view.camera.center.longitude;
        assert!(dragged < start, "{dragged} {start}");
        assert!(end < dragged, "it glides on: {end} {dragged}");
        release(runtime, map, recognizer);
    }

    /// A twist that leaves north and comes back reports the header's
    /// default notch there once, sampled from the map's snapshots, and
    /// asks for frames until its gesture ends.
    #[test]
    fn a_twist_back_to_north_reports_its_notch() {
        let (runtime, map, recognizer) = recognizer();
        let bearing = |b: f64| move |s: &mln_map_snapshot| (s.view.camera.bearing - b).abs() < 1e-6;
        let mut reported = Vec::new();
        let mut t = 0.0;
        let drag =
            |kind, x: f64, t: f64| notches(&press(recognizer, kind, x, t, MLN_GESTURE_MODE_ROTATE));
        reported.extend(drag(MLN_GESTURE_INPUT_POINTER_DOWN, 300.0, t));
        // Half a degree a pixel: to 20° and back.
        for x in [310.0, 320.0, 330.0, 340.0] {
            t += 16.0;
            reported.extend(drag(MLN_GESTURE_INPUT_POINTER_MOVE, x, t));
        }
        until(map, bearing(20.0));
        t += 16.0;
        let away = tick(recognizer, t);
        assert_eq!(away.next_tick, MLN_GESTURE_NEXT_TICK_FRAME);
        reported.extend(notches(&away));
        for x in [330.0, 320.0, 310.0, 300.0] {
            t += 16.0;
            reported.extend(drag(MLN_GESTURE_INPUT_POINTER_MOVE, x, t));
        }
        until(map, bearing(0.0));
        t += 100.0;
        reported.extend(drag(MLN_GESTURE_INPUT_POINTER_UP, 300.0, t));
        let start = Instant::now();
        loop {
            t += 16.0;
            let result = tick(recognizer, t);
            reported.extend(notches(&result));
            if result.next_tick == MLN_GESTURE_NEXT_TICK_NONE {
                break;
            }
            assert!(start.elapsed() < WAIT, "its watch never ended");
            std::thread::yield_now();
        }
        assert_eq!(reported, [(MLN_GESTURE_AXIS_BEARING, 0.0, 0)]);
        release(runtime, map, recognizer);
    }
}
