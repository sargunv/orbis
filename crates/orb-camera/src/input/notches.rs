//! Notches: where input moving the zoom, bearing or pitch reports
//! [`InteractEvent::Notch`] for haptic feedback, as MapLibre Compose's
//! `BearingHapticDetector` does at north, for every part (mln-rs's
//! `mln_gestures` notches, with their capture and rearm distances).
//!
//! The recognizer keeps no camera, so its host samples it
//! ([`Interaction::sample`](super::Interaction::sample)): a watch runs from
//! the input that starts moving the camera until the host says input no
//! longer moves it, and each sample checks the path from the last.

use super::{Axis, GestureOptions, InteractEvent, Notch};

/// The most steps of one [`Notch`] a sample checks: a move that crosses
/// more reports one event all the same.
const MAX_CROSSED: i64 = 64;
/// Feedback comes at most this often on a part, in milliseconds; a notch
/// reached sooner still disarms, so it can't play late.
const MIN_INTERVAL: f64 = 50.0;

impl Axis {
    /// How near a notch a value reaches it, and how far it then moves away
    /// before the notch rearms, in the part's unit: Compose's 1° and 2° for
    /// the bearing, and the pitch too; the zoom's are mln-rs's.
    fn capture_and_rearm(self) -> (f64, f64) {
        match self {
            Self::Zoom => (0.02, 0.04),
            Self::Bearing | Self::Pitch => (1.0, 2.0),
        }
    }

    fn circular(self) -> bool {
        self == Self::Bearing
    }
}

/// One notch of a part: its [`Notch`]'s index in the part's list, and its
/// step from the offset (within a turn, for the bearing).
type Key = (usize, i64);

/// The notches of one part.
#[derive(Debug, Default)]
struct PartWatch {
    /// The value at the last sample.
    last: Option<f64>,
    /// Notches reached and not yet left far enough to rearm.
    disarmed: Vec<Key>,
    /// When the last event came, on the inputs' clock.
    last_event: Option<f64>,
}

/// Watches the camera's zoom, bearing and pitch while input moves it. A
/// notch the first sample finds the camera at doesn't report until the
/// value leaves it and comes back.
#[derive(Debug, Default)]
pub(super) struct NotchWatch {
    parts: [PartWatch; 3],
}

impl NotchWatch {
    /// Samples the parts' `values` (in [`Axis::ALL`] order) at `now`, and
    /// pushes an event for each part that reached an armed notch since the
    /// last sample.
    pub(super) fn sample(
        &mut self,
        values: [f64; 3],
        options: &GestureOptions,
        now: f64,
        events: &mut Vec<InteractEvent>,
    ) {
        for ((axis, state), to) in Axis::ALL.into_iter().zip(&mut self.parts).zip(values) {
            if !to.is_finite() {
                continue;
            }
            let part = options.part(axis);
            let notches = &part.notches[..];
            let (capture, rearm) = axis.capture_and_rearm();
            let Some(from) = state.last.replace(to) else {
                state.disarmed = reached(axis, notches, to, 0.0, capture).collect();
                continue;
            };
            if !part.enabled || notches.is_empty() {
                continue;
            }
            let travel = match axis.circular() {
                true => bearing_delta(from, to),
                false => to - from,
            };
            if travel == 0.0 {
                // Resting on a notch reports it once.
                continue;
            }
            let mut first = None;
            for key in reached(axis, notches, from, travel, capture) {
                if !state.disarmed.contains(&key) {
                    state.disarmed.push(key);
                    first.get_or_insert(key);
                }
            }
            state.disarmed.retain(|&(index, step)| {
                let value = notch_value(axis, &notches[index], step);
                distance(axis, value, to) <= rearm
            });
            let Some((index, step)) = first else {
                continue;
            };
            if state.last_event.is_some_and(|at| now - at < MIN_INTERVAL) {
                continue;
            }
            state.last_event = Some(now);
            let notch = &notches[index];
            events.push(InteractEvent::Notch {
                axis,
                value: notch_value(axis, notch, step),
                kind: notch.kind,
            });
        }
    }
}

/// The notches a move from `from` by `travel` reaches, within `capture` of
/// its path, in the order of the part's list.
fn reached(
    axis: Axis,
    notches: &[Notch],
    from: f64,
    travel: f64,
    capture: f64,
) -> impl Iterator<Item = Key> + '_ {
    let low = from + travel.min(0.0) - capture;
    let high = from + travel.max(0.0) + capture;
    notches.iter().enumerate().flat_map(move |(index, notch)| {
        steps(axis, notch, low, high)
            .into_iter()
            .map(move |step| (index, step))
    })
}

/// The steps of `notch` from `low` to `high`, at most [`MAX_CROSSED`]. On
/// the bearing's circle, the steps lie within one turn from the offset, as
/// snaps' do.
fn steps(axis: Axis, notch: &Notch, low: f64, high: f64) -> Vec<i64> {
    // Saturating float-to-int casts keep a huge range finite.
    let between = |low: f64, high: f64, min: f64, max: f64| {
        let first = ((low - notch.offset) / notch.spacing).ceil().max(min) as i64;
        let end = ((high - notch.offset) / notch.spacing).floor().min(max) as i64;
        first..=end.min(first.saturating_add(MAX_CROSSED - 1))
    };
    if !axis.circular() {
        return between(low, high, f64::MIN, f64::MAX).collect();
    }
    // The steps within a turn, over each turn the range spans.
    let last = ((360.0 / notch.spacing).ceil() - 1.0).max(0.0);
    let first_turn = ((low - notch.offset) / 360.0).floor() as i64;
    let last_turn = ((high - notch.offset) / 360.0).floor() as i64;
    let mut steps = Vec::new();
    for turn in first_turn..=last_turn {
        let shift = 360.0 * turn as f64;
        steps.extend(between(low - shift, high - shift, 0.0, last));
        if steps.len() >= MAX_CROSSED as usize {
            break;
        }
    }
    steps
}

/// The value of `notch`'s step; on the bearing's circle, in [0°, 360°).
fn notch_value(axis: Axis, notch: &Notch, step: i64) -> f64 {
    let value = notch.offset + step as f64 * notch.spacing;
    match axis.circular() {
        true => value.rem_euclid(360.0),
        false => value,
    }
}

fn distance(axis: Axis, a: f64, b: f64) -> f64 {
    match axis.circular() {
        true => bearing_delta(a, b).abs(),
        false => (a - b).abs(),
    }
}

/// The shorter turn from `from` to `to`, in [-180°, 180°).
fn bearing_delta(from: f64, to: f64) -> f64 {
    (to - from + 180.0).rem_euclid(360.0) - 180.0
}
