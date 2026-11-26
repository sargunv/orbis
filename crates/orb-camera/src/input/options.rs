//! What gestures may move and how they end: MapLibre Compose's
//! `MapInteractions` camera settings (each movement on or off, with its
//! release momentum and settling), and the platform's thresholds, which
//! hosts take from their platform (Android's `ViewConfiguration`, say).

use super::rates::{CLICK_TOLERANCE, LONG_PRESS, TAP_INTERVAL, TAP_TOLERANCE};
use crate::{DecayModel, Easing, MomentumDecay, MomentumSnap};

/// What gestures may move and how releases continue. Changing it cancels
/// a gesture under way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GestureOptions {
    pub pan: Movement,
    pub zoom: Movement,
    /// The bearing.
    pub rotate: Movement,
    pub pitch: Movement,
    pub thresholds: Thresholds,
    /// Levels a double tap zooms in, eased, or out with
    /// [`Modes::zoom_out`](super::Modes::zoom_out): 1 by default, and 0
    /// leaves double taps to the host.
    pub double_tap_zoom: f64,
    /// Levels a two-finger tap zooms, eased, negative out: -1 by default,
    /// and 0 leaves two-finger taps to the host.
    pub two_finger_tap_zoom: f64,
}

impl Default for GestureOptions {
    /// Every movement on, with mln-ffi's default decays; the bearing
    /// settles on north within 7° (Compose's `MapInteractions.Standard`).
    fn default() -> Self {
        Self {
            pan: Movement::new(MomentumDecay::PAN),
            zoom: Movement::new(MomentumDecay::ZOOM),
            rotate: Movement {
                snap: Some(MomentumSnap::BEARING),
                ..Movement::new(MomentumDecay::BEARING)
            },
            pitch: Movement::new(MomentumDecay::PITCH),
            thresholds: Thresholds::default(),
            double_tap_zoom: 1.0,
            two_finger_tap_zoom: -1.0,
        }
    }
}

impl GestureOptions {
    /// Nothing moves the map (Compose's `MapInteractions.None`); clicks
    /// still report.
    pub fn none() -> Self {
        let off = |m: Movement| Movement {
            enabled: false,
            ..m
        };
        let all = Self::default();
        Self {
            pan: off(all.pan),
            zoom: off(all.zoom),
            rotate: off(all.rotate),
            pitch: off(all.pitch),
            ..all
        }
    }
}

/// One camera movement: whether gestures make it, how a release keeps it
/// going, and where it settles after input that changed it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Movement {
    pub enabled: bool,
    /// How a release keeps it moving; without one it stops where the
    /// gesture leaves it.
    pub momentum: Option<Fling>,
    /// Where it settles once input that changed it and its momentum end;
    /// pans don't settle.
    pub snap: Option<MomentumSnap>,
}

impl Movement {
    /// On, slowing by `decay`, settling nowhere.
    pub const fn new(decay: MomentumDecay) -> Self {
        Self {
            enabled: true,
            momentum: Some(Fling::Decay(decay)),
            snap: None,
        }
    }
}

/// How a release keeps a movement going, in its unit: logical pixels for
/// pan, zoom levels, degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fling {
    /// This decay for every release.
    Decay(MomentumDecay),
    /// A glide that lasts `duration` milliseconds × (speed /
    /// `reference_speed`) to the power of `exponent`, along `easing`, a
    /// cubic Bézier's control points, starting at the release's speed, so
    /// faster releases glide longer and farther (Compose's
    /// `MomentumPlanner`). A release slower than `min_speed` stays; one
    /// faster than `max_speed` counts as it; none moves farther than
    /// `max_distance`.
    Curve {
        reference_speed: f64,
        duration: f64,
        exponent: f64,
        easing: [f64; 4],
        min_speed: f64,
        max_speed: f64,
        max_distance: f64,
    },
}

impl Fling {
    /// Android's `OverScroller` fling (`SplineOverScroller`, at 160 dpi so
    /// its pixels are logical pixels), with its platform's limits: at least
    /// 50 px/s (`ViewConfiguration`'s minimum fling velocity), at most
    /// 8000.
    pub const ANDROID: Self = Self::android(50.0, 8000.0);

    /// Android's fling within a platform's limits of speed
    /// (`ViewConfiguration`'s minimum and maximum fling velocity), in
    /// logical pixels a second.
    pub const fn android(min_speed: f64, max_speed: f64) -> Self {
        Self::Curve {
            reference_speed: 1000.0,
            duration: 555.182,
            exponent: 0.7362676,
            easing: [0.175, 0.5, 0.35, 1.0],
            min_speed,
            max_speed,
            max_distance: f64::INFINITY,
        }
    }

    /// The decay a release at `speed` (not negative) continues with, or
    /// `None` when it stays.
    pub(crate) fn decay(&self, speed: f64) -> Option<MomentumDecay> {
        match *self {
            Self::Decay(decay) => (speed >= decay.min_speed).then_some(decay),
            Self::Curve {
                reference_speed,
                duration,
                exponent,
                easing,
                min_speed,
                max_speed,
                max_distance,
            } => {
                if speed.is_nan() || speed < min_speed {
                    return None;
                }
                let speed = speed.min(max_speed);
                let duration = duration * (speed / reference_speed).powf(exponent);
                // The curve starts at slope y1 / x1: the distance whose
                // glide starts at the release's speed.
                let distance = speed * duration / 1000.0 * easing[0] / easing[1];
                if !(distance > 0.0 && distance.is_finite() && duration.is_finite()) {
                    return None;
                }
                Some(MomentumDecay {
                    model: DecayModel::Ease {
                        distance,
                        duration,
                        easing: Easing::Bezier(easing),
                    },
                    min_speed,
                    max_speed: f64::INFINITY,
                    max_distance,
                })
            }
        }
    }
}

/// When presses are clicks, taps pair and holds are long: GL JS's values
/// by default, or the platform's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thresholds {
    /// Logical pixels a press may move and still be a click; drags start
    /// past it (GL JS 3; Android's touch slop is 8 dp).
    pub click_tolerance: f64,
    /// How far apart, in logical pixels, two taps may be to make a double
    /// tap (GL JS 30; Android's double-tap slop is 100 dp).
    pub tap_tolerance: f64,
    /// Milliseconds from one tap to the next for a double tap (GL JS 500;
    /// Android 300).
    pub tap_interval: f64,
    /// Milliseconds a touch or pen holds still for a long press (Compose
    /// and Android 400).
    pub long_press: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            click_tolerance: CLICK_TOLERANCE,
            tap_tolerance: TAP_TOLERANCE,
            tap_interval: TAP_INTERVAL,
            long_press: LONG_PRESS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The distance and duration a fling plans at `speed`.
    fn android_fling(speed: f64) -> Option<(f64, f64)> {
        match Fling::ANDROID.decay(speed)?.model {
            DecayModel::Ease {
                distance, duration, ..
            } => Some((distance, duration)),
            DecayModel::Friction { .. } => None,
        }
    }

    #[test]
    fn android_flings_as_its_overscroller_does() {
        // `SplineOverScroller` at 160 dpi: a 1000 dp/s fling glides 194 dp
        // over 555 ms, a 4000 dp/s one 2157 dp over 1541 ms.
        let (d, t) = android_fling(1000.0).unwrap();
        assert!(
            (d - 194.31).abs() < 0.01 && (t - 555.18).abs() < 0.01,
            "{d} {t}"
        );
        let (d, t) = android_fling(4000.0).unwrap();
        assert!(
            (d - 2156.95).abs() < 0.01 && (t - 1540.68).abs() < 0.01,
            "{d} {t}"
        );
        assert!(Fling::ANDROID.decay(40.0).is_none());
        let fast = Fling::ANDROID.decay(20_000.0).unwrap();
        let capped = Fling::ANDROID.decay(8000.0).unwrap();
        assert_eq!(fast, capped);
    }
}
