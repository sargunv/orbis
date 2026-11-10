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
    /// The distance and duration Android's `OverScroller` gives a fling
    /// at the release's speed, along its curve (Compose's
    /// `MomentumPlanner.AndroidFling`), planning logical pixels as dp. A
    /// release slower than `min_speed` stays; one faster than `max_speed`
    /// counts as it; none moves farther than `max_distance`.
    Android {
        min_speed: f64,
        max_speed: f64,
        max_distance: f64,
    },
}

impl Fling {
    /// Android's fling with its platform defaults: at least 50 dp/s
    /// (`ViewConfiguration`'s minimum fling velocity), at most 8000.
    pub const ANDROID: Self = Self::Android {
        min_speed: 50.0,
        max_speed: 8000.0,
        max_distance: f64::INFINITY,
    };

    /// The decay a release at `speed` (not negative) continues with, or
    /// `None` when it stays.
    pub(crate) fn decay(&self, speed: f64) -> Option<MomentumDecay> {
        match *self {
            Self::Decay(decay) => (speed >= decay.min_speed).then_some(decay),
            Self::Android {
                min_speed,
                max_speed,
                max_distance,
            } => {
                if speed.is_nan() || speed < min_speed {
                    return None;
                }
                let (distance, duration) = android_fling(speed.min(max_speed))?;
                Some(MomentumDecay {
                    model: DecayModel::Ease {
                        distance,
                        duration,
                        easing: Easing::Bezier([0.175, 0.5, 0.35, 1.0]),
                    },
                    min_speed,
                    max_speed: f64::INFINITY,
                    max_distance,
                })
            }
        }
    }
}

/// The distance in dp and duration in milliseconds of Android's
/// `OverScroller` spline fling at `speed` dp/s (`SplineOverScroller`, at
/// 160 dpi so its pixels are dp).
fn android_fling(speed: f64) -> Option<(f64, f64)> {
    const INFLEXION: f64 = 0.35;
    /// `ViewConfiguration.getScrollFriction()`.
    const FRICTION: f64 = 0.015;
    /// Earth's gravity in inches per second squared at 160 dpi, times
    /// Android's 0.84.
    const PHYSICAL: f64 = 9.80665 * 39.37 * 160.0 * 0.84;
    let rate = 0.78f64.ln() / 0.9f64.ln();
    if speed.is_nan() || speed <= 0.0 || speed.is_infinite() {
        return None;
    }
    let deceleration = (INFLEXION * speed / (FRICTION * PHYSICAL)).ln();
    let distance = FRICTION * PHYSICAL * (rate / (rate - 1.0) * deceleration).exp();
    let duration = 1000.0 * (deceleration / (rate - 1.0)).exp();
    (distance > 0.0 && distance.is_finite() && duration > 0.0 && duration.is_finite())
        .then_some((distance, duration))
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
