//! Feel profiles: the rates, thresholds and inertia that make gestures feel
//! like a platform's.

use orb_camera::Easing;

/// How a property keeps moving after a gesture lets go.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Inertia {
    /// GL JS's model: a fraction of the release speed, decelerating at a
    /// constant rate along an ease-out curve.
    Linear {
        linearity: f64,
        /// Units per second squared.
        deceleration: f64,
        /// Units per second.
        max_speed: f64,
    },
    /// Platform scroll views' model: velocity decays by `rate` per
    /// millisecond until it falls below `stop` units per second.
    Decay {
        rate: f64,
        max_speed: f64,
        stop: f64,
    },
}

impl Inertia {
    /// The glide for `amount` units moved over the last `duration`
    /// milliseconds: how far it goes, for how long, and how it eases.
    pub fn ease(&self, amount: f64, duration: f64) -> (f64, f64, Easing) {
        let seconds = (duration / 1000.0).max(1e-3);
        match *self {
            Self::Linear {
                linearity,
                deceleration,
                max_speed,
            } => {
                let speed = (amount * linearity / seconds).clamp(-max_speed, max_speed);
                let time = speed.abs() / (deceleration * linearity);
                (speed * time / 2.0, time * 1000.0, Easing::INERTIA)
            }
            Self::Decay {
                rate,
                max_speed,
                stop,
            } => {
                let speed = (amount / seconds).clamp(-max_speed, max_speed);
                if speed.abs() <= stop {
                    return (0.0, 0.0, Easing::Linear);
                }
                let ms = (stop / speed.abs()).ln() / rate.ln();
                let moved = speed / 1000.0 * (rate.powf(ms) - 1.0) / rate.ln();
                (moved, ms, Easing::Decay { rate, duration: ms })
            }
        }
    }
}

/// Gesture rates, thresholds and inertia.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feel {
    /// Pixels a press may move and still be a click; pans start past it.
    pub click_tolerance: f64,
    /// How far apart and how quickly two taps make a double tap.
    pub tap_tolerance: f64,
    pub tap_interval: f64,
    /// Degrees per pixel of mouse drag-rotation, sideways and upward.
    pub drag_rotate: f64,
    pub drag_pitch: f64,
    /// Zoom levels a pinch spreads before it zooms.
    pub pinch_zoom_threshold: f64,
    /// Pixels along the fingers' circle a twist travels before it rotates.
    pub rotate_threshold: f64,
    /// Wheel zoom: the rate for mouse wheels' large steps, and for
    /// trackpads and small steps.
    pub wheel_zoom_rate: f64,
    pub trackpad_zoom_rate: f64,
    /// Keyboard steps: pixels, degrees of bearing, degrees of pitch.
    pub key_pan: f64,
    pub key_bearing: f64,
    pub key_pitch: f64,
    pub pan_inertia: Inertia,
    pub zoom_inertia: Inertia,
    pub bearing_inertia: Inertia,
    pub pitch_inertia: Inertia,
}

impl Default for Feel {
    fn default() -> Self {
        Self::DESKTOP
    }
}

const fn linear(deceleration: f64, max_speed: f64) -> Inertia {
    Inertia::Linear {
        linearity: 0.3,
        deceleration,
        max_speed,
    }
}

impl Feel {
    /// GL JS's handlers and their defaults.
    pub const DESKTOP: Self = Self {
        click_tolerance: 3.0,
        tap_tolerance: 30.0,
        tap_interval: 500.0,
        drag_rotate: 0.8,
        drag_pitch: 0.5,
        pinch_zoom_threshold: 0.1,
        rotate_threshold: 25.0,
        wheel_zoom_rate: 1.0 / 450.0,
        trackpad_zoom_rate: 1.0 / 100.0,
        key_pan: 100.0,
        key_bearing: 15.0,
        key_pitch: 10.0,
        pan_inertia: linear(2500.0, 1400.0),
        zoom_inertia: linear(20.0, 1400.0),
        bearing_inertia: linear(1000.0, 360.0),
        pitch_inertia: linear(1000.0, 90.0),
    };

    /// Pans glide like a `UIScrollView` at its normal deceleration rate
    /// (0.998 per millisecond); the rest as on the desktop.
    pub const IOS: Self = Self {
        pan_inertia: Inertia::Decay {
            rate: 0.998,
            max_speed: 8000.0,
            stop: 20.0,
        },
        ..Self::DESKTOP
    };

    /// Pans glide about as far as Android's `OverScroller` flings: a decay
    /// matched to its spline at 2,000 px/s on a 2.75× display [inferred].
    pub const ANDROID: Self = Self {
        pan_inertia: Inertia::Decay {
            rate: 0.9935,
            max_speed: 8000.0,
            stop: 20.0,
        },
        ..Self::DESKTOP
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl_js_inertia_matches_its_formula() {
        // 100 px over 100 ms: 1000 px/s × 0.3 = 300 px/s, decelerating at
        // 2500 × 0.3 for 0.4 s, travelling 60 px.
        let (amount, duration, _) = Feel::DESKTOP.pan_inertia.ease(100.0, 100.0);
        assert!((amount - 60.0).abs() < 1e-9);
        assert!((duration - 400.0).abs() < 1e-9);
        let (amount, ..) = Feel::DESKTOP.pan_inertia.ease(-100.0, 100.0);
        assert!(amount < 0.0);
    }

    #[test]
    fn decay_inertia_stops_at_its_threshold() {
        let (amount, duration, easing) = Feel::IOS.pan_inertia.ease(200.0, 100.0);
        // 2000 px/s decaying 0.998 per ms: about 2 / 0.002 = 1000 px.
        assert!(amount > 900.0 && amount < 1000.0, "{amount}");
        assert!(duration > 2000.0, "{duration}");
        assert!((easing.at(1.0) - 1.0).abs() < 1e-9);
        assert_eq!(Feel::IOS.pan_inertia.ease(1.0, 100.0).0, 0.0);
    }
}
