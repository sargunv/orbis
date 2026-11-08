//! GL JS's gesture rates, thresholds and inertia, from its handlers'
//! defaults.

/// Pixels a press may move and still be a click; pans start past it.
pub(crate) const CLICK_TOLERANCE: f64 = 3.0;
/// How far apart, in pixels, and how quickly, in milliseconds, two taps
/// make a double tap.
pub(crate) const TAP_TOLERANCE: f64 = 30.0;
pub(crate) const TAP_INTERVAL: f64 = 500.0;
/// Degrees per pixel of mouse drag-rotation, sideways and upward.
pub(crate) const DRAG_ROTATE: f64 = 0.8;
pub(crate) const DRAG_PITCH: f64 = 0.5;
/// Zoom levels a pinch spreads before it zooms.
pub(crate) const PINCH_ZOOM_THRESHOLD: f64 = 0.1;
/// Pixels along the fingers' circle a twist travels before it rotates.
pub(crate) const ROTATE_THRESHOLD: f64 = 25.0;
/// Wheel zoom: the rate for mouse wheels' large steps, and for trackpads
/// and small steps.
pub(crate) const WHEEL_ZOOM_RATE: f64 = 1.0 / 450.0;
pub(crate) const TRACKPAD_ZOOM_RATE: f64 = 1.0 / 100.0;
/// Keyboard steps: pixels, degrees of bearing, degrees of pitch.
pub(crate) const KEY_PAN: f64 = 100.0;
pub(crate) const KEY_BEARING: f64 = 15.0;
pub(crate) const KEY_PITCH: f64 = 10.0;

pub(crate) const PAN_INERTIA: Inertia = Inertia::new(2500.0, 1400.0);
pub(crate) const ZOOM_INERTIA: Inertia = Inertia::new(20.0, 1400.0);
pub(crate) const BEARING_INERTIA: Inertia = Inertia::new(1000.0, 360.0);
pub(crate) const PITCH_INERTIA: Inertia = Inertia::new(1000.0, 90.0);

/// How a property keeps moving after a gesture lets go: a fraction of the
/// release speed, decelerating at a constant rate along GL JS's inertia
/// ease-out ([`Easing::INERTIA`](orb_camera::Easing::INERTIA)).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Inertia {
    /// Units per second squared.
    deceleration: f64,
    /// Units per second.
    max_speed: f64,
}

/// The fraction of the release speed a glide starts with.
const LINEARITY: f64 = 0.3;

impl Inertia {
    const fn new(deceleration: f64, max_speed: f64) -> Self {
        Self {
            deceleration,
            max_speed,
        }
    }

    /// The glide for `amount` units moved over the last `duration`
    /// milliseconds: how far it goes, and for how many milliseconds.
    pub(crate) fn ease(&self, amount: f64, duration: f64) -> (f64, f64) {
        let seconds = (duration / 1000.0).max(1e-3);
        let speed = (amount * LINEARITY / seconds).clamp(-self.max_speed, self.max_speed);
        let time = speed.abs() / (self.deceleration * LINEARITY);
        (speed * time / 2.0, time * 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl_js_inertia_matches_its_formula() {
        // 100 px over 100 ms: 1000 px/s × 0.3 = 300 px/s, decelerating at
        // 2500 × 0.3 for 0.4 s, travelling 60 px.
        let (amount, duration) = PAN_INERTIA.ease(100.0, 100.0);
        assert!((amount - 60.0).abs() < 1e-9);
        assert!((duration - 400.0).abs() < 1e-9);
        let (amount, _) = PAN_INERTIA.ease(-100.0, 100.0);
        assert!(amount < 0.0);
    }
}
