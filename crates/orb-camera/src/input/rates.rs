//! GL JS's gesture rates and thresholds, from its handlers' defaults.

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
/// Milliseconds a touch holds still for a long press (Compose's and
/// Android's `ViewConfiguration` default).
pub(crate) const LONG_PRESS: f64 = 400.0;
