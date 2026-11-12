//! The gestures' rates and thresholds: mln-rs's `mln_gestures` defaults,
//! which are maplibre-native-ffi's browser demo's where it has the gesture
//! and GL JS's handlers' elsewhere.

/// Pixels a press may move and still be a click; pans start past it.
pub(crate) const CLICK_TOLERANCE: f64 = 3.0;
/// How far apart, in pixels, and how quickly, in milliseconds, two taps
/// make a double tap.
pub(crate) const TAP_TOLERANCE: f64 = 30.0;
pub(crate) const TAP_INTERVAL: f64 = 500.0;
/// Degrees per pixel of mouse drag-rotation, sideways (ffi's demo's; GL JS
/// and Compose turn 0.8) and upward.
pub(crate) const DRAG_ROTATE: f64 = 0.5;
pub(crate) const DRAG_PITCH: f64 = 0.5;
/// Zoom levels a pinch spreads before it zooms.
pub(crate) const PINCH_ZOOM_THRESHOLD: f64 = 0.1;
/// Pixels along the fingers' circle a twist travels before it rotates.
pub(crate) const ROTATE_THRESHOLD: f64 = 25.0;
/// Zoom levels per wheel notch, eased over the milliseconds after: ffi's
/// demo's and Compose's 100 px at 1/300 a pixel, over GL JS's 200 ms.
/// Provisional: no one has measured the native map apps' notches yet.
pub(crate) const NOTCH_ZOOM: f64 = 1.0 / 3.0;
pub(crate) const NOTCH_DURATION: f64 = 200.0;
/// Zoom levels per pixel of continuous scroll (ffi's demo's and Compose's,
/// provisional as the notch), and with Ctrl, as a browser's trackpad pinch
/// sends (ffi's demo's).
pub(crate) const SCROLL_ZOOM_RATE: f64 = 1.0 / 300.0;
pub(crate) const PINCH_ZOOM_RATE: f64 = 1.0 / 100.0;
/// The most one scroll event zooms, in levels (GL JS's `maxScalePerFrame`).
pub(crate) const MAX_SCROLL_ZOOM: f64 = 1.0;
/// A continuous scroll stays one move until this long after its last
/// event (GL JS's and Compose's).
pub(crate) const SCROLL_IDLE: f64 = 200.0;
/// Keyboard steps: pixels, degrees of bearing, degrees of pitch.
pub(crate) const KEY_PAN: f64 = 100.0;
pub(crate) const KEY_BEARING: f64 = 15.0;
pub(crate) const KEY_PITCH: f64 = 10.0;
/// Milliseconds a touch holds still for a long press (Compose's and
/// Android's `ViewConfiguration` default).
pub(crate) const LONG_PRESS: f64 = 400.0;
