//! Camera animations: `easeTo` and `flyTo` after GL JS, and glides (inertia
//! and smooth wheel zoom) whose properties ease independently.

use glam::DVec2;
use orb_crs::Geodetic;
use orb_crs::Projection;

use crate::moves::wrap_bearing;
use crate::{CameraState, View, Viewport};
use orb_crs::Surface;

/// How an animation's progress follows time, from 0 to 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Easing {
    Linear,
    /// A CSS-style cubic Bézier through (0, 0), `(x1, y1)`, `(x2, y2)`,
    /// (1, 1).
    Bezier([f64; 4]),
}

impl Easing {
    /// GL JS's `defaultEasing`.
    pub const DEFAULT: Self = Self::Bezier([0.25, 0.1, 0.25, 1.0]);
    /// GL JS's `easeOut`, for keyboard and double-click moves.
    pub(crate) const EASE_OUT: Self = Self::Bezier([0.0, 0.0, 0.58, 1.0]);
    /// GL JS's inertia easing.
    pub(crate) const INERTIA: Self = Self::Bezier([0.0, 0.0, 0.3, 1.0]);

    /// Progress at time fraction `t` in [0, 1].
    pub fn at(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match *self {
            Self::Linear => t,
            Self::Bezier([x1, y1, x2, y2]) => bezier(x1, y1, x2, y2, t),
        }
    }
}

/// Solves a unit cubic Bézier for `x = t` and returns its `y`, after the
/// UnitBezier GL JS and browsers use.
fn bezier(x1: f64, y1: f64, x2: f64, y2: f64, t: f64) -> f64 {
    let (cx, cy) = (3.0 * x1, 3.0 * y1);
    let (bx, by) = (3.0 * (x2 - x1) - cx, 3.0 * (y2 - y1) - cy);
    let (ax, ay) = (1.0 - cx - bx, 1.0 - cy - by);
    let x = |u: f64| ((ax * u + bx) * u + cx) * u;
    let dx = |u: f64| (3.0 * ax * u + 2.0 * bx) * u + cx;
    let mut u = t;
    for _ in 0..8 {
        let error = x(u) - t;
        if error.abs() < 1e-7 {
            return ((ay * u + by) * u + cy) * u;
        }
        let slope = dx(u);
        if slope.abs() < 1e-7 {
            break;
        }
        u -= error / slope;
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    u = t;
    while hi - lo > 1e-7 {
        if x(u) < t {
            lo = u;
        } else {
            hi = u;
        }
        u = (lo + hi) / 2.0;
    }
    ((ay * u + by) * u + cy) * u
}

/// Where a camera move ends: GL JS's `CameraOptions`. Unset properties keep
/// their current values.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CameraTarget {
    pub center: Option<Geodetic>,
    pub zoom: Option<f64>,
    pub bearing: Option<f64>,
    pub pitch: Option<f64>,
    /// Degrees about the view axis (GL JS `roll`).
    pub roll: Option<f64>,
    /// The vertical field of view, degrees.
    pub fov: Option<f64>,
    /// The camera's padding from here on (GL JS `padding`).
    pub padding: Option<crate::Padding>,
    /// Metres the target stands above the surface (GL JS `elevation`).
    /// While the target is clamped to the ground, the terrain takes it.
    pub elevation: Option<f64>,
    /// Without a `center`: the point zoom and rotation keep in place.
    pub around: Option<Geodetic>,
    /// Where the center ends up, in pixels from the viewport's middle.
    pub offset: DVec2,
}

/// GL JS's `easeTo` timing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EaseOptions {
    /// Milliseconds; zero jumps.
    pub duration: f64,
    pub easing: Easing,
}

impl Default for EaseOptions {
    fn default() -> Self {
        Self {
            duration: 500.0,
            easing: Easing::DEFAULT,
        }
    }
}

/// GL JS's `flyTo` options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlyOptions {
    /// How far the flight zooms out: ρ in van Wijk and Nuij's curve.
    pub curve: f64,
    /// Average speed along the curve, in ρ-screenfuls per second.
    pub speed: f64,
    /// Average speed in screenfuls per second, overriding `speed`.
    pub screen_speed: Option<f64>,
    /// Milliseconds, overriding the speeds.
    pub duration: Option<f64>,
    /// A flight longer than this jumps instead.
    pub max_duration: Option<f64>,
    /// The flight zooms out no further than this.
    pub min_zoom: Option<f64>,
    pub easing: Easing,
}

impl Default for FlyOptions {
    fn default() -> Self {
        Self {
            curve: 1.42,
            speed: 1.2,
            screen_speed: None,
            duration: None,
            max_duration: None,
            min_zoom: None,
            easing: Easing::DEFAULT,
        }
    }
}

/// One property's share of a glide: a total change spread over time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Track<T> {
    pub amount: T,
    /// Milliseconds.
    pub duration: f64,
    pub easing: Easing,
}

/// A move continued over time by deltas: inertia after a gesture, or a
/// smooth wheel zoom. Each property eases on its own schedule, so a short
/// zoom fling needn't end with a long pan's.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Glide {
    /// Pixels the ground moves on screen.
    pub pan: Option<Track<DVec2>>,
    /// Zoom levels.
    pub zoom: Option<Track<f64>>,
    /// Degrees clockwise.
    pub bearing: Option<Track<f64>>,
    pub pitch: Option<Track<f64>>,
    /// The pixel zoom and rotation keep in place; the viewport's middle if
    /// unset.
    pub around: Option<DVec2>,
}

impl Glide {
    fn duration(&self) -> f64 {
        let d = |t: Option<f64>| t.unwrap_or(0.0);
        d(self.pan.map(|t| t.duration))
            .max(d(self.zoom.map(|t| t.duration)))
            .max(d(self.bearing.map(|t| t.duration)))
            .max(d(self.pitch.map(|t| t.duration)))
    }
}

fn progress<T>(track: &Option<Track<T>>, elapsed: f64) -> f64 {
    match track {
        Some(t) if t.duration > 0.0 => t.easing.at(elapsed / t.duration),
        _ => 1.0,
    }
}

/// A running animation's path.
#[derive(Clone, Debug)]
pub(crate) enum Motion {
    Ease {
        from: CameraState,
        to: CameraState,
        /// The ground point and the pixel it stays under, when zooming or
        /// rotating around a point.
        around: Option<(Geodetic, DVec2)>,
        /// The end target's world position, on the near side of the
        /// antimeridian.
        to_world: DVec2,
        options: EaseOptions,
    },
    Fly {
        from: CameraState,
        to: CameraState,
        to_world: DVec2,
        curve: FlyCurve,
        duration: f64,
        easing: Easing,
    },
    Glide {
        glide: Glide,
        /// Each track's progress so far.
        done: [f64; 4],
    },
}

/// van Wijk and Nuij's "Smooth and efficient zooming and panning", as GL JS
/// parameterizes it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlyCurve {
    rho: f64,
    r0: f64,
    w0: f64,
    u1: f64,
    /// Total length in ρ-screenfuls.
    s: f64,
    /// A flight with no path: `w(s) = exp(k ρ s)`.
    zoom_only: Option<f64>,
}

impl FlyCurve {
    fn new(w0: f64, w1: f64, u1: f64, rho: f64, w_max: f64) -> Self {
        let rho = if u1 > 0.0 {
            rho.min((w_max / u1 * 2.0).sqrt())
        } else {
            rho
        };
        let rho2 = rho * rho;
        let r = |descent: bool| {
            let sign = if descent { -1.0 } else { 1.0 };
            let b = (w1 * w1 - w0 * w0 + sign * rho2 * rho2 * u1 * u1)
                / (2.0 * if descent { w1 } else { w0 } * rho2 * u1);
            ((b * b + 1.0).sqrt() - b).ln()
        };
        let r0 = r(false);
        let s = (r(true) - r0) / rho;
        if u1.abs() < 2e-6 || !s.is_finite() {
            let k = if w1 < w0 { -1.0 } else { 1.0 };
            return Self {
                rho,
                r0: 0.0,
                w0,
                u1,
                s: (w1 / w0).ln().abs() / rho,
                zoom_only: Some(k),
            };
        }
        Self {
            rho,
            r0,
            w0,
            u1,
            s,
            zoom_only: None,
        }
    }

    /// The visible span relative to the start, and the fraction of the path
    /// covered, `k` of the way along.
    fn at(&self, k: f64) -> (f64, f64) {
        let s = k * self.s;
        if let Some(sign) = self.zoom_only {
            return ((sign * self.rho * s).exp(), 0.0);
        }
        let (rho, r0) = (self.rho, self.r0);
        let w = r0.cosh() / (r0 + rho * s).cosh();
        let u = self.w0 * ((r0.cosh() * (r0 + rho * s).tanh() - r0.sinh()) / (rho * rho)) / self.u1;
        (w, u)
    }
}

/// The end bearing nearest the start, so rotations take the short way.
fn nearest_bearing(from: f64, to: f64) -> f64 {
    from + wrap_bearing(to - from)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

impl Motion {
    pub(crate) fn ease(
        from: CameraState,
        to: CameraState,
        around: Option<(Geodetic, DVec2)>,
        options: EaseOptions,
        projection: &Projection,
    ) -> Self {
        let to_world =
            projection.nearest_copy(to.target_world(projection), from.target_world(projection));
        Self::Ease {
            from,
            to,
            around,
            to_world,
            options,
        }
    }

    pub(crate) fn fly(
        from: CameraState,
        to: CameraState,
        options: &FlyOptions,
        min_zoom: f64,
        projection: &Projection,
        viewport: Viewport,
    ) -> Self {
        let from_world = from.target_world(projection);
        let to_world = projection.nearest_copy(to.target_world(projection), from_world);
        let start_px = from.world_per_px(projection);
        let w0 = viewport.width.max(viewport.height);
        let w1 = w0 / (to.zoom(projection) - from.zoom(projection)).exp2();
        let u1 = (to_world - from_world).length() / start_px;
        let lowest = options
            .min_zoom
            .unwrap_or(min_zoom)
            .max(min_zoom)
            .min(from.zoom(projection))
            .min(to.zoom(projection));
        let w_max = w0 / (lowest - from.zoom(projection)).exp2();
        let curve = FlyCurve::new(w0, w1, u1, options.curve, w_max);
        let duration = options.duration.unwrap_or_else(|| {
            let speed = options
                .screen_speed
                .map_or(options.speed, |screens| screens / curve.rho);
            1000.0 * curve.s / speed
        });
        let duration = match options.max_duration {
            Some(max) if duration > max => 0.0,
            _ => duration,
        };
        Self::Fly {
            from,
            to,
            to_world,
            curve,
            duration,
            easing: options.easing,
        }
    }

    pub(crate) fn glide(glide: Glide) -> Self {
        Self::Glide {
            glide,
            done: [0.0; 4],
        }
    }

    pub(crate) fn duration(&self) -> f64 {
        match self {
            Self::Ease { options, .. } => options.duration,
            Self::Fly { duration, .. } => *duration,
            Self::Glide { glide, .. } => glide.duration(),
        }
    }

    /// Where an ease or a flight ends, in world coordinates.
    pub(crate) fn destination(&self) -> Option<DVec2> {
        match self {
            Self::Ease { to_world, .. } | Self::Fly { to_world, .. } => Some(*to_world),
            Self::Glide { .. } => None,
        }
    }

    /// Aims an ease's or a flight's elevation at `ground`, the terrain
    /// under its destination once that is known, keeping the elevation
    /// `elapsed` milliseconds in where it is: GL JS's `_updateElevation`,
    /// so DEM data landing mid-flight bends the path instead of moving the
    /// camera.
    pub(crate) fn aim_elevation(&mut self, ground: f64, elapsed: f64) {
        let t = elapsed / self.duration().max(f64::MIN_POSITIVE);
        let (from, to, k) = match self {
            Self::Ease {
                from, to, options, ..
            } => (from, to, options.easing.at(t)),
            Self::Fly {
                from, to, easing, ..
            } => (from, to, easing.at(t)),
            Self::Glide { .. } => return,
        };
        if ground == to.elevation || !(0.0..1.0).contains(&k) {
            return;
        }
        let now = lerp(from.elevation, to.elevation, k);
        from.elevation = (now - ground * k) / (1.0 - k);
        to.elevation = ground;
    }

    /// What a glide has still to move, for a glide that continues it.
    pub(crate) fn remaining_glide(&self) -> Option<Glide> {
        let Self::Glide { glide, done } = self else {
            return None;
        };
        let mut out = *glide;
        if let Some(t) = &mut out.pan {
            t.amount *= 1.0 - done[0];
        }
        if let Some(t) = &mut out.zoom {
            t.amount *= 1.0 - done[1];
        }
        if let Some(t) = &mut out.bearing {
            t.amount *= 1.0 - done[2];
        }
        if let Some(t) = &mut out.pitch {
            t.amount *= 1.0 - done[3];
        }
        Some(out)
    }

    /// Moves `camera` to the animation's state `elapsed` milliseconds in.
    pub(crate) fn apply(
        &mut self,
        camera: &mut CameraState,
        elapsed: f64,
        surface: &Surface,
        viewport: Viewport,
    ) {
        let projection = &surface.projection;
        let t = if self.duration() > 0.0 {
            elapsed / self.duration()
        } else {
            1.0
        };
        if t >= 1.0
            && let Self::Ease { to, .. } | Self::Fly { to, .. } = self
        {
            *camera = *to;
            return;
        }
        match self {
            Self::Ease {
                from,
                to,
                around,
                to_world,
                options,
            } => {
                let k = options.easing.at(t);
                let mut next = *from;
                next.set_zoom(
                    projection,
                    lerp(from.zoom(projection), to.zoom(projection), k),
                );
                next.bearing = wrap_bearing(lerp(
                    from.bearing,
                    nearest_bearing(from.bearing, to.bearing),
                    k,
                ));
                next.pitch = lerp(from.pitch, to.pitch, k);
                next.roll = wrap_bearing(lerp(from.roll, nearest_bearing(from.roll, to.roll), k));
                next.elevation = lerp(from.elevation, to.elevation, k);
                next.fov = lerp(from.fov, to.fov, k);
                next.padding = from.padding.lerp(to.padding, k);
                match around {
                    Some((ground, at)) => {
                        let world = projection.forward(*ground);
                        let view = View::new(&next, viewport, surface);
                        if let Some(now) = view.unproject(*at) {
                            let target =
                                view.camera_world() + projection.nearest_copy(world, now) - now;
                            next.set_target_world(projection, target, view.world_per_px());
                        }
                    }
                    None => {
                        let start = from.target_world(projection);
                        let world_per_px = next.world_per_px(projection);
                        next.set_target_world(
                            projection,
                            start + (*to_world - start) * k,
                            world_per_px,
                        );
                    }
                }
                *camera = next;
            }
            Self::Fly {
                from,
                to,
                to_world,
                curve,
                easing,
                ..
            } => {
                let k = easing.at(t);
                let (w, u) = curve.at(k);
                let mut next = *from;
                next.set_zoom(projection, from.zoom(projection) + (1.0 / w).log2());
                next.bearing = wrap_bearing(lerp(
                    from.bearing,
                    nearest_bearing(from.bearing, to.bearing),
                    k,
                ));
                next.pitch = lerp(from.pitch, to.pitch, k);
                next.roll = wrap_bearing(lerp(from.roll, nearest_bearing(from.roll, to.roll), k));
                next.elevation = lerp(from.elevation, to.elevation, k);
                next.fov = lerp(from.fov, to.fov, k);
                next.padding = from.padding.lerp(to.padding, k);
                let start = from.target_world(projection);
                let world_per_px = next.world_per_px(projection);
                next.set_target_world(projection, start + (*to_world - start) * u, world_per_px);
                *camera = next;
            }
            Self::Glide { glide, done } => {
                let center = camera.center(viewport);
                let around = glide.around.unwrap_or(center);
                let p = [
                    progress(&glide.pan, elapsed),
                    progress(&glide.zoom, elapsed),
                    progress(&glide.bearing, elapsed),
                    progress(&glide.pitch, elapsed),
                ];
                if let Some(pan) = glide.pan {
                    let delta = pan.amount * (p[0] - done[0]);
                    camera.pan(surface, viewport, center, center + delta);
                }
                if let Some(zoom) = glide.zoom {
                    let to = camera.zoom(projection) + zoom.amount * (p[1] - done[1]);
                    camera.zoom_around(surface, viewport, around, to);
                }
                if let Some(bearing) = glide.bearing {
                    let to = camera.bearing + bearing.amount * (p[2] - done[2]);
                    camera.rotate_around(surface, viewport, around, to);
                }
                if let Some(pitch) = glide.pitch {
                    camera.pitch += pitch.amount * (p[3] - done[3]);
                }
                *done = p;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easings_run_from_zero_to_one() {
        for easing in [
            Easing::Linear,
            Easing::DEFAULT,
            Easing::EASE_OUT,
            Easing::INERTIA,
        ] {
            assert!(easing.at(0.0).abs() < 1e-6, "{easing:?}");
            assert!((easing.at(1.0) - 1.0).abs() < 1e-6, "{easing:?}");
            let mut last = 0.0;
            for i in 1..=20 {
                let v = easing.at(f64::from(i) / 20.0);
                assert!(v >= last - 1e-9, "{easing:?} is monotonic");
                last = v;
            }
        }
        // CSS `ease` at its midpoint.
        assert!((Easing::DEFAULT.at(0.5) - 0.8024).abs() < 1e-3);
    }

    #[test]
    fn flights_zoom_out_between_distant_points() {
        let curve = FlyCurve::new(1024.0, 1024.0, 20_000.0, 1.42, f64::INFINITY);
        let (w_mid, u_mid) = curve.at(0.5);
        assert!(w_mid > 1.0, "the middle of the flight sees more: {w_mid}");
        assert!((u_mid - 0.5).abs() < 1e-6, "a symmetric flight is halfway");
        let (w_end, u_end) = curve.at(1.0);
        assert!((w_end - 1.0).abs() < 1e-9 && (u_end - 1.0).abs() < 1e-9);
        // Only zooming: no path, an exponential span.
        let zoom = FlyCurve::new(1024.0, 256.0, 0.0, 1.42, f64::INFINITY);
        assert_eq!(zoom.at(1.0).1, 0.0);
        assert!((zoom.at(1.0).0 - 0.25).abs() < 1e-9);
    }
}
