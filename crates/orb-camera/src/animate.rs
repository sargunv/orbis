//! Camera animations: `easeTo` and `flyTo` after GL JS, and the moves
//! frames step on by deltas, a gesture's momentum and a discrete delta's
//! eased step (`crate::momentum`).

use glam::DVec2;
use orb_crs::{Geodetic, Metres, sphere};

use crate::momentum::Momentum;
use crate::moves::wrap_bearing;
use crate::{CameraState, Viewport};
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
    /// MapLibre Native's default transition curve, which momentum snaps
    /// settle along.
    pub(crate) const SNAP: Self = Self::Bezier([0.0, 0.0, 0.25, 1.0]);

    /// Progress at time fraction `t` in [0, 1].
    pub fn at(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match *self {
            Self::Linear => t,
            Self::Bezier([x1, y1, x2, y2]) => bezier(x1, y1, x2, y2, t),
        }
    }

    /// The rate of progress at time fraction `t`: the curve's slope.
    pub fn slope(&self, t: f64) -> f64 {
        const H: f64 = 1e-5;
        let (a, b) = ((t - H).max(0.0), (t + H).min(1.0));
        if b <= a {
            return 0.0;
        }
        (self.at(b) - self.at(a)) / (b - a)
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
/// their current values, for a jump those of the camera asked for (see
/// [`Camera::jump_to`](crate::Camera::jump_to)).
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
    /// Without a `center`: the point zoom, rotation and tilt keep in place.
    pub around: Option<Geodetic>,
    /// Where the center ends up, in pixels from the viewport's middle.
    pub offset: DVec2,
}

/// GL JS's `easeTo` timing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EaseOptions {
    /// Milliseconds; zero ends at the first frame.
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

/// The longest a flight runs, in milliseconds, and the longest duration
/// mln-ffi's C API takes: MapLibre Native's clock's span (nanoseconds in
/// 64 bits, 292 years).
pub const MAX_DURATION_MS: f64 = i64::MAX as f64 / 1e6;

/// GL JS's `flyTo` options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlyOptions {
    /// How far the flight zooms out: ρ in van Wijk and Nuij's curve.
    pub curve: f64,
    /// Average speed along the curve, in ρ-screenfuls per second. One that
    /// gives no finite duration within [`MAX_DURATION_MS`] jumps.
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

/// The ground point a move turns about, the pixel it stays under and, on
/// terrain, the ground's own height there (else it stands level with the
/// target).
pub(crate) type Around = (Geodetic, DVec2, Option<Metres>);

/// A running animation.
#[derive(Clone, Debug)]
pub(crate) enum Motion {
    /// A path to a camera, which each frame samples.
    Path(Path),
    /// A gesture's momentum, which each frame steps on within the
    /// camera's limits.
    Momentum(Momentum),
    /// A discrete delta's eased step (a wheel notch), stepped as momentum
    /// is.
    Delta(Momentum),
}

impl Motion {
    pub(crate) fn duration(&self) -> f64 {
        match self {
            Self::Path(path) => path.duration(),
            Self::Momentum(momentum) | Self::Delta(momentum) => match momentum.active() {
                true => f64::INFINITY,
                false => 0.0,
            },
        }
    }
}

/// An animation's path from one camera to another.
#[derive(Clone, Debug)]
pub(crate) enum Path {
    Ease {
        from: CameraState,
        to: CameraState,
        /// The ground point zoom, rotation and tilt keep in place, if any.
        around: Option<Around>,
        /// The end target's world position, on the near side of the
        /// antimeridian.
        to_world: DVec2,
        options: EaseOptions,
        /// On the sphere: the target moves over it and the sphere's size
        /// eases (GL JS's globe `handleEaseTo`).
        sphere: bool,
    },
    Fly {
        from: CameraState,
        to: CameraState,
        to_world: DVec2,
        curve: FlyCurve,
        duration: f64,
        easing: Easing,
        sphere: bool,
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

/// A target `t` of the way from `from` to `to` over the sphere, the short
/// way round (GL JS's `interpolateLngLatForGlobe`): latitude moves
/// steadily, and longitude by the integral of the secant of latitude, so
/// the globe turns at a steady speed on screen.
fn over_sphere(from: Geodetic, to: Geodetic, t: f64) -> Geodetic {
    let d_lon = nearest_bearing(from.lon, to.lon) - from.lon;
    let lat = lerp(from.lat, to.lat, t).clamp(-90.0, 90.0);
    let edge = orb_crs::MERCATOR_MAX_LATITUDE;
    let (start, end) = (from.lat.clamp(-edge, edge), to.lat.clamp(-edge, edge));
    let k = if (end - start).abs() > 1.0 {
        // ∫ sec x dx, sampled across the equator with the start mirrored.
        let integral = |x: f64| {
            let (sin, cos) = (x / 2.0).sin_cos();
            (sin + cos).ln() - (cos - sin).ln()
        };
        let a = if start.signum() == end.signum() {
            start.abs()
        } else {
            -start.abs()
        }
        .to_radians();
        let b = end.abs().to_radians();
        (integral(a + t * (b - a)) - integral(a)) / (integral(b) - integral(a))
    } else {
        t
    };
    Geodetic::new(from.lon + d_lon * k, lat).wrapped()
}

impl Path {
    pub(crate) fn ease(
        from: CameraState,
        to: CameraState,
        around: Option<Around>,
        options: EaseOptions,
        surface: &Surface,
    ) -> Self {
        let projection = &surface.projection;
        let to_world =
            projection.nearest_copy(to.target_world(projection), from.target_world(projection));
        Self::Ease {
            from,
            to,
            around,
            to_world,
            options,
            sphere: from.on_sphere(surface),
        }
    }

    pub(crate) fn fly(
        from: CameraState,
        to: CameraState,
        options: &FlyOptions,
        min_zoom: f64,
        surface: &Surface,
        viewport: Viewport,
    ) -> Self {
        let projection = &surface.projection;
        let sphere = from.on_sphere(surface);
        let from_world = from.target_world(projection);
        let to_world = projection.nearest_copy(to.target_world(projection), from_world);
        // On the sphere the flight is over it, at its size: the distance
        // is the arc's on screen at the start, and zooms are the sphere's
        // size (GL JS's globe `handleFlyTo`).
        let zoom = |camera: &CameraState| match sphere {
            true => camera.globe_zoom(),
            false => camera.zoom(projection),
        };
        let w0 = viewport.width.max(viewport.height);
        let w1 = w0 / (zoom(&to) - zoom(&from)).exp2();
        let u1 = match sphere {
            true => {
                let (a, b) = (sphere::up(from.target), sphere::up(to.target));
                a.angle_between(b) * sphere::RADIUS / from.resolution
            }
            false => (to_world - from_world).length() / from.world_per_px(projection),
        };
        let floor = options.min_zoom.unwrap_or(min_zoom).max(min_zoom);
        // A least style zoom, as the sphere's size at the destination.
        let floor = match sphere {
            true => floor - orb_crs::style_parallel(to.target.lat).log2(),
            false => floor,
        };
        let lowest = floor.min(zoom(&from)).min(zoom(&to));
        let w_max = w0 / (lowest - zoom(&from)).exp2();
        let curve = FlyCurve::new(w0, w1, u1, options.curve, w_max);
        let duration = options.duration.unwrap_or_else(|| {
            let speed = options
                .screen_speed
                .map_or(options.speed, |screens| screens / curve.rho);
            1000.0 * curve.s / speed
        });
        // A flight longer than its maximum jumps instead, as GL JS's does,
        // and so does one that would never end: no speed, a NaN, or longer
        // than Native's clock holds.
        let limit = options
            .max_duration
            .map_or(MAX_DURATION_MS, |max| max.min(MAX_DURATION_MS));
        let duration = match (0.0..=limit).contains(&duration) {
            true => duration,
            false => 0.0,
        };
        Self::Fly {
            from,
            to,
            to_world,
            curve,
            duration,
            easing: options.easing,
            sphere,
        }
    }

    pub(crate) fn duration(&self) -> f64 {
        match self {
            Self::Ease { options, .. } => options.duration,
            Self::Fly { duration, .. } => *duration,
        }
    }

    /// Where the path ends, in world coordinates.
    pub(crate) fn destination(&self) -> DVec2 {
        match self {
            Self::Ease { to_world, .. } | Self::Fly { to_world, .. } => *to_world,
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
        };
        if ground == to.elevation || !(0.0..1.0).contains(&k) {
            return;
        }
        let now = lerp(from.elevation, to.elevation, k);
        from.elevation = (now - ground * k) / (1.0 - k);
        to.elevation = ground;
    }

    /// Moves `camera` to the path's state `elapsed` milliseconds in.
    pub(crate) fn apply(
        &self,
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
        if t >= 1.0 {
            let (Self::Ease { to, .. } | Self::Fly { to, .. }) = self;
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
                sphere,
            } => {
                let k = options.easing.at(t);
                let mut next = *from;
                if *sphere {
                    next.set_globe_zoom(lerp(from.globe_zoom(), to.globe_zoom(), k));
                } else {
                    next.set_zoom(
                        projection,
                        lerp(from.zoom(projection), to.zoom(projection), k),
                    );
                }
                lerp_orientation(&mut next, from, to, k);
                match around {
                    Some((ground, at, height)) if *sphere => {
                        let height = height.unwrap_or(Metres(next.elevation));
                        next.put_under(surface, viewport, *ground, *at, height);
                    }
                    None if *sphere => next.target = over_sphere(from.target, to.target, k),
                    Some((ground, at, height)) => {
                        let height = height.unwrap_or(Metres(next.elevation));
                        next.slide_under(surface, viewport, *ground, height, *at);
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
                sphere,
                ..
            } => {
                let k = easing.at(t);
                let (w, u) = curve.at(k);
                let mut next = *from;
                if *sphere {
                    next.set_globe_zoom(from.globe_zoom() + (1.0 / w).log2());
                } else {
                    next.set_zoom(projection, from.zoom(projection) + (1.0 / w).log2());
                }
                lerp_orientation(&mut next, from, to, k);
                if *sphere {
                    next.target = over_sphere(from.target, to.target, u);
                } else {
                    let start = from.target_world(projection);
                    let world_per_px = next.world_per_px(projection);
                    next.set_target_world(
                        projection,
                        start + (*to_world - start) * u,
                        world_per_px,
                    );
                }
                *camera = next;
            }
        }
    }
}

/// Sets `camera`'s orientation, elevation, field of view and padding `k`
/// of the way from `from`'s to `to`'s, turning the short way round.
fn lerp_orientation(camera: &mut CameraState, from: &CameraState, to: &CameraState, k: f64) {
    camera.bearing = wrap_bearing(lerp(
        from.bearing,
        nearest_bearing(from.bearing, to.bearing),
        k,
    ));
    camera.pitch = lerp(from.pitch, to.pitch, k);
    camera.roll = wrap_bearing(lerp(from.roll, nearest_bearing(from.roll, to.roll), k));
    camera.elevation = lerp(from.elevation, to.elevation, k);
    camera.fov = lerp(from.fov, to.fov, k);
    camera.padding = from.padding.lerp(to.padding, k);
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
            Easing::SNAP,
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
