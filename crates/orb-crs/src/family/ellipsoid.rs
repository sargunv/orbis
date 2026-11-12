//! Ellipsoids, and latitude functions of the WGS84 ellipsoid shared by the
//! ellipsoidal families (EPSG Guidance Note 7-2 notation). Angles are
//! radians.

use std::f64::consts::FRAC_PI_2;

use crate::wgs84;

pub(crate) const A: f64 = wgs84::SEMI_MAJOR_M;
pub(crate) const E2: f64 = wgs84::E2;

/// An ellipsoid of revolution: semi-major axis in metres and first
/// eccentricity squared.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ellipsoid {
    pub(crate) a: f64,
    pub(crate) e2: f64,
}

impl Ellipsoid {
    /// WGS 84, which GRS 1980 matches here (see [`wgs84::FLATTENING`]).
    pub(crate) const WGS84: Self = Self { a: A, e2: E2 };
    /// Bessel 1841 (EPSG:7004), under the Dutch and Swiss grids.
    pub(crate) const BESSEL_1841: Self = Self::flattened(6_377_397.155, 299.152_812_8);
    /// Airy 1830 (EPSG:7001), under the British National Grid.
    pub(crate) const AIRY_1830: Self = Self::flattened(6_377_563.396, 299.324_964_6);

    /// The ellipsoid of semi-major axis `a` and inverse flattening `rf`.
    const fn flattened(a: f64, rf: f64) -> Self {
        let f = 1.0 / rf;
        Self {
            a,
            e2: f * (2.0 - f),
        }
    }

    /// The first eccentricity.
    pub(crate) fn e(&self) -> f64 {
        self.e2.sqrt()
    }
}

pub(crate) fn e() -> f64 {
    E2.sqrt()
}

/// `m = cos φ / √(1 − e² sin² φ)`.
pub(crate) fn m(phi: f64) -> f64 {
    phi.cos() / (1.0 - E2 * phi.sin().powi(2)).sqrt()
}

/// The conformal function `t = tan(π/4 − φ/2) / ((1 − e sin φ)/(1 + e sin φ))^(e/2)`.
pub(crate) fn t(phi: f64) -> f64 {
    let e = e();
    let s = phi.sin();
    (std::f64::consts::FRAC_PI_4 - phi / 2.0).tan() / ((1.0 - e * s) / (1.0 + e * s)).powf(e / 2.0)
}

/// The geodetic latitude whose conformal function is `t`: the conformal
/// latitude's tangent `(1 − t²) / 2t`, inverted by Newton's method.
pub(crate) fn phi_from_t(t: f64) -> f64 {
    if t <= 0.0 {
        return FRAC_PI_2;
    }
    if !t.is_finite() {
        return -FRAC_PI_2;
    }
    super::tm::tau_from_taup((1.0 - t * t) / (2.0 * t), E2).atan()
}

/// The authalic function `q`.
pub(crate) fn q(phi: f64) -> f64 {
    let e = e();
    let s = phi.sin();
    (1.0 - E2) * (s / (1.0 - E2 * s * s) - (1.0 / (2.0 * e)) * ((1.0 - e * s) / (1.0 + e * s)).ln())
}

/// The geodetic latitude whose authalic function is `q`.
pub(crate) fn phi_from_q(q: f64) -> f64 {
    let e = e();
    let qp = self::q(FRAC_PI_2);
    if q.abs() >= qp {
        return FRAC_PI_2.copysign(q);
    }
    let mut phi = (q / 2.0).clamp(-1.0, 1.0).asin();
    for _ in 0..20 {
        let s = phi.sin();
        let c = phi.cos();
        let one = 1.0 - E2 * s * s;
        let step = one * one / (2.0 * c)
            * (q / (1.0 - E2) - s / one + (1.0 / (2.0 * e)) * ((1.0 - e * s) / (1.0 + e * s)).ln());
        phi += step;
        if step.abs() < 1e-14 {
            break;
        }
    }
    phi
}

/// The authalic latitude of `phi`.
pub(crate) fn authalic(phi: f64) -> f64 {
    (q(phi) / q(FRAC_PI_2)).clamp(-1.0, 1.0).asin()
}

/// The geodetic latitude of authalic latitude `beta`.
pub(crate) fn from_authalic(beta: f64) -> f64 {
    phi_from_q(beta.sin() * q(FRAC_PI_2))
}

/// The radius of the sphere with the ellipsoid's area.
pub(crate) fn authalic_radius() -> f64 {
    A * (q(FRAC_PI_2) / 2.0).sqrt()
}
