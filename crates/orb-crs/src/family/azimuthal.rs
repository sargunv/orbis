//! Ellipsoidal polar stereographic (variants A and B) and Lambert azimuthal
//! equal-area, oblique and polar (EPSG Guidance Note 7-2, §3.5.2 and §3.6.2).

use std::f64::consts::FRAC_PI_2;

use glam::DVec2;

use super::ellipsoid::{A, authalic, authalic_radius, e, from_authalic, m, phi_from_t, q, t};
use super::tm::wrap_pi;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PolarStereographic {
    lon0: f64,
    x0: f64,
    y0: f64,
    /// +1 at the north pole, −1 at the south.
    pole: f64,
    /// ρ = `scale` · t.
    scale: f64,
    /// The latitude of true scale (variant B), or the pole (variant A).
    true_lat: f64,
}

impl PolarStereographic {
    /// Variant B: true scale at latitude `lat_ts`, whose sign picks the pole.
    pub(crate) fn with_standard_parallel(lat_ts: f64, lon0: f64, x0: f64, y0: f64) -> Self {
        let pole = lat_ts.signum();
        let phi = lat_ts.abs();
        let scale = if (phi - FRAC_PI_2).abs() < 1e-12 {
            Self::variant_a_scale(1.0)
        } else {
            A * m(phi) / t(phi)
        };
        Self {
            lon0,
            x0,
            y0,
            pole,
            scale,
            true_lat: lat_ts,
        }
    }

    /// Variant A: scale factor `k0` at the pole.
    pub(crate) fn with_scale(north: bool, k0: f64, lon0: f64, x0: f64, y0: f64) -> Self {
        Self {
            lon0,
            x0,
            y0,
            pole: if north { 1.0 } else { -1.0 },
            scale: Self::variant_a_scale(k0),
            true_lat: if north { FRAC_PI_2 } else { -FRAC_PI_2 },
        }
    }

    fn variant_a_scale(k0: f64) -> f64 {
        let e = e();
        2.0 * A * k0 / ((1.0 + e).powf(1.0 + e) * (1.0 - e).powf(1.0 - e)).sqrt()
    }

    pub(crate) fn forward(&self, lon: f64, lat: f64) -> DVec2 {
        let rho = self.scale * t(self.pole * lat);
        let dlon = lon - self.lon0;
        DVec2::new(
            self.x0 + rho * dlon.sin(),
            self.y0 - self.pole * rho * dlon.cos(),
        )
    }

    pub(crate) fn inverse(&self, p: DVec2) -> (f64, f64) {
        let (dx, dy) = (p.x - self.x0, p.y - self.y0);
        let rho = dx.hypot(dy);
        let lat = self.pole * phi_from_t(rho / self.scale);
        let lon = self.lon0 + dx.atan2(-self.pole * dy);
        (wrap_pi(lon), lat)
    }

    /// Everything but the opposite pole.
    pub(crate) fn contains(&self, lat: f64) -> bool {
        self.pole * lat > -89.9f64.to_radians()
    }

    pub(crate) fn north(&self) -> bool {
        self.pole > 0.0
    }

    pub(crate) fn reference(&self) -> (f64, f64) {
        (self.lon0, self.true_lat)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Laea {
    lon0: f64,
    x0: f64,
    y0: f64,
    /// The authalic latitude of the origin.
    beta0: f64,
    rq: f64,
    d: f64,
    /// ±1 for a polar aspect, 0 for oblique.
    pole: f64,
    /// The latitude of the origin, the centre of true scale.
    lat0: f64,
}

impl Laea {
    pub(crate) fn new(lat0: f64, lon0: f64, x0: f64, y0: f64) -> Self {
        let rq = authalic_radius();
        let beta0 = authalic(lat0);
        let polar = (lat0.abs() - FRAC_PI_2).abs() < 1e-12;
        Self {
            lon0,
            x0,
            y0,
            beta0,
            rq,
            d: if polar {
                1.0
            } else {
                A * m(lat0) / (rq * beta0.cos())
            },
            pole: if polar { lat0.signum() } else { 0.0 },
            lat0,
        }
    }

    pub(crate) fn forward(&self, lon: f64, lat: f64) -> DVec2 {
        let dlon = lon - self.lon0;
        if self.pole != 0.0 {
            let qp = q(FRAC_PI_2);
            let rho = A * (qp - self.pole * q(lat)).max(0.0).sqrt();
            return DVec2::new(
                self.x0 + rho * dlon.sin(),
                self.y0 - self.pole * rho * dlon.cos(),
            );
        }
        let beta = authalic(lat);
        let (sb0, cb0) = self.beta0.sin_cos();
        let (sb, cb) = beta.sin_cos();
        let b = self.rq
            * (2.0 / (1.0 + sb0 * sb + cb0 * cb * dlon.cos()))
                .max(0.0)
                .sqrt();
        DVec2::new(
            self.x0 + b * self.d * cb * dlon.sin(),
            self.y0 + b / self.d * (cb0 * sb - sb0 * cb * dlon.cos()),
        )
    }

    pub(crate) fn inverse(&self, p: DVec2) -> (f64, f64) {
        let (dx, dy) = (p.x - self.x0, p.y - self.y0);
        if self.pole != 0.0 {
            let rho = dx.hypot(dy);
            let qp = q(FRAC_PI_2);
            let beta = (self.pole * (1.0 - rho * rho / (A * A * qp)))
                .clamp(-1.0, 1.0)
                .asin();
            let lon = self.lon0 + dx.atan2(-self.pole * dy);
            return (wrap_pi(lon), from_authalic(beta));
        }
        let rho = (dx / self.d).hypot(self.d * dy);
        if rho < 1e-9 {
            return (self.lon0, from_authalic(self.beta0));
        }
        let c = 2.0 * (rho / (2.0 * self.rq)).clamp(-1.0, 1.0).asin();
        let (sc, cc) = c.sin_cos();
        let (sb0, cb0) = self.beta0.sin_cos();
        let beta = (cc * sb0 + self.d * dy * sc * cb0 / rho)
            .clamp(-1.0, 1.0)
            .asin();
        let lon =
            self.lon0 + (dx * sc).atan2(self.d * rho * cb0 * cc - self.d * self.d * dy * sb0 * sc);
        (wrap_pi(lon), from_authalic(beta))
    }

    pub(crate) fn reference(&self) -> (f64, f64) {
        (self.lon0, self.lat0)
    }

    /// Everything but the antipode of the origin.
    pub(crate) fn contains(&self, lon: f64, lat: f64) -> bool {
        self.angular_cos(lon, lat) > -0.9999
    }

    /// The cosine of the angle between the origin and a position on the
    /// authalic sphere.
    pub(crate) fn angular_cos(&self, lon: f64, lat: f64) -> f64 {
        let beta = authalic(lat);
        self.beta0.sin() * beta.sin() + self.beta0.cos() * beta.cos() * (lon - self.lon0).cos()
    }
}
