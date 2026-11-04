//! Ellipsoidal Lambert conformal conic (two standard parallels) and Albers
//! equal-area conic (EPSG Guidance Note 7-2, §3.1.1 and §3.4.1).

use glam::DVec2;

use super::ellipsoid::{A, m, phi_from_q, phi_from_t, q, t};
use super::tm::wrap_pi;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Conic {
    lon0: f64,
    x0: f64,
    y0: f64,
    n: f64,
    /// LCC: `a F`; Albers: `C`.
    c: f64,
    /// The radius of the latitude of origin.
    rho0: f64,
    equal_area: bool,
    /// Midway between the standard parallels, where scale is nearly true.
    middle: f64,
}

impl Conic {
    pub(crate) fn lcc(lat1: f64, lat2: f64, lat0: f64, lon0: f64, x0: f64, y0: f64) -> Self {
        let (m1, m2, t1, t2) = (m(lat1), m(lat2), t(lat1), t(lat2));
        let n = if (lat1 - lat2).abs() < 1e-12 {
            lat1.sin()
        } else {
            (m1.ln() - m2.ln()) / (t1.ln() - t2.ln())
        };
        let af = A * m1 / (n * t1.powf(n));
        Self {
            lon0,
            x0,
            y0,
            n,
            c: af,
            rho0: af * t(lat0).powf(n),
            equal_area: false,
            middle: (lat1 + lat2) / 2.0,
        }
    }

    pub(crate) fn albers(lat1: f64, lat2: f64, lat0: f64, lon0: f64, x0: f64, y0: f64) -> Self {
        let (m1, m2, q1, q2) = (m(lat1), m(lat2), q(lat1), q(lat2));
        let n = if (lat1 - lat2).abs() < 1e-12 {
            lat1.sin()
        } else {
            (m1 * m1 - m2 * m2) / (q2 - q1)
        };
        let c = m1 * m1 + n * q1;
        let mut conic = Self {
            lon0,
            x0,
            y0,
            n,
            c,
            rho0: 0.0,
            equal_area: true,
            middle: (lat1 + lat2) / 2.0,
        };
        conic.rho0 = conic.rho(lat0);
        conic
    }

    fn rho(&self, phi: f64) -> f64 {
        if self.equal_area {
            A * (self.c - self.n * q(phi)).max(0.0).sqrt() / self.n
        } else {
            self.c * t(phi).powf(self.n)
        }
    }

    pub(crate) fn forward(&self, lon: f64, lat: f64) -> DVec2 {
        let theta = self.n * wrap_pi(lon - self.lon0);
        let rho = self.rho(lat);
        DVec2::new(
            self.x0 + rho * theta.sin(),
            self.y0 + self.rho0 - rho * theta.cos(),
        )
    }

    pub(crate) fn inverse(&self, p: DVec2) -> (f64, f64) {
        let s = self.n.signum();
        let (dx, dy) = (p.x - self.x0, self.rho0 - (p.y - self.y0));
        let rho = s * dx.hypot(dy);
        let theta = (s * dx).atan2(s * dy);
        let lat = if self.equal_area {
            phi_from_q((self.c - rho * rho * self.n * self.n / (A * A)) / self.n)
        } else if rho == 0.0 {
            std::f64::consts::FRAC_PI_2.copysign(self.n)
        } else {
            phi_from_t((rho / self.c).powf(1.0 / self.n))
        };
        (self.lon0 + theta / self.n, lat)
    }

    /// Whether a latitude has a finite image: LCC excludes the pole the cone
    /// opens toward.
    pub(crate) fn contains(&self, lat: f64) -> bool {
        self.equal_area || lat * self.n.signum() > -89.9f64.to_radians()
    }

    pub(crate) fn lon0(&self) -> f64 {
        self.lon0
    }

    pub(crate) fn reference(&self) -> (f64, f64) {
        (self.lon0, self.middle)
    }
}
