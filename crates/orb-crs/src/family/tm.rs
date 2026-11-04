//! Ellipsoidal transverse Mercator by Krüger's n-series to sixth order
//! (Karney 2011, as PROJ's default `tmerc`): sub-millimetre within 4,000 km
//! of the central meridian.

use glam::DVec2;

use super::ellipsoid::{A, E2, e};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TransverseMercator {
    lon0: f64,
    k0: f64,
    x0: f64,
    y0: f64,
    /// Rectifying radius times `k0`.
    ka: f64,
    /// Northing of the latitude of origin before the false northing.
    m0: f64,
    alpha: [f64; 6],
    beta: [f64; 6],
}

impl TransverseMercator {
    pub(crate) fn new(lon0: f64, lat0: f64, k0: f64, x0: f64, y0: f64) -> Self {
        let f = E2 / (1.0 + (1.0 - E2).sqrt());
        let n = f / (2.0 - f);
        let (n2, n3) = (n * n, n * n * n);
        let (n4, n5, n6) = (n2 * n2, n2 * n3, n3 * n3);
        let a = A / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0 + n6 / 256.0);
        let alpha = [
            n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0 + 41.0 * n4 / 180.0 - 127.0 * n5 / 288.0
                + 7891.0 * n6 / 37800.0,
            13.0 * n2 / 48.0 - 3.0 * n3 / 5.0 + 557.0 * n4 / 1440.0 + 281.0 * n5 / 630.0
                - 1983433.0 * n6 / 1935360.0,
            61.0 * n3 / 240.0 - 103.0 * n4 / 140.0
                + 15061.0 * n5 / 26880.0
                + 167603.0 * n6 / 181440.0,
            49561.0 * n4 / 161280.0 - 179.0 * n5 / 168.0 + 6601661.0 * n6 / 7257600.0,
            34729.0 * n5 / 80640.0 - 3418889.0 * n6 / 1995840.0,
            212378941.0 * n6 / 319334400.0,
        ];
        let beta = [
            n / 2.0 - 2.0 * n2 / 3.0 + 37.0 * n3 / 96.0 - n4 / 360.0 - 81.0 * n5 / 512.0
                + 96199.0 * n6 / 604800.0,
            n2 / 48.0 + n3 / 15.0 - 437.0 * n4 / 1440.0 + 46.0 * n5 / 105.0
                - 1118711.0 * n6 / 3870720.0,
            17.0 * n3 / 480.0 - 37.0 * n4 / 840.0 - 209.0 * n5 / 4480.0 + 5569.0 * n6 / 90720.0,
            4397.0 * n4 / 161280.0 - 11.0 * n5 / 504.0 - 830251.0 * n6 / 7257600.0,
            4583.0 * n5 / 161280.0 - 108847.0 * n6 / 3991680.0,
            20648693.0 * n6 / 638668800.0,
        ];
        let mut tm = Self {
            lon0,
            k0,
            x0,
            y0,
            ka: k0 * a,
            m0: 0.0,
            alpha,
            beta,
        };
        tm.m0 = tm.xi_eta(lat0, 0.0).x * tm.ka;
        tm
    }

    /// Gauss–Krüger ξ (northing) and η (easting) before scaling.
    fn xi_eta(&self, phi: f64, dlon: f64) -> DVec2 {
        let e = e();
        let tau = phi.tan();
        let sigma = (e * (e * phi.sin()).atanh()).sinh();
        let taup = tau * sigma.hypot(1.0) - sigma * tau.hypot(1.0);
        let xip = taup.atan2(dlon.cos());
        let etap = (dlon.sin() / taup.hypot(dlon.cos())).asinh();
        let (mut xi, mut eta) = (xip, etap);
        for (j, a) in self.alpha.iter().enumerate() {
            let k = 2.0 * (j + 1) as f64;
            xi += a * (k * xip).sin() * (k * etap).cosh();
            eta += a * (k * xip).cos() * (k * etap).sinh();
        }
        DVec2::new(xi, eta)
    }

    pub(crate) fn forward(&self, lon: f64, lat: f64) -> DVec2 {
        let dlon = wrap_pi(lon - self.lon0);
        let v = self.xi_eta(lat, dlon);
        DVec2::new(self.x0 + self.ka * v.y, self.y0 + self.ka * v.x - self.m0)
    }

    pub(crate) fn inverse(&self, p: DVec2) -> (f64, f64) {
        let xi = (p.y - self.y0 + self.m0) / self.ka;
        let eta = (p.x - self.x0) / self.ka;
        let (mut xip, mut etap) = (xi, eta);
        for (j, b) in self.beta.iter().enumerate() {
            let k = 2.0 * (j + 1) as f64;
            xip -= b * (k * xi).sin() * (k * eta).cosh();
            etap -= b * (k * xi).cos() * (k * eta).sinh();
        }
        let dlon = etap.sinh().atan2(xip.cos());
        let taup = xip.sin() / etap.sinh().hypot(xip.cos());
        let lat = tau_from_taup(taup).atan();
        (self.lon0 + dlon, lat)
    }

    /// Whether the series is meaningful at a longitude offset: within a
    /// quarter turn of the central meridian.
    pub(crate) fn contains(&self, lon: f64) -> bool {
        wrap_pi(lon - self.lon0).abs() < 80f64.to_radians()
    }

    pub(crate) fn lon0(&self) -> f64 {
        self.lon0
    }
}

/// tan φ from the conformal latitude's tangent, by Newton's method
/// (Karney 2011, eq. 19–21).
pub(crate) fn tau_from_taup(taup: f64) -> f64 {
    let e = e();
    let mut tau = taup;
    for _ in 0..8 {
        let sigma = (e * (e * tau / tau.hypot(1.0)).atanh()).sinh();
        let tp = tau * sigma.hypot(1.0) - sigma * tau.hypot(1.0);
        let d = (taup - tp) / tp.hypot(1.0) * (1.0 + (1.0 - E2) * tau * tau)
            / ((1.0 - E2) * tau.hypot(1.0));
        tau += d;
        if d.abs() < 1e-14 * tau.abs().max(1.0) {
            break;
        }
    }
    tau
}

/// An angle wrapped into [-π, π).
pub(crate) fn wrap_pi(a: f64) -> f64 {
    use std::f64::consts::PI;
    (a + PI).rem_euclid(2.0 * PI) - PI
}
