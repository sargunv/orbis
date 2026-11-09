//! Conformal projections of an ellipsoid through Gauss's conformal sphere:
//! the oblique stereographic (EPSG method 9809, PROJ's `sterea`) of the
//! Dutch grid, and the Swiss oblique Mercator (EPSG method 9814, PROJ's
//! `somerc`) of the Swiss one (EPSG Guidance Note 7-2, §3.5.1, and
//! swisstopo's formulas for the Swiss projection).

use std::f64::consts::{FRAC_PI_2, PI};

use glam::DVec2;

use super::ellipsoid::Ellipsoid;
use super::tm::{tau_from_taup, wrap_pi};

/// Gauss's conformal sphere about an origin: true to the ellipsoid's
/// scale and its curvature at the origin, so a sphere projection on it
/// stays conformal and near true scale around there.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GaussSphere {
    e: f64,
    e2: f64,
    lon0: f64,
    /// The sphere's longitude scale `n`.
    n: f64,
    /// The isometric latitudes' offset: ψ' = n ψ + `c`.
    c: f64,
    /// The origin's latitude on the sphere, χ0.
    chi0: f64,
    /// The sphere's radius √(ρ0 ν0).
    radius: f64,
}

impl GaussSphere {
    fn new(ellipsoid: Ellipsoid, lon0: f64, lat0: f64) -> Self {
        let Ellipsoid { a, e2 } = ellipsoid;
        let e = ellipsoid.e();
        let s = lat0.sin();
        let n = (1.0 + e2 * lat0.cos().powi(4) / (1.0 - e2)).sqrt();
        let chi0 = (s / n).asin();
        let mut sphere = Self {
            e,
            e2,
            lon0,
            n,
            c: 0.0,
            chi0,
            radius: a * (1.0 - e2).sqrt() / (1.0 - e2 * s * s),
        };
        sphere.c = chi0.tan().asinh() - n * sphere.isometric(lat0);
        sphere
    }

    /// The ellipsoid's isometric latitude ψ.
    fn isometric(&self, phi: f64) -> f64 {
        phi.tan().asinh() - self.e * (self.e * phi.sin()).atanh()
    }

    /// A position's longitude from the origin and latitude on the sphere.
    fn onto(&self, lon: f64, lat: f64) -> (f64, f64) {
        let chi = (self.n * self.isometric(lat) + self.c).sinh().atan();
        (self.n * wrap_pi(lon - self.lon0), chi)
    }

    /// The ellipsoid's longitude and latitude of a sphere position.
    fn back(&self, dlon: f64, chi: f64) -> (f64, f64) {
        let psi = (chi.tan().asinh() - self.c) / self.n;
        let lat = tau_from_taup(psi.sinh(), self.e2).atan();
        (self.lon0 + dlon / self.n, lat)
    }

    /// The angular distance's cosine from the origin, on the sphere.
    fn angular_cos(&self, lon: f64, lat: f64) -> f64 {
        let (dlon, chi) = self.onto(lon, lat);
        chi.sin() * self.chi0.sin() + chi.cos() * self.chi0.cos() * dlon.cos()
    }
}

/// Oblique (double) stereographic: Gauss's sphere projected
/// stereographically from its origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ObliqueStereographic {
    sphere: GaussSphere,
    lat0: f64,
    /// Twice the sphere's radius times the scale at the origin.
    r2k: f64,
    x0: f64,
    y0: f64,
}

impl ObliqueStereographic {
    pub(crate) fn new(
        ellipsoid: Ellipsoid,
        lon0: f64,
        lat0: f64,
        k0: f64,
        x0: f64,
        y0: f64,
    ) -> Self {
        let sphere = GaussSphere::new(ellipsoid, lon0, lat0);
        Self {
            r2k: 2.0 * sphere.radius * k0,
            sphere,
            lat0,
            x0,
            y0,
        }
    }

    pub(crate) fn forward(&self, lon: f64, lat: f64) -> DVec2 {
        let (dlon, chi) = self.sphere.onto(lon, lat);
        let (s0, c0) = self.sphere.chi0.sin_cos();
        let (s, c) = chi.sin_cos();
        let b = 1.0 + s * s0 + c * c0 * dlon.cos();
        DVec2::new(
            self.x0 + self.r2k * c * dlon.sin() / b,
            self.y0 + self.r2k * (s * c0 - c * s0 * dlon.cos()) / b,
        )
    }

    pub(crate) fn inverse(&self, p: DVec2) -> (f64, f64) {
        let (x, y) = (p.x - self.x0, p.y - self.y0);
        let rho = x.hypot(y);
        let (s0, c0) = self.sphere.chi0.sin_cos();
        if rho == 0.0 {
            return self.sphere.back(0.0, self.sphere.chi0);
        }
        let (sc, cc) = (2.0 * (rho / self.r2k).atan()).sin_cos();
        let chi = (cc * s0 + y * sc * c0 / rho).clamp(-1.0, 1.0).asin();
        let dlon = (x * sc).atan2(rho * c0 * cc - y * s0 * sc);
        self.sphere.back(dlon, chi)
    }

    /// Everywhere but near the antipode of the origin, where the plane
    /// runs out to infinity.
    pub(crate) fn contains(&self, lon: f64, lat: f64) -> bool {
        lat.abs() < FRAC_PI_2 && self.sphere.angular_cos(lon, lat) > -0.9
    }

    /// Within 30° of the origin, where scale grows by under 7%.
    pub(crate) fn shows(&self, lon: f64, lat: f64) -> bool {
        self.sphere.angular_cos(lon, lat) >= 30f64.to_radians().cos()
    }

    /// The origin, where the scale is the projection's `k0`.
    pub(crate) fn reference(&self) -> (f64, f64) {
        (self.sphere.lon0, self.lat0)
    }
}

/// Swiss oblique Mercator: Gauss's sphere turned so the origin lies on
/// its equator, then a Mercator of it. True to scale along the great
/// circle through the origin at right angles to its meridian.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SwissObliqueMercator {
    sphere: GaussSphere,
    lat0: f64,
    /// The sphere's radius times the scale on the oblique equator.
    rk: f64,
    x0: f64,
    y0: f64,
}

impl SwissObliqueMercator {
    pub(crate) fn new(
        ellipsoid: Ellipsoid,
        lon0: f64,
        lat0: f64,
        k0: f64,
        x0: f64,
        y0: f64,
    ) -> Self {
        let sphere = GaussSphere::new(ellipsoid, lon0, lat0);
        Self {
            rk: sphere.radius * k0,
            sphere,
            lat0,
            x0,
            y0,
        }
    }

    /// Latitude and longitude on the turned sphere.
    fn oblique(&self, lon: f64, lat: f64) -> (f64, f64) {
        let (dlon, chi) = self.sphere.onto(lon, lat);
        let (s0, c0) = self.sphere.chi0.sin_cos();
        let (s, c) = chi.sin_cos();
        let lat = (c0 * s - s0 * c * dlon.cos()).clamp(-1.0, 1.0).asin();
        let lon = (c * dlon.sin()).atan2(s0 * s + c0 * c * dlon.cos());
        (lon, lat)
    }

    pub(crate) fn forward(&self, lon: f64, lat: f64) -> DVec2 {
        let (lon, lat) = self.oblique(lon, lat);
        DVec2::new(
            self.x0 + self.rk * lon,
            self.y0 + self.rk * lat.tan().asinh(),
        )
    }

    pub(crate) fn inverse(&self, p: DVec2) -> (f64, f64) {
        let lon = ((p.x - self.x0) / self.rk).clamp(-PI, PI);
        let lat = ((p.y - self.y0) / self.rk).sinh().atan();
        let (s0, c0) = self.sphere.chi0.sin_cos();
        let (s, c) = lat.sin_cos();
        let chi = (s0 * c * lon.cos() + c0 * s).clamp(-1.0, 1.0).asin();
        let dlon = (c * lon.sin()).atan2(c0 * c * lon.cos() - s0 * s);
        self.sphere.back(dlon, chi)
    }

    /// Within 80° of the oblique equator, short of the cylinder's poles.
    pub(crate) fn contains(&self, lon: f64, lat: f64) -> bool {
        lat.abs() < FRAC_PI_2 && self.oblique(lon, lat).1.abs() < 80f64.to_radians()
    }

    /// Within 30° of the origin.
    pub(crate) fn shows(&self, lon: f64, lat: f64) -> bool {
        self.sphere.angular_cos(lon, lat) >= 30f64.to_radians().cos()
    }

    /// The origin, where the scale is the projection's `k0`.
    pub(crate) fn reference(&self) -> (f64, f64) {
        (self.sphere.lon0, self.lat0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deg(d: f64) -> f64 {
        d.to_radians()
    }

    #[test]
    fn rd_new_matches_guidance_note_example() {
        // EPSG Guidance Note 7-2's example for Amersfoort / RD New:
        // 53°N 6°E on Bessel 1841 is at (196105.283, 557057.739).
        let rd = ObliqueStereographic::new(
            Ellipsoid::BESSEL_1841,
            deg(5.387_638_888_888_89),
            deg(52.156_160_555_555_6),
            0.999_907_9,
            155_000.0,
            463_000.0,
        );
        let p = rd.forward(deg(6.0), deg(53.0));
        assert!(
            (p - DVec2::new(196_105.283, 557_057.739))
                .abs()
                .max_element()
                < 1e-3,
            "{p}"
        );
        let (lon, lat) = rd.inverse(p);
        assert!((lon - deg(6.0)).abs() < 1e-11 && (lat - deg(53.0)).abs() < 1e-11);
        let origin = rd.forward(deg(5.387_638_888_888_89), deg(52.156_160_555_555_6));
        assert!((origin - DVec2::new(155_000.0, 463_000.0)).length() < 1e-6);
    }

    #[test]
    fn swiss_grid_matches_proj() {
        // CH1903+ / LV95's origin, and two points as PROJ 9.8's `somerc`
        // puts them on Bessel 1841.
        let lv95 = SwissObliqueMercator::new(
            Ellipsoid::BESSEL_1841,
            deg(7.439_583_333_333_33),
            deg(46.952_405_555_555_6),
            1.0,
            2_600_000.0,
            1_200_000.0,
        );
        for (lon, lat, x, y) in [
            (
                7.439_583_333_333_33,
                46.952_405_555_555_6,
                2_600_000.0,
                1_200_000.0,
            ),
            (7.0, 46.0, 2_565_948.043_5, 1_094_232.448_0),
            (8.5, 47.3, 2_680_187.805_5, 1_239_181.419_8),
        ] {
            let p = lv95.forward(deg(lon), deg(lat));
            assert!((p - DVec2::new(x, y)).abs().max_element() < 1e-3, "{p}");
            let (l, b) = lv95.inverse(p);
            assert!((l - deg(lon)).abs() < 1e-11 && (b - deg(lat)).abs() < 1e-11);
        }
    }
}
