//! Pseudocylindrical world projections: Equal Earth on the ellipsoid's
//! authalic sphere (Šavrič, Patterson and Jenny 2018; EPSG method 1078) and
//! Natural Earth on the sphere of the semi-major axis (PROJ `natearth`).

use glam::DVec2;

use super::ellipsoid::{A, authalic, authalic_radius, from_authalic};
use super::tm::wrap_pi;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Pseudo {
    EqualEarth { lon0: f64, rq: f64 },
    NaturalEarth { lon0: f64 },
}

const EE: [f64; 4] = [1.340264, -0.081106, 0.000893, 0.003796];
const NE_A: [f64; 5] = [0.8707, -0.131979, -0.013791, 0.003971, -0.001529];
const NE_B: [f64; 5] = [1.007226, 0.015085, -0.044475, 0.028874, -0.005916];
const SQRT3: f64 = 1.732_050_807_568_877_2;

impl Pseudo {
    pub(crate) fn equal_earth(lon0: f64) -> Self {
        Self::EqualEarth {
            lon0,
            rq: authalic_radius(),
        }
    }

    pub(crate) fn natural_earth(lon0: f64) -> Self {
        Self::NaturalEarth { lon0 }
    }

    pub(crate) fn lon0(&self) -> f64 {
        match *self {
            Self::EqualEarth { lon0, .. } | Self::NaturalEarth { lon0 } => lon0,
        }
    }

    pub(crate) fn forward(&self, lon: f64, lat: f64) -> DVec2 {
        let dlon = wrap_pi(lon - self.lon0());
        match *self {
            Self::EqualEarth { rq, .. } => {
                let theta = (SQRT3 / 2.0 * authalic(lat).sin()).asin();
                let t2 = theta * theta;
                let t6 = t2 * t2 * t2;
                let [a1, a2, a3, a4] = EE;
                let d = 9.0 * a4 * t6 * t2 + 7.0 * a3 * t6 + 3.0 * a2 * t2 + a1;
                DVec2::new(
                    rq * 2.0 * SQRT3 * dlon * theta.cos() / (3.0 * d),
                    rq * theta * (a4 * t6 * t2 + a3 * t6 + a2 * t2 + a1),
                )
            }
            Self::NaturalEarth { .. } => {
                let (p2, p4) = (lat * lat, lat.powi(4));
                let [a0, a1, a2, a3, a4] = NE_A;
                let [b0, b1, b2, b3, b4] = NE_B;
                DVec2::new(
                    A * dlon * (a0 + p2 * (a1 + p2 * (a2 + p4 * p2 * (a3 + p2 * a4)))),
                    A * lat * (b0 + p2 * (b1 + p4 * (b2 + b3 * p2 + b4 * p4))),
                )
            }
        }
    }

    pub(crate) fn inverse(&self, p: DVec2) -> (f64, f64) {
        match *self {
            Self::EqualEarth { lon0, rq } => {
                let [a1, a2, a3, a4] = EE;
                let y = p.y / rq;
                let mut theta = y;
                for _ in 0..20 {
                    let t2 = theta * theta;
                    let t6 = t2 * t2 * t2;
                    let f = theta * (a4 * t6 * t2 + a3 * t6 + a2 * t2 + a1) - y;
                    let fp = 9.0 * a4 * t6 * t2 + 7.0 * a3 * t6 + 3.0 * a2 * t2 + a1;
                    let step = f / fp;
                    theta -= step;
                    if step.abs() < 1e-14 {
                        break;
                    }
                }
                let theta = theta.clamp(-std::f64::consts::FRAC_PI_3, std::f64::consts::FRAC_PI_3);
                let t2 = theta * theta;
                let t6 = t2 * t2 * t2;
                let d = 9.0 * a4 * t6 * t2 + 7.0 * a3 * t6 + 3.0 * a2 * t2 + a1;
                let dlon = 3.0 * p.x * d / (2.0 * SQRT3 * rq * theta.cos());
                let beta = (2.0 * theta.sin() / SQRT3).clamp(-1.0, 1.0).asin();
                (lon0 + dlon, from_authalic(beta))
            }
            Self::NaturalEarth { lon0 } => {
                let [a0, a1, a2, a3, a4] = NE_A;
                let [b0, b1, b2, b3, b4] = NE_B;
                let y = p.y / A;
                let mut lat = y;
                for _ in 0..30 {
                    let (p2, p4) = (lat * lat, lat.powi(4));
                    let f = lat * (b0 + p2 * (b1 + p4 * (b2 + b3 * p2 + b4 * p4))) - y;
                    let fp =
                        b0 + p2 * (3.0 * b1 + p4 * (7.0 * b2 + 9.0 * b3 * p2 + 11.0 * b4 * p4));
                    let step = f / fp;
                    lat -= step;
                    if step.abs() < 1e-14 {
                        break;
                    }
                }
                let lat = lat.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
                let (p2, p4) = (lat * lat, lat.powi(4));
                let dlon = p.x / A / (a0 + p2 * (a1 + p2 * (a2 + p4 * p2 * (a3 + p2 * a4))));
                (lon0 + dlon, lat)
            }
        }
    }
}
