//! Points and number conversions with JavaScript semantics.
//!
//! Layout reproduces GL JS's doubles bit for bit, so these follow
//! `@mapbox/point-geometry` and the ECMAScript conversions exactly.
//! Trigonometry goes through `libm`, the same code natively and in wasm.

use std::f64::consts::PI;

/// GL JS's `EXTENT`: tile units per tile side.
pub const EXTENT: f64 = 8192.0;

/// A point in tile units or layout pixels (`@mapbox/point-geometry`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Point {
    pub(crate) x: f64,
    pub(crate) y: f64,
}

impl Point {
    pub(crate) const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub(crate) fn dist_sqr(self, p: Self) -> f64 {
        let dx = p.x - self.x;
        let dy = p.y - self.y;
        dx * dx + dy * dy
    }

    pub(crate) fn dist(self, p: Self) -> f64 {
        self.dist_sqr(p).sqrt()
    }

    /// The angle from `b` to this point (`angleTo`).
    pub(crate) fn angle_to(self, b: Self) -> f64 {
        libm::atan2(self.y - b.y, self.x - b.x)
    }

    pub(crate) fn rotate(self, angle: f64) -> Self {
        let (cos, sin) = (libm::cos(angle), libm::sin(angle));
        Self::new(cos * self.x - sin * self.y, sin * self.x + cos * self.y)
    }

    pub(crate) fn rotate_around(self, angle: f64, p: Self) -> Self {
        let (cos, sin) = (libm::cos(angle), libm::sin(angle));
        Self::new(
            p.x + cos * (self.x - p.x) - sin * (self.y - p.y),
            p.y + sin * (self.x - p.x) + cos * (self.y - p.y),
        )
    }

    pub(crate) fn mat_mult(self, m: [f64; 4]) -> Self {
        Self::new(m[0] * self.x + m[1] * self.y, m[2] * self.x + m[3] * self.y)
    }

    pub(crate) fn round(self) -> Self {
        Self::new(js_round(self.x), js_round(self.y))
    }

    pub(crate) fn add(self, p: Self) -> Self {
        Self::new(self.x + p.x, self.y + p.y)
    }
}

/// The rotation matrix GL JS builds for `*-rotate` (`[cos, -sin, sin, cos]`).
pub(crate) fn rotation_matrix(radians: f64) -> [f64; 4] {
    let (sin, cos) = (libm::sin(radians), libm::cos(radians));
    [cos, -sin, sin, cos]
}

pub(crate) fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

pub(crate) use orb_text::{js_max, js_min};
pub(crate) use orb_tile::js_round;

/// ECMAScript `ToInt16`, what storing into an `Int16Array` does.
pub(crate) fn to_i16(x: f64) -> i16 {
    to_int_bits(x, 16) as u16 as i16
}

/// ECMAScript `ToUint16`, what storing into a `Uint16Array` does.
pub(crate) fn to_u16(x: f64) -> u16 {
    to_int_bits(x, 16) as u16
}

/// Truncates toward zero and wraps modulo 2^bits; NaN and ±∞ become 0.
fn to_int_bits(x: f64, bits: u32) -> u64 {
    if !x.is_finite() {
        return 0;
    }
    let modulus = (1u64 << bits) as f64;
    x.trunc().rem_euclid(modulus) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn conversions_follow_javascript() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(js_round(0.499_999_999_999_999_94), 0.0);
        assert_eq!(js_round(-0.6), -1.0);
        assert_eq!(to_i16(-1.9), -1);
        assert_eq!(to_i16(32768.0), -32768);
        assert_eq!(to_i16(f64::NAN), 0);
        assert_eq!(to_u16(-1.0), 65535);
        assert_eq!(to_u16(1711.9), 1711);
        assert!(js_max(1.0, f64::NAN).is_nan());
    }
}
