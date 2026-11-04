use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
use std::fmt;

use glam::DVec2;
use orb_geo::{Geodetic, wgs84};

use crate::WorldRect;

/// A map projection into a plane's world space (projected metres, y up).
///
/// Built from a projection descriptor's `type`, as in the scene document's
/// `{"type": "..."}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Projection {
    family: Family,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Family {
    /// Spherical Web Mercator on the WGS84 semi-major axis.
    WebMercator,
}

/// A projection descriptor that names no known projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownProjection(pub String);

impl fmt::Display for UnknownProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown projection `{}`", self.0)
    }
}

impl std::error::Error for UnknownProjection {}

impl Projection {
    /// The projection named by a descriptor `type`.
    pub fn from_descriptor(kind: &str) -> Result<Self, UnknownProjection> {
        let family = match kind {
            "web-mercator" | "mercator" | "EPSG:3857" => Family::WebMercator,
            _ => return Err(UnknownProjection(kind.to_string())),
        };
        Ok(Self { family })
    }

    /// Projects a geodetic position into world space.
    pub fn forward(&self, g: Geodetic) -> DVec2 {
        match self.family {
            Family::WebMercator => {
                let lat = g
                    .lat
                    .to_radians()
                    .clamp(-MERCATOR_MAX_LAT, MERCATOR_MAX_LAT);
                DVec2::new(
                    wgs84::SEMI_MAJOR_M * g.lon.to_radians(),
                    wgs84::SEMI_MAJOR_M * (FRAC_PI_4 + lat / 2.0).tan().ln(),
                )
            }
        }
    }

    /// The geodetic position of a world-space point.
    pub fn inverse(&self, p: DVec2) -> Geodetic {
        match self.family {
            Family::WebMercator => Geodetic::new(
                (p.x / wgs84::SEMI_MAJOR_M).to_degrees(),
                (2.0 * (p.y / wgs84::SEMI_MAJOR_M).exp().atan() - FRAC_PI_2).to_degrees(),
            ),
        }
    }

    /// Projected length per true ground length at `g`, along the parallel.
    pub fn scale_factor(&self, g: Geodetic) -> f64 {
        match self.family {
            Family::WebMercator => {
                1.0 / g
                    .lat
                    .to_radians()
                    .clamp(-MERCATOR_MAX_LAT, MERCATOR_MAX_LAT)
                    .cos()
            }
        }
    }

    /// World units per metre of v8 feature height at `g`. The style's height
    /// convention uses the IUGG mean sphere, while projected XY uses WGS84.
    pub fn style_height_scale(&self, g: Geodetic) -> f64 {
        match self.family {
            Family::WebMercator => self.scale_factor(g) * wgs84::SEMI_MAJOR_M / 6_371_008.8,
        }
    }

    /// The projected extent of the world.
    pub fn world_bounds(&self) -> WorldRect {
        match self.family {
            Family::WebMercator => {
                let half = std::f64::consts::PI * wgs84::SEMI_MAJOR_M;
                WorldRect {
                    min: DVec2::splat(-half),
                    max: DVec2::splat(half),
                }
            }
        }
    }

    /// The x period after which the world repeats, if it wraps.
    pub fn wrap_period(&self) -> Option<f64> {
        match self.family {
            Family::WebMercator => Some(self.world_bounds().size().x),
        }
    }
}

/// Latitude at which Web Mercator's world is square, in radians.
const MERCATOR_MAX_LAT: f64 = 1.484_422_229_745_332_4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mercator_round_trips() {
        let p = Projection::from_descriptor("web-mercator").unwrap();
        for g in [
            Geodetic::new(0.0, 0.0),
            Geodetic::new(-122.4, 47.6),
            Geodetic::new(179.9, -80.0),
        ] {
            let back = p.inverse(p.forward(g));
            assert!(
                (back.lon - g.lon).abs() < 1e-9 && (back.lat - g.lat).abs() < 1e-9,
                "{g:?} -> {back:?}"
            );
        }
    }

    #[test]
    fn mercator_world_is_square() {
        let p = Projection::from_descriptor("web-mercator").unwrap();
        let corner = p.forward(Geodetic::new(180.0, 85.051_128_779_806_59));
        let bounds = p.world_bounds();
        assert!((corner.x - bounds.max.x).abs() < 1e-6);
        assert!((corner.y - bounds.max.y).abs() < 1e-6);
    }

    #[test]
    fn rejects_unknown_descriptors() {
        assert!(Projection::from_descriptor("equal-earth").is_err());
    }
}
