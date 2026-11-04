//! f64 geodesy primitives shared by every crate that reasons about positions
//! on the Earth.

#![forbid(unsafe_code)]

/// The WGS84 ellipsoid.
pub mod wgs84 {
    /// Semi-major axis in metres.
    pub const SEMI_MAJOR_M: f64 = 6_378_137.0;
    /// Flattening. GRS80's differs in the eleventh significant digit, a
    /// tenth of a millimetre at the pole, so it is the same ellipsoid here.
    pub const FLATTENING: f64 = 1.0 / 298.257_223_563;
    /// First eccentricity squared.
    pub const E2: f64 = FLATTENING * (2.0 - FLATTENING);
    /// Mean radius of the IUGG sphere, which v8 feature heights use.
    pub const MEAN_RADIUS_M: f64 = 6_371_008.8;
}

/// A geodetic position in degrees on the WGS84 datum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geodetic {
    /// Longitude in degrees, east positive.
    pub lon: f64,
    /// Latitude in degrees, north positive.
    pub lat: f64,
}

impl Geodetic {
    pub const fn new(lon: f64, lat: f64) -> Self {
        Self { lon, lat }
    }

    /// The same position with longitude wrapped into `[-180, 180)`.
    pub fn wrapped(self) -> Self {
        Self {
            lon: (self.lon + 180.0).rem_euclid(360.0) - 180.0,
            lat: self.lat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_longitude() {
        assert_eq!(
            Geodetic::new(190.0, 10.0).wrapped(),
            Geodetic::new(-170.0, 10.0)
        );
        assert_eq!(Geodetic::new(-180.0, 0.0).wrapped().lon, -180.0);
        assert_eq!(Geodetic::new(540.0, 0.0).wrapped().lon, -180.0);
    }
}
