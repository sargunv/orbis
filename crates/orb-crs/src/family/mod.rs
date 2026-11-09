//! Projection families: f64 forward and inverse maps between geodetic
//! radians and a plane's world units. Private: the public face is
//! [`Projection`](crate::Projection), chosen by descriptor.

mod azimuthal;
mod conic;
mod datum;
mod ellipsoid;
mod oblique;
mod pseudo;
mod tm;

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use glam::DVec2;

pub(crate) use azimuthal::{Laea, PolarStereographic};
pub(crate) use conic::Conic;
pub(crate) use datum::Datum;
pub(crate) use ellipsoid::{A, Ellipsoid};
pub(crate) use oblique::{ObliqueStereographic, SwissObliqueMercator};
pub(crate) use pseudo::Pseudo;
pub(crate) use tm::{TransverseMercator, wrap_pi};

/// Latitude at which Web Mercator's world is square, in radians.
pub(crate) const MERCATOR_MAX_LAT: f64 = 1.484_422_229_745_332_4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Family {
    /// Spherical Web Mercator on the WGS84 semi-major axis.
    WebMercator,
    /// Longitude and latitude in degrees as x and y.
    Geographic,
    /// Plate carrée in metres on the sphere of the semi-major axis.
    Equirectangular {
        lon0: f64,
    },
    TransverseMercator(TransverseMercator),
    PolarStereographic(PolarStereographic),
    Conic(Conic),
    Laea(Laea),
    Pseudo(Pseudo),
    ObliqueStereographic(ObliqueStereographic),
    SwissObliqueMercator(SwissObliqueMercator),
}

impl Family {
    /// Projects radians into world units.
    pub(crate) fn forward(&self, lon: f64, lat: f64) -> DVec2 {
        match self {
            Self::WebMercator => {
                let lat = lat.clamp(-MERCATOR_MAX_LAT, MERCATOR_MAX_LAT);
                DVec2::new(A * lon, A * (FRAC_PI_4 + lat / 2.0).tan().ln())
            }
            Self::Geographic => DVec2::new(lon.to_degrees(), lat.to_degrees()),
            Self::Equirectangular { lon0 } => DVec2::new(A * wrap_pi(lon - lon0), A * lat),
            Self::TransverseMercator(f) => f.forward(lon, lat),
            Self::PolarStereographic(f) => f.forward(lon, lat),
            Self::Conic(f) => f.forward(lon, lat),
            Self::Laea(f) => f.forward(lon, lat),
            Self::Pseudo(f) => f.forward(lon, lat),
            Self::ObliqueStereographic(f) => f.forward(lon, lat),
            Self::SwissObliqueMercator(f) => f.forward(lon, lat),
        }
    }

    /// Radians of a world position.
    pub(crate) fn inverse(&self, p: DVec2) -> (f64, f64) {
        match self {
            Self::WebMercator => (p.x / A, 2.0 * (p.y / A).exp().atan() - FRAC_PI_2),
            Self::Geographic => (p.x.to_radians(), p.y.to_radians()),
            Self::Equirectangular { lon0 } => (lon0 + p.x / A, p.y / A),
            Self::TransverseMercator(f) => f.inverse(p),
            Self::PolarStereographic(f) => f.inverse(p),
            Self::Conic(f) => f.inverse(p),
            Self::Laea(f) => f.inverse(p),
            Self::Pseudo(f) => f.inverse(p),
            Self::ObliqueStereographic(f) => f.inverse(p),
            Self::SwissObliqueMercator(f) => f.inverse(p),
        }
    }

    /// Whether the family maps a position to a finite, meaningful point.
    pub(crate) fn contains(&self, lon: f64, lat: f64) -> bool {
        match self {
            Self::WebMercator | Self::Geographic | Self::Equirectangular { .. } => true,
            Self::TransverseMercator(f) => f.contains(lon),
            Self::PolarStereographic(f) => f.contains(lat),
            Self::Conic(f) => f.contains(lat),
            Self::Laea(f) => f.contains(lon, lat),
            Self::Pseudo(_) => true,
            Self::ObliqueStereographic(f) => f.contains(lon, lat),
            Self::SwissObliqueMercator(f) => f.contains(lon, lat),
        }
    }

    /// Whether `forward` puts a position it contains on the world's edge
    /// instead of projecting it: Web Mercator's past ±85.05°.
    pub(crate) fn clamps(&self, lat: f64) -> bool {
        matches!(self, Self::WebMercator) && lat.abs() > MERCATOR_MAX_LAT
    }

    /// Where a projection's default world extent comes from: the positions
    /// it shows with moderate distortion.
    pub(crate) fn shows(&self, lon: f64, lat: f64) -> bool {
        let near = |lon0: f64, max: f64| wrap_pi(lon - lon0).abs() <= max.to_radians();
        match self {
            Self::WebMercator | Self::Geographic | Self::Equirectangular { .. } => true,
            Self::Pseudo(_) => true,
            Self::TransverseMercator(f) => near(f.lon0(), 30.0) && lat.abs() <= 84f64.to_radians(),
            Self::PolarStereographic(f) => {
                (if f.north() { lat } else { -lat }) >= 30f64.to_radians()
            }
            Self::Conic(f) => near(f.lon0(), 60.0) && lat.abs() <= 84f64.to_radians(),
            Self::Laea(f) => f.angular_cos(lon, lat) >= 0.0,
            Self::ObliqueStereographic(f) => f.shows(lon, lat),
            Self::SwissObliqueMercator(f) => f.shows(lon, lat),
        }
    }

    /// The position whose scale style zoom is measured at:
    /// where the family is true to scale, else its natural origin; `None`
    /// for Web Mercator, whose style zoom is its own convention.
    pub(crate) fn zoom_reference(&self) -> Option<(f64, f64)> {
        Some(match self {
            Self::WebMercator => return None,
            Self::Geographic => (0.0, 0.0),
            Self::Equirectangular { lon0 } => (*lon0, 0.0),
            Self::TransverseMercator(f) => f.reference(),
            Self::PolarStereographic(f) => f.reference(),
            Self::Conic(f) => f.reference(),
            Self::Laea(f) => f.reference(),
            Self::Pseudo(f) => (f.lon0(), 0.0),
            Self::ObliqueStereographic(f) => f.reference(),
            Self::SwissObliqueMercator(f) => f.reference(),
        })
    }

    /// The central meridian of a plane cut along its antimeridian, in
    /// radians: the cut's two sides lie at opposite edges of the world.
    pub(crate) fn seam(&self) -> Option<f64> {
        match self {
            Self::Pseudo(f) => Some(f.lon0()),
            Self::Conic(f) => Some(f.lon0()),
            _ => None,
        }
    }

    /// The x period after which the world repeats, if it wraps.
    pub(crate) fn wrap_period(&self) -> Option<f64> {
        match self {
            Self::WebMercator | Self::Equirectangular { .. } => Some(2.0 * PI * A),
            Self::Geographic => Some(360.0),
            _ => None,
        }
    }
}

/// A family on a datum: positions in and out are WGS 84 radians, shifted
/// to and from the datum the family's formulas take, if not WGS 84.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Plane {
    pub(crate) family: Family,
    datum: Option<Datum>,
}

impl Plane {
    pub(crate) fn new(family: Family, datum: Option<Datum>) -> Self {
        Self { family, datum }
    }

    /// A WGS 84 position on the family's datum.
    fn local(&self, lon: f64, lat: f64) -> (f64, f64) {
        match &self.datum {
            Some(datum) => datum.local(lon, lat),
            None => (lon, lat),
        }
    }

    fn wgs84(&self, (lon, lat): (f64, f64)) -> (f64, f64) {
        match &self.datum {
            Some(datum) => datum.wgs84(lon, lat),
            None => (lon, lat),
        }
    }

    pub(crate) fn forward(&self, lon: f64, lat: f64) -> DVec2 {
        let (lon, lat) = self.local(lon, lat);
        self.family.forward(lon, lat)
    }

    pub(crate) fn inverse(&self, p: DVec2) -> (f64, f64) {
        let local = self.family.inverse(p);
        if !(local.0.is_finite() && local.1.is_finite()) {
            return local;
        }
        self.wgs84(local)
    }

    pub(crate) fn contains(&self, lon: f64, lat: f64) -> bool {
        let (lon, lat) = self.local(lon, lat);
        self.family.contains(lon, lat)
    }

    pub(crate) fn clamps(&self, lon: f64, lat: f64) -> bool {
        self.family.clamps(self.local(lon, lat).1)
    }

    pub(crate) fn shows(&self, lon: f64, lat: f64) -> bool {
        let (lon, lat) = self.local(lon, lat);
        self.family.shows(lon, lat)
    }

    pub(crate) fn zoom_reference(&self) -> Option<(f64, f64)> {
        self.family.zoom_reference().map(|p| self.wgs84(p))
    }

    pub(crate) fn seam(&self) -> Option<f64> {
        self.family.seam()
    }

    pub(crate) fn wrap_period(&self) -> Option<f64> {
        self.family.wrap_period()
    }
}
