use std::fmt;

use crate::{Geodetic, wgs84};
use glam::DVec2;

use crate::WorldRect;
use crate::family::{
    Conic, Ellipsoid, Family, Laea, MERCATOR_MAX_LAT, ObliqueStereographic, Plane,
    PolarStereographic, Pseudo, SwissObliqueMercator, TransverseMercator,
};

/// A map projection into a plane's world space (world units with y up:
/// projected metres, or degrees for a geographic plane).
///
/// Built from a [`Descriptor`], as in a style's root
/// `{"type": "...", ...}` `projection`, and turned back into one by
/// [`descriptor`](Self::descriptor).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    plane: Plane,
    name: Name,
    extent: WorldRect,
    northing_first: bool,
}

/// How a projection was named, for [`Projection::descriptor`].
#[derive(Clone, Copy, Debug, PartialEq)]
enum Name {
    Epsg(u32),
    Kind(Kind, Params),
}

/// A parametric projection kind of the descriptor syntax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    WebMercator,
    Geographic,
    Equirectangular,
    TransverseMercator,
    Utm,
    PolarStereographic,
    Lcc,
    Albers,
    Laea,
    EqualEarth,
    NaturalEarth,
    /// EPSG codes' own kinds, outside the descriptor syntax.
    ObliqueStereographic,
    SwissObliqueMercator,
}

impl Kind {
    const ALL: [(Kind, &'static str); 11] = [
        (Kind::WebMercator, "web-mercator"),
        (Kind::Geographic, "geographic"),
        (Kind::Equirectangular, "equirectangular"),
        (Kind::TransverseMercator, "transverse-mercator"),
        (Kind::Utm, "utm"),
        (Kind::PolarStereographic, "polar-stereographic"),
        (Kind::Lcc, "lcc"),
        (Kind::Albers, "albers"),
        (Kind::Laea, "laea"),
        (Kind::EqualEarth, "equal-earth"),
        (Kind::NaturalEarth, "natural-earth"),
    ];

    fn parse(name: &str) -> Option<Self> {
        match name {
            "mercator" => Some(Kind::WebMercator),
            "OGC:CRS84" | "CRS84" => Some(Kind::Geographic),
            _ => Self::ALL.iter().find(|(_, n)| *n == name).map(|(k, _)| *k),
        }
    }

    fn name(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(k, _)| *k == self)
            .expect("listed")
            .1
    }

    /// The parameters the kind reads.
    fn params(self) -> &'static [&'static str] {
        match self {
            Kind::WebMercator | Kind::Geographic => &[],
            Kind::Equirectangular | Kind::EqualEarth | Kind::NaturalEarth => &["lon_0"],
            Kind::TransverseMercator => &["lon_0", "lat_0", "k_0", "x_0", "y_0"],
            Kind::Utm => &["zone", "south"],
            Kind::PolarStereographic => &["lat_0", "lat_ts", "lon_0", "k_0", "x_0", "y_0"],
            Kind::Lcc | Kind::Albers => &["lat_1", "lat_2", "lat_0", "lon_0", "x_0", "y_0"],
            Kind::Laea => &["lat_0", "lon_0", "x_0", "y_0"],
            Kind::ObliqueStereographic | Kind::SwissObliqueMercator => {
                &["lat_0", "lon_0", "k_0", "x_0", "y_0"]
            }
        }
    }
}

/// Parameters of a parametric projection, named as PROJ names them:
/// degrees for angles, world units for false eastings and northings.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Params {
    pub(crate) lon_0: Option<f64>,
    pub(crate) lat_0: Option<f64>,
    pub(crate) lat_1: Option<f64>,
    pub(crate) lat_2: Option<f64>,
    pub(crate) lat_ts: Option<f64>,
    pub(crate) k_0: Option<f64>,
    pub(crate) x_0: Option<f64>,
    pub(crate) y_0: Option<f64>,
    pub(crate) zone: Option<f64>,
    pub(crate) south: Option<f64>,
}

impl Params {
    fn slot(&mut self, name: &str) -> Option<&mut Option<f64>> {
        Some(match name {
            "lon_0" => &mut self.lon_0,
            "lat_0" => &mut self.lat_0,
            "lat_1" => &mut self.lat_1,
            "lat_2" => &mut self.lat_2,
            "lat_ts" => &mut self.lat_ts,
            "k_0" => &mut self.k_0,
            "x_0" => &mut self.x_0,
            "y_0" => &mut self.y_0,
            "zone" => &mut self.zone,
            "south" => &mut self.south,
            _ => return None,
        })
    }

    fn get(&self, name: &str) -> Option<f64> {
        let mut copy = *self;
        copy.slot(name).and_then(|v| *v)
    }
}

/// A projection descriptor: a `type` naming a projection kind or an EPSG
/// code (`"EPSG:2193"`), with the kind's parameters by their PROJ names.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Descriptor {
    pub kind: String,
    pub params: Vec<(String, f64)>,
}

impl Descriptor {
    /// A descriptor without parameters.
    pub fn named(kind: &str) -> Self {
        Self {
            kind: kind.to_string(),
            params: Vec::new(),
        }
    }
}

/// A projection descriptor that names no known projection, or gives it
/// parameters it can't use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownProjection(pub String);

impl fmt::Display for UnknownProjection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnknownProjection {}

impl Projection {
    /// The projection named by a descriptor `type` alone.
    pub fn from_descriptor(kind: &str) -> Result<Self, UnknownProjection> {
        Self::new(&Descriptor::named(kind))
    }

    /// The projection a descriptor names.
    pub fn new(descriptor: &Descriptor) -> Result<Self, UnknownProjection> {
        let kind = descriptor.kind.as_str();
        if let Some(code) = kind
            .strip_prefix("EPSG:")
            .or_else(|| kind.strip_prefix("urn:ogc:def:crs:EPSG::"))
        {
            if !descriptor.params.is_empty() {
                return Err(UnknownProjection(format!("`{kind}` takes no parameters")));
            }
            let unknown = || UnknownProjection(format!("unknown projection `{kind}`"));
            let code: u32 = code.parse().map_err(|_| unknown())?;
            let entry = crate::epsg::lookup(code).ok_or_else(unknown)?;
            let ellipsoid = entry.datum.map_or(Ellipsoid::WGS84, |d| d.ellipsoid);
            let family = family(entry.kind, &entry.params, ellipsoid)?;
            return Ok(Self::with_plane(
                Plane::new(family, entry.datum),
                Name::Epsg(code),
                entry.northing_first,
            ));
        }
        let parsed = Kind::parse(kind)
            .ok_or_else(|| UnknownProjection(format!("unknown projection `{kind}`")))?;
        let mut params = Params::default();
        for (name, value) in &descriptor.params {
            if !parsed.params().contains(&name.as_str()) || !value.is_finite() {
                return Err(UnknownProjection(format!(
                    "`{kind}` has no parameter `{name}`"
                )));
            }
            *params.slot(name).expect("listed parameter") = Some(*value);
        }
        // Descriptor kinds list easting first, as `OGC:CRS84` does.
        let family = family(parsed, &params, Ellipsoid::WGS84)?;
        Ok(Self::with_plane(
            Plane::new(family, None),
            Name::Kind(parsed, params),
            false,
        ))
    }

    /// The projection of a coordinate reference system URI, as tile matrix
    /// sets name it: `EPSG:n`, an OGC URN or an opengis.net URL.
    pub(crate) fn from_crs_uri(uri: &str) -> Result<Self, UnknownProjection> {
        let uri = uri.trim();
        let tail = uri
            .strip_prefix("http://www.opengis.net/def/crs/")
            .or_else(|| uri.strip_prefix("https://www.opengis.net/def/crs/"));
        let kind = match tail {
            Some(rest) => match rest.split('/').collect::<Vec<_>>()[..] {
                ["EPSG", _, code] => format!("EPSG:{code}"),
                ["OGC", _, "CRS84"] => "OGC:CRS84".to_string(),
                _ => return Err(UnknownProjection(format!("unknown CRS `{uri}`"))),
            },
            None => match uri {
                "urn:ogc:def:crs:OGC:1.3:CRS84" | "urn:ogc:def:crs:OGC::CRS84" => {
                    "OGC:CRS84".to_string()
                }
                _ => uri.to_string(),
            },
        };
        Self::from_descriptor(&kind)
    }

    /// Metres per world unit at the equator: 1 for projected planes, a
    /// degree's arc for geographic ones (OGC's convention for tile matrix
    /// scale denominators).
    pub fn metres_per_unit(&self) -> f64 {
        match self.plane.family {
            Family::Geographic => 2.0 * std::f64::consts::PI * wgs84::SEMI_MAJOR_M / 360.0,
            _ => 1.0,
        }
    }

    fn with_plane(plane: Plane, name: Name, northing_first: bool) -> Self {
        Self {
            plane,
            name,
            extent: default_extent(&plane),
            northing_first,
        }
    }

    /// The descriptor that names this projection.
    pub fn descriptor(&self) -> Descriptor {
        match self.name {
            Name::Epsg(code) => Descriptor::named(&format!("EPSG:{code}")),
            Name::Kind(kind, params) => Descriptor {
                kind: kind.name().to_string(),
                params: kind
                    .params()
                    .iter()
                    .filter_map(|n| params.get(n).map(|v| (n.to_string(), v)))
                    .collect(),
            },
        }
    }

    /// Whether two projections map positions identically, however they
    /// are named and whatever their extents.
    pub fn same_plane(&self, other: &Projection) -> bool {
        self.plane == other.plane
    }

    /// The same projection with another world extent, such as the area of
    /// a tile matrix set.
    pub fn with_extent(mut self, extent: WorldRect) -> Self {
        self.extent = extent;
        self
    }

    /// Whether the projection's coordinate reference system lists northing
    /// before easting, as tile matrix set origins written in CRS axis order
    /// do.
    pub fn northing_first(&self) -> bool {
        self.northing_first
    }

    /// Projects a geodetic position into world space.
    pub fn forward(&self, g: Geodetic) -> DVec2 {
        self.plane.forward(g.lon.to_radians(), g.lat.to_radians())
    }

    /// The geodetic position of a world-space point.
    pub fn inverse(&self, p: DVec2) -> Geodetic {
        let (lon, lat) = self.plane.inverse(p);
        Geodetic::new(lon.to_degrees(), lat.to_degrees())
    }

    /// Whether `forward` maps a position to a finite, meaningful point.
    pub fn contains(&self, g: Geodetic) -> bool {
        self.plane.contains(g.lon.to_radians(), g.lat.to_radians())
    }

    /// World units per ground metre at `g`, along the parallel. Ground
    /// metres are on the sphere of the WGS84 semi-major axis, the sphere
    /// style zoom is defined on, so that zoom, resolution and scale agree
    /// in every projection.
    pub fn scale_factor(&self, g: Geodetic) -> f64 {
        match self.plane.family {
            Family::WebMercator => {
                1.0 / g
                    .lat
                    .to_radians()
                    .clamp(-MERCATOR_MAX_LAT, MERCATOR_MAX_LAT)
                    .cos()
            }
            Family::Geographic => {
                180.0
                    / std::f64::consts::PI
                    / (wgs84::SEMI_MAJOR_M * g.lat.to_radians().cos().max(1e-9))
            }
            _ => {
                let plane = &self.plane;
                let (lon, lat) = (g.lon.to_radians(), g.lat.to_radians());
                let h = 1e-7;
                let ground = wgs84::SEMI_MAJOR_M * lat.cos();
                if ground > 1.0 {
                    let d = plane.forward(lon + h, lat) - plane.forward(lon - h, lat);
                    d.length() / (2.0 * h * ground)
                } else {
                    // At a pole, the meridian scale: the same for the
                    // conformal families centred there.
                    let lat = lat.clamp(-FRAC_PI_2_LESS, FRAC_PI_2_LESS);
                    let d = plane.forward(lon, lat + h) - plane.forward(lon, lat - h);
                    d.length() / (2.0 * h * wgs84::SEMI_MAJOR_M)
                }
            }
        }
    }

    /// The position this plane's style zoom is measured at,
    /// or `None` for Web Mercator, whose style zoom follows the target's
    /// latitude as GL JS's does.
    pub fn zoom_reference(&self) -> Option<Geodetic> {
        self.plane
            .zoom_reference()
            .map(|(lon, lat)| Geodetic::new(lon.to_degrees(), lat.to_degrees()))
    }

    /// The style zoom of a view of `world_per_px` on this plane: the zoom
    /// at which Web Mercator's ground resolution at the plane's reference
    /// latitude is the view's there. The same everywhere on the plane, so
    /// panning keeps it. `None` for Web Mercator (see
    /// [`zoom_reference`](Self::zoom_reference)).
    pub fn zoom_at_scale(&self, world_per_px: f64) -> Option<f64> {
        let reference = self.zoom_reference()?;
        // Ground metres per pixel at the reference, from the plane's scale
        // there (world units per ground metre).
        let ground_per_px = world_per_px / self.scale_factor(reference);
        Some(
            (2.0 * std::f64::consts::PI
                * wgs84::SEMI_MAJOR_M
                * crate::style_parallel(reference.lat)
                / (STYLE_TILE_PX * ground_per_px))
                .log2(),
        )
    }

    /// World units per pixel at a style zoom: [`zoom_at_scale`](Self::zoom_at_scale)'s
    /// inverse.
    pub fn scale_at_zoom(&self, zoom: f64) -> Option<f64> {
        let reference = self.zoom_reference()?;
        Some(
            2.0 * std::f64::consts::PI * wgs84::SEMI_MAJOR_M * crate::style_parallel(reference.lat)
                / (STYLE_TILE_PX * zoom.exp2())
                * self.scale_factor(reference),
        )
    }

    /// World units per metre of v8 feature height at `g`. The style's height
    /// convention uses the IUGG mean sphere, while ground metres use the
    /// semi-major axis.
    pub fn style_height_scale(&self, g: Geodetic) -> f64 {
        self.scale_factor(g) * wgs84::SEMI_MAJOR_M / wgs84::MEAN_RADIUS_M
    }

    /// The projected extent of the world.
    pub fn world_bounds(&self) -> WorldRect {
        self.extent
    }

    /// The x period after which the world repeats, if it wraps.
    pub fn wrap_period(&self) -> Option<f64> {
        self.plane.wrap_period()
    }

    /// `world` in the copy of a wrapping world nearest `near`; unchanged on
    /// a plane that doesn't wrap.
    pub fn nearest_copy(&self, world: DVec2, near: DVec2) -> DVec2 {
        match self.wrap_period() {
            Some(period) => DVec2::new(
                world.x - ((world.x - near.x) / period).round() * period,
                world.y,
            ),
            None => world,
        }
    }

    /// The central meridian of a plane cut along its antimeridian (Equal
    /// Earth's, a conic's), in degrees: longitudes on the cut lie on either
    /// edge of the world.
    pub(crate) fn seam(&self) -> Option<f64> {
        self.plane.seam().map(f64::to_degrees)
    }
}

/// Logical pixels per tile side at its own style zoom.
const STYLE_TILE_PX: f64 = 512.0;

const FRAC_PI_2_LESS: f64 = std::f64::consts::FRAC_PI_2 - 1e-6;

/// The family a kind and its parameters name, on `ellipsoid` where the
/// kind is ellipsoidal and not tied to WGS 84.
fn family(kind: Kind, p: &Params, ellipsoid: Ellipsoid) -> Result<Family, UnknownProjection> {
    let deg = |v: Option<f64>| v.unwrap_or(0.0).to_radians();
    let (x_0, y_0) = (p.x_0.unwrap_or(0.0), p.y_0.unwrap_or(0.0));
    let bad = |message: &str| Err(UnknownProjection(message.to_string()));
    Ok(match kind {
        Kind::WebMercator => Family::WebMercator,
        Kind::Geographic => Family::Geographic,
        Kind::Equirectangular => Family::Equirectangular { lon0: deg(p.lon_0) },
        Kind::TransverseMercator => Family::TransverseMercator(TransverseMercator::new(
            ellipsoid,
            deg(p.lon_0),
            deg(p.lat_0),
            p.k_0.unwrap_or(1.0),
            x_0,
            y_0,
        )),
        Kind::Utm => {
            let zone = p.zone.unwrap_or(0.0);
            if !(1.0..=60.0).contains(&zone) || zone.fract() != 0.0 {
                return bad("`utm` needs a `zone` from 1 to 60");
            }
            let south = p.south.unwrap_or(0.0) != 0.0;
            Family::TransverseMercator(TransverseMercator::new(
                ellipsoid,
                (-183.0 + 6.0 * zone).to_radians(),
                0.0,
                0.9996,
                500_000.0,
                if south { 10_000_000.0 } else { 0.0 },
            ))
        }
        Kind::PolarStereographic => match (p.lat_ts, p.lat_0) {
            (Some(lat_ts), _) if lat_ts != 0.0 && lat_ts.abs() <= 90.0 => {
                Family::PolarStereographic(PolarStereographic::with_standard_parallel(
                    lat_ts.to_radians(),
                    deg(p.lon_0),
                    x_0,
                    y_0,
                ))
            }
            (None, Some(lat_0)) if lat_0.abs() == 90.0 => {
                Family::PolarStereographic(PolarStereographic::with_scale(
                    lat_0 > 0.0,
                    p.k_0.unwrap_or(1.0),
                    deg(p.lon_0),
                    x_0,
                    y_0,
                ))
            }
            _ => return bad("`polar-stereographic` needs `lat_ts`, or `lat_0` of ±90"),
        },
        Kind::Lcc | Kind::Albers => {
            let Some(lat_1) = p.lat_1 else {
                return bad("a conic needs `lat_1`");
            };
            let lat_2 = p.lat_2.unwrap_or(lat_1);
            if lat_1 + lat_2 == 0.0 || lat_1.abs() >= 90.0 || lat_2.abs() >= 90.0 {
                return bad("a conic's standard parallels can't be opposite or polar");
            }
            let make = if kind == Kind::Lcc {
                Conic::lcc
            } else {
                Conic::albers
            };
            Family::Conic(make(
                lat_1.to_radians(),
                lat_2.to_radians(),
                deg(p.lat_0),
                deg(p.lon_0),
                x_0,
                y_0,
            ))
        }
        Kind::Laea => Family::Laea(Laea::new(deg(p.lat_0), deg(p.lon_0), x_0, y_0)),
        Kind::EqualEarth => Family::Pseudo(Pseudo::equal_earth(deg(p.lon_0))),
        Kind::NaturalEarth => Family::Pseudo(Pseudo::natural_earth(deg(p.lon_0))),
        Kind::ObliqueStereographic => Family::ObliqueStereographic(ObliqueStereographic::new(
            ellipsoid,
            deg(p.lon_0),
            deg(p.lat_0),
            p.k_0.unwrap_or(1.0),
            x_0,
            y_0,
        )),
        Kind::SwissObliqueMercator => Family::SwissObliqueMercator(SwissObliqueMercator::new(
            ellipsoid,
            deg(p.lon_0),
            deg(p.lat_0),
            p.k_0.unwrap_or(1.0),
            x_0,
            y_0,
        )),
    })
}

/// The bounding box of the positions a family shows with moderate
/// distortion; Web Mercator's is its square world.
fn default_extent(plane: &Plane) -> WorldRect {
    if let Family::WebMercator = plane.family {
        let half = std::f64::consts::PI * wgs84::SEMI_MAJOR_M;
        return WorldRect {
            min: DVec2::splat(-half),
            max: DVec2::splat(half),
        };
    }
    let degrees = |n: u32, from: f64| (from + 5.0 * f64::from(n)).to_radians();
    let grid =
        (0..=72).flat_map(|i| (0..=36).map(move |j| (degrees(i, -180.0), degrees(j, -90.0))));
    WorldRect::around(
        grid.filter(|&(lon, lat)| plane.contains(lon, lat) && plane.shows(lon, lat))
            .map(|(lon, lat)| plane.forward(lon, lat))
            .filter(|p| p.is_finite()),
    )
}

/// Web Mercator's northing, written out for its tests.
#[cfg(test)]
fn mercator_y(lat: f64) -> f64 {
    wgs84::SEMI_MAJOR_M
        * (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0)
            .tan()
            .ln()
}

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
            assert_eq!(p.forward(g).y, mercator_y(g.lat));
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
        assert!(Projection::from_descriptor("EPSG:31467").is_err());
        assert!(Projection::from_descriptor("EPSG:1").is_err());
        assert!(Projection::from_descriptor("orthographic-ish").is_err());
        let bad = Descriptor {
            kind: "lcc".into(),
            params: vec![("lat_9".into(), 1.0)],
        };
        assert!(Projection::new(&bad).is_err());
        assert!(Projection::from_descriptor("lcc").is_err());
        assert!(Projection::from_descriptor("utm").is_err());
    }

    #[test]
    fn descriptors_round_trip() {
        for d in [
            Descriptor::named("EPSG:2193"),
            Descriptor::named("EPSG:3413"),
            Descriptor::named("web-mercator"),
            Descriptor::named("equal-earth"),
            Descriptor {
                kind: "lcc".into(),
                params: vec![
                    ("lat_1".into(), 33.0),
                    ("lat_2".into(), 45.0),
                    ("lon_0".into(), -96.0),
                ],
            },
            Descriptor {
                kind: "utm".into(),
                params: vec![("zone".into(), 33.0), ("south".into(), 1.0)],
            },
        ] {
            let p = Projection::new(&d).unwrap();
            assert_eq!(p.descriptor(), d);
            assert_eq!(Projection::new(&p.descriptor()).unwrap(), p);
        }
        let mercator = Projection::from_descriptor("mercator").unwrap();
        assert!(mercator.same_plane(&Projection::from_descriptor("EPSG:3857").unwrap()));
        assert_eq!(mercator.descriptor(), Descriptor::named("web-mercator"));
    }

    #[test]
    fn national_grids_shift_datums() {
        let close = |code: &str, lon: f64, lat: f64, x: f64, y: f64, metres: f64| {
            let p = Projection::from_descriptor(code).unwrap();
            let g = Geodetic::new(lon, lat);
            let w = p.forward(g);
            assert!(
                (w - DVec2::new(x, y)).length() < metres,
                "{code} {g:?}: {w} not within {metres} m of ({x}, {y})"
            );
            let back = p.inverse(w);
            assert!(
                (back.lon - lon).abs() < 1e-11 && (back.lat - lat).abs() < 1e-11,
                "{code}: {back:?}"
            );
        };
        // As PROJ 9.8 transforms them with the same EPSG Helmert
        // transformations, from ETRS89 (WGS 84 for Britain).
        for (code, lon, lat, x, y) in [
            ("EPSG:28992", 6.0, 53.0, 196_139.438_6, 557_179.094_0),
            ("EPSG:28992", 3.6, 51.45, 30_775.060_4, 386_068.515_9),
            ("EPSG:28992", 7.0, 52.2, 265_253.172_4, 469_213.651_7),
            ("EPSG:27700", -0.1276, 51.5072, 530_043.195_0, 180_358.208_6),
            ("EPSG:27700", -3.19, 55.95, 325_784.748_0, 673_635.768_1),
            ("EPSG:27700", -1.6, 60.15, 422_310.805_3, 1_140_678.863_7),
            ("EPSG:2056", 6.14, 46.2, 2_499_760.951_7, 1_117_336.097_8),
            ("EPSG:2056", 10.4, 46.6, 2_826_856.477_1, 1_165_252.546_2),
        ] {
            close(code, lon, lat, x, y, 0.005);
        }
        // Published reference points. The RD grid's origin at Amersfoort's
        // tower in ETRS89, as Rijkswaterstaat's approximate formulas place
        // it; swisstopo's worked example for its approximate formulas
        // (46°2′38.87″N 8°43′49.79″E); both formulas are good to a metre.
        close(
            "EPSG:28992",
            5.387_206_21,
            52.155_174_40,
            155_000.0,
            463_000.0,
            0.5,
        );
        close(
            "EPSG:2056",
            8.730_497_22,
            46.044_130_56,
            2_700_000.0,
            1_100_000.0,
            1.0,
        );
        close(
            "EPSG:21781",
            8.730_497_22,
            46.044_130_56,
            700_000.0,
            100_000.0,
            1.0,
        );
        // OSTN15, the British grid shift, as PROJ applies it: the Helmert
        // transformation is within a few metres of it.
        close(
            "EPSG:27700",
            -0.1276,
            51.5072,
            530_041.453_9,
            180_358.008_5,
            2.0,
        );
        close("EPSG:27700", -6.3, 49.92, 91_461.709_1, 11_068.126_9, 5.0);
        // The grid's scale at its origin is its k0, times the ratio of the
        // prime vertical radius to the semi-major axis (as for transverse
        // Mercator below), up to the ellipsoids' difference.
        let rd = Projection::from_descriptor("EPSG:28992").unwrap();
        let origin = rd.zoom_reference().unwrap();
        let n = 1.0 / (1.0 - wgs84::E2 * origin.lat.to_radians().sin().powi(2)).sqrt();
        let k = rd.scale_factor(origin);
        assert!((k - 0.999_907_9 * n).abs() < 2e-4, "{k}");
        assert!(!rd.same_plane(&Projection::from_descriptor("EPSG:2056").unwrap()));
    }

    #[test]
    fn scale_factors_match_known_values() {
        // Transverse Mercator's scale on its central meridian is k0, times
        // the ratio of the prime vertical radius to the semi-major axis.
        let nztm = Projection::from_descriptor("EPSG:2193").unwrap();
        let k = nztm.scale_factor(Geodetic::new(173.0, -41.0));
        let n = 1.0 / (1.0 - wgs84::E2 * 41f64.to_radians().sin().powi(2)).sqrt();
        assert!((k - 0.9996 * n).abs() < 1e-7, "{k}");
        // Polar stereographic variant B has unit scale at its standard
        // parallel (on the ellipsoid, so up to that same ratio).
        let polar = Projection::from_descriptor("EPSG:3413").unwrap();
        let k = polar.scale_factor(Geodetic::new(10.0, 70.0));
        let n = 1.0 / (1.0 - wgs84::E2 * 70f64.to_radians().sin().powi(2)).sqrt();
        assert!((k - n).abs() < 1e-7, "{k}");
        assert!(polar.scale_factor(Geodetic::new(0.0, 90.0)).is_finite());
    }
}
