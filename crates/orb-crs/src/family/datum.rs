//! Geodetic datums other than WGS 84, reached from it by a seven-parameter
//! Helmert transformation of Earth-centred coordinates (EPSG methods 1033
//! and 1032, as PROJ's `helmert` applies them).

use glam::{DMat3, DVec3};

use super::ellipsoid::Ellipsoid;

/// A datum on its own ellipsoid, and the Helmert transformation from it to
/// WGS 84 (taken as ETRS89, NAD83 and the other datums within a metre of
/// it, as the EPSG table does).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Datum {
    pub(crate) ellipsoid: Ellipsoid,
    /// Translation in metres.
    shift: DVec3,
    /// Rotation and scale from the datum's axes to WGS 84's, linearized
    /// for small angles.
    to_wgs84: DMat3,
    /// Its inverse to the linearization's order, as PROJ inverts it: the
    /// rotation transposed, over the scale.
    from_wgs84: DMat3,
}

impl Datum {
    /// A datum from EPSG's position vector parameters for the datum to
    /// WGS 84: translations in metres, rotations in arc-seconds, scale
    /// difference in parts per million.
    pub(crate) fn position_vector(
        ellipsoid: Ellipsoid,
        shift: [f64; 3],
        rotation: [f64; 3],
        ppm: f64,
    ) -> Self {
        let [rx, ry, rz] = rotation.map(|r| (r / 3600.0).to_radians());
        // Columns of R = [[1, −rz, ry], [rz, 1, −rx], [−ry, rx, 1]].
        let rotation = DMat3::from_cols(
            DVec3::new(1.0, rz, -ry),
            DVec3::new(-rz, 1.0, rx),
            DVec3::new(ry, -rx, 1.0),
        );
        let scale = 1.0 + ppm * 1e-6;
        Self {
            ellipsoid,
            shift: DVec3::from(shift),
            to_wgs84: rotation * scale,
            from_wgs84: rotation.transpose() / scale,
        }
    }

    /// The same transformation from EPSG's coordinate frame parameters,
    /// whose rotations turn the other way.
    pub(crate) fn coordinate_frame(
        ellipsoid: Ellipsoid,
        shift: [f64; 3],
        rotation: [f64; 3],
        ppm: f64,
    ) -> Self {
        Self::position_vector(ellipsoid, shift, rotation.map(|r| -r), ppm)
    }

    /// A WGS 84 position (radians) on this datum. Its height on the
    /// datum's ellipsoid is dropped, as for any map position.
    pub(crate) fn local(&self, lon: f64, lat: f64) -> (f64, f64) {
        let p = geocentric(Ellipsoid::WGS84, lon, lat);
        geodetic(self.ellipsoid, self.from_wgs84 * (p - self.shift))
    }

    /// A position on this datum (radians) in WGS 84: the position on the
    /// WGS 84 surface that [`local`](Self::local) takes there.
    pub(crate) fn wgs84(&self, lon: f64, lat: f64) -> (f64, f64) {
        let p = geocentric(self.ellipsoid, lon, lat);
        let (l, b) = geodetic(Ellipsoid::WGS84, self.to_wgs84 * p + self.shift);
        // The two ways differ by a millimetre (the surfaces lie tens of
        // metres apart, their normals arc-seconds, and the inverse is
        // linearized), which one step corrects.
        let (l1, b1) = self.local(l, b);
        (l + lon - l1, b + lat - b1)
    }
}

/// Earth-centred, Earth-fixed metres of a position on the ellipsoid's
/// surface.
fn geocentric(ellipsoid: Ellipsoid, lon: f64, lat: f64) -> DVec3 {
    let Ellipsoid { a, e2 } = ellipsoid;
    let (s, c) = lat.sin_cos();
    let nu = a / (1.0 - e2 * s * s).sqrt();
    DVec3::new(nu * c * lon.cos(), nu * c * lon.sin(), nu * (1.0 - e2) * s)
}

/// Longitude and latitude of an Earth-centred point, by fixed-point
/// iteration on the latitude (to well under a micrometre within a few
/// kilometres of the surface).
fn geodetic(ellipsoid: Ellipsoid, p: DVec3) -> (f64, f64) {
    let Ellipsoid { a, e2 } = ellipsoid;
    let r = p.x.hypot(p.y);
    let mut lat = p.z.atan2(r * (1.0 - e2));
    for _ in 0..6 {
        let s = lat.sin();
        let nu = a / (1.0 - e2 * s * s).sqrt();
        lat = (p.z + e2 * nu * s).atan2(r);
    }
    (p.y.atan2(p.x), lat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geocentric_round_trips() {
        for (lon, lat) in [(0.0, 0.0), (5.0, 52.0), (-120.0, -45.0), (179.0, 89.9)] {
            let (lon, lat) = (f64::to_radians(lon), f64::to_radians(lat));
            let (l, b) = geodetic(
                Ellipsoid::BESSEL_1841,
                geocentric(Ellipsoid::BESSEL_1841, lon, lat),
            );
            assert!((l - lon).abs() < 1e-14 && (b - lat).abs() < 1e-13);
        }
    }

    #[test]
    fn helmert_round_trips() {
        let datum = Datum::position_vector(
            Ellipsoid::AIRY_1830,
            [446.448, -125.157, 542.06],
            [0.15, 0.247, 0.842],
            -20.489,
        );
        let (lon, lat) = (f64::to_radians(-2.0), f64::to_radians(54.0));
        let (l, b) = datum.local(lon, lat);
        // About a hundred metres apart.
        assert!((l - lon).abs() > 1e-6);
        let (l, b) = datum.wgs84(l, b);
        // Within a millimetre: the dropped height (tens of metres) moves
        // the way back by the arc-seconds between the ellipsoids' normals.
        assert!((l - lon).abs() < 1e-13 && (b - lat).abs() < 1e-13);
    }
}
