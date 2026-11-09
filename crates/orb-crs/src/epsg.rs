//! The EPSG codes Orbis knows, as families and parameters taken from their
//! PROJ 9 definitions. Most listed datums (WGS 84, NZGD2000, ETRS89, NAD83,
//! GDA94, RGF93, SWEREF99, EUREF-FIN) are within a metre of WGS 84 on
//! GRS 1980 or WGS 84, so no datum shift applies. The Dutch, British and
//! Swiss grids are on Bessel 1841 and Airy 1830 datums, reached by EPSG's
//! Helmert transformation from ETRS89 (taken as WGS 84): within 0.25 m of
//! the national grid shifts (RDNAPTRANS, CHENyx06) in the Netherlands and
//! Switzerland, a few metres of OSTN15 in Britain. Every code not listed
//! is unknown: an error, never a fallback.

use crate::family::{Datum, Ellipsoid};
use crate::projection::{Kind, Params};

/// A known code: its family, parameters, datum if not WGS 84, and whether
/// its first axis is northing (which tile matrix set origins follow).
pub(crate) struct Entry {
    pub(crate) kind: Kind,
    pub(crate) params: Params,
    pub(crate) datum: Option<Datum>,
    pub(crate) northing_first: bool,
}

pub(crate) fn lookup(code: u32) -> Option<Entry> {
    let p = Params::default();
    let entry = |kind, params, northing_first| {
        Some(Entry {
            kind,
            params,
            datum: None,
            northing_first,
        })
    };
    let on = |datum, kind, params| {
        Some(Entry {
            kind,
            params,
            datum: Some(datum),
            northing_first: false,
        })
    };
    let swiss = |x_0, y_0| Params {
        lat_0: Some(46.952_405_555_555_6),
        lon_0: Some(7.439_583_333_333_33),
        k_0: Some(1.0),
        x_0: Some(x_0),
        y_0: Some(y_0),
        ..p
    };
    // CH1903 and CH1903+ to ETRS89 (EPSG:1646, EPSG:1647).
    let ch1903 = Datum::position_vector(
        Ellipsoid::BESSEL_1841,
        [674.374, 15.056, 405.346],
        [0.0; 3],
        0.0,
    );
    let utm = |zone: u32, south: bool, northing_first: bool| {
        entry(
            Kind::TransverseMercator,
            Params {
                lon_0: Some(-183.0 + 6.0 * f64::from(zone)),
                k_0: Some(0.9996),
                x_0: Some(500_000.0),
                y_0: Some(if south { 10_000_000.0 } else { 0.0 }),
                ..p
            },
            northing_first,
        )
    };
    let stere = |lat_ts: f64, lon_0: f64| {
        entry(
            Kind::PolarStereographic,
            Params {
                lat_ts: Some(lat_ts),
                lon_0: Some(lon_0),
                ..p
            },
            false,
        )
    };
    let ups = |north: bool, northing_first: bool| {
        entry(
            Kind::PolarStereographic,
            Params {
                lat_0: Some(if north { 90.0 } else { -90.0 }),
                k_0: Some(0.994),
                x_0: Some(2_000_000.0),
                y_0: Some(2_000_000.0),
                ..p
            },
            northing_first,
        )
    };
    let conic = |kind, lat_1: f64, lat_2: f64, lat_0: f64, lon_0: f64, x_0: f64, y_0: f64, nf| {
        entry(
            kind,
            Params {
                lat_1: Some(lat_1),
                lat_2: Some(lat_2),
                lat_0: Some(lat_0),
                lon_0: Some(lon_0),
                x_0: Some(x_0),
                y_0: Some(y_0),
                ..p
            },
            nf,
        )
    };
    let laea = |lat_0: f64, lon_0: f64, x_0: f64, y_0: f64, nf| {
        entry(
            Kind::Laea,
            Params {
                lat_0: Some(lat_0),
                lon_0: Some(lon_0),
                x_0: Some(x_0),
                y_0: Some(y_0),
                ..p
            },
            nf,
        )
    };
    match code {
        3857 | 900913 => entry(Kind::WebMercator, p, false),
        4326 => entry(Kind::Geographic, p, true),
        // NZGD2000 / New Zealand Transverse Mercator 2000.
        2193 => entry(
            Kind::TransverseMercator,
            Params {
                lon_0: Some(173.0),
                k_0: Some(0.9996),
                x_0: Some(1_600_000.0),
                y_0: Some(10_000_000.0),
                ..p
            },
            true,
        ),
        32601..=32660 => utm(code - 32600, false, false),
        32701..=32760 => utm(code - 32700, true, false),
        // ETRS89 / UTM zones 28N–38N.
        25828..=25838 => utm(code - 25800, false, false),
        // NAD83 / UTM zones 1N–23N.
        26901..=26923 => utm(code - 26900, false, false),
        // SWEREF99 TM, EUREF-FIN / TM35FIN(E,N).
        3006 => utm(33, false, true),
        3067 => utm(35, false, false),
        // NSIDC sea ice, Antarctic and Arctic polar stereographic.
        3413 => stere(70.0, -45.0),
        3976 => stere(-70.0, 0.0),
        3031 => stere(-71.0, 0.0),
        3995 => stere(71.0, 0.0),
        // UPS North and South, (E,N) and (N,E).
        5041 => ups(true, false),
        5042 => ups(false, false),
        32661 => ups(true, true),
        32761 => ups(false, true),
        2154 => conic(
            Kind::Lcc,
            49.0,
            44.0,
            46.5,
            3.0,
            700_000.0,
            6_600_000.0,
            false,
        ),
        3034 => conic(
            Kind::Lcc,
            35.0,
            65.0,
            52.0,
            10.0,
            4_000_000.0,
            2_800_000.0,
            true,
        ),
        3577 => conic(Kind::Albers, -18.0, -36.0, 0.0, 132.0, 0.0, 0.0, false),
        5070 => conic(Kind::Albers, 29.5, 45.5, 23.0, -96.0, 0.0, 0.0, false),
        3338 => conic(Kind::Albers, 55.0, 65.0, 50.0, -154.0, 0.0, 0.0, false),
        3035 => laea(52.0, 10.0, 4_321_000.0, 3_210_000.0, true),
        // NSIDC EASE-Grid 2.0 North and South; North Pole LAEA Canada.
        6931 => laea(90.0, 0.0, 0.0, 0.0, false),
        6932 => laea(-90.0, 0.0, 0.0, 0.0, false),
        3573 => laea(90.0, -100.0, 0.0, 0.0, false),
        8857 => entry(Kind::EqualEarth, p, false),
        // Amersfoort / RD New, by Amersfoort to ETRS89 (8) (EPSG:9281).
        28992 => on(
            Datum::coordinate_frame(
                Ellipsoid::BESSEL_1841,
                [565.7381, 50.4018, 465.2904],
                [
                    0.395_025_981_036_064,
                    -0.330_772_431_242_031,
                    1.876_073_294_628_21,
                ],
                4.072_44,
            ),
            Kind::ObliqueStereographic,
            Params {
                lat_0: Some(52.156_160_555_555_6),
                lon_0: Some(5.387_638_888_888_89),
                k_0: Some(0.999_907_9),
                x_0: Some(155_000.0),
                y_0: Some(463_000.0),
                ..p
            },
        ),
        // OSGB36 / British National Grid, by OSGB36 to WGS 84 (6)
        // (EPSG:1314), the Ordnance Survey's ETRS89 transformation.
        27700 => on(
            Datum::position_vector(
                Ellipsoid::AIRY_1830,
                [446.448, -125.157, 542.06],
                [0.15, 0.247, 0.842],
                -20.489,
            ),
            Kind::TransverseMercator,
            Params {
                lat_0: Some(49.0),
                lon_0: Some(-2.0),
                k_0: Some(0.999_601_271_7),
                x_0: Some(400_000.0),
                y_0: Some(-100_000.0),
                ..p
            },
        ),
        // CH1903+ / LV95 and CH1903 / LV03.
        2056 => on(
            ch1903,
            Kind::SwissObliqueMercator,
            swiss(2_600_000.0, 1_200_000.0),
        ),
        21781 => on(
            ch1903,
            Kind::SwissObliqueMercator,
            swiss(600_000.0, 200_000.0),
        ),
        _ => None,
    }
}
