//! The EPSG codes Orbis knows, as families and parameters taken from their
//! PROJ 9 definitions. Every listed datum (WGS 84, NZGD2000, ETRS89, NAD83,
//! GDA94, RGF93, SWEREF99, EUREF-FIN) is within a metre of WGS 84 and uses
//! GRS 1980 or WGS 84, so no datum shift applies. Codes on other datums
//! (Bessel, Airy, Clarke) need a transformation and stay unknown, as does
//! every code not listed: an error, never a fallback.

use crate::projection::{Kind, Params};

/// A known code: its family, parameters and whether its first axis is
/// northing (which tile matrix set origins follow).
pub(crate) struct Entry {
    pub(crate) kind: Kind,
    pub(crate) params: Params,
    pub(crate) northing_first: bool,
}

pub(crate) fn lookup(code: u32) -> Option<Entry> {
    let p = Params::default();
    let entry = |kind, params, northing_first| {
        Some(Entry {
            kind,
            params,
            northing_first,
        })
    };
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
        _ => None,
    }
}
