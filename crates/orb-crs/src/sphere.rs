//! The globe's sphere (plan §9.1): tiles of any tile matrix set wrapped
//! around a sphere of the WGS84 semi-major axis, as v8 `globe` draws them.
//! Positions are Earth-centred, Earth-fixed metres: x towards (0°, 0°),
//! y towards (90°E, 0°), z towards the North Pole.

use glam::{DMat3, DVec2, DVec3};
use orb_geo::{Geodetic, wgs84};

use crate::{Series, TileIndex, TileMatrixSet};

/// The sphere's radius: Web Mercator's, so a Mercator world unit at the
/// equator is a metre on the globe.
pub const RADIUS: f64 = wgs84::SEMI_MAJOR_M;

/// A geodetic position on the sphere, `height` metres above it.
pub fn ecef(g: Geodetic, height: f64) -> DVec3 {
    up(g) * (RADIUS + height)
}

/// The geodetic position over a point (longitude and latitude only).
pub fn geodetic(p: DVec3) -> Geodetic {
    Geodetic::new(
        p.y.atan2(p.x).to_degrees(),
        p.z.atan2(p.truncate().length()).to_degrees(),
    )
}

/// The unit vector up from the sphere at `g`.
pub fn up(g: Geodetic) -> DVec3 {
    let (lon, lat) = (g.lon.to_radians(), g.lat.to_radians());
    DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin())
}

/// The local east, north, up frame at `g`, as matrix columns: ECEF
/// directions from local ones.
pub fn frame(g: Geodetic) -> DMat3 {
    let (lon, lat) = (g.lon.to_radians(), g.lat.to_radians());
    let east = DVec3::new(-lon.sin(), lon.cos(), 0.0);
    let north = DVec3::new(-lat.sin() * lon.cos(), -lat.sin() * lon.sin(), lat.cos());
    DMat3::from_cols(east, north, up(g))
}

/// Which poles a tile leaves bare on the sphere, north then south: those
/// past its edge when it is in its set's top or bottom row and the set's
/// world stops short of the pole (Web Mercator's at ±85.05°). Caps reach
/// out to them.
pub fn poles(tms: &TileMatrixSet, tile: TileIndex) -> [bool; 2] {
    let rows = tms.matrix_size(tile.z).1;
    let b = tms.tile_bounds(tile);
    let projection = tms.projection();
    let short = |y: f64| {
        let lat = projection.inverse(DVec2::new(b.center().x, y)).lat;
        lat.is_finite() && lat.abs() < 90.0 - 1e-9
    };
    [
        tile.row == 0 && short(b.max.y),
        tile.row + 1 == rows && short(b.min.y),
    ]
}

/// A tile on the sphere as quadratic patches through exact positions, to
/// within `tolerance` of the tile's side at its centre (the globe's
/// `GpuPerVertex` and `Geodetic` plans), or `None` where its projection
/// doesn't reach or it needs more than 256 patches.
pub fn series(tms: &TileMatrixSet, tile: TileIndex, tolerance: f64) -> Option<Series<DVec3>> {
    let b = tms.tile_bounds(tile);
    let projection = tms.projection();
    let map = |uv: DVec2| {
        let world = DVec2::new(b.min.x + uv.x * b.size().x, b.max.y - uv.y * b.size().y);
        let g = projection.inverse(world);
        (g.lon.is_finite() && g.lat.is_finite()).then(|| ecef(g, 0.0))
    };
    // The tile's side at its centre: the square root of its area scale.
    let h = 1.0 / 64.0;
    let c = DVec2::splat(0.5);
    let dx = map(c + DVec2::new(h, 0.0))? - map(c - DVec2::new(h, 0.0))?;
    let dy = map(c + DVec2::new(0.0, h))? - map(c - DVec2::new(0.0, h))?;
    let span = (dx.cross(dy).length() / (4.0 * h * h)).sqrt();
    Series::fit(map, tolerance * span, span, MAX_CELLS)
}

/// The most patches per side of a tile on the sphere: a whole-world tile
/// needs 32 at an eighth of a pixel.
const MAX_CELLS: u32 = 32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_round_trip() {
        for (lon, lat) in [(0.0, 0.0), (90.0, 0.0), (-170.5, 33.0), (12.0, -89.0)] {
            let g = Geodetic::new(lon, lat);
            let p = ecef(g, 0.0);
            assert!((p.length() - RADIUS).abs() < 1e-6);
            let back = geodetic(p);
            assert!((back.lon - lon).abs() < 1e-9 && (back.lat - lat).abs() < 1e-9);
            let f = frame(g);
            assert!((f.x_axis.cross(f.y_axis) - f.z_axis).length() < 1e-12);
        }
        assert!(ecef(Geodetic::new(90.0, 0.0), 0.0).distance(DVec3::new(0.0, RADIUS, 0.0)) < 1e-6);
    }

    #[test]
    fn mercator_tiles_wrap_the_sphere_within_the_tolerance() {
        let tms = TileMatrixSet::by_id("WebMercatorQuad").unwrap();
        for tile in [
            TileIndex::new(0, 0, 0),
            TileIndex::new(1, 1, 0),
            TileIndex::new(3, 2, 0),
            TileIndex::new(6, 40, 20),
            TileIndex::new(14, 9000, 6000),
        ] {
            let series = series(&tms, tile, 1.0 / 2048.0).unwrap();
            let limit = series.scale / 2048.0;
            let b = tms.tile_bounds(tile);
            for k in 0..=24 {
                for l in 0..=24 {
                    let uv = DVec2::new(f64::from(k), f64::from(l)) / 24.0;
                    let world =
                        DVec2::new(b.min.x + uv.x * b.size().x, b.max.y - uv.y * b.size().y);
                    let exact = ecef(tms.projection().inverse(world), 0.0);
                    let error = series.position(uv).distance(exact);
                    assert!(error <= limit * 1.5, "{tile:?} {uv}: {error} m of {limit}");
                }
            }
            assert!(series.max_segment > 0.0);
        }
        // The whole world in one tile still fits.
        let world = series(&tms, TileIndex::new(0, 0, 0), 1.0 / 2048.0).unwrap();
        assert!(
            world.cells <= 16 && world.grid() > 1,
            "{} cells",
            world.cells
        );
    }

    #[test]
    fn geodetic_tiles_reach_the_poles() {
        let tms = TileMatrixSet::by_id("WorldCRS84Quad").unwrap();
        let s = series(&tms, TileIndex::new(0, 0, 0), 1.0 / 2048.0).unwrap();
        let pole = s.position(DVec2::new(0.3, 0.0));
        assert!(pole.distance(DVec3::new(0.0, 0.0, RADIUS)) < s.scale / 1000.0);
    }
}
