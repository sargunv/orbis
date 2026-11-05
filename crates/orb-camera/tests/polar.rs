//! Polar views, part of M6's CRS suite: cameras on and around the pole of a
//! polar plane see ground that round-trips through the projection, and the
//! cover leaves no visible point without a tile.

use glam::DVec2;
use orb_camera::{CameraState, View, Viewport};
use orb_crs::{Projection, TileMatrixSet};
use orb_geo::Geodetic;

fn check(tms: &TileMatrixSet, target: Geodetic) {
    let projection: &Projection = tms.projection();
    let viewport = Viewport {
        width: 1024.0,
        height: 768.0,
    };
    for zoom in [1.0, 3.5, 8.0, 14.0] {
        for bearing in [0.0, 73.0, -150.0] {
            for pitch in [0.0, 45.0, 70.0] {
                let mut camera = CameraState::new(target, zoom, projection);
                camera.bearing = bearing;
                camera.pitch = pitch;
                let view = View::new(&camera, viewport, projection);
                let cover = view.cover(tms, 512.0, 16);
                assert!(!cover.is_empty());
                let case = format!("{} {target:?} z{zoom} b{bearing} p{pitch}", tms.id());
                let top = view.ground_horizon() + 1.0;
                let world_rect = tms.tile_bounds(tms.roots().next().unwrap());
                let mut checked = 0;
                for i in 0..=16 {
                    for j in 0..=16 {
                        let px = DVec2::new(
                            viewport.width * f64::from(i) / 16.0,
                            top + (viewport.height - top) * f64::from(j) / 16.0,
                        );
                        let world = view.unproject(px).unwrap();
                        // Off the set's square is past its data.
                        if !(world_rect.min.cmple(world).all() && world.cmple(world_rect.max).all())
                        {
                            continue;
                        }
                        // The ground round-trips through the projection...
                        let ground = projection.inverse(world);
                        let back = projection.forward(ground);
                        assert!(
                            back.distance(world) < 1e-6 * view.world_per_px().max(1e-3),
                            "{case}: {px} round-trips {} px off",
                            back.distance(world) / view.world_per_px()
                        );
                        // ...and some covered tile holds it.
                        let held = cover.iter().any(|t| {
                            let b = tms.tile_bounds(t.index);
                            b.min.cmple(world).all() && world.cmple(b.max).all()
                        });
                        assert!(held, "{case}: no tile under {px}");
                        checked += 1;
                    }
                }
                assert!(checked > 100, "{case}: {checked} points on the set");
                // Looking down at the pole, its tiles are in the cover.
                if pitch == 0.0 && target.lat.abs() == 90.0 {
                    let pole = projection.forward(target);
                    assert!(cover.iter().any(|t| {
                        let b = tms.tile_bounds(t.index);
                        b.min.cmple(pole).all() && pole.cmple(b.max).all()
                    }));
                }
            }
        }
    }
}

#[test]
fn views_over_the_poles_are_covered() {
    let arctic = TileMatrixSet::by_id("UPSArcticWGS84Quad").unwrap();
    let antarctic = TileMatrixSet::by_id("UPSAntarcticWGS84Quad").unwrap();
    for lon in [0.0, -45.0, 180.0] {
        check(&arctic, Geodetic::new(lon, 90.0));
        check(&antarctic, Geodetic::new(lon, -90.0));
    }
    check(&arctic, Geodetic::new(-45.0, 87.5));
    check(&antarctic, Geodetic::new(100.0, -80.0));
    // NSIDC sea ice: polar stereographic true at 70°N, on a square set.
    let nsidc = Projection::from_descriptor("EPSG:3413").unwrap();
    let rect = nsidc.world_bounds();
    let nsidc = TileMatrixSet::square(nsidc, rect);
    check(&nsidc, Geodetic::new(0.0, 90.0));
    check(&nsidc, Geodetic::new(-45.0, 75.0));
}
