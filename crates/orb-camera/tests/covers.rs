use std::f64::consts::SQRT_2;

use glam::DVec2;
use orb_camera::{CameraState, Cover, View, Viewport};
use orb_crs::Geodetic;
use orb_crs::Surface;
use orb_crs::TileIndex;
use orb_crs::TileMatrixSet;

/// Up to 60° of pitch, where neither GL JS nor Native's level of detail
/// varies the zoom, covers are GL JS's tile for tile: they cull alike.
/// Past it Orbis takes Native's level of detail, which keeps detail
/// farther out than GL JS's, and Native's far plane, which reaches past
/// GL JS's shortened horizon: its cover must hold every point of the view
/// GL JS's holds, in no more than four times GL JS's tiles (or 128).
#[test]
fn covers_match_pinned_gl_js() {
    let golden: serde_json::Value = serde_json::from_str(include_str!("data/covers.json")).unwrap();
    let tms = TileMatrixSet::web_mercator();
    let period = tms.tile_span(0);
    let mut failures = Vec::new();
    for case in golden["cases"].as_array().unwrap() {
        let n = |key: &str| case[key].as_f64().unwrap();
        let mut camera = CameraState::new(
            Geodetic::new(n("lon"), n("lat")),
            n("zoom"),
            tms.projection(),
        );
        camera.pitch = n("pitch");
        camera.bearing = n("bearing");
        camera.fov = n("fov");
        let view = View::new(
            &camera,
            Viewport {
                width: n("width"),
                height: n("height"),
            },
            &Surface::from(tms.projection()),
        );
        let mut actual: Vec<_> = view
            .cover(
                tms,
                n("tileSize"),
                n("maxzoom") as u8,
                Cover {
                    height: n("maxContentElevation")
                        * tms.projection().style_height_scale(camera.target),
                    ..Cover::default()
                },
            )
            .iter()
            .filter(|t| t.index.z >= n("minzoom") as u8)
            .map(|t| {
                [
                    i64::from(t.wrap),
                    i64::from(t.index.z),
                    i64::from(t.index.col),
                    i64::from(t.index.row),
                ]
            })
            .collect();
        actual.sort_unstable();
        let expected: Vec<[i64; 4]> = serde_json::from_value(case["tiles"].clone()).unwrap();
        if n("pitch") > 60.0 {
            if actual.len() > 4 * expected.len().max(32) {
                failures.push(format!(
                    "{}: {} tiles to {}",
                    case["name"],
                    actual.len(),
                    expected.len()
                ));
            }
            let (width, height) = (n("width"), n("height"));
            for i in 0..=32 {
                for j in 0..=32 {
                    let px = DVec2::new(width * f64::from(i), height * f64::from(j)) / 32.0;
                    let Some(world) = view.unproject(px) else {
                        continue;
                    };
                    // Ground past the far plane isn't drawn.
                    if view
                        .project_depth(world, orb_crs::Metres::ZERO)
                        .is_none_or(|(_, depth)| depth > 1.0)
                    {
                        continue;
                    }
                    let held = |cover: &[[i64; 4]]| {
                        cover.iter().any(|&[wrap, z, col, row]| {
                            let index = TileIndex::new(z as u8, col as u32, row as u32);
                            let offset = DVec2::new(wrap as f64 * period, 0.0);
                            let b = tms.tile_bounds(index).translated(offset);
                            b.min.cmple(world).all() && world.cmple(b.max).all()
                        })
                    };
                    if held(&expected) && !held(&actual) {
                        failures.push(format!("{}: {px} uncovered", case["name"]));
                    }
                }
            }
            continue;
        }
        if actual != expected {
            let extra: Vec<_> = actual.iter().filter(|t| !expected.contains(t)).collect();
            let missing: Vec<_> = expected.iter().filter(|t| !actual.contains(t)).collect();
            failures.push(format!(
                "{}: extra {extra:?}, missing {missing:?}",
                case["name"]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} differing covers:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The globe's covers against GL JS's for its own globe cases and a sweep
/// of cameras. Its covers differ in two ways, so they are compared by what
/// they show rather than tile by tile:
///
/// - Culling: each culls with its own conservative volumes, so each keeps
///   a few tiles the other drops that are out of view.
/// - Detail: GL JS coarsens tiles by distance at any pitch; Orbis takes
///   Native's level of detail, as on the plane, with no coarsening up to
///   60° of pitch. Orbis also stops at tiles that show small on screen,
///   toward a pole or the limb, where GL JS keeps the view's zoom.
///
/// So for every visible point, Orbis's cover must hold it (up to 60° of
/// pitch, at most a level coarser than GL JS's tile, which may be finer
/// than the target's near the eye, unless Orbis's tile there shows under
/// √2 times the centre tile's width), and no case may take many
/// more tiles than GL JS's. Covers match exactly where neither coarsens;
/// the count of those is a floor, raised as covers come closer.
#[test]
fn globe_covers_match_pinned_gl_js() {
    let golden: serde_json::Value = serde_json::from_str(include_str!("data/covers.json")).unwrap();
    let tms = TileMatrixSet::web_mercator();
    let surface = Surface::globe(*tms.projection(), orb_crs::Globe::V8);
    let edge = tms.projection().world_bounds().max.y;
    let period = tms.tile_span(0);
    let cases = golden["globe"].as_array().unwrap();
    let (mut exact, mut failures) = (0, Vec::new());
    for case in cases {
        let n = |key: &str| case[key].as_f64().unwrap();
        let name = case["name"].as_str().unwrap();
        let mut camera = CameraState::new(
            Geodetic::new(n("lon"), n("lat")),
            n("zoom"),
            tms.projection(),
        );
        camera.pitch = n("pitch");
        camera.bearing = n("bearing");
        camera.fov = n("fov");
        let (width, height) = (n("width"), n("height"));
        let view = View::new(&camera, Viewport { width, height }, &surface);
        // Rounded covers (GL JS's raster sources) are floored covers of
        // tiles √2 smaller.
        let round = case["roundZoom"].as_bool().unwrap_or(false);
        let tile_px = n("tileSize") / if round { std::f64::consts::SQRT_2 } else { 1.0 };
        // The globe has one world: GL JS's wraps only name the copy
        // nearest its centre.
        let mut actual: Vec<TileIndex> = view
            .cover(tms, tile_px, n("maxzoom") as u8, Cover::default())
            .iter()
            .map(|t| t.index)
            .filter(|t| t.z >= n("minzoom") as u8)
            .collect();
        actual.sort_by_key(|t| (t.z, t.col, t.row));
        let mut expected: Vec<TileIndex> = case["tiles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                let [_, z, col, row]: [i64; 4] = serde_json::from_value(t.clone()).unwrap();
                TileIndex::new(z as u8, col as u32, row as u32)
            })
            .collect();
        expected.sort_by_key(|t| (t.z, t.col, t.row));
        if actual == expected {
            exact += 1;
            continue;
        }
        if actual.len() as f64 > 2.5 * expected.len() as f64 + 2.0 {
            failures.push(format!(
                "{name}: {} tiles to {}",
                actual.len(),
                expected.len()
            ));
        }
        let holding = |cover: &[TileIndex], world: DVec2| {
            cover
                .iter()
                .find(|t| {
                    // A metre's slack: a sample on a tile's edge (the view's
                    // diagonal down a meridian) can fall a rounding outside.
                    let b = tms.tile_bounds(**t);
                    (b.min - 1.0).cmple(world).all() && world.cmple(b.max + 1.0).all()
                })
                .map(|t| t.z)
        };
        // How wide a tile of zoom `z` at `world` shows: the square root of
        // the area a small square there covers on screen.
        let shows = |z: u8, world: DVec2| {
            let step = tms.tile_span(z) / 1024.0;
            let at = |dx: f64, dy: f64| view.project(world + DVec2::new(dx, dy));
            let (o, e, n) = (at(0.0, 0.0)?, at(step, 0.0)?, at(0.0, step)?);
            Some((e - o).perp_dot(n - o).abs().sqrt() * 1024.0)
        };
        let wrap =
            |world: DVec2| DVec2::new(world.x - (world.x / period).round() * period, world.y);
        let center = wrap(view.unproject(DVec2::new(width, height) / 2.0).unwrap());
        let center_tile = shows(holding(&actual, center).unwrap(), center).unwrap();
        for i in 0..=32 {
            for j in 0..=32 {
                let px = DVec2::new(width * f64::from(i), height * f64::from(j)) / 32.0;
                let Some(world) = view.unproject(px) else {
                    continue;
                };
                // Past ±85.05°, the edge tiles' caps show.
                if world.y.abs() >= edge {
                    continue;
                }
                let world = wrap(world);
                let ours = holding(&actual, world);
                let theirs = holding(&expected, world);
                // Tiles that show under √2 times the centre tile's width may
                // be coarser still.
                let coarser = |z: u8| {
                    let ours = ours.unwrap();
                    let small = shows(ours, world).is_some_and(|w| w < SQRT_2 * center_tile);
                    n("pitch") <= 60.0 && z > ours + 1 && !small
                };
                if ours.is_none() || theirs.is_some_and(coarser) {
                    failures.push(format!("{name}: {px} in z{ours:?}, GL JS z{theirs:?}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty() && exact >= 27,
        "{exact} of {} match; {} failures:\n{}",
        cases.len(),
        failures.len(),
        failures.join("\n")
    );
}

/// The globe's cover against Web Mercator's for the same cameras. Where the
/// globe fills the view (zoom 5 up) and up to 45° of pitch, it takes about
/// as many tiles: at most 1.3× as many all told and 1.5× in any view. Below
/// that the globe shows more of the world than the plane does at the same
/// zoom (GL JS's globe cases bound it there).
#[test]
fn globe_cover_takes_about_the_planes_tiles() {
    let tms = TileMatrixSet::web_mercator();
    let projection = *tms.projection();
    let viewport = Viewport {
        width: 1024.0,
        height: 768.0,
    };
    let (mut plane_total, mut globe_total) = (0, 0);
    for zoom in [1.0, 3.0, 5.0, 8.0, 11.0] {
        for pitch in [0.0, 45.0, 70.0] {
            let mut camera = CameraState::new(Geodetic::new(139.7, 35.7), zoom, &projection);
            camera.pitch = pitch;
            camera.bearing = 20.0;
            let tiles = |surface: Surface| {
                let view = View::new(&camera, viewport, &surface);
                view.cover(tms, 512.0, 22, Cover::default()).len()
            };
            let plane = tiles(Surface::plane(projection));
            let globe = tiles(Surface::globe(projection, orb_crs::Globe::V8));
            assert!(globe > 0 && globe < 200);
            if zoom >= 5.0 && pitch <= 45.0 {
                assert!(
                    globe as f64 <= 1.5 * plane as f64,
                    "z{zoom} pitch {pitch}: {globe} tiles to {plane}"
                );
                plane_total += plane;
                globe_total += globe;
            }
        }
    }
    assert!(
        globe_total as f64 <= 1.3 * plane_total as f64,
        "{globe_total} tiles to {plane_total}"
    );
}
