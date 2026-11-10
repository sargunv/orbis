//! Rounded extrusion corners (`fill-extrusion-rounded-corner-distance`),
//! following the pinned GL JS `round_polygon_corners.ts`: each corner an
//! arc of steps of 30° or less, snapped to the integer tile grid, and walls
//! whose normals blend across an arc so it shades as one curved surface.

use orb_tile::{GeometryType, Layer, LayerBuilder};

/// A point in tile units.
type Point = [f64; 2];

/// `features` of `layer` scaled to a grid of `extent` units a side, their
/// polygons' corners rounded by `distance` grid units (none at 0): a layer
/// whose features are their slots. Extrusions lay out from it
/// (`fill-extrusion-rounded-corner-distance`), and so do fills with
/// style-spec#1047's `dev.sargunv.orbis:fill-rounded-corner-distance`, on
/// the same arcs.
pub fn rounded_layer(layer: &Layer, features: &[u32], extent: u32, distance: f64) -> Layer {
    let mut out = LayerBuilder::new(layer.name(), extent);
    let rings = || {
        (features.iter())
            .flat_map(|&f| layer.geometry(f as usize).parts())
            .flat_map(|part| part.rings())
    };
    out.reserve(rings().map(<[_]>::len).sum());
    let scale = |p: &[i32; 2]| {
        p.map(|v| {
            (f64::from(v) * f64::from(extent) / f64::from(layer.extent()) + 0.5).floor() as i32
        })
    };
    let mut scaled = Vec::new();
    for &feature in features {
        let feature = feature as usize;
        out.begin_feature(None, layer.geometry_type(feature));
        let polygon = layer.geometry_type(feature) == GeometryType::Polygon;
        for part in layer.geometry(feature).parts() {
            out.begin_part();
            for ring in part.rings() {
                out.begin_ring();
                if distance > 0.0 && polygon {
                    scaled.clear();
                    scaled.extend(ring.iter().map(scale));
                    for p in round_ring(&scaled, distance, extent as i32) {
                        out.push_vertex(p);
                    }
                } else {
                    for p in ring {
                        out.push_vertex(scale(p));
                    }
                }
            }
        }
    }
    out.finish()
}

/// GL JS's `isBoundaryEdge`: an edge along the tile's buffer, which
/// clipping made, outside the tile square.
pub(crate) fn is_boundary_edge(a: [i32; 2], b: [i32; 2], extent: i32) -> bool {
    (0..2).any(|axis| a[axis] == b[axis] && (a[axis] < 0 || a[axis] > extent))
}

/// A ring with its corners rounded by `distance` tile units, open (no
/// repeated last vertex), or the ring itself if it would collapse. Corners
/// that tile clipping made stay sharp: the neighbouring tile cuts the same
/// feature elsewhere.
pub(crate) fn round_ring(ring: &[[i32; 2]], distance: f64, extent: i32) -> Vec<[i32; 2]> {
    let mut n = ring.len();
    if n > 1 && ring[0] == ring[n - 1] {
        n -= 1;
    }
    if n < 3 {
        return ring.to_vec();
    }
    let at = |i: usize| ring[i % n].map(f64::from);
    let mut rounded: Vec<Point> = Vec::new();
    let mut previous_boundary = is_boundary_edge(ring[n - 1], ring[0], extent);
    for i in 0..n {
        let next_boundary = is_boundary_edge(ring[i], ring[(i + 1) % n], extent);
        if previous_boundary || next_boundary {
            rounded.push(at(i));
        } else {
            round_corner(&mut rounded, at(i + n - 1), at(i), at(i + 1), distance);
        }
        previous_boundary = next_boundary;
    }
    // Snap to the grid, dropping points that land on their neighbour.
    let mut snapped: Vec<[i32; 2]> = Vec::with_capacity(rounded.len());
    for p in rounded {
        let p = p.map(|v| v.round() as i32);
        if snapped.last() != Some(&p) {
            snapped.push(p);
        }
    }
    while snapped.len() > 1 && snapped[0] == snapped[snapped.len() - 1] {
        snapped.pop();
    }
    if snapped.len() < 3 {
        return ring[..n].to_vec();
    }
    snapped
}

/// Appends the arc that replaces corner `c` between `a` and `b`, or the
/// corner itself where it is too shallow or too sharp to round.
fn round_corner(out: &mut Vec<Point>, a: Point, c: Point, b: Point, distance: f64) {
    let (mut ua, mut ub) = ([a[0] - c[0], a[1] - c[1]], [b[0] - c[0], b[1] - c[1]]);
    let (la, lb) = (ua[0].hypot(ua[1]), ub[0].hypot(ub[1]));
    if la < 1e-6 || lb < 1e-6 {
        out.push(c);
        return;
    }
    ua = ua.map(|v| v / la);
    ub = ub.map(|v| v / lb);
    let dot = ua[0] * ub[0] + ua[1] * ub[1];
    if dot.abs() > (5f64).to_radians().cos() {
        out.push(c);
        return;
    }
    // At most a fifth of either edge, so arcs don't meet.
    let r = distance.min(la * 0.2).min(lb * 0.2);
    let tangent_a = [c[0] + ua[0] * r, c[1] + ua[1] * r];
    let tangent_b = [c[0] + ub[0] * r, c[1] + ub[1] * r];
    let cos_half = ((1.0 + dot) / 2.0).sqrt();
    let bisector = [ua[0] + ub[0], ua[1] + ub[1]];
    let len = bisector[0].hypot(bisector[1]);
    let center = [
        c[0] + bisector[0] / len * r / cos_half,
        c[1] + bisector[1] / len * r / cos_half,
    ];
    let (va, vb) = (
        [tangent_a[0] - center[0], tangent_a[1] - center[1]],
        [tangent_b[0] - center[0], tangent_b[1] - center[1]],
    );
    let sweep = (va[0] * vb[1] - va[1] * vb[0]).atan2(va[0] * vb[0] + va[1] * vb[1]);
    let steps = ((sweep.abs() / (std::f64::consts::PI / 6.0) - 1e-6).ceil() as usize).max(2);
    for s in 0..=steps {
        let (sin, cos) = (sweep * s as f64 / steps as f64).sin_cos();
        out.push([
            center[0] + cos * va[0] - sin * va[1],
            center[1] + sin * va[0] + cos * va[1],
        ]);
    }
}

/// The normals at a wall's two ends, (start, end), unit length.
pub(crate) type WallNormals = ([f64; 2], [f64; 2]);

/// GL JS's `roundedWallNormals` for a closed ring given open: the normals
/// of the wall from vertex `i` to `i + 1`, or `None` for boundary and
/// zero-length walls. Where walls meet at a shallow turn they share one
/// normal (the much longer wall's, or their length-weighted average), so a
/// rounded corner shades as one curved surface.
pub(crate) fn wall_normals(ring: &[[i32; 2]], extent: i32) -> Vec<Option<WallNormals>> {
    let n = ring.len();
    let perps: Vec<Option<[f64; 2]>> = (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            let edge = [f64::from(b[0] - a[0]), f64::from(b[1] - a[1])];
            (!is_boundary_edge(a, b, extent) && edge != [0.0; 2]).then_some([-edge[1], edge[0]])
        })
        .collect();
    (0..n)
        .map(|i| {
            let perp = perps[i]?;
            Some((
                smooth_normal(perp, perps[(i + n - 1) % n]),
                smooth_normal(perp, perps[(i + 1) % n]),
            ))
        })
        .collect()
}

fn smooth_normal(perp: [f64; 2], neighbour: Option<[f64; 2]>) -> [f64; 2] {
    let unit = |v: [f64; 2]| {
        let len = v[0].hypot(v[1]);
        [v[0] / len, v[1] / len]
    };
    let Some(neighbour) = neighbour else {
        return unit(perp);
    };
    let (length, neighbour_length) = (perp[0].hypot(perp[1]), neighbour[0].hypot(neighbour[1]));
    let sharp = perp[0] * neighbour[0] + perp[1] * neighbour[1]
        < (35f64).to_radians().cos() * length * neighbour_length;
    if sharp || length >= 2.0 * neighbour_length {
        return unit(perp);
    }
    if neighbour_length >= 2.0 * length {
        return unit(neighbour);
    }
    unit([perp[0] + neighbour[0], perp[1] + neighbour[1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_gets_four_arcs_on_the_grid() {
        let square = [[1000, 1000], [3000, 1000], [3000, 3000], [1000, 3000]];
        let rounded = round_ring(&square, 200.0, 8192);
        // Each right angle becomes three 30° steps: four points per corner.
        assert_eq!(rounded.len(), 16);
        // Every point stays within the square, and the corners are gone.
        assert!(
            rounded
                .iter()
                .all(|p| (1000..=3000).contains(&p[0]) && (1000..=3000).contains(&p[1]))
        );
        assert!(!rounded.contains(&[1000, 1000]));
        // The arc's tangent points are 200 units along each edge.
        assert!(rounded.contains(&[1200, 1000]) && rounded.contains(&[1000, 1200]));
    }

    #[test]
    fn clipped_corners_stay_sharp() {
        // The left edge lies on the buffer, outside the tile.
        let clipped = [[-128, 1000], [3000, 1000], [3000, 3000], [-128, 3000]];
        let rounded = round_ring(&clipped, 200.0, 8192);
        assert!(rounded.contains(&[-128, 1000]) && rounded.contains(&[-128, 3000]));
        assert!(!rounded.contains(&[3000, 1000]));
    }

    #[test]
    fn arcs_shade_as_one_surface() {
        let rounded = round_ring(&[[0, 0], [4000, 0], [4000, 4000], [0, 4000]], 400.0, 8192);
        let normals = wall_normals(&rounded, 8192);
        // Along an arc, a wall's end normal is its neighbour's start normal.
        for i in 0..rounded.len() {
            let (Some((_, end)), Some((start, _))) = (normals[i], normals[(i + 1) % rounded.len()])
            else {
                continue;
            };
            let short = |j: usize| {
                let (a, b) = (rounded[j], rounded[(j + 1) % rounded.len()]);
                f64::from(b[0] - a[0]).hypot(f64::from(b[1] - a[1])) < 1000.0
            };
            if short(i) && short((i + 1) % rounded.len()) {
                assert!((end[0] - start[0]).abs() < 1e-9 && (end[1] - start[1]).abs() < 1e-9);
            }
        }
    }
}
