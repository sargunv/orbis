//! The fill family: polygons triangulated with earcut, and their outlines
//! as line segments for antialiasing (GL JS's `fill-antialias`).
//!
//! Vertex layout, 2 words: `x | y << 16` as i16 tile units, then the feature
//! slot.

use orb_tile::{GeometryType, Layer, clip_ring};

use crate::{Mesh, MeshBuilder};

/// Words per fill vertex.
pub const STRIDE: usize = 2;

/// The y of a vertex at the North Pole, and of one at the South Pole: on a
/// globe the shader puts them there; on a plane, on the tile's top or
/// bottom edge (`project.wgsl`).
pub const NORTH_POLE_Y: i16 = i16::MIN;
pub const SOUTH_POLE_Y: i16 = i16::MAX;

/// How a tile's fills bend onto a curved surface: cut along a grid of
/// `cells` per side, and in a world's top or bottom row on a globe, their
/// edges on the world's edge fanned out to the pole.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Bend {
    pub cells: u32,
    /// Whether the north and south poles take caps.
    pub poles: [bool; 2],
}

/// Lays out the polygons among `features` (indices into `layer`); other
/// geometry types are skipped. Each feature's slot is its position in
/// `features`.
pub fn layout(layer: &Layer, features: &[u32]) -> Mesh {
    layout_subdivided(layer, features, None)
}

/// [`layout`], with triangles cut along a grid of `cells` × `cells` over the
/// tile, for a tile drawn through a curved warp onto another plane: edges
/// then follow the curve (earcut drops the collinear vertices that cut the
/// rings), and the vertices on the tile's edges are its neighbours'. Edges
/// on a pole's side extend to it.
pub fn layout_subdivided(layer: &Layer, features: &[u32], bend: Option<Bend>) -> Mesh {
    layout_clipped(layer, features, bend, &|_| true)
}

/// [`layout_subdivided`], clipping to the tile square only the features
/// (by slot) `clip` says: the rest keep the source's buffer around it.
pub(crate) fn layout_clipped(
    layer: &Layer,
    features: &[u32],
    bend: Option<Bend>,
    clip: &dyn Fn(u32) -> bool,
) -> Mesh {
    let extent = i32::try_from(layer.extent()).expect("extent fits i32");
    assert!(
        extent <= i32::from(i16::MAX),
        "extent {extent} exceeds i16 positions"
    );

    let mut builder = MeshBuilder::new(STRIDE);
    let mut earcut = earcut::Earcut::new();
    let mut ring = Vec::new();
    let mut flat: Vec<[f64; 2]> = Vec::new();
    let mut points: Vec<[i16; 2]> = Vec::new();
    let mut holes: Vec<u32> = Vec::new();
    let mut triangles: Vec<u32> = Vec::new();

    for (slot, &feature) in features.iter().enumerate() {
        let feature = feature as usize;
        if layer.geometry_type(feature) != GeometryType::Polygon {
            continue;
        }
        let slot = slot as u32;
        for part in layer.geometry(feature).parts() {
            flat.clear();
            points.clear();
            holes.clear();
            for (r, source) in part.rings().enumerate() {
                if clip(slot) {
                    clip_ring(source, 0, extent, &mut ring);
                } else {
                    ring.clear();
                    let closing = source.len() > 1 && source.first() == source.last();
                    ring.extend_from_slice(&source[..source.len() - usize::from(closing)]);
                }
                if ring.is_empty() {
                    if r == 0 {
                        break;
                    }
                    continue;
                }
                if r > 0 {
                    holes.push(points.len() as u32);
                }
                for &[x, y] in &ring {
                    flat.push([f64::from(x), f64::from(y)]);
                    points.push([x as i16, y as i16]);
                }
            }
            if points.is_empty() {
                continue;
            }
            earcut.earcut(flat.iter().copied(), &holes, &mut triangles);
            if let Some(bend) = bend {
                if bend.cells > 1 {
                    subdivide(
                        &mut points,
                        &mut triangles,
                        f64::from(extent) / f64::from(bend.cells),
                    );
                }
                let edge = extent as i16;
                let [north, south] = bend.poles;
                for (on, y, pole) in [(north, 0, NORTH_POLE_Y), (south, edge, SOUTH_POLE_Y)] {
                    if on {
                        cap(&mut points, &mut triangles, y, pole);
                    }
                }
            }
            let base = builder.vertex_count();
            for &[x, y] in &points {
                builder.push_vertex([x, y], &[pack_position(x, y), slot]);
            }
            for t in triangles.as_chunks::<3>().0 {
                builder.push_triangle(base + t[0], base + t[1], base + t[2]);
            }
        }
    }
    builder.finish()
}

/// Lays out the outlines of the polygons among `features` as line
/// segments, with the same vertex layout as [`layout`]. Edges along the
/// tile square that clipping created are left out: they aren't polygon
/// edges, and GL JS's stencil hides them too.
pub fn outline(layer: &Layer, features: &[u32]) -> Mesh {
    let extent = i32::try_from(layer.extent()).expect("extent fits i32");
    assert!(
        extent <= i32::from(i16::MAX),
        "extent {extent} exceeds i16 positions"
    );
    let synthetic = |a: [i32; 2], b: [i32; 2]| {
        (0..2).any(|axis| a[axis] == b[axis] && (a[axis] == 0 || a[axis] == extent))
    };

    let mut builder = MeshBuilder::new(STRIDE);
    let mut ring = Vec::new();
    for (slot, &feature) in features.iter().enumerate() {
        let feature = feature as usize;
        if layer.geometry_type(feature) != GeometryType::Polygon {
            continue;
        }
        let slot = slot as u32;
        for part in layer.geometry(feature).parts() {
            for source in part.rings() {
                clip_ring(source, 0, extent, &mut ring);
                if ring.is_empty() {
                    continue;
                }
                let base = builder.vertex_count();
                for &[x, y] in &ring {
                    let p = [x as i16, y as i16];
                    builder.push_vertex(p, &[pack_position(p[0], p[1]), slot]);
                }
                let n = ring.len();
                for i in 0..n {
                    let j = (i + 1) % n;
                    if !synthetic(ring[i], ring[j]) {
                        builder.push_segment(base + i as u32, base + j as u32);
                    }
                }
            }
        }
    }
    builder.finish()
}

/// Fans the triangles' edges along the tile edge `y` out to a vertex at a
/// pole. Only a polygon's boundary lies on the tile's edge after clipping.
fn cap(points: &mut Vec<[i16; 2]>, triangles: &mut Vec<u32>, y: i16, pole: i16) {
    let mut fan = Vec::new();
    for t in triangles.as_chunks::<3>().0 {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if points[a as usize][1] == y && points[b as usize][1] == y {
                fan.push((a, b));
            }
        }
    }
    let Some(&(first, _)) = fan.first() else {
        return;
    };
    let tip = points.len() as u32;
    points.push([points[first as usize][0], pole]);
    for (a, b) in fan {
        triangles.extend([a, b, tip]);
    }
}

/// Cuts triangles along a grid of lines `step` apart into convex pieces,
/// fanned into triangles. A cut on an edge is computed from the edge's ends
/// in a fixed order, so the triangles (and tiles) sharing an edge cut it
/// alike and the mesh stays watertight.
fn subdivide(points: &mut Vec<[i16; 2]>, triangles: &mut Vec<u32>, step: f64) {
    let mut index: std::collections::HashMap<[i32; 2], u32> = points
        .iter()
        .enumerate()
        .map(|(i, p)| ([i32::from(p[0]), i32::from(p[1])], i as u32))
        .collect();
    let mut out = Vec::with_capacity(triangles.len());
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for t in triangles.as_chunks::<3>().0 {
        let corners = t.map(|i| points[i as usize].map(i32::from));
        let cell = |v: i32| (f64::from(v) / step).floor() as i64;
        let (x0, x1) = (
            corners.iter().map(|p| cell(p[0])).min(),
            corners.iter().map(|p| cell(p[0])).max(),
        );
        let (y0, y1) = (
            corners.iter().map(|p| cell(p[1])).min(),
            corners.iter().map(|p| cell(p[1])).max(),
        );
        let (Some(x0), Some(x1), Some(y0), Some(y1)) = (x0, x1, y0, y1) else {
            continue;
        };
        if x0 == x1 && y0 == y1 {
            out.extend_from_slice(t);
            continue;
        }
        for cy in y0..=y1 {
            for cx in x0..=x1 {
                a.clear();
                a.extend_from_slice(&corners);
                let bounds = [
                    (0, (cx as f64 * step).round() as i32, true),
                    (0, ((cx + 1) as f64 * step).round() as i32, false),
                    (1, (cy as f64 * step).round() as i32, true),
                    (1, ((cy + 1) as f64 * step).round() as i32, false),
                ];
                for (axis, bound, keep_above) in bounds {
                    clip(&a, axis, bound, keep_above, &mut b);
                    std::mem::swap(&mut a, &mut b);
                }
                a.dedup();
                while a.len() > 1 && a.first() == a.last() {
                    a.pop();
                }
                if a.len() < 3 {
                    continue;
                }
                let mut id = |p: [i32; 2]| {
                    *index.entry(p).or_insert_with(|| {
                        points.push([p[0] as i16, p[1] as i16]);
                        (points.len() - 1) as u32
                    })
                };
                let first = id(a[0]);
                for k in 1..a.len() - 1 {
                    let (p, q) = (id(a[k]), id(a[k + 1]));
                    if first != p && p != q && q != first {
                        out.extend([first, p, q]);
                    }
                }
            }
        }
    }
    *triangles = out;
}

/// One Sutherland–Hodgman step of a convex polygon against an axis-aligned
/// line, keeping the side above (`keep_above`) or below it.
fn clip(polygon: &[[i32; 2]], axis: usize, bound: i32, keep_above: bool, out: &mut Vec<[i32; 2]>) {
    out.clear();
    let inside = |p: &[i32; 2]| {
        if keep_above {
            p[axis] >= bound
        } else {
            p[axis] <= bound
        }
    };
    let Some(&last) = polygon.last() else { return };
    let mut prev = last;
    for &p in polygon {
        let cross = || {
            let (lo, hi) = if prev <= p { (prev, p) } else { (p, prev) };
            let t = f64::from(bound - lo[axis]) / f64::from(hi[axis] - lo[axis]);
            let other = 1 - axis;
            let mut q = [0; 2];
            q[axis] = bound;
            q[other] = (f64::from(lo[other]) + t * f64::from(hi[other] - lo[other])).round() as i32;
            q
        };
        match (inside(&prev), inside(&p)) {
            (true, true) => out.push(p),
            (true, false) => out.push(cross()),
            (false, true) => {
                out.push(cross());
                out.push(p);
            }
            (false, false) => {}
        }
        prev = p;
    }
}

pub(crate) fn pack_position(x: i16, y: i16) -> u32 {
    u32::from(x as u16) | u32::from(y as u16) << 16
}

#[cfg(test)]
mod tests {
    use super::*;
    use orb_tile::LayerBuilder;

    fn polygon(b: &mut LayerBuilder, rings: &[&[[i32; 2]]]) {
        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        for ring in rings {
            b.begin_ring();
            for &v in *ring {
                b.push_vertex(v);
            }
        }
    }

    fn area(mesh: &Mesh) -> f64 {
        let pos = |i: u32| {
            let w = mesh.vertices[i as usize * STRIDE];
            [
                f64::from(w as u16 as i16),
                f64::from((w >> 16) as u16 as i16),
            ]
        };
        mesh.indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let [a, b, c] = [pos(t[0]), pos(t[1]), pos(t[2])];
                ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs() / 2.0
            })
            .sum()
    }

    #[test]
    fn triangulates_a_square_with_a_hole() {
        let mut b = LayerBuilder::new("water", 4096);
        polygon(
            &mut b,
            &[
                &[[0, 0], [100, 0], [100, 100], [0, 100]],
                &[[25, 25], [25, 75], [75, 75], [75, 25]],
            ],
        );
        let layer = b.finish();
        let mesh = layout(&layer, &[0]);
        assert_eq!(area(&mesh), 100.0 * 100.0 - 50.0 * 50.0);
        assert_eq!(mesh.chunks.len(), 1);
        assert_eq!(mesh.vertices[1], 0, "feature slot");
    }

    #[test]
    fn outlines_skip_clipped_tile_edges() {
        let mut b = LayerBuilder::new("water", 4096);
        // A square crossing the tile's left edge: three real edges inside,
        // one created by clipping along x = 0.
        polygon(
            &mut b,
            &[&[[-100, 100], [100, 100], [100, 200], [-100, 200]]],
        );
        let layer = b.finish();
        let mesh = outline(&layer, &[0]);
        assert_eq!(mesh.indices.len(), 6, "three segments");
        assert_eq!(mesh.chunks.len(), 1);
    }

    #[test]
    fn clips_to_the_tile_and_skips_unselected() {
        let mut b = LayerBuilder::new("water", 4096);
        polygon(&mut b, &[&[[0, 0], [10, 0], [10, 10]]]);
        polygon(
            &mut b,
            &[&[[-100, -100], [5000, -100], [5000, 5000], [-100, 5000]]],
        );
        let layer = b.finish();
        let mesh = layout(&layer, &[1]);
        assert_eq!(area(&mesh), 4096.0 * 4096.0);
        let bounds = mesh.chunks[0].bounds;
        assert_eq!((bounds.min, bounds.max), ([0, 0], [4096, 4096]));
    }
}

#[cfg(test)]
mod subdivide_tests {
    use super::{NORTH_POLE_Y, SOUTH_POLE_Y, cap, subdivide};

    #[test]
    fn subdivision_keeps_area_and_cuts_at_grid_lines() {
        let mut points = vec![[0i16, 0], [4096, 0], [4096, 4096], [0, 4096]];
        let mut triangles = vec![0, 1, 2, 0, 2, 3];
        subdivide(&mut points, &mut triangles, 512.0);
        let area: f64 = triangles
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| points[i as usize].map(f64::from));
                ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs() / 2.0
            })
            .sum();
        assert_eq!(area, 4096.0 * 4096.0);
        // Every triangle lies in one grid cell.
        for t in triangles.as_chunks::<3>().0 {
            let cells: Vec<_> = t
                .iter()
                .map(|&i| points[i as usize].map(|v| f64::from(v) / 512.0))
                .collect();
            for axis in 0..2 {
                let lo = cells.iter().map(|c| c[axis]).fold(f64::INFINITY, f64::min);
                let hi = cells
                    .iter()
                    .map(|c| c[axis])
                    .fold(f64::NEG_INFINITY, f64::max);
                assert!(hi - lo <= 1.0 && lo.floor() >= hi.ceil() - 1.0);
            }
        }
        // The square's top edge has a vertex at each grid line.
        assert_eq!(points.iter().filter(|p| p[1] == 0).count(), 9);
    }

    #[test]
    fn caps_fan_the_world_edge_to_the_pole() {
        let mut points = vec![[0i16, 0], [4096, 0], [4096, 4096], [0, 4096]];
        let mut triangles = vec![0, 1, 2, 0, 2, 3];
        subdivide(&mut points, &mut triangles, 512.0);
        let before = triangles.len() / 3;
        cap(&mut points, &mut triangles, 0, NORTH_POLE_Y);
        // One triangle per grid cell along the top edge, to one pole vertex.
        assert_eq!(triangles.len() / 3 - before, 8);
        assert_eq!(points.iter().filter(|p| p[1] == NORTH_POLE_Y).count(), 1);
        // Nothing on the bottom edge's side without a south cap.
        let n = triangles.len();
        cap(&mut points, &mut triangles, 1000, SOUTH_POLE_Y);
        assert_eq!(triangles.len(), n);
    }
}
