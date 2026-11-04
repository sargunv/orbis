//! The fill family: polygons triangulated with earcut, and their outlines
//! as line segments for antialiasing (GL JS's `fill-antialias`).
//!
//! Vertex layout, 2 words: `x | y << 16` as i16 tile units, then the feature
//! slot.

use orb_tile::{GeometryType, Layer};

use crate::clip::clip_ring;
use crate::{Mesh, MeshBuilder};

/// Words per fill vertex.
pub const STRIDE: usize = 2;

/// Lays out the polygons among `features` (indices into `layer`); other
/// geometry types are skipped. Each feature's slot is its position in
/// `features`.
pub fn layout(layer: &Layer, features: &[u32]) -> Mesh {
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
                clip_ring(source, extent, &mut ring);
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
                clip_ring(source, extent, &mut ring);
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
