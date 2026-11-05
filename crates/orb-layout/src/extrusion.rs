//! Flat extrusions, following GL JS's pinned `fill_extrusion_bucket.ts`.
//! Four words: i16 position, feature slot, i16 normal.xy (scale 16384,
//! bottom/top in x's low bit), then i16 normal.z and u16 edge distance.
//! Roofs reuse fill triangulation. Walls clip original segments independently,
//! so clipping never introduces a wall across the tile square.
use crate::round_corners::{round_ring, wall_normals};
use crate::{Mesh, MeshBuilder, clip::clip_segment, fill};
use orb_tile::{GeometryType, Layer, LayerBuilder};

pub const STRIDE: usize = 4;

pub fn layout(layer: &Layer, features: &[u32]) -> Mesh {
    layout_in_extent(layer, features, layer.extent(), 0.0, None)
}

/// Layout in a normalized tile grid. The pinned bucket uses 8192 units,
/// including for edge-distance resets and integer vertex snapping.
/// `rounding` rounds polygon corners by that many grid units
/// (`fill-extrusion-rounded-corner-distance`), 0 for none. Roofs are cut
/// along `bend`'s grid for a curved surface, as fills are (walls follow
/// the source geometry, which the jobs cut along it too).
pub fn layout_in_extent(
    layer: &Layer,
    features: &[u32],
    output_extent: u32,
    rounding: f64,
    bend: Option<fill::Bend>,
) -> Mesh {
    // Roofs don't reach the poles: caps are the ground's.
    let bend = bend.map(|b| fill::Bend {
        north: false,
        south: false,
        ..b
    });
    let rounded = rounding > 0.0;
    if output_extent == layer.extent() && !rounded {
        return layout_normalized(layer, features, false, bend);
    }
    let extent = output_extent as i32;
    let mut normalized = LayerBuilder::new("extrusion", output_extent);
    for &feature in features {
        let feature = feature as usize;
        normalized.begin_feature(None, layer.geometry_type(feature));
        for part in layer.geometry(feature).parts() {
            normalized.begin_part();
            for ring in part.rings() {
                normalized.begin_ring();
                let scaled: Vec<[i32; 2]> = ring
                    .iter()
                    .map(|p| {
                        p.map(|v| {
                            (f64::from(v) * f64::from(output_extent) / f64::from(layer.extent())
                                + 0.5)
                                .floor() as i32
                        })
                    })
                    .collect();
                let polygon = layer.geometry_type(feature) == GeometryType::Polygon;
                let ring = if rounded && polygon {
                    round_ring(&scaled, rounding, extent)
                } else {
                    scaled
                };
                for p in ring {
                    normalized.push_vertex(p);
                }
            }
        }
    }
    layout_normalized(
        &normalized.finish(),
        &(0..features.len() as u32).collect::<Vec<_>>(),
        rounded,
        bend,
    )
}

/// A feature's centroid in a tile grid of `output_extent` units: the mean
/// of its rings' vertices, each closed ring's repeated last vertex left
/// out, floored, as GL JS's `accumulatePointsToCentroid` takes it per
/// polygon. Terrain raises extrusions by the ground under it.
pub fn centroid(layer: &Layer, feature: u32, output_extent: u32) -> [i16; 2] {
    let (mut x, mut y, mut n) = (0i64, 0i64, 0i64);
    for part in layer.geometry(feature as usize).parts() {
        for ring in part.rings() {
            let last = ring.len().saturating_sub(1);
            for (i, p) in ring.iter().enumerate() {
                if i == last && i > 0 && ring[0] == *p {
                    continue;
                }
                x += i64::from(p[0]);
                y += i64::from(p[1]);
                n += 1;
            }
        }
    }
    if n == 0 {
        return [0; 2];
    }
    let scale = f64::from(output_extent) / f64::from(layer.extent());
    let mean = |s: i64| ((s as f64 / n as f64) * scale).floor();
    let clamp = |v: f64| v.clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16;
    [clamp(mean(x)), clamp(mean(y))]
}

/// `rounded`: polygon walls take normals that blend across rounded corners.
fn layout_normalized(
    layer: &Layer,
    features: &[u32],
    rounded: bool,
    bend: Option<fill::Bend>,
) -> Mesh {
    let roof = fill::layout_subdivided(layer, features, bend);
    let mut builder = MeshBuilder::new(STRIDE);
    // Preserve roof chunks by feeding their triangles through the builder.
    for v in roof.vertices.as_chunks::<2>().0 {
        let p = [v[0] as i16, (v[0] >> 16) as i16];
        builder.push_vertex(p, &[v[0], v[1], 1, 16384]);
    }
    for t in roof.indices.as_chunks::<3>().0 {
        // Tile Y points south; world Y points north. Reverse winding
        // so outward faces are counterclockwise after projection.
        builder.push_triangle(t[0], t[2], t[1]);
    }
    let extent = layer.extent() as i32;
    for (slot, &feature) in features.iter().enumerate() {
        let feature = feature as usize;
        let closed = layer.geometry_type(feature) == GeometryType::Polygon;
        if !closed && layer.geometry_type(feature) != GeometryType::LineString {
            continue;
        }
        for part in layer.geometry(feature).parts() {
            for ring in part.rings() {
                let mut distance = 0.0f32;
                let n = ring.len();
                if n < 2 {
                    continue;
                }
                let count = if closed { n } else { n - 1 };
                let smooth = (rounded && closed).then(|| wall_normals(ring, extent));
                for i in 0..count {
                    if smooth.as_ref().is_some_and(|s| s[i].is_none()) {
                        continue;
                    }
                    let a = ring[i];
                    let b = ring[(i + 1) % n];
                    // The reference's synthetic clip lines are on the source
                    // buffer, outside the tile square. Real edges on 0/extent
                    // remain walls. Clip original edges, not a clipped polygon.
                    if (0..2).any(|axis| a[axis] == b[axis] && (a[axis] < 0 || a[axis] > extent)) {
                        continue;
                    }
                    let dx = b[0] as f32 - a[0] as f32;
                    let dy = b[1] as f32 - a[1] as f32;
                    let length = dx.hypot(dy);
                    if length == 0.0 {
                        continue;
                    }
                    if distance + length > 32768.0 {
                        distance = 0.0;
                    }
                    if let Some((ca, cb, _, _)) = clip_segment(a, b, 0, extent) {
                        let flat = [f64::from(-dy / length), f64::from(dx / length)];
                        // The wall's normals at its end and its start.
                        let (start_normal, end_normal) =
                            smooth.as_ref().and_then(|s| s[i]).unwrap_or((flat, flat));
                        let start = builder.vertex_count();
                        for (p, normal) in [(cb, end_normal), (ca, start_normal)] {
                            let nx = (normal[0] * 8192.0).floor() as i16 * 2;
                            let ny = (normal[1] * 16384.0) as i16;
                            // GL JS assigns the cumulative distance to the
                            // edge's end, then distance + length to its start.
                            // Retain that phase even when either end is clipped.
                            let edge =
                                distance + ((b[0] - p[0]) as f32).hypot((b[1] - p[1]) as f32);
                            for top in [0u16, 1] {
                                let p = [p[0] as i16, p[1] as i16];
                                builder.push_vertex(
                                    p,
                                    &[
                                        u32::from(p[0] as u16) | u32::from(p[1] as u16) << 16,
                                        slot as u32,
                                        u32::from(nx as u16 | top) | u32::from(ny as u16) << 16,
                                        (edge.round() as u32) << 16,
                                    ],
                                );
                            }
                        }
                        builder.push_triangle(start, start + 2, start + 1);
                        builder.push_triangle(start + 1, start + 2, start + 3);
                    }
                    distance += length;
                }
            }
        }
    }
    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalized_source_extents_have_identical_geometry_and_pattern_phase() {
        let make = |extent: u32| {
            let mut b = LayerBuilder::new("building", extent);
            b.begin_feature(None, GeometryType::Polygon);
            b.begin_part();
            b.begin_ring();
            for p in [[0, 0], [1, 0], [1, 1], [0, 1]] {
                b.push_vertex(p.map(|v| v * extent as i32 / 2));
            }
            layout_in_extent(&b.finish(), &[0], 8192, 0.0, None)
        };
        let a = make(4096);
        let b = make(8192);
        assert_eq!(a.vertices, b.vertices);
        assert_eq!(a.indices, b.indices);
    }

    #[test]
    fn clipped_roof_has_no_synthetic_walls() {
        let mut b = LayerBuilder::new("building", 100);
        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        for p in [[-10, 20], [50, 20], [50, 80], [-10, 80]] {
            b.push_vertex(p);
        }
        let mesh = layout(&b.finish(), &[0]);
        // Two roof triangles, three walls; no wall on x = 0.
        assert_eq!(mesh.indices.len(), 6 + 3 * 6);
        for v in mesh.vertices.as_chunks::<4>().0 {
            assert!((0..=100).contains(&(v[0] as i16)));
        }
    }
    #[test]
    fn original_edge_on_tile_square_is_a_real_wall() {
        let mut b = LayerBuilder::new("building", 100);
        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        for p in [[0, 20], [50, 20], [50, 80], [0, 80]] {
            b.push_vertex(p);
        }
        let mesh = layout(&b.finish(), &[0]);
        assert_eq!(mesh.indices.len(), 6 + 4 * 6);
    }
    #[test]
    fn surrounding_polygon_has_only_a_roof() {
        let mut b = LayerBuilder::new("building", 100);
        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        for p in [[-10, -10], [110, -10], [110, 110], [-10, 110]] {
            b.push_vertex(p);
        }
        let mesh = layout(&b.finish(), &[0]);
        assert_eq!(mesh.indices.len(), 6);
        assert!(
            mesh.vertices
                .as_chunks::<4>()
                .0
                .iter()
                .all(|v| v[3] == 16384)
        );
    }
}
