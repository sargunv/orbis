//! Flat extrusions, following GL JS's pinned `fill_extrusion_bucket.ts`.
//! Four words: i16 position, feature slot, i16 normal.xy (scale 16384,
//! bottom/top in x's low bit, and in y's which end of its wall a wall
//! vertex is), then i16 normal.z and u16 edge distance. A wall is four
//! vertices in a row: its end's bottom and top, then its start's.
//! Roofs reuse fill triangulation. A building whole in the tile's data, one
//! polygon with no edge on the source's buffer clip that the tile owning
//! its centroid keeps whole too ([`Layer::is_whole`]), keeps its roof and
//! walls past the tile square: the placement holding its centroid draws it
//! all (`whole`).
//! A building the source cut at its buffer is clipped to the square, roof
//! and walls; walls clip original segments independently, so clipping never
//! introduces a wall across the tile square.
use crate::round_corners::{rounded_layer, wall_normals};
use crate::{Mesh, MeshBuilder, Owner, clip::clip_segment, fill};
use orb_tile::{GeometryType, Layer};

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
        poles: [false; 2],
        ..b
    });
    let rounded = rounding > 0.0;
    // From the source's rings, as the property rows take them.
    let whole = whole(layer, features, bend.is_some());
    let owners: Vec<Owner> = (features.iter().zip(whole))
        .map(|(&f, whole)| match whole {
            true => Owner::of_whole(centroid(layer, f, output_extent), output_extent),
            false => Owner::Every,
        })
        .collect();
    if output_extent == layer.extent() && !rounded {
        return layout_normalized(layer, features, &owners, false, bend);
    }
    layout_normalized(
        &rounded_layer(layer, features, output_extent, rounding),
        &(0..features.len() as u32).collect::<Vec<_>>(),
        &owners,
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

/// Which of `features` are whole in `layer`'s data ([`Layer::is_whole`]:
/// one polygon each, the tile owning its centroid keeping it whole too)
/// and can be drawn entire by that tile, where its placement maps past
/// the square: none for a tile `bent` onto a curved surface, whose
/// patches end there.
pub fn whole(layer: &Layer, features: &[u32], bent: bool) -> Vec<bool> {
    let buffer = layer.clip_buffer();
    (features.iter())
        .map(|&f| !bent && layer.is_whole(f as usize, buffer))
        .collect()
}

/// `owners`, per slot: whose the building is, a whole one keeping its
/// roof and walls past the tile square. A chunk holds one owner's
/// buildings, so the cull's candidates keep it only where its owner draws
/// it: the square's first, then those every placement cuts, then each
/// neighbour's.
fn layout_normalized(
    layer: &Layer,
    features: &[u32],
    owners: &[Owner],
    rounded: bool,
    bend: Option<fill::Bend>,
) -> Mesh {
    let rank = |slot: u32| match owners[slot as usize] {
        Owner::Square => 0,
        Owner::Every => 1,
        Owner::Neighbour(cell) => 2 + cell,
    };
    let roof = fill::layout_clipped(layer, features, bend, &|slot| rank(slot) == 1);
    let mut builder = MeshBuilder::new(STRIDE);
    for v in roof.vertices.as_chunks::<2>().0 {
        let p = [v[0] as i16, (v[0] >> 16) as i16];
        builder.push_vertex(p, &[v[0], v[1], 1, 16384]);
    }
    // A roof vertex's second word is its slot.
    let slot_of = |v: u32| roof.vertices[v as usize * fill::STRIDE + 1];
    let mut roofs: Vec<[u32; 3]> = roof
        .vertex_indices()
        .collect::<Vec<_>>()
        .as_chunks()
        .0
        .to_vec();
    roofs.sort_by_key(|t| rank(slot_of(t[0])));
    for [a, b, c] in roofs {
        builder.set_owner(owners[slot_of(a) as usize]);
        // Tile Y points south; world Y points north. Reverse winding
        // so outward faces are counterclockwise after projection.
        builder.push_triangle(a, c, b);
    }
    let extent = layer.extent() as i32;
    let mut slots: Vec<u32> = (0..features.len() as u32).collect();
    slots.sort_by_key(|&slot| rank(slot));
    for slot in slots {
        builder.set_owner(owners[slot as usize]);
        let slot = slot as usize;
        let feature = features[slot] as usize;
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
                    // A cut building's walls clip to the square, which drops
                    // the source's clip lines (outside it); real edges on
                    // 0/extent remain walls. Original edges clip, not a
                    // clipped polygon. A whole building has no clip lines.
                    let dx = b[0] as f32 - a[0] as f32;
                    let dy = b[1] as f32 - a[1] as f32;
                    let length = dx.hypot(dy);
                    if length == 0.0 {
                        continue;
                    }
                    if distance + length > 32768.0 {
                        distance = 0.0;
                    }
                    let clipped = if owners[slot] != Owner::Every {
                        Some((a, b))
                    } else {
                        clip_segment(a.map(i64::from), b.map(i64::from), 0, extent)
                            .map(|(ca, cb, _, _)| (ca, cb))
                    };
                    if let Some((ca, cb)) = clipped {
                        let flat = [f64::from(-dy / length), f64::from(dx / length)];
                        // The wall's normals at its end and its start.
                        let (start_normal, end_normal) =
                            smooth.as_ref().and_then(|s| s[i]).unwrap_or((flat, flat));
                        let start = builder.vertex_count();
                        for (p, normal) in [(cb, end_normal), (ca, start_normal)] {
                            let nx = (normal[0] * 8192.0).floor() as i16 * 2;
                            let ny = (normal[1] * 8192.0).floor() as i16 * 2;
                            // GL JS assigns the cumulative distance to the
                            // edge's end, then distance + length to its start.
                            // Retain that phase even when either end is clipped.
                            let [ex, ey] = [0, 1].map(|k| b[k] as f32 - p[k] as f32);
                            let edge = distance + ex.hypot(ey);
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
    use orb_tile::LayerBuilder;
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
    fn a_whole_building_keeps_its_walls_past_the_tile_edge() {
        let mut b = LayerBuilder::new("building", 100);
        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        // Across x = 0, no edge on a buffer line: whole.
        for p in [[-10, 20], [50, 20], [50, 80], [-5, 80]] {
            b.push_vertex(p);
        }
        let layer = b.finish();
        assert!(layer.is_whole(0, layer.clip_buffer()));
        let mesh = layout(&layer, &[0]);
        // Two roof triangles and four whole walls, reaching x = −10.
        assert_eq!(mesh.vertex_indices().count(), 6 + 4 * 6);
        let min_x = mesh
            .vertices
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| v[0] as i16)
            .min();
        assert_eq!(min_x, Some(-10));
    }

    #[test]
    fn chunks_hold_one_owners_buildings() {
        let mut b = LayerBuilder::new("building", 100);
        let square = |b: &mut LayerBuilder, x: i32, y: i32| {
            b.begin_feature(None, GeometryType::Polygon);
            b.begin_part();
            b.begin_ring();
            for p in [[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]] {
                b.push_vertex(p);
            }
        };
        // West of the square, inside it, cut at the buffer, inside again.
        square(&mut b, -8, 40);
        square(&mut b, 40, 40);
        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        for p in [[-10, 0], [5, 0], [5, 10], [-10, 10]] {
            b.push_vertex(p);
        }
        square(&mut b, 60, 60);
        let layer = b.finish();
        let mesh = layout(&layer, &[0, 1, 2, 3]);
        let owners: Vec<Owner> = mesh.chunks.iter().map(|c| c.owner).collect();
        // Roofs, then walls, each the square's, the cut's, the west's.
        let order = [Owner::Square, Owner::Every, Owner::Neighbour(3)];
        assert_eq!(owners, [order, order].concat());
        let slot = |c: &crate::Chunk| {
            let mut v = (c.first_index..c.first_index + c.index_count).map(|i| {
                mesh.vertices[(c.first_vertex + u32::from(crate::mesh::index_at(&mesh.indices, i)))
                    as usize
                    * STRIDE
                    + 1]
            });
            let first = v.next().expect("indices");
            v.all(|s| match c.owner {
                Owner::Square => s == 1 || s == 3,
                _ => s == first,
            })
            .then_some(first)
        };
        let slots: Vec<_> = mesh.chunks.iter().map(slot).collect();
        assert!(slots.iter().all(Option::is_some));
        assert_eq!(slots[1], Some(2));
        assert_eq!(slots[2], Some(0));
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
        let layer = b.finish();
        assert!(
            !layer.is_whole(0, layer.clip_buffer()),
            "cut along the buffer at x = −10"
        );
        let mesh = layout(&layer, &[0]);
        // Two roof triangles, three walls; no wall on x = 0.
        assert_eq!(mesh.vertex_indices().count(), 6 + 3 * 6);
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
        assert_eq!(mesh.vertex_indices().count(), 6 + 4 * 6);
    }
    #[test]
    fn walls_from_coordinates_far_outside_the_tile() {
        // Edges between i32::MIN and the tile overflow i32; their walls
        // still clip to the tile.
        let mut b = LayerBuilder::new("building", 100);
        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        for p in [[i32::MIN, 20], [50, 20], [50, 80], [i32::MIN, 80]] {
            b.push_vertex(p);
        }
        let mesh = layout(&b.finish(), &[0]);
        assert!(mesh.vertex_indices().count() > 0);
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
        assert_eq!(mesh.vertex_indices().count(), 6);
    }
}
