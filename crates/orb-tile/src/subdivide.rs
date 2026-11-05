//! Subdivision along a grid, for tiles drawn through a curved warp onto
//! another plane (plan §9.2, `GpuPerVertex`): mapped vertices then follow the
//! curve between them, and the vertices on a tile's edges fall on grid lines
//! its neighbours share.

use crate::{GeometryType, Layer, TileFrame, clip_ring};

impl TileFrame {
    /// The frame with a vertex wherever a line or polygon-ring segment
    /// crosses a line of a grid of `cells` × `cells` over each layer's
    /// extent (polygon rings' implicit closing segments included).
    /// Polygons are first clipped to the box line layout keeps (the tile
    /// square and a sixteenth of it around), so the edges clipping makes
    /// are cut too and lie outside the tile, where no stroke of them shows.
    /// Points, ids and properties are unchanged.
    pub fn subdivided(&self, cells: u32) -> Self {
        Self {
            layers: self.layers.iter().map(|l| subdivide(l, cells)).collect(),
        }
    }
}

fn subdivide(layer: &Layer, cells: u32) -> Layer {
    let extent = layer.extent as i32;
    let step = f64::from(layer.extent) / f64::from(cells.max(1));
    let mut out = Layer {
        feature_parts: vec![0],
        part_rings: vec![0],
        ring_vertices: vec![0],
        vertices: Vec::with_capacity(layer.vertices.len()),
        ..layer.clone()
    };
    let mut clipped = Vec::new();
    let mut crossings = Vec::new();
    for feature in 0..layer.len() {
        let kind = layer.types[feature];
        let parts =
            layer.feature_parts[feature] as usize..layer.feature_parts[feature + 1] as usize;
        for part in parts {
            let rings = layer.part_rings[part] as usize..layer.part_rings[part + 1] as usize;
            let first_ring = out.ring_vertices.len();
            for ring in rings.clone() {
                let mut points = &layer.vertices
                    [layer.ring_vertices[ring] as usize..layer.ring_vertices[ring + 1] as usize];
                if kind == GeometryType::Polygon {
                    clip_ring(points, -extent / 16, extent + extent / 16, &mut clipped);
                    if clipped.is_empty() {
                        if ring == rings.start {
                            // Without its exterior the polygon is gone.
                            break;
                        }
                        continue;
                    }
                    points = &clipped;
                }
                let closed = kind == GeometryType::Polygon && points.len() > 2;
                for (i, &p) in points.iter().enumerate() {
                    out.vertices.push(p);
                    let next = match points.get(i + 1) {
                        Some(&next) => next,
                        None if closed => points[0],
                        None => break,
                    };
                    if kind == GeometryType::Point {
                        continue;
                    }
                    grid_crossings(p, next, step, &mut crossings);
                    out.vertices.extend(crossings.iter().copied());
                }
                out.ring_vertices.push(out.vertices.len() as u32);
            }
            if out.ring_vertices.len() > first_ring {
                out.part_rings.push((out.ring_vertices.len() - 1) as u32);
            }
        }
        out.feature_parts.push((out.part_rings.len() - 1) as u32);
    }
    out
}

/// The points where segment `a`–`b` crosses grid lines `step` apart, in
/// order from `a`, without its ends. Each crossing is computed from the
/// segment's ends in a fixed order, so a segment shared by two rings (or
/// two tiles) gets the same points either way.
pub(crate) fn grid_crossings(a: [i32; 2], b: [i32; 2], step: f64, out: &mut Vec<[i32; 2]>) {
    out.clear();
    let (lo, hi, flip) = if a <= b { (a, b, false) } else { (b, a, true) };
    let mut ts: Vec<(f64, [i32; 2])> = Vec::new();
    for axis in 0..2 {
        let (p, q) = (f64::from(lo[axis]), f64::from(hi[axis]));
        if p == q {
            continue;
        }
        let (min, max) = (p.min(q), p.max(q));
        let mut k = (min / step).floor() + 1.0;
        while k * step < max {
            let line = k * step;
            let t = (line - p) / (q - p);
            let other = 1 - axis;
            let mut point = [0; 2];
            point[axis] = line.round() as i32;
            point[other] =
                (f64::from(lo[other]) + t * f64::from(hi[other] - lo[other])).round() as i32;
            ts.push((t, point));
            k += 1.0;
        }
    }
    ts.sort_by(|x, y| x.0.total_cmp(&y.0));
    if flip {
        ts.reverse();
    }
    for (_, point) in ts {
        if out.last() != Some(&point) && point != a && point != b {
            out.push(point);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::grid_crossings;
    use crate::{GeometryType, LayerBuilder, TileFrame};

    #[test]
    fn cuts_segments_at_grid_lines() {
        let mut b = LayerBuilder::new("l", 4096);
        b.begin_feature(None, GeometryType::LineString);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([0, 100]);
        b.push_vertex([4096, 100]);
        b.begin_feature(None, GeometryType::Point);
        b.begin_part();
        b.begin_ring();
        b.push_vertex([10, 10]);
        b.push_vertex([4000, 4000]);
        b.begin_feature(None, GeometryType::Polygon);
        b.begin_part();
        b.begin_ring();
        for p in [[-1000, -1000], [5096, -1000], [5096, 2048], [-1000, 2048]] {
            b.push_vertex(p);
        }
        let frame = TileFrame {
            layers: vec![b.finish()],
        };
        let cut = frame.subdivided(8);
        let layer = &cut.layers[0];
        let ring = |f: usize| -> Vec<[i32; 2]> {
            let part = layer.geometry(f).parts().next().unwrap();
            part.rings().next().unwrap().to_vec()
        };
        // Seven grid lines across: eight pieces.
        assert_eq!(ring(0).len(), 9);
        assert_eq!(ring(0)[4], [2048, 100]);
        assert_eq!(ring(1), [[10, 10], [4000, 4000]]);
        // The polygon is clipped to line layout's box first (a sixteenth of
        // the tile around it); the edges along the box are cut at the grid
        // lines too.
        let square = ring(2);
        assert!(
            square
                .iter()
                .all(|p| (-256..=4352).contains(&p[0]) && (-256..=4352).contains(&p[1]))
        );
        let top: Vec<_> = square.iter().filter(|p| p[1] == -256).collect();
        assert_eq!(top.len(), 2 + 9, "{top:?}");
    }

    #[test]
    fn shared_segments_cut_alike_both_ways() {
        let (mut ab, mut ba) = (Vec::new(), Vec::new());
        grid_crossings([3, 7], [4001, 2999], 512.0, &mut ab);
        grid_crossings([4001, 2999], [3, 7], 512.0, &mut ba);
        ba.reverse();
        assert_eq!(ab, ba);
        assert!(ab.len() > 8);
    }
}
