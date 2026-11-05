//! Vector overzoom, following pinned GL JS's `sliceVectorTileLayer` and
//! `clipGeometry`. Clip before layout so each child starts its own line phase.

use crate::{GeometryType, Layer, LayerBuilder, TileFrame};

type Point = [f64; 2];

impl TileFrame {
    /// Slices a descendant at relative `(col, row)` and `depth` into normalized
    /// extent 8192. Intersections stay in f64 until the layout grid is rounded.
    pub fn overzoom(&self, depth: u8, col: u32, row: u32) -> Self {
        assert!(depth <= 30 && col < 1u32 << depth && row < 1u32 << depth);
        Self {
            layers: self
                .layers
                .iter()
                .map(|l| slice(l, depth, col, row))
                .filter(|l| !l.is_empty())
                .collect(),
        }
    }
}

fn slice(layer: &Layer, depth: u8, col: u32, row: u32) -> Layer {
    let scale = 2f64.powi(i32::from(depth)) * 8192.0 / f64::from(layer.extent());
    let offset = [f64::from(col) * 8192.0, f64::from(row) * 8192.0];
    let buffer = (128.0 * f64::from(layer.extent()) / 4096.0)
        .round()
        .max(128.0)
        * 8192.0
        / f64::from(layer.extent());
    let mut builder = LayerBuilder::new(layer.name(), 8192);
    builder.clipped_at(buffer.round() as i32);
    for feature in 0..layer.len() {
        let kind = layer.geometry_type(feature);
        let mut parts = Vec::new();
        for part in layer.geometry(feature).parts() {
            let mut rings = Vec::new();
            for ring in part.rings() {
                let mut points: Vec<Point> = ring
                    .iter()
                    .map(|p| {
                        [
                            f64::from(p[0]) * scale - offset[0],
                            f64::from(p[1]) * scale - offset[1],
                        ]
                    })
                    .collect();
                if kind == GeometryType::Polygon && !points.is_empty() {
                    points.push(points[0]);
                }
                let mut geometry = vec![points];
                for axis in 0..2 {
                    geometry = clip(geometry, kind, -buffer, 8192.0 + buffer, axis);
                }
                rings.extend(geometry);
            }
            if !rings.is_empty() {
                parts.push(rings);
            }
        }
        if parts.is_empty() {
            continue;
        }
        builder.begin_feature(None, kind);
        if let Some(id) = layer.ids().get(feature) {
            builder.set_id(id.to_owned());
        }
        for (key, column) in layer.properties() {
            if let Some(value) = column.get(feature) {
                builder.set_property(key, value.to_owned());
            }
        }
        for part in parts {
            builder.begin_part();
            for ring in part {
                builder.begin_ring();
                let end = ring.len()
                    - usize::from(
                        kind == GeometryType::Polygon
                            && ring.len() > 1
                            && ring.first() == ring.last(),
                    );
                for point in &ring[..end] {
                    builder.push_vertex(point.map(|v| (v + 0.5).floor() as i32));
                }
            }
        }
    }
    builder.finish()
}

fn clip(
    geometry: Vec<Vec<Point>>,
    kind: GeometryType,
    min: f64,
    max: f64,
    axis: usize,
) -> Vec<Vec<Point>> {
    if kind == GeometryType::Point {
        return geometry
            .into_iter()
            .flatten()
            .filter(|p| p[axis] >= min && p[axis] <= max)
            .map(|p| vec![p])
            .collect();
    }
    let polygon = kind == GeometryType::Polygon;
    let mut out = Vec::new();
    for line in geometry {
        let mut slice = Vec::new();
        let intersection = |a: Point, b: Point, edge: f64| {
            let t = (edge - a[axis]) / (b[axis] - a[axis]);
            let mut p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            p[axis] = edge;
            p
        };
        for pair in line.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let mut exited = false;
            if a[axis] < min {
                if b[axis] >= min {
                    slice.push(intersection(a, b, min));
                }
            } else if a[axis] > max {
                if b[axis] <= max {
                    slice.push(intersection(a, b, max));
                }
            } else {
                slice.push(a);
            }
            if b[axis] < min && a[axis] >= min {
                slice.push(intersection(a, b, min));
                exited = true;
            }
            if b[axis] > max && a[axis] <= max {
                slice.push(intersection(a, b, max));
                exited = true;
            }
            if !polygon && exited && !slice.is_empty() {
                out.push(std::mem::take(&mut slice));
            }
        }
        if let Some(&last) = line.last()
            && last[axis] >= min
            && last[axis] <= max
        {
            slice.push(last);
        }
        if polygon && !slice.is_empty() && slice.first() != slice.last() {
            slice.push(slice[0]);
        }
        if !slice.is_empty() {
            out.push(slice);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Value, ValueRef};
    #[test]
    fn child_clips_before_rounding_and_retains_columns() {
        let mut b = LayerBuilder::new("roads", 4096);
        b.begin_feature(Some(42), GeometryType::LineString);
        b.set_property("class", Value::Str("primary".into()));
        b.begin_part();
        b.begin_ring();
        b.push_vertex([0, 100]);
        b.push_vertex([4608, 101]);
        let child = TileFrame {
            layers: vec![b.finish()],
        }
        .overzoom(1, 1, 0);
        let l = &child.layers[0];
        assert_eq!(l.extent(), 8192);
        assert_eq!(l.vertices, [[-256, 402], [8448, 404]]);
        assert_eq!(l.ids().get(0), Some(ValueRef::U64(42)));
        assert_eq!(
            l.property("class").unwrap().get(0),
            Some(ValueRef::Str("primary"))
        );
    }
}
