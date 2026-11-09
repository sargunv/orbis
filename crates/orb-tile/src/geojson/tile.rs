use super::{Key, Line, Options, Point, Projected, Shape};
use crate::{GeometryType, LayerBuilder, TileFrame, Value};

pub(super) fn create(features: &[Projected], (z, x, y): Key, options: &Options) -> TileFrame {
    let scale = f64::from(1u32 << z);
    let extent = f64::from(options.extent);
    let tolerance = if z == options.max_zoom {
        0.0
    } else {
        options.tolerance / (scale * extent)
    };
    let transform = |p: &Point| {
        [
            ((p.xy[0] * scale - f64::from(x)) * extent + 0.5).floor() as i32,
            ((p.xy[1] * scale - f64::from(y)) * extent + 0.5).floor() as i32,
        ]
    };
    let mut b = LayerBuilder::new(&options.layer, options.extent);
    b.clipped_at(options.buffer.round() as i32);
    for f in features {
        let ty = match *f.shape {
            Shape::Points(_) => GeometryType::Point,
            Shape::Lines(_) => GeometryType::LineString,
            Shape::Polygons(_) => GeometryType::Polygon,
        };
        b.begin_feature(None, ty);
        if let Some(id) = &f.id {
            b.set_id(id.clone());
        }
        for (key, value) in f.tags.iter() {
            b.set_property(key, value.clone());
        }
        let mut parts = 0;
        match &*f.shape {
            Shape::Points(points) => {
                b.begin_part();
                b.begin_ring();
                parts += 1;
                for p in points {
                    b.push_vertex(transform(p));
                }
            }
            Shape::Lines(lines) => {
                for line in lines {
                    if let Some(points) = simplify_line(line, tolerance, false, false) {
                        b.begin_part();
                        b.begin_ring();
                        parts += 1;
                        for p in &points {
                            b.push_vertex(transform(p));
                        }
                    }
                }
                if options.line_metrics && lines.len() == 1 {
                    b.set_property(
                        "geojsonvt_clip_start",
                        Value::F64(lines[0].start / lines[0].size),
                    );
                    b.set_property(
                        "geojsonvt_clip_end",
                        Value::F64(lines[0].end / lines[0].size),
                    );
                }
            }
            Shape::Polygons(polygons) => {
                for polygon in polygons {
                    let mut begun = false;
                    for (i, line) in polygon.iter().enumerate() {
                        if let Some(points) = simplify_line(line, tolerance, true, i == 0) {
                            // TileFrame uses implicit closure. Keep winding and ring order,
                            // and omit a ring that degenerates during quantization.
                            let mut ring: Vec<_> = points.iter().map(transform).collect();
                            ring.dedup();
                            if ring.len() > 1 && ring.first() == ring.last() {
                                ring.pop();
                            }
                            if ring.len() < 3 {
                                continue;
                            }
                            if !begun {
                                b.begin_part();
                                parts += 1;
                                begun = true;
                            }
                            b.begin_ring();
                            for p in ring {
                                b.push_vertex(p);
                            }
                        }
                    }
                }
            }
        }
        if parts == 0 {
            b.abandon_feature();
        }
    }
    TileFrame {
        layers: vec![b.finish()],
    }
}

fn simplify_line(line: &Line, tolerance: f64, polygon: bool, outer: bool) -> Option<Vec<Point>> {
    let sq = tolerance * tolerance;
    if tolerance > 0.0 && line.size < if polygon { sq } else { tolerance } {
        return None;
    }
    let mut points: Vec<_> = line
        .points
        .iter()
        .filter(|p| tolerance == 0.0 || p.importance > sq)
        .copied()
        .collect();
    if polygon && !points.is_empty() {
        let mut area = 0.0;
        let mut prev = points.last().expect("nonempty").xy;
        for p in &points {
            area += (p.xy[0] - prev[0]) * (p.xy[1] + prev[1]);
            prev = p.xy;
        }
        if (area > 0.0) == outer {
            points.reverse();
        }
    }
    Some(points)
}
