use super::{Feature, Geometry, Line, Options, Point, Projected, Shape};
use crate::Value;
use std::sync::Arc;

pub(super) fn convert(features: Vec<Feature>, options: &Options) -> Vec<Projected> {
    let mut out = Vec::new();
    let tolerance = (options.tolerance
        / (f64::from(1u32 << options.max_zoom) * f64::from(options.extent)))
    .powi(2);
    for (index, f) in features.into_iter().enumerate() {
        let id = if let Some(key) = &options.promote_id {
            f.properties
                .iter()
                .rev()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
        } else if options.generate_id {
            Some(Value::U64(index as u64))
        } else {
            f.id
        };
        if let Some(geometry) = f.geometry {
            geometry_to_features(
                geometry,
                id,
                Arc::new(f.properties),
                tolerance,
                options,
                &mut out,
            );
        }
    }
    out
}

fn geometry_to_features(
    geometry: Geometry,
    id: Option<Value>,
    tags: Arc<Vec<(String, Value)>>,
    tolerance: f64,
    options: &Options,
    out: &mut Vec<Projected>,
) {
    let shape = match geometry {
        Geometry::Points(points) => Shape::Points(
            points
                .into_iter()
                .filter_map(|p| project(p, options.plane.as_deref()))
                .collect(),
        ),
        Geometry::Lines(lines) => {
            let lines: Vec<_> = lines
                .into_iter()
                .filter_map(|l| line(l, tolerance, false, options.plane.as_deref()))
                .collect();
            if options.line_metrics {
                for l in lines {
                    out.push(Projected::new(
                        id.clone(),
                        Arc::clone(&tags),
                        Shape::Lines(vec![l]),
                    ));
                }
                return;
            }
            Shape::Lines(lines)
        }
        Geometry::Polygons(polygons) => Shape::Polygons(
            polygons
                .into_iter()
                .map(|p| {
                    p.into_iter()
                        .filter_map(|l| line(l, tolerance, true, options.plane.as_deref()))
                        .collect::<Vec<_>>()
                })
                .filter(|p| !p.is_empty())
                .collect(),
        ),
        Geometry::Collection(geometries) => {
            for g in geometries {
                geometry_to_features(g, id.clone(), Arc::clone(&tags), tolerance, options, out);
            }
            return;
        }
    };
    let feature = Projected::new(id, tags, shape);
    if feature.count() > 0 {
        out.push(feature);
    }
}

/// A position in the index's unit square, or `None` outside the plane's
/// projection.
fn project([lon, lat]: [f64; 2], plane: Option<&orb_crs::TileMatrixSet>) -> Option<Point> {
    let xy = match plane {
        None => {
            let sin = (lat * std::f64::consts::PI / 180.0).sin();
            let y = 0.5 - 0.25 * ((1.0 + sin) / (1.0 - sin)).ln() / std::f64::consts::PI;
            [lon / 360.0 + 0.5, y.clamp(0.0, 1.0)]
        }
        Some(tms) => {
            let g = orb_geo::Geodetic::new(lon, lat);
            let projection = tms.projection();
            if !projection.contains(g) {
                return None;
            }
            let unit = tms.unit(projection.forward(g));
            if !unit.is_finite() {
                return None;
            }
            unit.to_array()
        }
    };
    Some(Point {
        xy,
        importance: 0.0,
    })
}

/// Longitude and latitude steps no longer than this many degrees: a
/// straight segment in longitude and latitude curves on other planes.
const DENSIFY_DEG: f64 = 1.0;

fn line(
    coords: Vec<[f64; 2]>,
    tolerance: f64,
    polygon: bool,
    plane: Option<&orb_crs::TileMatrixSet>,
) -> Option<Line> {
    let coords = match plane {
        None => coords,
        Some(_) => densify(&coords),
    };
    let mut points: Vec<_> = coords
        .into_iter()
        .filter_map(|p| project(p, plane))
        .collect();
    if points.is_empty() {
        return None;
    }
    let mut size = 0.0;
    for pair in points.windows(2) {
        let [a, b] = [pair[0].xy, pair[1].xy];
        size += if polygon {
            (a[0] * b[1] - b[0] * a[1]) / 2.0
        } else {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        };
    }
    let last = points.len() - 1;
    points[0].importance = 1.0;
    if tolerance > 0.0 {
        simplify(&mut points, tolerance);
    }
    points[last].importance = 1.0;
    Some(Line {
        points,
        size: size.abs(),
        start: 0.0,
        end: size.abs(),
    })
}

fn simplify(points: &mut [Point], tolerance: f64) {
    let mut stack = vec![(0, points.len() - 1)];
    while let Some((first, last)) = stack.pop() {
        let mut max = tolerance;
        let mid = first * 3 + (((last - first) * 3) >> 1);
        let mut closest = (last - first) * 3;
        let mut pivot = first;
        for i in first + 1..last {
            let distance = segment_distance(points[i].xy, points[first].xy, points[last].xy);
            if distance > max {
                pivot = i;
                max = distance;
            } else if distance == max {
                let position = (i * 3).abs_diff(mid);
                if position < closest {
                    pivot = i;
                    closest = position;
                }
            }
        }
        if max > tolerance {
            points[pivot].importance = max;
            if pivot - first > 1 {
                stack.push((first, pivot));
            }
            if last - pivot > 1 {
                stack.push((pivot, last));
            }
        }
    }
}

fn segment_distance(p: [f64; 2], mut a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    if dx != 0.0 || dy != 0.0 {
        let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy);
        if t > 1.0 {
            a = b;
        } else if t > 0.0 {
            a[0] += dx * t;
            a[1] += dy * t;
        }
    }
    (p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)
}

fn densify(coords: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(coords.len());
    for (i, &b) in coords.iter().enumerate() {
        if let Some(&a) = i.checked_sub(1).and_then(|j| coords.get(j)) {
            let span = (b[0] - a[0]).abs().max((b[1] - a[1]).abs());
            let steps = (span / DENSIFY_DEG).ceil().max(1.0) as usize;
            for k in 1..steps {
                let t = k as f64 / steps as f64;
                out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
            }
        }
        out.push(b);
    }
    out
}
