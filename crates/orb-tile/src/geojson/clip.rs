use super::{Line, Options, Point, Projected, Shape};
use std::sync::Arc;

pub(super) fn clip(
    features: &[Projected],
    start: f64,
    end: f64,
    axis: usize,
    options: &Options,
) -> Vec<Projected> {
    let mut out = Vec::new();
    for f in features {
        if f.min[axis] >= start && f.max[axis] < end {
            out.push(f.clone());
            continue;
        }
        if f.max[axis] < start || f.min[axis] >= end {
            continue;
        }
        let shape = match &f.shape {
            Shape::Points(points) => Shape::Points(
                points
                    .iter()
                    .filter(|p| p.xy[axis] >= start && p.xy[axis] <= end)
                    .copied()
                    .collect(),
            ),
            Shape::Lines(lines) => {
                let lines: Vec<_> = lines
                    .iter()
                    .flat_map(|l| clip_line(l, start, end, axis, false, options.line_metrics))
                    .collect();
                if options.line_metrics {
                    for line in lines {
                        out.push(Projected::new(
                            f.id.clone(),
                            Arc::clone(&f.tags),
                            Shape::Lines(vec![line]),
                        ));
                    }
                    continue;
                }
                Shape::Lines(lines)
            }
            Shape::Polygons(polygons) => Shape::Polygons(
                polygons
                    .iter()
                    .map(|p| {
                        p.iter()
                            .flat_map(|l| clip_line(l, start, end, axis, true, false))
                            .collect::<Vec<_>>()
                    })
                    .filter(|p| !p.is_empty())
                    .collect(),
            ),
        };
        let f = Projected::new(f.id.clone(), Arc::clone(&f.tags), shape);
        if f.count() > 0 {
            out.push(f);
        }
    }
    out
}

fn clip_line(
    line: &Line,
    start: f64,
    end: f64,
    axis: usize,
    polygon: bool,
    metrics: bool,
) -> Vec<Line> {
    let new_slice = || Line {
        points: Vec::new(),
        size: line.size,
        start: line.start,
        end: line.end,
    };
    let mut slice = new_slice();
    let mut out = Vec::new();
    let mut length = line.start;
    for pair in line.points.windows(2) {
        let [a, b] = [pair[0], pair[1]];
        let av = a.xy[axis];
        let bv = b.xy[axis];
        let seg_length = if metrics {
            ((a.xy[0] - b.xy[0]).powi(2) + (a.xy[1] - b.xy[1]).powi(2)).sqrt()
        } else {
            0.0
        };
        let mut intersect = |edge| {
            let t = (edge - av) / (bv - av);
            let mut xy = [
                a.xy[0] + (b.xy[0] - a.xy[0]) * t,
                a.xy[1] + (b.xy[1] - a.xy[1]) * t,
            ];
            xy[axis] = edge;
            slice.points.push(Point {
                xy,
                importance: 1.0,
            });
            t
        };
        if av < start {
            if bv >= start {
                let t = intersect(start);
                if metrics {
                    slice.start = length + seg_length * t;
                }
            }
        } else if av > end {
            if bv <= end {
                let t = intersect(end);
                if metrics {
                    slice.start = length + seg_length * t;
                }
            }
        } else {
            slice.points.push(a);
        }
        let edge = if bv < start && av >= start {
            Some(start)
        } else if bv > end && av <= end {
            Some(end)
        } else {
            None
        };
        if let Some(edge) = edge {
            let t = (edge - av) / (bv - av);
            let mut xy = [
                a.xy[0] + (b.xy[0] - a.xy[0]) * t,
                a.xy[1] + (b.xy[1] - a.xy[1]) * t,
            ];
            xy[axis] = edge;
            slice.points.push(Point {
                xy,
                importance: 1.0,
            });
            if !polygon {
                if metrics {
                    slice.end = length + seg_length * t;
                }
                out.push(slice);
                slice = new_slice();
            }
        }
        if metrics {
            length += seg_length;
        }
    }
    if let Some(&p) = line.points.last()
        && p.xy[axis] >= start
        && p.xy[axis] <= end
    {
        slice.points.push(p);
    }
    if polygon
        && slice.points.len() > 1
        && slice.points.first().map(|p| p.xy) != slice.points.last().map(|p| p.xy)
    {
        slice.points.push(slice.points[0]);
    }
    if !slice.points.is_empty() {
        out.push(slice);
    }
    out
}

/// Copies features across a wrapping world's edges. Other planes don't
/// repeat: features keep to the unit square and its buffer.
pub(super) fn wrap(features: Vec<Projected>, options: &Options) -> Vec<Projected> {
    let buffer = options.buffer / f64::from(options.extent);
    if options.plane.is_some() {
        return clip(&features, -buffer, 1.0 + buffer, 0, options);
    }
    let left = clip(&features, -1.0 - buffer, buffer, 0, options);
    let right = clip(&features, 1.0 - buffer, 2.0 + buffer, 0, options);
    if left.is_empty() && right.is_empty() {
        return features;
    }
    let center = clip(&features, -buffer, 1.0 + buffer, 0, options);
    let mut merged = shift(left, 1.0);
    merged.extend(center);
    merged.extend(shift(right, -1.0));
    merged
}

fn shift(features: Vec<Projected>, offset: f64) -> Vec<Projected> {
    features
        .into_iter()
        .map(|mut f| {
            let shift_line = |l: &mut Line| {
                for p in &mut l.points {
                    p.xy[0] += offset;
                }
            };
            match &mut f.shape {
                Shape::Points(points) => {
                    for p in points {
                        p.xy[0] += offset;
                    }
                }
                Shape::Lines(lines) => lines.iter_mut().for_each(shift_line),
                Shape::Polygons(polygons) => {
                    for p in polygons {
                        p.iter_mut().for_each(shift_line);
                    }
                }
            }
            f.min[0] += offset;
            f.max[0] += offset;
            f
        })
        .collect()
}
