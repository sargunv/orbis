//! Lines in one buffer, and GL JS's `clipLine` (`clip_line.ts`).

use std::ops::Range;

use crate::geometry::Point;

/// Several lines stored back to back, reused across features.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Lines {
    points: Vec<Point>,
    ends: Vec<usize>,
}

impl Lines {
    pub(crate) fn clear(&mut self) {
        self.points.clear();
        self.ends.clear();
    }

    pub(crate) fn len(&self) -> usize {
        self.ends.len()
    }

    fn range(&self, i: usize) -> Range<usize> {
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        start..self.ends[i]
    }

    pub(crate) fn line(&self, i: usize) -> &[Point] {
        &self.points[self.range(i)]
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &[Point]> {
        (0..self.len()).map(|i| self.line(i))
    }

    fn begin_line(&mut self, p: Point) {
        self.points.push(p);
        self.ends.push(self.points.len());
    }

    fn extend_last(&mut self, p: Point) {
        self.points.push(p);
        *self.ends.last_mut().expect("a line is open") = self.points.len();
    }

    fn last_point(&self) -> Option<Point> {
        self.points.last().copied()
    }
}

/// The parts of `lines` inside the box `[x1, x2) × [y1, y2)`, with
/// intersections rounded (GL JS `clipLine`). Appends to `out`.
pub(crate) fn clip_line<'a>(
    lines: impl IntoIterator<Item = &'a [Point]>,
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    out: &mut Lines,
) {
    clip_line_rounding(lines, [x1, y1, x2, y2], Point::round, out);
}

/// [`clip_line`] with intersections rounded by `round`.
pub(crate) fn clip_line_rounding<'a>(
    lines: impl IntoIterator<Item = &'a [Point]>,
    [x1, y1, x2, y2]: [f64; 4],
    round: impl Fn(Point) -> Point,
    out: &mut Lines,
) {
    for line in lines {
        let mut open = false;
        for pair in line.windows(2) {
            let (mut p0, mut p1) = (pair[0], pair[1]);

            if p0.x < x1 && p1.x < x1 {
                continue;
            } else if p0.x < x1 {
                p0 = round(Point::new(
                    x1,
                    p0.y + (p1.y - p0.y) * ((x1 - p0.x) / (p1.x - p0.x)),
                ));
            } else if p1.x < x1 {
                p1 = round(Point::new(
                    x1,
                    p0.y + (p1.y - p0.y) * ((x1 - p0.x) / (p1.x - p0.x)),
                ));
            }

            if p0.y < y1 && p1.y < y1 {
                continue;
            } else if p0.y < y1 {
                p0 = round(Point::new(
                    p0.x + (p1.x - p0.x) * ((y1 - p0.y) / (p1.y - p0.y)),
                    y1,
                ));
            } else if p1.y < y1 {
                p1 = round(Point::new(
                    p0.x + (p1.x - p0.x) * ((y1 - p0.y) / (p1.y - p0.y)),
                    y1,
                ));
            }

            if p0.x >= x2 && p1.x >= x2 {
                continue;
            } else if p0.x >= x2 {
                p0 = round(Point::new(
                    x2,
                    p0.y + (p1.y - p0.y) * ((x2 - p0.x) / (p1.x - p0.x)),
                ));
            } else if p1.x >= x2 {
                p1 = round(Point::new(
                    x2,
                    p0.y + (p1.y - p0.y) * ((x2 - p0.x) / (p1.x - p0.x)),
                ));
            }

            if p0.y >= y2 && p1.y >= y2 {
                continue;
            } else if p0.y >= y2 {
                p0 = round(Point::new(
                    p0.x + (p1.x - p0.x) * ((y2 - p0.y) / (p1.y - p0.y)),
                    y2,
                ));
            } else if p1.y >= y2 {
                p1 = round(Point::new(
                    p0.x + (p1.x - p0.x) * ((y2 - p0.y) / (p1.y - p0.y)),
                    y2,
                ));
            }

            if !open || out.last_point() != Some(p0) {
                out.begin_line(p0);
                open = true;
            }
            out.extend_last(p1);
        }
    }
}
