//! Polygon label anchors: GL JS `find_pole_of_inaccessibility.ts`
//! (polylabel), with its `tinyqueue` ordering, so ties break the same way.

use std::f64::consts::SQRT_2;

use crate::geometry::{Point, js_max, js_min};

/// The point of `polygon` (an exterior ring, then its holes) farthest from
/// its outline, to `precision` tile units. Rings may be open or closed.
pub(crate) fn pole_of_inaccessibility(polygon: &[&[Point]], precision: f64) -> Point {
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in polygon[0] {
        min_x = js_min(min_x, p.x);
        min_y = js_min(min_y, p.y);
        max_x = js_max(max_x, p.x);
        max_y = js_max(max_y, p.y);
    }

    let cell_size = js_min(max_x - min_x, max_y - min_y);
    let mut h = cell_size / 2.0;
    let mut queue = Queue::default();

    if cell_size == 0.0 {
        return Point::new(min_x, min_y);
    }

    let mut x = min_x;
    while x < max_x {
        let mut y = min_y;
        while y < max_y {
            queue.push(Cell::new(x + h, y + h, h, polygon));
            y += cell_size;
        }
        x += cell_size;
    }

    let centroid = centroid_cell(polygon);
    let mut best = centroid;

    while let Some(cell) = queue.pop() {
        if cell.d > best.d || best.d == 0.0 || best.d.is_nan() {
            best = cell;
        }
        if cell.max - best.d <= precision {
            continue;
        }
        h = cell.h / 2.0;
        queue.push(Cell::new(cell.p.x - h, cell.p.y - h, h, polygon));
        queue.push(Cell::new(cell.p.x + h, cell.p.y - h, h, polygon));
        queue.push(Cell::new(cell.p.x - h, cell.p.y + h, h, polygon));
        queue.push(Cell::new(cell.p.x + h, cell.p.y + h, h, polygon));
    }

    // Near-convex polygons read better labelled at the centroid.
    if centroid.d > 0.0 && best.d - centroid.d <= precision {
        return centroid.p;
    }
    best.p
}

#[derive(Clone, Copy, Debug)]
struct Cell {
    p: Point,
    /// Half the cell size.
    h: f64,
    /// Distance from the cell centre to the outline, negative outside.
    d: f64,
    /// The largest distance any point of the cell can have.
    max: f64,
}

impl Cell {
    fn new(x: f64, y: f64, h: f64, polygon: &[&[Point]]) -> Self {
        let p = Point::new(x, y);
        let d = point_to_polygon_dist(p, polygon);
        Self {
            p,
            h,
            d,
            max: d + h * SQRT_2,
        }
    }
}

fn point_to_polygon_dist(p: Point, polygon: &[&[Point]]) -> f64 {
    let mut inside = false;
    let mut min_dist_sq = f64::INFINITY;
    for ring in polygon {
        let len = ring.len();
        let mut j = len.wrapping_sub(1);
        for i in 0..len {
            let a = ring[i];
            let b = ring[j];
            if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
                inside = !inside;
            }
            min_dist_sq = js_min(min_dist_sq, dist_to_segment_squared(p, a, b));
            j = i;
        }
    }
    (if inside { 1.0 } else { -1.0 }) * min_dist_sq.sqrt()
}

/// GL JS `distToSegmentSquared` (`intersection_tests.ts`).
pub(crate) fn dist_to_segment_squared(p: Point, v: Point, w: Point) -> f64 {
    let l2 = v.dist_sqr(w);
    if l2 == 0.0 {
        return p.dist_sqr(v);
    }
    let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
    if t < 0.0 {
        return p.dist_sqr(v);
    }
    if t > 1.0 {
        return p.dist_sqr(w);
    }
    p.dist_sqr(Point::new((w.x - v.x) * t + v.x, (w.y - v.y) * t + v.y))
}

fn centroid_cell(polygon: &[&[Point]]) -> Cell {
    let mut area = 0.0;
    let mut x = 0.0;
    let mut y = 0.0;
    let points = polygon[0];
    let len = points.len();
    let mut j = len.wrapping_sub(1);
    for i in 0..len {
        let a = points[i];
        let b = points[j];
        let f = a.x * b.y - b.x * a.y;
        x += (a.x + b.x) * f;
        y += (a.y + b.y) * f;
        area += f * 3.0;
        j = i;
    }
    Cell::new(x / area, y / area, 0.0, polygon)
}

/// `tinyqueue` with polylabel's comparator (`b.max - a.max`): a binary
/// heap whose sift order, not just its priority, decides ties.
#[derive(Debug, Default)]
struct Queue {
    data: Vec<Cell>,
}

impl Queue {
    fn compare(a: &Cell, b: &Cell) -> f64 {
        b.max - a.max
    }

    fn push(&mut self, item: Cell) {
        self.data.push(item);
        let mut pos = self.data.len() - 1;
        while pos > 0 {
            let parent = (pos - 1) >> 1;
            let current = self.data[parent];
            if Self::compare(&item, &current) >= 0.0 {
                break;
            }
            self.data[pos] = current;
            pos = parent;
        }
        self.data[pos] = item;
    }

    fn pop(&mut self) -> Option<Cell> {
        let top = *self.data.first()?;
        let bottom = self.data.pop().expect("queue is not empty");
        if !self.data.is_empty() {
            self.data[0] = bottom;
            self.down(0);
        }
        Some(top)
    }

    fn down(&mut self, mut pos: usize) {
        let len = self.data.len();
        let half = len >> 1;
        let item = self.data[pos];
        while pos < half {
            let mut best = (pos << 1) + 1;
            let right = best + 1;
            if right < len && Self::compare(&self.data[right], &self.data[best]) < 0.0 {
                best = right;
            }
            if Self::compare(&self.data[best], &item) >= 0.0 {
                break;
            }
            self.data[pos] = self.data[best];
            pos = best;
        }
        self.data[pos] = item;
    }
}
