//! A static 2D index of points, ported from `kdbush` 4.0.2 (ISC, see
//! LICENSE in this directory), with its `Float32Array` coordinates:
//! supercluster's queries return points in its order, which decides which
//! cluster a point joins.

#[derive(Debug)]
pub(super) struct KdBush {
    node_size: usize,
    ids: Vec<u32>,
    coords: Vec<f32>,
}

impl KdBush {
    pub(super) fn new(points: impl Iterator<Item = [f64; 2]>, node_size: usize) -> Self {
        let mut coords = Vec::new();
        for [x, y] in points {
            coords.extend([x as f32, y as f32]);
        }
        let n = coords.len() / 2;
        let mut index = Self {
            node_size,
            ids: (0..n as u32).collect(),
            coords,
        };
        if n > 0 {
            index.sort(0, n - 1, 0);
        }
        index
    }

    fn at(&self, i: usize) -> (f64, f64) {
        (
            f64::from(self.coords[2 * i]),
            f64::from(self.coords[2 * i + 1]),
        )
    }

    /// The points in `min..=max`.
    pub(super) fn range(&self, min: [f64; 2], max: [f64; 2]) -> Vec<u32> {
        let inside = |(x, y): (f64, f64)| x >= min[0] && x <= max[0] && y >= min[1] && y <= max[1];
        self.search(inside, |axis, v| (min[axis] <= v, max[axis] >= v))
    }

    /// The points within `r` of `q`.
    pub(super) fn within(&self, q: [f64; 2], r: f64) -> Vec<u32> {
        let r2 = r * r;
        let inside = |(x, y): (f64, f64)| {
            let (dx, dy) = (x - q[0], y - q[1]);
            dx * dx + dy * dy <= r2
        };
        self.search(inside, |axis, v| (q[axis] - r <= v, q[axis] + r >= v))
    }

    /// kdbush's traversal: `sides(axis, v)` says whether the query reaches
    /// below and above a node's split value.
    fn search(
        &self,
        inside: impl Fn((f64, f64)) -> bool,
        sides: impl Fn(usize, f64) -> (bool, bool),
    ) -> Vec<u32> {
        let mut result = Vec::new();
        if self.ids.is_empty() {
            return result;
        }
        let mut stack = vec![(0usize, self.ids.len() - 1, 0usize)];
        while let Some((left, right, axis)) = stack.pop() {
            if right - left <= self.node_size {
                for i in left..=right {
                    if inside(self.at(i)) {
                        result.push(self.ids[i]);
                    }
                }
                continue;
            }
            let m = (left + right) >> 1;
            let p = self.at(m);
            if inside(p) {
                result.push(self.ids[m]);
            }
            let v = if axis == 0 { p.0 } else { p.1 };
            let (below, above) = sides(axis, v);
            if below {
                stack.push((left, m - 1, 1 - axis));
            }
            if above {
                stack.push((m + 1, right, 1 - axis));
            }
        }
        result
    }

    fn sort(&mut self, left: usize, right: usize, axis: usize) {
        if right - left <= self.node_size {
            return;
        }
        let m = (left + right) >> 1;
        self.select(m, left, right, axis);
        self.sort(left, m - 1, 1 - axis);
        self.sort(m + 1, right, 1 - axis);
    }

    /// Floyd–Rivest selection: the `k`th point along `axis` at `k`, smaller
    /// ones before it and larger after.
    fn select(&mut self, k: usize, mut left: usize, mut right: usize, axis: usize) {
        let c = |s: &Self, i: usize| s.coords[2 * i + axis];
        while right > left {
            if right - left > 600 {
                let n = (right - left + 1) as f64;
                let m = (k - left + 1) as f64;
                let z = n.ln();
                let s = 0.5 * (2.0 * z / 3.0).exp();
                let sd =
                    0.5 * (z * s * (n - s) / n).sqrt() * if m - n / 2.0 < 0.0 { -1.0 } else { 1.0 };
                let k_ = k as f64;
                let new_left = (left as f64).max((k_ - m * s / n + sd).floor()) as usize;
                let new_right = (right as f64).min((k_ + (n - m) * s / n + sd).floor()) as usize;
                self.select(k, new_left, new_right, axis);
            }
            let t = c(self, k);
            let mut i = left as isize;
            let mut j = right as isize;
            self.swap(left, k);
            if c(self, right) > t {
                self.swap(left, right);
            }
            while i < j {
                self.swap(i as usize, j as usize);
                i += 1;
                j -= 1;
                while c(self, i as usize) < t {
                    i += 1;
                }
                while c(self, j as usize) > t {
                    j -= 1;
                }
            }
            if c(self, left) == t {
                self.swap(left, j as usize);
            } else {
                j += 1;
                self.swap(j as usize, right);
            }
            let j = j as usize;
            if j <= k {
                left = j + 1;
            }
            if k <= j {
                if j == 0 {
                    break;
                }
                right = j - 1;
            }
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.ids.swap(i, j);
        self.coords.swap(2 * i, 2 * j);
        self.coords.swap(2 * i + 1, 2 * j + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_and_within_find_what_a_scan_finds() {
        // A deterministic scatter, more points than a node holds, with
        // repeats.
        let points: Vec<[f64; 2]> = (0..2000u32)
            .map(|i| {
                let h = i.wrapping_mul(2_654_435_761);
                [
                    f64::from(h % 1000) / 1000.0,
                    f64::from((h / 1000) % 1000) / 1000.0,
                ]
            })
            .collect();
        let index = KdBush::new(points.iter().copied(), 16);
        let f = |p: [f64; 2]| [f64::from(p[0] as f32), f64::from(p[1] as f32)];
        let mut found = index.range([0.2, 0.3], [0.5, 0.45]);
        found.sort_unstable();
        let scan: Vec<u32> = (0..points.len() as u32)
            .filter(|&i| {
                let [x, y] = f(points[i as usize]);
                (0.2..=0.5).contains(&x) && (0.3..=0.45).contains(&y)
            })
            .collect();
        assert_eq!(found, scan);
        let mut near = index.within([0.5, 0.5], 0.1);
        near.sort_unstable();
        let scan: Vec<u32> = (0..points.len() as u32)
            .filter(|&i| {
                let [x, y] = f(points[i as usize]);
                (x - 0.5).powi(2) + (y - 0.5).powi(2) <= 0.01
            })
            .collect();
        assert_eq!(near, scan);
    }
}
