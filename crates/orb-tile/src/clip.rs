//! Polygon clipping to axis-aligned lines (Sutherland–Hodgman), shared by
//! layout, subdivision and queries.

/// A coordinate polygons clip in: crossings are found in f64, and land
/// back on an integer grid rounded.
pub trait Coord: Copy + PartialOrd + Into<f64> {
    fn from_f64(v: f64) -> Self;
}

impl Coord for i32 {
    fn from_f64(v: f64) -> Self {
        v.round() as i32
    }
}

impl Coord for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// One Sutherland–Hodgman step: polygon `ring` (implicitly closed) cut
/// along `p[axis] == bound`, keeping the side above it (`keep_above`) or
/// below, into `out`. A crossing is found from its edge's ends in a fixed
/// order, so polygons sharing an edge, either way round and in one tile
/// or two, cut it at the same point.
pub fn clip_step<T: Coord>(
    ring: &[[T; 2]],
    axis: usize,
    bound: T,
    keep_above: bool,
    out: &mut Vec<[T; 2]>,
) {
    out.clear();
    let inside = |p: &[T; 2]| {
        if keep_above {
            p[axis] >= bound
        } else {
            p[axis] <= bound
        }
    };
    let Some(&last) = ring.last() else { return };
    let mut prev = last;
    for &p in ring {
        let cross = || {
            let (lo, hi) = if prev <= p { (prev, p) } else { (p, prev) };
            let at = |a: usize| (lo[a].into(), hi[a].into());
            let ((a0, a1), (o0, o1)) = (at(axis), at(1 - axis));
            let t = (bound.into() - a0) / (a1 - a0);
            let mut q = [bound; 2];
            q[1 - axis] = T::from_f64(o0 + t * (o1 - o0));
            q
        };
        match (inside(&prev), inside(&p)) {
            (true, true) => out.push(p),
            (true, false) => out.push(cross()),
            (false, true) => {
                out.push(cross());
                out.push(p);
            }
            (false, false) => {}
        }
        prev = p;
    }
}

/// Polygon `ring` (implicitly closed) clipped to the square `lo..=hi` on
/// both axes into `out`, empty when nothing of it is inside. It may run
/// along the square's edges and repeat a point where it touches one. A
/// ring wholly inside or wholly outside costs one pass over it.
pub fn clip_square<T: Coord>(ring: &[[T; 2]], lo: T, hi: T, out: &mut Vec<[T; 2]>) {
    out.clear();
    let Some(&first) = ring.first() else { return };
    let (min, max) = ring.iter().fold((first, first), |(min, max), p| {
        (
            [0, 1].map(|a| if p[a] < min[a] { p[a] } else { min[a] }),
            [0, 1].map(|a| if p[a] > max[a] { p[a] } else { max[a] }),
        )
    });
    if (0..2).any(|a| max[a] < lo || min[a] > hi) {
        return;
    }
    if (0..2).all(|a| min[a] >= lo && max[a] <= hi) {
        out.extend_from_slice(ring);
        return;
    }
    let mut scratch = Vec::with_capacity(ring.len() + 4);
    clip_step(ring, 0, lo, true, out);
    clip_step(out, 0, hi, false, &mut scratch);
    clip_step(&scratch, 1, lo, true, out);
    clip_step(out, 1, hi, false, &mut scratch);
    std::mem::swap(out, &mut scratch);
}

/// Polygon `ring` clipped to the square `lo..=hi` on both axes into `out`
/// ([`clip_square`]), without repeated points, a closing one included, and
/// empty when less than a triangle is left.
pub fn clip_ring(ring: &[[i32; 2]], lo: i32, hi: i32, out: &mut Vec<[i32; 2]>) {
    clip_square(ring, lo, hi, out);
    out.dedup();
    while out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    if out.len() < 3 {
        out.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{clip_ring, clip_step};

    #[test]
    fn keeps_inside_clips_crossing_and_drops_outside_rings() {
        let mut out = Vec::new();
        clip_ring(&[[1, 1], [9, 1], [9, 9], [9, 9], [1, 1]], 0, 10, &mut out);
        assert_eq!(out, [[1, 1], [9, 1], [9, 9]]);
        clip_ring(
            &[[-10, -10], [20, -10], [20, 20], [-10, 20]],
            0,
            10,
            &mut out,
        );
        out.sort();
        assert_eq!(out, [[0, 0], [0, 10], [10, 0], [10, 10]]);
        clip_ring(&[[20, 20], [30, 20], [30, 30]], 0, 10, &mut out);
        assert!(out.is_empty());
        // Crossing nothing but surrounding the square, it is the square.
        clip_ring(&[[-5, -5], [30, -5], [-5, 30]], 0, 10, &mut out);
        assert_eq!(out.len(), 4);
    }

    /// An edge two polygons share, one each way round, is cut at one point.
    #[test]
    fn shared_edges_cut_alike() {
        // Taken from either end in f64, this edge crosses x = 0 at -14.5
        // or just past it.
        let (a, b) = ([-38, 14], [18, -28]);
        let (mut ab, mut ba) = (Vec::new(), Vec::new());
        clip_step(&[a, b, [18, 40]], 0, 0, true, &mut ab);
        clip_step(&[b, a, [-38, -60]], 0, 0, true, &mut ba);
        let on = |ring: &[[i32; 2]]| ring.iter().find(|p| p[0] == 0 && (-20..-10).contains(&p[1])).copied();
        assert_eq!(on(&ab), on(&ba));
        assert!(on(&ab).is_some());
    }
}
