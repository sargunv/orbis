//! Ring clipping to a square, shared by layout and subdivision.

/// Clips a closed ring to the square `lo..=hi` on both axes
/// (Sutherland–Hodgman). The result may run along the square's edges; it is
/// empty when nothing is inside.
pub fn clip_ring(ring: &[[i32; 2]], lo: i32, hi: i32, out: &mut Vec<[i32; 2]>) {
    let mut a: Vec<[i32; 2]> = ring.to_vec();
    let mut b = Vec::with_capacity(ring.len() + 4);
    // (axis, keep >= bound when `min`, else keep <= bound)
    for (axis, bound, min) in [(0, lo, true), (0, hi, false), (1, lo, true), (1, hi, false)] {
        b.clear();
        let inside = |p: &[i32; 2]| {
            if min {
                p[axis] >= bound
            } else {
                p[axis] <= bound
            }
        };
        let Some(&last) = a.last() else { break };
        let mut prev = last;
        for &p in &a {
            match (inside(&prev), inside(&p)) {
                (true, true) => b.push(p),
                (true, false) => b.push(intersect(prev, p, axis, bound)),
                (false, true) => {
                    b.push(intersect(prev, p, axis, bound));
                    b.push(p);
                }
                (false, false) => {}
            }
            prev = p;
        }
        std::mem::swap(&mut a, &mut b);
    }
    out.clear();
    for p in a {
        if out.last() != Some(&p) {
            out.push(p);
        }
    }
    while out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    if out.len() < 3 {
        out.clear();
    }
}

/// The point where segment `a → b` crosses `p[axis] == bound`, rounded to
/// the grid.
fn intersect(a: [i32; 2], b: [i32; 2], axis: usize, bound: i32) -> [i32; 2] {
    let other = 1 - axis;
    let t = f64::from(bound - a[axis]) / f64::from(b[axis] - a[axis]);
    let mut p = [0; 2];
    p[axis] = bound;
    p[other] = (f64::from(a[other]) + t * f64::from(b[other] - a[other])).round() as i32;
    p
}

#[cfg(test)]
mod tests {
    use super::clip_ring;

    #[test]
    fn keeps_inside_clips_crossing_and_drops_outside_rings() {
        let mut out = Vec::new();
        clip_ring(&[[1, 1], [9, 1], [9, 9]], 0, 10, &mut out);
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
    }
}
