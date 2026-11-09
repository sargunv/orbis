//! Polyline clipping to tile squares.

/// One connected piece of a clipped line. Closed rings have no repeated end.
#[derive(Debug, PartialEq)]
pub(crate) struct ClippedLine {
    pub points: Vec<[i32; 2]>,
    pub closed: bool,
    pub(crate) start_clipped: bool,
    pub(crate) end_clipped: bool,
    /// How far along the source line, or a ring from its first vertex,
    /// the piece starts.
    pub start_distance: f64,
    /// Where a clipped ring's piece passes the ring's first vertex, the
    /// point there and the distance it starts again with (GL JS's ring
    /// starts at its first vertex with the length of its closing edge).
    pub(crate) seam: Option<(usize, f64)>,
}

/// Clips segments independently, without introducing edges along the square.
/// Consecutive duplicates and zero-length intersections are discarded. A ring
/// crossing the square becomes open pieces with both ends marked as clipped.
/// Source positions may lie far outside the square (an overzoom slice lays
/// out its data tile's lines in its own units).
pub(crate) fn clip_polyline(
    source: &[[i64; 2]],
    closed: bool,
    min: i32,
    max: i32,
) -> Vec<ClippedLine> {
    let mut points = source.to_vec();
    points.dedup();
    if closed && points.len() > 1 && points.first() == points.last() {
        points.pop();
    }
    if points.len() < if closed { 3 } else { 2 } {
        return Vec::new();
    }
    let length =
        |a: [i64; 2], b: [i64; 2]| (b[0] as f64 - a[0] as f64).hypot(b[1] as f64 - a[1] as f64);
    let inside = |p: &[i64; 2]| {
        p.iter()
            .all(|&c| (i64::from(min)..=i64::from(max)).contains(&c))
    };
    // A rotated ring's distance from its first vertex to where it starts,
    // its perimeter, and where its first vertex lands.
    let mut rotation = None;
    if closed {
        if let Some(outside) = points.iter().position(|p| !inside(p)) {
            // Starting outside avoids splitting a connected piece at the ring
            // seam. Every surviving piece then starts and ends at a clip edge.
            let n = points.len();
            let offset: f64 = (0..outside).map(|i| length(points[i], points[i + 1])).sum();
            let perimeter: f64 = (0..n).map(|i| length(points[i], points[(i + 1) % n])).sum();
            let closing = length(points[n - 1], points[0]);
            points.rotate_left(outside);
            points.push(points[0]);
            rotation = Some((offset, perimeter, closing, (n - outside) % n));
        } else {
            return vec![ClippedLine {
                points: points.iter().map(|p| p.map(|v| v as i32)).collect(),
                closed: true,
                start_clipped: false,
                end_clipped: false,
                start_distance: 0.0,
                seam: None,
            }];
        }
    }

    let mut out = Vec::new();
    let mut piece: Option<ClippedLine> = None;
    let mut distance = 0.0;
    for (i, segment) in points.windows(2).enumerate() {
        let segment_start = distance;
        distance += length(segment[0], segment[1]);
        let Some((a, b, start_clipped, end_clipped)) =
            clip_segment(segment[0], segment[1], min, max)
        else {
            finish_line(&mut piece, &mut out);
            continue;
        };
        if start_clipped {
            finish_line(&mut piece, &mut out);
        }
        // Keep a single-point intersection until the next segment so an
        // entry exactly at an existing vertex still suppresses its cap.
        let line = piece.get_or_insert_with(|| {
            let start = segment_start + length(segment[0], a.map(i64::from));
            ClippedLine {
                points: vec![a],
                closed: false,
                start_clipped,
                end_clipped: false,
                start_distance: match rotation {
                    Some((offset, perimeter, closing, _)) => closing + (offset + start) % perimeter,
                    None => start,
                },
                seam: None,
            }
        });
        if let Some((.., closing, seam)) = rotation
            && seam == i
            && i > 0
            && !start_clipped
        {
            line.seam = Some((line.points.len() - 1, closing));
        }
        if line.points.last() != Some(&b) {
            line.points.push(b);
        }
        line.end_clipped = end_clipped;
        if end_clipped {
            finish_line(&mut piece, &mut out);
        }
    }
    finish_line(&mut piece, &mut out);
    out
}

fn finish_line(piece: &mut Option<ClippedLine>, out: &mut Vec<ClippedLine>) {
    if let Some(line) = piece.take()
        && line.points.len() >= 2
    {
        out.push(line);
    }
}

// Liang–Barsky, retaining the original endpoints' cap eligibility. Arithmetic
// is in f64 before subtraction, exact for source positions up to 2^53.
pub(crate) fn clip_segment(
    a: [i64; 2],
    b: [i64; 2],
    min: i32,
    max: i32,
) -> Option<([i32; 2], [i32; 2], bool, bool)> {
    let mut enter: f64 = 0.0;
    let mut exit: f64 = 1.0;
    for axis in 0..2 {
        let origin = a[axis] as f64;
        let delta = b[axis] as f64 - origin;
        if delta == 0.0 {
            if !(i64::from(min)..=i64::from(max)).contains(&a[axis]) {
                return None;
            }
        } else {
            let t0 = (f64::from(min) - origin) / delta;
            let t1 = (f64::from(max) - origin) / delta;
            enter = enter.max(t0.min(t1));
            exit = exit.min(t0.max(t1));
            if enter > exit {
                return None;
            }
        }
    }
    let point = |t: f64| {
        std::array::from_fn(|axis| {
            let (a, b) = (a[axis] as f64, b[axis] as f64);
            (a + t * (b - a))
                .round()
                .clamp(f64::from(min), f64::from(max)) as i32
        })
    };
    Some((point(enter), point(exit), enter > 0.0, exit < 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_and_deduplicates_inside_lines() {
        let lines = clip_polyline(&[[1, 1], [1, 1], [9, 9], [9, 9]], false, 0, 10);
        assert_eq!(
            lines,
            [ClippedLine {
                points: vec![[1, 1], [9, 9]],
                closed: false,
                start_clipped: false,
                end_clipped: false,
                start_distance: 0.0,
                seam: None,
            }]
        );
        assert!(clip_polyline(&[[1, 1], [1, 1]], false, 0, 10).is_empty());
        assert!(clip_polyline(&[], false, 0, 10).is_empty());
    }

    #[test]
    fn clips_crossings_even_when_both_endpoints_are_outside() {
        let lines = clip_polyline(&[[-20, 5], [20, 5]], false, -1, 11);
        assert_eq!(
            lines,
            [ClippedLine {
                points: vec![[-1, 5], [11, 5]],
                closed: false,
                start_clipped: true,
                end_clipped: true,
                start_distance: 19.0,
                seam: None,
            }]
        );
        let diagonal = clip_polyline(&[[-10, -10], [20, 20]], false, 0, 10);
        assert_eq!(diagonal[0].points, [[0, 0], [10, 10]]);
        let vertical = clip_polyline(&[[5, -10], [5, 20]], false, 0, 10);
        assert_eq!(vertical[0].points, [[5, 0], [5, 10]]);
    }

    #[test]
    fn splits_exit_and_reentry_without_connecting_clip_edges() {
        let lines = clip_polyline(&[[1, 1], [20, 1], [20, 9], [1, 9]], false, 0, 10);
        assert_eq!(
            lines,
            [
                ClippedLine {
                    points: vec![[1, 1], [10, 1]],
                    closed: false,
                    start_clipped: false,
                    end_clipped: true,
                    start_distance: 0.0,
                    seam: None,
                },
                ClippedLine {
                    points: vec![[10, 9], [1, 9]],
                    closed: false,
                    start_clipped: true,
                    end_clipped: false,
                    start_distance: 37.0,
                    seam: None,
                },
            ]
        );
        // An excursion must split even if both crossings round to one point.
        let same = clip_polyline(&[[1, 5], [20, 5], [1, 5]], false, 0, 10);
        assert_eq!(same.len(), 2);
    }

    #[test]
    fn closed_rings_stay_closed_only_when_unclipped() {
        let ring = [[1, 1], [9, 1], [9, 9], [1, 1], [1, 1]];
        let lines = clip_polyline(&ring, true, 0, 10);
        assert_eq!(lines[0].points, [[1, 1], [9, 1], [9, 9]]);
        assert!(lines[0].closed);
        // Start inside, cross twice: the original seam must remain connected.
        let clipped = clip_polyline(&[[5, 1], [20, 1], [20, 9], [5, 9]], true, 0, 10);
        assert_eq!(
            clipped,
            [ClippedLine {
                points: vec![[10, 9], [5, 9], [5, 1], [10, 1]],
                closed: false,
                start_clipped: true,
                end_clipped: true,
                // As GL JS counts it from the ring's first vertex, [5, 1],
                // which starts with the closing edge's length, 8: the
                // distance starts again there.
                start_distance: 8.0 + 33.0,
                seam: Some((2, 8.0)),
            }]
        );
        // A surrounding polygon has no visible outline, unlike clip_ring.
        assert!(
            clip_polyline(&[[-20, -20], [20, -20], [20, 20], [-20, 20]], true, 0, 10).is_empty()
        );
    }

    #[test]
    fn handles_tile_boundaries_tangencies_and_extreme_source_positions() {
        assert!(clip_polyline(&[[-1, 1], [1, -1]], false, 0, 10).is_empty());
        assert!(clip_polyline(&[[-10, -1], [20, -1]], false, 0, 10).is_empty());
        let edge = clip_polyline(&[[0, 0], [0, 10]], false, 0, 10);
        assert_eq!(edge[0].points, [[0, 0], [0, 10]]);
        assert!(!edge[0].start_clipped && !edge[0].end_clipped);
        let at_vertices = clip_polyline(&[[-1, 5], [0, 5], [10, 5], [11, 5]], false, 0, 10);
        assert_eq!(at_vertices[0].points, [[0, 5], [10, 5]]);
        assert!(at_vertices[0].start_clipped && at_vertices[0].end_clipped);
        let extreme = clip_polyline(&[[i64::MIN >> 11, 5], [i64::MAX >> 11, 5]], false, 0, 10);
        assert_eq!(extreme[0].points, [[0, 5], [10, 5]]);
    }
}
