//! Release velocities, estimated as Compose's (and Android's)
//! `VelocityTracker1D` does with its default strategy: a least-squares
//! quadratic through the samples of the last 100 ms, back to the first gap
//! longer than 40 ms, whose slope at the newest sample is the velocity.
//! Like the platform's, a release adds no sample, and one more than 40 ms
//! after the last move has no velocity: the pointer held still.

/// How far back samples count, in milliseconds.
const HORIZON: f64 = 100.0;
/// A gap this long between samples means the pointer stopped.
const STOPPED: f64 = 40.0;
/// Samples kept at most.
const HISTORY: usize = 20;

/// One value's samples over time.
#[derive(Clone, Debug, Default)]
pub(crate) struct VelocityTracker {
    /// (time in milliseconds, value), oldest first.
    samples: Vec<(f64, f64)>,
}

impl VelocityTracker {
    pub(crate) fn reset(&mut self) {
        self.samples.clear();
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Adds the value at `time`. A sample at the newest one's time
    /// replaces it, as Compose's coalescing of host samples does; one
    /// before it starts over.
    pub(crate) fn add(&mut self, time: f64, value: f64) {
        match self.samples.last() {
            Some(&(last, _)) if time < last => self.samples.clear(),
            Some(&(last, _)) if time == last => {
                self.samples.pop();
            }
            _ => {}
        }
        self.samples.push((time, value));
        if self.samples.len() > HISTORY {
            self.samples.remove(0);
        }
    }

    /// The value's rate of change per second at a release at `now`.
    pub(crate) fn velocity(&self, now: f64) -> f64 {
        let Some(&(newest, _)) = self.samples.last() else {
            return 0.0;
        };
        if now - newest > STOPPED {
            return 0.0;
        }
        let mut points = Vec::with_capacity(self.samples.len());
        let mut previous = newest;
        for &(time, value) in self.samples.iter().rev() {
            if newest - time > HORIZON || previous - time > STOPPED {
                break;
            }
            previous = time;
            points.push((time - newest, value));
        }
        // Compose's quadratic needs three samples.
        if points.len() < 3 {
            return 0.0;
        }
        fit_slope(&points) * 1000.0
    }
}

/// The slope at x = 0 of the least-squares quadratic through `points`,
/// falling back to a line when the quadratic is degenerate.
fn fit_slope(points: &[(f64, f64)]) -> f64 {
    // Normal equations over powers of x, in seconds-scaled units for
    // conditioning.
    let mut s = [0.0; 5];
    let mut t = [0.0; 3];
    for &(x, y) in points {
        let x = x / HORIZON;
        let mut p = 1.0;
        for (i, si) in s.iter_mut().enumerate() {
            *si += p;
            if i < 3 {
                t[i] += p * y;
            }
            p *= x;
        }
    }
    let m = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(m);
    let scale = s[0] * s[2] * s[4];
    if d.abs() > 1e-12 * scale.abs().max(1e-300) {
        // Cramer's rule for the linear coefficient.
        let mut m1 = m;
        for (row, ti) in m1.iter_mut().zip(t) {
            row[1] = ti;
        }
        return det(m1) / d / HORIZON;
    }
    let d = s[0] * s[2] - s[1] * s[1];
    if d.abs() < 1e-12 {
        return 0.0;
    }
    (s[0] * t[1] - s[1] * t[0]) / d / HORIZON
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker(samples: &[(f64, f64)]) -> VelocityTracker {
        let mut tracker = VelocityTracker::default();
        for &(t, v) in samples {
            tracker.add(t, v);
        }
        tracker
    }

    #[test]
    fn steady_motion_reads_its_speed() {
        // 2 px per 8 ms: 250 px/s.
        let samples: Vec<_> = (0..10)
            .map(|i| (f64::from(i) * 8.0, f64::from(i) * 2.0))
            .collect();
        assert!((tracker(&samples).velocity(samples.last().unwrap().0) - 250.0).abs() < 1e-6);
    }

    #[test]
    fn a_quadratic_reads_its_final_slope() {
        // x = t² / 100 over 0..80 ms: the slope at 80 ms is 1.6 px/ms.
        let samples: Vec<_> = (0..=10)
            .map(|i| {
                let t = f64::from(i) * 8.0;
                (t, t * t / 100.0)
            })
            .collect();
        assert!((tracker(&samples).velocity(samples.last().unwrap().0) - 1600.0).abs() < 1e-6);
    }

    #[test]
    fn only_the_last_100_ms_count() {
        // Fast for a second, then 100 px/s over the last 100 ms.
        let mut samples: Vec<_> = (0..100)
            .map(|i| (f64::from(i) * 10.0, f64::from(i) * 50.0))
            .collect();
        let base = samples.last().unwrap().1;
        samples.extend((1..=10).map(|i| (990.0 + f64::from(i) * 10.0, base + f64::from(i))));
        assert!((tracker(&samples).velocity(samples.last().unwrap().0) - 100.0).abs() < 1e-6);
    }

    #[test]
    fn holding_still_has_no_velocity() {
        // A fast drag, then a pause, then moves again: the pause cuts the
        // history.
        let mut samples: Vec<_> = (0..10)
            .map(|i| (f64::from(i) * 8.0, f64::from(i) * 20.0))
            .collect();
        samples.extend([(122.0, 180.0), (130.0, 181.0), (138.0, 182.0)]);
        assert!((tracker(&samples).velocity(138.0) - 125.0).abs() < 1e-6);
        // Released 50 ms after the last move.
        assert_eq!(tracker(&samples).velocity(188.0), 0.0);
        assert!(tracker(&samples).velocity(170.0) > 0.0);
    }

    #[test]
    fn two_samples_are_not_enough() {
        assert_eq!(tracker(&[(0.0, 0.0), (8.0, 10.0)]).velocity(8.0), 0.0);
        assert_eq!(tracker(&[]).velocity(0.0), 0.0);
    }

    #[test]
    fn samples_at_one_time_coalesce_and_earlier_ones_restart() {
        let t = tracker(&[(0.0, 0.0), (8.0, 1.0), (8.0, 2.0), (16.0, 4.0)]);
        assert_eq!(t.samples, [(0.0, 0.0), (8.0, 2.0), (16.0, 4.0)]);
        let t = tracker(&[(0.0, 0.0), (8.0, 1.0), (4.0, 2.0)]);
        assert_eq!(t.samples, [(4.0, 2.0)]);
    }

    #[test]
    fn history_keeps_twenty_samples() {
        let samples: Vec<_> = (0..30).map(|i| (f64::from(i), f64::from(i))).collect();
        let t = tracker(&samples);
        assert_eq!(t.samples.len(), 20);
        assert!((t.velocity(29.0) - 1000.0).abs() < 1e-6);
    }
}
