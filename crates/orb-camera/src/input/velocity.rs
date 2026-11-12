//! Release velocities, as mln-rs's `mln_gestures`, maplibre-native-ffi's
//! browser demo and GL JS's `HandlerInertia` estimate them: the motion
//! over the last 100 ms, over the time since the window starts. An average
//! can't point against the motion it averages, nor read a flick as faster
//! than it moved, both of which a least-squares fit's final slope
//! (Compose's `VelocityTracker1D`) does when the pointer brakes or speeds
//! up just before it lifts.
//!
//! As Android's and Compose's trackers do, a release more than 40 ms after
//! the last move has no velocity: the pointer held still. A shorter pause
//! slows it.

/// How far back the motion counts, in milliseconds.
const HORIZON: f64 = 100.0;
/// A release this long after the last move means the pointer stopped
/// (Compose's `AssumePointerMoveStoppedMilliseconds`).
const STOPPED: f64 = 40.0;

/// One value's samples over time.
#[derive(Clone, Debug, Default)]
pub(crate) struct VelocityTracker {
    /// (time in milliseconds, value), oldest first: the samples of the
    /// [`HORIZON`] before the newest and the one before them, where the
    /// window starts.
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
        let keep = self
            .samples
            .iter()
            .rposition(|&(t, _)| t <= time - HORIZON)
            .unwrap_or(0);
        self.samples.drain(..keep);
    }

    /// The value's rate of change per second at a release at `now`: its
    /// change since the window that ends at `now` starts, over the time
    /// since then. None when the pointer stopped before the release.
    pub(crate) fn velocity(&self, now: f64) -> f64 {
        let Some(&(newest, end)) = self.samples.last() else {
            return 0.0;
        };
        if now - newest > STOPPED {
            return 0.0;
        }
        let start = (self.samples.iter().rev())
            .find(|&&(t, _)| t <= now - HORIZON)
            .unwrap_or(&self.samples[0]);
        let seconds = (now - start.0) / 1000.0;
        if seconds > 0.0 {
            (end - start.1) / seconds
        } else {
            0.0
        }
    }
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
    fn speeding_up_reads_the_average_not_the_final_slope() {
        // x = t² / 100 over 0..80 ms: 64 px in 80 ms is 800 px/s, where the
        // final slope (which a least-squares quadratic reads) is 1600.
        let samples: Vec<_> = (0..=10)
            .map(|i| {
                let t = f64::from(i) * 8.0;
                (t, t * t / 100.0)
            })
            .collect();
        assert!((tracker(&samples).velocity(80.0) - 800.0).abs() < 1e-6);
    }

    #[test]
    fn braking_before_lifting_never_reads_backwards() {
        // 50 px in 16 ms, then 2 px more 8 ms later, released 1 ms after: a
        // least-squares quadratic reads about -2750 px/s here.
        let t = tracker(&[(0.0, 0.0), (16.0, 50.0), (24.0, 52.0)]);
        assert!((t.velocity(25.0) - 52.0 / 0.025).abs() < 1e-6);
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
        // 20 px per 8 ms: 2500 px/s.
        let samples: Vec<_> = (0..10)
            .map(|i| (f64::from(i) * 8.0, f64::from(i) * 20.0))
            .collect();
        assert!((tracker(&samples).velocity(72.0) - 2500.0).abs() < 1e-6);
        // A short pause before the release slows it: 180 px in 102 ms.
        assert!((tracker(&samples).velocity(102.0) - 180.0 / 0.102).abs() < 1e-6);
        // Released 50 ms after the last move: none.
        assert_eq!(tracker(&samples).velocity(122.0), 0.0);
    }

    #[test]
    fn a_lone_sample_has_no_velocity() {
        assert_eq!(tracker(&[(0.0, 5.0)]).velocity(0.0), 0.0);
        assert_eq!(tracker(&[(0.0, 5.0)]).velocity(8.0), 0.0);
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
    fn history_keeps_the_window_and_where_it_starts() {
        // A sample a millisecond: the window's 100 and the one before.
        let samples: Vec<_> = (0..300).map(|i| (f64::from(i), f64::from(i))).collect();
        let t = tracker(&samples);
        assert_eq!(t.samples.first(), Some(&(199.0, 199.0)));
        assert_eq!(t.samples.len(), 101);
        assert!((t.velocity(299.0) - 1000.0).abs() < 1e-6);
    }
}
