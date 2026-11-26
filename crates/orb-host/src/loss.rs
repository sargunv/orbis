//! When a lost GPU gets a new device and when the map gives up, on every
//! platform: [`LossRuns`].

use std::fmt;
use std::time::Duration;

/// A device that ran this long before it was lost ends a run of losses:
/// the next loss is the first of a new run.
const STABLE_S: u32 = 30;

/// How long after each loss of a run a new device is made: the first at
/// once (an app switch, a driver reset), later ones after a pause. A loss
/// past the last isn't answered: a driver that loses every device it gives
/// (Adreno X1's Vulkan lost one a second, #141) would otherwise reset the
/// map forever.
const BACKOFF: [Duration; 3] = [
    Duration::ZERO,
    Duration::from_millis(250),
    Duration::from_secs(1),
];

/// Device losses in a row, on a clock of milliseconds such as
/// [`Clock::now`](crate::Clock::now)'s.
#[derive(Clone, Copy, Debug)]
pub struct LossRuns {
    /// When the device was made.
    made: f64,
    /// Losses in the current run.
    losses: u32,
}

impl LossRuns {
    /// A device made at `now`.
    pub fn new(now: f64) -> Self {
        Self {
            made: now,
            losses: 0,
        }
    }

    /// Another device was made at `now`.
    pub fn made(&mut self, now: f64) {
        self.made = now;
    }

    /// The device was lost at `now`: how long to wait before making
    /// another, or, past the run's backoff, that no other is made.
    pub fn lost(&mut self, now: f64) -> Result<Duration, GaveUp> {
        self.losses = if now - self.made < f64::from(STABLE_S) * 1000.0 {
            self.losses + 1
        } else {
            1
        };
        BACKOFF
            .get(self.losses as usize - 1)
            .copied()
            .ok_or(GaveUp {
                losses: self.losses,
            })
    }
}

/// No device is made after a loss: too many came in a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GaveUp {
    /// Losses in the run.
    pub losses: u32,
}

impl fmt::Display for GaveUp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the GPU was lost {} times in a row, each device within {} s of the last, so no \
             other is made",
            self.losses, STABLE_S
        )
    }
}

impl std::error::Error for GaveUp {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_backs_off_then_gives_up() {
        let mut runs = LossRuns::new(0.0);
        assert_eq!(runs.lost(1_000.0), Ok(Duration::ZERO));
        runs.made(1_000.0);
        assert_eq!(runs.lost(2_000.0), Ok(Duration::from_millis(250)));
        runs.made(2_250.0);
        assert_eq!(runs.lost(3_000.0), Ok(Duration::from_secs(1)));
        runs.made(4_000.0);
        let gave_up = runs.lost(5_000.0).unwrap_err();
        assert_eq!(gave_up, GaveUp { losses: 4 });
        assert_eq!(
            gave_up.to_string(),
            "the GPU was lost 4 times in a row, each device within 30 s of the last, so no \
             other is made"
        );
    }

    #[test]
    fn a_device_that_lasted_ends_the_run() {
        let mut runs = LossRuns::new(0.0);
        for at in [1_000.0, 2_000.0, 3_000.0] {
            runs.lost(at).unwrap();
            runs.made(at);
        }
        // The last device lasted 30 s: its loss is the first of a run.
        assert_eq!(runs.lost(33_000.0), Ok(Duration::ZERO));
    }
}
