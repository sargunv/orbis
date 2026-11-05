//! Wall-clock time for CPU timings.

use std::time::Duration;

/// Time since an arbitrary origin.
pub(crate) fn now() -> Duration {
    static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    ORIGIN.get_or_init(std::time::Instant::now).elapsed()
}

/// The smallest step [`now`] takes, sampled over up to 50 ms.
pub(crate) fn resolution() -> Duration {
    let started = now();
    let mut last = started;
    let mut steps = 0;
    let mut smallest = Duration::MAX;
    while steps < 50 && last - started < Duration::from_millis(50) {
        let t = now();
        if t > last {
            smallest = smallest.min(t - last);
            steps += 1;
            last = t;
        }
    }
    smallest
}
