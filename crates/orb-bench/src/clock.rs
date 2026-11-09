//! Wall-clock time for CPU timings, and the process's CPU time.

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

/// The process's CPU time so far, every thread's.
pub(crate) fn cpu() -> Option<Duration> {
    #[cfg(unix)]
    {
        let t = rustix::time::clock_gettime(rustix::time::ClockId::ProcessCPUTime);
        Some(Duration::new(t.tv_sec as u64, t.tv_nsec as u32))
    }
    #[cfg(not(unix))]
    None
}

/// This thread's CPU time so far where the platform counts it (time the
/// scheduler gives other work doesn't count), else [`now`].
pub(crate) fn thread() -> Duration {
    #[cfg(unix)]
    {
        let t = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
        Duration::new(t.tv_sec as u64, t.tv_nsec as u32)
    }
    #[cfg(not(unix))]
    now()
}
