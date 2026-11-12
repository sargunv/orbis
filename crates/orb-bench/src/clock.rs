//! Wall-clock time for CPU timings, the process's CPU time, and the
//! instructions a thread retires.

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

/// Instructions this thread has retired so far, where the platform counts
/// them (macOS). Unlike CPU time, they barely move with other work on the
/// machine or with the cores the scheduler picks, so they compare two
/// builds on a loaded machine.
pub(crate) fn instructions() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        #[allow(unsafe_code)]
        unsafe extern "C" {
            /// libsystem_kernel's per-thread counters: kind 1 writes
            /// instructions, then cycles.
            fn thread_selfcounts(kind: i32, counts: *mut u64, size: usize) -> i32;
        }
        let mut counts = [0u64; 2];
        // SAFETY: `counts` is writable for the size passed.
        #[allow(unsafe_code)]
        let status =
            unsafe { thread_selfcounts(1, counts.as_mut_ptr(), std::mem::size_of_val(&counts)) };
        (status == 0).then_some(counts[0])
    }
    #[cfg(not(target_os = "macos"))]
    None
}
