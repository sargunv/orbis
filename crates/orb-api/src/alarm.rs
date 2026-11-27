//! A map's alarm: a wake at the time it last set.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use orb_host_native::NativeClock;

/// Wakes a map once at the time it last set: when the data a tile it
/// draws shows expires or a failed load is due to be tried again
/// ([`orb_runtime::FrameReport::wake_at`]), so that a host drawing only
/// when woken draws the frame that loads it. Its
/// thread starts with the first time set and ends with the alarm.
///
/// It waits on the monotonic clock, which on macOS and Linux doesn't
/// count the system's sleep, so an expiry rings that much late on an idle
/// map; the first frame after a wake (input, a redraw) measures the wall
/// clock again and brings it forward.
pub(crate) struct Alarm {
    /// The map's services' clock, which its times are read on.
    clock: NativeClock,
    shared: Arc<Shared>,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}

#[derive(Default)]
struct State {
    due: Option<Instant>,
    closed: bool,
}

impl Alarm {
    pub(crate) fn new(clock: NativeClock, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            clock,
            shared: Arc::default(),
            wake: Some(wake),
        }
    }

    /// Wakes the map at `at` (ms on its clock) instead of when set last,
    /// or not at all with `None` or a time past what the clock holds. A
    /// time already due wakes it at once; one set again as it was, as
    /// each frame sets the same expiry, changes nothing.
    pub(crate) fn set(&mut self, at: Option<f64>) {
        let due = at.and_then(|at| self.clock.instant(at));
        {
            let mut state = lock(&self.shared.state);
            if state.due == due {
                return;
            }
            state.due = due;
        }
        self.shared.changed.notify_one();
        if due.is_some()
            && let Some(wake) = self.wake.take()
        {
            let shared = Arc::clone(&self.shared);
            let started = std::thread::Builder::new()
                .name("orbis alarm".into())
                .spawn(move || ring(&shared, &*wake));
            // Without a thread the map wakes for other reasons alone.
            drop(started);
        }
    }
}

impl Drop for Alarm {
    fn drop(&mut self) {
        lock(&self.shared.state).closed = true;
        self.shared.changed.notify_one();
    }
}

/// Calls `wake` at each time due until the alarm closes.
fn ring(shared: &Shared, wake: &(dyn Fn() + Send + Sync)) {
    let mut state = lock(&shared.state);
    loop {
        if state.closed {
            return;
        }
        let now = Instant::now();
        state = match state.due {
            None => (shared.changed.wait(state)).unwrap_or_else(PoisonError::into_inner),
            Some(due) if due > now => {
                let waited = shared.changed.wait_timeout(state, due - now);
                waited.unwrap_or_else(PoisonError::into_inner).0
            }
            Some(_) => {
                state.due = None;
                drop(state);
                wake();
                lock(&shared.state)
            }
        };
    }
}

fn lock(state: &Mutex<State>) -> MutexGuard<'_, State> {
    state.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    fn alarm() -> (NativeClock, Alarm, mpsc::Receiver<Instant>) {
        let clock = NativeClock::default();
        let (rang, rings) = mpsc::channel();
        let alarm = Alarm::new(
            clock,
            Arc::new(move || {
                let _ = rang.send(Instant::now());
            }),
        );
        (clock, alarm, rings)
    }

    fn ms(clock: NativeClock) -> f64 {
        orb_host::Clock::now(&clock)
    }

    #[test]
    fn rings_once_at_the_time_last_set() {
        let (clock, mut alarm, rings) = alarm();
        alarm.set(Some(ms(clock) + 60_000.0));
        let at = ms(clock) + 20.0;
        alarm.set(Some(at));
        let rang = rings.recv_timeout(Duration::from_secs(60)).expect("rang");
        assert!(rang >= clock.instant(at).unwrap(), "rang early");
        assert!(
            rings.recv_timeout(Duration::from_millis(100)).is_err(),
            "rang twice"
        );
    }

    #[test]
    fn a_time_set_again_as_it_was_stands() {
        let (clock, mut alarm, rings) = alarm();
        let at = ms(clock) + 50.0;
        alarm.set(Some(at));
        // Each frame till then sets it again.
        while rings.try_recv().is_err() && ms(clock) < at + 5_000.0 {
            alarm.set(Some(at));
        }
        assert!(ms(clock) >= at, "rang early");
        // The frame it rang for sets it again, still due: it rings again.
        alarm.set(Some(at));
        rings
            .recv_timeout(Duration::from_secs(60))
            .expect("rang again");
    }

    #[test]
    fn unset_before_it_rings() {
        let (clock, mut alarm, rings) = alarm();
        // Unset long before it's due.
        alarm.set(Some(ms(clock) + 250.0));
        alarm.set(None);
        alarm.set(Some(f64::MAX));
        assert!(
            rings.recv_timeout(Duration::from_millis(500)).is_err(),
            "rang after it was unset"
        );
    }
}
