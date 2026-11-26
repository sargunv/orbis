//! A map's alarm: a wake at the time it last set.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// Wakes a map once at the time it last set: when the data a tile it
/// draws shows expires ([`orb_runtime::FrameReport::wake_in`]), so that a
/// host drawing only when woken draws the frame that loads it again. Its
/// thread starts with the first time set and ends with the alarm.
pub(crate) struct Alarm {
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

/// How far a time set may move before the alarm's thread hears of it:
/// each frame sets the same expiry again, measured anew.
const SLACK: Duration = Duration::from_millis(10);

impl Alarm {
    pub(crate) fn new(wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        Self {
            shared: Arc::default(),
            wake: Some(wake),
        }
    }

    /// Wakes the map `after` from now instead of when set last, or not
    /// at all with `None`.
    pub(crate) fn set(&mut self, after: Option<Duration>) {
        let due = after.map(|after| Instant::now() + after);
        {
            let mut state = lock(&self.shared.state);
            let near = match (state.due, due) {
                (Some(old), Some(new)) => old.max(new) - old.min(new) < SLACK,
                (old, new) => old == new,
            };
            if near {
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
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn rings_once_at_the_time_last_set() {
        let rings = Arc::new(AtomicU32::new(0));
        let counted = Arc::clone(&rings);
        let mut alarm = Alarm::new(Arc::new(move || {
            counted.fetch_add(1, Ordering::SeqCst);
        }));
        alarm.set(Some(Duration::from_secs(60)));
        alarm.set(Some(Duration::from_millis(20)));
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(rings.load(Ordering::SeqCst), 1);
        alarm.set(Some(Duration::from_millis(20)));
        alarm.set(None);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(rings.load(Ordering::SeqCst), 1, "unset before it rang");
    }
}
