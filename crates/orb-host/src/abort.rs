//! Giving fetches up: [`Abort`].

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use crate::{FetchDone, FetchError};

type Listener = Box<dyn FnOnce() + Send>;

#[derive(Default)]
struct State {
    aborted: bool,
    /// Numbers the listeners, for [`Abort::forget`].
    next: u64,
    listeners: Vec<(u64, Listener)>,
}

/// A listener [`Abort::on_abort`] added, which [`Abort::forget`] takes
/// off again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Listening(Option<u64>);

/// Gives up the fetches it is passed with, as a browser's `AbortSignal`
/// does: a requester makes one for a load it may abandon, keeps a clone,
/// and calls [`abort`](Self::abort) once it no longer wants the answer.
/// [`NEVER`](Self::NEVER) is for loads nobody gives up.
#[derive(Clone)]
pub struct Abort(Option<Arc<Mutex<State>>>);

impl fmt::Debug for Abort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Abort").field(&self.is_aborted()).finish()
    }
}

impl Abort {
    /// For loads nobody gives up: it never fires.
    pub const NEVER: Self = Self(None);

    /// One that fires when [`abort`](Self::abort) is called on it or a clone.
    pub fn new() -> Self {
        Self(Some(Arc::default()))
    }

    fn state(&self) -> Option<MutexGuard<'_, State>> {
        let state = self.0.as_ref()?;
        Some(
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// Gives the loads up: runs each listener once, on this thread. Later
    /// calls do nothing.
    pub fn abort(&self) {
        let listeners = match self.state() {
            Some(mut state) if !state.aborted => {
                state.aborted = true;
                std::mem::take(&mut state.listeners)
            }
            _ => return,
        };
        for (_, listener) in listeners {
            listener();
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.state().is_some_and(|s| s.aborted)
    }

    /// Runs `listener` when this fires, or now if it has. Dropped unrun
    /// when every clone is, unfired, or when [`forget`](Self::forget)
    /// takes it off: a load that ends unaborted takes off what it added,
    /// so that one abort can serve many loads (an offline download's).
    pub fn on_abort(&self, listener: impl FnOnce() + Send + 'static) -> Listening {
        let Some(mut state) = self.state() else {
            return Listening(None);
        };
        if state.aborted {
            drop(state);
            listener();
            return Listening(None);
        }
        let number = state.next;
        state.next += 1;
        state.listeners.push((number, Box::new(listener)));
        Listening(Some(number))
    }

    /// Takes a listener off unrun, if it hasn't run.
    pub fn forget(&self, listening: Listening) {
        let (Some(number), Some(mut state)) = (listening.0, self.state()) else {
            return;
        };
        if let Some(i) = state.listeners.iter().position(|(n, _)| *n == number) {
            // Dropped once the lock is let go: it may hold a clone.
            let (_, listener) = state.listeners.swap_remove(i);
            drop(state);
            drop(listener);
        }
    }

    /// `done`, answered [`FetchError::Aborted`] as soon as this fires
    /// unless the fetch answered first: what comes after the first answer
    /// goes nowhere. An answer first takes the abort's listener off.
    pub fn answer(&self, done: FetchDone) -> FetchDone {
        if self.0.is_none() {
            return done;
        }
        let slot = Arc::new(Mutex::new(Some(done)));
        let take = |slot: &Mutex<Option<FetchDone>>| {
            slot.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
        };
        let aborted = Arc::clone(&slot);
        let listening = self.on_abort(move || {
            if let Some(done) = take(&aborted) {
                done(Err(FetchError::Aborted));
            }
        });
        let abort = self.clone();
        Box::new(move |result| {
            if let Some(done) = take(&slot) {
                abort.forget(listening);
                done(result);
            }
        })
    }

    /// A future that completes when this fires: for hosts that stop their
    /// loads from an async task (the browser's `AbortController`).
    pub fn aborted(&self) -> Aborted {
        Aborted {
            abort: self.clone(),
            waker: None,
            listening: Listening(None),
        }
    }

    /// Whether `other` is this one or a clone of it.
    pub fn same(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Default for Abort {
    fn default() -> Self {
        Self::new()
    }
}

/// [`Abort::aborted`]'s future.
#[derive(Debug)]
pub struct Aborted {
    abort: Abort,
    /// The waker the abort wakes, once it listens.
    waker: Option<Arc<Mutex<Option<Waker>>>>,
    /// Its listener, taken off when the future is dropped unfired.
    listening: Listening,
}

impl Drop for Aborted {
    fn drop(&mut self) {
        self.abort.forget(self.listening);
    }
}

impl Future for Aborted {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.abort.is_aborted() {
            return Poll::Ready(());
        }
        if self.waker.is_none() {
            let slot: Arc<Mutex<Option<Waker>>> = Arc::default();
            let woken = Arc::clone(&slot);
            self.listening = self.abort.on_abort(move || {
                if let Some(waker) = woken.lock().ok().and_then(|mut w| w.take()) {
                    waker.wake();
                }
            });
            self.waker = Some(slot);
        }
        let slot = self.waker.as_ref().expect("listening");
        *slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cx.waker().clone());
        // It may have fired before the waker was in place.
        if self.abort.is_aborted() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn an_abort_answers_once_and_drops_the_late_answer() {
        let abort = Abort::new();
        let answers = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&answers);
        let done = abort.answer(Box::new(move |r| seen.lock().unwrap().push(r)));
        abort.abort();
        abort.abort();
        done(Ok(vec![1]));
        assert_eq!(*answers.lock().unwrap(), vec![Err(FetchError::Aborted)]);
    }

    #[test]
    fn an_answer_before_the_abort_stands() {
        let abort = Abort::new();
        let answers = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&answers);
        let done = abort.answer(Box::new(move |r| seen.lock().unwrap().push(r)));
        done(Ok(vec![1]));
        abort.abort();
        assert_eq!(*answers.lock().unwrap(), vec![Ok(vec![1])]);
    }

    /// Loads that end take their listeners off: one abort serving many
    /// loads holds none for those that ended.
    #[test]
    fn ended_loads_leave_no_listener() {
        let abort = Abort::new();
        for _ in 0..3 {
            abort.answer(Box::new(|_| {}))(Ok(Vec::new()));
        }
        let mut future = abort.aborted();
        let mut cx = Context::from_waker(Waker::noop());
        assert!(Pin::new(&mut future).poll(&mut cx).is_pending());
        drop(future);
        assert!(abort.state().unwrap().listeners.is_empty());
        // One still waiting keeps its listener, and is answered.
        let answers = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&answers);
        let _done = abort.answer(Box::new(move |r| seen.lock().unwrap().push(r)));
        assert_eq!(abort.state().unwrap().listeners.len(), 1);
        abort.abort();
        assert_eq!(*answers.lock().unwrap(), vec![Err(FetchError::Aborted)]);
    }

    #[test]
    fn listeners_run_once_and_late_ones_at_once() {
        let abort = Abort::new();
        let runs = Arc::new(AtomicUsize::new(0));
        let r = Arc::clone(&runs);
        abort.clone().on_abort(move || {
            r.fetch_add(1, Ordering::Relaxed);
        });
        abort.abort();
        let r = Arc::clone(&runs);
        abort.on_abort(move || {
            r.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert!(!Abort::NEVER.is_aborted());
        Abort::NEVER.abort();
        assert!(!Abort::NEVER.is_aborted());
    }

    #[test]
    fn the_future_wakes_on_abort() {
        struct Count(AtomicUsize);
        impl std::task::Wake for Count {
            fn wake(self: Arc<Self>) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let count = Arc::new(Count(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&count));
        let mut cx = Context::from_waker(&waker);
        let abort = Abort::new();
        let mut future = abort.aborted();
        assert!(Pin::new(&mut future).poll(&mut cx).is_pending());
        assert!(Pin::new(&mut future).poll(&mut cx).is_pending());
        abort.abort();
        assert_eq!(count.0.load(Ordering::Relaxed), 1);
        assert!(Pin::new(&mut future).poll(&mut cx).is_ready());
    }
}
