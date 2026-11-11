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
    listeners: Vec<Listener>,
}

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
        for listener in listeners {
            listener();
        }
    }

    pub fn is_aborted(&self) -> bool {
        self.state().is_some_and(|s| s.aborted)
    }

    /// Runs `listener` when this fires, or now if it has. Dropped unrun
    /// when every clone is, unfired.
    pub fn on_abort(&self, listener: impl FnOnce() + Send + 'static) {
        let Some(mut state) = self.state() else {
            return;
        };
        if state.aborted {
            drop(state);
            listener();
        } else {
            state.listeners.push(Box::new(listener));
        }
    }

    /// `done`, answered [`FetchError::Aborted`] as soon as this fires
    /// unless the fetch answered first: what comes after the first answer
    /// goes nowhere.
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
        self.on_abort(move || {
            if let Some(done) = take(&aborted) {
                done(Err(FetchError::Aborted));
            }
        });
        Box::new(move |result| {
            if let Some(done) = take(&slot) {
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
}

impl Future for Aborted {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.abort.is_aborted() {
            return Poll::Ready(());
        }
        let abort = self.abort.clone();
        let slot = self.waker.get_or_insert_with(|| {
            let slot: Arc<Mutex<Option<Waker>>> = Arc::default();
            let woken = Arc::clone(&slot);
            abort.on_abort(move || {
                if let Some(waker) = woken.lock().ok().and_then(|mut w| w.take()) {
                    waker.wake();
                }
            });
            slot
        });
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
