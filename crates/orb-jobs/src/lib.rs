//! Job executors (plan §7.3).
//!
//! Core work is split into share-nothing jobs; an [`Executor`] runs them.
//! Natively that's a [`ThreadPool`]; [`Inline`] runs jobs on the caller's
//! thread, for tests and hosts that drive the runtime themselves.

#![forbid(unsafe_code)]

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

/// A unit of work.
pub type Job = Box<dyn FnOnce() + Send + 'static>;

pub trait Executor: Send + Sync {
    fn spawn(&self, job: Job);
}

/// Runs each job immediately on the calling thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct Inline;

impl Executor for Inline {
    fn spawn(&self, job: Job) {
        job();
    }
}

/// A fixed set of worker threads taking jobs first in, first out. Dropped,
/// its workers finish the jobs queued and exit; they aren't joined, since a
/// job stuck in a driver (a hung pipeline compile) would hold the drop.
#[derive(Debug)]
pub struct ThreadPool {
    sender: Sender<Job>,
}

impl ThreadPool {
    /// A pool of `threads` workers named `{name}-{i}`.
    pub fn new(name: &str, threads: usize) -> Self {
        let (sender, receiver) = channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        for i in 0..threads.max(1) {
            let receiver = Arc::clone(&receiver);
            std::thread::Builder::new()
                .name(format!("{name}-{i}"))
                .spawn(move || worker(&receiver))
                .expect("failed to spawn a worker thread");
        }
        Self { sender }
    }

    /// The plan's native default: one worker per core but one, at most six.
    pub fn default_size() -> usize {
        let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
        (cores - 1).clamp(1, 6)
    }
}

fn worker(receiver: &Mutex<Receiver<Job>>) {
    loop {
        let job = receiver.lock().expect("job queue poisoned").recv();
        match job {
            Ok(job) => job(),
            Err(_) => return,
        }
    }
}

impl Executor for ThreadPool {
    fn spawn(&self, job: Job) {
        self.sender.send(job).expect("workers exited early");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_runs_every_job() {
        let (tx, rx) = channel();
        {
            let pool = ThreadPool::new("test", 3);
            for i in 0..100 {
                let tx = tx.clone();
                pool.spawn(Box::new(move || tx.send(i).unwrap()));
            }
        }
        drop(tx);
        let mut got: Vec<i32> = rx.iter().collect();
        got.sort();
        assert_eq!(got, (0..100).collect::<Vec<_>>());
    }
}
