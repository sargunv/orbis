//! Job executors (plan §7.3).
//!
//! Core work is split into share-nothing jobs; an [`Executor`] runs them.
//! Natively that's a [`ThreadPool`]; [`Inline`] runs jobs on the caller's
//! thread, for tests and hosts that drive the runtime themselves.

#![forbid(unsafe_code)]

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

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

/// A fixed set of worker threads taking jobs first in, first out.
#[derive(Debug)]
pub struct ThreadPool {
    sender: Option<Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// A pool of `threads` workers named `{name}-{i}`.
    pub fn new(name: &str, threads: usize) -> Self {
        let (sender, receiver) = channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let workers = (0..threads.max(1))
            .map(|i| {
                let receiver = Arc::clone(&receiver);
                std::thread::Builder::new()
                    .name(format!("{name}-{i}"))
                    .spawn(move || worker(&receiver))
                    .expect("failed to spawn a worker thread")
            })
            .collect();
        Self {
            sender: Some(sender),
            workers,
        }
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
        self.sender
            .as_ref()
            .expect("pool is running")
            .send(job)
            .expect("workers exited early");
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
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
        let mut got: Vec<i32> = rx.try_iter().collect();
        got.sort();
        assert_eq!(got, (0..100).collect::<Vec<_>>());
    }
}
