//! Job executors.
//!
//! Core work is split into share-nothing jobs; an [`Executor`] runs them.
//! Natively that's a [`ThreadPool`]; [`Inline`] runs jobs on the caller's
//! thread, for tests and hosts that drive the runtime themselves.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, Condvar, Mutex};

/// A unit of work.
pub type Job = Box<dyn FnOnce() + Send + 'static>;

pub trait Executor: Send + Sync {
    /// Runs `job`, as soon as any job [`spawn_ranked`](Self::spawn_ranked)
    /// at rank 0.
    fn spawn(&self, job: Job);

    /// Runs `job` once the jobs queued at lower ranks have started: a
    /// map's tiles nearer the view's centre rank lower. Equal ranks run in
    /// the order queued. Executors that queue nothing run it as
    /// [`spawn`](Self::spawn) does.
    fn spawn_ranked(&self, _rank: u32, job: Job) {
        self.spawn(job);
    }
}

/// Runs each job immediately on the calling thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct Inline;

impl Executor for Inline {
    fn spawn(&self, job: Job) {
        job();
    }
}

/// A fixed set of worker threads taking jobs by rank, lowest first, and
/// at one rank first in, first out. Dropped, its workers finish the jobs
/// queued and exit; they aren't joined, since a job stuck in a driver (a
/// hung pipeline compile) would hold the drop.
#[derive(Debug)]
pub struct ThreadPool {
    queue: Arc<Queue>,
}

#[derive(Debug, Default)]
struct Queue {
    jobs: Mutex<Queued>,
    ready: Condvar,
}

#[derive(Debug, Default)]
struct Queued {
    heap: BinaryHeap<Ranked>,
    /// Counts the jobs queued, for their order within a rank.
    next: u64,
    /// The pool was dropped: workers exit once the heap is empty.
    closed: bool,
}

/// A queued job: the heap's greatest is the lowest rank queued first.
struct Ranked {
    rank: u32,
    order: u64,
    job: Job,
}

impl std::fmt::Debug for Ranked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ranked")
            .field("rank", &self.rank)
            .field("order", &self.order)
            .finish_non_exhaustive()
    }
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        (other.rank, other.order).cmp(&(self.rank, self.order))
    }
}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ranked {}

impl ThreadPool {
    /// A pool of `threads` workers named `{name}-{i}`.
    pub fn new(name: &str, threads: usize) -> Self {
        let queue = Arc::new(Queue::default());
        for i in 0..threads.max(1) {
            let queue = Arc::clone(&queue);
            std::thread::Builder::new()
                .name(format!("{name}-{i}"))
                .spawn(move || queue.work())
                .expect("failed to spawn a worker thread");
        }
        Self { queue }
    }

    /// The plan's native default: one worker per core but one, at most six.
    pub fn default_size() -> usize {
        let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
        (cores - 1).clamp(1, 6)
    }
}

impl Queue {
    /// A worker's loop: the next job by rank, until the pool is dropped
    /// and nothing is left.
    fn work(&self) {
        loop {
            let mut jobs = self.jobs.lock().expect("job queue poisoned");
            let job = loop {
                if let Some(ranked) = jobs.heap.pop() {
                    break ranked.job;
                }
                if jobs.closed {
                    return;
                }
                jobs = self.ready.wait(jobs).expect("job queue poisoned");
            };
            drop(jobs);
            job();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.queue.jobs.lock().expect("job queue poisoned").closed = true;
        self.queue.ready.notify_all();
    }
}

impl Executor for ThreadPool {
    fn spawn(&self, job: Job) {
        self.spawn_ranked(0, job);
    }

    fn spawn_ranked(&self, rank: u32, job: Job) {
        let mut jobs = self.queue.jobs.lock().expect("job queue poisoned");
        let order = jobs.next;
        jobs.next += 1;
        jobs.heap.push(Ranked { rank, order, job });
        drop(jobs);
        self.queue.ready.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

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
        got.sort_unstable();
        assert_eq!(got, (0..100).collect::<Vec<_>>());
    }

    /// Jobs queued behind a busy worker start lowest rank first, and in
    /// the order queued within a rank.
    #[test]
    fn queued_jobs_start_by_rank() {
        let pool = ThreadPool::new("test", 1);
        let (hold, held) = channel::<()>();
        let (tx, rx) = channel();
        pool.spawn(Box::new(move || held.recv().unwrap()));
        for (rank, name) in [(5, "far"), (0, "centre"), (2, "near"), (0, "centre too")] {
            let tx = tx.clone();
            pool.spawn_ranked(rank, Box::new(move || tx.send(name).unwrap()));
        }
        hold.send(()).unwrap();
        drop(tx);
        drop(pool);
        let order: Vec<_> = rx.iter().collect();
        assert_eq!(order, ["centre", "centre too", "near", "far"]);
    }
}
