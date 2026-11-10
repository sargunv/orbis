//! The placement lane: `orb-symbol`'s
//! [`Placer`] off the scene thread. The scene hands it buckets as their
//! tiles become resident and camera snapshots as the view changes; it
//! answers each snapshot with the state words that changed.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};

use orb_symbol::{Committed, Frame, PlacementTables, Placer};

/// Called once with a placement's commit, or why the lane couldn't place,
/// on any thread.
pub type PlaceDone = Box<dyn FnOnce(Result<Committed, String>) + Send + 'static>;

/// Where placement runs.
pub trait PlacementLane: Send + Sync {
    /// Adds a resident bucket's tables, or replaces them.
    fn ingest(&self, bucket: u64, tables: Arc<PlacementTables>);
    fn remove(&self, bucket: u64);
    /// Places against `frame`; buckets ingested before the call are seen.
    fn place(&self, frame: Frame, done: PlaceDone);
    /// Whether the lane started afresh, without the buckets it had, since
    /// last asked: its placer failed (a trap, a panic) and was replaced.
    /// The map ingests every bucket again before it places.
    fn restarted(&self) -> bool {
        false
    }
}

enum Message {
    Ingest(u64, Arc<PlacementTables>),
    Remove(u64),
    Place(Box<Frame>, PlaceDone),
}

fn run(placer: &mut Placer, message: Message) {
    match message {
        Message::Ingest(bucket, tables) => placer.ingest(bucket, tables),
        Message::Remove(bucket) => placer.remove(bucket),
        Message::Place(frame, done) => done(Ok(placer.place(&frame))),
    }
}

/// Placement on a thread of its own: the native lane. A placement that
/// panics fails, and the thread goes on with a fresh placer; a thread that
/// died is replaced. Either way the lane has `restarted`.
#[derive(Debug)]
pub(crate) struct ThreadPlacement {
    sender: Mutex<Sender<Message>>,
    flags: Arc<Flags>,
}

/// What the lane and its thread share.
#[derive(Debug, Default)]
struct Flags {
    /// The placer started afresh.
    restarted: AtomicBool,
    /// A test's trap in this lane's next placement.
    #[cfg(test)]
    trap: AtomicBool,
}

impl ThreadPlacement {
    pub(crate) fn new() -> Arc<Self> {
        let flags = Arc::new(Flags::default());
        Arc::new(Self {
            sender: Mutex::new(spawn(Arc::clone(&flags))),
            flags,
        })
    }

    /// Has this lane's next placement panic, as a bug in it would.
    #[cfg(test)]
    pub(crate) fn trap_next(&self) {
        self.flags.trap.store(true, Ordering::Release);
    }

    fn send(&self, message: Message) {
        let mut sender = self.sender.lock().expect("lane poisoned");
        if let Err(std::sync::mpsc::SendError(message)) = sender.send(message) {
            *sender = spawn(Arc::clone(&self.flags));
            self.flags.restarted.store(true, Ordering::Release);
            // The map sends its buckets again; a placement asks again.
            if let Message::Place(_, done) = message {
                done(Err("the placement thread was lost".into()));
            }
        }
    }
}

/// The placement thread: a panic leaves a fresh placer.
fn spawn(flags: Arc<Flags>) -> Sender<Message> {
    let (sender, receiver) = channel::<Message>();
    std::thread::Builder::new()
        .name("placement".into())
        .spawn(move || {
            let mut placer = Placer::default();
            let failed = |placer: &mut Placer| {
                *placer = Placer::default();
                flags.restarted.store(true, Ordering::Release);
            };
            while let Ok(message) = receiver.recv() {
                let Message::Place(frame, done) = message else {
                    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run(&mut placer, message);
                    }));
                    if ran.is_err() {
                        failed(&mut placer);
                    }
                    continue;
                };
                let placed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    #[cfg(test)]
                    if flags.trap.swap(false, Ordering::AcqRel) {
                        panic!("an injected trap");
                    }
                    placer.place(&frame)
                }));
                match placed {
                    Ok(committed) => done(Ok(committed)),
                    Err(_) => {
                        failed(&mut placer);
                        done(Err("placement failed; the lane starts afresh".into()));
                    }
                }
            }
        })
        .expect("failed to spawn the placement thread");
    sender
}

impl std::fmt::Debug for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("placement message")
    }
}

impl PlacementLane for ThreadPlacement {
    fn ingest(&self, bucket: u64, tables: Arc<PlacementTables>) {
        self.send(Message::Ingest(bucket, tables));
    }

    fn remove(&self, bucket: u64) {
        self.send(Message::Remove(bucket));
    }

    fn place(&self, frame: Frame, done: PlaceDone) {
        self.send(Message::Place(Box::new(frame), done));
    }

    fn restarted(&self) -> bool {
        self.flags.restarted.swap(false, Ordering::Acquire)
    }
}

/// Placement on the caller's thread: hosts without threads, and tests.
#[derive(Debug, Default)]
pub struct InlinePlacement {
    placer: Mutex<Placer>,
}

impl PlacementLane for InlinePlacement {
    fn ingest(&self, bucket: u64, tables: Arc<PlacementTables>) {
        run(
            &mut self.placer.lock().expect("lane poisoned"),
            Message::Ingest(bucket, tables),
        );
    }

    fn remove(&self, bucket: u64) {
        run(
            &mut self.placer.lock().expect("lane poisoned"),
            Message::Remove(bucket),
        );
    }

    fn place(&self, frame: Frame, done: PlaceDone) {
        let committed = self.placer.lock().expect("lane poisoned").place(&frame);
        done(Ok(committed));
    }
}
