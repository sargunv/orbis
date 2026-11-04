//! The placement lane (plan §7.1 [P], decision 0009): `orb-symbol`'s
//! [`Placer`] off the scene thread. The scene hands it buckets as their
//! tiles become resident and camera snapshots as the view changes; it
//! answers each snapshot with the state words that changed.

use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};

use orb_symbol::{Committed, Frame, PlacementTables, Placer};

/// Called once with a placement's commit, on any thread.
pub type PlaceDone = Box<dyn FnOnce(Committed) + Send + 'static>;

/// Where placement runs.
pub trait PlacementLane: Send + Sync {
    /// Adds a resident bucket's tables, or replaces them.
    fn ingest(&self, bucket: u64, tables: Arc<PlacementTables>);
    fn remove(&self, bucket: u64);
    /// Places against `frame`; buckets ingested before the call are seen.
    fn place(&self, frame: Frame, done: PlaceDone);
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
        Message::Place(frame, done) => done(placer.place(&frame)),
    }
}

/// Placement on a thread of its own: the native lane.
#[derive(Debug)]
pub struct ThreadPlacement {
    sender: Mutex<Sender<Message>>,
}

impl ThreadPlacement {
    pub fn new() -> Arc<Self> {
        let (sender, receiver) = channel::<Message>();
        std::thread::Builder::new()
            .name("placement".into())
            .spawn(move || {
                let mut placer = Placer::default();
                while let Ok(message) = receiver.recv() {
                    run(&mut placer, message);
                }
            })
            .expect("failed to spawn the placement thread");
        Arc::new(Self {
            sender: Mutex::new(sender),
        })
    }

    fn send(&self, message: Message) {
        // The thread lives as long as the sender.
        let _ = self.sender.lock().expect("lane poisoned").send(message);
    }
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
        done(committed);
    }
}
