//! Tile fetches taking turns: at most a limit of them at once, and the
//! waiting ones started lowest rank first. A load's rank is its map's
//! fetch order ([`fetch_rank`](crate::frame::fetch_rank)), which the map
//! sets again every frame for the loads still waiting
//! ([`Turns::reorder`]), so that the view the camera is at loads first
//! whichever frame or source asked. A load given up before its turn never
//! starts; one given up in its turn is aborted. A load beside the view
//! ([`YIELDING`]) runs in a turn no load ranked before it waits for: one
//! that waits takes its turn, and the load beside the view is aborted, so
//! loading on beside the view never holds up the view. Natively each
//! map's job host holds its turns; in a browser the page holds them for
//! every job worker, so that one order holds across them all.

use std::sync::{Arc, Mutex};

use orb_crs::TileIndex;
use orb_host::Abort;

/// A load its map can give up: its style, source and tile.
pub type LoadKey = (u64, u32, TileIndex);

/// Ranks from here on are loads beside the view, which load on after the
/// view leaves them: each yields its turn to a waiting load ranked before
/// it.
pub const YIELDING: u64 = crate::frame::Tier::Beside.first_rank();

/// Tile fetches started, aborted once started, and completed, since a
/// map's jobs began: [`FrameReport::fetches`](crate::FrameReport::fetches)
/// counts them a frame at a time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FetchCounts {
    pub started: u64,
    pub aborted: u64,
    pub completed: u64,
}

impl std::ops::Sub for FetchCounts {
    type Output = Self;

    fn sub(self, before: Self) -> Self {
        Self {
            started: self.started - before.started,
            aborted: self.aborted - before.aborted,
            completed: self.completed - before.completed,
        }
    }
}

impl std::ops::AddAssign for FetchCounts {
    fn add_assign(&mut self, other: Self) {
        self.started += other.started;
        self.aborted += other.aborted;
        self.completed += other.completed;
    }
}

/// Starts a fetch in its turn, or, given none, answers it cancelled.
pub type Start = Box<dyn FnOnce(Option<Turn>) + Send>;

/// Fetches taking turns: see the module.
#[derive(Clone)]
pub struct Turns(Arc<Mutex<State>>);

impl std::fmt::Debug for Turns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Turns").finish_non_exhaustive()
    }
}

struct State {
    limit: usize,
    counts: FetchCounts,
    running: Vec<Running>,
    waiting: Vec<Waiting>,
    /// Counts the fetches queued, for their order within a rank.
    queued: u64,
    /// A call starts waiting fetches: one whose fetch completes as it
    /// starts hands its turn on without starting more itself.
    starting: bool,
}

/// A fetch in its turn: its load, its abort and its latest rank.
struct Running {
    key: Option<LoadKey>,
    abort: Abort,
    rank: u64,
}

/// A fetch waiting for its turn.
struct Waiting {
    rank: u64,
    order: u64,
    key: Option<LoadKey>,
    abort: Abort,
    start: Start,
}

/// A running fetch's turn, and the abort its fetch takes: dropped, the
/// next waiting fetch starts.
pub struct Turn {
    turns: Arc<Mutex<State>>,
    abort: Abort,
    rank: u64,
}

impl std::fmt::Debug for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Turn")
            .field("rank", &self.rank)
            .finish_non_exhaustive()
    }
}

impl Turn {
    /// What gives the fetch up: [`Turns::cancel`] fires it.
    pub fn abort(&self) -> &Abort {
        &self.abort
    }

    /// The fetch's rank when its turn came.
    pub fn rank(&self) -> u64 {
        self.rank
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        let mut state = self.turns.lock().expect("turns poisoned");
        let running = &mut state.running;
        if let Some(i) = running.iter().position(|r| r.abort.same(&self.abort)) {
            running.swap_remove(i);
        }
        if !self.abort.is_aborted() {
            state.counts.completed += 1;
        }
        drop(state);
        start_waiting(&self.turns);
    }
}

/// Starts the lowest ranked waiting fetches while turns are free, or held
/// by loads beside the view ranked after them, unless another call
/// already does.
fn start_waiting(turns: &Arc<Mutex<State>>) {
    let mut state = turns.lock().expect("turns poisoned");
    if state.starting {
        return;
    }
    state.starting = true;
    loop {
        let next = (state.waiting.iter().enumerate())
            .min_by_key(|(_, w)| (w.rank, w.order))
            .map(|(i, w)| (i, w.rank));
        let Some((next, rank)) = next else {
            break;
        };
        if state.running.len() >= state.limit {
            let yielding = (state.running.iter().enumerate())
                .filter(|(_, r)| r.rank >= YIELDING && r.rank > rank)
                .max_by_key(|(_, r)| r.rank)
                .map(|(i, _)| i);
            let Some(yielding) = yielding else {
                break;
            };
            // Its turn ends here; its fetch answers aborted, which may
            // end it at once.
            let running = state.running.swap_remove(yielding);
            state.counts.aborted += 1;
            drop(state);
            running.abort.abort();
            state = turns.lock().expect("turns poisoned");
            continue;
        }
        let Waiting {
            rank,
            key,
            abort,
            start,
            ..
        } = state.waiting.swap_remove(next);
        state.running.push(Running {
            key,
            abort: abort.clone(),
            rank,
        });
        state.counts.started += 1;
        drop(state);
        start(Some(Turn {
            turns: Arc::clone(turns),
            abort,
            rank,
        }));
        state = turns.lock().expect("turns poisoned");
    }
    state.starting = false;
}

impl Turns {
    /// Turns for `limit` fetches at once.
    pub fn new(limit: usize) -> Self {
        Self(Arc::new(Mutex::new(State {
            limit: limit.max(1),
            counts: FetchCounts::default(),
            running: Vec::new(),
            waiting: Vec::new(),
            queued: 0,
            starting: false,
        })))
    }

    /// Runs `start` in its turn: at once while fewer than the limit run,
    /// or once one ends and none ranked lower waits. A load (`key`) its
    /// map may give up, reorder and count; a fetch with none (a relayout's
    /// data, say) just takes a turn.
    pub fn wait(&self, rank: u64, key: Option<LoadKey>, start: Start) {
        let mut state = self.0.lock().expect("turns poisoned");
        let order = state.queued;
        state.queued += 1;
        state.waiting.push(Waiting {
            rank,
            order,
            key,
            abort: Abort::new(),
            start,
        });
        drop(state);
        start_waiting(&self.0);
    }

    /// Ranks the loads again, waiting and running: each that `rank` ranks
    /// takes that rank; others keep theirs. A running load ranked beside
    /// the view now yields its turn to a load waiting before it.
    pub fn reorder(&self, rank: &dyn Fn(LoadKey) -> Option<u64>) {
        let mut state = self.0.lock().expect("turns poisoned");
        let State {
            waiting, running, ..
        } = &mut *state;
        let keyed = (waiting.iter_mut().map(|w| (w.key, &mut w.rank)))
            .chain(running.iter_mut().map(|r| (r.key, &mut r.rank)));
        for (key, old) in keyed {
            if let Some(new) = key.and_then(rank) {
                *old = new;
            }
        }
        drop(state);
        start_waiting(&self.0);
    }

    /// Gives up `key`'s load: waiting, it never starts and its start hears
    /// so; running, its abort fires. Returns whether it was running.
    pub fn cancel(&self, key: LoadKey) -> bool {
        let (given_up, running) = {
            let mut state = self.0.lock().expect("turns poisoned");
            let (given_up, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut state.waiting)
                .into_iter()
                .partition(|w| w.key == Some(key));
            state.waiting = kept;
            let running: Vec<Abort> = (state.running.iter())
                .filter(|r| r.key == Some(key))
                .map(|r| r.abort.clone())
                .collect();
            state.counts.aborted += running.len() as u64;
            (given_up, running)
        };
        for waiting in given_up {
            (waiting.start)(None);
        }
        let was_running = !running.is_empty();
        for abort in running {
            abort.abort();
        }
        was_running
    }

    /// The fetches these turns started, aborted and completed so far.
    pub fn counts(&self) -> FetchCounts {
        self.0.lock().expect("turns poisoned").counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loads whose turns came, each with its turn until it ends.
    type Started = Arc<Mutex<Vec<(u32, Option<Turn>)>>>;

    fn key(col: u32) -> LoadKey {
        (1, 1, TileIndex::new(4, col, 0))
    }

    /// Queues `col` at `rank`; its turn lands in `started`.
    fn queue(turns: &Turns, started: &Started, col: u32, rank: u64) {
        let started = Arc::clone(started);
        let start: Start = Box::new(move |turn| started.lock().unwrap().push((col, turn)));
        turns.wait(rank, Some(key(col)), start);
    }

    fn cols(started: &Started) -> Vec<u32> {
        started.lock().unwrap().iter().map(|(c, _)| *c).collect()
    }

    /// Waiting loads start lowest rank first, by their ranks when a turn
    /// frees: a reorder moves them, and one given up never starts.
    #[test]
    fn waiting_loads_start_by_their_latest_rank() {
        let turns = Turns::new(1);
        let started = Arc::default();
        for (col, rank) in [(0, 0), (1, 5), (2, 3), (3, 4)] {
            queue(&turns, &started, col, rank);
        }
        assert_eq!(cols(&started), [0]);
        // The view moved: 1 is nearest now.
        turns.reorder(&|(_, _, t)| (t.col == 1).then_some(1));
        assert!(!turns.cancel(key(2)));
        let next = |started: &Started| {
            let turn = started
                .lock()
                .unwrap()
                .iter_mut()
                .rev()
                .find_map(|(_, t)| t.take());
            drop(turn);
        };
        next(&started);
        next(&started);
        // 2 heard it was given up, with no turn.
        assert_eq!(cols(&started), [0, 2, 1, 3]);
        let given = started.lock().unwrap();
        assert!(given[1].1.is_none());
    }

    /// A load the view left, loading on beside it, gives its turn to a
    /// load of the view that waits, and its fetch is aborted; one of the
    /// view keeps its turn.
    #[test]
    fn loads_beside_the_view_yield_their_turns_to_the_view() {
        let turns = Turns::new(2);
        let started = Arc::default();
        queue(&turns, &started, 0, 1);
        queue(&turns, &started, 1, 2);
        let abort = |i: usize| {
            started.lock().unwrap()[i]
                .1
                .as_ref()
                .unwrap()
                .abort()
                .clone()
        };
        let (first, second) = (abort(0), abort(1));
        // The view left 1: it loads on beside it, and nothing waits.
        turns.reorder(&|(_, _, t)| (t.col == 1).then_some(YIELDING + 2));
        assert!(!second.is_aborted());
        // A load of the view comes: 1 yields; 0 runs on.
        queue(&turns, &started, 2, 3);
        assert!(second.is_aborted() && !first.is_aborted());
        assert_eq!(cols(&started), [0, 1, 2]);
        // Another waits behind the view's two.
        queue(&turns, &started, 3, 4);
        assert_eq!(cols(&started), [0, 1, 2]);
        assert_eq!(turns.counts().aborted, 1);
    }

    /// A load given up in its turn is aborted, its turn goes to the next,
    /// and the counts tell started, aborted and completed apart.
    #[test]
    fn running_loads_given_up_are_aborted_and_counted() {
        let turns = Turns::new(2);
        let started = Arc::default();
        for col in 0..3 {
            queue(&turns, &started, col, u64::from(col));
        }
        let abort = {
            let given = started.lock().unwrap();
            given[0].1.as_ref().unwrap().abort().clone()
        };
        assert!(turns.cancel(key(0)));
        assert!(abort.is_aborted());
        // Its fetch answers aborted, and its turn ends.
        let turn = started.lock().unwrap()[0].1.take();
        drop(turn);
        assert_eq!(cols(&started), [0, 1, 2]);
        let turn = started.lock().unwrap()[1].1.take();
        drop(turn);
        let counts = turns.counts();
        assert_eq!(
            counts,
            FetchCounts {
                started: 3,
                aborted: 1,
                completed: 1
            }
        );
    }
}
