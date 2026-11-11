use std::sync::Mutex;

use orb_host::{FetchDone, ResourceKind};
use orb_runtime::TileLocation;

use super::*;

/// A fetch that never answers: no test prefetches glyphs.
struct NoFetch;

impl Fetch for NoFetch {
    fn get(&self, _: ResourceKind, _: &str, _: FetchDone) {}
    fn get_range(&self, _: ResourceKind, _: &str, _: u64, _: u64, _: FetchDone) {}
}

/// The requests posted, with the worker each went to.
#[derive(Clone, Default)]
struct Posted(Arc<Mutex<Vec<(u32, JobRequest)>>>);

impl Posted {
    /// Takes what was posted so far.
    fn take(&self) -> Vec<(u32, JobRequest)> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }

    /// Takes what was posted so far, as the loads among it: each one's
    /// worker, id and tile.
    fn loads(&self) -> Vec<(u32, u32, TileIndex)> {
        (self.take().into_iter())
            .filter_map(|(worker, request)| match request {
                JobRequest::Load { id, request } => Some((worker, id, request.tile)),
                _ => None,
            })
            .collect()
    }
}

fn jobs(workers: u32) -> (Arc<WorkerJobs>, Posted) {
    let posted = Posted::default();
    let sink = posted.clone();
    let post = move |to: &[u32], message: &[u8]| {
        for &worker in to {
            let request = wire::decode(message).unwrap();
            sink.0.lock().unwrap().push((worker, request));
        }
    };
    (
        WorkerJobs::new(workers, Box::new(post), Arc::new(NoFetch)),
        posted,
    )
}

fn tile(col: u32) -> TileIndex {
    TileIndex::new(10, col, 0)
}

/// A load's result, once it has one.
type Outcome = Arc<Mutex<Option<Result<TileResult, String>>>>;

fn load(jobs: &WorkerJobs, tile: TileIndex) -> Outcome {
    let location = TileLocation::Url("test://tile.mvt".into());
    let request = TileRequest::whole(1, 1, tile, location, orb_style::Encoding::Mvt);
    let outcome = Outcome::default();
    let into = outcome.clone();
    jobs.load(request, Box::new(move |r| *into.lock().unwrap() = Some(r)));
    outcome
}

/// Answers load `id` with a laid-out tile.
fn answer(jobs: &WorkerJobs, id: u32) {
    jobs.complete(JobReply::Tile { id }, |_| Ok(TileResult::default()))
        .unwrap();
}

fn answered(outcome: &Outcome) -> bool {
    matches!(*outcome.lock().unwrap(), Some(Ok(ref r)) if !r.missing)
}

/// The warning a load answered missing gave, if it was.
fn left_out(outcome: &Outcome) -> Option<Vec<String>> {
    match &*outcome.lock().unwrap() {
        Some(Ok(r)) if r.missing => Some(r.warnings.clone()),
        _ => None,
    }
}

#[test]
fn a_lost_workers_loads_go_again_one_suspect_at_a_time() {
    let (jobs, posted) = jobs(1);
    let outcomes = [load(&jobs, tile(0)), load(&jobs, tile(1))];
    assert_eq!(posted.loads().len(), 2);
    jobs.lost(0, "trapped", Some(None));
    // One suspect in flight; the other waits for it.
    let [(0, first, t)] = posted.loads()[..] else {
        panic!("one load sent again")
    };
    let first_col = t.col as usize;
    answer(&jobs, first);
    assert!(answered(&outcomes[first_col]));
    let [(0, second, t)] = posted.loads()[..] else {
        panic!("the next suspect sent once the first answered")
    };
    assert_eq!(t.col as usize, 1 - first_col);
    answer(&jobs, second);
    assert!(answered(&outcomes[1 - first_col]));
    // Answered, a suspect is cleared: a later loss makes it a suspect
    // again, not a quarantined tile.
    let a = load(&jobs, tile(0));
    posted.take();
    jobs.lost(0, "trapped", Some(None));
    let [(0, id, _)] = posted.loads()[..] else {
        panic!("sent again")
    };
    answer(&jobs, id);
    assert!(answered(&a));
    assert_eq!(jobs.done(), 3, "replies counted");
}

#[test]
fn a_suspect_lost_again_is_quarantined() {
    let (jobs, posted) = jobs(1);
    let outcomes = [load(&jobs, tile(0)), load(&jobs, tile(1))];
    posted.take();
    jobs.lost(0, "trapped", Some(None));
    // One is the suspect in flight, the other waits behind it; the
    // replacement traps on the suspect too.
    let [(0, _, suspect)] = posted.loads()[..] else {
        panic!("one load sent again")
    };
    let suspect = suspect.col as usize;
    jobs.lost(0, "trapped", Some(None));
    let warnings = left_out(&outcomes[suspect]).expect("quarantined: answered missing");
    assert!(
        warnings[0].contains("is left out from now on: job worker 0 lost: trapped"),
        "{warnings:?}"
    );
    // The suspect behind it goes next.
    let [(0, id, t)] = posted.loads()[..] else {
        panic!("the waiting suspect sent")
    };
    assert_eq!(t.col as usize, 1 - suspect);
    answer(&jobs, id);
    assert!(answered(&outcomes[1 - suspect]));
    // Loaded again, a quarantined tile answers missing at once, quietly.
    let again = load(&jobs, tile(suspect as u32));
    assert!(posted.take().is_empty());
    assert_eq!(left_out(&again), Some(vec![]));
}

#[test]
fn a_replacement_gets_the_style_icons_and_archive_heads_before_the_loads() {
    let (jobs, posted) = jobs(2);
    jobs.add_archive_head("pmtiles://a", &Arc::new(vec![1, 2, 3]));
    let _a = load(&jobs, tile(0));
    let [(lost, _, _)] = posted.loads()[..] else {
        panic!("one load")
    };
    let style = (7, StyleUpdate::default(), SpriteIcons::default());
    jobs.lost(lost, "trapped", Some(Some(style)));
    let sent: Vec<_> = (posted.take().into_iter())
        .map(|(worker, request)| {
            let kind = match request {
                JobRequest::Style { style: 7, .. } => "style",
                JobRequest::Icons { style: 7, .. } => "icons",
                JobRequest::ArchiveHead { url, head } => {
                    assert_eq!((url.as_str(), &head[..]), ("pmtiles://a", &[1, 2, 3][..]));
                    "head"
                }
                JobRequest::Load { .. } => "load",
                other => panic!("{other:?}"),
            };
            (worker, kind)
        })
        .collect();
    // The load goes to its tile's home, the replacement.
    let expected = ["style", "icons", "head", "load"].map(|kind| (lost, kind));
    assert_eq!(sent, expected);
}

#[test]
fn without_a_replacement_the_next_worker_takes_the_share() {
    let (jobs, posted) = jobs(2);
    let a = load(&jobs, tile(0));
    let [(lost, _, _)] = posted.loads()[..] else {
        panic!("one load")
    };
    let other = 1 - lost;
    jobs.lost(lost, "died", None);
    let [(worker, id, _)] = posted.loads()[..] else {
        panic!("sent again")
    };
    assert_eq!(worker, other);
    // New loads skip the lost worker too.
    let b = load(&jobs, tile(1));
    let [(worker, _, _)] = posted.loads()[..] else {
        panic!("one load")
    };
    assert_eq!(worker, other);
    answer(&jobs, id);
    assert!(answered(&a));
    // With every worker lost, the loads it had and new ones fail at once.
    jobs.lost(other, "died", None);
    let c = load(&jobs, tile(2));
    assert!(posted.take().is_empty());
    for outcome in [b, c] {
        let failed = outcome.lock().unwrap().take();
        assert_eq!(failed.unwrap().unwrap_err(), "every job worker was lost");
    }
}

#[test]
fn a_lost_workers_rasters_go_again() {
    let (jobs, posted) = jobs(1);
    let raster = GlyphRaster::Pbf {
        url: "test://0-255.pbf".into(),
        code_points: vec![65],
    };
    let bitmaps = Arc::new(Mutex::new(None));
    let into = bitmaps.clone();
    jobs.raster(
        raster.clone(),
        Box::new(move |r| *into.lock().unwrap() = Some(r)),
    );
    posted.take();
    jobs.lost(0, "trapped", Some(None));
    let [(0, JobRequest::Raster { id, request })] = &posted.take()[..] else {
        panic!("the raster sent again")
    };
    assert_eq!(request, &raster);
    let reply = JobReply::Glyphs {
        id: *id,
        bitmaps: vec![],
    };
    jobs.complete(reply, |_| unreachable!()).unwrap();
    assert_eq!(bitmaps.lock().unwrap().take(), Some(Ok(vec![])));
}
