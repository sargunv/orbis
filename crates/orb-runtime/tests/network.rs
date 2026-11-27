//! Loads over a modeled network, frame by frame: it keeps [`SLOTS`]
//! requests in flight and answers each [`DELAY`] frames after it got a
//! slot, and jobs run inline, so every run counts the same frames. A
//! camera that moves fast and stops must find the view it stopped at
//! loaded within the frames its own requests need: what a fresh map at
//! that camera takes, whatever it asked for on the way. A globe turned to
//! its far side shows coarse tiles there at once, and a view turned back
//! and forth receives each vector tile once. (mln-rs's fast camera test
//! models its network alike.) A server that puts a reload off is asked
//! again no sooner than it said, on a clock the test moves.

mod common;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use common::{Geodetic, Map, Offscreen};
use orb_host::{Abort, Fetch, FetchDone, FetchError, ResourceKind};
use orb_runtime::{CameraState, FrameReport, Services};

/// Requests the network keeps in flight, as a browser's or MapLibre
/// Native's file source does.
const SLOTS: usize = 20;
/// Frames from a request's slot to its answer: 100 ms at 60 Hz.
const DELAY: u64 = 6;

/// The background, which no tile is: where it shows, nothing stands in.
const BACKGROUND: [u8; 3] = [255, 0, 255];

/// A vector source with a fill layer and a raster source of solid tiles,
/// every tile of both answered, over a background no tile matches.
fn style(globe: bool) -> String {
    let projection = match globe {
        true => r#""projection": {"type": "globe"},"#,
        false => "",
    };
    format!(
        r##"{{"version": 8, {projection}
        "sources": {{
            "r": {{"type": "raster", "tiles": ["model://r/{{z}}/{{x}}/{{y}}.png"], "tileSize": 256}},
            "v": {{"type": "vector", "tiles": ["model://v/{{z}}/{{x}}/{{y}}.mvt"]}}}},
        "layers": [
            {{"id": "bg", "type": "background", "paint": {{"background-color": "#ff00ff"}}}},
            {{"id": "r", "type": "raster", "source": "r", "paint": {{"raster-fade-duration": 0}}}},
            {{"id": "v", "type": "fill", "source": "v", "source-layer": "building",
              "paint": {{"fill-color": "#0000ff"}}}}]}}"##
    )
}

/// A request in flight: when it's answered, and how.
struct InFlight {
    due: u64,
    done: FetchDone,
    abort: Abort,
    url: String,
}

#[derive(Default)]
struct State {
    frame: u64,
    in_flight: Vec<InFlight>,
    queued: VecDeque<(String, FetchDone, Abort)>,
    /// The tile requests made.
    requests: Vec<String>,
    /// The requests answered: made and not given up before their answer.
    answered: Vec<String>,
}

/// The modeled network: see the module.
#[derive(Default)]
struct Network {
    state: Mutex<State>,
    raster: Vec<u8>,
    vector: Vec<u8>,
}

impl Network {
    fn new() -> Arc<Self> {
        let mut raster = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut raster, 256, 256);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let pixels = [0u8, 160, 0, 255].repeat(256 * 256);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&pixels).unwrap();
        }
        let tile = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/tiles/14-14552-6450.mvt"
        );
        Arc::new(Self {
            state: Mutex::default(),
            raster,
            vector: std::fs::read(tile).unwrap(),
        })
    }

    /// The body `url` answers with.
    fn body(&self, url: &str) -> Result<Vec<u8>, FetchError> {
        match url {
            u if u.starts_with("model://r/") => Ok(self.raster.clone()),
            u if u.starts_with("model://v/") => Ok(self.vector.clone()),
            _ => Err(FetchError::NotFound),
        }
    }

    /// The next frame: answers what is due, then gives free slots to the
    /// requests queued, oldest first. A request given up leaves its slot
    /// at once.
    fn tick(&self) {
        let due = {
            let mut state = self.state.lock().unwrap();
            state.frame += 1;
            let frame = state.frame;
            state.in_flight.retain(|f| !f.abort.is_aborted());
            let (due, kept) = std::mem::take(&mut state.in_flight)
                .into_iter()
                .partition(|f| f.due <= frame);
            state.in_flight = kept;
            due
        };
        let due: Vec<InFlight> = due;
        for f in due {
            let body = self.body(&f.url);
            self.state.lock().unwrap().answered.push(f.url);
            (f.done)(body.map(Into::into));
        }
        let mut state = self.state.lock().unwrap();
        let frame = state.frame;
        while state.in_flight.len() < SLOTS {
            let Some((url, done, abort)) = state.queued.pop_front() else {
                break;
            };
            if !abort.is_aborted() {
                let due = frame + DELAY;
                state.in_flight.push(InFlight {
                    due,
                    done,
                    abort,
                    url,
                });
            }
        }
    }

    fn requests(&self) -> usize {
        self.state.lock().unwrap().requests.len()
    }
}

impl Fetch for Network {
    fn get(&self, _: ResourceKind, url: &str, abort: &Abort, done: FetchDone) {
        let done = abort.answer(done);
        let mut state = self.state.lock().unwrap();
        state.requests.push(url.to_string());
        state
            .queued
            .push_back((url.to_string(), done, abort.clone()));
    }

    fn get_range(&self, _: ResourceKind, _: &str, _: u64, _: u64, _: &Abort, done: FetchDone) {
        done(Err(FetchError::NotFound));
    }
}

/// A map over `network` at `camera`, drawing `style` offscreen, its jobs
/// and placement inline.
fn offscreen(network: &Arc<Network>, style: &str, camera: CameraState) -> Offscreen {
    let fetch: Arc<dyn Fetch> = network.clone();
    let jobs = orb_runtime::PoolJobs::new(
        fetch.clone(),
        Arc::new(common::NativeImageDecode),
        Arc::new(orb_host::jobs::Inline),
    );
    let services = Services {
        clock: Arc::new(common::StillClock),
        image_decode: Arc::new(common::NativeImageDecode),
        images: Default::default(),
        jobs,
        fetch,
        wake: Arc::new(|| {}),
    };
    let mut map = Map::new(services, Some(camera));
    map.set_placement_lane(Arc::new(orb_runtime::InlinePlacement::default()));
    map.set_style_json(style).unwrap();
    Offscreen::with_map(map).unwrap()
}

/// The next frame, after the network's.
fn frame(network: &Network, offscreen: &mut Offscreen) -> FrameReport {
    network.tick();
    offscreen.frame()
}

/// Frames until the view has loaded, at most `limit`.
fn frames_to_load(network: &Network, offscreen: &mut Offscreen, limit: u32) -> u32 {
    for n in 1..=limit {
        if frame(network, offscreen).loaded {
            return n;
        }
    }
    panic!("not loaded after {limit} frames");
}

fn at(lon: f64, lat: f64, zoom: f64) -> CameraState {
    CameraState::new(Geodetic::new(lon, lat), zoom, &common::web_mercator())
}

/// Moves from `path`'s first camera through the rest, one a frame, after
/// loading the first; then the frames until the last loads, against a
/// fresh map's there, and the first frame after the stop's pixels.
fn after_path(globe: bool, path: &[CameraState]) -> (u32, u32, Vec<u8>) {
    let style = style(globe);
    let stop = *path.last().unwrap();
    let fresh = {
        let network = Network::new();
        let mut map = offscreen(&network, &style, stop);
        frames_to_load(&network, &mut map, 600)
    };
    let network = Network::new();
    let mut map = offscreen(&network, &style, path[0]);
    frames_to_load(&network, &mut map, 600);
    for &camera in &path[1..] {
        map.map.jump_to_state(camera);
        frame(&network, &mut map);
    }
    let stopped = map.pixels();
    let moved = 1 + frames_to_load(&network, &mut map, 600);
    eprintln!(
        "{} requests; loaded {moved} frames after the stop, a fresh map in {fresh}",
        network.requests()
    );
    (moved, fresh, stopped)
}

/// A fling at zoom 6 from Paris to Beijing, a tile a frame.
#[test]
fn a_fling_loads_where_it_stops_as_soon_as_a_fresh_map_would() {
    let Ok(_) = pollster::block_on(common::WgpuGpu::offscreen()) else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let (from, to) = ((2.35, 48.85), (116.4, 39.9));
    let steps = 20;
    let path: Vec<CameraState> = (0..=steps)
        .map(|i| {
            let f = f64::from(i) / f64::from(steps);
            at(
                from.0 + (to.0 - from.0) * f,
                from.1 + (to.1 - from.1) * f,
                6.0,
            )
        })
        .collect();
    let (moved, fresh, _) = after_path(false, &path);
    assert!(moved <= fresh, "{moved} frames, a fresh map's {fresh}");
}

/// A pinch out from zoom 5.5 to the whole world, a jump every second
/// frame, as mln-rs's fast camera test pinches.
#[test]
fn a_pinch_out_loads_the_world_as_soon_as_a_fresh_map_would() {
    let Ok(_) = pollster::block_on(common::WgpuGpu::offscreen()) else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let mut path = vec![at(10.0, 50.0, 5.5)];
    for zoom in [4.5, 3.5, 2.5, 1.5, 1.23] {
        path.extend([at(10.0, 50.0, zoom); 2]);
    }
    let (moved, fresh, _) = after_path(false, &path);
    assert!(moved <= fresh, "{moved} frames, a fresh map's {fresh}");
}

/// A globe spun from Switzerland to Japan in a second loads where it
/// stops as soon as a fresh map would.
#[test]
fn a_spun_globe_loads_where_it_stops_as_soon_as_a_fresh_map_would() {
    let Ok(_) = pollster::block_on(common::WgpuGpu::offscreen()) else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let steps = 60;
    let path: Vec<CameraState> = (0..=steps)
        .map(|i| {
            let f = f64::from(i) / f64::from(steps);
            at(
                SWITZERLAND.0 + (JAPAN.0 - SWITZERLAND.0) * f,
                SWITZERLAND.1 + (JAPAN.1 - SWITZERLAND.1) * f,
                4.0,
            )
        })
        .collect();
    let (moved, fresh, _) = after_path(true, &path);
    assert!(moved <= fresh, "{moved} frames, a fresh map's {fresh}");
}

/// A globe turned to its far side shows the sources' coarse tiles there
/// at once, before any of its own tiles come: never the background.
#[test]
fn a_globe_turned_to_its_far_side_shows_coarse_tiles_at_once() {
    let Ok(_) = pollster::block_on(common::WgpuGpu::offscreen()) else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let path = [
        at(SWITZERLAND.0, SWITZERLAND.1, 4.0),
        at(JAPAN.0, JAPAN.1, 4.0),
    ];
    let (_, _, turned) = after_path(true, &path);
    let (pixels, _) = turned.as_chunks::<4>();
    let bare = pixels.iter().filter(|p| p[..3] == BACKGROUND).count();
    assert_eq!(bare, 0, "background pixels on the far side");
}

const SWITZERLAND: (f64, f64) = (8.23, 46.82);
const JAPAN: (f64, f64) = (138.25, 36.2);

/// A view tilted over Paris, which [`swing`] turns.
fn pitched() -> CameraState {
    CameraState {
        pitch: 60.0,
        ..at(2.35, 48.85, 12.0)
    }
}

/// Speeds of a swing, in frames each way: a flick to a slow turn.
const SWINGS: [u32; 6] = [3, 5, 10, 20, 40, 80];

/// Turns `map` 40 degrees and back four times from `start`, `frames`
/// frames each way, and lets it load where it started.
fn swing(network: &Network, map: &mut Offscreen, start: CameraState, frames: u32) {
    for swing in 0..8 {
        for i in 0..=frames {
            let f = f64::from(i) / f64::from(frames);
            let turned = if swing % 2 == 0 { f } else { 1.0 - f };
            map.map.jump_to_state(CameraState {
                bearing: 40.0 * turned,
                ..start
            });
            frame(network, map);
        }
    }
    frames_to_load(network, map, 600);
}

/// The first idle frame's report: the map at rest.
fn at_rest(network: &Network, map: &mut Offscreen) -> FrameReport {
    for _ in 0..600 {
        let report = frame(network, map);
        if report.idle() {
            return report;
        }
    }
    panic!("not idle after 600 frames");
}

/// A pitched view turned back and forth, slowly or fast, receives each
/// vector tile once: a tile beside the view loads on when the view leaves
/// it, and the cache keeps what came, so a swing back finds it loaded
/// rather than asking again. (A load beside the view gives its turn to one
/// of the view's, and comes again when wanted; image tiles out of view
/// keep only the layers the view's own tiles grew their page to, so
/// they're given up: [`a_view_turned_back_and_forth_rests_on_a_fresh_maps_tile_pages`].)
#[test]
fn a_view_turned_back_and_forth_receives_each_vector_tile_once() {
    let Ok(_) = pollster::block_on(common::WgpuGpu::offscreen()) else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    for frames in SWINGS {
        let network = Network::new();
        let mut map = offscreen(&network, &style(false), pitched());
        frames_to_load(&network, &mut map, 600);
        let before = network.state.lock().unwrap().answered.len();
        swing(&network, &mut map, pitched(), frames);
        let state = network.state.lock().unwrap();
        let answered: Vec<&String> = (state.answered[before..].iter())
            .filter(|u| u.starts_with("model://v/"))
            .collect();
        let mut tiles = answered.clone();
        tiles.sort();
        tiles.dedup();
        assert!(!tiles.is_empty());
        assert_eq!(
            answered.len(),
            tiles.len(),
            "vector tiles received again at {frames} frames a swing"
        );
    }
}

/// A pitched view turned back and forth, slowly or fast, rests on the
/// tile pages a fresh map at its camera holds: what's kept of image
/// tiles out of view, a whole page layer each, lives in layers the view's
/// own tiles need, so the swing leaves no page grown for them.
#[test]
fn a_view_turned_back_and_forth_rests_on_a_fresh_maps_tile_pages() {
    let Ok(_) = pollster::block_on(common::WgpuGpu::offscreen()) else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let fresh = {
        let network = Network::new();
        let mut map = offscreen(&network, &style(false), pitched());
        frames_to_load(&network, &mut map, 600);
        at_rest(&network, &mut map).memory.tile_textures
    };
    assert!(fresh > 0);
    for frames in SWINGS {
        let network = Network::new();
        let mut map = offscreen(&network, &style(false), pitched());
        frames_to_load(&network, &mut map, 600);
        swing(&network, &mut map, pitched(), frames);
        let swung = at_rest(&network, &mut map).memory.tile_textures;
        assert_eq!(
            swung, fresh,
            "tile pages at rest after swings of {frames} frames, against a fresh map's"
        );
    }
}

/// A clock a test moves: milliseconds since it was made, and as many past
/// a fixed Unix time, plus what the system slept, which the monotonic
/// clock (`Instant`, `performance.now()`) doesn't count.
#[derive(Default)]
struct HandClock {
    now: Mutex<f64>,
    slept: Mutex<f64>,
}

impl HandClock {
    const EPOCH_MS: f64 = 1.7e12;

    fn advance(&self, ms: f64) {
        *self.now.lock().unwrap() += ms;
    }

    fn sleep(&self, ms: f64) {
        *self.slept.lock().unwrap() += ms;
    }
}

impl orb_host::Clock for HandClock {
    fn now(&self) -> f64 {
        *self.now.lock().unwrap()
    }

    fn unix_ms(&self) -> f64 {
        Self::EPOCH_MS + *self.slept.lock().unwrap() + self.now()
    }
}

/// A server whose one tile comes fresh for a minute, whose reload of it
/// it answers with 429 and `Retry-After: 30`, and which then serves it
/// with no lifetime: each answer at once, on the asking thread.
struct Throttling {
    vector: Vec<u8>,
    requests: Mutex<Vec<f64>>,
    clock: Arc<HandClock>,
}

impl Fetch for Throttling {
    fn get(&self, _: ResourceKind, url: &str, _: &Abort, done: FetchDone) {
        assert!(url.starts_with("model://v/"), "{url}");
        let n = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(orb_host::Clock::now(&*self.clock));
            requests.len()
        };
        let unix = orb_host::Clock::unix_ms(&*self.clock) / 1000.0;
        done(match n {
            1 => Ok(orb_host::Fetched {
                bytes: self.vector.clone(),
                fresh_until: Some(unix as u64 + 60),
                stale: false,
            }),
            2 => Err(FetchError::RetryAfter {
                error: Box::new(FetchError::Status(429)),
                after: std::time::Duration::from_secs(30),
            }),
            _ => Ok(self.vector.clone().into()),
        });
    }

    fn get_range(&self, _: ResourceKind, _: &str, _: u64, _: u64, _: &Abort, done: FetchDone) {
        done(Err(FetchError::NotFound));
    }
}

/// A tile whose data expired reloads, once the wall clock passes its
/// lifetime, the system's sleep included; a server that puts the reload
/// off (429 with `Retry-After`) is asked again no sooner than it said,
/// not on the backoff's first second, as a first load waits.
#[test]
fn a_reload_its_server_puts_off_waits_as_long_as_it_asks() {
    let Ok(_) = pollster::block_on(common::WgpuGpu::offscreen()) else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    let clock = Arc::new(HandClock::default());
    let server = Arc::new(Throttling {
        vector: Network::new().vector.clone(),
        requests: Mutex::default(),
        clock: Arc::clone(&clock),
    });
    let fetch: Arc<dyn Fetch> = server.clone();
    let jobs = orb_runtime::PoolJobs::new(
        fetch.clone(),
        Arc::new(common::NativeImageDecode),
        Arc::new(orb_host::jobs::Inline),
    );
    let services = Services {
        clock: clock.clone(),
        image_decode: Arc::new(common::NativeImageDecode),
        images: Default::default(),
        jobs,
        fetch,
        wake: Arc::new(|| {}),
    };
    let mut map = Map::new(services, Some(at(0.0, 0.0, 0.0)));
    map.set_placement_lane(Arc::new(orb_runtime::InlinePlacement::default()));
    map.set_style_json(
        r##"{"version": 8,
        "sources": {"v": {"type": "vector", "tiles": ["model://v/{z}/{x}/{y}.mvt"], "maxzoom": 0}},
        "layers": [{"id": "v", "type": "fill", "source": "v", "source-layer": "building",
            "paint": {"fill-color": "#0000ff"}}]}"##,
    )
    .unwrap();
    let mut map = Offscreen::with_map(map).unwrap();
    let requests = || server.requests.lock().unwrap().clone();
    // Frames, 100 ms of the clock apart, until the map is idle: its wake.
    let settle = |map: &mut Offscreen| {
        for _ in 0..200 {
            let report = map.frame();
            if report.idle() {
                return report.wake_in;
            }
            clock.advance(100.0);
        }
        panic!("the map never idled");
    };
    let wake = settle(&mut map).expect("woken for the expiry");
    assert_eq!(requests().len(), 1);
    assert!(wake.as_secs_f64() <= 60.0, "woken in {wake:?}");
    // Asleep past the tile's minute: the first frame on waking reloads it.
    clock.sleep(3_600_000.0);
    map.frame();
    assert_eq!(requests().len(), 2, "the expired tile drawn on waking");
    // The reload, put off: woken for the server's wait.
    let wake = settle(&mut map).expect("woken for the retry");
    let asked = requests();
    assert_eq!(asked.len(), 2, "{asked:?}");
    // From when the map heard the answer, a frame or so after the ask.
    let due = orb_host::Clock::now(&*clock) + wake.as_secs_f64() * 1000.0;
    let wait = due - asked[1];
    assert!(
        (30_000.0..31_000.0).contains(&wait),
        "woken {wait} ms after the ask"
    );
    // Not asked before then, whatever frames come.
    while orb_host::Clock::now(&*clock) + 500.0 < due {
        clock.advance(500.0);
        map.frame();
    }
    assert_eq!(requests().len(), 2, "asked again before the server's wait");
    clock.advance(500.0);
    assert_eq!(settle(&mut map), None, "the reload brought no lifetime");
    assert_eq!(requests().len(), 3);
}
