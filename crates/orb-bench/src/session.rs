//! The session bench, `pnpm session`'s native twin: a user's first
//! session on a live style. Each session starts cold, on a runtime of its
//! own with an empty state directory (no HTTP cache, no pipeline cache)
//! whose shader modules are unique to the run
//! ([`Runtime::set_shader_nonce`], as `?nonce=` does on the web), so no
//! cache of wgpu's, the driver's or the system's holds a pipeline. Real
//! input then goes through [`Map::input`] from 500 ms after the first
//! frame that draws tiles, while tiles and pipelines still load: the wheel
//! notches, drags and right-button tilts and turns of `pnpm session`,
//! timed as it times them. Frames come on a 60 Hz beat, as a host's
//! display asks for them: a beat whose map has news draws a frame,
//! another does the map's upkeep, and input waits for the beat after it
//! comes, as an event loop's would.
//!
//! The sessions: `manhattan` (Liberty over Midtown, z13 wheel-zoomed to
//! about z16 as the buildings load, tilted to 70, turned and dragged),
//! `paris` (Liberty tilted from 60 to the limit, panned and turned),
//! `alps` (the Alps terrain style at pitch 72) and `globe` (Liberty on the
//! globe with the Globe entry's Mapterhorn terrain, wheel-zoomed from z2
//! over New York, tilting on the way) and `nepal` (the same, flung in and
//! out between z1 and z7 over the Himalaya at a spun wheel's and a
//! trackpad's rates). A style given that sets a projection of its own
//! draws as it is in the globe sessions: the web demo's Globe entry, say,
//! as `load.html` designs it (its `mapStyle`). `--dump prefix` writes each
//! session's frames to `<prefix>-<session>.tsv`. `--filter` picks them by name;
//! `--warm` runs each once untimed first and keeps its state directory
//! and shaders, as a user's later launch.
//!
//! It reports the frame intervals while input moves the camera, each
//! frame's CPU and GPU time, the time to centre (the first frame whose
//! tiles under the view's centre all arrived, and after each move how
//! long that took), the settle time, what the device did (pipelines asked
//! for, memory made, uploads) over the session, while input held the
//! camera and in frames that moved it, the pipelines that finished
//! compiling during the moves, and what the slowest intervals spent.

use std::fmt::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use glam::DVec2;
use orb_api::{
    Button, CameraState, FrameReport, Geodetic, GpuCounters, Input, Modifiers, PointerKind,
    Runtime, RuntimeOptions, Target, WheelDelta,
};

use crate::report::{ms, percentile};
use crate::{Settings, clock};

/// A 60 Hz display's frame.
const BEAT: Duration = Duration::from_micros(16_667);
/// How long after the first frame drawing tiles input starts.
const START: f64 = 500.0;
/// The pause between moves.
const PAUSE: f64 = 400.0;
/// How long the map may take to settle after the last input.
const SETTLE_TIMEOUT: Duration = Duration::from_secs(60);
/// How long the first frame drawing tiles may take.
const LOAD_TIMEOUT: Duration = Duration::from_secs(120);
/// Slow intervals the report lists.
const WORST: usize = 6;
/// The pixels a wheel notch scrolls: Chromium hears each of `pnpm
/// session`'s 100 px notches as 50 (its report's `firstWheel`), with a
/// legacy `wheelDeltaY` of -60, which the page reads as a trackpad's
/// pixels, not a notch.
const NOTCH: f64 = 0.5;

/// One move: wheel notches, or a drag with a button held.
#[derive(Clone, Copy, Debug)]
enum Move {
    /// `n` notches of `delta` pixels at a point (as a fraction of the
    /// viewport), `ms` apart: negative zooms in.
    Wheel {
        at: [f64; 2],
        delta: f64,
        n: u32,
        ms: f64,
    },
    /// From a point (a fraction of the viewport) by `by` logical pixels
    /// over `ms`, held still 150 ms before the release.
    Drag {
        at: [f64; 2],
        by: [f64; 2],
        ms: f64,
        button: Button,
    },
}

const C: [f64; 2] = [0.5, 0.5];
const LOW: [f64; 2] = [0.5, 0.8];

const fn wheel(delta: f64, n: u32) -> Move {
    fling(delta, n, 50.0)
}

/// Wheel notches `ms` apart: a spun wheel's or a trackpad's.
const fn fling(delta: f64, n: u32, ms: f64) -> Move {
    Move::Wheel {
        at: C,
        delta,
        n,
        ms,
    }
}

/// Zooms `levels` (whole, or halves) in, or out where negative, with
/// notches `ms` apart, each a third of a level (100 px heard at 1/300 a
/// pixel), as `pnpm session`'s `fling` does.
const fn levels(levels: i32, ms: f64) -> Move {
    let delta = (if levels < 0 { 100.0 } else { -100.0 }) / NOTCH;
    fling(delta, levels.unsigned_abs() * 3, ms)
}

/// Zooms whole `levels` with a trackpad's scroll, 8 px heard every 8 ms
/// (37.5 events a level), as `pnpm session`'s `scroll` does.
const fn scroll(levels: i32) -> Move {
    let delta = (if levels < 0 { 8.0 } else { -8.0 }) / NOTCH;
    fling(delta, levels.unsigned_abs() * 75 / 2, 8.0)
}

const fn drag(at: [f64; 2], by: [f64; 2], ms: f64) -> Move {
    Move::Drag {
        at,
        by,
        ms,
        button: Button::Primary,
    }
}

const fn turn(by: [f64; 2], ms: f64) -> Move {
    Move::Drag {
        at: C,
        by,
        ms,
        button: Button::Secondary,
    }
}

/// Which style a session draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    Liberty,
    Alps,
    /// Liberty on the globe with the Globe entry's terrain.
    Globe,
}

/// A session: its style, first camera (lon, lat, zoom, bearing, pitch)
/// and moves, as `web/scripts/session.mjs` has them.
struct Session {
    name: &'static str,
    style: Style,
    at: [f64; 5],
    moves: &'static [Move],
    /// The pause between moves, in ms.
    pause: f64,
}

const SESSIONS: &[Session] = &[
    Session {
        name: "manhattan",
        style: Style::Liberty,
        at: [-73.9855, 40.758, 13.0, 0.0, 0.0],
        moves: &[
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            turn([0.0, -140.0], 800.0),
            turn([160.0, 0.0], 800.0),
            drag(LOW, [0.0, -200.0], 800.0),
            drag(LOW, [-300.0, -100.0], 800.0),
            drag(LOW, [0.0, -200.0], 800.0),
            wheel(100.0, 8),
            wheel(-100.0, 8),
        ],
        pause: PAUSE,
    },
    Session {
        name: "paris",
        style: Style::Liberty,
        at: [2.3417, 48.8559, 15.0, 30.0, 60.0],
        moves: &[
            turn([0.0, -200.0], 600.0),
            drag(LOW, [0.0, -150.0], 800.0),
            turn([200.0, 0.0], 1000.0),
            drag(LOW, [250.0, -100.0], 800.0),
            drag(LOW, [0.0, -150.0], 800.0),
            turn([-300.0, 0.0], 1200.0),
            drag(LOW, [-250.0, -100.0], 800.0),
            wheel(-100.0, 8),
            drag(LOW, [0.0, -150.0], 800.0),
        ],
        pause: PAUSE,
    },
    Session {
        name: "alps",
        style: Style::Alps,
        at: [crate::ALPS.0, crate::ALPS.1, 11.6, -32.0, 72.0],
        moves: &[
            drag(LOW, [0.0, -400.0], 800.0),
            turn([250.0, 0.0], 1200.0),
            drag(LOW, [200.0, -300.0], 800.0),
            wheel(-100.0, 8),
            turn([-300.0, 0.0], 1200.0),
            drag(LOW, [0.0, -400.0], 800.0),
            wheel(100.0, 12),
            drag(LOW, [-300.0, -200.0], 800.0),
        ],
        pause: PAUSE,
    },
    Session {
        name: "globe",
        style: Style::Globe,
        at: [-73.9855, 40.758, 2.0, 0.0, 0.0],
        moves: &[
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            turn([0.0, -200.0], 800.0),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            wheel(-100.0, 6),
            drag(LOW, [0.0, -400.0], 800.0),
            turn([200.0, 0.0], 1000.0),
        ],
        pause: PAUSE,
    },
    Session {
        name: "nepal",
        style: Style::Globe,
        at: [84.0, 28.0, 1.0, 0.0, 0.0],
        moves: &[
            levels(6, 20.0),
            levels(-6, 20.0),
            levels(6, 16.0),
            levels(-3, 16.0),
            levels(3, 25.0),
            levels(-6, 16.0),
            scroll(6),
            scroll(-6),
            levels(6, 20.0),
            levels(-6, 20.0),
            levels(6, 16.0),
            levels(-3, 16.0),
            levels(3, 25.0),
            levels(-6, 16.0),
            scroll(6),
            scroll(-6),
        ],
        pause: 150.0,
    },
];

/// The Globe entry's terrain, on Liberty on the globe.
const GLOBE: &str = r#"[
    ["setProjection", {"type": "globe"}],
    ["addSource", "terrain", {"type": "raster-dem", "tiles": ["https://tiles.mapterhorn.com/{z}/{x}/{y}.webp"], "encoding": "terrarium", "tileSize": 512, "maxzoom": 17}],
    ["setTerrain", {"source": "terrain", "exaggeration": 1.3}]
]"#;

/// The inputs a session's moves make from `start` (ms), each at its
/// time, and each move's window.
fn timeline(
    moves: &[Move],
    pause: f64,
    viewport: [f64; 2],
    start: f64,
) -> (Vec<(f64, Input)>, Vec<[f64; 2]>) {
    let px = |[x, y]: [f64; 2]| DVec2::new(x * viewport[0], y * viewport[1]);
    let mut inputs = Vec::new();
    let mut windows = Vec::new();
    let mut t = start;
    for &m in moves {
        let from = t;
        match m {
            Move::Wheel { at, delta, n, ms } => {
                for i in 0..n {
                    t = from + f64::from(i) * ms;
                    inputs.push((
                        t,
                        Input::Wheel {
                            position: px(at),
                            delta: WheelDelta::Pixels(delta * NOTCH),
                            modifiers: Modifiers::default(),
                        },
                    ));
                }
                t += ms;
            }
            Move::Drag { at, by, ms, button } => {
                let at = px(at);
                let by = DVec2::from(by);
                inputs.push((
                    t,
                    Input::PointerDown {
                        id: 1,
                        kind: PointerKind::Mouse,
                        button,
                        position: at,
                        modifiers: Modifiers::default(),
                    },
                ));
                let steps = (ms / 16.0).ceil() as u32;
                for i in 1..=steps {
                    let f = f64::from(i) / f64::from(steps);
                    inputs.push((
                        from + f * ms,
                        Input::PointerMove {
                            id: 1,
                            position: at + by * f,
                        },
                    ));
                }
                t = from + ms + 150.0;
                inputs.push((
                    t,
                    Input::PointerUp {
                        id: 1,
                        position: at + by,
                    },
                ));
            }
        }
        windows.push([from, t]);
        t += pause;
    }
    (inputs, windows)
}

/// One frame drawn: when it ended (ms on the session's clock), its CPU
/// and GPU time and its report.
#[derive(Debug)]
struct Frame {
    t: f64,
    cpu: Duration,
    gpu: Option<Duration>,
    /// Its costliest passes on the GPU, summed by label (tiled GPUs
    /// overlap them: where work waits, not where it is).
    passes: Vec<(String, Duration)>,
    /// The camera's zoom once drawn.
    zoom: f64,
    report: FrameReport,
}

/// A session's run.
#[derive(Debug)]
pub(crate) struct SessionResult {
    name: &'static str,
    style: String,
    warm: bool,
    device_ms: f64,
    first_ms: Option<f64>,
    frames: Vec<Frame>,
    windows: Vec<[f64; 2]>,
    settle_ms: Option<f64>,
    /// Pipelines that finished compiling during the moves, with their
    /// compile times.
    compiled_in_moves: Vec<(String, Duration)>,
    camera: CameraState,
}

/// The sessions' runs.
#[derive(Debug)]
pub struct SessionReport {
    about: Vec<String>,
    size: (u32, u32, f32),
    results: Vec<SessionResult>,
    warnings: Vec<String>,
}

/// A directory of the session's own under the system's temporary one.
struct StateDir(PathBuf);

impl StateDir {
    fn new() -> Result<Self, String> {
        let base = std::env::temp_dir();
        for n in 0..100u32 {
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let dir = base.join(format!("orb-session-{}-{stamp}-{n}", std::process::id()));
            if std::fs::create_dir(&dir).is_ok() {
                return Ok(Self(dir));
            }
        }
        Err(format!("no state directory in {}", base.display()))
    }
}

impl Drop for StateDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Runs each session the filter keeps, cold (or `--warm`), on runtimes of
/// their own.
pub(crate) fn run(
    settings: &Settings,
    about: Vec<String>,
    (liberty, alps): (&str, &str),
    mut progress: impl FnMut(&str),
) -> Result<SessionReport, String> {
    let mut results = Vec::new();
    let mut warnings = Vec::new();
    let chosen = SESSIONS
        .iter()
        .filter(|s| (settings.filter.as_ref()).is_none_or(|f| s.name.contains(f.as_str())));
    for session in chosen {
        let style = match session.style {
            Style::Liberty | Style::Globe => liberty,
            Style::Alps => alps,
        };
        let state = StateDir::new()?;
        let nonce = Some(std::process::id() ^ clock::now().subsec_nanos());
        if settings.warm {
            progress(&format!("{}: once untimed", session.name));
            run_session(settings, session, style, (&state, nonce), &mut warnings)?;
        }
        progress(&format!(
            "{}: {}",
            session.name,
            if settings.warm { "warm" } else { "cold" }
        ));
        let mut result = run_session(settings, session, style, (&state, nonce), &mut warnings)?;
        result.warm = settings.warm;
        results.push(result);
    }
    if results.is_empty() {
        return Err("no session matches the filter".into());
    }
    warnings.sort();
    warnings.dedup();
    Ok(SessionReport {
        about,
        size: (settings.width, settings.height, settings.ratio),
        results,
        warnings,
    })
}

fn run_session(
    settings: &Settings,
    session: &Session,
    style: &str,
    (state, nonce): (&StateDir, Option<u32>),
    warnings: &mut Vec<String>,
) -> Result<SessionResult, String> {
    let made = Instant::now();
    let runtime = Runtime::new(&RuntimeOptions {
        state_dir: Some(state.0.clone()),
        ..RuntimeOptions::default()
    })
    .map_err(|e| e.to_string())?;
    runtime.set_shader_nonce(nonce);
    let device_ms = made.elapsed().as_secs_f64() * 1000.0;
    let (width, height) = settings.physical_size();
    let mut target = Target::texture(&runtime, width, height, f64::from(settings.ratio));
    let [lon, lat, zoom, bearing, pitch] = session.at;
    let camera = CameraState {
        bearing,
        pitch,
        ..CameraState::new(Geodetic::new(lon, lat), zoom, &crate::web_mercator())
    };
    let mut map = crate::new_map(&runtime, style, camera, settings.antialias);
    let mut edit = match session.style {
        Style::Globe => Some(orb_api::Transaction::from_commands(GLOBE)?),
        _ => None,
    };
    let mut edited = false;
    let gpu_timestamps = runtime.stats().gpu_timestamps;
    runtime.set_gpu_timing(gpu_timestamps);
    let epoch = Instant::now();
    let now = || epoch.elapsed().as_secs_f64() * 1000.0;
    let viewport = [f64::from(settings.width), f64::from(settings.height)];
    let mut frames: Vec<Frame> = Vec::new();
    let mut inputs: std::collections::VecDeque<(f64, Input)> = Default::default();
    let mut windows = Vec::new();
    let mut first_ms = None;
    let mut last_input = None;
    let mut settled = None;
    // The pipelines compiled when the first move started and the last
    // ended.
    let mut compiled = [None, None];
    let mut beat = 0u32;
    loop {
        beat += 1;
        let due = epoch + BEAT * beat;
        std::thread::sleep(due.saturating_duration_since(Instant::now()));
        // A beat missed while a frame ran is skipped, as a display's is.
        while epoch + BEAT * (beat + 1) <= Instant::now() {
            beat += 1;
        }
        let t = now();
        while inputs.front().is_some_and(|&(at, _)| at <= t) {
            let (at, input) = inputs.pop_front().expect("checked");
            map.input(input, at);
        }
        if let Some(tx) = &edit {
            map.apply_arrivals().map_err(|e| e.to_string())?;
            if let Some(doc) = map.style() {
                // A style that sets its projection itself (the Globe entry
                // as the web demo designs it, `load.html`'s `mapStyle`)
                // draws as it is.
                if doc.root("projection").is_none() {
                    map.edit(tx).map_err(|e| e.to_string())?;
                    edited = true;
                }
                edit = None;
            }
        }
        if map.has_news() || frames.is_empty() {
            let started = clock::now();
            let report = map
                .render(&mut target, t)
                .map_err(|e| e.to_string())?
                .ok_or("a texture target always draws")?;
            let cpu = clock::now() - started;
            warnings.extend(crate::take_warnings(&mut map));
            // The frame's submissions: collecting them waits for the GPU,
            // as a present would.
            let timings = match gpu_timestamps {
                true => runtime.gpu_timings(),
                false => Vec::new(),
            };
            let gpu = gpu_timestamps.then(|| timings.iter().map(|s| s.gpu()).sum());
            let mut passes: Vec<(String, Duration)> = Vec::new();
            for (label, took) in timings.iter().flat_map(|s| &s.passes) {
                match passes.iter_mut().find(|(l, _)| l == label) {
                    Some((_, sum)) => *sum += *took,
                    None => passes.push((label.clone(), *took)),
                }
            }
            passes.sort_by_key(|&(_, took)| std::cmp::Reverse(took));
            passes.truncate(3);
            let t = now();
            if first_ms.is_none() && report.tiles_drawn > 0 {
                first_ms = Some(t);
                let (moves, w) = timeline(session.moves, session.pause, viewport, t + START);
                inputs.extend(moves);
                last_input = w.last().map(|w| w[1]);
                windows = w;
            }
            frames.push(Frame {
                t,
                cpu,
                gpu,
                passes,
                zoom: map.camera().zoom(&crate::web_mercator()),
                report,
            });
        } else {
            map.upkeep();
        }
        let t = now();
        let edges = [windows.first().map(|w: &[f64; 2]| w[0]), last_input];
        for (count, edge) in compiled.iter_mut().zip(edges) {
            if count.is_none() && edge.is_some_and(|e| t >= e) {
                *count = Some(runtime.stats().pipeline_timings.len());
            }
        }
        match last_input {
            None if Duration::from_secs_f64(t / 1000.0) > LOAD_TIMEOUT => {
                return Err(format!(
                    "{}: nothing drawn after {LOAD_TIMEOUT:?}",
                    session.name
                ));
            }
            Some(end) if t > end && inputs.is_empty() => {
                if frames.last().is_some_and(|f| f.report.idle && f.t > end) {
                    settled = Some(frames.last().expect("checked").t - end);
                    break;
                }
                if t - end > SETTLE_TIMEOUT.as_secs_f64() * 1000.0 {
                    break;
                }
            }
            _ => {}
        }
    }
    runtime.set_gpu_timing(false);
    // Each frame's end (ms), the camera's zoom, tiles drawn, cull
    // candidates, CPU and GPU ms, draws, drape slots redrawn and whether
    // input moved the camera, for attributing slow frames.
    if let Some(prefix) = &settings.dump {
        let mut out =
            String::from("t\tzoom\ttiles\tcandidates\tcpu\tgpu\tdraws\tdrape_slots\tmoving\n");
        for f in &frames {
            let moving = windows.iter().any(|w| (w[0]..=w[1]).contains(&f.t));
            writeln!(
                out,
                "{:.1}\t{:.3}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{}",
                f.t,
                f.zoom,
                f.report.tiles_drawn,
                f.report.frame.candidates,
                f.cpu.as_secs_f64() * 1000.0,
                f.gpu.map_or(0.0, |g| g.as_secs_f64() * 1000.0),
                f.report.frame.draws,
                f.report.frame.drape_slots,
                u8::from(moving),
            )
            .map_err(|e| e.to_string())?;
        }
        let path = format!("{prefix}-{}.tsv", session.name);
        std::fs::write(&path, out).map_err(|e| format!("{path}: {e}"))?;
    }
    let compiled_in_moves = match compiled {
        [Some(from), Some(to)] => runtime.stats().pipeline_timings[from..to].to_vec(),
        _ => Vec::new(),
    };
    Ok(SessionResult {
        name: session.name,
        style: match edited {
            true => format!("{style} on the globe, with the Globe entry's terrain"),
            false => style.to_string(),
        },
        warm: false,
        device_ms,
        first_ms,
        camera: map.camera(),
        frames,
        windows,
        settle_ms: settled,
        compiled_in_moves,
    })
}

/// Percentiles and counts over 25 and 50 ms of intervals in ms.
fn pacing(xs: &mut [f64]) -> String {
    if xs.is_empty() {
        return "— | — | — | — | — | —".into();
    }
    xs.sort_by(f64::total_cmp);
    let at = |p: f64| xs[((xs.len() - 1) as f64 * p).round() as usize];
    format!(
        "{:.1} | {:.1} | {:.1} | {:.1} | {} | {}",
        at(0.5),
        at(0.95),
        at(0.99),
        at(1.0),
        xs.iter().filter(|&&x| x > 25.0).count(),
        xs.iter().filter(|&&x| x > 50.0).count(),
    )
}

/// The device's counts over `frames`, summed, the drape slots redrawn
/// and how many frames.
/// The frames' device counts summed, their drape slots redrawn, their
/// pipelines asked for outside the style's set, and how many they are.
/// The frames' device counts, drape slots, late pipelines and number;
/// the counts with what their preparations at rest did, or without.
fn summed<'a>(
    frames: impl Iterator<Item = &'a Frame>,
    at_rest: bool,
) -> (GpuCounters, u32, u32, usize) {
    let mut sum = GpuCounters::default();
    let (mut slots, mut late, mut n) = (0, 0, 0);
    for f in frames {
        sum += match at_rest {
            true => f.report.gpu,
            false => f.report.gpu.since(&f.report.prepared_at_rest),
        };
        slots += f.report.frame.drape_slots;
        late += f.report.pipelines_late;
        n += 1;
    }
    (sum, slots, late, n)
}

fn mib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / f64::from(1 << 20))
}

fn opt_ms(x: Option<f64>) -> String {
    x.map_or("—".into(), |x| format!("{x:.0}"))
}

impl SessionResult {
    /// Intervals between frames that begin and end inside a move, each
    /// with the index of the frame that ends it.
    fn intervals(&self) -> Vec<(usize, f64)> {
        (1..self.frames.len())
            .filter_map(|i| {
                let (a, b) = (self.frames[i - 1].t, self.frames[i].t);
                (self.windows.iter())
                    .any(|w| a >= w[0] && b <= w[1])
                    .then_some((i, b - a))
            })
            .collect()
    }

    /// The first frame whose centre loaded, and per move how long after
    /// its end that took.
    fn centre(&self) -> (Option<f64>, Vec<Option<f64>>) {
        let first = (self.frames.iter()).find(|f| f.report.centre_loaded);
        let after = (self.windows.iter())
            .map(|w| {
                // The frame showing the move's end: none follows it when
                // nothing changes.
                if (self.frames.iter().rev())
                    .find(|f| f.t <= w[1])
                    .is_some_and(|f| f.report.centre_loaded)
                {
                    return Some(0.0);
                }
                (self.frames.iter())
                    .find(|f| f.t >= w[1] && f.report.centre_loaded)
                    .map(|f| f.t - w[1])
            })
            .collect();
        (first.map(|f| f.t), after)
    }
}

impl SessionReport {
    pub(crate) fn write(&self, out: &mut String) -> std::fmt::Result {
        writeln!(out, "### Session bench")?;
        writeln!(out)?;
        for line in &self.about {
            writeln!(out, "- {line}")?;
        }
        let (w, h, ratio) = self.size;
        writeln!(
            out,
            "- {w}×{h} @{ratio} offscreen, a 60 Hz beat; each session on a fresh runtime with an \
             empty state directory and shaders unique to it (`--warm`: its second run), input \
             from {START:.0} ms after the first frame drawing tiles"
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| session | first tiles ms | centre ms | centre after moves p50 / max ms | interval \
             p50 | p95 | p99 | max | > 25 ms | > 50 ms | CPU p99 | max | GPU p99 | max | settle \
             ms | device ms |"
        )?;
        writeln!(out, "|{}", "---|".repeat(16))?;
        for r in &self.results {
            let mut gaps: Vec<f64> = r.intervals().into_iter().map(|(_, g)| g).collect();
            let (centre, after) = r.centre();
            let mut known: Vec<f64> = after.iter().flatten().copied().collect();
            known.sort_by(f64::total_cmp);
            let cpu = {
                let mut v: Vec<Duration> = r.frames.iter().map(|f| f.cpu).collect();
                v.sort_unstable();
                v
            };
            let gpu = {
                let mut v: Vec<Duration> = r.frames.iter().filter_map(|f| f.gpu).collect();
                v.sort_unstable();
                v
            };
            let at = |v: &[Duration], q: f64| {
                if v.is_empty() {
                    "—".into()
                } else {
                    ms(percentile(v, q))
                }
            };
            writeln!(
                out,
                "| {}{} | {} | {} | {} / {}{} | {} | {} | {} | {} | {} | {} | {:.0} |",
                r.name,
                if r.warm { " (warm)" } else { "" },
                opt_ms(r.first_ms),
                opt_ms(centre),
                opt_ms(known.get((known.len().max(1) - 1) / 2).copied()),
                opt_ms(known.last().copied()),
                match after.iter().filter(|x| x.is_none()).count() {
                    0 => String::new(),
                    n => format!(" ({n} never)"),
                },
                pacing(&mut gaps),
                at(&cpu, 0.99),
                at(&cpu, 1.0),
                at(&gpu, 0.99),
                at(&gpu, 1.0),
                opt_ms(r.settle_ms),
                r.device_ms,
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Times in ms. Interval: between frames inside a move, from one frame's end to the \
             next's (its GPU work waited for); first tiles and centre from the map's creation; \
             centre: the first frame whose tiles under the view's centre all arrived; after \
             moves: from each move's end to that; settle: from the last input until the map is \
             idle. The slowest intervals' passes: the frame's three costliest GPU passes by \
             label, summed (tiled GPUs overlap passes, whose times show where work waits)."
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| session | frames | pipelines asked | late | made | buffers + textures made | MiB made | MiB \
             copied on the GPU | pool moves | MiB they made | MiB they copied | MiB uploaded | \
             upload calls | texture-write submissions | drape slots | peak frame MiB uploaded | \
             peak frame MiB of tiles | layout ms on arrival frames p50 / max | most \
             chunks left out | most label chunks capped, in frames | MiB of slots moved |"
        )?;
        writeln!(out, "|{}", "---|".repeat(21))?;
        for r in &self.results {
            for what in ["all", "input", "moving"] {
                let keep = |f: &&Frame| match what {
                    "input" => f.report.input,
                    "moving" => f.report.moved || f.report.input,
                    _ => true,
                };
                // The upkeep's moves at rest count with the frame drawn after
                // them, which input may have reached since: not its own.
                let (c, slots, late, n) = summed(r.frames.iter().filter(keep), what == "all");
                // Moves in upkeep between frames happened at rest.
                let mut resized = orb_api::Resized::default();
                for f in r.frames.iter().filter(keep) {
                    resized += f.report.resized;
                    if what == "all" {
                        resized += f.report.prepared_resized;
                    }
                }
                let peak = (r.frames.iter().filter(keep))
                    .map(|f| f.report.gpu.uploaded_bytes())
                    .max()
                    .unwrap_or(0);
                // The candidate layout's stage (with the placement rows),
                // in frames that took tiles (only a drawn frame lays out).
                let mut layout: Vec<Duration> = (r.frames.iter().filter(keep))
                    .filter(|f| f.report.tiles_arrived > 0)
                    .map(|f| Duration::from_micros(u64::from(f.report.times.render.cull)))
                    .collect();
                layout.sort_unstable();
                let layout = match layout.last() {
                    Some(&max) => format!("{} / {}", ms(percentile(&layout, 0.5)), ms(max)),
                    None => "—".into(),
                };
                // Chunks the cull's lists had no room for.
                let left_out = (r.frames.iter().filter(keep))
                    .map(|f| f.report.scene.candidates_dropped)
                    .max()
                    .unwrap_or(0);
                // Label chunks that didn't draw for the placement target.
                let (most_capped, capped_frames) = (r.frames.iter().filter(keep))
                    .map(|f| f.report.scene.labels_capped)
                    .filter(|&c| c > 0)
                    .fold((0, 0), |(most, n), c| (most.max(c), n + 1));
                let slot_moves: u64 = (r.frames.iter().filter(keep))
                    .map(|f| f.report.frame.slot_bytes_moved)
                    .sum();
                let tiles = (r.frames.iter().filter(keep))
                    .map(|f| f.report.tile_bytes)
                    .max()
                    .unwrap_or(0);
                writeln!(
                    out,
                    "| {} {what} | {n} | {} | {late} | {} | {} | {} | {} | {} | {} | {} | {} | {} | \
                     {} | {slots} | {} | {} | {layout} | {left_out} | {most_capped}, \
                     {capped_frames} | {} |",
                    r.name,
                    c.pipelines_requested,
                    c.pipelines_compiled,
                    c.buffers_created + c.textures_created,
                    mib(c.bytes_allocated),
                    mib(c.bytes_copied),
                    resized.moves,
                    mib(resized.bytes_allocated),
                    mib(resized.bytes_copied),
                    mib(c.uploaded_bytes()),
                    c.upload_calls,
                    c.texture_write_submissions,
                    mib(peak),
                    mib(tiles),
                    mib(slot_moves),
                )?;
            }
        }
        writeln!(out)?;
        writeln!(
            out,
            "Each frame's device counts since the frame before. late: pipelines asked for that \
             the style's pipeline set left out. input: frames drawn while input \
             held the camera (a pipeline made then compiled mid-gesture); moving: frames \
             that moved the camera, or with input (memory made then is made while it moves); \
             input and moving leave out what the map's upkeep did at rest before them; pool \
             moves: pools, tables, tile pages and the drape atlas moving into memory of another \
             size, and what they made and copied (part of the device's counts), in the frames \
             (all: and in the map's upkeep between them, at rest); peak frame: the most one frame \
             uploaded, with what was prepared before it; of tiles: the most of tiles' and edits' \
             results one frame uploaded, which its budget's bytes bound; layout: the render's \
             cull stage (the candidates laid out again, and tile placement rows written) in \
             frames that took tiles; chunks left out: the most one frame's cull lists had no \
             room for; label chunks capped: the most one frame's symbol nodes left undrawn \
             past the placement target, which doesn't grow while memory is held, and the \
             frames that left any; slots moved: compacted index slots the candidate layouts moved \
             (natively copied twice, through the scratch buffer)."
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| session | allocated at the end | peak | vertices used / allocated | \
             indices | props | compacted | tables | tile pages | glyph pages | image atlas | drape \
             atlas | cached |"
        )?;
        writeln!(out, "|{}", "---|".repeat(13))?;
        for r in &self.results {
            let Some(end) = r.frames.last().map(|f| f.report.memory) else {
                continue;
            };
            let peak = (r.frames.iter())
                .map(|f| f.report.memory.allocated())
                .max()
                .unwrap_or(0);
            let pool = |p: orb_api::PoolMemory| format!("{} / {}", mib(p.used), mib(p.allocated));
            let s = &end.scene;
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} ({} of {} tiles) |",
                r.name,
                mib(end.allocated()),
                mib(peak),
                pool(s.vertices),
                pool(s.indices),
                pool(s.props),
                pool(s.compacted),
                mib(s.tables),
                mib(end.tile_textures),
                mib(end.glyph_textures),
                mib(end.image_atlas),
                mib(end.drape_atlas),
                mib(end.cached),
                end.cached_tiles,
                end.cache_limit,
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "GPU memory in MiB when the map settled after the moves, and the most it held in \
             any frame; the cache keeps at most as many tiles as the view draws."
        )?;
        for r in &self.results {
            writeln!(out)?;
            writeln!(out, "#### {}", r.name)?;
            writeln!(out)?;
            writeln!(out, "- style: {}", r.style)?;
            let c = &r.camera;
            writeln!(
                out,
                "- the camera ended at {:.4}, {:.4}, z{:.2}, bearing {:.0}, pitch {:.0}",
                c.target.lon,
                c.target.lat,
                c.zoom(&crate::web_mercator()),
                c.bearing,
                c.pitch
            )?;
            if r.compiled_in_moves.is_empty() {
                writeln!(out, "- no pipeline finished compiling during the moves")?;
            } else {
                let list: Vec<String> = (r.compiled_in_moves.iter())
                    .map(|(label, took)| format!("{label} {:.0}", took.as_secs_f64() * 1000.0))
                    .collect();
                writeln!(
                    out,
                    "- {} pipelines finished compiling during the moves (compile ms): {}",
                    list.len(),
                    list.join(", ")
                )?;
            }
            writeln!(out)?;
            writeln!(
                out,
                "| at ms | interval | CPU | GPU | passes | stages | between frames | \
                 pipelines made | MiB made | KiB uploaded | upload calls | tiles arrived | \
                 drape slots |"
            )?;
            writeln!(out, "|{}", "---|".repeat(13))?;
            let mut worst = r.intervals();
            worst.sort_by(|a, b| b.1.total_cmp(&a.1));
            for &(i, gap) in worst.iter().take(WORST) {
                let f = &r.frames[i];
                writeln!(
                    out,
                    "| {:.0} | {gap:.1} | {} | {} | {} | {} | {} | {} | {} | {:.0} | {} | {} | {} |",
                    f.t,
                    ms(f.cpu),
                    f.gpu.map_or("—".into(), ms),
                    (f.passes.iter())
                        .map(|(label, took)| format!("{label} {:.1}", took.as_secs_f64() * 1000.0))
                        .collect::<Vec<_>>()
                        .join(", "),
                    top_stages(&f.report.times),
                    top_stages(&f.report.prepared),
                    f.report.gpu.pipelines_compiled,
                    mib(f.report.gpu.bytes_allocated),
                    f.report.gpu.uploaded_bytes() as f64 / 1024.0,
                    f.report.gpu.upload_calls,
                    f.report.tiles_arrived,
                    f.report.frame.drape_slots,
                )?;
            }
        }
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
        }
        Ok(())
    }
}

/// A frame's three costliest stages of 1 ms or more.
fn top_stages(t: &orb_api::FrameTimes) -> String {
    let r = &t.render;
    let mut stages = [
        ("arrivals", t.arrivals),
        ("upload", t.upload),
        ("terrain", t.terrain),
        ("plan", t.plan),
        ("drape plan", t.drape),
        ("place", t.place),
        ("placement", t.placement),
        ("cull", r.cull),
        ("bundles", r.bundles),
        ("drape passes", r.drape),
        ("encode", r.encode),
        ("submit", r.submit),
    ];
    stages.sort_by_key(|&(_, us)| std::cmp::Reverse(us));
    let named: Vec<String> = (stages.iter())
        .take(3)
        .filter(|&&(_, us)| us >= 1000)
        .map(|&(name, us)| format!("{name} {:.1}", f64::from(us) / 1000.0))
        .collect();
    match named.is_empty() {
        true => format!("total {:.1}", f64::from(t.total) / 1000.0),
        false => named.join(", "),
    }
}
