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
//! display asks for them: a beat whose map has news draws a frame, and
//! input waits for the beat after it comes, as an event loop's would.
//!
//! The sessions, their styles, first cameras and moves, are
//! `web/scripts/sessions.json`'s, which `pnpm session` runs too; its
//! `about` says what each does. A session names its style by the
//! catalogue key `load.html` takes, and a style given the bench stands in
//! for live Liberty. The web demo's own styles (the Globe entry's) open as
//! the demo opens them, with the map options' defaults, unless the session
//! puts the style on the globe, which puts in its projection and sky
//! alone, as `load.html`'s `?globe` does. Over terrain on
//! the plane (`alps`, `rotate`) the report says how still the terrain's
//! cover held between frames ([`crate::steadiness`]). `--dump prefix`
//! writes each session's frames to `<prefix>-<session>.tsv`. `--filter`
//! picks them by name;
//! `--warm` runs each once untimed first and keeps its state directory
//! and shaders, as a user's later launch; `--stale` as well, its HTTP
//! cache made stale, as a launch after the servers' lifetimes lapsed
//! (what a cache revalidates before it answers waits for the server).
//!
//! It reports the frame intervals while input moves the camera, each
//! frame's CPU and GPU time, the time to centre (the first frame whose
//! tiles under the view's centre all arrived, and after each move how
//! long that took), the settle time, what the device did (pipelines asked
//! for, memory made, uploads) over the session, while input held the
//! camera and in frames that moved it, the pipelines that finished
//! compiling during the moves, and what the slowest intervals spent.

use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use glam::DVec2;
use orb_api::{
    Button, CameraState, FrameReport, Geodetic, GpuCounters, Input, Modifiers, PointerKind,
    Runtime, RuntimeOptions, Target, Transaction, WheelDelta,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::report::{ms, percentile};
use crate::{Settings, clock};

/// A 60 Hz display's frame.
const BEAT: Duration = Duration::from_micros(16_667);
/// How long after the first frame drawing tiles input starts.
const START: f64 = 500.0;
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

/// The sessions, as `pnpm session` runs them too.
const SESSIONS: &str = include_str!("../../../web/scripts/sessions.json");
/// The demo's map catalogue, which names the sessions' styles.
const CATALOGUE: &str = include_str!("../../../web/demo/data/scenes.json");
/// The sky a session on the globe has around it, as on the web.
const GLOBE_SKY: &str = include_str!("../../../web/demo/data/globe-sky.json");
/// The demo's map options, whose defaults its own styles open with.
const OPTIONS: &str = include_str!("../../../web/demo/data/options.json");

#[derive(Deserialize)]
struct Sessions {
    sessions: Vec<Session>,
}

/// A session, as `web/scripts/sessions.json` describes its fields.
#[derive(Debug, Deserialize)]
struct Session {
    name: String,
    /// A catalogue key.
    style: String,
    /// Whether it draws on the globe, under [`GLOBE_SKY`].
    #[serde(default)]
    globe: bool,
    /// The first camera; the style's own fills what it leaves out.
    at: crate::StyleCamera,
    /// The pause between moves, in ms.
    pause: f64,
    moves: Vec<Move>,
}

/// One move: points are fractions of the viewport, times in ms.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
enum Move {
    /// `n` notches of `delta` pixels at a point, `ms` apart: negative
    /// zooms in.
    Wheel {
        at: [f64; 2],
        delta: f64,
        n: u32,
        ms: f64,
    },
    /// The primary button from a point by `by` logical pixels over `ms`.
    Drag {
        from: [f64; 2],
        by: [f64; 2],
        ms: f64,
    },
    /// The secondary button from the centre: a tilt and turn.
    Turn { by: [f64; 2], ms: f64 },
    /// `levels` zoomed in (out, negative) by notches `ms` apart, each a
    /// third of a level (100 px heard at 1/300 a pixel).
    Fling { levels: f64, ms: f64 },
    /// `levels` zoomed by a trackpad's scroll, 8 px heard every 8 ms.
    Scroll { levels: f64 },
    /// The camera's centre carried straight from one lon, lat to another,
    /// a step every 16 ms, as an animation carries it.
    Fly {
        from: [f64; 2],
        to: [f64; 2],
        ms: f64,
    },
}

/// What a session does at a time: input, or the camera's centre set.
#[derive(Debug)]
enum Step {
    Input(Input),
    Centre([f64; 2]),
}

const C: [f64; 2] = [0.5, 0.5];

/// The demo's catalogue (`web/demo/data/scenes.json`): each entry's key
/// and style URL, relative to the data directory for the demo's own.
#[derive(Deserialize)]
struct Catalogue {
    groups: Vec<Group>,
}

#[derive(Deserialize)]
struct Group {
    scenes: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    key: String,
    url: String,
    /// The switches its style designs that start off.
    #[serde(default)]
    off: Vec<String>,
}

/// A map option (`web/demo/data/options.json`), as far as its default
/// reads it.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Opt {
    key: String,
    #[serde(default)]
    orbis_only: bool,
}

/// A session's style, found by its catalogue key.
#[derive(Debug)]
struct Style {
    /// What the map loads.
    url: String,
    /// The file in the demo data it is, where it is one.
    file: Option<PathBuf>,
    /// Whether it is the demo's own (a catalogue entry in the data
    /// directory), which the web opens with the map options' defaults.
    own: bool,
    /// The switches its catalogue entry starts off.
    off: Vec<String>,
}

impl Style {
    /// The style `key` names as `load.html` takes it, in the demo data
    /// directory `data`: `alps` (`styles.js`'s entry by key only) or a
    /// catalogue entry, `liberty` standing in for live Liberty.
    fn find(key: &str, data: &Path, liberty: &str) -> Result<Self, String> {
        let local = |file: PathBuf, own, off| Self {
            url: crate::file_url(&file),
            file: Some(file),
            own,
            off,
        };
        if key == "alps" {
            return Ok(local(data.join("alps.json"), false, Vec::new()));
        }
        let catalogue: Catalogue = serde_json::from_str(CATALOGUE).map_err(|e| e.to_string())?;
        let entry = (catalogue.groups.iter().flat_map(|g| &g.scenes))
            .find(|e| e.key == key)
            .ok_or_else(|| format!("sessions.json: no catalogue entry {key}"))?;
        if !entry.url.contains("://") {
            return Ok(local(data.join(&entry.url), true, entry.off.clone()));
        }
        Ok(Self {
            url: match entry.url == crate::LIBERTY {
                true => liberty.to_string(),
                false => entry.url.clone(),
            },
            file: None,
            own: false,
            off: Vec::new(),
        })
    }
}

impl Session {
    /// The session's first camera: its `at`, the style's own filling what
    /// it leaves out.
    fn camera(&self, style: &Style) -> Result<CameraState, String> {
        let at = match (self.at.state(), &style.file) {
            (Some(camera), _) => return Ok(camera),
            (None, Some(file)) => self.at.or(crate::StyleCamera::read(file)?),
            (None, None) => self.at,
        };
        at.state()
            .ok_or_else(|| format!("{}: its camera needs a centre and zoom", self.name))
    }

    /// The edits the session makes to its style once it arrives (`style`,
    /// as JSON): on the globe, its projection and sky alone, as
    /// `load.html`'s `?globe` puts them in; else the web demo's defaults
    /// for its own.
    fn opening(&self, style: &Style, json: &str) -> Result<Option<Transaction>, String> {
        let mut commands = Vec::new();
        if style.own && !self.globe {
            defaults_known(&self.name, style)?;
            let doc: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
            commands.extend(unblended(&doc));
        }
        if self.globe {
            let sky: Value = serde_json::from_str(GLOBE_SKY).map_err(|e| e.to_string())?;
            commands.push(json!(["setProjection", {"type": "globe"}]));
            commands.push(json!(["setSky", sky]));
        }
        if commands.is_empty() {
            return Ok(None);
        }
        Transaction::from_commands(&Value::Array(commands).to_string()).map(Some)
    }
}

/// Whether [`unblended`] is all the web demo's map options change of
/// `style` by default (`web/demo/app/options.js`'s `opening`): an
/// Orbis-only option besides the blend, or a switch the catalogue entry
/// starts off, would draw the session differently from the web's until
/// the bench learns it.
fn defaults_known(session: &str, style: &Style) -> Result<(), String> {
    let options: Vec<Opt> = serde_json::from_str(OPTIONS).map_err(|e| e.to_string())?;
    if let Some(option) = options.iter().find(|o| o.orbis_only && o.key != "blend") {
        return Err(format!(
            "{session}: the bench opens no Orbis-only option but the blend off, not {}",
            option.key
        ));
    }
    match style.off.first() {
        Some(key) => Err(format!(
            "{session}: the bench starts no switch off as its style's catalogue entry does {key}"
        )),
        None => Ok(()),
    }
}

/// What the web demo's map options change of its own styles by default
/// (`web/demo/app/options.js`'s `opening`): their one Orbis-only option,
/// style-spec#1661's blend of a hillshade or colour relief, starts off,
/// so each blended layer loses its blend and takes the paint its
/// `demo:unblended` metadata keeps for it drawn normally.
fn unblended(style: &Value) -> Vec<Value> {
    const BLENDS: [(&str, &str); 2] = [
        ("hillshade", "dev.sargunv.orbis:hillshade-layer-blend"),
        ("color-relief", "dev.sargunv.orbis:color-relief-layer-blend"),
    ];
    let mut commands = Vec::new();
    for layer in style["layers"].as_array().into_iter().flatten() {
        let Some(&(_, blend)) = BLENDS.iter().find(|(kind, _)| layer["type"] == *kind) else {
            continue;
        };
        if matches!(layer["paint"].get(blend), None | Some(Value::Null))
            || layer["paint"][blend] == "normal"
        {
            continue;
        }
        let id = &layer["id"];
        for (name, value) in layer["metadata"]["demo:unblended"]
            .as_object()
            .into_iter()
            .flatten()
        {
            commands.push(json!(["setPaintProperty", id, name, value]));
        }
        commands.push(json!(["setPaintProperty", id, blend, null]));
    }
    commands
}

/// The sessions sessions.json describes.
fn sessions() -> Result<Vec<Session>, String> {
    let parsed: Sessions =
        serde_json::from_str(SESSIONS).map_err(|e| format!("sessions.json: {e}"))?;
    Ok(parsed.sessions)
}

/// The inputs a session's moves make from `start` (ms), each at its
/// time, and each move's window.
fn timeline(
    moves: &[Move],
    pause: f64,
    viewport: [f64; 2],
    start: f64,
) -> (Vec<(f64, Step)>, Vec<[f64; 2]>) {
    let px = |[x, y]: [f64; 2]| DVec2::new(x * viewport[0], y * viewport[1]);
    let mut inputs = Vec::new();
    let mut windows = Vec::new();
    let mut t = start;
    for &m in moves {
        let from = t;
        // `n` notches of `delta` pixels sent at `at`, `ms` apart.
        let mut notches = |at: [f64; 2], delta: f64, n: u32, ms: f64| {
            for i in 0..n {
                inputs.push((
                    from + f64::from(i) * ms,
                    Step::Input(Input::Wheel {
                        position: px(at),
                        delta: WheelDelta::Pixels(delta * NOTCH),
                        modifiers: Modifiers::default(),
                    }),
                ));
            }
            from + f64::from(n.max(1)) * ms
        };
        // `levels` zoomed by notches of `heard` pixels, `ms` apart, at
        // 1/300 a pixel.
        let mut zoom = |levels: f64, heard: f64, ms: f64| {
            let n = (levels.abs() * 300.0 / heard).round() as u32;
            notches(C, -levels.signum() * heard / NOTCH, n, ms)
        };
        t = match m {
            Move::Wheel { at, delta, n, ms } => notches(at, delta, n, ms),
            Move::Fling { levels, ms } => zoom(levels, 100.0, ms),
            Move::Scroll { levels } => zoom(levels, 8.0, 8.0),
            Move::Drag { from: at, by, ms } => {
                drag(&mut inputs, (px(at), by, Button::Primary), from, ms)
            }
            Move::Turn { by, ms } => drag(&mut inputs, (px(C), by, Button::Secondary), from, ms),
            Move::Fly { from: a, to: b, ms } => {
                let steps = (ms / 16.0).ceil() as u32;
                for i in 0..=steps {
                    let f = f64::from(i) / f64::from(steps);
                    let centre = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
                    inputs.push((from + f * ms, Step::Centre(centre)));
                }
                from + ms
            }
        };
        windows.push([from, t]);
        t += pause;
    }
    (inputs, windows)
}

/// A drag with `button` held from `at` by `by` logical pixels, starting
/// at `from` (ms) and moving over `ms`, held still 150 ms before the
/// release: when it ends.
fn drag(
    inputs: &mut Vec<(f64, Step)>,
    (at, by, button): (DVec2, [f64; 2], Button),
    from: f64,
    ms: f64,
) -> f64 {
    let by = DVec2::from(by);
    inputs.push((
        from,
        Step::Input(Input::PointerDown {
            id: 1,
            kind: PointerKind::Mouse,
            button,
            position: at,
            modifiers: Modifiers::default(),
        }),
    ));
    let steps = (ms / 16.0).ceil() as u32;
    for i in 1..=steps {
        let f = f64::from(i) / f64::from(steps);
        inputs.push((
            from + f * ms,
            Step::Input(Input::PointerMove {
                id: 1,
                position: at + by * f,
            }),
        ));
    }
    let end = from + ms + 150.0;
    inputs.push((
        end,
        Step::Input(Input::PointerUp {
            id: 1,
            position: at + by,
        }),
    ));
    end
}

/// One frame drawn: when it ended (ms on the session's clock), its CPU
/// and GPU time and its report.
#[derive(Debug)]
struct Frame {
    t: f64,
    cpu: Duration,
    gpu: Option<Duration>,
    /// Its costliest passes on the GPU, summed by label.
    passes: Vec<(String, Duration)>,
    /// The camera's zoom once drawn.
    zoom: f64,
    report: FrameReport,
    /// The terrain it drew, on a plane (the steadiness reads it).
    terrain: Option<orb_runtime::TerrainProbe>,
}

/// A session's run.
#[derive(Debug)]
pub(crate) struct SessionResult {
    name: String,
    style: String,
    /// How it launched: "" cold, else " (warm)" or " (warm, stale)".
    launch: &'static str,
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
/// their own: `liberty` stands in for live Liberty, and `data` is the
/// demo data directory, which holds the demo's own styles.
pub(crate) fn run(
    settings: &Settings,
    about: Vec<String>,
    liberty: &str,
    data: &Path,
    mut progress: impl FnMut(&str),
) -> Result<SessionReport, String> {
    let mut results = Vec::new();
    let mut warnings = Vec::new();
    let sessions = sessions()?;
    let chosen = (sessions.iter())
        .filter(|s| (settings.filter.as_ref()).is_none_or(|f| s.name.contains(f.as_str())));
    for session in chosen {
        let style = Style::find(&session.style, data, liberty)?;
        let state = StateDir::new()?;
        let nonce = Some(std::process::id() ^ clock::now().subsec_nanos());
        if settings.warm {
            progress(&format!("{}: once untimed", session.name));
            run_session(
                settings,
                session,
                &style,
                (&state, nonce, false),
                &mut warnings,
            )?;
        }
        let launch = match (settings.warm, settings.stale) {
            (false, _) => "",
            (true, false) => " (warm)",
            (true, true) => " (warm, stale)",
        };
        progress(&format!("{}:{}", session.name, launch));
        let stale = settings.stale;
        let mut result = run_session(
            settings,
            session,
            &style,
            (&state, nonce, stale),
            &mut warnings,
        )?;
        result.launch = launch;
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
    style: &Style,
    (state, nonce, stale): (&StateDir, Option<u32>, bool),
    warnings: &mut Vec<String>,
) -> Result<SessionResult, String> {
    let made = Instant::now();
    let runtime = Runtime::new(&RuntimeOptions {
        state_dir: Some(state.0.clone()),
        ..RuntimeOptions::default()
    })
    .map_err(|e| e.to_string())?;
    if stale {
        runtime.invalidate_http_cache();
    }
    runtime.set_shader_nonce(nonce);
    let device_ms = made.elapsed().as_secs_f64() * 1000.0;
    let (width, height) = settings.physical_size();
    let mut target = Target::texture(&runtime, width, height, f64::from(settings.ratio));
    let camera = session.camera(style)?;
    let mut map = crate::new_map(&runtime, &style.url, camera, settings.antialias);
    // The style arrives before the first frame: its sprites load at the
    // ratio the frames draw at.
    map.set_viewport(
        f64::from(settings.width),
        f64::from(settings.height),
        f64::from(settings.ratio),
    );
    // Whether the style has yet to arrive for its opening edits.
    let mut opening = true;
    let gpu_timestamps = runtime.stats().gpu_timestamps;
    runtime.set_gpu_timing(gpu_timestamps);
    let epoch = Instant::now();
    let now = || epoch.elapsed().as_secs_f64() * 1000.0;
    let viewport = [f64::from(settings.width), f64::from(settings.height)];
    let mut frames: Vec<Frame> = Vec::new();
    let mut inputs: std::collections::VecDeque<(f64, Step)> = Default::default();
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
            match inputs.pop_front().expect("checked") {
                (at, Step::Input(input)) => {
                    map.input(input, at);
                }
                (_, Step::Centre([lon, lat])) => map.jump_to(&orb_api::CameraTarget {
                    center: Some(Geodetic::new(lon, lat)),
                    ..orb_api::CameraTarget::default()
                }),
            }
        }
        if opening {
            map.apply_arrivals().map_err(|e| e.to_string())?;
            if let Some(json) = map.style_json() {
                if let Some(tx) = session.opening(style, &json)? {
                    map.edit(&tx).map_err(|e| e.to_string())?;
                }
                opening = false;
            }
        }
        if map.needs_frame() || frames.is_empty() {
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
                let (moves, w) = timeline(&session.moves, session.pause, viewport, t + START);
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
                // Kept, and measured once the session ends.
                terrain: (map.surface().globe.is_none())
                    .then(|| map.terrain_probe())
                    .flatten(),
            });
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
                if (frames.last()).is_some_and(|f| f.report.rests() && f.t > end) {
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
    // candidates, CPU and GPU ms, draws, drape slots redrawn, whether
    // input moved the camera and the tiles that arrived since the frame
    // before, for attributing slow frames and seeing loads come again;
    // the MiB the device made, and the drape atlas's and tile pages' MiB
    // after it, for seeing pages made anew; the vertex, index and
    // compacted index pools' MiB used and held, and all the map holds;
    // over terrain on the plane,
    // what it changed of the frame before ([`crate::steadiness`]): the
    // share of the ground re-zoomed, tile drops and the near ground's
    // wobble in pixels.
    if let Some(prefix) = &settings.dump {
        let mut out = String::from(
            "t\tzoom\ttiles\tcandidates\tcpu\tgpu\tdraws\tdrape_slots\tmoving\tarrived\tmade\t\
             drape_atlas\ttile_pages\tvertices\tvertices_held\tindices\tindices_held\t\
             compacted\tcompacted_held\tallocated\trezoomed\tdrops\twobble\n",
        );
        for (i, f) in frames.iter().enumerate() {
            let moving = windows.iter().any(|w| (w[0]..=w[1]).contains(&f.t));
            let scene = &f.report.memory.scene;
            let before = i.checked_sub(1).and_then(|p| frames[p].terrain.as_ref());
            let steadiness = match before.zip(f.terrain.as_ref()) {
                Some((a, b)) => {
                    let c = crate::steadiness::change(a, b);
                    format!("{:.4}\t{}\t{:.2}", c.rezoomed_share(), c.drops, c.wobble_px)
                }
                None => "\t\t".into(),
            };
            writeln!(
                out,
                "{:.1}\t{:.3}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
                 {steadiness}",
                f.t,
                f.zoom,
                f.report.tiles_drawn,
                f.report.frame.candidates,
                f.cpu.as_secs_f64() * 1000.0,
                f.gpu.map_or(0.0, |g| g.as_secs_f64() * 1000.0),
                f.report.frame.draws,
                f.report.frame.drape_slots,
                u8::from(moving),
                f.report.tiles_arrived,
                mib(f.report.gpu.bytes_allocated),
                mib(f.report.memory.drape_atlas),
                mib(f.report.memory.tile_textures),
                mib(scene.vertices.used),
                mib(scene.vertices.allocated),
                mib(scene.indices.used),
                mib(scene.indices.allocated),
                mib(scene.compacted.used),
                mib(scene.compacted.allocated),
                mib(f.report.memory.allocated()),
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
        name: session.name.clone(),
        style: match session.globe {
            true => format!("{} on the globe", style.url),
            false => style.url.clone(),
        },
        launch: "",
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

/// The frames' device counts summed, their drape slots redrawn, their
/// pipelines asked for outside the style's set, and how many they are.
fn summed<'a>(frames: impl Iterator<Item = &'a Frame>) -> (GpuCounters, u32, u32, usize) {
    let mut sum = GpuCounters::default();
    let (mut slots, mut late, mut n) = (0, 0, 0);
    for f in frames {
        sum += f.report.gpu;
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
             empty state directory and shaders unique to it (`--warm`: its second run, \
             `--stale`: with its HTTP cache made stale), input \
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
                r.launch,
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
             label, summed."
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| session | frames | pipelines asked | late | made | buffers + textures made | MiB made | MiB \
             copied on the GPU | pool moves | MiB they made | MiB they copied | MiB uploaded | \
             upload calls | texture-write submissions | render commands | peak frame commands | drape \
             slots | peak frame MiB uploaded | \
             peak frame MiB of tiles | layout ms on arrival frames p50 / max | most \
             chunks left out | MiB of slots moved |"
        )?;
        writeln!(out, "|{}", "---|".repeat(22))?;
        for r in &self.results {
            for what in ["all", "input", "moving"] {
                let keep = |f: &&Frame| match what {
                    "input" => f.report.input,
                    "moving" => f.report.moved || f.report.input,
                    _ => true,
                };
                let (c, slots, late, n) = summed(r.frames.iter().filter(keep));
                let mut resized = orb_api::Resized::default();
                for f in r.frames.iter().filter(keep) {
                    resized += f.report.resized;
                }
                let peak = (r.frames.iter().filter(keep))
                    .map(|f| f.report.gpu.uploaded_bytes())
                    .max()
                    .unwrap_or(0);
                let commands = (r.frames.iter().filter(keep))
                    .map(|f| f.report.gpu.render_commands)
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
                     {} | {} | {commands} | {slots} | {} | {} | {layout} | {left_out} | {} |",
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
                    c.render_commands,
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
             pool moves: blocks made, and tables, tile pages and the drape atlas moving into \
             memory of another size, and what they made and copied (part of the device's \
             counts); render commands: encoded into passes and bundles, each a call into \
             the graphics API (on the web, into the browser), and the most one frame encoded \
             (peak frame commands); \
             peak frame: the most one frame \
             uploaded; of tiles: the most of tiles' and edits' \
             results one frame uploaded, which its budget's bytes bound; layout: the render's \
             cull stage (the candidates laid out again, and tile placement rows written) in \
             frames that took tiles; chunks left out: the most one frame's cull lists had no \
             room for; slots moved: compacted index slots the candidate layouts moved \
             (natively copied twice, through the scratch buffer)."
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| session | allocated at the end | peak | vertices used / allocated | \
             indices | props | compacted | tables | tile pages | glyph pages | image atlas | drape \
             atlas | cached | peak vertices used / allocated | peak indices | peak compacted | \
             peak tile pages | peak drape atlas |"
        )?;
        writeln!(out, "|{}", "---|".repeat(18))?;
        for r in &self.results {
            let Some(end) = r.frames.last().map(|f| f.report.memory) else {
                continue;
            };
            let peak = (r.frames.iter())
                .map(|f| f.report.memory.allocated())
                .max()
                .unwrap_or(0);
            let pool = |p: orb_api::PoolMemory| format!("{} / {}", mib(p.used), mib(p.allocated));
            // Each pool's most used and most allocated in any frame.
            let peak_pool = |of: fn(&orb_api::Memory) -> orb_api::PoolMemory| {
                let (used, allocated) = (r.frames.iter())
                    .map(|f| of(&f.report.memory))
                    .fold((0, 0), |(u, a), p| (p.used.max(u), p.allocated.max(a)));
                format!("{} / {}", mib(used), mib(allocated))
            };
            let most = |of: fn(&orb_api::Memory) -> u64| {
                mib((r.frames.iter())
                    .map(|f| of(&f.report.memory))
                    .max()
                    .unwrap_or(0))
            };
            let s = &end.scene;
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} ({} of {} tiles) | {} | {} | {} | {} | {} |",
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
                peak_pool(|m| m.scene.vertices),
                peak_pool(|m| m.scene.indices),
                peak_pool(|m| m.scene.compacted),
                most(|m| m.tile_textures),
                most(|m| m.drape_atlas),
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "GPU memory in MiB when the map settled after the moves, and the most it held in \
             any frame (a pool's peaks: the most it used and the most it held, each in any \
             frame); the cache keeps at most two views' worth of tiles."
        )?;
        self.write_fetches(out)?;
        self.write_steadiness(out)?;
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
                "| at ms | interval | CPU | GPU | passes | stages | \
                 pipelines made | MiB made | KiB uploaded | upload calls | render commands | \
                 tiles arrived | drape slots |"
            )?;
            writeln!(out, "|{}", "---|".repeat(13))?;
            let mut worst = r.intervals();
            worst.sort_by(|a, b| b.1.total_cmp(&a.1));
            for &(i, gap) in worst.iter().take(WORST) {
                let f = &r.frames[i];
                writeln!(
                    out,
                    "| {:.0} | {gap:.1} | {} | {} | {} | {} | {} | {} | {:.0} | {} | {} | {} | {} |",
                    f.t,
                    ms(f.cpu),
                    f.gpu.map_or("—".into(), ms),
                    (f.passes.iter())
                        .map(|(label, took)| format!("{label} {:.1}", took.as_secs_f64() * 1000.0))
                        .collect::<Vec<_>>()
                        .join(", "),
                    top_stages(&f.report.times),
                    f.report.gpu.pipelines_compiled,
                    mib(f.report.gpu.bytes_allocated),
                    f.report.gpu.uploaded_bytes() as f64 / 1024.0,
                    f.report.gpu.upload_calls,
                    f.report.gpu.render_commands,
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

impl SessionReport {
    /// The tile fetches each session started, aborted and completed, in
    /// all and after its last input.
    fn write_fetches(&self, out: &mut String) -> std::fmt::Result {
        writeln!(
            out,
            "| session | fetches started | aborted | completed | started after the last input | \
             tiles in after the last input ms | nodes waiting on pipelines then | \
             loaded after it ms |"
        )?;
        writeln!(out, "|{}", "---|".repeat(8))?;
        for r in &self.results {
            let mut all = orb_api::FetchCounts::default();
            let mut after = 0;
            let end = r.windows.last().map_or(f64::INFINITY, |w| w[1]);
            for f in &r.frames {
                all += f.report.fetches;
                if f.t > end {
                    after += f.report.fetches.started;
                }
            }
            // The first frame after the last input with every tile the
            // view needs in.
            let tiles_in = (r.frames.iter()).find(|f| f.t > end && f.report.tiles_loading == 0);
            let waiting = tiles_in.map_or(0, |f| f.report.frame.nodes_waiting);
            let tiles_in = tiles_in.map(|f| f.t - end);
            let loaded = (r.frames.iter())
                .find(|f| f.t > end && f.report.loaded)
                .map(|f| f.t - end);
            writeln!(
                out,
                "| {} | {} | {} | {} | {after} | {} | {waiting} | {} |",
                r.name,
                all.started,
                all.aborted,
                all.completed,
                opt_ms(tiles_in),
                opt_ms(loaded)
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Tile fetches the map's jobs started, aborted once started (the view left their \
             tiles), and completed; tiles in: from the last input to the first frame with \
             nothing the view needs loading, and how many of its nodes (layers' draws) \
             waited on pipelines still compiling there; loaded: to the first with its labels placed and every \
             layer drawn too (GL JS's `loaded()`; fades may run on). Cold, under load, the \
             pipelines of the first view's labels may compile seconds after its tiles came."
        )?;
        writeln!(out)
    }

    /// How still the terrain's cover held between successive frames, in
    /// the sessions on the plane with terrain.
    fn write_steadiness(&self, out: &mut String) -> std::fmt::Result {
        let measured: Vec<_> = (self.results.iter())
            .filter(|r| r.frames.iter().any(|f| f.terrain.is_some()))
            .collect();
        if measured.is_empty() {
            return Ok(());
        }
        writeln!(
            out,
            "| session | frame pairs | ground re-zoomed mean | p95 | max | pairs re-zooming \
             | ground coarser mean | tile drops | pairs dropping | near wobble \
             p50 px | p95 | max |"
        )?;
        writeln!(out, "|{}", "---|".repeat(12))?;
        for r in measured {
            for what in ["all", "input"] {
                let changes: Vec<crate::steadiness::Change> = (r.frames.windows(2))
                    .filter(|w| what == "all" || w[1].report.input)
                    .filter_map(|w| {
                        let (a, b) = (w[0].terrain.as_ref()?, w[1].terrain.as_ref()?);
                        Some(crate::steadiness::change(a, b))
                    })
                    .collect();
                let n = changes.len().max(1) as f64;
                let mut shares: Vec<f64> = changes.iter().map(|c| c.rezoomed_share()).collect();
                let mut wobble: Vec<f64> = changes.iter().map(|c| c.wobble_px).collect();
                shares.sort_by(f64::total_cmp);
                wobble.sort_by(f64::total_cmp);
                let at = |v: &[f64], q: f64| match v.is_empty() {
                    true => 0.0,
                    false => v[((v.len() - 1) as f64 * q).round() as usize],
                };
                let coarser: f64 = (changes.iter())
                    .map(|c| f64::from(c.coarser) / f64::from(c.points.max(1)))
                    .sum();
                writeln!(
                    out,
                    "| {} {what} | {} | {:.2}% | {:.1}% | {:.1}% | {} | {:.2}% | {} | {} | {:.2} | \
                     {:.2} | {:.2} |",
                    r.name,
                    changes.len(),
                    100.0 * shares.iter().sum::<f64>() / n,
                    100.0 * at(&shares, 0.95),
                    100.0 * at(&shares, 1.0),
                    changes.iter().filter(|c| c.rezoomed > 0).count(),
                    100.0 * coarser / n,
                    changes.iter().map(|c| c.drops).sum::<u32>(),
                    changes.iter().filter(|c| c.drops > 0).count(),
                    at(&wobble, 0.5),
                    at(&wobble, 0.95),
                    at(&wobble, 1.0),
                )?;
            }
        }
        writeln!(out)?;
        writeln!(
            out,
            "Terrain steadiness between successive frames drawn (input: the later one drawn \
             while input held the camera), sampled at a {}×{} grid of the screen: the share of \
             the earlier frame's ground whose terrain tile took another zoom in the later one, \
             and a coarser one; tile drops: the earlier cover's tiles an ancestor replaced; near \
             wobble: the most a ground point in the screen's bottom quarter moved on screen as \
             its elevation changed, beyond the camera's move.",
            crate::steadiness::GRID[0],
            crate::steadiness::GRID[1],
        )?;
        writeln!(out)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every session `pnpm session` runs, natively: its style found in
    /// the demo data, a first camera, and opening edits a map takes.
    #[test]
    fn sessions_run_natively() {
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/demo/data");
        let sessions = sessions().unwrap();
        assert!(!sessions.is_empty());
        for session in &sessions {
            let style = Style::find(&session.style, &data, crate::LIBERTY).unwrap();
            session.camera(&style).unwrap();
            let json = match &style.file {
                Some(file) => std::fs::read_to_string(file).unwrap(),
                None => r#"{"version": 8, "sources": {}, "layers": []}"#.into(),
            };
            session.opening(&style, &json).unwrap();
            assert!(!session.moves.is_empty(), "{}", session.name);
        }
    }

    /// The Alps sessions open at the Alps style's own camera, `rotate`
    /// with its own zoom and pitch over it.
    #[test]
    fn a_camera_left_out_is_the_styles() {
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/demo/data");
        let own = crate::StyleCamera::read(&data.join("alps.json")).unwrap();
        let sessions = sessions().unwrap();
        let camera = |name: &str| {
            let session = sessions.iter().find(|s| s.name == name).unwrap();
            session
                .camera(&Style::find(&session.style, &data, crate::LIBERTY).unwrap())
                .unwrap()
        };
        assert_eq!(camera("alps"), own.state().unwrap());
        let rotate = camera("rotate");
        assert_eq!(rotate.target, own.state().unwrap().target);
        let at = sessions.iter().find(|s| s.name == "rotate").unwrap().at;
        assert_eq!(
            (rotate.pitch, rotate.bearing),
            (at.pitch.unwrap(), own.bearing.unwrap())
        );
    }
}
