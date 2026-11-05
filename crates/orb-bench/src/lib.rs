//! GPU and CPU frame timings through `orb-api` (`cargo xtask bench`), as
//! hosts draw.
//!
//! By default one map visits each view, loads until it is idle and renders
//! `rounds` runs of still frames offscreen; `experiment=gestures` times
//! rotate, tilt, zoom and pan paths with tiles arriving. [`run`] returns a
//! [`Report`], which formats as Markdown.

#![forbid(unsafe_code)]

mod clock;
mod gestures;
mod report;

use std::time::Duration;

pub use gestures::GesturesReport;
use orb_api::{CameraState, Event, FrameReport, Geodetic, Map, MapEvent, Runtime, Target};
use report::{FramesReport, Summary, ViewResult};

/// The committed Tokyo extract's view, by the Imperial Palace
/// (`web/demo/data/README.md`).
pub const TOKYO: (f64, f64) = (139.74, 35.685);
/// The terrain demo's view over the Valais Alps: Zermatt and the
/// Matterhorn, as `web/demo/data/alps-flight.json` ends.
pub const ALPS: (f64, f64) = (7.7491, 46.0207);
/// Name, zoom, pitch, and whether the style gains [`BUILDINGS_3D`].
const VIEWS: &[(&str, f64, f64, bool)] = &[
    ("z12", 12.0, 0.0, false),
    ("z15", 15.0, 0.0, false),
    ("z15 pitch 60", 15.0, 60.0, false),
    ("z17", 17.0, 0.0, false),
    ("z16 pitch 60, 3D buildings", 16.0, 60.0, true),
];
/// OpenFreeMap Liberty's extrusion layer, which the reduced style leaves
/// out, added before the labels.
const BUILDINGS_3D: &str = r#"{"id": "building-3d", "type": "fill-extrusion", "source": "openmaptiles", "source-layer": "building", "minzoom": 14, "filter": ["!", ["has", "hide_3d"]], "paint": {"fill-extrusion-base": ["get", "render_min_height"], "fill-extrusion-color": "hsl(35,8%,85%)", "fill-extrusion-height": ["get", "render_height"], "fill-extrusion-opacity": 0.8}}"#;
/// How long a view may take to load.
const LOAD_TIMEOUT: Duration = Duration::from_secs(120);

/// What the bench compares.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Experiment {
    /// Still frames of each view.
    #[default]
    Frames,
    /// Rotate, tilt and zoom paths at a 60 Hz pace with tiles arriving:
    /// frame-time distributions.
    Gestures,
}

/// The gestures bench's camera paths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GesturePaths {
    /// Rotate and tilt at z14, zoom z12 → z16 → z12: a city's streets.
    #[default]
    City,
    /// The terrain demo's view over the Alps (`web/demo/data/alps.json`):
    /// rotate and pan at z11.6, pitch 72, and its 60 s flight.
    Alps,
}

/// A bench's results.
#[derive(Debug)]
pub enum Report {
    Frames(FramesReport),
    Gestures(GesturesReport),
}

impl Report {
    /// The report as Markdown.
    pub fn markdown(&self) -> String {
        let mut out = String::new();
        match self {
            Self::Frames(r) => r.write(&mut out),
            Self::Gestures(r) => r.write(&mut out),
        }
        .expect("writing to a String");
        out
    }
}

/// How GPU time is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Timing {
    /// Timestamp queries at the beginning and end of every pass, two frames
    /// in flight.
    Timestamps,
    /// A proxy for devices without timestamp queries: wall time from the
    /// start of each frame until the GPU reports its work done, one frame
    /// in flight. It includes the CPU's work and the queue's latency, so it
    /// is not GPU time.
    WorkDone,
}

/// What to measure, and how.
#[derive(Clone, Debug)]
pub struct Settings {
    /// Measured frames per submission per round.
    pub frames: usize,
    /// Frames rendered before measuring, per submission per round.
    pub warmup: usize,
    pub rounds: usize,
    /// Viewport in logical pixels.
    pub width: u32,
    pub height: u32,
    pub ratio: f32,
    pub center: Geodetic,
    /// `None` uses timestamps where the device has them.
    pub timing: Option<Timing>,
    pub experiment: Experiment,
    pub paths: GesturePaths,
    /// The frames bench's views as `(zoom, pitch, bearing)`, in place of
    /// its own (`--view`, repeatable).
    pub views: Vec<[f64; 3]>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            frames: 100,
            warmup: 50,
            rounds: 3,
            width: 1024,
            height: 768,
            ratio: 2.0,
            center: Geodetic::new(TOKYO.0, TOKYO.1),
            timing: None,
            experiment: Experiment::Frames,
            paths: GesturePaths::City,
            views: Vec::new(),
        }
    }
}

impl Settings {
    /// Sets one option by name, as the tool's `--name value` gives it: `frames`, `warmup`, `rounds`, `size`
    /// (`WxH`), `ratio`, `at` (`lon,lat`), `timing` (`timestamps` or
    /// `work-done`), `experiment` (`frames` or `gestures`), `paths` (`city` or `alps`,
    /// the gestures bench's) or `view` (`zoom,pitch,bearing`, adding a view
    /// to measure instead of the frames bench's own).
    pub fn set(&mut self, name: &str, value: &str) -> Result<(), String> {
        let number = |value: &str| {
            value
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("{name}: {e}"))
        };
        let count = |value: &str| {
            value
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("{name}: {e}"))
        };
        match name {
            "frames" => self.frames = count(value)?.max(1),
            "warmup" => self.warmup = count(value)?,
            "rounds" => self.rounds = count(value)?.max(1),
            "ratio" => self.ratio = number(value)? as f32,
            "size" => {
                let (w, h) = value.split_once('x').ok_or("size needs WxH")?;
                self.width = count(w)? as u32;
                self.height = count(h)? as u32;
            }
            "view" => {
                let v: Vec<f64> = value.split(',').map(number).collect::<Result<_, _>>()?;
                let [zoom, pitch, bearing] = v[..] else {
                    return Err("view needs zoom,pitch,bearing".into());
                };
                self.views.push([zoom, pitch, bearing]);
            }
            "at" => {
                let (lon, lat) = value.split_once(',').ok_or("at needs lon,lat")?;
                self.center = Geodetic::new(number(lon)?, number(lat)?);
            }
            "timing" => {
                self.timing = Some(match value {
                    "timestamps" => Timing::Timestamps,
                    "work-done" => Timing::WorkDone,
                    _ => return Err("timing needs timestamps or work-done".into()),
                })
            }
            "experiment" => {
                self.experiment = match value {
                    "frames" => Experiment::Frames,
                    "gestures" => Experiment::Gestures,
                    _ => return Err("experiment needs frames or gestures".into()),
                }
            }
            "paths" => {
                self.paths = match value {
                    "city" => GesturePaths::City,
                    "alps" => GesturePaths::Alps,
                    _ => return Err("paths needs city or alps".into()),
                }
            }
            _ => return Err(format!("unknown option {name}")),
        }
        Ok(())
    }

    fn physical_size(&self) -> (u32, u32) {
        (
            (self.width as f32 * self.ratio).round() as u32,
            (self.height as f32 * self.ratio).round() as u32,
        )
    }
}

/// How `runtime` times the GPU: `asked`, or timestamps where it has them.
fn timing(runtime: &Runtime, asked: Option<Timing>) -> Result<Timing, String> {
    let timestamps = runtime.stats().gpu_timestamps;
    match asked {
        Some(Timing::Timestamps) if !timestamps => {
            Err("the device has no timestamp queries".into())
        }
        Some(timing) => Ok(timing),
        None if timestamps => Ok(Timing::Timestamps),
        None => Ok(Timing::WorkDone),
    }
}

/// A view's still frames.
#[derive(Default)]
struct Samples {
    /// GPU time per frame (every pass), or the proxy.
    gpu: Vec<Duration>,
    /// Passes alone; empty under the proxy. Extrusions' offscreen pass only
    /// where the view has them.
    cull: Vec<Duration>,
    main: Vec<Duration>,
    extrusion: Vec<Duration>,
    /// `Map::render`.
    render_cpu: Vec<Duration>,
    submit_cpu: Vec<Duration>,
    report: FrameReport,
}

/// Runs every view on `runtime`. `about` lines head the report (adapter,
/// data); `new_map` makes a map loading the benchmarked style at a camera;
/// `progress` hears what runs.
pub fn run(
    runtime: &Runtime,
    settings: &Settings,
    about: Vec<String>,
    mut new_map: impl FnMut(CameraState) -> Map,
    mut progress: impl FnMut(&str),
) -> Result<Report, String> {
    let timing = timing(runtime, settings.timing)?;
    let cpu_resolution = clock::resolution();
    let (width, height) = settings.physical_size();
    let mut target = Target::texture(runtime, width, height, f64::from(settings.ratio));
    let size = (width, height);
    if settings.experiment == Experiment::Gestures {
        return gestures::run(
            runtime,
            timing,
            (&mut target, size),
            settings,
            about,
            new_map,
            progress,
        )
        .map(Report::Gestures);
    }
    // The frames' clock: fades and transitions run on frame times.
    let epoch = clock::now();

    let mut warnings = Vec::new();
    let mut views = Vec::new();
    let mut gpu_resolution = None;
    // One map for every view, as a host keeps one while it moves.
    let mut map: Option<Map> = None;
    let cameras: Vec<(String, f64, f64, f64, bool)> = if settings.views.is_empty() {
        VIEWS
            .iter()
            .map(|&(name, zoom, pitch, buildings)| (name.to_string(), zoom, pitch, 0.0, buildings))
            .collect()
    } else {
        settings
            .views
            .iter()
            .map(|&[zoom, pitch, bearing]| {
                let name = format!("z{zoom} pitch {pitch} bearing {bearing}");
                (name, zoom, pitch, bearing, false)
            })
            .collect()
    };
    for (name, zoom, pitch, bearing, buildings) in cameras {
        progress(&format!("{name}: loading"));
        let camera = CameraState {
            pitch,
            bearing,
            ..CameraState::new(settings.center, zoom, &crate::web_mercator())
        };
        let map = match &mut map {
            Some(map) => {
                map.jump_to_state(camera);
                map
            }
            None => map.insert(new_map(camera)),
        };
        settle(map, &mut target, &mut warnings)?;
        if buildings {
            let tx = orb_api::Transaction::from_commands(&add_buildings(map)?)?;
            map.edit(&tx).map_err(|e| e.to_string())?;
            settle(map, &mut target, &mut warnings)?;
        }

        let mut s = Samples::default();
        runtime.set_gpu_timing(true);
        for round in 0..settings.rounds {
            progress(&format!(
                "{name}: round {} of {}",
                round + 1,
                settings.rounds
            ));
            let count = settings.warmup + settings.frames;
            let (mut cpu, mut wall) = (Vec::with_capacity(count), Vec::with_capacity(count));
            for f in 0..count {
                let started = clock::now();
                let time = (started - epoch).as_secs_f64() * 1000.0;
                let (report, time) = frame(map, &mut target, time)?;
                // The proxy waits for each frame; timestamps let two
                // queue, as a display's swapchain does, so the GPU keeps
                // its clocks.
                if timing == Timing::WorkDone || f % 2 == 1 {
                    runtime.finish();
                }
                if timing == Timing::WorkDone {
                    wall.push(clock::now() - started);
                }
                s.report = report;
                cpu.push(time);
            }
            let times = runtime.gpu_timings();
            if times.len() != count {
                return Err(format!("{} submissions for {count} frames", times.len()));
            }
            for f in settings.warmup..count {
                s.gpu.push(match timing {
                    Timing::Timestamps => times[f].gpu(),
                    Timing::WorkDone => wall[f],
                });
                for (label, samples) in [
                    ("cull", &mut s.cull),
                    ("main", &mut s.main),
                    ("extrusion", &mut s.extrusion),
                ] {
                    samples.extend(times[f].pass(label));
                }
                s.render_cpu.push(cpu[f]);
                s.submit_cpu.push(times[f].cpu);
            }
        }
        gpu_resolution = gpu_resolution.or(runtime.stats().timestamp_resolution);
        runtime.set_gpu_timing(false);
        views.push(view_result(&name, s));
    }
    Ok(Report::Frames(FramesReport {
        about,
        settings: settings.clone(),
        size: (width, height),
        timing,
        gpu_resolution,
        cpu_resolution,
        views,
        warnings,
    }))
}

/// The edit adding Liberty's extrusions beneath its labels.
fn add_buildings(map: &Map) -> Result<String, String> {
    let style: serde_json::Value =
        serde_json::from_str(&map.style_json().ok_or("no style")?).map_err(|e| e.to_string())?;
    let before = style["layers"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|l| l["type"] == "symbol")
        .map(|l| l["id"].clone());
    Ok(match before {
        Some(before) => format!("[[\"addLayer\", {BUILDINGS_3D}, {before}]]"),
        None => format!("[[\"addLayer\", {BUILDINGS_3D}]]"),
    })
}

fn view_result(name: &str, mut s: Samples) -> ViewResult {
    let summary = |v: &mut Vec<Duration>| (!v.is_empty()).then(|| Summary::of(v));
    let stats = s.report.frame;
    ViewResult {
        name: name.to_string(),
        gpu: Summary::of(&mut s.gpu),
        cull: summary(&mut s.cull),
        main: summary(&mut s.main),
        extrusion: summary(&mut s.extrusion),
        render_cpu: Summary::of(&mut s.render_cpu),
        submit_cpu: Summary::of(&mut s.submit_cpu),
        draws: stats.draws,
        dispatches: stats.dispatches,
        tiles: s.report.tiles_drawn,
        candidates: stats.candidates,
        meshes: s.report.scene.meshes,
        memory: {
            let m = &s.report.memory;
            let pools = [
                m.scene.vertices,
                m.scene.indices,
                m.scene.compacted,
                m.scene.props,
            ];
            (m.allocated(), pools.iter().map(|p| p.used).sum())
        },
    }
}

/// The default surface's projection, which the benchmarks draw on.
pub(crate) fn web_mercator() -> orb_api::Projection {
    *orb_api::TileMatrixSet::by_id("WebMercatorQuad")
        .expect("built-in tile matrix set")
        .projection()
}

/// The warnings and errors since the last call.
pub(crate) fn take_warnings(map: &mut Map) -> Vec<String> {
    std::iter::from_fn(|| map.poll_event())
        .filter_map(|event| match event {
            Event::Map(MapEvent::Warning { message } | MapEvent::Error { message }) => {
                Some(message)
            }
            _ => None,
        })
        .collect()
}

/// One frame at `time` (milliseconds), and its CPU time.
pub(crate) fn frame(
    map: &mut Map,
    target: &mut Target,
    time: f64,
) -> Result<(FrameReport, Duration), String> {
    let started = clock::now();
    let report = map
        .render(target, time)
        .map_err(|e| e.to_string())?
        .ok_or("a texture target always draws")?;
    Ok((report, clock::now() - started))
}

/// Draws until the map is idle, then once more.
pub(crate) fn settle(
    map: &mut Map,
    target: &mut Target,
    warnings: &mut Vec<String>,
) -> Result<(), String> {
    map.render_until_idle(target, LOAD_TIMEOUT)
        .map_err(|e| format!("loading: {e}"))?;
    warnings.extend(take_warnings(map));
    Ok(())
}
