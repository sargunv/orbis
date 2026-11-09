//! GPU and CPU timings through `orb-api` (`cargo xtask bench`), as hosts
//! draw.
//!
//! - `frames` (the default): one map visits each view, loads until it is
//!   idle and renders `rounds` runs of still frames offscreen; with
//!   `nudge`, each frame moves the zoom by 0.001 and is read back, so it
//!   times whole frames as mln-rs's bench does.
//! - `gestures`: rotate, tilt, zoom and pan paths with tiles arriving.
//! - `load` ([`load`]): a fresh map per sample from loading its style to
//!   its first complete still, on a new device (cold) and on one that drew
//!   the view before (warm).
//! - `tiles` ([`tiles`]): the committed Tokyo extract's tiles through a
//!   tile job's stages in process: decode, layout and pack.
//!
//! [`run`] returns a [`Report`], which formats as Markdown.

#![forbid(unsafe_code)]

mod clock;
mod gestures;
mod load;
mod report;
mod tiles;

use std::collections::BTreeMap;
use std::time::Duration;

pub use gestures::GesturesReport;
pub use load::LoadReport;
use orb_api::{
    CameraState, Event, FrameReport, Geodetic, Map, MapEvent, MapOptions, Runtime, Target,
};
use report::{FramesReport, Summary, ViewResult};
pub use tiles::TilesReport;

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
/// The zoom step a nudged frame takes, as mln-rs's still bench.
const NUDGE: f64 = 0.001;

/// What the bench compares.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Experiment {
    /// Still frames of each view.
    #[default]
    Frames,
    /// Rotate, tilt and zoom paths at a 60 Hz pace with tiles arriving:
    /// frame-time distributions.
    Gestures,
    /// A fresh map from style load to its first complete still.
    Load,
    /// The committed extract's tiles through decode, layout and pack.
    Tiles,
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
    Load(LoadReport),
    Tiles(TilesReport),
}

impl Report {
    /// The report as Markdown.
    pub fn markdown(&self) -> String {
        let mut out = String::new();
        match self {
            Self::Frames(r) => r.write(&mut out),
            Self::Gestures(r) => r.write(&mut out),
            Self::Load(r) => r.write(&mut out),
            Self::Tiles(r) => r.write(&mut out),
        }
        .expect("writing to a String");
        out
    }
}

/// A camera to measure.
#[derive(Clone, Debug)]
pub struct View {
    pub name: String,
    /// The settings' centre when `None`.
    pub center: Option<Geodetic>,
    pub zoom: f64,
    pub pitch: f64,
    pub bearing: f64,
    /// Whether the style gains Liberty's extrusions first.
    pub buildings: bool,
    /// The settings' viewport when `None`.
    pub size: Option<ViewSize>,
}

/// A viewport: logical pixels and their ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewSize {
    pub width: u32,
    pub height: u32,
    pub ratio: f32,
}

impl ViewSize {
    fn physical(self) -> (u32, u32) {
        let px = |v: u32| (v as f32 * self.ratio).round() as u32;
        (px(self.width), px(self.height))
    }
}

/// What to measure, and how.
#[derive(Clone, Debug)]
pub struct Settings {
    /// Measured frames per submission per round.
    pub frames: usize,
    /// Frames rendered before measuring, per submission per round.
    pub warmup: usize,
    pub rounds: usize,
    /// Samples per view (`load`) or per tile (`tiles`).
    pub samples: usize,
    /// Viewport in logical pixels.
    pub width: u32,
    pub height: u32,
    pub ratio: f32,
    pub center: Geodetic,
    pub experiment: Experiment,
    pub paths: GesturePaths,
    /// The views to measure instead of the bench's own (`--view`,
    /// repeatable, or `--views <file>`).
    pub views: Vec<View>,
    /// Only views whose name contains this.
    pub filter: Option<String>,
    /// Frames alternate the zoom by [`NUDGE`] and are read back.
    pub nudge: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            frames: 100,
            warmup: 50,
            rounds: 3,
            samples: 5,
            width: 1024,
            height: 768,
            ratio: 2.0,
            center: Geodetic::new(TOKYO.0, TOKYO.1),
            experiment: Experiment::Frames,
            paths: GesturePaths::City,
            views: Vec::new(),
            filter: None,
            nudge: false,
        }
    }
}

impl Settings {
    /// Sets one option by name, as the tool's `--name value` gives it:
    /// `frames`, `warmup`, `rounds`, `samples`, `size` (`WxH`), `ratio`, `at`
    /// (`lon,lat`), `experiment` (`frames`, `gestures`, `load` or `tiles`),
    /// `paths` (`city` or `alps`, the gestures bench's), `view`
    /// (`zoom,pitch,bearing`, adding a view to measure instead of the
    /// bench's own) or `filter` (a substring of the views' names).
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
            "samples" => self.samples = count(value)?.max(1),
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
                self.views.push(View {
                    name: format!("z{zoom} pitch {pitch} bearing {bearing}"),
                    center: None,
                    zoom,
                    pitch,
                    bearing,
                    buildings: false,
                    size: None,
                });
            }
            "filter" => self.filter = Some(value.to_string()),
            "at" => {
                let (lon, lat) = value.split_once(',').ok_or("at needs lon,lat")?;
                self.center = Geodetic::new(number(lon)?, number(lat)?);
            }
            "experiment" => {
                self.experiment = match value {
                    "frames" => Experiment::Frames,
                    "gestures" => Experiment::Gestures,
                    "load" => Experiment::Load,
                    "tiles" => Experiment::Tiles,
                    _ => return Err("experiment needs frames, gestures, load or tiles".into()),
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

    fn size(&self) -> ViewSize {
        ViewSize {
            width: self.width,
            height: self.height,
            ratio: self.ratio,
        }
    }

    fn physical_size(&self) -> (u32, u32) {
        self.size().physical()
    }

    /// The views to measure, with their cameras and viewports.
    fn cameras(&self) -> Vec<(View, CameraState, ViewSize)> {
        let own = || {
            VIEWS.iter().map(|&(name, zoom, pitch, buildings)| View {
                name: name.to_string(),
                center: None,
                zoom,
                pitch,
                bearing: 0.0,
                buildings,
                size: None,
            })
        };
        let views: Vec<View> = if self.views.is_empty() {
            own().collect()
        } else {
            self.views.clone()
        };
        views
            .into_iter()
            .filter(|v| self.filter.as_ref().is_none_or(|f| v.name.contains(f)))
            .map(|v| {
                let camera = view_camera(&v, self.center, 0.0);
                let size = v.size.unwrap_or(self.size());
                (v, camera, size)
            })
            .collect()
    }
}

/// A view's camera, its zoom moved by `nudge`.
fn view_camera(view: &View, center: Geodetic, nudge: f64) -> CameraState {
    CameraState {
        pitch: view.pitch,
        bearing: view.bearing,
        ..CameraState::new(
            view.center.unwrap_or(center),
            view.zoom + nudge,
            &web_mercator(),
        )
    }
}

/// The views of a file in mln-rs's `tools/liberty-compare/views.txt`
/// format, `style name lon lat zoom bearing pitch ratio width height` a
/// line (`#` comments), whose style column is `style`.
pub fn read_views(path: &str, style: &str) -> Result<Vec<View>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    parse_views(&text, path, style)
}

fn parse_views(text: &str, path: &str, style: &str) -> Result<Vec<View>, String> {
    let mut views = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let at = || format!("{path}:{}", n + 1);
        let f: Vec<&str> = line.split_whitespace().collect();
        let [view_style, name, numbers @ ..] = &f[..] else {
            return Err(format!("{}: too few columns", at()));
        };
        let numbers: Vec<f64> = numbers
            .iter()
            .map(|v| v.parse::<f64>().map_err(|e| format!("{}: {e}", at())))
            .collect::<Result<_, _>>()?;
        let [lon, lat, zoom, bearing, pitch, ratio, width, height] = numbers[..] else {
            return Err(format!(
                "{}: want style name lon lat zoom bearing pitch ratio width height",
                at()
            ));
        };
        if *view_style != style {
            continue;
        }
        views.push(View {
            name: name.to_string(),
            center: Some(Geodetic::new(lon, lat)),
            zoom,
            pitch,
            bearing,
            buildings: false,
            size: Some(ViewSize {
                width: width as u32,
                height: height as u32,
                ratio: ratio as f32,
            }),
        });
    }
    if views.is_empty() {
        return Err(format!("{path} has no view of style {style}"));
    }
    Ok(views)
}

/// A view's still frames.
#[derive(Default)]
struct Samples {
    /// GPU time per frame (every pass), or the proxy.
    gpu: Vec<Duration>,
    /// Passes alone; empty under the proxy.
    cull: Vec<Duration>,
    main: Vec<Duration>,
    /// Per pass label, its passes' time in each frame, and how many ran.
    passes: BTreeMap<String, (Vec<Duration>, usize)>,
    /// `Map::render`.
    render_cpu: Vec<Duration>,
    submit_cpu: Vec<Duration>,
    /// Render and readback of nudged frames.
    wall: Vec<Duration>,
    report: FrameReport,
}

/// A map on `runtime` loading `style` at `camera`.
pub(crate) fn new_map(runtime: &Runtime, style: &str, camera: CameraState) -> Map {
    let mut map = Map::new(
        runtime,
        MapOptions {
            camera: Some(camera),
            ..MapOptions::default()
        },
    );
    map.load_style_url(style);
    map
}

/// Runs the experiment on `runtime` for `style`, a URL. `about` lines head
/// the report (adapter, data); `data` is the directory holding the
/// committed extract (the `tiles` experiment's); `progress` hears what
/// runs.
pub fn run(
    runtime: &Runtime,
    settings: &Settings,
    about: Vec<String>,
    style: &str,
    data: &std::path::Path,
    mut progress: impl FnMut(&str),
) -> Result<Report, String> {
    match settings.experiment {
        Experiment::Load => {
            return load::run(runtime, settings, about, style, progress).map(Report::Load);
        }
        Experiment::Tiles => {
            return tiles::run(settings, about, data, progress).map(Report::Tiles);
        }
        Experiment::Frames | Experiment::Gestures => {}
    }
    if !runtime.stats().gpu_timestamps {
        return Err("the device has no timestamp queries".into());
    }
    let cpu_resolution = clock::resolution();
    if settings.experiment == Experiment::Gestures {
        let (width, height) = settings.physical_size();
        let mut target = Target::texture(runtime, width, height, f64::from(settings.ratio));
        return gestures::run(
            runtime,
            (&mut target, (width, height)),
            settings,
            about,
            |camera| new_map(runtime, style, camera),
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
    let mut target: Option<(ViewSize, Target)> = None;
    for (view, camera, size) in settings.cameras() {
        let name = view.name.clone();
        progress(&format!("{name}: loading"));
        let target = match &mut target {
            Some((s, t)) if *s == size => t,
            _ => {
                let (w, h) = size.physical();
                let t = Target::texture(runtime, w, h, f64::from(size.ratio));
                &mut target.insert((size, t)).1
            }
        };
        let map = match &mut map {
            Some(map) => {
                map.jump_to_state(camera);
                map
            }
            None => map.insert(new_map(runtime, style, camera)),
        };
        settle(map, target, &mut warnings)?;
        if view.buildings {
            let tx = orb_api::Transaction::from_commands(&add_buildings(map)?)?;
            map.edit(&tx).map_err(|e| e.to_string())?;
            settle(map, target, &mut warnings)?;
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
            let mut cpu = Vec::with_capacity(count);
            let mut wall = Vec::with_capacity(count);
            // The target keeps two frames in flight, as a swapchain does,
            // unless nudged frames are read back one by one.
            for i in 0..count {
                if settings.nudge {
                    let nudge = if i % 2 == 1 { NUDGE } else { 0.0 };
                    map.jump_to_state(view_camera(&view, settings.center, nudge));
                }
                let started = clock::now();
                let time = (started - epoch).as_secs_f64() * 1000.0;
                let (report, time) = frame(map, target, time)?;
                if settings.nudge {
                    target.read_pixels().map_err(|e| e.to_string())?;
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
                s.gpu.push(times[f].gpu());
                for (label, samples) in [("cull", &mut s.cull), ("main", &mut s.main)] {
                    samples.extend(times[f].pass(label));
                }
                let mut frame: BTreeMap<&str, (Duration, usize)> = BTreeMap::new();
                for (label, d) in &times[f].passes {
                    let e = frame.entry(label).or_default();
                    *e = (e.0 + *d, e.1 + 1);
                }
                for (label, (d, n)) in frame {
                    let e = s.passes.entry(label.to_string()).or_default();
                    e.0.push(d);
                    e.1 = n;
                }
                s.render_cpu.push(cpu[f]);
                s.submit_cpu.push(times[f].cpu);
                s.wall.extend(wall.get(f));
            }
        }
        if settings.nudge {
            map.jump_to_state(camera);
        }
        gpu_resolution = gpu_resolution.or(runtime.stats().timestamp_resolution);
        runtime.set_gpu_timing(false);
        views.push(view_result(&name, size.physical(), s));
    }
    Ok(Report::Frames(FramesReport {
        about,
        settings: settings.clone(),
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

fn view_result(name: &str, size: (u32, u32), mut s: Samples) -> ViewResult {
    let summary = |v: &mut Vec<Duration>| (!v.is_empty()).then(|| Summary::of(v));
    let stats = s.report.frame;
    ViewResult {
        name: name.to_string(),
        size,
        gpu: Summary::of(&mut s.gpu),
        cull: summary(&mut s.cull),
        main: summary(&mut s.main),
        passes: (s.passes.into_iter())
            .map(|(label, (mut d, n))| (label, Summary::of(&mut d), n))
            .collect(),
        render_cpu: Summary::of(&mut s.render_cpu),
        submit_cpu: Summary::of(&mut s.submit_cpu),
        wall: summary(&mut s.wall),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_of_a_style() {
        let text = "# style name lon lat zoom bearing pitch ratio width height\n\
            liberty ams13 4.89 52.37 13 0 0 1 512 512\n\
            bright ams13 4.89 52.37 13 0 0 1 512 512\n\
            liberty paris16r2 2.35 48.857 16 30 45 2 256 256 # retina\n";
        let views = parse_views(text, "views.txt", "liberty").unwrap();
        let names: Vec<&str> = views.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(names, ["ams13", "paris16r2"]);
        let paris = &views[1];
        assert_eq!((paris.zoom, paris.bearing, paris.pitch), (16.0, 30.0, 45.0));
        let size = paris.size.unwrap();
        assert_eq!(
            (size.width, size.ratio, size.physical()),
            (256, 2.0, (512, 512))
        );
        assert!(parse_views(text, "views.txt", "dark").is_err());
        assert!(parse_views("liberty x 1 2 3\n", "views.txt", "liberty").is_err());
    }
}
