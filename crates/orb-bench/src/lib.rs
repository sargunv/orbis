//! GPU and CPU frame timings: natively (`cargo xtask bench`) and in the
//! browser (`orb-web`'s `harness` feature, `web/demo/bench.html`), through
//! the same loop.
//!
//! By default one map visits each view, loads until it is idle and renders
//! `rounds` runs of still frames offscreen; `experiment=gestures` times
//! rotate, tilt, zoom and pan paths with tiles arriving. [`run`] returns a
//! [`Report`], which formats as Markdown.
//!
//! The loop is async so it runs on the browser's event loop; natively,
//! every future it awaits is ready when first polled.

#![forbid(unsafe_code)]

mod clock;
mod gestures;
mod report;
mod timed;

use std::time::Duration;

pub use gestures::GesturesReport;
use orb_crs::Geodetic;
use orb_hal::{Gpu, TextureDesc, TextureFormat, TextureUsages};
use orb_hal_wgpu::WgpuGpu;
use orb_runtime::{CameraState, FrameReport, Map, TargetFormat, Transaction, Viewport};
use report::{FramesReport, Summary, ViewResult};
use timed::TimedGpu;

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
/// Words of the cull pass's counters in a node's draw arguments
/// (`cull.wgsl`): chunks kept, indices written, vertex inputs.
const KEPT_WORD: usize = 5;
const WRITTEN_WORD: usize = 6;
const INPUTS_WORD: usize = 7;
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
    /// Sets one option by name, as the native tool's `--name value` and the
    /// page's `?name=value` give it: `frames`, `warmup`, `rounds`, `size`
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

/// A device for [`run`], and how it times the GPU.
#[derive(Debug)]
pub struct Device {
    gpu: WgpuGpu,
    /// wgpu's adapter name, backend and driver.
    pub adapter: String,
    pub timing: Timing,
}

/// A device on the high-performance adapter, with timestamp queries unless
/// `timing` asks for the proxy or the adapter lacks them.
pub async fn device(timing: Option<Timing>) -> Result<Device, String> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("no adapter: {e}"))?;
    let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let timing = match timing {
        Some(Timing::Timestamps) if !timestamps => {
            return Err("the adapter has no timestamp queries".into());
        }
        Some(timing) => timing,
        None if timestamps => Timing::Timestamps,
        None => Timing::WorkDone,
    };
    let info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("orb-bench"),
            required_features: match timing {
                Timing::Timestamps => wgpu::Features::TIMESTAMP_QUERY,
                Timing::WorkDone => wgpu::Features::empty(),
            },
            ..Default::default()
        })
        .await
        .map_err(|e| format!("no device: {e}"))?;
    Ok(Device {
        gpu: WgpuGpu::new(device, queue),
        adapter: format!("{} ({:?}, {})", info.name, info.backend, info.driver),
        timing,
    })
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
    /// `Map::render`, less the timer's own overhead.
    render_cpu: Vec<Duration>,
    submit_cpu: Vec<Duration>,
    report: FrameReport,
    /// The last frame's cull counters, summed over nodes.
    counters: [u64; 3],
}

/// The cull counters (chunks kept, indices written, vertex inputs) summed
/// over the first `nodes` nodes of a draw-argument buffer's words.
fn counters(words: &[u32], nodes: u32) -> [u64; 3] {
    let mut sum = [0u64; 3];
    for node in words.chunks(orb_runtime::ARGS_WORDS).take(nodes as usize) {
        for (s, word) in sum.iter_mut().zip([KEPT_WORD, WRITTEN_WORD, INPUTS_WORD]) {
            *s += u64::from(node[word]);
        }
    }
    sum
}

/// Runs every view on `device`. `about` lines head the report (adapter,
/// browser, data); `new_map` makes a map loading the benchmarked style at
/// a camera; `progress` hears what runs.
pub async fn run(
    device: Device,
    settings: &Settings,
    about: Vec<String>,
    mut new_map: impl FnMut(CameraState) -> Result<Map, String>,
    mut progress: impl FnMut(&str),
) -> Result<Report, String> {
    let timing = device.timing;
    let mut gpu = TimedGpu::new(device.gpu, timing == Timing::Timestamps);
    let cpu_resolution = clock::resolution();
    let (width, height) = settings.physical_size();
    let format = TargetFormat {
        color: TextureFormat::Rgba8Unorm,
        sample_count: 1,
    };
    let target = gpu.create_texture(&TextureDesc {
        label: "bench",
        width,
        height,
        array_layers: 1,
        dimension: orb_hal::TextureViewDimension::D2,
        format: format.color,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        sample_count: 1,
        mip_levels: 1,
    });
    let viewport = Viewport {
        width: f64::from(settings.width),
        height: f64::from(settings.height),
    };
    let size = (width, height);
    match settings.experiment {
        Experiment::Gestures => {
            let run = gestures::run(
                &mut gpu,
                timing,
                (target, size),
                format,
                settings,
                about,
                new_map,
                progress,
            );
            return run.await.map(Report::Gestures);
        }
        Experiment::Frames => {}
    }
    // The one map's renderer.
    let renderer = std::cell::RefCell::new(None);
    let render = |gpu: &mut TimedGpu, map: &mut Map| {
        let started = clock::now();
        let mut renderer = renderer.borrow_mut();
        let renderer = renderer.get_or_insert_with(|| map.create_renderer(gpu, format));
        let report = map.render(gpu, renderer, target, viewport, settings.ratio, 0.0);
        (report, started, clock::now() - started)
    };

    let mut warnings = Vec::new();
    let mut views = Vec::new();
    // One map for every view: a dropped map keeps its GPU buffers (#20), and
    // a phone's browser has no room for a map per view.
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
            None => map.insert(new_map(camera)?),
        };
        settle(&mut gpu, map, &|g, m| render(g, m).0, &mut warnings).await?;
        if buildings {
            // Liberty's extrusions, beneath its labels.
            let before = map
                .style_doc()
                .into_iter()
                .flat_map(|doc| doc.style().layers.iter())
                .find(|l| l.ty == orb_style::LayerType::Symbol)
                .map(|l| l.id.replace('\\', "\\\\").replace('"', "\\\""));
            let call = match before {
                Some(before) => format!("[[\"addLayer\", {BUILDINGS_3D}, \"{before}\"]]"),
                None => format!("[[\"addLayer\", {BUILDINGS_3D}]]"),
            };
            map.edit(&Transaction::from_commands(&call)?)?;
            settle(&mut gpu, map, &|g, m| render(g, m).0, &mut warnings).await?;
        }

        let mut s = Samples::default();
        for round in 0..settings.rounds {
            progress(&format!(
                "{name}: round {} of {}",
                round + 1,
                settings.rounds
            ));
            let count = settings.warmup + settings.frames;
            let (mut cpu, mut wall) = (Vec::with_capacity(count), Vec::with_capacity(count));
            for _ in 0..count {
                let (report, started, time) = render(&mut gpu, map);
                match timing {
                    Timing::Timestamps => gpu.throttle().await,
                    Timing::WorkDone => {
                        gpu.finish().await;
                        wall.push(clock::now() - started);
                    }
                }
                s.report = report;
                cpu.push(time);
            }
            let times = gpu.take_frames().await;
            assert_eq!(times.len(), count, "one submission per frame");
            for f in settings.warmup..count {
                s.gpu.push(match timing {
                    Timing::Timestamps => times[f].gpu(),
                    Timing::WorkDone => wall[f].saturating_sub(times[f].overhead),
                });
                for (label, samples) in [
                    ("cull", &mut s.cull),
                    ("main", &mut s.main),
                    ("extrusion", &mut s.extrusion),
                ] {
                    samples.extend(times[f].pass(label));
                }
                s.render_cpu.push(cpu[f].saturating_sub(times[f].overhead));
                s.submit_cpu.push(times[f].submit_cpu);
            }
        }
        let (args, nodes) = renderer
            .borrow()
            .as_ref()
            .map(|r| r.draw_args())
            .ok_or("no renderer")?;
        s.counters = counters(&words(&gpu.inner().read_buffer(args).await), nodes);

        views.push(view_result(&name, s));
    }
    Ok(Report::Frames(FramesReport {
        about,
        settings: settings.clone(),
        size: (width, height),
        timing,
        gpu_resolution: gpu.timestamp_resolution(),
        cpu_resolution,
        views,
        warnings,
    }))
}

/// Little-endian words.
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
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
        chunks: s.counters[0],
        inputs: s.counters[2],
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
pub(crate) fn web_mercator() -> orb_crs::Projection {
    *orb_crs::TileMatrixSet::by_id("WebMercatorQuad")
        .expect("built-in tile matrix set")
        .projection()
}

/// The warnings and errors since the last frame.
pub(crate) fn take_warnings(map: &mut Map) -> Vec<String> {
    map.take_events()
        .into_iter()
        .filter_map(|event| match event {
            orb_runtime::MapEvent::Warning { message }
            | orb_runtime::MapEvent::Error { message } => Some(message),
            _ => None,
        })
        .collect()
}

/// Draws until the map is idle, then once more.
pub(crate) async fn settle(
    gpu: &mut TimedGpu,
    map: &mut Map,
    render: &impl Fn(&mut TimedGpu, &mut Map) -> FrameReport,
    warnings: &mut Vec<String>,
) -> Result<(), String> {
    let started = clock::now();
    loop {
        let report = render(gpu, map);
        warnings.extend(take_warnings(map));
        gpu.take_frames().await;
        if report.idle {
            render(gpu, map);
            gpu.take_frames().await;
            return Ok(());
        }
        if clock::now() - started > LOAD_TIMEOUT {
            return Err("timed out waiting for tiles".into());
        }
        clock::sleep(Duration::from_millis(5)).await;
    }
}
