//! GPU and CPU frame timings of the default render path (decision 0002)
//! and of paint bindings (decision 0017): natively (`cargo xtask bench`)
//! and in the browser (`orb-web`'s `harness` feature, `web/demo/bench.html`),
//! through the same loop.
//!
//! One map visits each view, loads until it is idle and renders `rounds`
//! runs of camera-only frames offscreen. Then the camera pans around a
//! circle, once to load what it shows and `rounds` times measured, with the
//! cull pass's counters read back per frame: how often compacted indices
//! are rewritten, and what those frames cost. [`run`] returns a [`Report`],
//! which formats as Markdown.
//!
//! The loop is async so it runs on the browser's event loop; natively,
//! every future it awaits is ready when first polled.

#![forbid(unsafe_code)]

mod clock;
mod gestures;
mod labels;
mod paint;
mod report;
mod timed;

use std::time::Duration;

pub use gestures::GesturesReport;
pub use labels::LabelsReport;
use orb_geo::Geodetic;
use orb_hal::{Gpu, TextureDesc, TextureFormat, TextureUsages};
use orb_hal_wgpu::WgpuGpu;
use orb_runtime::{CameraState, FrameReport, Map, TargetFormat, Transaction, Viewport};
pub use paint::PaintReport;
use report::{FramesReport, Moving, Summary, ViewResult};
use timed::TimedGpu;

/// The committed Tokyo extract's view, by the Imperial Palace
/// (`web/demo/data/README.md`).
pub const TOKYO: (f64, f64) = (139.74, 35.685);
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
/// The moving camera pans around a circle of this radius, in logical
/// pixels, over this many frames: about 6.7 px per frame.
const PATH_RADIUS: f64 = 128.0;
const PATH_FRAMES: usize = 120;
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
    /// The default path's frames, still and moving (decision 0002).
    #[default]
    Frames,
    /// Decision 0017's paint bindings: property rows against GPU bytecode.
    Paint,
    /// M2's label exits: placement time while turning, hidden symbols' GPU
    /// cost.
    Labels,
    /// Rotate, tilt and zoom paths at a 60 Hz pace with tiles arriving:
    /// frame-time distributions.
    Gestures,
}

/// A bench's results.
#[derive(Debug)]
pub enum Report {
    Frames(FramesReport),
    Paint(PaintReport),
    Labels(LabelsReport),
    Gestures(GesturesReport),
}

impl Report {
    /// The report as Markdown.
    pub fn markdown(&self) -> String {
        let mut out = String::new();
        match self {
            Self::Frames(r) => r.write(&mut out),
            Self::Paint(r) => r.write(&mut out),
            Self::Labels(r) => r.write(&mut out),
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
    /// Keeps the style's symbol layers.
    pub labels: bool,
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
            labels: true,
        }
    }
}

impl Settings {
    /// Sets one option by name, as the native tool's `--name value` and the
    /// page's `?name=value` give it: `frames`, `warmup`, `rounds`, `size`
    /// (`WxH`), `ratio`, `at` (`lon,lat`), `timing` (`timestamps` or
    /// `work-done`), `experiment` (`frames`, `paint`, `labels` or
    /// `gestures`) or
    /// `labels` (`on` or `off`).
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
            "labels" => {
                self.labels = match value {
                    "on" => true,
                    "off" => false,
                    _ => return Err("labels needs on or off".into()),
                }
            }
            "experiment" => {
                self.experiment = match value {
                    "frames" => Experiment::Frames,
                    "paint" => Experiment::Paint,
                    "labels" => Experiment::Labels,
                    "gestures" => Experiment::Gestures,
                    _ => {
                        return Err("experiment needs frames, paint, labels or gestures".into());
                    }
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
    if settings.experiment == Experiment::Paint {
        let size = (width, height);
        return paint::run(
            &mut gpu,
            timing,
            (target, size),
            format,
            settings,
            about,
            new_map,
            progress,
        )
        .await
        .map(Report::Paint);
    }
    if settings.experiment == Experiment::Labels {
        let size = (width, height);
        return labels::run(
            &mut gpu,
            timing,
            (target, size),
            format,
            settings,
            about,
            new_map,
            progress,
        )
        .await
        .map(Report::Labels);
    }
    if settings.experiment == Experiment::Gestures {
        let size = (width, height);
        return gestures::run(
            &mut gpu,
            timing,
            (target, size),
            format,
            settings,
            about,
            new_map,
            progress,
        )
        .await
        .map(Report::Gestures);
    }
    let render = |gpu: &mut TimedGpu, map: &mut Map| {
        let started = clock::now();
        let report = map.render(gpu, target, format, viewport, settings.ratio);
        (report, started, clock::now() - started)
    };

    let mut warnings = Vec::new();
    let mut views = Vec::new();
    // One map for every view: a dropped map keeps its GPU buffers (#20), and
    // a phone's browser has no room for a map per view.
    let mut map: Option<Map> = None;
    let mut labels_removed = false;
    for &(name, zoom, pitch, buildings) in VIEWS {
        progress(&format!("{name}: loading"));
        let camera = CameraState {
            pitch,
            ..CameraState::new(settings.center, zoom, &crate::web_mercator())
        };
        let map = match &mut map {
            Some(map) => {
                *map.camera_mut() = camera;
                map
            }
            None => map.insert(new_map(camera)?),
        };
        // `labels=off` measures the views without the style's labels.
        let mut edit = (!settings.labels && !labels_removed) || buildings;
        let started = clock::now();
        loop {
            let (report, _, _) = render(&mut gpu, map);
            warnings.extend(map.take_warnings());
            gpu.take_frames().await;
            if report.idle && edit {
                edit = false;
                let symbols = map
                    .style_doc()
                    .into_iter()
                    .flat_map(|doc| doc.style().layers.iter())
                    .filter(|l| matches!(l.kind, orb_v8::LayerKind::Symbol(_)))
                    .map(|l| l.id.replace('\\', "\\\\").replace('"', "\\\""));
                let mut calls = Vec::new();
                let mut first_symbol = None;
                for id in symbols {
                    first_symbol.get_or_insert_with(|| id.clone());
                    if !settings.labels && !labels_removed {
                        calls.push(format!("[\"removeLayer\", \"{id}\"]"));
                    }
                }
                labels_removed = !settings.labels;
                if buildings {
                    calls.push(match first_symbol.filter(|_| settings.labels) {
                        Some(before) => format!("[\"addLayer\", {BUILDINGS_3D}, \"{before}\"]"),
                        None => format!("[\"addLayer\", {BUILDINGS_3D}]"),
                    });
                }
                if !calls.is_empty() {
                    let tx = Transaction::from_commands(&format!("[{}]", calls.join(",")))?;
                    map.edit(&tx)?;
                    continue;
                }
            }
            if report.idle {
                // One more frame draws what the last one received.
                render(&mut gpu, map);
                gpu.take_frames().await;
                break;
            }
            if clock::now() - started > LOAD_TIMEOUT {
                return Err(format!("{name}: timed out waiting for tiles"));
            }
            clock::sleep(Duration::from_millis(5)).await;
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
        let (args, nodes) = map.draw_args().ok_or("no renderer")?;
        s.counters = counters(&words(&gpu.inner().read_buffer(args).await), nodes);

        progress(&format!("{name}: loading the moving camera's path"));
        let path = Path::new(viewport);
        for f in 0..PATH_FRAMES {
            path.step(map, f);
            render(&mut gpu, map);
            gpu.throttle().await;
        }
        settle(&mut gpu, map, &render, name).await?;
        let mut moving = Moving::default();
        for round in 0..settings.rounds {
            progress(&format!(
                "{name}: moving, round {} of {}",
                round + 1,
                settings.rounds
            ));
            pan_circle(&mut gpu, map, &render, &path, timing, &mut moving).await?;
        }
        views.push(view_result(name, s, moving));
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

/// The moving camera's path: around a circle of [`PATH_RADIUS`], back to
/// where it started.
struct Path {
    viewport: Viewport,
}

impl Path {
    fn new(viewport: Viewport) -> Self {
        Self { viewport }
    }

    fn point(&self, f: usize) -> glam::DVec2 {
        let center = glam::DVec2::new(self.viewport.width / 2.0, self.viewport.height / 2.0);
        let angle = std::f64::consts::TAU * f as f64 / PATH_FRAMES as f64;
        center + PATH_RADIUS * glam::DVec2::new(angle.cos() - 1.0, angle.sin())
    }

    /// Pans from frame `f`'s point to the next.
    fn step(&self, map: &mut Map, f: usize) {
        let projection = map.projection();
        map.camera_mut()
            .pan(&projection, self.viewport, self.point(f), self.point(f + 1));
    }
}

/// Renders until the map is idle.
async fn settle(
    gpu: &mut TimedGpu,
    map: &mut Map,
    render: &impl Fn(&mut TimedGpu, &mut Map) -> (FrameReport, Duration, Duration),
    name: &str,
) -> Result<(), String> {
    let started = clock::now();
    loop {
        let (report, _, _) = render(gpu, map);
        gpu.take_frames().await;
        if report.idle {
            return Ok(());
        }
        if clock::now() - started > LOAD_TIMEOUT {
            return Err(format!("{name}: timed out loading the path"));
        }
        clock::sleep(Duration::from_millis(5)).await;
    }
}

/// Pans once around `path`, measured, adding the frames to `moving`.
async fn pan_circle(
    gpu: &mut TimedGpu,
    map: &mut Map,
    render: &impl Fn(&mut TimedGpu, &mut Map) -> (FrameReport, Duration, Duration),
    path: &Path,
    timing: Timing,
    moving: &mut Moving,
) -> Result<(), String> {
    gpu.take_frames().await;
    let (args, nodes) = map.draw_args().ok_or("no renderer")?;
    gpu.capture(
        args,
        u64::from(nodes) * orb_runtime::ARGS_WORDS as u64 * 4,
        PATH_FRAMES as u64,
    );
    let mut cpu = Vec::with_capacity(PATH_FRAMES);
    let mut wall = Vec::with_capacity(PATH_FRAMES);
    for f in 0..PATH_FRAMES {
        path.step(map, f);
        let (report, started, time) = render(gpu, map);
        match timing {
            Timing::Timestamps => gpu.throttle().await,
            Timing::WorkDone => {
                gpu.finish().await;
                wall.push(clock::now() - started);
            }
        }
        moving.loading += usize::from(report.tiles_loading > 0);
        cpu.push(time);
    }
    let times = gpu.take_frames().await;
    let captures = gpu.take_captures().await;
    moving
        .frames
        .extend((0..PATH_FRAMES).map(|f| report::MovingFrame {
            gpu: match timing {
                Timing::Timestamps => times[f].gpu(),
                Timing::WorkDone => wall[f].saturating_sub(times[f].overhead),
            },
            cull: times[f].pass("cull"),
            cpu: cpu[f].saturating_sub(times[f].overhead),
            written: counters(&captures[f], nodes)[1],
        }));
    Ok(())
}

fn view_result(name: &str, mut s: Samples, moving: Moving) -> ViewResult {
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
        moving,
    }
}

/// Reproducible steady-state cost of the Liberty building layer in the Tokyo
/// extract. Kept outside the A/B/C experiment, which compares 2D submissions.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod extrusion_benchmark {
    use super::*;
    use orb_host_native::{NativeFetch, NativeImageDecode};
    use orb_runtime::{PoolJobs, Services};
    use std::sync::Arc;

    #[test]
    #[ignore = "GPU benchmark: cargo test --release -p orb-bench tokyo_extrusion_frame_cost -- --ignored --nocapture"]
    fn tokyo_extrusion_frame_cost() {
        pollster::block_on(async {
            let device = device(Some(Timing::Timestamps)).await.unwrap();
            eprintln!("adapter: {}", device.adapter);
            let mut gpu = TimedGpu::new(device.gpu, true);
            let path =
                std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/demo/data"))
                    .canonicalize()
                    .unwrap();
            let style_path = std::env::var_os("ORB_BENCH_STYLE")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| path.join("tokyo-mvt.json"));
            let style = std::fs::read_to_string(style_path).unwrap().replace(
                "pmtiles://./data/",
                &format!("pmtiles://file://{}/", path.display()),
            );
            let fetch = NativeFetch::new();
            let services = Services {
                clock: Arc::new(orb_host_native::NativeClock::default()),
                image_decode: Arc::new(NativeImageDecode),
                images: Default::default(),
                jobs: PoolJobs::new(
                    fetch.clone(),
                    Arc::new(NativeImageDecode),
                    Arc::new(orb_jobs::ThreadPool::new("jobs", 2)),
                ),
                fetch,
                wake: Arc::new(|| {}),
            };
            let mut map = Map::new(
                services,
                Some(CameraState {
                    bearing: 30.0,
                    pitch: 60.0,
                    ..CameraState::new(
                        Geodetic::new(TOKYO.0, TOKYO.1),
                        16.0,
                        &crate::web_mercator(),
                    )
                }),
            );
            map.set_style_json(&style).unwrap();
            let format = TargetFormat {
                color: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            };
            let target = gpu.create_texture(&TextureDesc {
                label: "extrusion benchmark",
                width: 1024,
                height: 768,
                array_layers: 1,
                dimension: orb_hal::TextureViewDimension::D2,
                format: format.color,
                usage: TextureUsages::RENDER_ATTACHMENT,
                sample_count: 1,
                mip_levels: 1,
            });
            let render = |gpu: &mut TimedGpu, map: &mut Map| {
                map.render(
                    gpu,
                    target,
                    format,
                    Viewport {
                        width: 1024.0,
                        height: 768.0,
                    },
                    1.0,
                )
            };
            let start = std::time::Instant::now();
            loop {
                let report = render(&mut gpu, &mut map);
                gpu.take_frames().await;
                if report.idle {
                    break;
                }
                assert!(start.elapsed() < Duration::from_secs(120));
                std::thread::sleep(Duration::from_millis(5));
            }
            for _ in 0..50 {
                render(&mut gpu, &mut map);
                gpu.throttle().await;
            }
            gpu.take_frames().await;
            let mut cpu = Vec::new();
            let mut report = FrameReport::default();
            for _ in 0..300 {
                let start = std::time::Instant::now();
                report = render(&mut gpu, &mut map);
                cpu.push(start.elapsed());
                gpu.throttle().await;
            }
            let frames = gpu.take_frames().await;
            for (cpu, frame) in cpu.iter_mut().zip(&frames) {
                *cpu = cpu.saturating_sub(frame.overhead);
            }
            let median = |mut samples: Vec<Duration>| {
                samples.sort();
                samples[samples.len() / 2].as_secs_f64() * 1000.0
            };
            eprintln!(
                "Tokyo 139.74,35.685,16,30,60; 1024x768; 50 warmup + 300 samples; milliseconds, medians"
            );
            eprintln!(
                "CPU render {:.6}; GPU total {:.6}; cull {:.6}; extrusion {:.6}; main {:.6}",
                median(cpu),
                median(frames.iter().map(|f| f.gpu()).collect()),
                median(frames.iter().map(|f| f.pass("cull").unwrap()).collect()),
                median(
                    frames
                        .iter()
                        .map(|f| f.pass("extrusion").unwrap_or_default())
                        .collect()
                ),
                median(frames.iter().map(|f| f.pass("main").unwrap()).collect())
            );
            eprintln!("{report:?}");
            assert_eq!(report.frame.dispatches, 1);
            assert!(!report.frame.bundle_recorded);
            assert_eq!(report.frame.paints_written, 0);
        });
    }

    /// Terrain's frame cost on the Alps demo view (decision 0019): frames
    /// at rest, frames while the camera turns (no drape redraw), and the
    /// frame that redraws every drape slot. Fetches tiles over the network.
    #[test]
    #[ignore = "GPU benchmark over the network: cargo test --release -p orb-bench alps_terrain_frame_cost -- --ignored --nocapture"]
    fn alps_terrain_frame_cost() {
        pollster::block_on(async {
            let device = device(Some(Timing::Timestamps)).await.unwrap();
            eprintln!("adapter: {}", device.adapter);
            let mut gpu = TimedGpu::new(device.gpu, true);
            let style = std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../web/demo/data/alps.json"
            ))
            .unwrap();
            let fetch = NativeFetch::new();
            let services = Services {
                clock: Arc::new(orb_host_native::NativeClock::default()),
                image_decode: Arc::new(NativeImageDecode),
                images: Default::default(),
                jobs: PoolJobs::new(
                    fetch.clone(),
                    Arc::new(NativeImageDecode),
                    Arc::new(orb_jobs::ThreadPool::new("jobs", 4)),
                ),
                fetch,
                wake: Arc::new(|| {}),
            };
            let mut map = Map::new(services, None);
            map.set_style_json(&style).unwrap();
            let format = TargetFormat {
                color: TextureFormat::Rgba8Unorm,
                sample_count: 1,
            };
            let target = gpu.create_texture(&TextureDesc {
                label: "terrain benchmark",
                width: 1024,
                height: 768,
                array_layers: 1,
                dimension: orb_hal::TextureViewDimension::D2,
                format: format.color,
                usage: TextureUsages::RENDER_ATTACHMENT,
                sample_count: 1,
                mip_levels: 1,
            });
            let viewport = Viewport {
                width: 1024.0,
                height: 768.0,
            };
            let render = |gpu: &mut TimedGpu, map: &mut Map| {
                map.constrain_camera(viewport);
                map.render(gpu, target, format, viewport, 1.0)
            };
            let start = std::time::Instant::now();
            loop {
                let report = render(&mut gpu, &mut map);
                gpu.take_frames().await;
                if report.idle {
                    break;
                }
                assert!(start.elapsed() < Duration::from_secs(180));
                std::thread::sleep(Duration::from_millis(5));
            }
            let median = |mut samples: Vec<Duration>| {
                samples.sort();
                samples[samples.len() / 2].as_secs_f64() * 1000.0
            };
            let pass = |frames: &[timed::FrameTimes], label: &str| {
                median(
                    frames
                        .iter()
                        .map(|f| f.pass(label).unwrap_or_default())
                        .collect(),
                )
            };
            let sample = |gpu: &mut TimedGpu, map: &mut Map, turn: f64| {
                let mut cpu = Vec::new();
                let mut report = FrameReport::default();
                for _ in 0..200 {
                    map.camera_mut().bearing += turn;
                    let start = std::time::Instant::now();
                    report = render(gpu, map);
                    cpu.push(start.elapsed());
                    pollster::block_on(gpu.throttle());
                }
                (cpu, report)
            };
            for (label, turn) in [("at rest", 0.0), ("turning 0.2°/frame", 0.2)] {
                for _ in 0..30 {
                    render(&mut gpu, &mut map);
                    gpu.throttle().await;
                }
                gpu.take_frames().await;
                let (mut cpu, report) = sample(&mut gpu, &mut map, turn);
                let frames = gpu.take_frames().await;
                for (cpu, frame) in cpu.iter_mut().zip(&frames) {
                    *cpu = cpu.saturating_sub(frame.overhead);
                }
                eprintln!(
                    "{label}: CPU render {:.3}; GPU total {:.3}; cull {:.3}; terrain {:.3}; main {:.3}; drape redraws {}",
                    median(cpu),
                    median(frames.iter().map(|f| f.gpu()).collect()),
                    pass(&frames, "cull"),
                    pass(&frames, "terrain"),
                    pass(&frames, "main"),
                    frames.iter().filter(|f| f.pass("drape").is_some()).count()
                );
                eprintln!("{report:?}");
            }
            // A zoom change waits for the camera to stop, then redraws
            // every slot in one frame.
            let mut redraws = Vec::new();
            for i in 0..20 {
                let projection = map.projection();
                map.camera_mut()
                    .set_zoom(&projection, 11.6 + f64::from(i % 2) * 0.01);
                render(&mut gpu, &mut map);
                render(&mut gpu, &mut map);
                gpu.throttle().await;
                redraws.extend(
                    gpu.take_frames()
                        .await
                        .into_iter()
                        .filter(|f| f.pass("drape").is_some()),
                );
            }
            eprintln!(
                "full drape redraw ({} frames): drape {:.3}; mips {:.3}; GPU total {:.3}",
                redraws.len(),
                pass(&redraws, "drape"),
                pass(&redraws, "drape mips"),
                median(redraws.iter().map(|f| f.gpu()).collect())
            );
        });
    }
}

/// The default surface's projection, which the benchmarks draw on.
pub(crate) fn web_mercator() -> orb_crs::Projection {
    *orb_crs::TileMatrixSet::by_id("WebMercatorQuad")
        .expect("built-in tile matrix set")
        .projection()
}
