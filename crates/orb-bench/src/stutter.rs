//! The stutter bench: camera paths that bring many tiles in, frame by
//! frame, with each frame's CPU time split by stage, its GPU time, the
//! bytes it uploaded and the tiles that arrived in it. The report names
//! the worst frames and what they spent their time on, and a histogram of
//! each stage's per-frame times.
//!
//! The scenarios: Liberty's 3D buildings over Manhattan, zooming z12 → z16
//! at pitch 60 and panning on across new ground, plain and then with
//! style-spec#1797's shadows and #1796's ambient occlusion; the Alps terrain style's fast
//! turns and its flight (`web/demo/data/alps.json`); and Liberty on the
//! globe, zooming from the whole world onto Tokyo, and onto Manhattan as
//! its buildings arrive, plain and then lit as the demo's Globe entry
//! lights its own, ground and night side included. Each starts cold from
//! its first camera at rest and runs at a 60 Hz pace, as the gestures bench
//! does. `cargo xtask bench stutter`; in the browser
//! `bench.html?experiment=stutter` runs the same paths through the page's
//! map package and, with `engine=gl`, through GL JS.

use std::fmt::Write;
use std::time::Duration;

use orb_api::{CameraState, FrameTimes, Geodetic, Map, Runtime, Target};

use crate::report::{ms, percentile, resolution};
use crate::{Settings, clock, frame, settle};

/// A 60 Hz display's frame, and two.
const FRAME: Duration = Duration::from_micros(16_667);
const TWO_FRAMES: Duration = Duration::from_micros(33_333);
/// How long the map may take to settle after a path.
const SETTLE_TIMEOUT: Duration = Duration::from_secs(120);
/// Frames the report lists, slowest first.
const WORST: usize = 8;
/// The stage histogram's bucket edges, in ms: under the first, between
/// each two, over the last.
const BUCKETS: [f64; 7] = [0.25, 1.0, 2.0, 4.0, 8.0, 16.7, 33.3];

/// OpenFreeMap's live Liberty, with its own 3D buildings.
pub const LIBERTY: &str = "https://tiles.openfreemap.org/styles/liberty";
/// Midtown Manhattan, by Times Square.
const MANHATTAN: (f64, f64) = (-73.9855, 40.758);
/// Where the Manhattan pan ends: down the avenues, by Washington Square.
const MANHATTAN_SOUTH: (f64, f64) = (-73.9975, 40.731);
/// Where the globe zoom onto Manhattan starts: over the Atlantic.
const ATLANTIC: (f64, f64) = (-40.0, 30.0);
/// Tokyo Station, and where the globe zoom starts over Asia.
const TOKYO_STATION: (f64, f64) = (139.767, 35.681);
const ASIA: (f64, f64) = (105.0, 30.0);

/// Which style a scenario draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    /// Liberty, or the style URL the bench was given.
    Liberty,
    /// The Alps terrain style, `alps.json`.
    Alps,
}

/// A scripted camera path.
struct Scenario {
    name: &'static str,
    style: Style,
    /// An edit to the style once it loaded: the globe projection, or the
    /// buildings' shadows and ambient occlusion.
    edit: Option<&'static str>,
    /// Frames the path takes; a flight takes its legs instead.
    frames: usize,
    /// The camera at `f` in `0..=1`; a flight's first camera.
    camera: fn(f64) -> CameraState,
    flight: bool,
}

/// 5 s zooming in over Midtown at pitch 60, 5 s panning down the avenues.
fn manhattan(f: f64) -> CameraState {
    let (lon, lat) = MANHATTAN;
    let zoom = lerp(12.0, 16.0, ease((2.0 * f).min(1.0)));
    let pan = (2.0 * f - 1.0).clamp(0.0, 1.0);
    let (lon, lat) = (
        lerp(lon, MANHATTAN_SOUTH.0, pan),
        lerp(lat, MANHATTAN_SOUTH.1, pan),
    );
    camera(lon, lat, zoom, 29.0, 60.0)
}

/// 6 s from the whole globe down to Midtown at z16, tilting to pitch 60
/// as the buildings come in, then 2 s turning there.
fn globe_manhattan(f: f64) -> CameraState {
    let zoom_in = (f / 0.75).min(1.0);
    let e = ease(zoom_in);
    let g = ease((zoom_in / 0.6).min(1.0));
    let lon = lerp(ATLANTIC.0, MANHATTAN.0, g);
    let lat = lerp(ATLANTIC.1, MANHATTAN.1, g);
    let turn = ((f - 0.75) / 0.25).clamp(0.0, 1.0);
    camera(
        lon,
        lat,
        lerp(1.5, 16.0, e),
        29.0 + 30.0 * turn,
        lerp(0.0, 60.0, e),
    )
}

/// The globe, as the globe zoom onto Manhattan draws it.
const GLOBE: &str = r#"[["setProjection", {"type": "globe"}]]"#;
/// The globe with Liberty's buildings lit as the demo's Globe entry lights
/// its own: the sun 60° from overhead, shadows at 0.5, occlusion 0.6 on
/// the walls and 0.5 on the ground, and the ground lit, night side and
/// all (`web/demo/data/globe.json`'s `light`).
const GLOBE_LIT: &str = r##"[
    ["setProjection", {"type": "globe"}],
    ["setLight", {"anchor": "viewport", "position": [1.15, 210, 60], "intensity": 0.35, "dev.sargunv.orbis:shadow-intensity": 0.5,
        "dev.sargunv.orbis:ground-lighting": 1, "dev.sargunv.orbis:ambient-color": "#8d9fd6",
        "dev.sargunv.orbis:ambient-intensity": 0.5, "dev.sargunv.orbis:twilight": 14}],
    ["setPaintProperty", "building-3d", "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-wall-intensity", 0.6],
    ["setPaintProperty", "building-3d", "dev.sargunv.orbis:fill-extrusion-ambient-occlusion-ground-intensity", 0.5]
]"##;

fn camera(lon: f64, lat: f64, zoom: f64, bearing: f64, pitch: f64) -> CameraState {
    CameraState {
        bearing,
        pitch,
        ..CameraState::new(Geodetic::new(lon, lat), zoom, &crate::web_mercator())
    }
}

/// Ease in and out over `0..=1`.
fn ease(f: f64) -> f64 {
    f * f * (3.0 - 2.0 * f)
}

fn lerp(a: f64, b: f64, f: f64) -> f64 {
    a + (b - a) * f
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "buildings: zoom z12 → z16 at pitch 60, then pan, Manhattan",
        style: Style::Liberty,
        edit: None,
        frames: 600,
        camera: manhattan,
        flight: false,
    },
    Scenario {
        name: "buildings, shadows and AO: zoom z12 → z16 at pitch 60, then pan, Manhattan",
        style: Style::Liberty,
        edit: Some(crate::SHADOWS_AND_OCCLUSION),
        frames: 600,
        camera: manhattan,
        flight: false,
    },
    Scenario {
        name: "terrain: fast turns at z11.6, pitch 80, Alps",
        style: Style::Alps,
        edit: None,
        frames: 360,
        camera: |f| {
            camera(
                crate::ALPS.0,
                crate::ALPS.1,
                11.6,
                -32.0 + 3.0 * 360.0 * f,
                80.0,
            )
        },
        flight: false,
    },
    Scenario {
        name: "terrain: the Alps flight",
        style: Style::Alps,
        edit: None,
        frames: 0,
        camera: |_| camera(crate::ALPS.0, crate::ALPS.1, 11.6, -32.0, 72.0),
        flight: true,
    },
    Scenario {
        name: "globe: zoom z1.5 → z13 onto Tokyo",
        style: Style::Liberty,
        edit: Some(r#"[["setProjection", {"type": "globe"}]]"#),
        frames: 480,
        camera: |f| {
            let e = ease(f);
            let lon = lerp(ASIA.0, TOKYO_STATION.0, ease(f.min(0.6) / 0.6));
            let lat = lerp(ASIA.1, TOKYO_STATION.1, ease(f.min(0.6) / 0.6));
            camera(lon, lat, lerp(1.5, 13.0, e), 0.0, lerp(0.0, 40.0, e))
        },
        flight: false,
    },
    Scenario {
        name: "globe, buildings: zoom z1.5 → z16 onto Manhattan",
        style: Style::Liberty,
        edit: Some(GLOBE),
        frames: 480,
        camera: globe_manhattan,
        flight: false,
    },
    Scenario {
        name: "globe, buildings, shadows and AO: zoom z1.5 → z16 onto Manhattan",
        style: Style::Liberty,
        edit: Some(GLOBE_LIT),
        frames: 480,
        camera: globe_manhattan,
        flight: false,
    },
];

/// The stages a frame's CPU time splits into, in the report's order.
const STAGES: [&str; 14] = [
    "arrivals",
    "upload",
    "terrain",
    "plan",
    "drape plan",
    "place",
    "placement",
    "cull",
    "bundles",
    "drape passes",
    "encode",
    "submit",
    "runtime other",
    "host",
];

/// One frame: its CPU time (`Map::render`) and stages, GPU time, bytes
/// uploaded and tiles arrived.
#[derive(Clone, Debug)]
struct Frame {
    index: usize,
    zoom: f64,
    cpu: Duration,
    /// `Map::render`'s instructions, where counted.
    instructions: Option<u64>,
    /// Per stage in [`STAGES`].
    stages: [Duration; STAGES.len()],
    /// Its submissions' GPU time, each's from its first pass's start to
    /// its last one's end, as the gestures and frames benches time them.
    gpu: Option<Duration>,
    /// The GPU's time by pass label, slowest first. On tiled GPUs passes
    /// overlap, so they sum to more than [`gpu`](Self::gpu), and a pass
    /// split in two takes longer here without the frame doing so.
    passes: Vec<(String, Duration)>,
    uploaded: u64,
    arrived: u32,
    /// Layers it left out while their pipelines compiled.
    waiting: u32,
}

impl Frame {
    fn stages(times: &FrameTimes, cpu: Duration) -> [Duration; STAGES.len()] {
        let r = &times.render;
        let named = [
            times.arrivals,
            times.upload,
            times.terrain,
            times.plan,
            times.drape,
            times.place,
            times.placement,
            r.cull,
            r.bundles,
            r.drape,
            r.encode,
            r.submit,
        ];
        let us = |v: u32| Duration::from_micros(u64::from(v));
        let total = us(times.total);
        let mut out = [Duration::ZERO; STAGES.len()];
        for (o, v) in out.iter_mut().zip(named) {
            *o = us(v);
        }
        let sum: Duration = out.iter().sum();
        out[STAGES.len() - 2] = total.saturating_sub(sum);
        out[STAGES.len() - 1] = cpu.saturating_sub(total);
        out
    }

    /// What a display waits for: the slower of the CPU and the GPU.
    fn time(&self) -> Duration {
        self.cpu.max(self.gpu.unwrap_or_default())
    }
}

#[derive(Debug)]
struct PathResult {
    name: &'static str,
    style: String,
    frames: Vec<Frame>,
    settle: Duration,
}

/// The scenarios' frames.
#[derive(Debug)]
pub struct StutterReport {
    about: Vec<String>,
    size: (u32, u32),
    gpu_timestamps: bool,
    cpu_resolution: Duration,
    paths: Vec<PathResult>,
    warnings: Vec<String>,
}

/// Runs each scenario the settings' filter keeps. `liberty` is the style
/// URL Liberty's scenarios draw, `alps` the Alps terrain style.
pub(crate) fn run(
    runtime: &Runtime,
    settings: &Settings,
    about: Vec<String>,
    (liberty, alps): (&str, &str),
    mut progress: impl FnMut(&str),
) -> Result<StutterReport, String> {
    let (width, height) = settings.physical_size();
    let mut target = Target::texture(runtime, width, height, f64::from(settings.ratio));
    let gpu_timestamps = runtime.stats().gpu_timestamps;
    let mut warnings = Vec::new();
    let mut paths = Vec::new();
    let chosen = SCENARIOS.iter().filter(|s| {
        settings
            .filter
            .as_ref()
            .is_none_or(|f| s.name.contains(f.as_str()))
    });
    for scenario in chosen {
        let style = match scenario.style {
            Style::Liberty => liberty,
            Style::Alps => alps,
        };
        progress(&format!("{}: loading its first view", scenario.name));
        let mut map = crate::new_map(runtime, style, (scenario.camera)(0.0), settings.antialias);
        let epoch = clock::now();
        frame(&mut map, &mut target, 0.0)?;
        settle(&mut map, &mut target, &mut warnings)?;
        if let Some(edit) = scenario.edit {
            let tx = orb_api::Transaction::from_commands(edit)?;
            map.edit(&tx).map_err(|e| e.to_string())?;
            settle(&mut map, &mut target, &mut warnings)?;
        }
        let base = (clock::now() - epoch).as_secs_f64() * 1000.0;
        progress(&format!("{}: moving", scenario.name));
        let frames = run_path(
            runtime,
            &mut map,
            &mut target,
            (scenario, base),
            settings.legs,
            &mut warnings,
        )?;
        let ended = clock::now();
        let mut now = base + frames.len() as f64 * FRAME.as_secs_f64() * 1000.0;
        loop {
            now += FRAME.as_secs_f64() * 1000.0;
            let (report, _) = frame(&mut map, &mut target, now)?;
            warnings.extend(crate::take_warnings(&mut map));
            if report.idle || clock::now() - ended > SETTLE_TIMEOUT {
                break;
            }
            std::thread::sleep(FRAME);
        }
        paths.push(PathResult {
            name: scenario.name,
            style: style.to_string(),
            frames,
            settle: clock::now() - ended,
        });
    }
    if paths.is_empty() {
        return Err("no scenario matches the filter".into());
    }
    warnings.sort();
    warnings.dedup();
    Ok(StutterReport {
        about,
        size: (width, height),
        gpu_timestamps,
        cpu_resolution: clock::resolution(),
        paths,
        warnings,
    })
}

/// Moves the camera along the path once each 60 Hz frame from `base`
/// (milliseconds), timing each; a flight flies its first `first_legs`.
fn run_path(
    runtime: &Runtime,
    map: &mut Map,
    target: &mut Target,
    (scenario, base): (&Scenario, f64),
    first_legs: Option<usize>,
    warnings: &mut Vec<String>,
) -> Result<Vec<Frame>, String> {
    let mut legs = match scenario.flight {
        true => crate::gestures::flight_legs(),
        false => Vec::new(),
    };
    legs.truncate(first_legs.unwrap_or(usize::MAX));
    legs.reverse();
    let gpu_timestamps = runtime.stats().gpu_timestamps;
    let mut frames = Vec::new();
    runtime.set_gpu_timing(gpu_timestamps);
    let started = clock::now();
    for f in 1.. {
        let now = base + (FRAME * f).as_secs_f64() * 1000.0;
        if scenario.flight {
            if !map.is_moving() {
                let Some((target, options)) = legs.pop() else {
                    break;
                };
                map.fly_to(&target, &options);
            }
        } else if f as usize > scenario.frames {
            break;
        } else {
            map.jump_to_state((scenario.camera)(f64::from(f) / scenario.frames as f64));
        }
        let before = runtime.stats().uploaded_bytes;
        // The target keeps two frames in flight, as a swapchain does.
        let (report, cpu, instructions) = crate::frame_counted(map, target, now)?;
        warnings.extend(crate::take_warnings(map));
        frames.push(Frame {
            index: f as usize,
            zoom: map.zoom(),
            cpu,
            instructions,
            stages: Frame::stages(&report.times, cpu),
            gpu: None,
            passes: Vec::new(),
            uploaded: runtime.stats().uploaded_bytes - before,
            arrived: report.tiles_arrived,
            waiting: report.frame.nodes_waiting,
        });
        let next = started + FRAME * f;
        std::thread::sleep(next.saturating_sub(clock::now()));
        // The frame's submissions, done by now unless the GPU is behind:
        // collecting them waits for it then.
        if gpu_timestamps {
            let frame = frames.last_mut().expect("pushed");
            let timings = runtime.gpu_timings();
            frame.gpu = Some(timings.iter().map(|s| s.gpu()).sum());
            frame.passes = by_name(timings.into_iter().flat_map(|s| s.passes));
        }
    }
    runtime.set_gpu_timing(false);
    Ok(frames)
}

/// Durations summed by name, the longest first.
fn by_name(parts: impl Iterator<Item = (String, Duration)>) -> Vec<(String, Duration)> {
    let mut totals: Vec<(String, Duration)> = Vec::new();
    for (name, d) in parts {
        match totals.iter_mut().find(|(n, _)| *n == name) {
            Some((_, sum)) => *sum += d,
            None => totals.push((name, d)),
        }
    }
    totals.sort_by_key(|&(_, d)| std::cmp::Reverse(d));
    totals
}

fn sorted(xs: impl Iterator<Item = Duration>) -> Vec<Duration> {
    let mut v: Vec<Duration> = xs.collect();
    v.sort_unstable();
    v
}

fn mib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / f64::from(1 << 20))
}

fn over(xs: &[Duration], limit: Duration) -> usize {
    xs.iter().filter(|&&t| t > limit).count()
}

impl StutterReport {
    pub(crate) fn write(&self, out: &mut String) -> std::fmt::Result {
        writeln!(out, "### Stutter bench")?;
        writeln!(out)?;
        for line in &self.about {
            writeln!(out, "- {line}")?;
        }
        writeln!(
            out,
            "- {}×{} px offscreen; the camera moved once each 60 Hz frame, each path from a cold \
             map at rest; GPU: {}; CPU clock resolution {}",
            self.size.0,
            self.size.1,
            if self.gpu_timestamps {
                "timestamp queries around every pass, a frame's GPU time from its first pass's \
                 start to its last one's end"
            } else {
                "**no timestamp queries**"
            },
            resolution(Some(self.cpu_resolution)),
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| path | frames | CPU p50 | p99 | max | > 16.7 ms | > 33.3 ms | GPU p50 | p99 | max | \
             > 16.7 ms | worst of both > 16.7 ms | tiles arrived | MiB uploaded (peak frame) | \
             frames with layers compiling | settle ms | render M instructions |"
        )?;
        writeln!(
            out,
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        )?;
        for p in &self.paths {
            let cpu = sorted(p.frames.iter().map(|f| f.cpu));
            let gpu = sorted(p.frames.iter().filter_map(|f| f.gpu));
            let worst = sorted(p.frames.iter().map(Frame::time));
            let gpu_at = |q: f64| {
                if gpu.is_empty() {
                    "—".to_string()
                } else {
                    ms(percentile(&gpu, q))
                }
            };
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} ({}) | {} | {} | {} |",
                p.name,
                p.frames.len(),
                ms(percentile(&cpu, 0.5)),
                ms(percentile(&cpu, 0.99)),
                ms(percentile(&cpu, 1.0)),
                over(&cpu, FRAME),
                over(&cpu, TWO_FRAMES),
                gpu_at(0.5),
                gpu_at(0.99),
                gpu_at(1.0),
                over(&gpu, FRAME),
                over(&worst, FRAME),
                p.frames.iter().map(|f| f.arrived).sum::<u32>(),
                mib(p.frames.iter().map(|f| f.uploaded).sum()),
                mib(p.frames.iter().map(|f| f.uploaded).max().unwrap_or(0)),
                p.frames.iter().filter(|f| f.waiting > 0).count(),
                if p.settle > SETTLE_TIMEOUT {
                    format!("> {}", SETTLE_TIMEOUT.as_millis())
                } else {
                    format!("{:.0}", p.settle.as_secs_f64() * 1000.0)
                },
                (p.frames.iter().map(|f| f.instructions))
                    .sum::<Option<u64>>()
                    .map_or("—".into(), |n| format!("{:.1}", n as f64 / 1e6)),
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Times in ms. CPU: `Map::render`; GPU: its submissions, from the first pass's start \
             to the last one's end; worst of both: the slower of the two, what a display waits \
             for; frames with layers compiling: frames that left a layer out while its \
             pipelines compiled; settle: from the path's end until the map is idle; render M instructions: those \
             `Map::render` retired on its thread over the path (macOS), which other work on the \
             machine barely moves. Passes overlap on tiled GPUs, so their times below sum to \
             more than the frame's."
        )?;
        for p in &self.paths {
            writeln!(out)?;
            writeln!(out, "#### {}", p.name)?;
            writeln!(out)?;
            writeln!(out, "- style: {}", p.style)?;
            write_shares(out, p)?;
            writeln!(out)?;
            write_passes(out, p)?;
            writeln!(out)?;
            write_worst(out, p)?;
            writeln!(out)?;
            write_histogram(out, p)?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Stages: arrivals: applying arrived results but for uploads and label commits; upload: \
             arrived tiles' meshes, rows and images into the GPU, room made for them; terrain: the \
             DEM tiles and cover; plan: the sources' tiles for the view; drape plan: drape slots \
             chosen and their placements written; place: the view's tiles put in the scene; \
             placement: building the placement lane's frame, applying its commits; cull: the \
             cull's candidates laid out and placement rows written; bundles: the main bundles \
             recorded again; drape passes: redrawn drape slots' bundles and passes; encode: the \
             other passes listed; submit: the passes and the buffer writes staged for them \
             recorded into wgpu and submitted; runtime other: paint, the camera, picking; host: \
             `orb-api` around the runtime's frame."
        )?;
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
        }
        Ok(())
    }
}

/// Each stage's share of the CPU time of the frames over 16.7 ms of it,
/// and each pass's of the GPU time of those over 16.7 ms of that.
fn write_shares(out: &mut String, p: &PathResult) -> std::fmt::Result {
    let slow = |f: &&Frame| f.cpu > FRAME;
    let stages = p
        .frames
        .iter()
        .filter(slow)
        .flat_map(|f| (STAGES.iter().zip(f.stages)).map(|(stage, d)| (stage.to_string(), d)));
    write_share(out, "CPU", p.frames.iter().filter(slow).count(), stages)?;
    let slow = |f: &&Frame| f.gpu.is_some_and(|g| g > FRAME);
    let passes = p.frames.iter().filter(slow).flat_map(|f| f.passes.clone());
    write_share(out, "GPU", p.frames.iter().filter(slow).count(), passes)
}

fn write_share(
    out: &mut String,
    what: &str,
    frames: usize,
    parts: impl Iterator<Item = (String, Duration)>,
) -> std::fmt::Result {
    if frames == 0 {
        return writeln!(out, "- no frame took over 16.7 ms of {what}");
    }
    let totals = by_name(parts);
    let all: Duration = totals.iter().map(|(_, d)| *d).sum();
    let shares: Vec<String> = (totals.iter())
        .map(|(name, d)| (name, 100.0 * d.as_secs_f64() / all.as_secs_f64().max(1e-9)))
        .filter(|&(_, share)| share >= 0.5)
        .map(|(name, share)| format!("{name} {share:.0}%"))
        .collect();
    writeln!(
        out,
        "- the {frames} frames over 16.7 ms of {what} spent it on: {}",
        shares.join(", ")
    )
}

/// Per GPU pass label, the frames it ran in and its time per frame: the
/// path's total, p50, p99 and max over those frames. Totals compare runs
/// whose tiles arrived in other frames.
fn write_passes(out: &mut String, p: &PathResult) -> std::fmt::Result {
    let mut labels: Vec<&str> = Vec::new();
    for (label, _) in p.frames.iter().flat_map(|f| &f.passes) {
        if !labels.contains(&label.as_str()) {
            labels.push(label);
        }
    }
    if labels.is_empty() {
        return Ok(());
    }
    let mut per_label: Vec<(&str, Vec<Duration>)> = (labels.into_iter())
        .map(|label| {
            let times = p
                .frames
                .iter()
                .filter_map(|f| (f.passes.iter().find(|(l, _)| l == label)).map(|&(_, d)| d));
            (label, sorted(times))
        })
        .collect();
    per_label.sort_by_key(|(_, t)| std::cmp::Reverse(t.iter().sum::<Duration>()));
    writeln!(out, "| GPU pass | frames | total | p50 | p99 | max |")?;
    writeln!(out, "|---|---|---|---|---|---|")?;
    for (label, times) in per_label {
        writeln!(
            out,
            "| {label} | {} | {:.1} | {} | {} | {} |",
            times.len(),
            times.iter().sum::<Duration>().as_secs_f64() * 1000.0,
            ms(percentile(&times, 0.5)),
            ms(percentile(&times, 0.99)),
            ms(percentile(&times, 1.0)),
        )?;
    }
    Ok(())
}

/// The slowest frames, by what a display waits for, with their stages.
fn write_worst(out: &mut String, p: &PathResult) -> std::fmt::Result {
    let mut worst: Vec<&Frame> = p.frames.iter().collect();
    worst.sort_by_key(|f| std::cmp::Reverse(f.time()));
    worst.truncate(WORST);
    write!(
        out,
        "| frame | zoom | CPU | GPU | GPU's slowest passes | tiles arrived | KiB uploaded |"
    )?;
    for stage in STAGES {
        write!(out, " {stage} |")?;
    }
    writeln!(out)?;
    writeln!(out, "|{}", "---|".repeat(7 + STAGES.len()))?;
    for f in worst {
        write!(
            out,
            "| {} | {:.2} | {} | {} | {} | {} | {:.0} |",
            f.index,
            f.zoom,
            ms(f.cpu),
            f.gpu.map_or("—".into(), ms),
            (f.passes.iter().take(3))
                .map(|(label, d)| format!("{label} {:.1}", d.as_secs_f64() * 1000.0))
                .collect::<Vec<_>>()
                .join(", "),
            f.arrived,
            f.uploaded as f64 / 1024.0,
        )?;
        for s in f.stages {
            write!(out, " {} |", ms(s))?;
        }
        writeln!(out)?;
    }
    Ok(())
}

/// Per stage, how many frames spent each bucket's time on it, its p99 and
/// max, and its share of the path's CPU time.
fn write_histogram(out: &mut String, p: &PathResult) -> std::fmt::Result {
    let edges: Vec<String> = BUCKETS.iter().map(|e| format!("{e}")).collect();
    write!(out, "| stage | < {} |", edges[0])?;
    for pair in edges.windows(2) {
        write!(out, " {}–{} |", pair[0], pair[1])?;
    }
    writeln!(out, " > {} | p99 | max | share |", edges[edges.len() - 1])?;
    writeln!(out, "|{}", "---|".repeat(BUCKETS.len() + 5))?;
    let all: Duration = p.frames.iter().map(|f| f.cpu).sum();
    for (i, stage) in STAGES.iter().enumerate() {
        let times = sorted(p.frames.iter().map(|f| f.stages[i]));
        let mut counts = [0usize; BUCKETS.len() + 1];
        for t in &times {
            let t = t.as_secs_f64() * 1000.0;
            counts[BUCKETS.iter().take_while(|&&e| t >= e).count()] += 1;
        }
        write!(out, "| {stage} |")?;
        for c in counts {
            write!(out, " {c} |")?;
        }
        let total: Duration = times.iter().sum();
        writeln!(
            out,
            " {} | {} | {:.0}% |",
            ms(percentile(&times, 0.99)),
            ms(percentile(&times, 1.0)),
            100.0 * total.as_secs_f64() / all.as_secs_f64().max(1e-9),
        )?;
    }
    Ok(())
}
