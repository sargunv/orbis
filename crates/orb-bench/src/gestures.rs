//! The gestures bench: what a user's rotate, tilt and zoom cost the render
//! thread frame by frame, with tiles arriving as they load.
//!
//! Each path starts cold from its first camera at rest and runs at a 60 Hz
//! pace: one frame each 16.7 ms, the camera moved before each, the map's
//! jobs loading meanwhile. Per frame: `Map::render`'s CPU time and the bytes
//! uploaded. A frame over 16.7 ms misses a 60 Hz display's refresh; the
//! tail (p99, max) is where tile arrivals and per-frame work that should
//! wait for a change show. Natively `cargo xtask bench --experiment
//! gestures`; in the browser `bench.html?experiment=gestures` times the same
//! paths through the page's map package, render worker included.

use std::fmt::Write;
use std::time::Duration;

use orb_hal::TextureId;
use orb_runtime::{CameraState, FrameReport, Map, TargetFormat, Viewport};

use crate::report::{ms, resolution};
use crate::timed::TimedGpu;
use crate::{GesturePaths, Settings, Timing, clock, settle};

/// A 60 Hz display's frame.
const FRAME: Duration = Duration::from_micros(16_667);
/// Frames per path: 6 s at 60 Hz.
const PATH_FRAMES: usize = 360;
/// A label that all but vanishes (opacity under 0.1) and is back in full
/// (0.9) within this many frames blinked: no fade is that quick (300 ms).
const FLASH_FRAMES: usize = 10;
/// How long the map may take to settle after a path.
const SETTLE_TIMEOUT: Duration = Duration::from_secs(120);

/// A camera path: its name, the camera at `f` in `0..=1` (moved once a
/// frame for [`PATH_FRAMES`]), or, for a flight, its first camera.
struct Path {
    name: &'static str,
    camera: fn(&Settings, f64) -> CameraState,
    /// Flies the terrain demo's flight from the first camera instead.
    flight: bool,
}

const CITY_PATHS: &[Path] = &[
    Path {
        name: "rotate and tilt at z14",
        camera: |s, f| CameraState {
            bearing: 360.0 * f,
            pitch: 30.0 - 30.0 * (f * std::f64::consts::TAU).cos(),
            ..CameraState::new(s.center, 14.0, &crate::web_mercator())
        },
        flight: false,
    },
    Path {
        name: "zoom z12 → z16 → z12",
        camera: |s, f| {
            let zoom = 14.0 - 2.0 * (f * std::f64::consts::TAU).cos();
            CameraState::new(s.center, zoom, &crate::web_mercator())
        },
        flight: false,
    },
];

/// The terrain demo's view: z11.6, bearing −32, pitch 72.
fn alps_view(s: &Settings, lon: f64, bearing: f64) -> CameraState {
    let center = orb_crs::Geodetic::new(s.center.lon + lon, s.center.lat);
    CameraState {
        bearing,
        pitch: 72.0,
        ..CameraState::new(center, 11.6, &crate::web_mercator())
    }
}

const ALPS_PATHS: &[Path] = &[
    Path {
        name: "rotate at z11.6, pitch 72",
        camera: |s, f| alps_view(s, 0.0, -32.0 + 360.0 * f),
        flight: false,
    },
    Path {
        name: "pan at z11.6, pitch 72",
        camera: |s, f| alps_view(s, 0.15 * (f * std::f64::consts::TAU).sin(), -32.0),
        flight: false,
    },
    Path {
        name: "Alps flight (60 s)",
        camera: |s, _| alps_view(s, 0.0, -32.0),
        flight: true,
    },
];

/// The terrain demo's flight (`web/demo/data/alps-flight.json`): `flyTo`
/// legs, as the demos fly them.
fn flight_legs() -> Vec<(orb_camera::CameraTarget, orb_camera::FlyOptions)> {
    let json: serde_json::Value =
        serde_json::from_str(include_str!("../../../web/demo/data/alps-flight.json"))
            .expect("the flight parses");
    json["legs"]
        .as_array()
        .expect("the flight's legs")
        .iter()
        .map(|leg| {
            let number = |key: &str| leg[key].as_f64();
            let center = &leg["center"];
            (
                orb_camera::CameraTarget {
                    center: Some(orb_crs::Geodetic::new(
                        center[0].as_f64().expect("lon"),
                        center[1].as_f64().expect("lat"),
                    )),
                    zoom: number("zoom"),
                    bearing: number("bearing"),
                    pitch: number("pitch"),
                    ..Default::default()
                },
                orb_camera::FlyOptions {
                    duration: number("duration"),
                    ..Default::default()
                },
            )
        })
        .collect()
}

#[derive(Debug)]
struct PathResult {
    name: &'static str,
    /// `Map::render` per frame, sorted.
    cpu: Vec<Duration>,
    /// The render thread's placement work per frame
    /// (`FrameReport::placement_us`), sorted.
    placement: Vec<Duration>,
    /// `Map::render` until the GPU finished the frame, sorted; empty with
    /// timestamps, which don't wait for each frame.
    done: Vec<Duration>,
    /// Bytes uploaded in the frame that uploaded most.
    peak_bytes: u64,
    /// Frames that drew with tiles still loading.
    loading: usize,
    /// Frames that laid the cull's candidates out again, and tile rows
    /// written over the path.
    rebuilds: usize,
    rows: u64,
    /// Labels shown in full that all but vanished and were back in full
    /// within [`FLASH_FRAMES`] (#55), and the frames between.
    blinks: usize,
    blink_frames: usize,
    /// From the path's end until the map is idle.
    settle: Duration,
}

impl PathResult {
    fn at(&self, p: f64) -> Duration {
        percentile(&self.cpu, p)
    }
}

/// A column of the frames to GPU completion, or – without them.
fn done(times: &[Duration], f: impl Fn(&[Duration]) -> String) -> String {
    if times.is_empty() {
        "–".into()
    } else {
        f(times)
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted.get(i).copied().unwrap_or_default()
}

/// The paths' frame times.
#[derive(Debug)]
pub struct GesturesReport {
    about: Vec<String>,
    size: (u32, u32),
    settings: Settings,
    cpu_resolution: Duration,
    paths: Vec<PathResult>,
    warnings: Vec<String>,
}

fn uploaded(gpu: &TimedGpu) -> u64 {
    let u = gpu.inner().uploaded();
    u.from_core + u.from_js + u.texels_from_core + u.texels_from_js
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run(
    gpu: &mut TimedGpu,
    timing: Timing,
    target: (TextureId, (u32, u32)),
    format: TargetFormat,
    settings: &Settings,
    about: Vec<String>,
    mut new_map: impl FnMut(CameraState) -> Result<Map, String>,
    mut progress: impl FnMut(&str),
) -> Result<GesturesReport, String> {
    let (target, size) = target;
    let viewport = Viewport {
        width: f64::from(settings.width),
        height: f64::from(settings.height),
    };
    // The path's map's renderer: a new map gets a new one.
    let renderer = std::cell::RefCell::new(None);
    let render = |gpu: &mut TimedGpu, map: &mut Map, time: f64| -> (FrameReport, Duration) {
        let started = clock::now();
        let mut renderer = renderer.borrow_mut();
        let renderer = renderer.get_or_insert_with(|| map.create_renderer(gpu, format));
        let report = map.render(gpu, renderer, target, viewport, settings.ratio, time);
        (report, clock::now() - started)
    };
    let mut warnings = Vec::new();
    let mut paths = Vec::new();
    let paths_run = match settings.paths {
        GesturePaths::City => CITY_PATHS,
        GesturePaths::Alps => ALPS_PATHS,
    };
    for path in paths_run {
        // A map per path, so each starts cold.
        progress(&format!("{}: loading its first view", path.name));
        let start = (path.camera)(settings, 0.0);
        let mut map = new_map(start)?;
        *renderer.borrow_mut() = None;
        // Frames' times: the wall clock while the view loads, then 60 Hz
        // from there, so fades and transitions run on.
        let epoch = clock::now();
        let since = || (clock::now() - epoch).as_secs_f64() * 1000.0;
        settle(
            gpu,
            &mut map,
            &|g, m| render(g, m, since()).0,
            &mut warnings,
        )
        .await?;
        let base = since();
        let mut legs = if path.flight {
            flight_legs()
        } else {
            Vec::new()
        };
        legs.reverse();
        progress(&format!("{}: moving", path.name));
        let (mut cpu, mut done, mut peak_bytes, mut loading) = (Vec::new(), Vec::new(), 0, 0);
        let mut placement = Vec::new();
        // Per label text: shown in full, and since which frame it is gone.
        let mut labels: std::collections::HashMap<Box<str>, Option<usize>> = Default::default();
        let (mut blinks, mut blink_frames) = (0, 0);
        let (mut rebuilds, mut rows) = (0, 0);
        let started = clock::now();
        let mut now = base;
        for f in 1.. {
            now = base + (FRAME * f).as_secs_f64() * 1000.0;
            // A flight's camera stands on the terrain, as the demos' do.
            if path.flight {
                if !map.is_moving() {
                    let Some((target, options)) = legs.pop() else {
                        break;
                    };
                    map.fly_to(&target, &options);
                }
            } else if f as usize > PATH_FRAMES {
                break;
            } else {
                map.jump_to_state((path.camera)(settings, f64::from(f) / PATH_FRAMES as f64));
            }
            let before = uploaded(gpu);
            let start = clock::now();
            let (report, time) = render(gpu, &mut map, now);
            match timing {
                Timing::Timestamps => gpu.throttle().await,
                Timing::WorkDone => {
                    gpu.finish().await;
                    done.push(clock::now() - start);
                }
            }
            // A flight's thousands of frames outrun the timestamp queries;
            // this bench reads CPU times only.
            if f % 64 == 0 {
                gpu.take_frames().await;
            }
            cpu.push(time);
            placement.push(Duration::from_micros(u64::from(report.placement_us)));
            peak_bytes = peak_bytes.max(uploaded(gpu) - before);
            loading += usize::from(report.tiles_loading > 0);
            rebuilds += usize::from(report.frame.candidates_rebuilt);
            rows += u64::from(report.frame.placements_written);
            let shown = map.shown_labels();
            for (name, dim) in &mut labels {
                let opacity = shown.get(name).copied().unwrap_or(0.0);
                match *dim {
                    None if opacity < 0.1 => *dim = Some(f as usize),
                    Some(since) if opacity >= 0.9 => {
                        if f as usize - since <= FLASH_FRAMES {
                            blinks += 1;
                            blink_frames += f as usize - since;
                        }
                        *dim = None;
                    }
                    _ => {}
                }
            }
            for (name, opacity) in shown {
                if opacity >= 0.9 {
                    labels.entry(name).or_insert(None);
                }
            }
            warnings.extend(crate::take_warnings(&mut map));
            let next = started + FRAME * f;
            clock::sleep(next.saturating_sub(clock::now())).await;
        }
        let ended = clock::now();
        loop {
            now += FRAME.as_secs_f64() * 1000.0;
            let (report, _) = render(gpu, &mut map, now);
            gpu.take_frames().await;
            warnings.extend(crate::take_warnings(&mut map));
            if report.idle {
                break;
            }
            if clock::now() - ended > SETTLE_TIMEOUT {
                return Err(format!("{}: timed out settling", path.name));
            }
            clock::sleep(FRAME).await;
        }
        gpu.take_frames().await;
        cpu.sort_unstable();
        done.sort_unstable();
        placement.sort_unstable();
        paths.push(PathResult {
            done,
            name: path.name,
            cpu,
            placement,
            peak_bytes,
            loading,
            rebuilds,
            rows,
            blinks,
            blink_frames,
            settle: clock::now() - ended,
        });
    }
    warnings.sort();
    warnings.dedup();
    Ok(GesturesReport {
        about,
        size,
        settings: settings.clone(),
        cpu_resolution: clock::resolution(),
        paths,
        warnings,
    })
}

impl GesturesReport {
    pub(crate) fn write(&self, out: &mut String) -> std::fmt::Result {
        let s = &self.settings;
        writeln!(out, "### Gestures bench")?;
        writeln!(out)?;
        for line in &self.about {
            writeln!(out, "- {line}")?;
        }
        writeln!(
            out,
            "- {}×{} px offscreen at {:.4}, {:.4}; {PATH_FRAMES} frames a path (a flight: its \
             legs) at 60 Hz, each path from a cold map at rest; CPU clock resolution {}",
            self.size.0,
            self.size.1,
            s.center.lon,
            s.center.lat,
            resolution(Some(self.cpu_resolution)),
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| path | render CPU p50 | p95 | p99 | max | frames > 16.7 ms | to GPU done p50 | p95 | frames > 16.7 ms | frames loading | candidate rebuilds | rows written per frame | peak MiB/frame | settle ms | label blinks (frames) | placement p50 | p95 | max |"
        )?;
        writeln!(
            out,
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        )?;
        for p in &self.paths {
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {:.1} | {:.0} | {} ({}) | {} | {} | {} |",
                p.name,
                ms(p.at(0.5)),
                ms(p.at(0.95)),
                ms(p.at(0.99)),
                ms(p.at(1.0)),
                p.cpu.iter().filter(|&&t| t > FRAME).count(),
                done(&p.done, |d| ms(percentile(d, 0.5))),
                done(&p.done, |d| ms(percentile(d, 0.95))),
                done(&p.done, |d| d
                    .iter()
                    .filter(|&&t| t > FRAME)
                    .count()
                    .to_string()),
                p.loading,
                p.rebuilds,
                p.rows as f64 / p.cpu.len().max(1) as f64,
                p.peak_bytes as f64 / (1024.0 * 1024.0),
                p.settle.as_secs_f64() * 1000.0,
                p.blinks,
                p.blink_frames,
                ms(percentile(&p.placement, 0.5)),
                ms(percentile(&p.placement, 0.95)),
                ms(percentile(&p.placement, 1.0)),
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Times in ms of `Map::render` (applying arrived tiles, uploads, placement, encoding), \
             and with `--timing work-done` until the GPU finished the frame; settle: from the \
             path's end until the map is idle; candidate rebuilds: frames that laid the cull's \
             candidates out again; label blinks: labels shown in full that all but \
             vanished (opacity under 0.1) and were back in full within {FLASH_FRAMES} frames (no \
             fade is that quick), and the frames they were gone; placement: the render thread's \
             share of it (building the frame the lane places, applying commits; plan §6.1 \
             step 10's 0.2 ms)."
        )?;
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
        }
        Ok(())
    }
}
