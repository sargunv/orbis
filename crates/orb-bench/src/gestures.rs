//! The gestures bench: what a user's rotate, tilt and zoom cost the render
//! thread frame by frame, with tiles arriving as they load.
//!
//! Each path starts cold from its first camera at rest and runs at a 60 Hz
//! pace: one frame each 16.7 ms, the camera moved before each, the map's
//! jobs loading meanwhile. Per frame: `Map::render`'s CPU time and the bytes
//! uploaded. A frame over 16.7 ms misses a 60 Hz display's refresh; the
//! tail (p99, max) is where tile arrivals and per-frame work that should
//! wait for a change show. `cargo xtask bench --experiment gestures`; in
//! the browser `bench.html?experiment=gestures` times the same paths
//! through the page's map package.

use std::fmt::Write;
use std::time::Duration;

use orb_api::{CameraState, CameraTarget, FlyOptions, Map, Runtime, Target};

use crate::report::{ms, percentile, resolution};
use crate::{GesturePaths, Settings, clock, frame, settle};

/// A 60 Hz display's frame.
const FRAME: Duration = Duration::from_micros(16_667);
/// Frames per path: 6 s at 60 Hz.
const PATH_FRAMES: usize = 360;
/// A label that all but vanishes (opacity under 0.1) and is back in full
/// (0.9) within this many frames blinked: no fade is that quick (300 ms).
const FLASH_FRAMES: usize = 10;
/// A label back in full within this many frames of all but vanishing
/// dropped and returned: 1 s, a fade out and in with a placement between.
const RETURN_FRAMES: usize = 60;
/// How long the map may take to settle after a path.
const SETTLE_TIMEOUT: Duration = Duration::from_secs(120);

/// A camera path: its name, the camera at `f` in `0..=1` (moved once a
/// frame for [`PATH_FRAMES`]), or, for a flight, its first camera.
struct Path {
    name: &'static str,
    camera: fn(&Settings, f64) -> CameraState,
    /// Flies the Alps terrain style's flight from the first camera instead.
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
    Path {
        name: "zoom z12 → z16 → z12, 3 times",
        camera: |s, f| {
            let zoom = 14.0 - 2.0 * (3.0 * f * std::f64::consts::TAU).cos();
            CameraState::new(s.center, zoom, &crate::web_mercator())
        },
        flight: false,
    },
];

/// The Alps terrain style's view: z11.6, bearing −32, pitch 72.
fn alps_view(s: &Settings, lon: f64, bearing: f64) -> CameraState {
    alps_pitched(s, lon, bearing, 72.0)
}

fn alps_pitched(s: &Settings, lon: f64, bearing: f64, pitch: f64) -> CameraState {
    let center = orb_api::Geodetic::new(s.center.lon + lon, s.center.lat);
    CameraState {
        bearing,
        pitch,
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
        name: "fast turns at z11.6, pitch 80",
        camera: |s, f| alps_pitched(s, 0.0, -32.0 + 3.0 * 360.0 * f, 80.0),
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

/// The Alps terrain style's flight (`web/demo/data/alps-flight.json`): `flyTo`
/// legs, as the demos fly them.
pub(crate) fn flight_legs() -> Vec<(CameraTarget, FlyOptions)> {
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
                CameraTarget {
                    center: Some(orb_api::Geodetic::new(
                        center[0].as_f64().expect("lon"),
                        center[1].as_f64().expect("lat"),
                    )),
                    zoom: number("zoom"),
                    bearing: number("bearing"),
                    pitch: number("pitch"),
                    ..Default::default()
                },
                FlyOptions {
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
    /// Each submission's GPU time (first pass's start to last one's end),
    /// sorted; empty without timestamp queries.
    gpu: Vec<Duration>,
    /// The render thread's placement work per frame
    /// (`FrameTimes::placement`), sorted.
    placement: Vec<Duration>,
    /// Bytes uploaded in the frame that uploaded most.
    peak_bytes: u64,
    /// The drape atlas's largest and summed bytes over the frames, and how
    /// many times it was made again.
    drape: (u64, u64, u32),
    /// Frames that drew with tiles still loading.
    loading: usize,
    /// Frames that laid the cull's candidates out again, and tile rows
    /// written over the path.
    rebuilds: usize,
    rows: u64,
    /// Labels shown in full that all but vanished and were back in full
    /// within [`FLASH_FRAMES`], and the frames between.
    blinks: usize,
    blink_frames: usize,
    /// Labels shown in full that all but vanished, and of those, the ones
    /// back in full within [`RETURN_FRAMES`] (blinks included).
    drops: usize,
    returns: usize,
    /// From the path's end until the map is idle.
    settle: Duration,
}

impl PathResult {
    fn at(&self, p: f64) -> Duration {
        percentile(&self.cpu, p)
    }
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

fn uploaded(runtime: &Runtime) -> u64 {
    runtime.stats().counters.uploaded_bytes()
}

pub(crate) fn run(
    runtime: &Runtime,
    (target, size): (&mut Target, (u32, u32)),
    settings: &Settings,
    about: Vec<String>,
    mut new_map: impl FnMut(CameraState) -> Map,
    mut progress: impl FnMut(&str),
) -> Result<GesturesReport, String> {
    let mut warnings = Vec::new();
    let mut paths = Vec::new();
    let paths_run = match settings.paths {
        GesturePaths::City => CITY_PATHS,
        GesturePaths::Alps => ALPS_PATHS,
    };
    let chosen = paths_run.iter().filter(|p| {
        settings
            .filter
            .as_ref()
            .is_none_or(|f| p.name.contains(f.as_str()))
    });
    for path in chosen {
        // A map per path, so each starts cold.
        progress(&format!("{}: loading its first view", path.name));
        let start = (path.camera)(settings, 0.0);
        let mut map = new_map(start);
        // Frames' times: the wall clock while the view loads (the map times
        // its own frames from its last one), then 60 Hz from there, so
        // fades and transitions run on.
        let epoch = clock::now();
        frame(&mut map, target, 0.0)?;
        settle(&mut map, target, &mut warnings)?;
        let base = (clock::now() - epoch).as_secs_f64() * 1000.0;
        let mut legs = if path.flight {
            flight_legs()
        } else {
            Vec::new()
        };
        legs.reverse();
        progress(&format!("{}: moving", path.name));
        let (mut cpu, mut peak_bytes, mut loading) = (Vec::new(), 0, 0);
        let mut drape = (0, 0, 0);
        let mut atlas_was = None;
        let mut placement = Vec::new();
        // Per label text: shown in full, and since which frame it is gone.
        let mut labels: std::collections::HashMap<Box<str>, Option<usize>> = Default::default();
        let (mut blinks, mut blink_frames, mut returns, mut drops) = (0, 0, 0, 0);
        let (mut rebuilds, mut rows) = (0, 0);
        // GPU times, collected after the path: a collection waits for the
        // GPU.
        runtime.set_gpu_timing(true);
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
            let before = uploaded(runtime);
            // The target keeps two frames in flight, as a swapchain does.
            let (report, time) = frame(&mut map, target, now)?;
            cpu.push(time);
            placement.push(Duration::from_micros(u64::from(report.times.placement)));
            peak_bytes = peak_bytes.max(uploaded(runtime) - before);
            let atlas = report.memory.drape_atlas;
            let resized = atlas_was.is_some_and(|was| was != atlas);
            drape = (
                drape.0.max(atlas),
                drape.1 + atlas,
                drape.2 + u32::from(resized),
            );
            atlas_was = Some(atlas);
            loading += usize::from(report.tiles_loading > 0);
            rebuilds += usize::from(report.frame.candidates_rebuilt);
            rows += u64::from(report.frame.placements_written);
            let shown = map.shown_labels();
            for (name, dim) in &mut labels {
                let opacity = shown.get(name).copied().unwrap_or(0.0);
                match *dim {
                    None if opacity < 0.1 => {
                        *dim = Some(f as usize);
                        drops += 1;
                    }
                    Some(since) if opacity >= 0.9 => {
                        let gone = f as usize - since;
                        if gone <= FLASH_FRAMES {
                            blinks += 1;
                            blink_frames += gone;
                        }
                        returns += usize::from(gone <= RETURN_FRAMES);
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
            std::thread::sleep(next.saturating_sub(clock::now()));
        }
        // Submissions without timed passes (a page's growth copies) took no
        // measured GPU time, so they aren't samples.
        let mut gpu: Vec<_> = (runtime.gpu_timings().into_iter())
            .filter(|s| !s.passes.is_empty())
            .map(|s| s.gpu())
            .collect();
        runtime.set_gpu_timing(false);
        gpu.sort_unstable();
        let ended = clock::now();
        loop {
            now += FRAME.as_secs_f64() * 1000.0;
            let (report, _) = frame(&mut map, target, now)?;
            warnings.extend(crate::take_warnings(&mut map));
            // A map that doesn't settle in time is a result: reported as
            // the timeout passed.
            if report.idle || clock::now() - ended > SETTLE_TIMEOUT {
                break;
            }
            std::thread::sleep(FRAME);
        }
        cpu.sort_unstable();
        placement.sort_unstable();
        paths.push(PathResult {
            name: path.name,
            cpu,
            gpu,
            placement,
            peak_bytes,
            drape,
            loading,
            rebuilds,
            rows,
            blinks,
            blink_frames,
            returns,
            drops,
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
            "| path | render CPU p50 | p95 | p99 | max | frames > 16.7 ms | GPU p50 | p99 | max | frames loading | candidate rebuilds | rows written per frame | peak MiB/frame | drape atlas MiB max (mean) | drape resizes | settle ms | label blinks (frames) | label returns (drops) | placement p50 | p95 | max |"
        )?;
        writeln!(
            out,
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        )?;
        for p in &self.paths {
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {:.1} | {:.0} ({:.0}) | {} | {} | {} ({}) | {} ({}) | {} | {} | {} |",
                p.name,
                ms(p.at(0.5)),
                ms(p.at(0.95)),
                ms(p.at(0.99)),
                ms(p.at(1.0)),
                p.cpu.iter().filter(|&&t| t > FRAME).count(),
                ms(percentile(&p.gpu, 0.5)),
                ms(percentile(&p.gpu, 0.99)),
                ms(percentile(&p.gpu, 1.0)),
                p.loading,
                p.rebuilds,
                p.rows as f64 / p.cpu.len().max(1) as f64,
                p.peak_bytes as f64 / (1024.0 * 1024.0),
                p.drape.0 as f64 / (1024.0 * 1024.0),
                p.drape.1 as f64 / (1024.0 * 1024.0) / p.cpu.len().max(1) as f64,
                p.drape.2,
                if p.settle > SETTLE_TIMEOUT {
                    format!("> {}", SETTLE_TIMEOUT.as_millis())
                } else {
                    format!("{:.0}", p.settle.as_secs_f64() * 1000.0)
                },
                p.blinks,
                p.blink_frames,
                p.returns,
                p.drops,
                ms(percentile(&p.placement, 0.5)),
                ms(percentile(&p.placement, 0.95)),
                ms(percentile(&p.placement, 1.0)),
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Times in ms of `Map::render` (applying arrived tiles, uploads, placement, encoding); \
             GPU: each submission's, from its first pass's start to its last one's end, where \
             the device has timestamp queries; \
             settle: from the \
             path's end until the map is idle; candidate rebuilds: frames that laid the cull's \
             candidates out again; label blinks: labels shown in full that all but \
             vanished (opacity under 0.1) and were back in full within {FLASH_FRAMES} frames (no \
             fade is that quick), and the frames they were gone; label returns: labels \
             that all but vanished and were back in full within {RETURN_FRAMES} frames (1 s), \
             blinks included, of the labels shown in full that all but vanished; placement: the render thread's \
             share of it (building the frame the lane places, applying commits; 0.2 ms)."
        )?;
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
        }
        Ok(())
    }
}
