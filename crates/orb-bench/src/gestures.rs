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
use crate::{Settings, Timing, clock, paint::settle};

/// A 60 Hz display's frame.
const FRAME: Duration = Duration::from_micros(16_667);
/// Frames per path: 6 s at 60 Hz.
const PATH_FRAMES: usize = 360;
/// How long the map may take to settle after a path.
const SETTLE_TIMEOUT: Duration = Duration::from_secs(120);

/// A camera path: its name and the camera at `f` in `0..=1`.
struct Path {
    name: &'static str,
    camera: fn(&Settings, f64) -> CameraState,
}

const PATHS: &[Path] = &[
    Path {
        name: "rotate and tilt at z14",
        camera: |s, f| CameraState {
            bearing: 360.0 * f,
            pitch: 30.0 - 30.0 * (f * std::f64::consts::TAU).cos(),
            ..CameraState::new(s.center, 14.0, &crate::web_mercator())
        },
    },
    Path {
        name: "zoom z12 → z16 → z12",
        camera: |s, f| {
            let zoom = 14.0 - 2.0 * (f * std::f64::consts::TAU).cos();
            CameraState::new(s.center, zoom, &crate::web_mercator())
        },
    },
];

#[derive(Debug)]
struct PathResult {
    name: &'static str,
    /// `Map::render` per frame, sorted.
    cpu: Vec<Duration>,
    /// Bytes uploaded in the frame that uploaded most.
    peak_bytes: u64,
    /// Frames that drew with tiles still loading.
    loading: usize,
    /// From the path's end until the map is idle.
    settle: Duration,
}

impl PathResult {
    fn at(&self, p: f64) -> Duration {
        let i = ((self.cpu.len() as f64 - 1.0) * p).round() as usize;
        self.cpu.get(i).copied().unwrap_or_default()
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
    let render = |gpu: &mut TimedGpu, map: &mut Map| -> (FrameReport, Duration) {
        let started = clock::now();
        let report = map.render(gpu, target, format, viewport, settings.ratio);
        (report, clock::now() - started)
    };
    let mut warnings = Vec::new();
    let mut paths = Vec::new();
    for path in PATHS {
        // A map per path, so each starts cold.
        progress(&format!("{}: loading its first view", path.name));
        let mut map = new_map((path.camera)(settings, 0.0))?;
        settle(gpu, &mut map, &render, &mut warnings).await?;
        progress(&format!("{}: moving", path.name));
        let (mut cpu, mut peak_bytes, mut loading) = (Vec::new(), 0, 0);
        let started = clock::now();
        for f in 1..=PATH_FRAMES {
            *map.camera_mut() = (path.camera)(settings, f as f64 / PATH_FRAMES as f64);
            let before = uploaded(gpu);
            let (report, time) = render(gpu, &mut map);
            match timing {
                Timing::Timestamps => gpu.throttle().await,
                Timing::WorkDone => gpu.finish().await,
            }
            cpu.push(time);
            peak_bytes = peak_bytes.max(uploaded(gpu) - before);
            loading += usize::from(report.tiles_loading > 0);
            warnings.extend(map.take_warnings());
            let next = started + FRAME * f as u32;
            clock::sleep(next.saturating_sub(clock::now())).await;
        }
        let ended = clock::now();
        loop {
            let (report, _) = render(gpu, &mut map);
            gpu.take_frames().await;
            warnings.extend(map.take_warnings());
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
        paths.push(PathResult {
            name: path.name,
            cpu,
            peak_bytes,
            loading,
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
            "- {}×{} px offscreen at {:.4}, {:.4}; {PATH_FRAMES} frames a path at 60 Hz, \
             each path from a cold map at rest; CPU clock resolution {}",
            self.size.0,
            self.size.1,
            s.center.lon,
            s.center.lat,
            resolution(Some(self.cpu_resolution)),
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| path | render CPU p50 | p95 | p99 | max | frames > 16.7 ms | frames loading | peak MiB/frame | settle ms |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|")?;
        for p in &self.paths {
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {:.1} | {:.0} |",
                p.name,
                ms(p.at(0.5)),
                ms(p.at(0.95)),
                ms(p.at(0.99)),
                ms(p.at(1.0)),
                p.cpu.iter().filter(|&&t| t > FRAME).count(),
                p.loading,
                p.peak_bytes as f64 / (1024.0 * 1024.0),
                p.settle.as_secs_f64() * 1000.0,
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Times in ms of `Map::render` (applying arrived tiles, uploads, placement, picking, encoding); \
             settle: from the path's end until the map is idle."
        )?;
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
        }
        Ok(())
    }
}
