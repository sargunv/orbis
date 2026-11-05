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
use crate::{GesturePaths, Settings, Timing, clock, paint::settle};

/// A 60 Hz display's frame.
const FRAME: Duration = Duration::from_micros(16_667);
/// Frames per path: 6 s at 60 Hz.
const PATH_FRAMES: usize = 360;
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
    let center = orb_geo::Geodetic::new(s.center.lon + lon, s.center.lat);
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
                    center: Some(orb_geo::Geodetic::new(
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
    /// Bytes uploaded in the frame that uploaded most.
    peak_bytes: u64,
    /// Frames that drew with tiles still loading.
    loading: usize,
    /// From the path's end until the map is idle.
    settle: Duration,
    picking: Picking,
    /// The terrain's elevation pyramid, published to a UI thread's camera
    /// whenever it changed over the path (the web's ui role and placement
    /// worker take it, decision 0019).
    elevation: Publishes,
}

/// What publishing picking to a UI thread's index costs (decision 0024):
/// the render thread's `Map::publish_hits` and the UI's `HitIndex::apply`
/// after each frame, and a first hit test at rest.
#[derive(Debug, Default)]
struct Picking {
    /// Over the path: publications, their total and largest time and bytes.
    moving: Publishes,
    /// From the path's end until idle.
    settling: Publishes,
    /// Hit tests at the centre at rest, through the UI's index: the first
    /// and the next.
    hit_test: Duration,
    hit_test_again: Duration,
}

#[derive(Debug, Default)]
struct Publishes {
    count: usize,
    publish: Duration,
    publish_max: Duration,
    apply: Duration,
    apply_max: Duration,
    bytes: usize,
}

impl Publishes {
    fn publish(
        &mut self,
        map: &Map,
        to: &mut orb_runtime::HitPublication,
        index: &mut orb_runtime::HitIndex,
    ) {
        let started = clock::now();
        let Some(bytes) = map.publish_hits(to) else {
            return;
        };
        let published = clock::now();
        index.apply(&bytes).expect("a publication applies");
        let applied = clock::now();
        self.count += 1;
        self.publish += published - started;
        self.publish_max = self.publish_max.max(published - started);
        self.apply += applied - published;
        self.apply_max = self.apply_max.max(applied - published);
        self.bytes += bytes.len();
    }
}

impl Publishes {
    /// Encodes the elevation pyramid on the render thread and decodes it as
    /// a UI thread does, when its revision moved past `seen`.
    fn publish_elevation(&mut self, map: &Map, seen: &mut u64) {
        let revision = map.elevation_revision();
        if revision == std::mem::replace(seen, revision) {
            return;
        }
        let started = clock::now();
        let bytes = map.elevation_pyramid();
        let published = clock::now();
        if !bytes.is_empty() {
            orb_runtime::TerrainElevation::from_pyramid(&bytes).expect("a pyramid decodes");
        }
        let applied = clock::now();
        self.count += 1;
        self.publish += published - started;
        self.publish_max = self.publish_max.max(published - started);
        self.apply += applied - published;
        self.apply_max = self.apply_max.max(applied - published);
        self.bytes += bytes.len();
    }

    /// The row's cells: count, publish total / max, apply total / max, MiB.
    fn cells(&self) -> String {
        format!(
            "{} | {} / {} | {} / {} | {:.2}",
            self.count,
            ms(self.publish),
            ms(self.publish_max),
            ms(self.apply),
            ms(self.apply_max),
            self.bytes as f64 / (1024.0 * 1024.0)
        )
    }
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
    let paths_run = match settings.paths {
        GesturePaths::City => CITY_PATHS,
        GesturePaths::Alps => ALPS_PATHS,
    };
    let projection = crate::web_mercator();
    for path in paths_run {
        // A map per path, so each starts cold.
        progress(&format!("{}: loading its first view", path.name));
        let start = (path.camera)(settings, 0.0);
        let mut map = new_map(start)?;
        settle(gpu, &mut map, &render, &mut warnings).await?;
        let mut to = orb_runtime::HitPublication::default();
        let mut index = orb_runtime::HitIndex::default();
        let mut picking = Picking::default();
        let mut elevation = Publishes::default();
        let mut revision = map.elevation_revision();
        Publishes::default().publish(&map, &mut to, &mut index);
        // A flight's camera stands on the terrain, as the demos' do.
        let mut camera = orb_camera::Camera::new(start, projection, viewport);
        camera.set_terrain(map.elevation());
        let mut legs = if path.flight {
            flight_legs()
        } else {
            Vec::new()
        };
        legs.reverse();
        progress(&format!("{}: moving", path.name));
        let (mut cpu, mut peak_bytes, mut loading) = (Vec::new(), 0, 0);
        let started = clock::now();
        for f in 1.. {
            let now = (FRAME * f).as_secs_f64() * 1000.0;
            if path.flight {
                if !camera.is_animating() {
                    let Some((target, options)) = legs.pop() else {
                        break;
                    };
                    camera.fly_to(&target, &options, now);
                }
                camera.frame(now);
                camera.take_events();
                *map.camera_mut() = *camera.state();
            } else if f as usize > PATH_FRAMES {
                break;
            } else {
                *map.camera_mut() = (path.camera)(settings, f64::from(f) / PATH_FRAMES as f64);
            }
            let before = uploaded(gpu);
            let (report, time) = render(gpu, &mut map);
            match timing {
                Timing::Timestamps => gpu.throttle().await,
                Timing::WorkDone => gpu.finish().await,
            }
            // A flight's thousands of frames outrun the timestamp queries;
            // this bench reads CPU times only.
            if f % 64 == 0 {
                gpu.take_frames().await;
            }
            cpu.push(time);
            picking.moving.publish(&map, &mut to, &mut index);
            if map.elevation_revision() != revision {
                camera.set_terrain(map.elevation());
            }
            elevation.publish_elevation(&map, &mut revision);
            peak_bytes = peak_bytes.max(uploaded(gpu) - before);
            loading += usize::from(report.tiles_loading > 0);
            warnings.extend(map.take_warnings());
            let next = started + FRAME * f;
            clock::sleep(next.saturating_sub(clock::now())).await;
        }
        let ended = clock::now();
        loop {
            let (report, _) = render(gpu, &mut map);
            picking.settling.publish(&map, &mut to, &mut index);
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
        let center = glam::DVec2::new(viewport.width / 2.0, viewport.height / 2.0);
        let started = clock::now();
        let _ = index.hit_test(center, map.camera(), viewport, &map.projection());
        picking.hit_test = clock::now() - started;
        let started = clock::now();
        let _ = index.hit_test(center, map.camera(), viewport, &map.projection());
        picking.hit_test_again = clock::now() - started;
        cpu.sort_unstable();
        paths.push(PathResult {
            name: path.name,
            cpu,
            peak_bytes,
            loading,
            settle: clock::now() - ended,
            picking,
            elevation,
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
            "Times in ms of `Map::render` (applying arrived tiles, uploads, placement, encoding); \
             settle: from the path's end until the map is idle."
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| path | picking while moving: publications | publish total / max | apply total / max | MiB | \
             while settling: publications | publish total / max | apply total / max | MiB | hit tests at rest: first / next |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|---|")?;
        for p in &self.paths {
            writeln!(
                out,
                "| {} | {} | {} | {} / {} |",
                p.name,
                p.picking.moving.cells(),
                p.picking.settling.cells(),
                ms(p.picking.hit_test),
                ms(p.picking.hit_test_again)
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Picking: `Map::publish_hits` on the render thread and `HitIndex::apply` on the UI's after each \
             frame (a publication carries what changed: tiles, the drawn list, labels, the layer table)."
        )?;
        if self.paths.iter().any(|p| p.elevation.count > 0) {
            writeln!(out)?;
            writeln!(
                out,
                "| path | elevation pyramid while moving: publications | encode total / max | decode total / max | MiB |"
            )?;
            writeln!(out, "|---|---|---|---|---|")?;
            for p in &self.paths {
                writeln!(out, "| {} | {} |", p.name, p.elevation.cells())?;
            }
            writeln!(out)?;
            writeln!(
                out,
                "Elevation: `Map::elevation_pyramid` on the render thread and `TerrainElevation::from_pyramid` \
                 on the UI's, at every revision (the web publishes at most every 30 frames while moving; \
                 its placement worker takes one at each new snapshot)."
            )?;
        }
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
        }
        Ok(())
    }
}
