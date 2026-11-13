//! The `quake` experiment: the Alps style with its DEM sources made one
//! that the bench feeds heights (`Map::set_host_elevations`), as the
//! demo's Earthquake does, its ground held still and then rippling from an
//! epicentre while the camera turns at a 60 Hz pace: CPU and GPU time a
//! frame, the steps of heights shown a second, the bytes a frame uploads,
//! and what memory the frames made. The heights are synthetic hills over
//! the Valais (the demo's are Mapterhorn's), 256² a tile, and the wave is
//! the demo's.

use std::fmt::Write;
use std::time::Duration;

use orb_api::{Event, Map, MapEvent, Runtime, Target, TileIndex};

use crate::report::{Summary, ms};
use crate::{Settings, clock, frame_counted};

const SOURCE: &str = "quake";
const SIDE: u32 = 256;
/// The demo's epicentre, the Matterhorn, and its camera.
const EPICENTRE: (f64, f64) = (7.6586, 45.9763);
const CAMERA: (f64, f64, f64, f64, f64) = (7.79, 46.06, 10.4, -28.0, 62.0);

/// One phase's frames.
#[derive(Debug)]
struct Phase {
    name: &'static str,
    frames: usize,
    cpu: Summary,
    gpu: Summary,
    shown_per_second: f64,
    /// Texel and buffer bytes uploaded a frame.
    upload: (u64, u64),
    bytes_allocated: u64,
    textures_created: u32,
    buffers_created: u32,
    tiles: usize,
}

#[derive(Debug)]
pub struct QuakeReport {
    about: Vec<String>,
    phases: Vec<Phase>,
    warnings: Vec<String>,
}

/// The Alps style with one host-fed DEM source for its terrain and hillshade.
fn style(alps: &str) -> Result<String, String> {
    let mut style: serde_json::Value = serde_json::from_str(alps).map_err(|e| e.to_string())?;
    let dem: Vec<String> = (style["sources"].as_object().into_iter().flatten())
        .filter(|(_, s)| s["type"] == "raster-dem")
        .map(|(id, _)| id.clone())
        .collect();
    let sources = style["sources"].as_object_mut().ok_or("no sources")?;
    sources.retain(|id, _| !dem.contains(id));
    sources.insert(
        SOURCE.into(),
        serde_json::json!({"type": "raster-dem", "tiles": ["orbis-host:"],
            "encoding": "terrarium", "tileSize": 512, "maxzoom": 11}),
    );
    for layer in style["layers"].as_array_mut().into_iter().flatten() {
        if layer["source"]
            .as_str()
            .is_some_and(|s| dem.iter().any(|d| d == s))
        {
            layer["source"] = SOURCE.into();
        }
    }
    style["terrain"] = serde_json::json!({"source": SOURCE, "exaggeration": 1.6});
    Ok(style.to_string())
}

/// Per texel of `tile`, its longitude and latitude.
fn texels(tile: TileIndex) -> impl Iterator<Item = (f64, f64)> {
    let (n, side) = (f64::from(1u32 << tile.z), f64::from(SIDE));
    (0..SIDE * SIDE).map(move |i| {
        let x = (f64::from(tile.col) + (f64::from(i % SIDE) + 0.5) / side) / n;
        let y = (f64::from(tile.row) + (f64::from(i / SIDE) + 0.5) / side) / n;
        let lat = (std::f64::consts::PI * (1.0 - 2.0 * y))
            .sinh()
            .atan()
            .to_degrees();
        (x * 360.0 - 180.0, lat)
    })
}

/// A tile's hills, and per texel its distance from the epicentre in metres.
struct Ground {
    base: Vec<f32>,
    metres: Vec<f32>,
}

impl Ground {
    fn of(tile: TileIndex) -> Self {
        let (base, metres) = texels(tile)
            .map(|(lon, lat)| {
                let hills = 1800.0
                    + 1200.0 * (lon * 30.0).sin() * (lat * 40.0).cos()
                    + 300.0 * (lon * 110.0 + lat * 70.0).sin();
                let dx = (lon - EPICENTRE.0) * 111_320.0 * EPICENTRE.1.to_radians().cos();
                let dy = (lat - EPICENTRE.1) * 110_540.0;
                (hills as f32, dx.hypot(dy) as f32)
            })
            .unzip();
        Self { base, metres }
    }

    /// Its heights `t` seconds after the shock: the demo's wave
    /// (`web/demo/app/quake-worker.js`), straight rather than tabled.
    fn at(&self, t: f64) -> Vec<f32> {
        (self.base.iter().zip(&self.metres))
            .map(|(&b, &d)| {
                let s = t - f64::from(d) / 3000.0;
                if s <= 0.0 {
                    return b;
                }
                let swing = 900.0 / (1.0 + f64::from(d) / 15_000.0).sqrt();
                let wave = swing * (-s / 4.0).exp() * (std::f64::consts::TAU * 0.5 * s).sin();
                b + wave as f32
            })
            .collect()
    }
}

pub(crate) fn run(
    runtime: &Runtime,
    settings: &Settings,
    mut about: Vec<String>,
    alps: &std::path::Path,
    mut progress: impl FnMut(&str),
) -> Result<QuakeReport, String> {
    if !runtime.stats().gpu_timestamps {
        return Err("the device has no timestamp queries".into());
    }
    let text = std::fs::read_to_string(alps).map_err(|e| e.to_string())?;
    let (w, h) = (settings.width, settings.height);
    about.push(format!(
        "{w}×{h} @{}, synthetic hills over the Valais, {SIDE}² heights a tile",
        settings.ratio
    ));
    let mut target = Target::texture(runtime, w, h, f64::from(settings.ratio));
    let (lon, lat, zoom, bearing, pitch) = CAMERA;
    let target_camera = orb_api::CameraTarget {
        center: Some(orb_api::Geodetic::new(lon, lat)),
        zoom: Some(zoom),
        bearing: Some(bearing),
        pitch: Some(pitch),
        ..Default::default()
    };
    let mut map = Map::new(
        runtime,
        orb_api::MapOptions {
            previews: orb_api::Previews::ALL,
            ..Default::default()
        },
    );
    map.set_style_json(&style(&text)?)
        .map_err(|e| e.to_string())?;
    map.jump_to(&target_camera);
    let epoch = clock::now();
    let now = || (clock::now() - epoch).as_secs_f64() * 1000.0;
    let mut grounds: std::collections::BTreeMap<TileIndex, Ground> = Default::default();
    let mut warnings = Vec::new();
    // Answers the map's tile requests with still ground; true once a
    // batch it waited for showed.
    let mut answer = |map: &mut Map, grounds: &mut std::collections::BTreeMap<_, _>| {
        let mut shown = false;
        let mut asked = Vec::new();
        while let Some(event) = map.poll_event() {
            match event {
                Event::Map(MapEvent::TileRequest { source, tile }) if source == SOURCE => {
                    asked.push(tile);
                }
                Event::Map(MapEvent::ElevationsShown { .. }) => shown = true,
                Event::Map(MapEvent::Error { message } | MapEvent::Warning { message }) => {
                    warnings.push(message);
                }
                _ => {}
            }
        }
        if !asked.is_empty() {
            let tiles = (asked.iter())
                .map(|&t| {
                    let ground = Ground::of(t);
                    let heights = ground.base.clone();
                    grounds.insert(t, ground);
                    (t, heights)
                })
                .collect();
            if let Ok(done) = map.set_host_elevations(SOURCE, SIDE, tiles) {
                for t in done.dropped {
                    grounds.remove(&t);
                }
            }
        }
        shown
    };
    progress("loading");
    let deadline = clock::now() + Duration::from_secs(120);
    let mut idle = 0;
    while idle < 2 {
        let report = map
            .render(&mut target, now())
            .map_err(|e| e.to_string())?
            .ok_or("a texture target always draws")?;
        answer(&mut map, &mut grounds);
        idle = if report.idle { idle + 1 } else { 0 };
        if clock::now() > deadline {
            return Err("the map didn't settle".into());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let mut phases = Vec::new();
    for (name, moving) in [("still", false), ("quake", true)] {
        progress(name);
        runtime.set_gpu_timing(true);
        let count = settings.warmup + settings.frames;
        let (mut cpu, mut reports) = (Vec::new(), Vec::new());
        let mut in_flight = false;
        let (mut shown, started) = (0u32, clock::now());
        let frame = Duration::from_secs_f64(1.0 / 60.0);
        for i in 0..count {
            let tick = clock::now();
            if moving && !in_flight {
                let t = (clock::now() - started).as_secs_f64() % 12.0;
                let tiles = (grounds.iter()).map(|(&t_, g)| (t_, g.at(t))).collect();
                let done = map
                    .set_host_elevations(SOURCE, SIDE, tiles)
                    .map_err(|e| e.to_string())?;
                for t in done.dropped {
                    grounds.remove(&t);
                }
                in_flight = true;
            }
            map.jump_to(&orb_api::CameraTarget {
                bearing: Some(bearing + 15.0 * i as f64 / count as f64),
                ..target_camera
            });
            let (report, time, _) = frame_counted(&mut map, &mut target, now())?;
            if answer(&mut map, &mut grounds) {
                in_flight = false;
                shown += u32::from(i >= settings.warmup);
            }
            if i >= settings.warmup {
                cpu.push(time);
                reports.push(report);
            }
            if let Some(rest) = frame.checked_sub(clock::now() - tick) {
                std::thread::sleep(rest);
            }
        }
        let seconds =
            (clock::now() - started).as_secs_f64() * settings.frames as f64 / count as f64;
        let mut gpu: Vec<Duration> = (runtime.gpu_timings().into_iter())
            .filter(|t| t.draws)
            .skip(settings.warmup)
            .map(|t| t.gpu())
            .collect();
        runtime.set_gpu_timing(false);
        let n = reports.len().max(1) as u64;
        let sum = |f: &dyn Fn(&orb_api::FrameReport) -> u64| reports.iter().map(f).sum::<u64>();
        phases.push(Phase {
            name,
            frames: reports.len(),
            cpu: Summary::of(&mut cpu),
            gpu: Summary::of(&mut gpu),
            shown_per_second: f64::from(shown) / seconds,
            upload: (
                sum(&|r| r.gpu.texel_bytes_from_core + r.gpu.texel_bytes_from_host) / n,
                sum(&|r| r.gpu.buffer_bytes_from_core + r.gpu.buffer_bytes_from_host) / n,
            ),
            bytes_allocated: sum(&|r| r.gpu.bytes_allocated),
            textures_created: reports.iter().map(|r| r.gpu.textures_created).sum(),
            buffers_created: reports.iter().map(|r| r.gpu.buffers_created).sum(),
            tiles: grounds.len(),
        });
    }
    Ok(QuakeReport {
        about,
        phases,
        warnings,
    })
}

impl QuakeReport {
    pub(crate) fn write(&self, out: &mut String) -> std::fmt::Result {
        writeln!(out, "# Earthquake: host-fed heights\n")?;
        for line in &self.about {
            writeln!(out, "- {line}")?;
        }
        writeln!(
            out,
            "\n| phase | frames | CPU median / p90 | GPU median / p90 | steps shown / s | texel KiB / frame | buffer KiB / frame | bytes allocated | textures, buffers made | tiles fed |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|---|")?;
        for p in &self.phases {
            writeln!(
                out,
                "| {} | {} | {} / {} | {} / {} | {:.1} | {} | {} | {} | {}, {} | {} |",
                p.name,
                p.frames,
                ms(p.cpu.median),
                ms(p.cpu.p90),
                ms(p.gpu.median),
                ms(p.gpu.p90),
                p.shown_per_second,
                p.upload.0 / 1024,
                p.upload.1 / 1024,
                p.bytes_allocated,
                p.textures_created,
                p.buffers_created,
                p.tiles,
            )?;
        }
        for warning in &self.warnings {
            writeln!(out, "\nwarning: {warning}")?;
        }
        Ok(())
    }
}
