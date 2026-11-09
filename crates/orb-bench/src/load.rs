//! The load bench: for each view, a fresh map per sample from loading its
//! style to its first complete still (`Map::snapshot`: Still mode, every
//! tile and label of the view, read back), as `orbis render` draws one.
//!
//! Cold samples make a new runtime first (outside the time), so the map
//! compiles its pipelines and starts its job threads' work from nothing;
//! warm samples share one runtime that drew the view before, as a second
//! map in an app does. The two interleave, so drift hits both alike. The
//! process and the driver's own shader caches stay warm across runtimes,
//! so a first start from nothing is `orbis render` in a new process
//! (`cargo xtask h2h stills`).

use std::fmt::Write;
use std::time::Duration;

use orb_api::{CameraState, Runtime, RuntimeOptions};

use crate::report::{Summary, ms};
use crate::{Settings, ViewSize, clock, new_map, take_warnings};

/// Every view's loads.
#[derive(Debug)]
pub struct LoadReport {
    about: Vec<String>,
    settings: Settings,
    views: Vec<LoadView>,
    warnings: Vec<String>,
}

#[derive(Debug)]
struct LoadView {
    name: String,
    size: (u32, u32),
    cold: Loads,
    warm: Loads,
}

/// One kind of sample of a view.
#[derive(Debug, Default)]
struct Loads {
    wall: Vec<Duration>,
    /// The process's CPU time, every thread's.
    cpu: Vec<Duration>,
    /// Pipelines the device made while loading.
    pipelines: Vec<u32>,
    /// Making the runtime, for cold samples (not in `wall`).
    setup: Vec<Duration>,
}

pub(crate) fn run(
    runtime: &Runtime,
    settings: &Settings,
    about: Vec<String>,
    style: &str,
    mut progress: impl FnMut(&str),
) -> Result<LoadReport, String> {
    let mut views = Vec::new();
    let mut warnings = Vec::new();
    for (view, camera, size) in settings.cameras() {
        progress(&format!("{}: warming the shared runtime", view.name));
        load(runtime, style, camera, size, &mut warnings)?;
        let (mut cold, mut warm) = (Loads::default(), Loads::default());
        for sample in 0..settings.samples {
            progress(&format!(
                "{}: sample {} of {}",
                view.name,
                sample + 1,
                settings.samples
            ));
            let started = clock::now();
            let fresh = Runtime::new(&RuntimeOptions::default()).map_err(|e| e.to_string())?;
            cold.setup.push(clock::now() - started);
            cold.push(load(&fresh, style, camera, size, &mut warnings)?);
            drop(fresh);
            warm.push(load(runtime, style, camera, size, &mut warnings)?);
        }
        let name = if view.buildings {
            format!("{} (without them)", view.name)
        } else {
            view.name
        };
        views.push(LoadView {
            name,
            size: size.physical(),
            cold,
            warm,
        });
    }
    warnings.sort();
    warnings.dedup();
    Ok(LoadReport {
        about,
        settings: settings.clone(),
        views,
        warnings,
    })
}

impl Loads {
    fn push(&mut self, (wall, cpu, pipelines): (Duration, Option<Duration>, u32)) {
        self.wall.push(wall);
        self.cpu.extend(cpu);
        self.pipelines.push(pipelines);
    }
}

/// One map from `Map::new` to its first complete still: wall time, CPU
/// time and pipelines made.
fn load(
    runtime: &Runtime,
    style: &str,
    camera: CameraState,
    size: ViewSize,
    warnings: &mut Vec<String>,
) -> Result<(Duration, Option<Duration>, u32), String> {
    let pipelines = runtime.stats().pipelines_created;
    let cpu = clock::cpu();
    let started = clock::now();
    let mut map = new_map(runtime, style, camera);
    map.snapshot(size.width, size.height, f64::from(size.ratio))
        .map_err(|e| format!("loading: {e}"))?;
    let wall = clock::now() - started;
    let cpu = clock::cpu().zip(cpu).map(|(after, before)| after - before);
    warnings.extend(take_warnings(&mut map));
    Ok((wall, cpu, runtime.stats().pipelines_created - pipelines))
}

impl LoadReport {
    pub(crate) fn write(&self, out: &mut String) -> std::fmt::Result {
        writeln!(out, "### Load bench")?;
        writeln!(out)?;
        for line in &self.about {
            writeln!(out, "- {line}")?;
        }
        writeln!(
            out,
            "- a fresh map per sample from `Map::new` to its first complete still, read back; \
             {} samples a view; cold: a new runtime (made outside the time), warm: a shared \
             runtime that drew the view before; CPU is the whole process's; new runtime: \
             making the cold sample's device and threads, outside its time",
            self.settings.samples,
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| view | px | cold wall median | min–max | cold CPU median | pipelines made | new runtime | warm wall median | min–max | warm CPU median |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|---|")?;
        let (mut cold_total, mut warm_total) = (Duration::ZERO, Duration::ZERO);
        for v in &self.views {
            let cold = Summary::of(&mut v.cold.wall.clone());
            let warm = Summary::of(&mut v.warm.wall.clone());
            cold_total += cold.median;
            warm_total += warm.median;
            let cpu = |l: &Loads| {
                if l.cpu.is_empty() {
                    "—".to_string()
                } else {
                    ms(Summary::of(&mut l.cpu.clone()).median)
                }
            };
            let range = |w: &[Duration]| {
                let (min, max) = (w.iter().min(), w.iter().max());
                format!(
                    "{}–{}",
                    ms(*min.unwrap_or(&Duration::ZERO)),
                    ms(*max.unwrap_or(&Duration::ZERO))
                )
            };
            let mut pipelines = v.cold.pipelines.clone();
            pipelines.sort_unstable();
            writeln!(
                out,
                "| {} | {}×{} | {} | {} | {} | {} | {} | {} | {} | {} |",
                v.name,
                v.size.0,
                v.size.1,
                ms(cold.median),
                range(&v.cold.wall),
                cpu(&v.cold),
                pipelines.get(pipelines.len() / 2).copied().unwrap_or(0),
                ms(Summary::of(&mut v.cold.setup.clone()).median),
                ms(warm.median),
                range(&v.warm.wall),
                cpu(&v.warm),
            )?;
        }
        writeln!(
            out,
            "| **sum of medians** | | **{}** | | | | | **{}** | | |",
            ms(cold_total),
            ms(warm_total)
        )?;
        writeln!(out)?;
        writeln!(out, "Times in ms.")?;
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
        }
        Ok(())
    }
}
