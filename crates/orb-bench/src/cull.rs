//! The `cull` experiment: at each view, what the GPU cull keeps of the
//! candidates the frame lays out, by draw kind, and what keeping them all
//! (`cull.wgsl`'s `keep_all`, as a driver without the cull draws) costs
//! the GPU, with the pixels of both compared.

use std::fmt::Write;
use std::time::Duration;

use orb_api::{Map, Runtime, Target};

use crate::report::Summary;
use crate::{Settings, clock, frame, prepare_view};

/// One view, culled and kept whole.
#[derive(Debug)]
pub(crate) struct CullView {
    name: String,
    /// Per draw kind (meshes, points, images): candidates, and those the
    /// cull kept.
    kinds: [(u64, u64); 3],
    /// GPU time per frame, and the main pass's, culled then kept whole.
    gpu: [Summary; 2],
    main: [Summary; 2],
    /// Pixels that differ between the two.
    differing: usize,
    pixels: usize,
}

#[derive(Debug)]
pub struct CullReport {
    about: Vec<String>,
    frames: usize,
    views: Vec<CullView>,
    warnings: Vec<String>,
}

/// The draw kind of a family as `orb-render` names it: meshes (0),
/// points (1) or images (2).
fn kind(family: &str) -> usize {
    match family {
        "Fill" | "FillOutline" | "Line" | "FillExtrusion" => 0,
        "Raster" | "Hillshade" | "ColorRelief" => 2,
        _ => 1,
    }
}

pub(crate) fn run(
    runtime: &Runtime,
    settings: &Settings,
    about: Vec<String>,
    style: &str,
    mut progress: impl FnMut(&str),
) -> Result<CullReport, String> {
    if !runtime.stats().gpu_timestamps {
        return Err("the device has no timestamp queries".into());
    }
    let epoch = clock::now();
    let mut warnings = Vec::new();
    let mut views = Vec::new();
    let mut map: Option<Map> = None;
    let mut target = None;
    for (view, camera, size) in settings.cameras() {
        progress(&format!("{}: loading", view.name));
        let (map, target) = prepare_view(
            runtime,
            (style, settings.antialias),
            (&mut map, &mut target),
            (&view, camera, size),
            &mut warnings,
        )?;
        let mut sample = |map: &mut Map, target: &mut Target| -> Result<_, String> {
            crate::settle(map, target, &mut warnings)?;
            runtime.set_gpu_timing(true);
            let count = settings.warmup + settings.frames;
            for _ in 0..count {
                let time = (clock::now() - epoch).as_secs_f64() * 1000.0;
                frame(map, target, time)?;
            }
            let times = runtime.gpu_timings();
            runtime.set_gpu_timing(false);
            let (mut gpu, mut main): (Vec<Duration>, Vec<Duration>) = (Vec::new(), Vec::new());
            for t in times.iter().skip(settings.warmup) {
                gpu.push(t.gpu());
                main.extend(t.pass("main"));
            }
            let pixels = target.read_pixels().map_err(|e| e.to_string())?;
            Ok((
                Summary::of(&mut gpu),
                Summary::of(&mut main),
                pixels,
                map.cull_counts(),
            ))
        };
        progress(&format!("{}: culled", view.name));
        let (gpu_culled, main_culled, culled, counts) = sample(map, target)?;
        progress(&format!("{}: kept whole", view.name));
        map.set_keep_all_cull(true);
        let (gpu_kept, main_kept, kept, all) = sample(map, target)?;
        map.set_keep_all_cull(false);
        let short: Vec<String> = (all.iter().enumerate())
            .filter(|(_, c)| c.kept != c.candidates)
            .map(|(i, c)| format!("node {i} {} {}/{}", c.family, c.kept, c.candidates))
            .collect();
        if !short.is_empty() {
            warnings.push(format!(
                "{}: keep-all kept fewer: {}",
                view.name,
                short.join(", ")
            ));
        }
        let mut kinds = [(0u64, 0u64); 3];
        for c in &counts {
            let k = &mut kinds[kind(&c.family)];
            k.0 += u64::from(c.candidates);
            k.1 += u64::from(c.kept);
        }
        let differing = (culled.as_chunks::<4>().0.iter())
            .zip(kept.as_chunks::<4>().0)
            .filter(|(a, b)| a != b)
            .count();
        views.push(CullView {
            name: view.name.clone(),
            kinds,
            gpu: [gpu_culled, gpu_kept],
            main: [main_culled, main_kept],
            differing,
            pixels: culled.len() / 4,
        });
    }
    Ok(CullReport {
        about,
        frames: settings.frames,
        views,
        warnings,
    })
}

impl CullReport {
    pub(crate) fn write(&self, out: &mut String) -> std::fmt::Result {
        writeln!(
            out,
            "# Cull: kept against candidates, and keeping them all\n"
        )?;
        for line in &self.about {
            writeln!(out, "- {line}")?;
        }
        writeln!(
            out,
            "- {} timed frames per arm; GPU medians in ms; candidates are (chunk, tile placement) pairs\n",
            self.frames
        )?;
        writeln!(
            out,
            "| View | Meshes kept | Points kept | Images kept | All kept | GPU culled | GPU keep-all | Main culled | Main keep-all | Pixels differing |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|---|")?;
        let ms = |d: Duration| format!("{:.3}", d.as_secs_f64() * 1000.0);
        let share = |(all, kept): (u64, u64)| {
            if all == 0 {
                "–".to_string()
            } else {
                format!("{kept} / {all} ({:.0}%)", kept as f64 * 100.0 / all as f64)
            }
        };
        for v in &self.views {
            let total = v
                .kinds
                .iter()
                .fold((0, 0), |(a, k), &(a2, k2)| (a + a2, k + k2));
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} of {} |",
                v.name,
                share(v.kinds[0]),
                share(v.kinds[1]),
                share(v.kinds[2]),
                share(total),
                ms(v.gpu[0].median),
                ms(v.gpu[1].median),
                ms(v.main[0].median),
                ms(v.main[1].median),
                v.differing,
                v.pixels,
            )?;
        }
        for w in &self.warnings {
            writeln!(out, "\n- warning: {w}")?;
        }
        Ok(())
    }
}
