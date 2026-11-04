//! The labels bench (M2's exit): what labels cost the render thread and the
//! GPU.
//!
//! - **Turning**: the camera turns a quarter degree a frame, so every frame
//!   asks for a placement. Per frame: the render thread's placement time
//!   (`FrameReport::placement_us`: building the placement frame and
//!   applying commits), `Map::render`'s CPU time and the bytes uploaded. Line
//!   labels walk their lines in the vertex shader, so turning uploads only
//!   placements and state words.
//! - **Hidden**: GPU time of the same view with the symbol layers removed,
//!   as they are, nearly all hidden (`text-padding` and `icon-padding`
//!   200), and all shown (`*-allow-overlap`). Hidden symbols drop their
//!   vertices in the vertex shader; the difference between "removed" and
//!   "nearly all hidden" is what they cost (#9).

use std::fmt::Write;
use std::time::Duration;

use orb_hal::TextureId;
use orb_runtime::{CameraState, FrameReport, Map, TargetFormat, Transaction, Viewport};
use orb_v8::LayerKind;

use crate::report::{Stat, Summary, ms};
use crate::timed::TimedGpu;
use crate::{Settings, Timing, clock, paint::settle};

/// Name, zoom and pitch.
const VIEWS: &[(&str, f64, f64)] = &[("z15", 15.0, 0.0), ("z15 pitch 60", 15.0, 60.0)];

/// Layout properties and their JSON values.
type Layout = &'static [(&'static str, &'static str)];

/// The hidden-symbol variants: a label and the layout each symbol layer
/// gets (`None` removes the layer).
const VARIANTS: &[(&str, Option<Layout>)] = &[
    ("removed", None),
    ("as styled", Some(&[])),
    (
        "nearly all hidden",
        Some(&[("text-padding", "200"), ("icon-padding", "200")]),
    ),
    (
        "all shown",
        Some(&[
            ("text-allow-overlap", "true"),
            ("icon-allow-overlap", "true"),
        ]),
    ),
];

#[derive(Debug)]
struct Turning {
    view: &'static str,
    placement: Summary,
    placement_max: Duration,
    /// Frames whose placement took over 0.2 ms (plan §6.1 step 10).
    over_budget: usize,
    cpu: Summary,
    /// Bytes uploaded per frame, mean.
    bytes: u64,
    placements_written: u32,
    paints_written: u32,
}

#[derive(Debug)]
struct Hidden {
    view: &'static str,
    variant: &'static str,
    gpu: Summary,
    main: Option<Summary>,
}

/// The turning and hidden-symbol results.
#[derive(Debug)]
pub struct LabelsReport {
    about: Vec<String>,
    size: (u32, u32),
    settings: Settings,
    timing: Timing,
    gpu_resolution: Option<Duration>,
    cpu_resolution: Duration,
    turning: Vec<Turning>,
    hidden: Vec<Hidden>,
    warnings: Vec<String>,
}

/// Edits every symbol layer: removes it, or sets `layout` on it.
fn edit_symbols(map: &mut Map, layout: Option<&[(&str, &str)]>) -> Result<(), String> {
    let doc = map.style_doc().ok_or("no style")?;
    let commands: Vec<String> = doc
        .style()
        .layers
        .iter()
        .filter(|l| matches!(l.kind, LayerKind::Symbol(_)))
        .flat_map(|l| {
            let id = l.id.replace('\\', "\\\\").replace('"', "\\\"");
            match layout {
                None => vec![format!(r#"["removeLayer", "{id}"]"#)],
                Some(layout) => layout
                    .iter()
                    .map(|(name, value)| {
                        format!(r#"["setLayoutProperty", "{id}", "{name}", {value}]"#)
                    })
                    .collect(),
            }
        })
        .collect();
    if commands.is_empty() {
        return Ok(());
    }
    let tx = Transaction::from_commands(&format!("[{}]", commands.join(",")))?;
    map.edit(&tx).map(|_| ())
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
) -> Result<LabelsReport, String> {
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
    let (mut turning, mut hidden) = (Vec::new(), Vec::new());
    for &(view, zoom, pitch) in VIEWS {
        let camera = CameraState {
            pitch,
            ..CameraState::new(settings.center, zoom)
        };
        progress(&format!("{view}: loading"));
        let mut map = new_map(camera)?;
        settle(gpu, &mut map, &render, &mut warnings).await?;

        progress(&format!("{view}: turning"));
        let count = settings.warmup + settings.frames;
        let (mut placement, mut cpu) = (Vec::new(), Vec::new());
        let (mut bytes, mut placements_written, mut paints_written) = (0, 0, 0);
        for f in 0..count {
            map.camera_mut().bearing += 0.25;
            let before = uploaded(gpu);
            let (report, time) = render(gpu, &mut map);
            match timing {
                Timing::Timestamps => gpu.throttle().await,
                Timing::WorkDone => gpu.finish().await,
            }
            if f >= settings.warmup {
                placement.push(Duration::from_micros(u64::from(report.placement_us)));
                cpu.push(time);
                bytes += uploaded(gpu) - before;
                placements_written += report.frame.placements_written;
                paints_written += report.frame.paints_written;
            }
        }
        gpu.take_frames().await;
        let frames = settings.frames as u32;
        turning.push(Turning {
            view,
            placement_max: placement.iter().copied().max().unwrap_or_default(),
            over_budget: placement
                .iter()
                .filter(|&&p| p > Duration::from_micros(200))
                .count(),
            placement: Summary::of(&mut placement),
            cpu: Summary::of(&mut cpu),
            bytes: bytes / u64::from(frames),
            placements_written: placements_written / frames,
            paints_written: paints_written / frames,
        });

        // One map per variant: edits that hide labels are not undone.
        for &(variant, layout) in VARIANTS {
            progress(&format!("{view}: {variant}"));
            let mut map = new_map(camera)?;
            settle(gpu, &mut map, &render, &mut warnings).await?;
            edit_symbols(&mut map, layout)?;
            settle(gpu, &mut map, &render, &mut warnings).await?;
            let (mut gpu_times, mut main) = (Vec::new(), Vec::new());
            for _ in 0..settings.rounds {
                let mut wall = Vec::new();
                for _ in 0..count {
                    let started = clock::now();
                    render(gpu, &mut map);
                    match timing {
                        Timing::Timestamps => gpu.throttle().await,
                        Timing::WorkDone => {
                            gpu.finish().await;
                            wall.push(clock::now() - started);
                        }
                    }
                }
                let times = gpu.take_frames().await;
                for f in settings.warmup..count {
                    gpu_times.push(match timing {
                        Timing::Timestamps => times[f].gpu(),
                        Timing::WorkDone => wall[f].saturating_sub(times[f].overhead),
                    });
                    if let Some(m) = times[f].pass("main") {
                        main.push(m);
                    }
                }
            }
            hidden.push(Hidden {
                view,
                variant,
                gpu: Summary::of(&mut gpu_times),
                main: (!main.is_empty()).then(|| Summary::of(&mut main)),
            });
        }
    }
    Ok(LabelsReport {
        about,
        size,
        settings: settings.clone(),
        timing,
        gpu_resolution: gpu.timestamp_resolution(),
        cpu_resolution: clock::resolution(),
        turning,
        hidden,
        warnings,
    })
}

impl LabelsReport {
    pub(crate) fn write(&self, out: &mut String) -> std::fmt::Result {
        let s = &self.settings;
        let gpu_stat = match self.timing {
            Timing::Timestamps => Stat::for_clock(self.gpu_resolution),
            Timing::WorkDone => Stat::for_clock(Some(self.cpu_resolution)),
        };
        let cpu_stat = Stat::for_clock(Some(self.cpu_resolution));
        writeln!(out, "### Labels bench (M2 exit)")?;
        writeln!(out)?;
        for line in &self.about {
            writeln!(out, "- {line}")?;
        }
        writeln!(
            out,
            "- {}×{} px offscreen at {:.4}, {:.4}; {} frames after {} warm-up; GPU {}s{}, CPU {}s",
            self.size.0,
            self.size.1,
            s.center.lon,
            s.center.lat,
            s.frames,
            s.warmup,
            gpu_stat.name(),
            if self.timing == Timing::WorkDone {
                " of the work-done proxy, not GPU time"
            } else {
                ""
            },
            cpu_stat.name(),
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "Turning 0.25° a frame (a placement asked for every frame):"
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| view | placement median | placement p90 | placement max | frames > 0.2 ms | render CPU | bytes/frame | placements/frame | paints/frame |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|")?;
        for t in &self.turning {
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                t.view,
                ms(t.placement.median),
                ms(t.placement.p90),
                ms(t.placement_max),
                t.over_budget,
                ms(cpu_stat.of(&t.cpu)),
                t.bytes,
                t.placements_written,
                t.paints_written,
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Symbol layers removed, as styled, nearly all hidden (padding 200), all shown (allow-overlap):"
        )?;
        writeln!(out)?;
        writeln!(out, "| view | symbols | GPU | main pass |")?;
        writeln!(out, "|---|---|---|---|")?;
        for h in &self.hidden {
            let main = h.main.as_ref().map_or("—".into(), |m| ms(gpu_stat.of(m)));
            writeln!(
                out,
                "| {} | {} | {} | {} |",
                h.view,
                h.variant,
                ms(gpu_stat.of(&h.gpu)),
                main
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Times in ms; placement time is the map clock's, in whole µs."
        )?;
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
        }
        Ok(())
    }
}
