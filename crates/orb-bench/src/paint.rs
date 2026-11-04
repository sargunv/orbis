//! The paint binding bench (decision 0017): GPU frame time with
//! data-driven paint bound as CPU property rows against GPU bytecode. For
//! the style as it is, and for every fill and line colour replaced by a
//! program of N instructions, two maps draw the same view: one with a cost
//! model that binds nothing as bytecode, one with the default. Rounds
//! alternate between them.

use std::fmt::Write;
use std::time::Duration;

use orb_hal::TextureId;
use orb_runtime::{CameraState, CostModel, FrameReport, Map, TargetFormat, Transaction, Viewport};
use orb_v8::LayerKind;

use crate::report::{Stat, Summary, ms};
use crate::timed::TimedGpu;
use crate::{LOAD_TIMEOUT, Settings, Timing, clock};

/// Name, zoom and pitch: the default view and its heaviest pitched one.
const VIEWS: &[(&str, f64, f64)] = &[("z15", 15.0, 0.0), ("z15 pitch 60", 15.0, 60.0)];

/// The swept colours: a typical `match` on `class` (two instructions, eight
/// labels), and arithmetic chains of `3 + 2k` instructions.
const SWEEP: &[(&str, Option<usize>)] = &[
    ("match on class, 8 labels (2 ops)", None),
    ("5 ops", Some(1)),
    ("9 ops", Some(3)),
    ("17 ops", Some(7)),
    ("31 ops", Some(14)),
];

/// A fill or line colour: the `match`, or `3 + 2k` instructions on the
/// feature id, which most features have: `to-number` (two), `k`
/// multiply-adds, and an interpolation.
fn colour(chain: Option<usize>) -> String {
    let Some(k) = chain else {
        let labels = [
            "primary",
            "secondary",
            "tertiary",
            "minor",
            "service",
            "residential",
            "wood",
            "grass",
        ];
        let arms: Vec<String> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| format!(r##""{l}", "#{:02x}40{:02x}""##, i * 30, 255 - i * 30))
            .collect();
        return format!(
            r##"["match", ["get", "class"], {}, "#888888"]"##,
            arms.join(", ")
        );
    };
    let mut x = r#"["to-number", ["id"]]"#.to_string();
    for _ in 0..k {
        x = format!(r#"["+", ["*", {x}, 1.0001], 1]"#);
    }
    format!(r##"["interpolate", ["linear"], {x}, 0, "#ff0000", 1000000000, "#0000ff"]"##)
}

/// One expression's rows and bytecode samples in one view.
#[derive(Debug)]
pub(crate) struct Row {
    view: &'static str,
    paint: &'static str,
    /// Rows, then bytecode.
    gpu: [Summary; 2],
    main: [Option<Summary>; 2],
    cpu: [Summary; 2],
    /// Pixels of the bytecode frame that differ from the rows frame, and
    /// the largest channel difference.
    differ: usize,
    max_delta: u8,
}

/// Every view's rows and bytecode timings.
#[derive(Debug)]
pub struct PaintReport {
    about: Vec<String>,
    size: (u32, u32),
    settings: Settings,
    timing: Timing,
    gpu_resolution: Option<Duration>,
    cpu_resolution: Duration,
    rows: Vec<Row>,
    warnings: Vec<String>,
}

#[derive(Default)]
struct Samples {
    gpu: Vec<Duration>,
    main: Vec<Duration>,
    cpu: Vec<Duration>,
    pixels: Vec<u8>,
}

/// Draws until the map is idle, then once more.
pub(crate) async fn settle(
    gpu: &mut TimedGpu,
    map: &mut Map,
    render: &impl Fn(&mut TimedGpu, &mut Map) -> (FrameReport, Duration),
    warnings: &mut Vec<String>,
) -> Result<(), String> {
    let started = clock::now();
    loop {
        let (report, _) = render(gpu, map);
        warnings.extend(map.take_warnings());
        gpu.take_frames().await;
        if report.idle {
            render(gpu, map);
            gpu.take_frames().await;
            return Ok(());
        }
        if clock::now() - started > LOAD_TIMEOUT {
            return Err("timed out waiting for tiles".into());
        }
        clock::sleep(Duration::from_millis(5)).await;
    }
}

/// Applies `value` to every fill and line colour of the map's style.
fn sweep(map: &mut Map, value: &str) -> Result<(), String> {
    let doc = map.style_doc().ok_or("no style")?;
    let commands: Vec<String> = doc
        .style()
        .layers
        .iter()
        .filter_map(|l| {
            let name = match l.kind {
                LayerKind::Fill(_) => "fill-color",
                LayerKind::Line(_) => "line-color",
                _ => return None,
            };
            Some(format!(
                r#"["setPaintProperty", "{}", "{name}", {value}]"#,
                l.id
            ))
        })
        .collect();
    let tx = Transaction::from_commands(&format!("[{}]", commands.join(",")))?;
    map.edit(&tx).map(|_| ())
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
) -> Result<PaintReport, String> {
    let (target, size) = target;
    let viewport = Viewport {
        width: f64::from(settings.width),
        height: f64::from(settings.height),
    };
    let render = |gpu: &mut TimedGpu, map: &mut Map| {
        let started = clock::now();
        let report = map.render(gpu, target, format, viewport, settings.ratio);
        (report, clock::now() - started)
    };
    let models = [CostModel::ROWS, CostModel::ALL];
    let mut warnings = Vec::new();
    let mut rows = Vec::new();
    for &(view, zoom, pitch) in VIEWS {
        let camera = CameraState {
            pitch,
            ..CameraState::new(settings.center, zoom)
        };
        // Two maps a view: each swept colour is an edit of both.
        progress(&format!("{view}: loading"));
        let mut maps = Vec::with_capacity(2);
        for model in models {
            let mut map = new_map(camera)?;
            map.set_cost_model(model);
            settle(gpu, &mut map, &render, &mut warnings).await?;
            maps.push(map);
        }
        for (label, chain) in [("style", None)]
            .into_iter()
            .chain(SWEEP.iter().map(|&(l, c)| (l, Some(c))))
        {
            if let Some(chain) = chain {
                progress(&format!("{view}, {label}: editing"));
                for map in &mut maps {
                    sweep(map, &colour(chain))?;
                    settle(gpu, map, &render, &mut warnings).await?;
                }
            }
            let mut samples = [Samples::default(), Samples::default()];
            for round in 0..settings.rounds {
                for k in 0..2 {
                    let i = (round + k) % 2;
                    progress(&format!(
                        "{view}, {label}: round {} of {}, {}",
                        round + 1,
                        settings.rounds,
                        ["rows", "bytecode"][i]
                    ));
                    let s = &mut samples[i];
                    let count = settings.warmup + settings.frames;
                    let (mut cpu, mut wall) = (Vec::new(), Vec::new());
                    for _ in 0..count {
                        let started = clock::now();
                        let (_, time) = render(gpu, &mut maps[i]);
                        match timing {
                            Timing::Timestamps => gpu.throttle().await,
                            Timing::WorkDone => {
                                gpu.finish().await;
                                wall.push(clock::now() - started);
                            }
                        }
                        cpu.push(time);
                    }
                    let times = gpu.take_frames().await;
                    for f in settings.warmup..count {
                        s.gpu.push(match timing {
                            Timing::Timestamps => times[f].gpu(),
                            Timing::WorkDone => wall[f].saturating_sub(times[f].overhead),
                        });
                        if let Some(main) = times[f].pass("main") {
                            s.main.push(main);
                        }
                        s.cpu.push(cpu[f].saturating_sub(times[f].overhead));
                    }
                    s.pixels = gpu.inner().read_texture(target).await;
                }
            }
            let pairs = || samples[0].pixels.chunks(4).zip(samples[1].pixels.chunks(4));
            let differ = pairs().filter(|(a, b)| a != b).count();
            let max_delta = pairs()
                .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)))
                .max()
                .unwrap_or(0);
            let [r, b] = &mut samples;
            rows.push(Row {
                view,
                paint: label,
                gpu: [Summary::of(&mut r.gpu), Summary::of(&mut b.gpu)],
                main: [&mut r.main, &mut b.main].map(|m| (!m.is_empty()).then(|| Summary::of(m))),
                cpu: [Summary::of(&mut r.cpu), Summary::of(&mut b.cpu)],
                differ,
                max_delta,
            });
        }
    }
    Ok(PaintReport {
        about,
        size,
        settings: settings.clone(),
        timing,
        gpu_resolution: gpu.timestamp_resolution(),
        cpu_resolution: clock::resolution(),
        rows,
        warnings,
    })
}

impl PaintReport {
    pub(crate) fn write(&self, out: &mut String) -> std::fmt::Result {
        let s = &self.settings;
        let gpu_stat = match self.timing {
            Timing::Timestamps => Stat::for_clock(self.gpu_resolution),
            Timing::WorkDone => Stat::for_clock(Some(self.cpu_resolution)),
        };
        let cpu_stat = Stat::for_clock(Some(self.cpu_resolution));
        writeln!(out, "### Paint binding bench (decision 0017)")?;
        writeln!(out)?;
        for line in &self.about {
            writeln!(out, "- {line}")?;
        }
        writeln!(
            out,
            "- {}×{} px offscreen at {:.4}, {:.4}; {} rounds × {} frames after {} warm-up per binding; GPU {}s{}, CPU {}s",
            self.size.0,
            self.size.1,
            s.center.lon,
            s.center.lat,
            s.rounds,
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
        writeln!(
            out,
            "- Rows: `CostModel::ROWS`; bytecode: `CostModel::ALL`, every value the GPU can evaluate interpreted. Below the style as it is, every fill and line colour is replaced: a `match` on `class`, then programs of N instructions on the feature id."
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| view | paint | rows GPU | bytecode GPU | ratio | difference | rows main | bytecode main | rows CPU | bytecode CPU | pixels ≠ | max Δ |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|---|")?;
        for row in &self.rows {
            let [r, b] = row.gpu.map(|g| gpu_stat.of(&g));
            let (rf, bf) = (r.as_secs_f64(), b.as_secs_f64());
            let main = |m: &Option<Summary>| m.as_ref().map_or("—".into(), |m| ms(gpu_stat.of(m)));
            writeln!(
                out,
                "| {} | {} | {} | {} | {:.2}× | {:+.3} | {} | {} | {} | {} | {} | {} |",
                row.view,
                row.paint,
                ms(r),
                ms(b),
                bf / rf.max(f64::MIN_POSITIVE),
                (bf - rf) * 1e3,
                main(&row.main[0]),
                main(&row.main[1]),
                ms(cpu_stat.of(&row.cpu[0])),
                ms(cpu_stat.of(&row.cpu[1])),
                row.differ,
                row.max_delta,
            )?;
        }
        writeln!(out)?;
        writeln!(out, "Times in ms.")?;
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
        }
        Ok(())
    }
}
