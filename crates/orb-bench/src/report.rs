//! Statistics and the frame bench's Markdown report.

use std::fmt::Write;
use std::time::Duration;

use crate::{Settings, Timing};

/// Clocks at least this fine give medians; coarser ones give means, which
/// quantization does not bias. Chrome quantizes timestamps (to 65.5 µs on a
/// Mac) and `performance.now()` (to 100 µs, or 5 µs cross-origin isolated).
const FINE: Duration = Duration::from_micros(1);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Summary {
    pub(crate) median: Duration,
    pub(crate) p90: Duration,
    pub(crate) mean: Duration,
}

impl Summary {
    pub(crate) fn of(samples: &mut [Duration]) -> Self {
        samples.sort_unstable();
        let at = |p: f64| {
            let i = ((samples.len() as f64 - 1.0) * p).round() as usize;
            samples.get(i).copied().unwrap_or_default()
        };
        let total: Duration = samples.iter().sum();
        Self {
            median: at(0.5),
            p90: at(0.9),
            mean: total / samples.len().max(1) as u32,
        }
    }
}

/// A statistic: medians for fine clocks, means for coarse ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stat {
    Median,
    Mean,
}

impl Stat {
    pub(crate) fn for_clock(resolution: Option<Duration>) -> Self {
        match resolution {
            Some(r) if r <= FINE => Stat::Median,
            _ => Stat::Mean,
        }
    }

    pub(crate) fn of(self, s: &Summary) -> Duration {
        match self {
            Stat::Median => s.median,
            Stat::Mean => s.mean,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Stat::Median => "median",
            Stat::Mean => "mean",
        }
    }
}

/// One view's still frames and its moving camera.
#[derive(Debug)]
pub(crate) struct ViewResult {
    pub(crate) name: String,
    pub(crate) gpu: Summary,
    /// Passes, where timed; extrusions' only where the view has them.
    pub(crate) cull: Option<Summary>,
    pub(crate) main: Option<Summary>,
    pub(crate) extrusion: Option<Summary>,
    pub(crate) render_cpu: Summary,
    pub(crate) submit_cpu: Summary,
    pub(crate) draws: u32,
    pub(crate) dispatches: u32,
    /// The cull pass's counts on the last frame: chunks kept, and vertex
    /// inputs (pulled vertices, or visible chunks' indices).
    pub(crate) chunks: u64,
    pub(crate) inputs: u64,
    pub(crate) tiles: u32,
    pub(crate) candidates: u32,
    pub(crate) meshes: u32,
    pub(crate) moving: Moving,
}

/// Frames along the moving camera's path.
#[derive(Debug, Default)]
pub(crate) struct Moving {
    pub(crate) frames: Vec<MovingFrame>,
    /// Frames during which tiles were loading.
    pub(crate) loading: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MovingFrame {
    /// GPU time or proxy.
    pub(crate) gpu: Duration,
    pub(crate) cull: Option<Duration>,
    pub(crate) cpu: Duration,
    /// Compacted indices the cull pass wrote.
    pub(crate) written: u64,
}

/// Every view's results, and how they were measured.
#[derive(Debug)]
pub struct FramesReport {
    pub(crate) about: Vec<String>,
    pub(crate) settings: Settings,
    /// The target in physical pixels.
    pub(crate) size: (u32, u32),
    pub(crate) timing: Timing,
    /// The timestamps' resolution, if any were taken.
    pub(crate) gpu_resolution: Option<Duration>,
    pub(crate) cpu_resolution: Duration,
    pub(crate) views: Vec<ViewResult>,
    pub(crate) warnings: Vec<String>,
}

impl FramesReport {
    fn gpu_stat(&self) -> Stat {
        match self.timing {
            Timing::Timestamps => Stat::for_clock(self.gpu_resolution),
            Timing::WorkDone => Stat::for_clock(Some(self.cpu_resolution)),
        }
    }

    fn cpu_stat(&self) -> Stat {
        Stat::for_clock(Some(self.cpu_resolution))
    }

    /// The report as Markdown: setup, still frames, the moving camera.
    pub(crate) fn write(&self, out: &mut String) -> std::fmt::Result {
        let s = &self.settings;
        let (gpu_stat, cpu_stat) = (self.gpu_stat(), self.cpu_stat());
        let (g, c) = (gpu_stat.name(), cpu_stat.name());
        writeln!(out, "### Frame bench (decision 0002)")?;
        writeln!(out)?;
        for line in &self.about {
            writeln!(out, "- {line}")?;
        }
        writeln!(
            out,
            "- {}×{} px ({}×{} @{}x) offscreen at {:.4}, {:.4}; {} rounds × {} frames after {} warm-up per view; labels {}",
            self.size.0,
            self.size.1,
            s.width,
            s.height,
            s.ratio,
            s.center.lon,
            s.center.lat,
            s.rounds,
            s.frames,
            s.warmup,
            if s.labels { "on" } else { "off" },
        )?;
        match self.timing {
            Timing::Timestamps => writeln!(
                out,
                "- GPU: timestamp queries around every pass, two frames in flight; resolution {}{}",
                resolution(self.gpu_resolution),
                coarse_note(gpu_stat),
            )?,
            Timing::WorkDone => writeln!(
                out,
                "- GPU: **no timestamp queries; the GPU column is a proxy, not GPU time**: wall time from the start of each frame until the GPU reports its work done, one frame in flight. It adds the frame's CPU work and the queue's latency.",
            )?,
        }
        writeln!(
            out,
            "- CPU clock resolution {}{}",
            resolution(Some(self.cpu_resolution)),
            coarse_note(cpu_stat),
        )?;
        writeln!(out)?;

        let gpu_label = match self.timing {
            Timing::Timestamps => format!("GPU {g}"),
            Timing::WorkDone => format!("proxy (work done) {g}"),
        };
        writeln!(
            out,
            "| view | {gpu_label} | p90 | cull pass {g} | main pass {g} | extrusion pass {g} | CPU render {c} | CPU encode+submit {c} | draws | dispatches | vertex inputs |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|")?;
        let pass = |p: &Option<Summary>| p.as_ref().map_or("—".into(), |m| ms(gpu_stat.of(m)));
        for view in &self.views {
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                view.name,
                ms(gpu_stat.of(&view.gpu)),
                ms(view.gpu.p90),
                pass(&view.cull),
                pass(&view.main),
                pass(&view.extrusion),
                ms(cpu_stat.of(&view.render_cpu)),
                ms(cpu_stat.of(&view.submit_cpu)),
                view.draws,
                view.dispatches,
                thousands(view.inputs),
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "Times in ms. Vertex inputs: the visible chunks' indices, and pulled point and image vertices."
        )?;
        for view in &self.views {
            writeln!(
                out,
                "- {}: {} tiles drawn, {} candidates, {} visible chunks, {} meshes",
                view.name, view.tiles, view.candidates, view.chunks, view.meshes,
            )?;
        }
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
        }
        writeln!(out)?;
        self.write_moving(out, gpu_stat, cpu_stat)
    }

    fn write_moving(&self, out: &mut String, gpu_stat: Stat, cpu_stat: Stat) -> std::fmt::Result {
        let (g, c) = (gpu_stat.name(), cpu_stat.name());
        writeln!(out, "#### Moving camera")?;
        writeln!(out)?;
        writeln!(
            out,
            "The camera pans around a circle of {} px radius in {} frames (about {:.1} px per frame), once to load the path, then measured in {} rounds. Frames that rewrite compacted indices are those where chunks entered or left the view, or the scene changed (every slot is rewritten then). While moving, the CPU's frame dominates and the GPU's clock varies.",
            crate::PATH_RADIUS,
            crate::PATH_FRAMES,
            std::f64::consts::TAU * crate::PATH_RADIUS / crate::PATH_FRAMES as f64,
            self.settings.rounds,
        )?;
        writeln!(out)?;
        writeln!(
            out,
            "| view | GPU {g} | p90 | max | CPU render {c} | max | frames rewriting | indices rewritten per such frame, mean | max | cull pass on those frames, {g} | max | cull pass on the others, {g} |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|---|---|---|")?;
        for view in &self.views {
            let m = &view.moving;
            let summary = |f: &dyn Fn(&MovingFrame) -> Option<Duration>,
                           which: &dyn Fn(&MovingFrame) -> bool| {
                let mut v: Vec<Duration> =
                    m.frames.iter().filter(|x| which(x)).filter_map(f).collect();
                (!v.is_empty()).then(|| {
                    let max = v.iter().copied().max().unwrap_or_default();
                    (Summary::of(&mut v), max)
                })
            };
            let show = |s: Option<(Summary, Duration)>, stat: Stat| {
                s.map_or(("—".to_string(), "—".to_string()), |(s, max)| {
                    (ms(stat.of(&s)), ms(max))
                })
            };
            let all = |_: &MovingFrame| true;
            let rewriting = |x: &MovingFrame| x.written > 0;
            let other = |x: &MovingFrame| x.written == 0;
            let (gpu, gpu_max) = summary(&|x| Some(x.gpu), &all).expect("frames");
            let (cpu, cpu_max) = summary(&|x| Some(x.cpu), &all).expect("frames");
            let (cull_rewriting, cull_rewriting_max) =
                show(summary(&|x| x.cull, &rewriting), gpu_stat);
            let (cull_other, _) = show(summary(&|x| x.cull, &other), gpu_stat);
            let written: Vec<u64> = m
                .frames
                .iter()
                .map(|x| x.written)
                .filter(|&w| w > 0)
                .collect();
            let (written_mean, written_max) = match written.iter().max() {
                Some(&max) => (
                    thousands(written.iter().sum::<u64>() / written.len() as u64),
                    thousands(max),
                ),
                None => ("—".to_string(), "—".to_string()),
            };
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} of {} | {} | {} | {} | {} | {} |",
                view.name,
                ms(gpu_stat.of(&gpu)),
                ms(gpu.p90),
                ms(gpu_max),
                ms(cpu_stat.of(&cpu)),
                ms(cpu_max),
                written.len(),
                m.frames.len(),
                written_mean,
                written_max,
                cull_rewriting,
                cull_rewriting_max,
                cull_other,
            )?;
        }
        writeln!(out)?;
        for view in self.views.iter().filter(|v| v.moving.loading > 0) {
            writeln!(
                out,
                "- {}: tiles were loading in {} of {} measured frames",
                view.name,
                view.moving.loading,
                view.moving.frames.len()
            )?;
        }
        Ok(())
    }
}

fn coarse_note(stat: Stat) -> &'static str {
    match stat {
        Stat::Median => " (medians)",
        Stat::Mean => ", coarse: the table shows means over every measured frame",
    }
}

pub(crate) fn resolution(r: Option<Duration>) -> String {
    match r {
        Some(r) if r < Duration::from_micros(1) => format!("{} ns", r.as_nanos()),
        Some(r) => format!("{:.1} µs", r.as_secs_f64() * 1e6),
        None => "unknown".into(),
    }
}

pub(crate) fn ms(d: Duration) -> String {
    format!("{:.3}", d.as_secs_f64() * 1000.0)
}

fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_and_statistic() {
        let us = Duration::from_micros;
        let mut samples = vec![us(300), us(100), us(200), us(200)];
        let s = Summary::of(&mut samples);
        assert_eq!((s.median, s.mean), (us(200), us(200)));
        assert_eq!(s.p90, us(300));
        assert_eq!(
            Stat::for_clock(Some(Duration::from_nanos(40))),
            Stat::Median
        );
        assert_eq!(Stat::for_clock(Some(us(100))), Stat::Mean);
        assert_eq!(Stat::for_clock(None), Stat::Mean);
    }
}
