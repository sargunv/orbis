//! Statistics and the frame bench's Markdown report.

use std::fmt::Write;
use std::time::Duration;

use crate::{Settings, Timing};

/// Clocks at least this fine give medians; coarser ones give means, which
/// quantization does not bias.
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

/// One view's still frames.
#[derive(Debug)]
pub(crate) struct ViewResult {
    pub(crate) name: String,
    pub(crate) gpu: Summary,
    /// Passes, where timed.
    pub(crate) cull: Option<Summary>,
    pub(crate) main: Option<Summary>,
    /// Every pass label's time per frame and how many passes it has.
    pub(crate) passes: Vec<(String, Summary, usize)>,
    pub(crate) render_cpu: Summary,
    pub(crate) submit_cpu: Summary,
    pub(crate) draws: u32,
    pub(crate) dispatches: u32,
    pub(crate) tiles: u32,
    pub(crate) candidates: u32,
    pub(crate) meshes: u32,
    /// GPU bytes the map holds, and those its scene pools use.
    pub(crate) memory: (u64, u64),
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

    /// The report as Markdown: setup and still frames.
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
            "- {}×{} px ({}×{} @{}x) offscreen at {:.4}, {:.4}; {} rounds × {} frames after {} warm-up per view",
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
            "| view | {gpu_label} | p90 | cull pass {g} | main pass {g} | CPU render {c} | CPU encode+submit {c} | draws | dispatches |"
        )?;
        writeln!(out, "|---|---|---|---|---|---|---|---|---|")?;
        let pass = |p: &Option<Summary>| p.as_ref().map_or("—".into(), |m| ms(gpu_stat.of(m)));
        for view in &self.views {
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                view.name,
                ms(gpu_stat.of(&view.gpu)),
                ms(view.gpu.p90),
                pass(&view.cull),
                pass(&view.main),
                ms(cpu_stat.of(&view.render_cpu)),
                ms(cpu_stat.of(&view.submit_cpu)),
                view.draws,
                view.dispatches,
            )?;
        }
        writeln!(out)?;
        writeln!(out, "Times in ms.")?;
        for view in &self.views {
            writeln!(
                out,
                "- {}: {} tiles drawn, {} candidates, {} meshes, \
                 {:.1} MiB of GPU memory allocated, {:.1} MiB of pools used",
                view.name,
                view.tiles,
                view.candidates,
                view.meshes,
                view.memory.0 as f64 / f64::from(1 << 20),
                view.memory.1 as f64 / f64::from(1 << 20),
            )?;
            if !view.passes.is_empty() {
                let passes: Vec<String> = (view.passes.iter())
                    .map(|(label, t, n)| format!("{label} ×{n} {}", ms(gpu_stat.of(t))))
                    .collect();
                writeln!(out, "  - GPU by pass: {}", passes.join(", "))?;
            }
        }
        for warning in &self.warnings {
            writeln!(out, "- warning: {warning}")?;
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
