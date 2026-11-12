//! Submission timing ([`WgpuGpu::set_timing`]): each submission's CPU time
//! to encode and submit, and, where the device has timestamp queries, each
//! compute and render pass's GPU time.

use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::WgpuGpu;
use crate::offscreen::map_read;

/// Timestamp queries in one set, WebGPU's most: two per pass. Submissions
/// between two [`WgpuGpu::take_timings`] that need more write further sets.
const QUERIES: u32 = 4096;

/// One submission's times.
#[derive(Clone, Debug, Default)]
pub struct SubmitTimes {
    /// Pass labels and their GPU durations, in submission order; empty
    /// without timestamp queries.
    pub passes: Vec<(String, Duration)>,
    /// From the first pass's start to the last one's end: what the
    /// submission took on the GPU. On tiled GPUs passes overlap, so their
    /// durations sum to more.
    pub span: Duration,
    /// Encoding the passes for wgpu and submitting them.
    pub cpu: Duration,
    /// Whether it ran a render or compute pass, as a frame does, rather
    /// than only copies (a table or page growing between frames).
    pub draws: bool,
}

impl SubmitTimes {
    /// The submission's GPU time, its [`span`](Self::span).
    pub fn gpu(&self) -> Duration {
        self.span
    }

    /// The GPU time of the pass labelled `label`, if it ran.
    pub fn pass(&self, label: &str) -> Option<Duration> {
        self.passes
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, d)| *d)
    }
}

/// Submissions timed since the last [`WgpuGpu::take_timings`].
#[derive(Debug)]
pub(crate) struct Timing {
    /// The queries the submissions write, without timestamp queries none.
    queries: Option<Queries>,
    /// Each submission's passes with their first query, its CPU time, and
    /// whether it draws.
    submissions: Vec<Submission>,
    /// The greatest common divisor of the timestamp steps seen, in ticks:
    /// the timer's resolution, once enough passes are timed.
    step: Arc<AtomicU64>,
    /// Sets a take has read, for later submissions to write again: making
    /// one costs more than the passes it times (on Metal a counter sample
    /// buffer, some tens of µs), and a bench takes every frame.
    spare: Arc<Mutex<Vec<wgpu::QuerySet>>>,
}

/// A submission's labelled passes and their first queries, its CPU time,
/// and whether it draws.
type Submission = (Vec<(String, u32)>, Duration, bool);

/// The query sets written since the last take, in order: query `i` is
/// set `i / QUERIES`'s `i % QUERIES`.
#[derive(Debug, Default)]
struct Queries {
    sets: Vec<wgpu::QuerySet>,
    /// The next query, counting through every set.
    next: u32,
}

impl Queries {
    /// The first of the next pass's two queries, in another set once the
    /// last is full: a spare one, or a new one.
    fn pair(&mut self, device: &wgpu::Device, spare: &Mutex<Vec<wgpu::QuerySet>>) -> u32 {
        if self.next == self.sets.len() as u32 * QUERIES {
            let set = spare.lock().expect("spare sets poisoned").pop();
            self.sets.push(set.unwrap_or_else(|| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("pass timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: QUERIES,
                })
            }));
        }
        let first = self.next;
        self.next += 2;
        first
    }
}

impl Timing {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            queries: (device.features())
                .contains(wgpu::Features::TIMESTAMP_QUERY)
                .then(Queries::default),
            submissions: Vec::new(),
            step: Arc::default(),
            spare: Arc::default(),
        }
    }

    /// The first of the two queries for each pass labelled in `labels`,
    /// which a submission is about to encode; `None` without timestamps.
    pub(crate) fn begin<'a>(
        &mut self,
        device: &wgpu::Device,
        labels: impl Iterator<Item = Option<&'a str>>,
    ) -> Vec<Option<u32>> {
        let mut passes = Vec::new();
        let mut draws = false;
        let firsts = labels
            .map(|label| {
                draws |= label.is_some();
                let (label, queries) = (label?, self.queries.as_mut()?);
                let first = queries.pair(device, &self.spare);
                passes.push((label.to_string(), first));
                Some(first)
            })
            .collect();
        self.submissions.push((passes, Duration::ZERO, draws));
        firsts
    }

    /// The set holding query `first`, which [`begin`](Self::begin) gave,
    /// and its index in it.
    pub(crate) fn query(&self, first: u32) -> Option<(&wgpu::QuerySet, u32)> {
        let sets = &self.queries.as_ref()?.sets;
        Some((sets.get((first / QUERIES) as usize)?, first % QUERIES))
    }

    /// The submission [`begin`](Self::begin) started took `cpu`.
    pub(crate) fn end(&mut self, cpu: Duration) {
        if let Some(last) = self.submissions.last_mut() {
            last.1 = cpu;
        }
    }
}

impl WgpuGpu {
    /// Times every submission from now on: its CPU time, and its passes'
    /// GPU times where the device has timestamp queries.
    /// [`take_timings`](Self::take_timings) collects them.
    pub fn set_timing(&mut self, on: bool) {
        self.timing = on.then(|| Timing::new(&self.device));
    }

    /// The timestamps' resolution, once a step between two has been seen.
    pub fn timestamp_resolution(&self) -> Option<Duration> {
        let step = self.timing.as_ref()?.step.load(Ordering::Relaxed);
        let ns = step as f64 * f64::from(self.timestamp_period_ns());
        (step > 0).then(|| Duration::from_nanos(ns.round() as u64))
    }

    /// The times of every submission since the last call, one per
    /// submission, once the GPU has finished them; none unless
    /// [`set_timing`](Self::set_timing) is on. The future holds no borrow
    /// of the device: frames may go on meanwhile, into the next call's.
    pub fn take_timings(&mut self) -> impl Future<Output = Vec<SubmitTimes>> + use<> {
        let done = self.work_done();
        let (device, queue) = (self.device.clone(), self.queue.clone());
        let period = f64::from(self.timestamp_period_ns());
        let (submissions, used, step, spare) = match &mut self.timing {
            Some(timing) => (
                std::mem::take(&mut timing.submissions),
                // Later submissions write other sets until these are read.
                timing.queries.as_mut().map(std::mem::take),
                timing.step.clone(),
                timing.spare.clone(),
            ),
            None => (Vec::new(), None, Arc::default(), Arc::default()),
        };
        async move {
            // The passes' last samples are written once their work
            // finishes: on tiled GPUs a pass's fragment work can outlast a
            // resolve submitted right after it, which then reads the old
            // value.
            done.await;
            let mut ticks = Vec::new();
            if let Some(Queries { sets, next: used }) = used.filter(|q| q.next > 0) {
                let size = u64::from(used) * 8;
                let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pass timestamps"),
                    size,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pass timestamps readback"),
                    size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("timestamps"),
                });
                for (set, start) in sets.iter().zip((0..).step_by(QUERIES as usize)) {
                    let count = (used - start).min(QUERIES);
                    encoder.resolve_query_set(set, 0..count, &resolve, u64::from(start) * 8);
                }
                encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, size);
                queue.submit([encoder.finish()]);
                ticks = map_read(&device, &readback)
                    .await
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| u64::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                let first = ticks.iter().copied().min().unwrap_or_default();
                let mut gcd_step = step.load(Ordering::Relaxed);
                for &t in &ticks {
                    gcd_step = gcd(gcd_step, t - first);
                }
                step.store(gcd_step, Ordering::Relaxed);
                spare.lock().expect("spare sets poisoned").extend(sets);
            }
            submissions
                .into_iter()
                .map(|(passes, cpu, draws)| {
                    let ns = |ticks: u64| Duration::from_nanos((ticks as f64 * period) as u64);
                    let (mut start, mut end) = (u64::MAX, 0);
                    let passes = passes
                        .into_iter()
                        .map(|(label, first)| {
                            let [begin, finish] =
                                [ticks[first as usize], ticks[first as usize + 1]];
                            start = start.min(begin);
                            end = end.max(finish);
                            (label, ns(finish.saturating_sub(begin)))
                        })
                        .collect();
                    SubmitTimes {
                        passes,
                        span: ns(end.saturating_sub(start)),
                        cpu,
                        draws,
                    }
                })
                .collect()
        }
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Now, on a clock CPU times are measured with.
pub(crate) fn now() -> Duration {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        EPOCH.get_or_init(std::time::Instant::now).elapsed()
    }
    #[cfg(target_arch = "wasm32")]
    {
        let ms = js_sys::Reflect::get(&js_sys::global(), &"performance".into())
            .ok()
            .and_then(|p| {
                let now = js_sys::Reflect::get(&p, &"now".into()).ok()?;
                js_sys::Function::from(now).call0(&p).ok()?.as_f64()
            })
            .unwrap_or_else(js_sys::Date::now);
        Duration::from_secs_f64(ms / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use orb_hal::{ComputePass, Gpu, Pass};

    use super::*;

    #[test]
    fn more_passes_than_a_query_set_holds_are_all_timed() {
        let Ok(mut gpu) = pollster::block_on(WgpuGpu::offscreen()) else {
            eprintln!("skipped: no GPU adapter");
            return;
        };
        gpu.set_timing(true);
        let pass = |label| {
            Pass::Compute(ComputePass {
                label,
                commands: Vec::new(),
            })
        };
        // Three passes a submission: a set's last pair falls in the middle
        // of a submission. Debug groups keep them apart, as they do in
        // captures: adjacent compute passes otherwise encode as one.
        gpu.set_debug_groups(true);
        let group = || [Pass::PushDebugGroup("g"), Pass::PopDebugGroup];
        let submissions = QUERIES as usize;
        for _ in 0..submissions {
            let passes: Vec<Pass> = [vec![pass("a")], group().into(), vec![pass("b")]]
                .into_iter()
                .chain([group().into(), vec![pass("c")]])
                .flatten()
                .collect();
            gpu.submit(&passes);
        }
        let times = pollster::block_on(gpu.take_timings());
        assert_eq!(times.len(), submissions);
        let timed = gpu
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY);
        for t in &times {
            let labels: Vec<_> = t.passes.iter().map(|(l, _)| l.as_str()).collect();
            assert_eq!(labels, if timed { &["a", "b", "c"][..] } else { &[] });
        }
        // Timing goes on after a take.
        gpu.submit(&[pass("a")]);
        assert_eq!(pollster::block_on(gpu.take_timings()).len(), 1);
    }
}
