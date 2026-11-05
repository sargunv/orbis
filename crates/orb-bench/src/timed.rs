//! A [`Gpu`] that times every pass with timestamp queries, if asked to, and
//! every submission on the CPU, and can copy a buffer the GPU writes after
//! every submission.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use orb_hal::*;
use orb_hal_wgpu::WgpuGpu;

use crate::clock;

/// Queries in the set: two per pass, flushed by [`TimedGpu::take_frames`].
const QUERIES: u32 = 4096;
/// Submissions in flight before [`TimedGpu::throttle`] waits, as a
/// swapchain would.
const IN_FLIGHT: usize = 2;

/// A pass's label and its timestamp queries.
#[derive(Debug)]
struct PassQueries {
    label: String,
    begin: u32,
}

/// One submission: its passes and CPU times.
#[derive(Debug, Default)]
struct Submission {
    passes: Vec<PassQueries>,
    cpu: Duration,
    overhead: Duration,
}

/// GPU time per pass label, and CPU time to encode and submit, for one
/// submission.
#[derive(Clone, Debug, Default)]
pub(crate) struct FrameTimes {
    /// Pass labels and their GPU durations, in submission order; empty
    /// without timestamps.
    pub(crate) passes: Vec<(String, Duration)>,
    /// Encoding passes for wgpu and submitting them.
    pub(crate) submit_cpu: Duration,
    /// Time spent adding timestamps, to subtract from the caller's.
    pub(crate) overhead: Duration,
}

impl FrameTimes {
    pub(crate) fn gpu(&self) -> Duration {
        self.passes.iter().map(|(_, d)| *d).sum()
    }

    pub(crate) fn pass(&self, label: &str) -> Option<Duration> {
        self.passes
            .iter()
            .find(|(l, _)| l == label)
            .map(|(_, d)| *d)
    }
}

/// The query set and its resolve buffer.
#[derive(Debug)]
struct Queries {
    set: QuerySetId,
    resolve: BufferId,
    next: u32,
    /// The greatest common divisor of the timestamp steps seen, in ticks:
    /// the timer's resolution, once enough passes are timed.
    step: u64,
}

pub(crate) struct TimedGpu {
    inner: WgpuGpu,
    queries: Option<Queries>,
    submissions: Vec<Submission>,
    in_flight: VecDeque<Pin<Box<dyn Future<Output = ()>>>>,
    capture: Option<Capture>,
}

/// Copies of a buffer's first `size` bytes, one per submission, into a
/// ring read back by [`TimedGpu::take_captures`].
#[derive(Debug)]
struct Capture {
    source: BufferId,
    size: u64,
    ring: BufferId,
    slots: u64,
    next: u64,
}

impl std::fmt::Debug for TimedGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TimedGpu")
            .field("queries", &self.queries)
            .field("in_flight", &self.in_flight.len())
            .finish_non_exhaustive()
    }
}

impl TimedGpu {
    /// With `timestamps`, every pass is timed; the device needs
    /// timestamp queries.
    pub(crate) fn new(mut inner: WgpuGpu, timestamps: bool) -> Self {
        let queries = timestamps.then(|| {
            assert!(inner.caps().timestamp_query, "the device has no timestamps");
            Queries {
                set: inner.create_query_set(&QuerySetDesc {
                    label: "pass timestamps",
                    count: QUERIES,
                }),
                resolve: inner.create_buffer(&BufferDesc {
                    label: "timestamps",
                    size: u64::from(QUERIES) * 8,
                    usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                }),
                next: 0,
                step: 0,
            }
        });
        Self {
            inner,
            queries,
            submissions: Vec::new(),
            in_flight: VecDeque::new(),
            capture: None,
        }
    }

    /// Copies `source`'s first `size` bytes after each of the next
    /// `submissions` submissions.
    pub(crate) fn capture(&mut self, source: BufferId, size: u64, submissions: u64) {
        assert!(self.capture.is_none(), "one capture at a time");
        let ring = self.inner.create_buffer(&BufferDesc {
            label: "captures",
            size: size * submissions,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });
        self.capture = Some(Capture {
            source,
            size,
            ring,
            slots: submissions,
            next: 0,
        });
    }

    /// The captured copies, in submission order, ending the capture.
    pub(crate) async fn take_captures(&mut self) -> Vec<Vec<u32>> {
        let capture = self.capture.take().expect("capturing");
        let bytes = self.inner.read_buffer(capture.ring).await;
        self.inner.destroy_buffer(capture.ring);
        bytes[..(capture.next * capture.size) as usize]
            .chunks(capture.size as usize)
            .map(|copy| {
                copy.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| u32::from_le_bytes(*w))
                    .collect()
            })
            .collect()
    }

    pub(crate) fn inner(&self) -> &WgpuGpu {
        &self.inner
    }

    /// The timestamps' resolution, once a step has been seen.
    pub(crate) fn timestamp_resolution(&self) -> Option<Duration> {
        let step = self.queries.as_ref()?.step;
        let ns = step as f64 * f64::from(self.inner.timestamp_period_ns());
        (step > 0).then(|| Duration::from_nanos(ns.round() as u64))
    }

    /// Waits until at most [`IN_FLIGHT`] submissions are unfinished.
    pub(crate) async fn throttle(&mut self) {
        self.in_flight.push_back(Box::pin(self.inner.work_done()));
        while self.in_flight.len() > IN_FLIGHT {
            self.in_flight.pop_front().expect("non-empty").await;
        }
    }

    /// Waits until the GPU has finished everything submitted.
    pub(crate) async fn finish(&mut self) {
        self.inner.work_done().await;
        self.in_flight.clear();
    }

    /// The times of every submission since the last call, one per
    /// submission, waiting until the GPU finishes them.
    pub(crate) async fn take_frames(&mut self) -> Vec<FrameTimes> {
        let mut ticks = Vec::new();
        if let Some(queries) = self.queries.as_mut().filter(|q| q.next > 0) {
            let used = std::mem::take(&mut queries.next);
            let (set, resolve) = (queries.set, queries.resolve);
            self.inner.submit(&[Pass::ResolveQuerySet {
                query_set: set,
                queries: 0..used,
                destination: resolve,
                offset: 0,
            }]);
            let bytes = self.inner.read_buffer(resolve).await;
            ticks = bytes[..used as usize * 8]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect();
            let first = ticks.iter().copied().min().unwrap_or_default();
            let queries = self.queries.as_mut().expect("checked");
            for &t in &ticks {
                queries.step = gcd(queries.step, t - first);
            }
        }
        self.finish().await;
        let period = f64::from(self.inner.timestamp_period_ns());
        std::mem::take(&mut self.submissions)
            .into_iter()
            .map(|submission| FrameTimes {
                passes: submission
                    .passes
                    .into_iter()
                    .map(|pass| {
                        let begin = pass.begin as usize;
                        let elapsed = ticks[begin + 1].saturating_sub(ticks[begin]);
                        let ns = elapsed as f64 * period;
                        (pass.label, Duration::from_nanos(ns as u64))
                    })
                    .collect(),
                submit_cpu: submission.cpu,
                overhead: submission.overhead,
            })
            .collect()
    }

    fn timestamps(&mut self, label: &str, passes: &mut Vec<PassQueries>) -> Option<PassTimestamps> {
        let queries = self.queries.as_mut()?;
        assert!(queries.next + 2 <= QUERIES, "take_frames more often");
        let begin = queries.next;
        queries.next += 2;
        passes.push(PassQueries {
            label: label.to_string(),
            begin,
        });
        Some(PassTimestamps {
            query_set: queries.set,
            beginning: Some(begin),
            end: Some(begin + 1),
        })
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl Gpu for TimedGpu {
    fn caps(&self) -> &Caps {
        self.inner.caps()
    }

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> BufferId {
        self.inner.create_buffer(desc)
    }

    fn destroy_buffer(&mut self, id: BufferId) {
        self.inner.destroy_buffer(id);
    }

    fn write_buffer(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        self.inner.write_buffer(id, offset, data);
    }

    fn write_buffer_from_host(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: HostBytesId,
        range: std::ops::Range<u64>,
    ) {
        self.inner.write_buffer_from_host(id, offset, bytes, range);
    }

    fn read_host_bytes(&self, bytes: HostBytesId, range: std::ops::Range<u64>) -> Vec<u8> {
        self.inner.read_host_bytes(bytes, range)
    }

    fn release_host_bytes(&mut self, bytes: HostBytesId) {
        self.inner.release_host_bytes(bytes);
    }

    fn create_texture(&mut self, desc: &TextureDesc<'_>) -> TextureId {
        self.inner.create_texture(desc)
    }

    fn create_texture_view(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
    ) -> TextureId {
        self.inner.create_texture_view(texture, levels)
    }

    fn create_layer_view(&mut self, texture: TextureId, layer: u32) -> TextureId {
        self.inner.create_layer_view(texture, layer)
    }

    fn destroy_texture(&mut self, id: TextureId) {
        self.inner.destroy_texture(id);
    }

    fn write_texture(&mut self, region: &TextureRegion, data: &[u8]) {
        self.inner.write_texture(region, data);
    }

    fn write_texture_from_host(
        &mut self,
        region: &TextureRegion,
        image: HostImageId,
        premultiply: bool,
    ) {
        self.inner
            .write_texture_from_host(region, image, premultiply);
    }

    fn release_host_image(&mut self, image: HostImageId) {
        self.inner.release_host_image(image);
    }

    fn create_sampler(&mut self, desc: &SamplerDesc<'_>) -> SamplerId {
        self.inner.create_sampler(desc)
    }

    fn create_shader_module(&mut self, desc: &ShaderModuleDesc<'_>) -> ShaderModuleId {
        self.inner.create_shader_module(desc)
    }

    fn create_bind_group_layout(&mut self, desc: &BindGroupLayoutDesc<'_>) -> BindGroupLayoutId {
        self.inner.create_bind_group_layout(desc)
    }

    fn create_bind_group(&mut self, desc: &BindGroupDesc<'_>) -> BindGroupId {
        self.inner.create_bind_group(desc)
    }

    fn destroy_bind_group(&mut self, id: BindGroupId) {
        self.inner.destroy_bind_group(id);
    }

    fn create_render_pipeline(&mut self, desc: &RenderPipelineDesc<'_>) -> RenderPipelineId {
        self.inner.create_render_pipeline(desc)
    }

    fn create_compute_pipeline(&mut self, desc: &ComputePipelineDesc<'_>) -> ComputePipelineId {
        self.inner.create_compute_pipeline(desc)
    }

    fn pipeline_ready(&mut self, id: PipelineId) -> bool {
        self.inner.pipeline_ready(id)
    }

    fn poll_pipelines(&mut self) -> PipelineCounts {
        self.inner.poll_pipelines()
    }

    fn prioritize_pipeline(&mut self, id: PipelineId) {
        self.inner.prioritize_pipeline(id);
    }

    fn create_render_bundle(
        &mut self,
        desc: &RenderBundleDesc<'_>,
        commands: &[RenderCommand],
    ) -> RenderBundleId {
        self.inner.create_render_bundle(desc, commands)
    }

    fn destroy_render_bundle(&mut self, id: RenderBundleId) {
        self.inner.destroy_render_bundle(id);
    }

    fn create_query_set(&mut self, desc: &QuerySetDesc<'_>) -> QuerySetId {
        self.inner.create_query_set(desc)
    }

    fn destroy_query_set(&mut self, id: QuerySetId) {
        self.inner.destroy_query_set(id);
    }

    /// Adds timestamps to every compute and render pass, if timing them,
    /// then submits.
    fn submit(&mut self, passes: &[Pass]) {
        let entered = clock::now();
        let mut timed = Vec::with_capacity(passes.len());
        let mut queries = Vec::new();
        for pass in passes {
            let mut pass = pass.clone();
            match &mut pass {
                Pass::Compute(p) => {
                    if let Some(t) = self.timestamps(&p.label, &mut queries) {
                        p.timestamps = Some(t);
                    }
                }
                Pass::Render(p) => {
                    if let Some(t) = self.timestamps(&p.label, &mut queries) {
                        p.timestamps = Some(t);
                    }
                }
                Pass::ClearBuffer { .. }
                | Pass::CopyBuffer { .. }
                | Pass::CopyTextureLayers { .. }
                | Pass::ResolveQuerySet { .. } => {}
            }
            timed.push(pass);
        }
        let started = clock::now();
        self.inner.submit(&timed);
        let cpu = clock::now() - started;
        drop(timed);
        if let Some(c) = self.capture.as_mut().filter(|c| c.next < c.slots) {
            let device = self.inner.device();
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(
                self.inner.buffer(c.source),
                0,
                self.inner.buffer(c.ring),
                c.next * c.size,
                c.size,
            );
            self.inner.queue().submit([encoder.finish()]);
            c.next += 1;
        }
        self.submissions.push(Submission {
            passes: queries,
            cpu,
            overhead: clock::now() - entered - cpu,
        });
    }
}
