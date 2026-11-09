//! [`orb_hal::Gpu`] on wgpu: browser WebGPU, Vulkan, Metal and D3D12, chosen
//! at runtime.
//!
//! [`WgpuGpu`] owns every wgpu handle behind the ids it hands out. Hosts
//! register their own targets (a surface frame, an offscreen texture) with
//! [`WgpuGpu::register_view`].

// `unsafe` only where a surface is made from a raw platform handle.
#![deny(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
mod cache;
mod compile;
mod convert;
mod offscreen;
#[cfg(not(target_arch = "wasm32"))]
mod surface;
mod timing;
mod writes;

pub use convert::from_texture_format;
use orb_hal::slots::Slots;
use orb_hal::*;
#[cfg(not(target_arch = "wasm32"))]
pub use surface::{RawSurface, Surface, SurfaceError};
pub use timing::SubmitTimes;
/// The wgpu this HAL is built on, for hosts that hand it windows.
pub use wgpu;

/// The wgpu device and queue, and every resource created through them.
#[derive(Debug)]
pub struct WgpuGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The instance and adapter behind the device, when it was made here,
    /// for surfaces.
    #[cfg(not(target_arch = "wasm32"))]
    native: Option<(wgpu::Instance, wgpu::Adapter)>,
    /// Pipelines are made through it once started (natively, Vulkan).
    #[cfg(not(target_arch = "wasm32"))]
    pipeline_cache: Option<wgpu::PipelineCache>,
    caps: Caps,
    /// Texture usages the platform defines. Browsers before
    /// `GPUTextureUsage.TRANSIENT_ATTACHMENT` (Samsung Internet 30,
    /// Chromium 143) reject it, so that hint is dropped there.
    texture_usages: wgpu::TextureUsages,
    buffers: Slots<wgpu::Buffer>,
    textures: Slots<Texture>,
    shader_modules: Slots<compile::Shared<compile::Module>>,
    bind_group_layouts: Slots<BindGroupLayout>,
    bind_groups: Slots<wgpu::BindGroup>,
    /// `None` while compiling.
    render_pipelines: Slots<Option<wgpu::RenderPipeline>>,
    compute_pipelines: Slots<Option<wgpu::ComputePipeline>>,
    compiler: compile::Compiler,
    /// Pipelines asked for and not yet compiled.
    pending: u32,
    /// A compile that ran past the watchdog's limit, or failed: the device
    /// is lost.
    hung: Option<String>,
    /// Each pipeline compiled, how long it took, in the order they
    /// finished.
    compiled: Vec<(String, std::time::Duration)>,
    render_bundles: Slots<wgpu::RenderBundle>,
    /// Submissions' times, while timing ([`WgpuGpu::set_timing`]).
    timing: Option<timing::Timing>,
    /// The last submission, and each unfinished frame's
    /// ([`WgpuGpu::end_frame`]).
    last: Option<wgpu::SubmissionIndex>,
    frames: std::collections::VecDeque<wgpu::SubmissionIndex>,
    samplers: Slots<wgpu::Sampler>,
    host_bytes: Slots<HostBytes>,
    /// `ImageBitmap`s and video elements for
    /// [`Gpu::write_texture_from_host`], copied with
    /// `GPUQueue.copyExternalImageToTexture`. Native hosts decode into core
    /// memory and register none.
    #[cfg(target_arch = "wasm32")]
    host_images: Slots<wgpu::ExternalImageSource>,
    uploaded: Uploaded,
    /// Small buffer writes, copied in at the next submission.
    writes: writes::Writes,
    /// Readback buffers back from their last readback, for the next of
    /// their size: a new one maps fresh pages each time.
    readbacks: offscreen::Readbacks,
    /// The texture the next submission copies out after its passes, and
    /// the copy the last one made, for the readback after it.
    copy_next: Option<TextureId>,
    copied: std::sync::Mutex<Option<(TextureId, offscreen::Readback)>>,
    pipelines_created: u32,
    /// Set when the device made by `offscreen` is lost.
    #[cfg(not(target_arch = "wasm32"))]
    lost: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Bytes written to buffers and textures since the device was created, by
/// where they came from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Uploaded {
    /// From Rust memory: [`Gpu::write_buffer`] and host bytes registered
    /// with [`WgpuGpu::register_bytes`].
    pub from_core: u64,
    /// From JS `ArrayBuffer`s registered with
    /// `WgpuGpu::register_array_buffer`, never copied into wasm memory.
    pub from_js: u64,
    /// Texel bytes written to textures from Rust memory.
    pub texels_from_core: u64,
    /// Texel bytes copied to textures from JS images registered with
    /// `WgpuGpu::register_image_bitmap` or `register_video`.
    pub texels_from_js: u64,
}

/// A write from core memory: gathered when small, or when it overlaps a
/// gathered one it must land after; otherwise straight to the queue.
fn write_core(
    writes: &mut writes::Writes,
    queue: &wgpu::Queue,
    buffers: &Slots<wgpu::Buffer>,
    id: BufferId,
    offset: u64,
    data: &[u8],
) {
    if data.len() <= writes::GATHERED || writes.overlaps(id, offset, data.len() as u64) {
        writes.gather(id, offset, data);
    } else {
        queue.write_buffer(&buffers[id.0], offset, data);
    }
}

/// Bytes held for [`Gpu::write_buffer_from_host`].
#[derive(Debug)]
enum HostBytes {
    /// A JS `ArrayBuffer`, written with `GPUQueue.writeBuffer` without
    /// entering wasm memory.
    #[cfg(target_arch = "wasm32")]
    Js(js_sys::ArrayBuffer),
    /// Bytes in Rust memory, for native hosts and tests.
    Owned(Vec<u8>),
}

#[derive(Debug)]
struct BindGroupLayout {
    layout: wgpu::BindGroupLayout,
    /// The browser's own, for async compiles.
    #[cfg(target_arch = "wasm32")]
    raw: wasm_bindgen::JsValue,
}

#[derive(Debug)]
struct Texture {
    /// `None` for views the host registered and owns.
    texture: Option<wgpu::Texture>,
    view: wgpu::TextureView,
    /// A view of these mip levels of another entry's texture, which it
    /// doesn't own.
    levels: Option<std::ops::Range<u32>>,
}

#[cfg(not(target_arch = "wasm32"))]
fn platform_texture_usages() -> wgpu::TextureUsages {
    wgpu::TextureUsages::all()
}

#[cfg(target_arch = "wasm32")]
fn platform_texture_usages() -> wgpu::TextureUsages {
    let defined = js_sys::Reflect::get(&js_sys::global(), &"GPUTextureUsage".into())
        .and_then(|usage| js_sys::Reflect::has(&usage, &"TRANSIENT_ATTACHMENT".into()))
        .unwrap_or(false);
    if defined {
        wgpu::TextureUsages::all()
    } else {
        wgpu::TextureUsages::all() - wgpu::TextureUsages::TRANSIENT_ATTACHMENT
    }
}

impl WgpuGpu {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let limits = device.limits();
        let caps = Caps {
            limits: Limits {
                max_buffer_size: limits.max_buffer_size,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
                max_texture_dimension_2d: limits.max_texture_dimension_2d,
                max_texture_array_layers: limits.max_texture_array_layers,
                min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
                min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
            },
            timestamp_query: device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            clip_distances: device.features().contains(wgpu::Features::CLIP_DISTANCES),
        };
        Self {
            device,
            queue,
            #[cfg(not(target_arch = "wasm32"))]
            native: None,
            #[cfg(not(target_arch = "wasm32"))]
            pipeline_cache: None,
            caps,
            texture_usages: platform_texture_usages(),
            buffers: Slots::default(),
            textures: Slots::default(),
            shader_modules: Slots::default(),
            bind_group_layouts: Slots::default(),
            bind_groups: Slots::default(),
            render_pipelines: Slots::default(),
            compute_pipelines: Slots::default(),
            compiler: compile::Compiler::new(),
            pending: 0,
            hung: None,
            compiled: Vec::new(),
            render_bundles: Slots::default(),
            timing: None,
            last: None,
            frames: Default::default(),
            samplers: Slots::default(),
            host_bytes: Slots::default(),
            #[cfg(target_arch = "wasm32")]
            host_images: Slots::default(),
            uploaded: Uploaded::default(),
            writes: writes::Writes::default(),
            readbacks: Default::default(),
            copy_next: None,
            copied: Default::default(),
            pipelines_created: 0,
            #[cfg(not(target_arch = "wasm32"))]
            lost: Default::default(),
        }
    }

    /// Copies the gathered writes into place ahead of what `encoder`
    /// records next.
    fn encode_writes(&self, encoder: &mut wgpu::CommandEncoder) {
        self.writes.encode(
            &self.device,
            &self.queue,
            |id| self.buffers[id.0].clone(),
            encoder,
        );
    }

    /// Bytes written to buffers so far.
    pub fn uploaded(&self) -> Uploaded {
        self.uploaded
    }

    /// Render and compute pipelines compiled so far.
    pub fn pipelines_created(&self) -> u32 {
        self.pipelines_created
    }

    /// Each pipeline compiled so far and how long it took, in the order
    /// they finished: natively the worker's time in wgpu and the driver;
    /// in the browser from the request to its promise resolving.
    pub fn pipeline_timings(&self) -> &[(String, std::time::Duration)] {
        &self.compiled
    }

    /// Compiles every later pipeline on `executor`, the host's job threads,
    /// or (`None`, the default) on the calling thread, so it is ready when
    /// [`Gpu::create_render_pipeline`] returns.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_compile_executor(
        &mut self,
        executor: Option<std::sync::Arc<dyn orb_host::jobs::Executor>>,
    ) {
        self.compiler.set_executor(executor);
    }

    /// Blocks until every pipeline asked for so far compiled: for tools
    /// and tests that want a complete frame now.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn finish_pipelines(&mut self) {
        self.compiler.wait(self.pending);
        self.poll_pipelines();
    }

    /// Calls `wake` from a compile thread whenever a pipeline compiles, so
    /// a host that stopped drawing draws the layers that waited for it.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_compile_wake(&self, wake: std::sync::Arc<dyn Fn() + Send + Sync>) {
        self.compiler.set_wake(wake);
    }

    /// Calls `wake` whenever a pipeline's `create*PipelineAsync` promise
    /// settles, so a host that stopped drawing draws the layers that waited
    /// for it.
    #[cfg(target_arch = "wasm32")]
    pub fn set_compile_wake(&self, wake: std::rc::Rc<dyn Fn()>) {
        self.compiler.set_wake(wake);
    }

    /// Makes later compiles stall forever, as in a driver hung in its
    /// compiler: fault injection for the watchdog's tests. Needs compile
    /// threads.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn inject_compile_hang(&self) {
        self.compiler.inject_hang();
    }

    /// How long a compile may run before the driver is taken to have hung
    /// (60 s by default).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_compile_watchdog(&self, limit: std::time::Duration) {
        self.compiler.set_watchdog(limit);
    }

    /// The compile that ran past the watchdog's limit, or failed, if one
    /// did: the
    /// driver is taken to have hung, and the device is lost.
    pub fn hung(&self) -> Option<&str> {
        self.hung.as_deref()
    }

    fn take_compiled(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        let done: Vec<_> = self.compiler.take().collect();
        #[cfg(target_arch = "wasm32")]
        let done: Vec<_> = self.compiler.take(&self.device).collect();
        for done in done {
            self.pending -= 1;
            let built = match done.built {
                Ok(built) => built,
                // A pipeline the driver can't make goes the way of a hung
                // compile: the device is lost, and recovery demotes it.
                Err(e) => {
                    self.hung
                        .get_or_insert_with(|| format!("pipeline {}: {e}", done.label));
                    #[cfg(not(target_arch = "wasm32"))]
                    self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
            };
            // A pipeline destroyed while it compiled is dropped.
            match built {
                compile::Built::Render(slot, p) if self.render_pipelines.contains(slot) => {
                    self.render_pipelines.set(slot, Some(p));
                }
                compile::Built::Compute(slot, p) if self.compute_pipelines.contains(slot) => {
                    self.compute_pipelines.set(slot, Some(p));
                }
                _ => {}
            }
            self.pipelines_created += 1;
            self.compiled.push((done.label, done.took));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.hung.is_none()
            && self.pending > 0
            && let Some(label) = self.compiler.hung()
        {
            self.hung = Some(label);
            self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn submit_job(&mut self, job: compile::Job, layouts: &[BindGroupLayoutId]) {
        self.pending += 1;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = layouts;
            self.compiler.submit(&self.device, job);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let raw = layouts
                .iter()
                .map(|id| self.bind_group_layouts[id.0].raw.clone())
                .collect();
            self.compiler.submit(&self.device, raw, job);
        }
    }

    /// The pipeline cache pipelines are made through, if any.
    fn cache(&self) -> Option<wgpu::PipelineCache> {
        #[cfg(not(target_arch = "wasm32"))]
        return self.pipeline_cache.clone();
        #[cfg(target_arch = "wasm32")]
        None
    }

    /// Time spent compiling pipelines so far, summed over the threads
    /// that compiled them (natively wgpu translates and the driver
    /// compiles there).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pipeline_time(&self) -> std::time::Duration {
        self.compiled.iter().map(|(_, took)| *took).sum()
    }

    /// Whether the device `offscreen` made was lost.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn lost(&self) -> bool {
        self.lost.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Loses the device on purpose (`destroy`), to exercise recovery.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn lose(&self) {
        self.lost.store(true, std::sync::atomic::Ordering::Relaxed);
        self.device.destroy();
    }

    /// The adapter behind a device `offscreen` made.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn adapter_info(&self) -> Option<wgpu::AdapterInfo> {
        self.native.as_ref().map(|(_, adapter)| adapter.get_info())
    }

    /// Registers bytes in Rust memory for [`Gpu::write_buffer_from_host`].
    pub fn register_bytes(&mut self, bytes: Vec<u8>) -> HostBytesId {
        HostBytesId(self.host_bytes.insert(HostBytes::Owned(bytes)))
    }

    /// Registers a JS `ArrayBuffer` for [`Gpu::write_buffer_from_host`].
    #[cfg(target_arch = "wasm32")]
    pub fn register_array_buffer(&mut self, buffer: js_sys::ArrayBuffer) -> HostBytesId {
        HostBytesId(self.host_bytes.insert(HostBytes::Js(buffer)))
    }

    /// Registers a JS `ImageBitmap` for [`Gpu::write_texture_from_host`].
    #[cfg(target_arch = "wasm32")]
    pub fn register_image_bitmap(&mut self, bitmap: web_sys::ImageBitmap) -> HostImageId {
        HostImageId(
            self.host_images
                .insert(wgpu::ExternalImageSource::ImageBitmap(bitmap)),
        )
    }

    /// Registers a video element, whose frame at the time of the write
    /// [`Gpu::write_texture_from_host`] copies.
    #[cfg(target_arch = "wasm32")]
    pub fn register_video(&mut self, video: web_sys::HtmlVideoElement) -> HostImageId {
        HostImageId(
            self.host_images
                .insert(wgpu::ExternalImageSource::HTMLVideoElement(video)),
        )
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Registers a texture view the host owns, such as the current surface
    /// frame, as a render target. Release it with
    /// [`Gpu::destroy_texture`] before the view is dropped or presented.
    pub fn register_view(&mut self, view: wgpu::TextureView) -> TextureId {
        TextureId(self.textures.insert(Texture {
            texture: None,
            view,
            levels: None,
        }))
    }

    /// The wgpu texture behind an id created with [`Gpu::create_texture`].
    pub fn texture(&self, id: TextureId) -> &wgpu::Texture {
        self.textures[id.0]
            .texture
            .as_ref()
            .expect("texture is a registered view")
    }

    /// The wgpu buffer behind an id.
    pub fn buffer(&self, id: BufferId) -> &wgpu::Buffer {
        &self.buffers[id.0]
    }

    /// Ends a frame drawn into a texture: waits until no more than
    /// `in_flight` frames are unfinished, as a swapchain does.
    pub fn end_frame(&mut self, in_flight: usize) {
        self.frames.extend(self.last.take());
        while self.frames.len() > in_flight {
            let index = self.frames.pop_front();
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: index,
                timeout: None,
            });
        }
    }

    /// Nanoseconds per timestamp tick in resolved query sets.
    pub(crate) fn timestamp_period_ns(&self) -> f32 {
        self.queue.get_timestamp_period()
    }

    fn pipeline_layout(&self, label: &str, layouts: &[BindGroupLayoutId]) -> wgpu::PipelineLayout {
        let layouts: Vec<_> = layouts
            .iter()
            .map(|id| Some(&self.bind_group_layouts[id.0].layout))
            .collect();
        self.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &layouts,
                immediate_size: 0,
            })
    }

    fn encode_render<'p>(&'p self, pass: &mut impl RenderEncoder<'p>, command: &'p RenderCommand) {
        match command {
            RenderCommand::SetPipeline(id) => pass.set_pipeline(
                self.render_pipelines[id.0]
                    .as_ref()
                    .expect("a pipeline used before it compiled"),
            ),
            RenderCommand::SetBindGroup { index, group } => {
                pass.set_bind_group(*index, &self.bind_groups[group.0])
            }
            RenderCommand::Draw {
                vertices,
                instances,
            } => pass.draw(vertices.clone(), instances.clone()),
            RenderCommand::SetIndexBuffer {
                buffer,
                format,
                offset,
            } => pass.set_index_buffer(
                self.buffers[buffer.0].slice(*offset..),
                match format {
                    IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
                    IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
                },
            ),
            RenderCommand::DrawIndexed {
                indices,
                base_vertex,
                instances,
            } => pass.draw_indexed(indices.clone(), *base_vertex, instances.clone()),
            RenderCommand::DrawIndirect { buffer, offset } => {
                pass.draw_indirect(&self.buffers[buffer.0], *offset)
            }
            RenderCommand::DrawIndexedIndirect { buffer, offset } => {
                pass.draw_indexed_indirect(&self.buffers[buffer.0], *offset)
            }
            RenderCommand::ExecuteBundles(ids) => {
                pass.execute_bundles(ids.iter().map(|id| &self.render_bundles[id.0]))
            }
            &RenderCommand::SetScissor([x, y, width, height]) => {
                pass.set_scissor(x, y, width, height)
            }
        }
    }
}

/// The commands render passes and render bundle encoders share.
trait RenderEncoder<'p> {
    fn set_pipeline(&mut self, pipeline: &'p wgpu::RenderPipeline);
    fn set_bind_group(&mut self, index: u32, group: &'p wgpu::BindGroup);
    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>);
    fn set_index_buffer(&mut self, slice: wgpu::BufferSlice<'p>, format: wgpu::IndexFormat);
    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    );
    fn draw_indirect(&mut self, buffer: &'p wgpu::Buffer, offset: u64);
    fn draw_indexed_indirect(&mut self, buffer: &'p wgpu::Buffer, offset: u64);
    fn execute_bundles(&mut self, bundles: impl Iterator<Item = &'p wgpu::RenderBundle>);
    fn set_scissor(&mut self, x: u32, y: u32, width: u32, height: u32);
}

impl<'p> RenderEncoder<'p> for wgpu::RenderPass<'_> {
    fn set_pipeline(&mut self, pipeline: &'p wgpu::RenderPipeline) {
        wgpu::RenderPass::set_pipeline(self, pipeline);
    }
    fn set_bind_group(&mut self, index: u32, group: &'p wgpu::BindGroup) {
        wgpu::RenderPass::set_bind_group(self, index, group, &[]);
    }
    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        wgpu::RenderPass::draw(self, vertices, instances);
    }
    fn set_index_buffer(&mut self, slice: wgpu::BufferSlice<'p>, format: wgpu::IndexFormat) {
        wgpu::RenderPass::set_index_buffer(self, slice, format);
    }
    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        wgpu::RenderPass::draw_indexed(self, indices, base_vertex, instances);
    }
    fn draw_indirect(&mut self, buffer: &'p wgpu::Buffer, offset: u64) {
        wgpu::RenderPass::draw_indirect(self, buffer, offset);
    }
    fn draw_indexed_indirect(&mut self, buffer: &'p wgpu::Buffer, offset: u64) {
        wgpu::RenderPass::draw_indexed_indirect(self, buffer, offset);
    }
    fn execute_bundles(&mut self, bundles: impl Iterator<Item = &'p wgpu::RenderBundle>) {
        wgpu::RenderPass::execute_bundles(self, bundles);
    }
    fn set_scissor(&mut self, x: u32, y: u32, width: u32, height: u32) {
        wgpu::RenderPass::set_scissor_rect(self, x, y, width, height);
    }
}

impl<'p> RenderEncoder<'p> for wgpu::RenderBundleEncoder<'p> {
    fn set_pipeline(&mut self, pipeline: &'p wgpu::RenderPipeline) {
        wgpu::RenderBundleEncoder::set_pipeline(self, pipeline);
    }
    fn set_bind_group(&mut self, index: u32, group: &'p wgpu::BindGroup) {
        wgpu::RenderBundleEncoder::set_bind_group(self, index, group, &[]);
    }
    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        wgpu::RenderBundleEncoder::draw(self, vertices, instances);
    }
    fn set_index_buffer(&mut self, slice: wgpu::BufferSlice<'p>, format: wgpu::IndexFormat) {
        wgpu::RenderBundleEncoder::set_index_buffer(self, slice, format);
    }
    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        wgpu::RenderBundleEncoder::draw_indexed(self, indices, base_vertex, instances);
    }
    fn draw_indirect(&mut self, buffer: &'p wgpu::Buffer, offset: u64) {
        wgpu::RenderBundleEncoder::draw_indirect(self, buffer, offset);
    }
    fn draw_indexed_indirect(&mut self, buffer: &'p wgpu::Buffer, offset: u64) {
        wgpu::RenderBundleEncoder::draw_indexed_indirect(self, buffer, offset);
    }
    fn execute_bundles(&mut self, _: impl Iterator<Item = &'p wgpu::RenderBundle>) {
        panic!("render bundles cannot execute bundles");
    }
    fn set_scissor(&mut self, _: u32, _: u32, _: u32, _: u32) {
        panic!("render bundles cannot set a scissor");
    }
}

impl Gpu for WgpuGpu {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> BufferId {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(desc.label),
            size: desc.size,
            usage: convert::buffer_usages(desc.usage),
            mapped_at_creation: false,
        });
        BufferId(self.buffers.insert(buffer))
    }

    fn destroy_buffer(&mut self, id: BufferId) {
        self.writes.forget(id);
        self.buffers.remove(id.0).destroy();
    }

    fn write_buffer(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        self.uploaded.from_core += data.len() as u64;
        write_core(
            &mut self.writes,
            &self.queue,
            &self.buffers,
            id,
            offset,
            data,
        );
    }

    fn write_buffer_from_host(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: HostBytesId,
        range: std::ops::Range<u64>,
    ) {
        let len = range.end - range.start;
        match &self.host_bytes[bytes.0] {
            HostBytes::Owned(data) => {
                self.uploaded.from_core += len;
                let range = range.start as usize..range.end as usize;
                write_core(
                    &mut self.writes,
                    &self.queue,
                    &self.buffers,
                    id,
                    offset,
                    &data[range],
                );
            }
            #[cfg(target_arch = "wasm32")]
            HostBytes::Js(data) => {
                self.uploaded.from_js += len;
                // It lands before gathered writes: those it overlaps go first.
                if self.writes.overlaps(id, offset, len) {
                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    self.encode_writes(&mut encoder);
                    self.queue.submit([encoder.finish()]);
                }
                let queue = self
                    .queue
                    .as_webgpu()
                    .expect("the web uses the WebGPU backend");
                let buffer = self.buffers[id.0]
                    .as_webgpu()
                    .expect("the web uses the WebGPU backend");
                queue
                    .write_buffer_with_f64_and_buffer_source_and_f64_and_f64(
                        buffer,
                        offset as f64,
                        data,
                        range.start as f64,
                        len as f64,
                    )
                    .expect("writeBuffer from a host ArrayBuffer");
            }
        }
    }

    fn read_host_bytes(&self, bytes: HostBytesId, range: std::ops::Range<u64>) -> Vec<u8> {
        match &self.host_bytes[bytes.0] {
            HostBytes::Owned(data) => data[range.start as usize..range.end as usize].to_vec(),
            #[cfg(target_arch = "wasm32")]
            HostBytes::Js(data) => js_sys::Uint8Array::new_with_byte_offset_and_length(
                data,
                range.start as u32,
                (range.end - range.start) as u32,
            )
            .to_vec(),
        }
    }

    fn release_host_bytes(&mut self, bytes: HostBytesId) {
        self.host_bytes.remove(bytes.0);
    }

    fn create_texture(&mut self, desc: &TextureDesc<'_>) -> TextureId {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(desc.label),
            size: wgpu::Extent3d {
                width: desc.width,
                height: desc.height,
                depth_or_array_layers: desc.array_layers,
            },
            mip_level_count: desc.mip_levels,
            sample_count: desc.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: convert::texture_format(desc.format),
            usage: convert::texture_usages(desc.usage) & self.texture_usages,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        TextureId(self.textures.insert(Texture {
            texture: Some(texture),
            view,
            levels: None,
        }))
    }

    fn create_texture_view(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
    ) -> TextureId {
        let texture = self.texture(texture).clone();
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: levels.start,
            mip_level_count: Some(levels.len() as u32),
            ..Default::default()
        });
        TextureId(self.textures.insert(Texture {
            texture: Some(texture),
            view,
            levels: Some(levels),
        }))
    }

    fn create_layer_view(&mut self, texture: TextureId, layer: u32) -> TextureId {
        let texture = self.texture(texture).clone();
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        TextureId(self.textures.insert(Texture {
            texture: Some(texture),
            view,
            levels: Some(0..1),
        }))
    }

    fn destroy_texture(&mut self, id: TextureId) {
        let entry = self.textures.remove(id.0);
        if let (Some(texture), None) = (entry.texture, entry.levels) {
            texture.destroy();
        }
    }

    fn write_texture(&mut self, region: &TextureRegion, data: &[u8]) {
        self.uploaded.texels_from_core += data.len() as u64;
        let texture = self.texture(region.texture);
        let bytes_per_row = region.width * convert::texel_bytes(texture.format());
        assert_eq!(
            data.len(),
            (bytes_per_row * region.height) as usize,
            "texture write size mismatch"
        );
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: region.level,
                origin: wgpu::Origin3d {
                    x: region.x,
                    y: region.y,
                    z: region.layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: region.width,
                height: region.height,
                depth_or_array_layers: 1,
            },
        );
    }

    #[cfg(target_arch = "wasm32")]
    fn write_texture_from_host(
        &mut self,
        region: &TextureRegion,
        image: HostImageId,
        premultiply: bool,
    ) {
        self.uploaded.texels_from_js += u64::from(region.width * region.height * 4);
        self.queue.copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                source: self.host_images[image.0].clone(),
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture: self.texture(region.texture),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: region.x,
                    y: region.y,
                    z: region.layer,
                },
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: premultiply,
            },
            wgpu::Extent3d {
                width: region.width,
                height: region.height,
                depth_or_array_layers: 1,
            },
        );
    }

    #[cfg(target_arch = "wasm32")]
    fn release_host_image(&mut self, image: HostImageId) {
        if let wgpu::ExternalImageSource::ImageBitmap(bitmap) = self.host_images.remove(image.0) {
            bitmap.close();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn write_texture_from_host(&mut self, _: &TextureRegion, image: HostImageId, _: bool) {
        unreachable!("{image:?}: native hosts register no images");
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn release_host_image(&mut self, image: HostImageId) {
        unreachable!("{image:?}: native hosts register no images");
    }

    fn create_sampler(&mut self, desc: &SamplerDesc<'_>) -> SamplerId {
        let mut descriptor = convert::sampler(desc);
        descriptor.label = Some(desc.label);
        SamplerId(
            self.samplers
                .insert(self.device.create_sampler(&descriptor)),
        )
    }

    fn destroy_sampler(&mut self, id: SamplerId) {
        self.samplers.remove(id.0);
    }

    fn destroy_shader_module(&mut self, id: ShaderModuleId) {
        self.shader_modules.remove(id.0);
    }

    fn destroy_bind_group_layout(&mut self, id: BindGroupLayoutId) {
        self.bind_group_layouts.remove(id.0);
    }

    fn destroy_pipeline(&mut self, id: PipelineId) {
        match id {
            PipelineId::Render(id) => drop(self.render_pipelines.remove(id.0)),
            PipelineId::Compute(id) => drop(self.compute_pipelines.remove(id.0)),
        }
    }

    fn live_resources(&self) -> usize {
        let counts = [
            self.buffers.len(),
            self.textures.len(),
            self.shader_modules.len(),
            self.bind_group_layouts.len(),
            self.bind_groups.len(),
            self.render_pipelines.len(),
            self.compute_pipelines.len(),
            self.render_bundles.len(),
            self.samplers.len(),
        ];
        counts.iter().sum()
    }

    fn create_shader_module(&mut self, desc: &ShaderModuleDesc<'_>) -> ShaderModuleId {
        let source = desc.source.to_string();
        ShaderModuleId(
            self.shader_modules
                .insert(compile::Module::new(desc.label, source, desc.ir)),
        )
    }

    fn create_bind_group_layout(&mut self, desc: &BindGroupLayoutDesc<'_>) -> BindGroupLayoutId {
        let entries: Vec<_> = desc
            .entries
            .iter()
            .map(convert::bind_group_layout_entry)
            .collect();
        let layout = BindGroupLayout {
            layout: self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(desc.label),
                    entries: &entries,
                }),
            #[cfg(target_arch = "wasm32")]
            raw: compile::raw_bind_group_layout(
                self.device
                    .as_webgpu()
                    .expect("the web uses the WebGPU backend")
                    .as_ref(),
                desc,
            ),
        };
        BindGroupLayoutId(self.bind_group_layouts.insert(layout))
    }

    fn create_bind_group(&mut self, desc: &BindGroupDesc<'_>) -> BindGroupId {
        // Texture views live until the bind group is created; wgpu keeps
        // what it needs.
        let views: Vec<Option<wgpu::TextureView>> = desc
            .entries
            .iter()
            .map(|entry| match entry.resource {
                BindingResource::Texture { texture, dimension } => {
                    let levels = self.textures[texture.0].levels.clone();
                    Some(
                        self.texture(texture)
                            .create_view(&wgpu::TextureViewDescriptor {
                                dimension: Some(convert::view_dimension(dimension)),
                                base_mip_level: levels.as_ref().map_or(0, |l| l.start),
                                mip_level_count: levels.map(|l| l.len() as u32),
                                ..Default::default()
                            }),
                    )
                }
                _ => None,
            })
            .collect();
        let entries: Vec<_> = desc
            .entries
            .iter()
            .zip(&views)
            .map(|(entry, view)| wgpu::BindGroupEntry {
                binding: entry.binding,
                resource: match entry.resource {
                    BindingResource::Buffer {
                        buffer,
                        offset,
                        size,
                    } => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.buffers[buffer.0],
                        offset,
                        size: size.map(|s| s.try_into().expect("binding size must be non-zero")),
                    }),
                    BindingResource::Texture { .. } => {
                        wgpu::BindingResource::TextureView(view.as_ref().expect("created above"))
                    }
                    BindingResource::Sampler(id) => {
                        wgpu::BindingResource::Sampler(&self.samplers[id.0])
                    }
                },
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(desc.label),
            layout: &self.bind_group_layouts[desc.layout.0].layout,
            entries: &entries,
        });
        BindGroupId(self.bind_groups.insert(group))
    }

    fn destroy_bind_group(&mut self, id: BindGroupId) {
        self.bind_groups.remove(id.0);
    }

    fn create_render_pipeline(&mut self, desc: &RenderPipelineDesc<'_>) -> RenderPipelineId {
        let module = |stage: &ShaderStage<'_>| self.shader_modules[stage.module.0].clone();
        let job = compile::RenderJob {
            label: desc.label.to_owned(),
            layout: self.pipeline_layout(desc.label, desc.bind_group_layouts),
            vertex: compile::Stage::new(&desc.vertex, module(&desc.vertex)),
            fragment: compile::Stage::new(&desc.fragment, module(&desc.fragment)),
            targets: desc.targets.to_vec(),
            depth: desc.depth,
            topology: desc.topology,
            cull_mode: desc.cull_mode,
            sample_count: desc.sample_count,
            cache: self.cache(),
        };
        let slot = self.render_pipelines.insert(None);
        self.submit_job(compile::Job::Render(slot, job), desc.bind_group_layouts);
        RenderPipelineId(slot)
    }

    fn create_compute_pipeline(&mut self, desc: &ComputePipelineDesc<'_>) -> ComputePipelineId {
        let job = compile::ComputeJob {
            label: desc.label.to_owned(),
            layout: self.pipeline_layout(desc.label, desc.bind_group_layouts),
            compute: compile::Stage::new(
                &desc.compute,
                self.shader_modules[desc.compute.module.0].clone(),
            ),
            cache: self.cache(),
        };
        let slot = self.compute_pipelines.insert(None);
        self.submit_job(compile::Job::Compute(slot, job), desc.bind_group_layouts);
        ComputePipelineId(slot)
    }

    fn pipeline_ready(&mut self, id: PipelineId) -> bool {
        let ready = |gpu: &Self| match id {
            PipelineId::Render(id) => gpu.render_pipelines[id.0].is_some(),
            PipelineId::Compute(id) => gpu.compute_pipelines[id.0].is_some(),
        };
        if ready(self) {
            return true;
        }
        self.take_compiled();
        ready(self)
    }

    fn poll_pipelines(&mut self) -> PipelineCounts {
        self.take_compiled();
        PipelineCounts {
            made: self.pipelines_created,
            pending: self.pending,
        }
    }

    fn prioritize_pipeline(&mut self, id: PipelineId) {
        self.compiler.prioritize(id);
    }

    fn create_render_bundle(
        &mut self,
        desc: &RenderBundleDesc<'_>,
        commands: &[RenderCommand],
    ) -> RenderBundleId {
        let color_formats: Vec<_> = desc
            .color_formats
            .iter()
            .map(|f| Some(convert::texture_format(*f)))
            .collect();
        let mut encoder =
            self.device
                .create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                    label: Some(desc.label),
                    color_formats: &color_formats,
                    depth_stencil: desc.depth_format.map(|f| wgpu::RenderBundleDepthStencil {
                        format: convert::texture_format(f),
                        depth_read_only: false,
                        stencil_read_only: true,
                    }),
                    sample_count: desc.sample_count,
                    multiview: None,
                });
        for command in commands {
            self.encode_render(&mut encoder, command);
        }
        let bundle = encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some(desc.label),
        });
        RenderBundleId(self.render_bundles.insert(bundle))
    }

    fn destroy_render_bundle(&mut self, id: RenderBundleId) {
        self.render_bundles.remove(id.0);
    }

    fn submit(&mut self, passes: &[Pass]) {
        let started = timing::now();
        // While timing, each compute and render pass's first query.
        let queries = match &mut self.timing {
            Some(timing) => timing.begin(passes.iter().map(|pass| match pass {
                Pass::Compute(p) => Some(p.label.as_str()),
                Pass::Render(p) => Some(p.label.as_str()),
                _ => None,
            })),
            None => Vec::new(),
        };
        let set = self.timing.as_ref().and_then(timing::Timing::set);
        let writes = |i: usize| Some((set?, (*queries.get(i)?)?));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_writes(&mut encoder);
        for (i, pass) in passes.iter().enumerate() {
            match pass {
                Pass::CopyTexture {
                    source,
                    destination,
                    copies,
                } => {
                    let source = self.texture(*source);
                    let destination = self.texture(*destination);
                    let levels = source.mip_level_count().min(destination.mip_level_count());
                    for (copy, level) in
                        copies.iter().flat_map(|c| (0..levels).map(move |l| (c, l)))
                    {
                        let at = |[x, y, z]: [u32; 3]| wgpu::Origin3d {
                            x: x >> level,
                            y: y >> level,
                            z,
                        };
                        let [w, h, layers] = copy.size;
                        let size = wgpu::Extent3d {
                            width: (w >> level).max(1),
                            height: (h >> level).max(1),
                            depth_or_array_layers: layers,
                        };
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                mip_level: level,
                                origin: at(copy.from),
                                ..source.as_image_copy()
                            },
                            wgpu::TexelCopyTextureInfo {
                                mip_level: level,
                                origin: at(copy.to),
                                ..destination.as_image_copy()
                            },
                            size,
                        );
                    }
                }
                Pass::ClearBuffer {
                    buffer,
                    offset,
                    size,
                } => {
                    encoder.clear_buffer(&self.buffers[buffer.0], *offset, *size);
                }
                Pass::CopyBuffer {
                    source,
                    source_offset,
                    destination,
                    destination_offset,
                    size,
                } => encoder.copy_buffer_to_buffer(
                    &self.buffers[source.0],
                    *source_offset,
                    &self.buffers[destination.0],
                    *destination_offset,
                    *size,
                ),
                Pass::Compute(compute) => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&compute.label),
                        timestamp_writes: writes(i).map(|(query_set, first)| {
                            wgpu::ComputePassTimestampWrites {
                                query_set,
                                beginning_of_pass_write_index: Some(first),
                                end_of_pass_write_index: Some(first + 1),
                            }
                        }),
                    });
                    for command in &compute.commands {
                        match command {
                            ComputeCommand::SetPipeline(id) => pass.set_pipeline(
                                self.compute_pipelines[id.0]
                                    .as_ref()
                                    .expect("a pipeline used before it compiled"),
                            ),
                            ComputeCommand::SetBindGroup { index, group } => {
                                pass.set_bind_group(*index, &self.bind_groups[group.0], &[]);
                            }
                            ComputeCommand::Dispatch { x, y, z } => {
                                pass.dispatch_workgroups(*x, *y, *z)
                            }
                        }
                    }
                }
                Pass::Render(render) => {
                    let colors: Vec<Option<wgpu::RenderPassColorAttachment>> = render
                        .colors
                        .iter()
                        .map(|color| {
                            Some(wgpu::RenderPassColorAttachment {
                                view: &self.textures[color.target.0].view,
                                depth_slice: None,
                                resolve_target: color.resolve.map(|id| &self.textures[id.0].view),
                                ops: wgpu::Operations {
                                    load: match color.load {
                                        LoadOp::Clear([r, g, b, a]) => {
                                            wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a })
                                        }
                                        LoadOp::Load => wgpu::LoadOp::Load,
                                    },
                                    store: match color.store {
                                        StoreOp::Store => wgpu::StoreOp::Store,
                                        StoreOp::Discard => wgpu::StoreOp::Discard,
                                    },
                                },
                            })
                        })
                        .collect();
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(&render.label),
                        color_attachments: &colors,
                        depth_stencil_attachment: render.depth.as_ref().map(|depth| {
                            wgpu::RenderPassDepthStencilAttachment {
                                view: &self.textures[depth.target.0].view,
                                depth_ops: Some(wgpu::Operations {
                                    load: match depth.load {
                                        LoadOp::Clear(v) => wgpu::LoadOp::Clear(v),
                                        LoadOp::Load => wgpu::LoadOp::Load,
                                    },
                                    store: match depth.store {
                                        StoreOp::Store => wgpu::StoreOp::Store,
                                        StoreOp::Discard => wgpu::StoreOp::Discard,
                                    },
                                }),
                                stencil_ops: None,
                            }
                        }),
                        timestamp_writes: writes(i).map(|(query_set, first)| {
                            wgpu::RenderPassTimestampWrites {
                                query_set,
                                beginning_of_pass_write_index: Some(first),
                                end_of_pass_write_index: Some(first + 1),
                            }
                        }),
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    for command in &render.commands {
                        self.encode_render(&mut pass, command);
                    }
                }
            }
        }
        self.encode_armed_copy(&mut encoder);
        self.last = Some(self.queue.submit([encoder.finish()]));
        if let Some(timing) = &mut self.timing {
            timing.end(timing::now() - started);
        }
    }
}
