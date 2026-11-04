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
mod convert;
mod offscreen;
mod slots;
#[cfg(not(target_arch = "wasm32"))]
mod surface;

pub use convert::from_texture_format;
use orb_hal::*;
use slots::Slots;
#[cfg(not(target_arch = "wasm32"))]
pub use surface::{RawSurface, Surface, SurfaceError};
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
    shader_modules: Slots<wgpu::ShaderModule>,
    bind_group_layouts: Slots<wgpu::BindGroupLayout>,
    bind_groups: Slots<wgpu::BindGroup>,
    render_pipelines: Slots<wgpu::RenderPipeline>,
    compute_pipelines: Slots<wgpu::ComputePipeline>,
    render_bundles: Slots<wgpu::RenderBundle>,
    query_sets: Slots<wgpu::QuerySet>,
    samplers: Slots<wgpu::Sampler>,
    host_bytes: Slots<HostBytes>,
    /// `ImageBitmap`s for [`Gpu::write_texture_from_host`], copied with
    /// `GPUQueue.copyExternalImageToTexture`. Native hosts decode into core
    /// memory and register none.
    #[cfg(target_arch = "wasm32")]
    host_images: Slots<web_sys::ImageBitmap>,
    uploaded: Uploaded,
    pipelines_created: u32,
    /// Time spent making shader modules and pipelines. Natively wgpu
    /// translates and the driver compiles there; browsers do it in the GPU
    /// process, out of sight.
    #[cfg(not(target_arch = "wasm32"))]
    pipeline_time: std::time::Duration,
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
    /// Texel bytes copied to textures from JS `ImageBitmap`s registered
    /// with `WgpuGpu::register_image_bitmap`.
    pub texels_from_js: u64,
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
            tier: Tier::Modern,
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
            rgba16float_blendable: {
                let format =
                    wgpu::TextureFormat::Rgba16Float.guaranteed_format_features(device.features());
                format.allowed_usages.contains(
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                ) && format.flags.contains(
                    wgpu::TextureFormatFeatureFlags::BLENDABLE
                        | wgpu::TextureFormatFeatureFlags::FILTERABLE,
                )
            },
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
            render_bundles: Slots::default(),
            query_sets: Slots::default(),
            samplers: Slots::default(),
            host_bytes: Slots::default(),
            #[cfg(target_arch = "wasm32")]
            host_images: Slots::default(),
            uploaded: Uploaded::default(),
            pipelines_created: 0,
            #[cfg(not(target_arch = "wasm32"))]
            pipeline_time: std::time::Duration::ZERO,
        }
    }

    /// Bytes written to buffers so far.
    pub fn uploaded(&self) -> Uploaded {
        self.uploaded
    }

    /// Render and compute pipelines created so far (plan §16).
    pub fn pipelines_created(&self) -> u32 {
        self.pipelines_created
    }

    /// The pipeline cache pipelines are made through, if any.
    fn cache(&self) -> Option<&wgpu::PipelineCache> {
        #[cfg(not(target_arch = "wasm32"))]
        return self.pipeline_cache.as_ref();
        #[cfg(target_arch = "wasm32")]
        None
    }

    /// Time spent making shader modules and pipelines so far.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pipeline_time(&self) -> std::time::Duration {
        self.pipeline_time
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
        HostImageId(self.host_images.insert(bitmap))
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

    /// Nanoseconds per timestamp tick in resolved query sets.
    pub fn timestamp_period_ns(&self) -> f32 {
        self.queue.get_timestamp_period()
    }

    fn timestamp_writes(
        &self,
        timestamps: Option<PassTimestamps>,
    ) -> Option<(&wgpu::QuerySet, Option<u32>, Option<u32>)> {
        timestamps.map(|t| (&self.query_sets[t.query_set.0], t.beginning, t.end))
    }

    fn pipeline_layout(&self, label: &str, layouts: &[BindGroupLayoutId]) -> wgpu::PipelineLayout {
        let layouts: Vec<_> = layouts
            .iter()
            .map(|id| Some(&self.bind_group_layouts[id.0]))
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
            RenderCommand::SetPipeline(id) => pass.set_pipeline(&self.render_pipelines[id.0]),
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
        self.buffers.remove(id.0).destroy();
    }

    fn write_buffer(&mut self, id: BufferId, offset: u64, data: &[u8]) {
        self.uploaded.from_core += data.len() as u64;
        self.queue.write_buffer(&self.buffers[id.0], offset, data);
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
                self.queue
                    .write_buffer(&self.buffers[id.0], offset, &data[range]);
            }
            #[cfg(target_arch = "wasm32")]
            HostBytes::Js(data) => {
                self.uploaded.from_js += len;
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
                mip_level: 0,
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
                source: wgpu::ExternalImageSource::ImageBitmap(self.host_images[image.0].clone()),
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
        self.host_images.remove(image.0).close();
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

    fn create_shader_module(&mut self, desc: &ShaderModuleDesc<'_>) -> ShaderModuleId {
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(desc.label),
                source: wgpu::ShaderSource::Wgsl(desc.wgsl.clone()),
            });
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pipeline_time += started.elapsed();
        }
        ShaderModuleId(self.shader_modules.insert(module))
    }

    fn create_bind_group_layout(&mut self, desc: &BindGroupLayoutDesc<'_>) -> BindGroupLayoutId {
        let entries: Vec<_> = desc
            .entries
            .iter()
            .map(convert::bind_group_layout_entry)
            .collect();
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(desc.label),
                entries: &entries,
            });
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
            layout: &self.bind_group_layouts[desc.layout.0],
            entries: &entries,
        });
        BindGroupId(self.bind_groups.insert(group))
    }

    fn destroy_bind_group(&mut self, id: BindGroupId) {
        self.bind_groups.remove(id.0);
    }

    fn create_render_pipeline(&mut self, desc: &RenderPipelineDesc<'_>) -> RenderPipelineId {
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();
        let layout = self.pipeline_layout(desc.label, desc.bind_group_layouts);
        let targets: Vec<_> = desc
            .targets
            .iter()
            .map(|t| Some(convert::color_target(t)))
            .collect();
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(desc.label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &self.shader_modules[desc.vertex.module.0],
                    entry_point: Some(desc.vertex.entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: desc.vertex.constants,
                        ..Default::default()
                    },
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: desc.cull_mode.map(|face| match face {
                        Face::Front => wgpu::Face::Front,
                        Face::Back => wgpu::Face::Back,
                    }),
                    topology: match desc.topology {
                        PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
                        PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
                        PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
                    },
                    ..Default::default()
                },
                depth_stencil: desc.depth.map(|d| wgpu::DepthStencilState {
                    format: convert::texture_format(d.format),
                    depth_write_enabled: Some(d.write),
                    depth_compare: Some(convert::compare(d.compare)),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: desc.sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader_modules[desc.fragment.module.0],
                    entry_point: Some(desc.fragment.entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: desc.fragment.constants,
                        ..Default::default()
                    },
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: self.cache(),
            });
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pipeline_time += started.elapsed();
        }
        self.pipelines_created += 1;
        RenderPipelineId(self.render_pipelines.insert(pipeline))
    }

    fn create_compute_pipeline(&mut self, desc: &ComputePipelineDesc<'_>) -> ComputePipelineId {
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();
        let layout = self.pipeline_layout(desc.label, desc.bind_group_layouts);
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(desc.label),
                layout: Some(&layout),
                module: &self.shader_modules[desc.compute.module.0],
                entry_point: Some(desc.compute.entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: desc.compute.constants,
                    ..Default::default()
                },
                cache: self.cache(),
            });
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pipeline_time += started.elapsed();
        }
        self.pipelines_created += 1;
        ComputePipelineId(self.compute_pipelines.insert(pipeline))
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

    fn create_query_set(&mut self, desc: &QuerySetDesc<'_>) -> QuerySetId {
        let set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some(desc.label),
            ty: wgpu::QueryType::Timestamp,
            count: desc.count,
        });
        QuerySetId(self.query_sets.insert(set))
    }

    fn destroy_query_set(&mut self, id: QuerySetId) {
        self.query_sets.remove(id.0);
    }

    fn submit(&mut self, passes: &[Pass]) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for pass in passes {
            match pass {
                Pass::CopyTextureLayers {
                    source,
                    destination,
                    layers,
                } => {
                    let source = self.texture(*source);
                    let size = wgpu::Extent3d {
                        width: source.width(),
                        height: source.height(),
                        depth_or_array_layers: *layers,
                    };
                    encoder.copy_texture_to_texture(
                        source.as_image_copy(),
                        self.texture(*destination).as_image_copy(),
                        size,
                    );
                }
                Pass::ClearBuffer {
                    buffer,
                    offset,
                    size,
                } => {
                    encoder.clear_buffer(&self.buffers[buffer.0], *offset, *size);
                }
                Pass::ResolveQuerySet {
                    query_set,
                    queries,
                    destination,
                    offset,
                } => encoder.resolve_query_set(
                    &self.query_sets[query_set.0],
                    queries.clone(),
                    &self.buffers[destination.0],
                    *offset,
                ),
                Pass::Compute(compute) => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&compute.label),
                        timestamp_writes: self.timestamp_writes(compute.timestamps).map(
                            |(query_set, beginning, end)| wgpu::ComputePassTimestampWrites {
                                query_set,
                                beginning_of_pass_write_index: beginning,
                                end_of_pass_write_index: end,
                            },
                        ),
                    });
                    for command in &compute.commands {
                        match command {
                            ComputeCommand::SetPipeline(id) => {
                                pass.set_pipeline(&self.compute_pipelines[id.0])
                            }
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
                    let color = &render.color;
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(&render.label),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
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
                        })],
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
                        timestamp_writes: self.timestamp_writes(render.timestamps).map(
                            |(query_set, beginning, end)| wgpu::RenderPassTimestampWrites {
                                query_set,
                                beginning_of_pass_write_index: beginning,
                                end_of_pass_write_index: end,
                            },
                        ),
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    for command in &render.commands {
                        self.encode_render(&mut pass, command);
                    }
                }
            }
        }
        self.queue.submit([encoder.finish()]);
    }
}
