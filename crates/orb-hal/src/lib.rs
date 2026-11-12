//! A small WebGPU-shaped GPU interface.
//!
//! Core crates talk to the GPU only through [`Gpu`]: they create resources
//! from descriptors, receive opaque ids, and describe work as plain lists of
//! passes and commands. The implementation (`orb-hal-wgpu`) owns the
//! real handles, so nothing below `orb-runtime` holds one, and command
//! lists recorded once can be replayed as render bundles.

#![forbid(unsafe_code)]

mod command;
mod resource;

pub use command::{
    ColorAttachment, ComputeCommand, ComputePass, DepthAttachment, LoadOp, Pass, RenderCommand,
    RenderPass, StoreOp, TextureCopy,
};
pub use resource::*;

/// The GPU as core crates see it.
///
/// Creation is synchronous, except for pipelines: those compile
/// asynchronously, and commands may name one only once
/// [`pipeline_ready`](Self::pipeline_ready) says it compiled. Ids are never
/// reused, and using a destroyed id is a bug that implementations may panic
/// on.
pub trait Gpu {
    fn caps(&self) -> &Caps;

    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> BufferId;
    fn destroy_buffer(&mut self, id: BufferId);
    /// Queues a write that lands before the next submitted pass.
    fn write_buffer(&mut self, id: BufferId, offset: u64, data: &[u8]);
    /// Like [`write_buffer`](Self::write_buffer), from a byte range of
    /// host-held bytes, which never pass through core memory (the web
    /// upload path).
    fn write_buffer_from_host(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: HostBytesId,
        range: std::ops::Range<u64>,
    );
    /// Copies a byte range of host-held bytes into core memory, for data
    /// the CPU reads too (symbol placement tables).
    fn read_host_bytes(&self, bytes: HostBytesId, range: std::ops::Range<u64>) -> Vec<u8>;
    /// Releases host-held bytes once every write from them is queued.
    fn release_host_bytes(&mut self, bytes: HostBytesId);

    fn create_texture(&mut self, desc: &TextureDesc<'_>) -> TextureId;
    /// Whether the device allocates a texture of `desc`: its first mip
    /// level, over the layers the device makes it with, within
    /// [`Limits::max_texture_bytes`]. Sides and layers within the other
    /// limits are the caller's to keep.
    fn texture_fits(&self, desc: &TextureDesc<'_>) -> bool {
        level_bytes(desc, desc.array_layers) <= self.caps().limits.max_texture_bytes
    }
    /// [`create_texture`](Self::create_texture), or `None` where the
    /// device says at once that it couldn't allocate it (GL's driver
    /// refusing a large one), for textures large enough that it might:
    /// the device then lowers its [`Limits::max_texture_bytes`] below
    /// it, so that the caller can ask for less.
    fn try_create_texture(&mut self, desc: &TextureDesc<'_>) -> Option<TextureId> {
        Some(self.create_texture(desc))
    }
    /// A view of some of a texture's mip levels, to render into or sample
    /// apart from the others (WebGPU's `createView`). Destroying it with
    /// [`destroy_texture`](Self::destroy_texture) leaves the texture.
    fn create_texture_view(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
    ) -> TextureId;
    /// A view of one layer of an array texture at one mip level as a 2D
    /// texture, to render into (WebGPU's `baseArrayLayer`). Destroying it
    /// leaves the texture.
    fn create_layer_view(&mut self, texture: TextureId, layer: u32, level: u32) -> TextureId;
    fn destroy_texture(&mut self, id: TextureId);
    /// Queues a write of tightly packed texel rows into a region; it lands
    /// before the next submitted pass.
    fn write_texture(&mut self, region: &TextureRegion, data: &[u8]);
    /// Like [`write_texture`](Self::write_texture), from a host-held image
    /// the size of `region`, which never passes through core memory (the
    /// web's `copyExternalImageToTexture`, so the texture needs
    /// [`TextureUsages::RENDER_ATTACHMENT`]). `premultiply` multiplies colour
    /// by alpha on the way.
    fn write_texture_from_host(
        &mut self,
        region: &TextureRegion,
        image: HostImageId,
        premultiply: bool,
    );
    /// Like [`write_texture`](Self::write_texture), from `region`'s
    /// tightly packed texels at byte `offset` of host-held bytes, which
    /// never pass through core memory (the web's `writeTexture` from a
    /// buffer a job worker transferred).
    fn write_texture_from_host_bytes(
        &mut self,
        region: &TextureRegion,
        bytes: HostBytesId,
        offset: u64,
    );
    /// Releases a host-held image once every write from it is queued.
    fn release_host_image(&mut self, image: HostImageId);
    fn create_sampler(&mut self, desc: &SamplerDesc<'_>) -> SamplerId;
    fn destroy_sampler(&mut self, id: SamplerId);

    fn create_shader_module(&mut self, desc: &ShaderModuleDesc<'_>) -> ShaderModuleId;
    /// Pipelines made from the module keep what they need of it.
    fn destroy_shader_module(&mut self, id: ShaderModuleId);
    fn create_bind_group_layout(&mut self, desc: &BindGroupLayoutDesc<'_>) -> BindGroupLayoutId;
    fn destroy_bind_group_layout(&mut self, id: BindGroupLayoutId);
    fn create_bind_group(&mut self, desc: &BindGroupDesc<'_>) -> BindGroupId;
    fn destroy_bind_group(&mut self, id: BindGroupId);
    /// Queues a pipeline's compile and returns its id at once.
    fn create_render_pipeline(&mut self, desc: &RenderPipelineDesc<'_>) -> RenderPipelineId;
    /// Queues a pipeline's compile and returns its id at once.
    fn create_compute_pipeline(&mut self, desc: &ComputePipelineDesc<'_>) -> ComputePipelineId;
    /// A pipeline still compiling is dropped when its compile finishes.
    fn destroy_pipeline(&mut self, id: PipelineId);
    /// Whether a pipeline compiled, so commands may use it.
    fn pipeline_ready(&mut self, id: PipelineId) -> bool;
    /// Takes compiles that finished, and counts pipelines made and still
    /// compiling (background ones aside).
    fn poll_pipelines(&mut self) -> PipelineCounts;
    /// Compiles a pending pipeline before those queued ahead of it, where
    /// the implementation keeps a queue: what the current view needs
    /// first.
    fn prioritize_pipeline(&mut self, id: PipelineId);
    /// Makes the pipelines asked for from now on background ones (`on`),
    /// or ones a frame waits for again: a background pipeline compiles
    /// after every other, a few at a time, while the device is otherwise
    /// idle, and isn't counted in [`PipelineCounts`] nor wakes the host.
    /// [`pipeline_ready`](Self::pipeline_ready) asking for one still
    /// compiling makes it an ordinary request, compiled first. An
    /// implementation without a queue compiles them as any other.
    fn set_background_compiles(&mut self, on: bool) {
        let _ = on;
    }
    /// Holds background compiles that haven't started while `paused`
    /// (the map isn't at rest): they run at low priority, after every
    /// compile a frame waits for and while the map rests. In a browser
    /// that compiles on the thread that presents (Firefox), each would
    /// hold frames back. One a frame asks for compiles at once.
    fn pause_background_compiles(&mut self, paused: bool) {
        let _ = paused;
    }
    /// Whether a frame presented now would wait behind compiles: on a
    /// device whose compiles hold presents
    /// ([`Caps::compiles_hold_presents`]), while one runs.
    fn presents_wait(&mut self) -> bool {
        false
    }

    /// Records render commands once for replay with
    /// [`RenderCommand::ExecuteBundles`]. Bundles may not contain
    /// `ExecuteBundles`.
    fn create_render_bundle(
        &mut self,
        desc: &RenderBundleDesc<'_>,
        commands: &[RenderCommand],
    ) -> RenderBundleId;
    fn destroy_render_bundle(&mut self, id: RenderBundleId);

    /// Encodes and submits passes in order.
    fn submit(&mut self, passes: &[Pass]);

    /// Resources alive, of every kind, for leak checks.
    fn live_resources(&self) -> usize;

    /// What the device has done since it was made, for measurements. An
    /// implementation that counts nothing reports zeros.
    fn counters(&self) -> GpuCounters {
        GpuCounters::default()
    }
}

/// What a device has done since it was made: every count only grows, so
/// what happened between two readings is their [`since`](Self::since).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuCounters {
    /// Render and compute pipelines asked for: each one compiles, which
    /// some browsers do on the thread that presents.
    pub pipelines_requested: u32,
    /// Pipelines whose compile finished, background ones too.
    pub pipelines_compiled: u32,
    /// Buffers and textures made (views aside), and their bytes, every mip
    /// level and layer: new GPU memory, which a pool or page growing then
    /// copies into.
    pub buffers_created: u32,
    pub textures_created: u32,
    pub bytes_allocated: u64,
    /// Bytes copied on the GPU from buffer to buffer and texture to
    /// texture: pools and pages moving into larger ones, candidates
    /// compacting, mip chains.
    pub bytes_copied: u64,
    /// Bytes written to buffers from core memory ([`Gpu::write_buffer`],
    /// and host bytes the host keeps in core memory), and from host
    /// memory that never enters it (JS `ArrayBuffer`s).
    pub buffer_bytes_from_core: u64,
    pub buffer_bytes_from_host: u64,
    /// Texel bytes written to textures from core memory, and copied from
    /// host images (`ImageBitmap`s, video).
    pub texel_bytes_from_core: u64,
    pub texel_bytes_from_host: u64,
    /// Uploads the device issued to the graphics API: queue writes and
    /// image copies, and natively each staged copy (gathered writes count
    /// once a submission).
    pub upload_calls: u32,
    /// Submissions made only to order texture writes that share a block
    /// (their waves), each an extra command buffer.
    pub texture_write_submissions: u32,
}

impl GpuCounters {
    /// Every byte uploaded, to buffers and textures.
    pub fn uploaded_bytes(&self) -> u64 {
        self.buffer_bytes_from_core
            + self.buffer_bytes_from_host
            + self.texel_bytes_from_core
            + self.texel_bytes_from_host
    }

    /// What happened between `earlier` and these. A device made since
    /// (after a loss) counts from zero again: then all of these.
    #[must_use]
    pub fn since(&self, earlier: &Self) -> Self {
        let restarted = self.zip(earlier, |a, b| u32::from(a < b), |a, b| u64::from(a < b));
        match restarted == Self::default() {
            true => self.zip(earlier, u32::wrapping_sub, u64::wrapping_sub),
            false => *self,
        }
    }

    /// Each count of these and `other`, combined field by field.
    fn zip(
        &self,
        other: &Self,
        count: impl Fn(u32, u32) -> u32,
        bytes: impl Fn(u64, u64) -> u64,
    ) -> Self {
        let (a, b) = (self, other);
        Self {
            pipelines_requested: count(a.pipelines_requested, b.pipelines_requested),
            pipelines_compiled: count(a.pipelines_compiled, b.pipelines_compiled),
            buffers_created: count(a.buffers_created, b.buffers_created),
            textures_created: count(a.textures_created, b.textures_created),
            bytes_allocated: bytes(a.bytes_allocated, b.bytes_allocated),
            bytes_copied: bytes(a.bytes_copied, b.bytes_copied),
            buffer_bytes_from_core: bytes(a.buffer_bytes_from_core, b.buffer_bytes_from_core),
            buffer_bytes_from_host: bytes(a.buffer_bytes_from_host, b.buffer_bytes_from_host),
            texel_bytes_from_core: bytes(a.texel_bytes_from_core, b.texel_bytes_from_core),
            texel_bytes_from_host: bytes(a.texel_bytes_from_host, b.texel_bytes_from_host),
            upload_calls: count(a.upload_calls, b.upload_calls),
            texture_write_submissions: count(
                a.texture_write_submissions,
                b.texture_write_submissions,
            ),
        }
    }
}

/// Counts over several spans, as [`GpuCounters::since`] gives them.
impl std::ops::AddAssign for GpuCounters {
    fn add_assign(&mut self, other: Self) {
        *self = self.zip(&other, u32::saturating_add, u64::saturating_add);
    }
}

impl std::iter::Sum for GpuCounters {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |mut sum, c| {
            sum += c;
            sum
        })
    }
}

/// A render or compute pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PipelineId {
    Render(RenderPipelineId),
    Compute(ComputePipelineId),
}

impl From<RenderPipelineId> for PipelineId {
    fn from(id: RenderPipelineId) -> Self {
        Self::Render(id)
    }
}

impl From<ComputePipelineId> for PipelineId {
    fn from(id: ComputePipelineId) -> Self {
        Self::Compute(id)
    }
}

/// Pipelines a device has compiled, and those it is still compiling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineCounts {
    /// Pipelines compiled since the device was made, background ones
    /// aside (unless a frame asked for one before it compiled); it only
    /// grows.
    pub made: u32,
    pub pending: u32,
}

/// What the device supports, detected at runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct Caps {
    pub limits: Limits,
    /// Timestamp queries on pass boundaries (`timestamp-query`).
    pub timestamp_query: bool,
    /// Clip distances in vertex shaders (`clip-distances`): a placement's
    /// clip rectangle as planes, where a fragment would otherwise discard.
    pub clip_distances: bool,
    /// Compute passes. WebGPU has them; WebGL 2 doesn't, and frames there
    /// cull and make mips without them.
    pub compute: bool,
    /// Indirect draws (`DrawIndirect*`), whose arguments a compute pass
    /// writes.
    pub indirect: bool,
    /// Storage buffers the vertex stage reads; without them, tables are
    /// data textures.
    pub vertex_storage: bool,
    /// Blendable `Rgba16Float` and renderable `R32Float` targets: heatmap
    /// density and the terrain's depth for what it hides.
    pub float_targets: bool,
    /// Clip-space depth from 0 to 1 as the shaders write it. On a GL
    /// context without clip control (`EXT_clip_control`: Firefox), wgpu's
    /// GL back end remaps it into GL's −1 to 1 (`2z − w`), which spends
    /// reversed Z's precision far from the camera.
    pub zero_to_one_depth: bool,
    /// The device compiles pipelines on the thread that runs its commands
    /// (Firefox's WebGPU, in its GPU process), so a present waits behind
    /// every compile asked for before it, and the page's frames behind
    /// that present. It has no background compiles, and a map on it shows
    /// nothing until its style's pipelines compiled, nor while any
    /// compiles ([`Gpu::presents_wait`]).
    pub compiles_hold_presents: bool,
}

impl Caps {
    /// Whether the GPU-driven frame (a compute cull writing indirect
    /// draws over storage-buffer tables) runs here.
    pub fn gpu_driven(&self) -> bool {
        self.compute && self.indirect && self.vertex_storage
    }

    /// Whether pipelines take WebGL 2's shaders (`Shader::gl_source`):
    /// on a device without compute or vertex-stage storage. Their vertex
    /// stages read a `u32` at `@location(0)`, which the HAL binds,
    /// instance-stepped, over zeros (#179).
    pub fn gl_shaders(&self) -> bool {
        !self.compute && !self.vertex_storage
    }
}

/// Bytes of `desc`'s first mip level with `layers` layers.
pub fn level_bytes(desc: &TextureDesc<'_>, layers: u32) -> u64 {
    u64::from(desc.width)
        * u64::from(desc.height)
        * u64::from(layers)
        * u64::from(desc.format.texel_bytes())
        * u64::from(desc.sample_count.max(1))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_storage_buffers_per_shader_stage: u32,
    /// The 2D texture side everything sizes by: the device's, up to
    /// WebGPU's default (8192), so that what a map draws doesn't depend on
    /// a device that offers more.
    pub max_texture_dimension_2d: u32,
    pub max_texture_array_layers: u32,
    /// The most bytes a texture's first mip level may take over its
    /// layers ([`Gpu::texture_fits`]): unbounded on WebGPU and the native
    /// back ends, whose limit is memory. GL has a limit it doesn't report:
    /// Chrome's ANGLE refuses past 2 GiB on GL and 1.25 GiB on Metal (an
    /// M6 with 32 GiB), so GL takes 1 GiB, lowered where the driver
    /// refuses less ([`Gpu::try_create_texture`]).
    pub max_texture_bytes: u64,
    pub min_storage_buffer_offset_alignment: u32,
    pub min_uniform_buffer_offset_alignment: u32,
}
