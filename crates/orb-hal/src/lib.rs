//! A small WebGPU-shaped GPU interface.
//!
//! Core crates talk to the GPU only through [`Gpu`]: they create resources
//! from descriptors, receive opaque ids, and describe work as plain lists of
//! passes and commands. The implementation (`orb-hal-wgpu`) owns the
//! real handles, so nothing below `orb-runtime` holds one, and command
//! lists recorded once can be replayed as render bundles.

#![forbid(unsafe_code)]

mod command;
#[cfg(feature = "recording")]
mod recording;
mod resource;

pub use command::{
    ColorAttachment, ComputeCommand, ComputePass, DepthAttachment, LoadOp, Pass, RenderCommand,
    RenderPass, StoreOp, TextureCopy,
};
#[cfg(feature = "recording")]
pub use recording::Recording;
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
    /// A view of some of a texture's mip levels, to render into or sample
    /// apart from the others (WebGPU's `createView`). Destroying it with
    /// [`destroy_texture`](Self::destroy_texture) leaves the texture.
    fn create_texture_view(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
    ) -> TextureId;
    /// [`create_texture_view`](Self::create_texture_view) in `format`, one
    /// of the texture's `view_formats`.
    fn create_texture_view_as(
        &mut self,
        texture: TextureId,
        levels: std::ops::Range<u32>,
        format: TextureFormat,
    ) -> TextureId;
    /// A view of one layer of an array texture at one mip level as a 2D
    /// texture, to render into (WebGPU's `baseArrayLayer`), in `format`
    /// when given, one of the texture's `view_formats`. Destroying it
    /// leaves the texture.
    fn create_layer_view(
        &mut self,
        texture: TextureId,
        layer: u32,
        level: u32,
        format: Option<TextureFormat>,
    ) -> TextureId;
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
    /// Many direct draws in one call (`WEBGL_multi_draw`).
    pub multi_draw: bool,
    /// Clip-space depth from 0 to 1 as the shaders write it. wgpu's GL
    /// back end remaps it into GL's −1 to 1 (`2z − w`), which spends
    /// reversed Z's precision far from the camera.
    pub zero_to_one_depth: bool,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_storage_buffers_per_shader_stage: u32,
    /// The 2D texture side everything sizes by: the device's, up to
    /// WebGPU's default (8192), so that what a map draws doesn't depend on
    /// a device that offers more.
    pub max_texture_dimension_2d: u32,
    /// The device's own largest 2D texture side, at least the above: for
    /// textures whose size changes only how they pack what they hold (the
    /// terrain drape's pages, whose stacks then share a layer).
    pub largest_texture_dimension_2d: u32,
    pub max_texture_array_layers: u32,
    pub min_storage_buffer_offset_alignment: u32,
    pub min_uniform_buffer_offset_alignment: u32,
}
