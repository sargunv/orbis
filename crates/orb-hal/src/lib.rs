//! A small WebGPU-shaped GPU interface (plan §8.1).
//!
//! Core crates talk to the GPU only through [`Gpu`]: they create resources
//! from descriptors, receive opaque ids, and describe work as plain lists of
//! passes and commands. Implementations (`orb-hal-wgpu`, later `orb-hal-gl`)
//! own the real handles, so nothing below `orb-runtime` holds one, and command
//! lists recorded once can be replayed as render bundles.

#![forbid(unsafe_code)]

mod command;
pub mod device;
mod resource;
pub mod slots;

pub use command::{
    ColorAttachment, ComputeCommand, ComputePass, DepthAttachment, LoadOp, Pass, RenderCommand,
    RenderPass, StoreOp,
};
pub use resource::*;

/// The GPU as core crates see it.
///
/// Creation is synchronous, except for pipelines: those compile
/// asynchronously (plan §8.4), and commands may name one only once
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
    /// host-held bytes, which never pass through core memory (plan §8.1's
    /// web upload path).
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
    /// A view of one layer of an array texture's first mip level as a 2D
    /// texture, to render into (WebGPU's `baseArrayLayer`). Destroying it
    /// leaves the texture.
    fn create_layer_view(&mut self, texture: TextureId, layer: u32) -> TextureId;
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

    fn create_shader_module(&mut self, desc: &ShaderModuleDesc<'_>) -> ShaderModuleId;
    fn create_bind_group_layout(&mut self, desc: &BindGroupLayoutDesc<'_>) -> BindGroupLayoutId;
    fn create_bind_group(&mut self, desc: &BindGroupDesc<'_>) -> BindGroupId;
    fn destroy_bind_group(&mut self, id: BindGroupId);
    /// Queues a pipeline's compile and returns its id at once.
    fn create_render_pipeline(&mut self, desc: &RenderPipelineDesc<'_>) -> RenderPipelineId;
    /// Queues a pipeline's compile and returns its id at once.
    fn create_compute_pipeline(&mut self, desc: &ComputePipelineDesc<'_>) -> ComputePipelineId;
    /// Whether a pipeline compiled, so commands may use it.
    fn pipeline_ready(&mut self, id: PipelineId) -> bool;
    /// Takes compiles that finished, and counts pipelines made and still
    /// compiling.
    fn poll_pipelines(&mut self) -> PipelineCounts;
    /// Compiles a pending pipeline before those queued ahead of it, where
    /// the implementation keeps a queue: what the current view needs
    /// first (plan §8.4's first-visible order).
    fn prioritize_pipeline(&mut self, id: PipelineId);

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
    /// Pipelines compiled since the device was made; it only grows.
    pub made: u32,
    pub pending: u32,
}

/// What the device supports, detected at runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct Caps {
    pub tier: Tier,
    pub limits: Limits,
    /// Timestamp queries on pass boundaries (`timestamp-query`).
    pub timestamp_query: bool,
    /// `rgba16float` render attachments with additive blending and filtering.
    /// Core WebGPU guarantees this; Compat needs EXT_color_buffer_float.
    pub rgba16float_blendable: bool,
}

/// The two render configurations (plan §4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Storage buffers in every stage, compute and indirect draws.
    Modern,
    /// WebGL2-class GL.
    Compat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_array_layers: u32,
    pub min_storage_buffer_offset_alignment: u32,
    pub min_uniform_buffer_offset_alignment: u32,
}
