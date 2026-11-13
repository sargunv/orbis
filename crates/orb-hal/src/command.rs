use crate::{
    BindGroupId, BufferId, ComputePipelineId, RenderBundleId, RenderPipelineId, TextureId,
};

/// One pass of a submission.
#[derive(Clone, Debug, PartialEq)]
pub enum Pass {
    Compute(ComputePass),
    Render(RenderPass),
    /// Copies boxes of `source`'s texels into `destination`, another
    /// texture of its format, at every mip level both have.
    CopyTexture {
        source: TextureId,
        destination: TextureId,
        copies: Vec<TextureCopy>,
    },
    /// Copies `size` bytes of `source` from `source_offset` into
    /// `destination` at `destination_offset`, another buffer: how a scene
    /// pool moves into a larger buffer. Both offsets and the size are whole
    /// words.
    CopyBuffer {
        source: BufferId,
        source_offset: u64,
        destination: BufferId,
        destination_offset: u64,
        size: u64,
    },
    /// Zeroes a buffer range; `size: None` clears to the end.
    ClearBuffer {
        buffer: BufferId,
        offset: u64,
        size: Option<u64>,
    },
    /// Opens a named group of the passes up to its [`PopDebugGroup`]
    /// (`Pass::PopDebugGroup`), for graphics debuggers' captures.
    PushDebugGroup(&'static str),
    PopDebugGroup,
}

impl Pass {
    /// Wraps the passes of `passes` from `start` on in a debug group, if
    /// there are any.
    pub fn group(passes: &mut Vec<Pass>, start: usize, label: &'static str) {
        if passes.len() > start {
            passes.insert(start, Pass::PushDebugGroup(label));
            passes.push(Pass::PopDebugGroup);
        }
    }
}

/// A box of texels at mip level 0, `size` (width, height, layers) from
/// `from` (x, y, layer) to `to`; each further level copies it scaled down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureCopy {
    pub from: [u32; 3],
    pub to: [u32; 3],
    pub size: [u32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComputePass {
    pub label: &'static str,
    pub commands: Vec<ComputeCommand>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ComputeCommand {
    SetPipeline(ComputePipelineId),
    SetBindGroup { index: u32, group: BindGroupId },
    Dispatch { x: u32, y: u32, z: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderPass {
    pub label: &'static str,
    /// Colour attachments, at the pipelines' target locations; none draws
    /// depth only.
    pub colors: Vec<ColorAttachment>,
    pub depth: Option<DepthAttachment>,
    pub commands: Vec<RenderCommand>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorAttachment {
    pub target: TextureId,
    /// Multisampled targets resolve into this texture.
    pub resolve: Option<TextureId>,
    pub store: StoreOp,
    /// Clear colour as premultiplied linear RGBA, or load.
    pub load: LoadOp<[f64; 4]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthAttachment {
    pub target: TextureId,
    pub load: LoadOp<f32>,
    pub store: StoreOp,
}

/// Whether attachment contents are needed after the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreOp {
    Store,
    Discard,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp<T> {
    Clear(T),
    Load,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RenderCommand {
    SetPipeline(RenderPipelineId),
    SetBindGroup {
        index: u32,
        group: BindGroupId,
    },
    Draw {
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    },
    /// Binds `buffer` from `offset` to the end as the index buffer, of
    /// u32 indices.
    SetIndexBuffer {
        buffer: BufferId,
        offset: u64,
    },
    DrawIndexed {
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    },
    /// Draw arguments `{vertex_count, instance_count, first_vertex,
    /// first_instance}` as four u32 at `offset`.
    DrawIndirect {
        buffer: BufferId,
        offset: u64,
    },
    /// Indexed draw arguments `{index_count, instance_count, first_index,
    /// base_vertex, first_instance}` as five 32-bit words at `offset`;
    /// `first_instance` must be 0 (core WebGPU).
    DrawIndexedIndirect {
        buffer: BufferId,
        offset: u64,
    },
    /// Replays bundles; only valid in a pass, not inside a bundle.
    ExecuteBundles(Vec<RenderBundleId>),
    /// Limits the draws after it, bundles' included, to a rectangle of
    /// the targets in texels: `[x, y, width, height]`. Only valid in a
    /// pass, not inside a bundle.
    SetScissor([u32; 4]),
    /// Maps clip space onto a rectangle `[x, y, width, height]` in texels
    /// for the draws after it, bundles' included: it may reach past the
    /// targets, whose texels it covers alone draw (by twice the device's
    /// largest 2D texture side at most), and is no larger than that side.
    /// Only valid in a pass, not inside a bundle.
    SetViewport([f32; 4]),
}
