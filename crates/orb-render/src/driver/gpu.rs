//! The GPU-driven frame: one compute dispatch culls and compacts every
//! node's candidates and writes its indirect draw arguments
//! (`cull.wgsl`), and each node draws with one indirect draw.
//!
//! The only place in `orb-render`, with the files `xtask check`'s
//! boundary rule lists, that names compute passes, indirect draws and
//! storage bindings: a device without them gets its frames from another
//! driver.

use orb_hal::*;
use orb_shaders::CULL_SHADER;

use crate::scene::{ARGS_WORDS, NodeDraw, SceneBuffers};

use super::Cull;

/// Compacted indices are full u32s.
pub(crate) const INDEX_BITS: u32 = 32;

/// The draw arguments: the cull dispatch writes them, indirect draws read
/// them, and tests read them back.
pub(crate) const ARGS_USAGE: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::INDIRECT)
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

/// The compacted indices: the cull dispatch writes them, draws read them.
pub(crate) const COMPACTED_USAGE: BufferUsages = BufferUsages::INDEX
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::STORAGE);

/// Where node `index`'s draw arguments start in the scene's args buffer.
pub(crate) fn args_offset(index: u32) -> u64 {
    u64::from(index) * ARGS_WORDS as u64 * 4
}

/// The cull's pipeline and the scene bound for it.
#[derive(Debug)]
pub(crate) struct GpuDriver {
    layout: BindGroupLayoutId,
    pipeline: ComputePipelineId,
    group: BindGroupId,
    /// The cull keeps every candidate (`cull.wgsl`'s `keep_all`).
    keep_all: bool,
}

impl GpuDriver {
    pub(crate) fn new(gpu: &mut dyn Gpu, buffers: &SceneBuffers) -> Self {
        use BindGroupLayoutEntry as E;
        let cs = ShaderStages::COMPUTE;
        let layout = gpu.create_bind_group_layout(&BindGroupLayoutDesc {
            label: "cull",
            entries: &[
                E::uniform(0, cs),
                E::storage(1, cs),
                E::storage(2, cs),
                E::storage_rw(3, cs),
                E::storage(4, cs),
                E::storage_rw(5, cs),
                E::storage_rw(6, cs),
                E::storage(7, cs),
                E::storage_rw(8, cs),
            ],
        });
        let group = group(gpu, layout, buffers);
        // The cull pipeline first: nothing draws without it.
        let pipeline = crate::pipeline::compute(gpu, &CULL_SHADER, &[layout], &[]);
        Self {
            layout,
            pipeline,
            group,
            keep_all: false,
        }
    }

    pub(crate) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
        gpu.pipeline_ready(self.pipeline.into())
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        vec![self.pipeline.into()]
    }

    pub(crate) fn rebind(&mut self, gpu: &mut dyn Gpu, buffers: &SceneBuffers) {
        gpu.destroy_bind_group(self.group);
        self.group = group(gpu, self.layout, buffers);
    }

    pub(crate) fn set_keep_all(&mut self, gpu: &mut dyn Gpu, on: bool) -> bool {
        if self.keep_all == on {
            return false;
        }
        self.keep_all = on;
        let pipeline = crate::pipeline::compute(
            gpu,
            &CULL_SHADER,
            &[self.layout],
            &[("keep_all", f64::from(u8::from(on)))],
        );
        gpu.destroy_pipeline(std::mem::replace(&mut self.pipeline, pipeline).into());
        true
    }

    /// The cull dispatch. It writes every node's arguments, zeros
    /// included, so it runs even when nothing is resident; indexed nodes'
    /// batches add to arguments cleared before it.
    pub(crate) fn cull(&self, gpu: &mut dyn Gpu, frame: Cull<'_>, passes: &mut Vec<Pass>) -> u32 {
        if frame.nodes == 0 || !self.ready(gpu) {
            return 0;
        }
        let start = passes.len();
        if frame.indexed {
            passes.push(Pass::ClearBuffer {
                buffer: frame.buffers.args,
                offset: 0,
                size: Some(args_offset(frame.nodes)),
            });
        }
        let [x, y] = frame.grid;
        passes.push(Pass::Compute(ComputePass {
            label: "cull",
            commands: vec![
                ComputeCommand::SetPipeline(self.pipeline),
                ComputeCommand::SetBindGroup {
                    index: 0,
                    group: self.group,
                },
                ComputeCommand::Dispatch { x, y, z: 1 },
            ],
        }));
        Pass::group(passes, start, "cull");
        1
    }

    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        gpu.destroy_pipeline(self.pipeline.into());
        gpu.destroy_bind_group(self.group);
        gpu.destroy_bind_group_layout(self.layout);
    }
}

/// Node `node`'s draw: its indirect arguments, which the cull wrote.
pub(crate) fn draw(buffers: &SceneBuffers, node: u32, draw: NodeDraw) -> RenderCommand {
    let (buffer, offset) = (buffers.args, args_offset(node));
    if draw == NodeDraw::Indexed {
        RenderCommand::DrawIndexedIndirect { buffer, offset }
    } else {
        RenderCommand::DrawIndirect { buffer, offset }
    }
}

/// The cull's group: the scene's tables and index buffers.
fn group(gpu: &mut dyn Gpu, layout: BindGroupLayoutId, b: &SceneBuffers) -> BindGroupId {
    let entries: Vec<BindGroupEntry> = [
        b.frame,
        b.tiles,
        b.chunks,
        b.candidates,
        b.node_cull,
        b.visible,
        b.args,
        b.indices,
        b.compacted,
    ]
    .iter()
    .enumerate()
    .map(|(i, &buffer)| BindGroupEntry::buffer(i as u32, buffer))
    .collect();
    gpu.create_bind_group(&BindGroupDesc {
        label: "cull",
        layout,
        entries: &entries,
    })
}
