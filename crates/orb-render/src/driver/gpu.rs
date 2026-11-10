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

#[cfg(cpu_driven)]
use crate::scene::Scene;
use crate::scene::{ARGS_WORDS, NodeDraw, SceneBuffers, Table, Tables};

use super::Cull;

/// Compacted indices are full u32s.
pub(crate) const INDEX_BITS: u32 = 32;

/// `types.wgsl`'s compacted index that draws nothing, which the cull
/// writes.
pub(crate) const HIDDEN_INDEX: u32 = 0xffff_fffe;

/// The draw arguments: the cull dispatch writes them, indirect draws read
/// them, and tests read them back.
pub(crate) const ARGS_USAGE: BufferUsages = BufferUsages::STORAGE
    .union(BufferUsages::INDIRECT)
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC);

/// The compacted indices: the cull dispatch writes them, draws read them,
/// and a layout moves slots within them.
pub(crate) const COMPACTED_USAGE: BufferUsages = BufferUsages::INDEX
    .union(BufferUsages::COPY_DST)
    .union(BufferUsages::COPY_SRC)
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
    pub(crate) fn new(gpu: &mut dyn Gpu, buffers: CullBuffers) -> Self {
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

    pub(crate) fn rebind(&mut self, gpu: &mut dyn Gpu, buffers: CullBuffers) {
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

/// What the cull binds: the scene's tables it reads and writes, and the
/// draw arguments and compacted indices it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CullBuffers {
    pub frame: BufferId,
    pub tiles: BufferId,
    pub chunks: BufferId,
    pub candidates: BufferId,
    pub node_cull: BufferId,
    pub visible: BufferId,
    pub args: BufferId,
    pub indices: BufferId,
    pub compacted: BufferId,
}

impl CullBuffers {
    /// The scene's, on the storage backing.
    pub(crate) fn of(tables: &Tables, buffers: &SceneBuffers) -> Self {
        Self {
            frame: buffers.frame,
            tiles: tables.buffer(Table::Tiles),
            chunks: tables.buffer(Table::Chunks),
            candidates: tables.buffer(Table::Candidates),
            node_cull: tables.buffer(Table::NodeCull),
            visible: tables.buffer(Table::Visible),
            args: buffers.args,
            indices: tables.buffer(Table::Indices),
            compacted: buffers.compacted,
        }
    }
}

/// The cull's group.
fn group(gpu: &mut dyn Gpu, layout: BindGroupLayoutId, b: CullBuffers) -> BindGroupId {
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

/// The GPU cull beside the CPU driver's, over the same scene and frame,
/// writing visible lists, draw arguments and compacted indices of its own
/// (and the candidates' drawn words, which the CPU driver keeps on the
/// CPU): what a test reads back to hold the CPU driver to it word for
/// word (`Renderer::check_driver`).
#[cfg(cpu_driven)]
#[derive(Debug)]
pub(crate) struct Twin {
    pub(crate) cull: GpuDriver,
    pub(crate) visible: BufferId,
    pub(crate) args: BufferId,
    pub(crate) compacted: BufferId,
    sizes: [u64; 3],
}

#[cfg(cpu_driven)]
impl Twin {
    pub(crate) fn new(gpu: &mut dyn Gpu, scene: &Scene) -> Self {
        let sizes = Self::sizes(scene);
        let [visible, args, compacted] = Self::create(gpu, sizes);
        let buffers = CullBuffers {
            visible,
            args,
            compacted,
            ..CullBuffers::of(scene.tables(), scene.buffers())
        };
        Self {
            cull: GpuDriver::new(gpu, buffers),
            visible,
            args,
            compacted,
            sizes,
        }
    }

    /// Bytes of the visible lists, draw arguments and compacted indices.
    fn sizes(scene: &Scene) -> [u64; 3] {
        let c = scene.capacity();
        [
            u64::from(c.candidates) * 8,
            u64::from(c.nodes) * ARGS_WORDS as u64 * 4,
            u64::from(c.compacted_words) * 4,
        ]
        .map(|bytes| bytes.max(16))
    }

    fn create(gpu: &mut dyn Gpu, [visible, args, compacted]: [u64; 3]) -> [BufferId; 3] {
        let mut buffer = |label, size, usage| gpu.create_buffer(&BufferDesc { label, size, usage });
        [
            buffer(
                "twin visible",
                visible,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            buffer("twin args", args, ARGS_USAGE),
            buffer(
                "twin compacted",
                compacted,
                COMPACTED_USAGE | BufferUsages::COPY_SRC,
            ),
        ]
    }

    /// The scene's buffers with the twin's own in place of what it writes.
    fn buffers(&self, tables: &Tables, buffers: &SceneBuffers) -> CullBuffers {
        CullBuffers {
            visible: self.visible,
            args: self.args,
            compacted: self.compacted,
            ..CullBuffers::of(tables, buffers)
        }
    }

    /// Binds the scene's tables again, its own buffers made again to
    /// their sizes.
    pub(crate) fn rebind(&mut self, gpu: &mut dyn Gpu, scene: &Scene) {
        let sizes = Self::sizes(scene);
        if sizes != self.sizes {
            for buffer in [self.visible, self.args, self.compacted] {
                gpu.destroy_buffer(buffer);
            }
            [self.visible, self.args, self.compacted] = Self::create(gpu, sizes);
            self.sizes = sizes;
        }
        let b = self.buffers(scene.tables(), scene.buffers());
        self.cull.rebind(gpu, b);
    }

    /// Its cull, writing its own buffers.
    pub(crate) fn cull(&self, gpu: &mut dyn Gpu, frame: Cull<'_>, passes: &mut Vec<Pass>) -> u32 {
        let buffers = SceneBuffers {
            args: self.args,
            compacted: self.compacted,
            ..*frame.buffers
        };
        self.cull.cull(
            gpu,
            Cull {
                buffers: &buffers,
                ..frame
            },
            passes,
        )
    }

    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        self.cull.destroy(gpu);
        for buffer in [self.visible, self.args, self.compacted] {
            gpu.destroy_buffer(buffer);
        }
    }
}
