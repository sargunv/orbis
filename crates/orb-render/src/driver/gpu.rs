//! The GPU-driven frame: one flat compute dispatch culls and compacts
//! every node's candidates and writes its indirect draw arguments
//! (`cull.wgsl`), and each node draws with one indirect draw.
//!
//! The only place in `orb-render`, with the files `xtask check`'s
//! boundary rule lists, that names compute passes, indirect draws and
//! storage bindings: a device without them gets its frames from another
//! driver.

use orb_hal::*;
use orb_shaders::CULL_SHADER;

use crate::scene::{ARGS_WORDS, BlockPool, NodeDraw, Scene, SceneBuffers, Table};

/// Compacted indices are full u32s.
pub(crate) const INDEX_BITS: u32 = 32;

/// `types.wgsl`'s compacted index that draws nothing, which the cull
/// writes: zero, so that clearing a range hides it.
#[cfg(cpu_driven)]
pub(crate) const HIDDEN_INDEX: u32 = 0;

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

/// The cull's pipeline and the scene bound for it, a group for each
/// block.
#[derive(Debug)]
pub(crate) struct GpuDriver {
    layout: BindGroupLayoutId,
    pipeline: ComputePipelineId,
    /// By block slot: its group, `None` for a free slot.
    groups: Vec<Option<BindGroupId>>,
    /// The cull keeps every candidate (`cull.wgsl`'s `keep_all`).
    keep_all: bool,
}

impl GpuDriver {
    /// The cull over `scene`, writing `swap`'s visible lists, draw
    /// arguments and compacted indices in place of the scene's if given.
    pub(crate) fn new(gpu: &mut dyn Gpu, scene: &Scene, swap: Option<Writes>) -> Self {
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
        let groups = groups(gpu, layout, scene, swap);
        // The cull pipeline first: nothing draws without it.
        let pipeline = crate::pipeline::compute(gpu, &CULL_SHADER, &[layout], &[]);
        Self {
            layout,
            pipeline,
            groups,
            keep_all: false,
        }
    }

    pub(crate) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
        gpu.pipeline_ready(self.pipeline.into())
    }

    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        vec![self.pipeline.into()]
    }

    /// Binds the scene's tables and blocks again, after one moved or a
    /// block came or went.
    pub(crate) fn rebind(&mut self, gpu: &mut dyn Gpu, scene: &Scene, swap: Option<Writes>) {
        for group in self.groups.drain(..).flatten() {
            gpu.destroy_bind_group(group);
        }
        self.groups = groups(gpu, self.layout, scene, swap);
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

    /// The cull: a dispatch for each block with rows to cull, of its
    /// workgroups as the node rows lay them out, after `args` is cleared,
    /// which leaves a node with no workgroups drawing nothing. The clear
    /// runs even when nothing is resident, to clear what nodes drew.
    /// Returns the dispatches.
    pub(crate) fn cull(
        &self,
        gpu: &mut dyn Gpu,
        scene: &Scene,
        args: BufferId,
        passes: &mut Vec<Pass>,
    ) -> u32 {
        if !self.ready(gpu) {
            return 0;
        }
        let start = passes.len();
        passes.push(Pass::ClearBuffer {
            buffer: args,
            offset: 0,
            size: None,
        });
        let mut commands = vec![ComputeCommand::SetPipeline(self.pipeline)];
        let mut dispatches = 0;
        for block in scene.blocks() {
            let groups = scene.block_groups(block);
            let Some(Some(group)) = self.groups.get(block as usize) else {
                continue;
            };
            if groups.is_empty() {
                continue;
            }
            commands.extend([
                ComputeCommand::SetBindGroup {
                    index: 0,
                    group: *group,
                },
                ComputeCommand::Dispatch {
                    x: groups.end - groups.start,
                    y: 1,
                    z: 1,
                },
            ]);
            dispatches += 1;
        }
        if dispatches > 0 {
            passes.push(Pass::Compute(ComputePass {
                label: "cull",
                commands,
            }));
        }
        Pass::group(passes, start, "cull");
        dispatches
    }

    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        gpu.destroy_pipeline(self.pipeline.into());
        for group in self.groups.into_iter().flatten() {
            gpu.destroy_bind_group(group);
        }
        gpu.destroy_bind_group_layout(self.layout);
    }
}

/// Node row `row`'s draw: its indirect arguments, which the cull wrote.
pub(crate) fn draw(buffers: &SceneBuffers, row: u32, draw: NodeDraw) -> RenderCommand {
    let (buffer, offset) = (buffers.args, args_offset(row));
    if draw == NodeDraw::Indexed {
        RenderCommand::DrawIndexedIndirect { buffer, offset }
    } else {
        RenderCommand::DrawIndirect { buffer, offset }
    }
}

/// What a cull writes in place of the scene's: the visible lists, draw
/// arguments and compacted indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Writes {
    pub visible: BufferId,
    pub args: BufferId,
    pub compacted: BufferId,
}

/// The cull's group for each of the scene's block slots: its frame and
/// index pool, the tables every block shares, and what the cull writes,
/// the scene's or `swap`'s.
fn groups(
    gpu: &mut dyn Gpu,
    layout: BindGroupLayoutId,
    scene: &Scene,
    swap: Option<Writes>,
) -> Vec<Option<BindGroupId>> {
    let (tables, buffers) = (scene.tables(), scene.buffers());
    let writes = swap.unwrap_or(Writes {
        visible: tables.buffer(Table::Visible),
        args: buffers.args,
        compacted: buffers.compacted,
    });
    let held: Vec<u32> = scene.blocks().collect();
    (0..scene.block_slots())
        .map(|block| {
            if !held.contains(&block) {
                return None;
            }
            let entries: Vec<BindGroupEntry> = [
                scene.block_frame(block),
                tables.buffer(Table::Tiles),
                tables.buffer(Table::Chunks),
                tables.buffer(Table::Candidates),
                tables.buffer(Table::NodeCull),
                writes.visible,
                writes.args,
                tables.block_buffer(block as usize, BlockPool::Indices),
                writes.compacted,
            ]
            .iter()
            .enumerate()
            .map(|(i, &buffer)| BindGroupEntry::buffer(i as u32, buffer))
            .collect();
            Some(gpu.create_bind_group(&BindGroupDesc {
                label: "cull",
                layout,
                entries: &entries,
            }))
        })
        .collect()
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
        let writes = Writes {
            visible,
            args,
            compacted,
        };
        Self {
            cull: GpuDriver::new(gpu, scene, Some(writes)),
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
            u64::from(scene.row_capacity()) * ARGS_WORDS as u64 * 4,
            u64::from(c.compacted_words) * 4,
        ]
        .map(|bytes| bytes.max(16))
    }

    fn create(gpu: &mut dyn Gpu, sizes: [u64; 3]) -> [BufferId; 3] {
        [0, 1, 2].map(|i| Self::buffer(gpu, i, sizes[i]))
    }

    /// Its visible lists (0), draw arguments (1) or compacted indices (2),
    /// `size` bytes.
    fn buffer(gpu: &mut dyn Gpu, which: usize, size: u64) -> BufferId {
        let (label, usage) = match which {
            0 => (
                "twin visible",
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            1 => ("twin args", ARGS_USAGE),
            _ => ("twin compacted", COMPACTED_USAGE | BufferUsages::COPY_SRC),
        };
        gpu.create_buffer(&BufferDesc { label, size, usage })
    }

    /// Binds the scene's tables again, each of its own buffers made again
    /// to its size if that changed: the compacted indices keep what the
    /// cull wrote while the scene's do, as the CPU driver's copies keep
    /// theirs.
    pub(crate) fn rebind(&mut self, gpu: &mut dyn Gpu, scene: &Scene) {
        let sizes = Self::sizes(scene);
        let mut buffers = [self.visible, self.args, self.compacted];
        for (i, buffer) in buffers.iter_mut().enumerate() {
            if sizes[i] != self.sizes[i] {
                gpu.destroy_buffer(*buffer);
                *buffer = Self::buffer(gpu, i, sizes[i]);
            }
        }
        [self.visible, self.args, self.compacted] = buffers;
        self.sizes = sizes;
        let writes = Writes {
            visible: self.visible,
            args: self.args,
            compacted: self.compacted,
        };
        self.cull.rebind(gpu, scene, Some(writes));
    }

    /// Its cull, writing its own buffers.
    pub(crate) fn cull(&self, gpu: &mut dyn Gpu, scene: &Scene, passes: &mut Vec<Pass>) -> u32 {
        self.cull.cull(gpu, scene, self.args, passes)
    }

    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        self.cull.destroy(gpu);
        for buffer in [self.visible, self.args, self.compacted] {
            gpu.destroy_buffer(buffer);
        }
    }
}
