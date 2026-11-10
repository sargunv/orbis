//! The frame drivers: how a frame's draws learn their counts.
//!
//! The renderer records its passes and bundles once for every device; what
//! differs between devices is only how a node's draw finds what to draw.
//! On the GPU-driven path ([`gpu`]), a compute cull writes each node's
//! indirect draw arguments every frame. A device without compute or
//! indirect draws (WebGL 2, #140) takes the CPU driver ([`cpu`]): the same
//! cull on the CPU, writing the same visible lists and compacted indices,
//! and direct draws, which bundles record again when they change.
//! Everything else (the scene, the candidate layout, the passes and
//! bundles) is shared and never asks which driver runs.

#[cfg(cpu_driven)]
pub(crate) mod cpu;
pub(crate) mod gpu;

use orb_hal::{BufferUsages, Caps, Gpu, Pass, PipelineId, RenderCommand};

use crate::scene::{NodeDraw, Scene, SceneBuffers, SceneCapacity, Table};

/// The limits the candidate layout packs within (`rebuild_candidates`),
/// one set for whichever driver runs, so both drop the same candidates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SubmitLimits {
    /// Bits of a compacted index (`run << run_shift | vertex`): 32 on
    /// WebGPU; WebGL rejects element indices from 2^31.
    pub index_bits: u32,
    /// Words of the compacted index buffer: indexed candidates' slots.
    pub compacted_words: u32,
    /// `(chunk, placement)` candidates over all nodes.
    pub candidates: u32,
}

impl SubmitLimits {
    pub(crate) fn new(capacity: &SceneCapacity, index_bits: u32) -> Self {
        Self {
            index_bits,
            compacted_words: capacity.compacted_words,
            candidates: capacity.candidates,
        }
    }

    /// The runs a node whose meshes' vertices take `run_shift` bits can
    /// address, the all-ones run (hidden indices) aside.
    pub(crate) fn max_runs(&self, run_shift: u32) -> u64 {
        (1u64 << self.index_bits.saturating_sub(run_shift)) - 1
    }
}

/// Which driver a device's frames take, which the scene is made for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Gpu,
    #[cfg(cpu_driven)]
    Cpu,
}

impl Kind {
    /// The device's: the GPU-driven one wherever it has compute, indirect
    /// draws and vertex-stage storage.
    pub(crate) fn of(caps: &Caps) -> Self {
        #[cfg(cpu_driven)]
        if !caps.gpu_driven() {
            return Self::Cpu;
        }
        assert!(
            caps.gpu_driven(),
            "the device lacks compute, indirect draws or vertex-stage storage, and this build has no CPU driver (#140)"
        );
        Self::Gpu
    }

    /// The draw arguments' usage.
    pub(crate) fn args_usage(self) -> BufferUsages {
        match self {
            Self::Gpu => gpu::ARGS_USAGE,
            #[cfg(cpu_driven)]
            Self::Cpu => cpu::ARGS_USAGE,
        }
    }

    /// The compacted indices' usage.
    pub(crate) fn compacted_usage(self) -> BufferUsages {
        match self {
            Self::Gpu => gpu::COMPACTED_USAGE,
            #[cfg(cpu_driven)]
            Self::Cpu => cpu::COMPACTED_USAGE,
        }
    }

    /// The compacted index that hides a slot, as the driver writes it.
    pub(crate) fn hidden_index(self) -> u32 {
        match self {
            Self::Gpu => gpu::HIDDEN_INDEX,
            #[cfg(cpu_driven)]
            Self::Cpu => cpu::RESTART_INDEX,
        }
    }

    /// Bits of a compacted index.
    pub(crate) fn index_bits(self) -> u32 {
        match self {
            Self::Gpu => gpu::INDEX_BITS,
            #[cfg(cpu_driven)]
            Self::Cpu => cpu::INDEX_BITS,
        }
    }

    /// The tables the CPU keeps copies of, for the driver to read.
    pub(crate) fn mirrored(self) -> &'static [Table] {
        match self {
            Self::Gpu => &[],
            #[cfg(cpu_driven)]
            Self::Cpu => cpu::MIRRORED,
        }
    }
}

/// The frame driver of a renderer's device.
#[derive(Debug)]
pub(crate) enum Driver {
    Gpu(gpu::GpuDriver),
    #[cfg(cpu_driven)]
    Cpu(cpu::CpuDriver),
}

impl Driver {
    /// The driver of `kind`, over `scene`'s tables.
    pub(crate) fn new(gpu: &mut dyn Gpu, kind: Kind, scene: &Scene) -> Self {
        match kind {
            Kind::Gpu => Self::Gpu(gpu::GpuDriver::new(
                gpu,
                gpu::CullBuffers::of(scene.tables(), scene.buffers()),
            )),
            #[cfg(cpu_driven)]
            Kind::Cpu => Self::Cpu(cpu::CpuDriver::default()),
        }
    }

    /// Whether frames can draw: its pipelines compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
        match self {
            Self::Gpu(d) => d.ready(gpu),
            #[cfg(cpu_driven)]
            Self::Cpu(_) => true,
        }
    }

    /// Its pipelines, which compile before everything else.
    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        match self {
            Self::Gpu(d) => d.pipelines(),
            #[cfg(cpu_driven)]
            Self::Cpu(d) => d.pipelines(),
        }
    }

    /// Binds the scene's tables again, after one moved.
    pub(crate) fn rebind(&mut self, gpu: &mut dyn Gpu, scene: &Scene) {
        match self {
            Self::Gpu(d) => d.rebind(gpu, gpu::CullBuffers::of(scene.tables(), scene.buffers())),
            #[cfg(cpu_driven)]
            Self::Cpu(d) => d.rebind(gpu, scene),
        }
    }

    /// Keeps every candidate when `on`; whether that changed.
    pub(crate) fn set_keep_all(&mut self, gpu: &mut dyn Gpu, on: bool) -> bool {
        match self {
            Self::Gpu(d) => d.set_keep_all(gpu, on),
            #[cfg(cpu_driven)]
            Self::Cpu(d) => d.set_keep_all(gpu, on),
        }
    }

    /// The frame's work before bundles record: on the CPU, the cull,
    /// which writes the visible lists, compacted indices and draws.
    /// Returns whether any node's draw changed, so bundles that recorded
    /// it must record again.
    #[cfg_attr(not(cpu_driven), allow(unused_variables))]
    pub(crate) fn prepare(&mut self, gpu: &mut dyn Gpu, scene: &mut Scene, rebuilt: bool) -> bool {
        match self {
            Self::Gpu(_) => false,
            #[cfg(cpu_driven)]
            Self::Cpu(d) => d.prepare(gpu, scene, rebuilt),
        }
    }

    /// The frame's passes before its draws: on the GPU, the cull. Returns
    /// the dispatches it adds.
    pub(crate) fn cull(&self, gpu: &mut dyn Gpu, frame: Cull<'_>, passes: &mut Vec<Pass>) -> u32 {
        match self {
            Self::Gpu(d) => d.cull(gpu, frame, passes),
            #[cfg(cpu_driven)]
            Self::Cpu(d) => d.cull(gpu, frame, passes),
        }
    }

    /// Node `node`'s draw, as a bundle records it: what every pass that
    /// draws a node's candidates replays.
    pub(crate) fn draw(&self, buffers: &SceneBuffers, node: u32, draw: NodeDraw) -> RenderCommand {
        match self {
            Self::Gpu(_) => gpu::draw(buffers, node, draw),
            #[cfg(cpu_driven)]
            Self::Cpu(d) => d.draw(node, draw),
        }
    }

    /// Whether node `node`'s draw draws nothing, so bundles may leave it
    /// out: known only to a driver whose draws are recorded with their
    /// counts.
    #[cfg_attr(not(cpu_driven), allow(unused_variables))]
    pub(crate) fn draws_nothing(&self, node: u32) -> bool {
        match self {
            Self::Gpu(_) => false,
            #[cfg(cpu_driven)]
            Self::Cpu(d) => d.draws_nothing(node),
        }
    }

    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        match self {
            Self::Gpu(d) => d.destroy(gpu),
            #[cfg(cpu_driven)]
            Self::Cpu(d) => d.destroy(gpu),
        }
    }
}

/// What a frame's cull reads of the scene.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cull<'a> {
    pub buffers: &'a SceneBuffers,
    pub nodes: u32,
    /// Whether any node draws indexed meshes.
    pub indexed: bool,
    /// The dispatch's grid (`Scene::cull_grid`).
    pub grid: [u32; 2],
}
