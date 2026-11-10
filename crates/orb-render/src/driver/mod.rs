//! The frame drivers: how a frame's draws learn their counts.
//!
//! The renderer records its passes and bundles once for every device; what
//! differs between devices is only how a node's draw finds what to draw.
//! On the GPU-driven path ([`gpu`]), a compute cull writes each node's
//! indirect draw arguments every frame. The WebGL 2 path (#140), which has
//! no compute or indirect draws, gets a CPU driver of direct draws behind
//! the same calls. Everything else (the scene, the candidate layout, the
//! passes and bundles) is shared and never asks which driver runs.

pub(crate) mod gpu;

use orb_hal::{Gpu, Pass, PipelineId, RenderCommand};

use crate::scene::{NodeDraw, SceneBuffers, SceneCapacity};

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

/// The frame driver of a renderer's device.
#[derive(Debug)]
pub(crate) enum Driver {
    Gpu(gpu::GpuDriver),
}

impl Driver {
    /// The driver the device runs: the GPU-driven one wherever it has
    /// compute, indirect draws and vertex-stage storage.
    pub(crate) fn new(gpu: &mut dyn Gpu, buffers: &SceneBuffers) -> Self {
        assert!(
            gpu.caps().gpu_driven(),
            "the device lacks compute, indirect draws or vertex-stage storage, and there is no CPU driver yet (#140)"
        );
        Self::Gpu(gpu::GpuDriver::new(gpu, buffers))
    }

    /// Bits of a compacted index on this driver.
    pub(crate) fn index_bits(&self) -> u32 {
        match self {
            Self::Gpu(_) => gpu::INDEX_BITS,
        }
    }

    /// Whether frames can draw: its pipelines compiled.
    pub(crate) fn ready(&self, gpu: &mut dyn Gpu) -> bool {
        match self {
            Self::Gpu(d) => d.ready(gpu),
        }
    }

    /// Its pipelines, which compile before everything else.
    pub(crate) fn pipelines(&self) -> Vec<PipelineId> {
        match self {
            Self::Gpu(d) => d.pipelines(),
        }
    }

    /// Binds the scene's buffers again, after a pool moved.
    pub(crate) fn rebind(&mut self, gpu: &mut dyn Gpu, buffers: &SceneBuffers) {
        match self {
            Self::Gpu(d) => d.rebind(gpu, buffers),
        }
    }

    /// Keeps every candidate when `on`; whether that changed.
    pub(crate) fn set_keep_all(&mut self, gpu: &mut dyn Gpu, on: bool) -> bool {
        match self {
            Self::Gpu(d) => d.set_keep_all(gpu, on),
        }
    }

    /// The frame's passes before its draws: on the GPU, the cull. Returns
    /// the dispatches it adds.
    pub(crate) fn cull(&self, gpu: &mut dyn Gpu, frame: Cull<'_>, passes: &mut Vec<Pass>) -> u32 {
        match self {
            Self::Gpu(d) => d.cull(gpu, frame, passes),
        }
    }

    /// Node `node`'s draw, as a bundle records it: what every pass that
    /// draws a node's candidates replays.
    pub(crate) fn draw(&self, buffers: &SceneBuffers, node: u32, draw: NodeDraw) -> RenderCommand {
        match self {
            Self::Gpu(_) => gpu::draw(buffers, node, draw),
        }
    }

    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        match self {
            Self::Gpu(d) => d.destroy(gpu),
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
