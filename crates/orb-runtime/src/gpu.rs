//! Interpreted paint (plan §11.3 `GpuBytecode`, decision 0017): the plan's
//! programs in the property arena, and the uniforms that point the
//! fill, line and circle shaders at them.
//!
//! One table holds every node's programs, encoded once per style revision.
//! An edit that changes only programs rewrites it: a buffer write, with no
//! column job when the node's input columns pack the same.

use std::ops::Range;

use orb_expr::gpu::Input;
use orb_hal::Gpu;
use orb_render::{GpuPaint, Renderer};

use crate::state::FeatureStates;
use crate::style::{NodePlan, StylePlan};

/// The plan's programs, uploaded.
#[derive(Debug, Default)]
pub(crate) struct ProgramTable {
    revision: Option<u64>,
    range: Option<Range<u32>>,
    words: Vec<u32>,
    /// Per plan node, the word its block starts at in `words`.
    blocks: Vec<Option<u32>>,
}

impl ProgramTable {
    /// Encodes and uploads `plan`'s programs, if it's a new revision. A
    /// node's block is its per-slot program offsets, then the programs.
    pub(crate) fn prepare(&mut self, gpu: &mut dyn Gpu, renderer: &mut Renderer, plan: &StylePlan) {
        if self.revision == Some(plan.revision) {
            return;
        }
        self.revision = Some(plan.revision);
        let mut words = Vec::new();
        self.blocks.clear();
        for node in &plan.nodes {
            let Some(gpu) = &node.gpu else {
                self.blocks.push(None);
                continue;
            };
            let start = words.len();
            self.blocks.push(Some(start as u32));
            words.resize(start + gpu.programs.len(), 0);
            let properties = plan.paint_properties(node);
            for (slot, program) in gpu.programs.iter().enumerate() {
                if let Some(program) = program {
                    words[start + slot] = (words.len() - start) as u32;
                    let default = properties[slot].default_value();
                    words.extend(gpu.layout.encode(program, &default));
                }
            }
        }
        if words == self.words && self.range.is_some() {
            return;
        }
        let fits = self.range.as_ref().is_some_and(|r| r.len() == words.len());
        if !fits {
            if let Some(range) = self.range.take() {
                renderer.scene().free_table(range);
            }
            if !words.is_empty() {
                // Out of property memory: interpreted paint draws defaults.
                self.range = renderer.scene().alloc_table(words.len() as u32).ok();
            }
        }
        if let Some(range) = &self.range {
            renderer.scene().write_table(gpu, range.start, &words);
        }
        self.words = words;
    }

    /// Where node `index` of the plan reads its interpreted paint.
    pub(crate) fn paint(&self, plan: &StylePlan, index: usize, states: &FeatureStates) -> GpuPaint {
        let node = &plan.nodes[index];
        match (
            &node.gpu,
            &self.range,
            self.blocks.get(index).copied().flatten(),
        ) {
            (Some(gpu), Some(range), Some(block)) => GpuPaint {
                mask: gpu.mask(),
                programs: range.start + block,
                state: states.raw_offset(index),
            },
            _ => GpuPaint::default(),
        }
    }

    pub(crate) fn release(&mut self, renderer: &mut Renderer) {
        if let Some(range) = self.range.take() {
            renderer.scene().free_table(range);
        }
        *self = Self::default();
    }
}

/// An upper bound of interpreted number slot `slot`'s magnitude at `zoom`,
/// for culling margins: over the input ranges the node's meshes reported
/// after their per-slot maxima.
pub(crate) fn bound(
    plan: &StylePlan,
    node: &NodePlan,
    slot: usize,
    zoom: f64,
    maxima: &[f32],
) -> f32 {
    let properties = plan.paint_properties(node);
    let (Some(gpu), Some(expression)) = (&node.gpu, properties[slot].expression()) else {
        return 0.0;
    };
    let slots = gpu.programs.len();
    let columns = gpu.layout.columns();
    let range = |input: &Input| {
        let i = columns.iter().position(|c| c == input)?;
        let max = f64::from(*maxima.get(slots + 2 * i)?);
        let min = -f64::from(*maxima.get(slots + 2 * i + 1)?);
        Some([min, max])
    };
    expression.expression().bound(zoom, &range) as f32
}
