//! Node uniforms: each render node's paint as words at its row, and its
//! second stream's at that stream's row, kept as written so a frame writes
//! only rows that changed, in one buffer write a run of them.

use orb_hal::*;

use crate::release::Release;

/// Bytes reserved per node uniform; every node struct fits. A symbol
/// layer's icon node binds its text node's after its own, this far apart
/// (`symbol.wgsl`'s `halves`).
pub(crate) const NODE_UNIFORM_SIZE: u64 = 512;
const _: () = assert!(orb_shaders::layout::symbol_node::WORDS as u64 * 4 == NODE_UNIFORM_SIZE);

/// Words a row holds.
const STRIDE: usize = (NODE_UNIFORM_SIZE / 4) as usize;

/// A buffer of node uniforms and its words as the next flush leaves it.
#[derive(Debug)]
pub(crate) struct NodeUniforms {
    pub(crate) buffer: BufferId,
    label: &'static str,
    usage: BufferUsages,
    words: Vec<u32>,
    /// Rows written since the last flush.
    dirty: Vec<u32>,
}

impl NodeUniforms {
    /// `rows` rows of zeros in a buffer of `usage`.
    pub(crate) fn new(
        gpu: &mut dyn Gpu,
        label: &'static str,
        rows: u32,
        usage: BufferUsages,
    ) -> Self {
        let buffer = Self::buffer(gpu, label, rows, usage);
        Self {
            buffer,
            label,
            usage,
            words: vec![0; STRIDE * rows as usize],
            dirty: Vec::new(),
        }
    }

    fn buffer(gpu: &mut dyn Gpu, label: &'static str, rows: u32, usage: BufferUsages) -> BufferId {
        gpu.create_buffer(&BufferDesc {
            label,
            size: NODE_UNIFORM_SIZE * u64::from(rows),
            usage,
        })
    }

    /// Grows to `rows` rows in a new buffer, which the next flush fills:
    /// the old one, to let go of once nothing binds it.
    pub(crate) fn grow(&mut self, gpu: &mut dyn Gpu, rows: u32) -> BufferId {
        let buffer = Self::buffer(gpu, self.label, rows, self.usage);
        self.words.resize(STRIDE * rows as usize, 0);
        self.dirty = (0..rows).collect();
        std::mem::replace(&mut self.buffer, buffer)
    }

    /// Writes `words` at `row`: whether they changed.
    pub(crate) fn set(&mut self, row: u32, words: &[u32]) -> bool {
        let at = &mut self.words[row as usize * STRIDE..][..words.len()];
        if at == words {
            return false;
        }
        at.copy_from_slice(words);
        self.dirty.push(row);
        true
    }

    /// Writes word `at` of `row`.
    pub(crate) fn set_word(&mut self, row: u32, at: usize, word: u32) {
        let mut words = self.row(row).to_vec();
        words[at] = word;
        self.set(row, &words);
    }

    /// Row `row`'s words.
    pub(crate) fn row(&self, row: u32) -> &[u32] {
        &self.words[row as usize * STRIDE..][..STRIDE]
    }

    /// Row `row` and, with `next`, the row after it, to bind.
    pub(crate) fn binding(&self, row: u32, next: bool) -> BindingResource {
        BindingResource::Buffer {
            buffer: self.buffer,
            offset: u64::from(row) * NODE_UNIFORM_SIZE,
            size: Some(NODE_UNIFORM_SIZE * (1 + u64::from(next))),
        }
    }

    /// Writes the rows changed since the last flush, a write a run of
    /// them.
    pub(crate) fn flush(&mut self, gpu: &mut dyn Gpu) {
        self.dirty.sort_unstable();
        self.dirty.dedup();
        for run in self.dirty.chunk_by(|a, b| *b == a + 1) {
            let rows = run[0] as usize..run[run.len() - 1] as usize + 1;
            let words = &self.words[rows.start * STRIDE..rows.end * STRIDE];
            let offset = rows.start as u64 * NODE_UNIFORM_SIZE;
            gpu.write_buffer(self.buffer, offset, bytemuck::cast_slice(words));
        }
        self.dirty.clear();
    }
}

impl Release for NodeUniforms {
    fn release(self, gpu: &mut dyn Gpu) {
        self.buffer.release(gpu);
    }
}
