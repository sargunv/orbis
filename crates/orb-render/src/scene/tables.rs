//! The scene's tables as shaders read them: placements, chunks,
//! candidates, node rows, visible lists and the vertex, index and property
//! arenas.
//!
//! Every write to them, and every move into a larger buffer, goes through
//! [`Tables`], so where they live is one decision: storage buffers on the
//! GPU-driven path, which its cull and draw stages bind. The WebGL 2 path
//! (#140) has no storage buffers in the vertex stage and reads data
//! textures (`fetch_texture.wgsl`), a second backing behind the same
//! writes. The image atlas's rect table is its own and stays outside.

use orb_hal::{BufferDesc, BufferId, BufferUsages, Gpu};

use super::{SceneBuffers, Words, move_buffer};

/// A table of the scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Table {
    Tiles,
    Chunks,
    Candidates,
    NodeCull,
    Visible,
    Vertices,
    Indices,
    Props,
}

impl Table {
    fn label(self) -> &'static str {
        match self {
            Self::Tiles => "tiles",
            Self::Chunks => "chunks",
            Self::Candidates => "candidates",
            Self::NodeCull => "node cull",
            Self::Visible => "visible",
            Self::Vertices => "vertices",
            Self::Indices => "indices",
            Self::Props => "props",
        }
    }

    /// Its buffer's usage on the storage backing: written from the CPU,
    /// and copied from where it grows by moving (the pools) or the cull
    /// writes what tests read back (candidates and visible lists).
    fn usage(self) -> BufferUsages {
        let written = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        match self {
            Self::Tiles | Self::NodeCull => written,
            Self::Chunks
            | Self::Candidates
            | Self::Visible
            | Self::Vertices
            | Self::Indices
            | Self::Props => written | BufferUsages::COPY_SRC,
        }
    }
}

/// Where the tables live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backing {
    /// Storage buffers, the GPU-driven path's.
    Storage,
}

/// One table, to write without the rest: what the glyph atlas holds of
/// the property table its rows open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TableWriter {
    backing: Backing,
    buffer: BufferId,
}

impl TableWriter {
    /// Queues `words` into the table from word `at`.
    pub(crate) fn write(&self, gpu: &mut dyn Gpu, at: u32, words: Words<'_>) {
        match self.backing {
            Backing::Storage => words.write_buffer(gpu, self.buffer, at),
        }
    }

    /// Queues core `words` into the table from word `at`.
    pub(crate) fn write_words(&self, gpu: &mut dyn Gpu, at: u32, words: &[u32]) {
        self.write(gpu, at, Words::Core(words));
    }
}

/// The scene's tables and their buffers.
#[derive(Debug)]
pub(crate) struct Tables {
    backing: Backing,
    /// What binds them, with the frame block, the draw arguments and the
    /// compacted indices the GPU driver writes.
    pub(super) buffers: SceneBuffers,
}

impl Tables {
    /// Tables in storage buffers, made by [`create`](Self::create).
    pub(super) fn storage(buffers: SceneBuffers) -> Self {
        Self {
            backing: Backing::Storage,
            buffers,
        }
    }

    /// A storage buffer for `table`, `words` long (at least four).
    pub(super) fn create(gpu: &mut dyn Gpu, table: Table, words: u64) -> BufferId {
        gpu.create_buffer(&BufferDesc {
            label: table.label(),
            size: (words * 4).max(16),
            usage: table.usage(),
        })
    }

    pub(crate) fn writer(&self, table: Table) -> TableWriter {
        let mut buffers = self.buffers;
        TableWriter {
            backing: self.backing,
            buffer: *slot(&mut buffers, table),
        }
    }

    /// Queues `words` into `table` from word `at`.
    pub(crate) fn write(&self, gpu: &mut dyn Gpu, table: Table, at: u32, words: Words<'_>) {
        self.writer(table).write(gpu, at, words);
    }

    /// Queues core `words` into `table` from word `at`.
    pub(crate) fn write_words(&self, gpu: &mut dyn Gpu, table: Table, at: u32, words: &[u32]) {
        self.write(gpu, table, at, Words::Core(words));
    }

    /// Moves `table` into a new one `words` long, its first `keep` words
    /// copied over on the GPU at the same offsets. Writers taken before
    /// write the old one.
    pub(super) fn resize(&mut self, gpu: &mut dyn Gpu, table: Table, words: u64, keep: u64) {
        let desc = BufferDesc {
            label: table.label(),
            size: (words * 4).max(16),
            usage: table.usage(),
        };
        move_buffer(gpu, slot(&mut self.buffers, table), &desc, keep * 4);
    }
}

fn slot(b: &mut SceneBuffers, table: Table) -> &mut BufferId {
    match table {
        Table::Tiles => &mut b.tiles,
        Table::Chunks => &mut b.chunks,
        Table::Candidates => &mut b.candidates,
        Table::NodeCull => &mut b.node_cull,
        Table::Visible => &mut b.visible,
        Table::Vertices => &mut b.vertices,
        Table::Indices => &mut b.indices,
        Table::Props => &mut b.props,
    }
}
