//! The scene's tables as shaders read them: placements, chunks,
//! candidates, node rows, visible lists, the property arena, and each
//! block's vertex and index pools.
//!
//! The vertex and index pools are blocks of fixed size (`BlockPool`): a
//! scene that needs more adds a block rather than moving the ones it has,
//! so growing copies nothing. A block's meshes draw with the block bound,
//! each node once per block it has candidates in.
//!
//! Every write to them, and every move into a larger one, goes through
//! [`Tables`], so where they live is one decision, the device's: storage
//! buffers where the vertex stage reads them, which the GPU driver's cull
//! and the draw stages bind (`fetch.wgsl`), or data textures where it
//! can't (WebGL 2, #140; `fetch_texture.wgsl`), each table in its own at
//! the same binding, so a table that grows moves only its own words. The
//! CPU driver also keeps copies of what its cull reads.
//!
//! Writes into the data textures wait for [`Tables::flush`], which issues
//! them as one write per run of adjacent words: each is a
//! `texSubImage2D` on WebGL 2, and a tile's chunks write their rows one by
//! one.

use orb_hal::*;

use super::{Words, move_buffer};

/// Words per row of a data texture on a device with `limits`
/// (`data_texture.wgsl`'s `data_width_log2`): 4096, or WebGL 2's least
/// texture side, 2048, on a device whose side is shorter (pipelines set
/// `orb_shaders::DATA_WIDTH_LOG2` for it).
#[cfg(cpu_driven)]
pub(crate) fn data_width(limits: &Limits) -> u32 {
    if limits.max_texture_dimension_2d >= 1 << 12 {
        1 << 12
    } else {
        1 << 11
    }
}

/// A table of the scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Table {
    Tiles,
    Chunks,
    Candidates,
    NodeCull,
    Visible,
    Props,
}

const TABLES: usize = 6;

/// A pool of a scene block: the words its meshes draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockPool {
    Vertices,
    Indices,
}

impl BlockPool {
    pub(crate) const ALL: [Self; 2] = [Self::Vertices, Self::Indices];
    /// A pool's buffer's usage: the block for what draws as points moves
    /// into others, copying from it.
    const USAGE: BufferUsages = BufferUsages::STORAGE
        .union(BufferUsages::COPY_DST)
        .union(BufferUsages::COPY_SRC);

    fn label(self) -> &'static str {
        match self {
            Self::Vertices => "vertices",
            Self::Indices => "indices",
        }
    }

    /// The scene group's binding of a block's pool (`fetch.wgsl`).
    fn binding(self) -> u32 {
        match self {
            Self::Vertices => 4,
            Self::Indices => 5,
        }
    }
}

impl Table {
    const ALL: [Self; TABLES] = [
        Self::Tiles,
        Self::Chunks,
        Self::Candidates,
        Self::NodeCull,
        Self::Visible,
        Self::Props,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Tiles => "tiles",
            Self::Chunks => "chunks",
            Self::Candidates => "candidates",
            Self::NodeCull => "node cull",
            Self::Visible => "visible",
            Self::Props => "props",
        }
    }

    /// Its buffer's usage on the storage backing: written from the CPU,
    /// and copied from where it grows by moving (the chunks and property
    /// rows) or the cull writes what tests read back (candidates and
    /// visible lists).
    fn usage(self) -> BufferUsages {
        let written = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        match self {
            Self::Tiles | Self::NodeCull => written,
            Self::Chunks | Self::Candidates | Self::Visible | Self::Props => {
                written | BufferUsages::COPY_SRC
            }
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Where the tables live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backing {
    /// Storage buffers, where the vertex stage reads them.
    Storage,
    /// Data textures, where it can't (WebGL 2).
    #[cfg(cpu_driven)]
    Texture,
}

impl Backing {
    /// The device's.
    pub(crate) fn of(caps: &Caps) -> Self {
        #[cfg(cpu_driven)]
        if !caps.vertex_storage {
            return Self::Texture;
        }
        assert!(
            caps.vertex_storage,
            "this build keeps no tables in textures (#140)"
        );
        Self::Storage
    }

    /// Words a table may hold on this backing at most.
    pub(crate) fn largest(self, limits: &Limits) -> u32 {
        let words = match self {
            Self::Storage => {
                limits
                    .max_storage_buffer_binding_size
                    .min(limits.max_buffer_size)
                    / 4
            }
            #[cfg(cpu_driven)]
            Self::Texture => {
                u64::from(data_width(limits)) * u64::from(limits.max_texture_dimension_2d)
            }
        };
        words.min(u64::from(u32::MAX)) as u32
    }
}

/// A table's place on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    Buffer(BufferId),
    /// A data texture.
    #[cfg(cpu_driven)]
    Texture(TextureId),
    /// Only the CPU reads it: the texture backing's candidates.
    #[cfg(cpu_driven)]
    Cpu,
}

impl Place {
    /// Its entry at `binding`: its buffer or texture.
    fn entry(self, binding: u32) -> BindGroupEntry {
        match self {
            Self::Buffer(buffer) => BindGroupEntry::buffer(binding, buffer),
            #[cfg(cpu_driven)]
            Self::Texture(texture) => {
                BindGroupEntry::texture(binding, texture, TextureViewDimension::D2)
            }
            #[cfg(cpu_driven)]
            Self::Cpu => panic!("a table only the CPU reads has no binding"),
        }
    }

    /// Frees its buffer or texture.
    fn destroy(self, gpu: &mut dyn Gpu) {
        match self {
            Self::Buffer(buffer) => gpu.destroy_buffer(buffer),
            #[cfg(cpu_driven)]
            Self::Texture(texture) => gpu.destroy_texture(texture),
            #[cfg(cpu_driven)]
            Self::Cpu => {}
        }
    }

    /// Queues `words` from word `at`: into a texture, in `staged` if
    /// given, else at once.
    fn write(
        self,
        gpu: &mut dyn Gpu,
        at: u32,
        words: Words<'_>,
        #[cfg(cpu_driven)] staged: Option<&mut Staged>,
    ) {
        match self {
            Self::Buffer(buffer) => words.write_buffer(gpu, buffer, at),
            #[cfg(cpu_driven)]
            Self::Texture(texture) => {
                let host;
                let words = match words {
                    Words::Core(words) => words,
                    Words::Host { bytes, offset, len } => {
                        let bytes = gpu.read_host_bytes(bytes, offset..offset + u64::from(len) * 4);
                        // Host bytes come unaligned (copied from a JS buffer).
                        host = bytemuck::pod_collect_to_vec::<u8, u32>(&bytes);
                        &host
                    }
                };
                match staged {
                    Some(staged) => staged.push(texture, at, words),
                    None => write_texels(gpu, texture, at, words),
                }
            }
            #[cfg(cpu_driven)]
            Self::Cpu => {}
        }
    }
}

/// Writes into data textures, held to be issued together.
#[cfg(cpu_driven)]
#[derive(Debug, Default)]
struct Staged {
    /// In the order made: the texture, the first word, and where its words
    /// are in `words`.
    writes: Vec<(TextureId, u32, std::ops::Range<usize>)>,
    words: Vec<u32>,
}

/// Words of held writes whose allocation outlives a flush.
#[cfg(cpu_driven)]
const KEPT_WORDS: usize = 1 << 18;

#[cfg(cpu_driven)]
impl Staged {
    fn push(&mut self, texture: TextureId, at: u32, words: &[u32]) {
        if words.is_empty() {
            return;
        }
        let start = self.words.len();
        self.words.extend_from_slice(words);
        self.writes.push((texture, at, start..self.words.len()));
    }

    /// Issues the held writes: one per run of a texture's words that they
    /// cover without a gap, the later of two writes winning where they
    /// overlap.
    fn flush(&mut self, gpu: &mut dyn Gpu) {
        let Self { writes, words } = self;
        let mut order: Vec<usize> = (0..writes.len()).collect();
        order.sort_unstable_by_key(|&i| (writes[i].0, writes[i].1, i));
        let end = |i: usize| writes[i].1 + writes[i].2.len() as u32;
        let mut run = Vec::new();
        let mut at = 0;
        while at < order.len() {
            let (texture, start, _) = writes[order[at]];
            let mut stop = end(order[at]);
            let mut next = at + 1;
            while next < order.len() && writes[order[next]].0 == texture {
                let (_, from, _) = writes[order[next]];
                if from > stop {
                    break;
                }
                stop = stop.max(end(order[next]));
                next += 1;
            }
            let members = &mut order[at..next];
            if let [only] = members {
                write_texels(gpu, texture, start, &words[writes[*only].2.clone()]);
            } else {
                // In the order they were made, so later words win.
                members.sort_unstable();
                run.clear();
                run.resize((stop - start) as usize, 0);
                for &i in members.iter() {
                    let (_, from, ref range) = writes[i];
                    let from = (from - start) as usize;
                    run[from..from + range.len()].copy_from_slice(&words[range.clone()]);
                }
                write_texels(gpu, texture, start, &run);
            }
            at = next;
        }
        writes.clear();
        words.clear();
        words.shrink_to(KEPT_WORDS);
    }
}

#[cfg(cpu_driven)]
/// Queues `words` into a data texture from word `at`: what's left of
/// its row, the whole rows after, then the start of the last.
fn write_texels(gpu: &mut dyn Gpu, texture: TextureId, at: u32, words: &[u32]) {
    let row = data_width(&gpu.caps().limits);
    let (mut at, mut words) = (at, words);
    while !words.is_empty() {
        let x = at % row;
        let (width, height) = if x == 0 && words.len() >= row as usize {
            (row, words.len() as u32 / row)
        } else {
            ((row - x).min(words.len() as u32), 1)
        };
        let n = (width * height) as usize;
        gpu.write_texture(
            &TextureRegion {
                texture,
                x,
                y: at / row,
                layer: 0,
                width,
                height,
                level: 0,
            },
            bytemuck::cast_slice(&words[..n]),
        );
        at += n as u32;
        words = &words[n..];
    }
}

/// The texel copies that move runs of words (`from`, `to`, `len`) between
/// data textures `width` words wide: whole rows where a run's ends line up
/// with them in both, else pieces that cross a row in neither.
#[cfg(cpu_driven)]
fn texel_copies(runs: &[(u32, u32, u32)], width: u32) -> Vec<TextureCopy> {
    let mut copies = Vec::new();
    for &(mut at, mut to, mut len) in runs {
        while len > 0 {
            let (x, to_x) = (at % width, to % width);
            let (w, h) = if x == 0 && to_x == 0 && len >= width {
                (width, len / width)
            } else {
                ((width - x).min(width - to_x).min(len), 1)
            };
            copies.push(TextureCopy {
                from: [x, at / width, 0],
                to: [to_x, to / width, 0],
                size: [w, h, 1],
            });
            (at, to, len) = (at + w * h, to + w * h, len - w * h);
        }
    }
    copies
}

/// A data texture of `rows` rows.
#[cfg(cpu_driven)]
fn data_texture(gpu: &mut dyn Gpu, label: &str, rows: u32) -> TextureId {
    let width = data_width(&gpu.caps().limits);
    gpu.create_texture(&TextureDesc {
        label,
        width,
        height: rows.max(1),
        array_layers: 1,
        dimension: TextureViewDimension::D2,
        format: TextureFormat::R32Uint,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC,
        sample_count: 1,
        mip_levels: 1,
    })
}

/// Rows of a data texture `width` words wide that `words` take.
#[cfg(cpu_driven)]
pub(super) fn rows(words: u64, width: u32) -> u32 {
    words.div_ceil(u64::from(width)).max(1) as u32
}

/// A table on its own, out of the scene's: the image atlas's rects, the
/// terrain's tiles and grid. A storage buffer or a data texture, as the
/// device's backing has it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DataTable {
    place: TablePlace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TablePlace {
    Buffer(BufferId),
    #[cfg(cpu_driven)]
    Texture(TextureId),
}

impl DataTable {
    /// A table of `words` words (at least four) on the device's backing.
    pub(crate) fn new(gpu: &mut dyn Gpu, label: &str, words: u64) -> Self {
        let place = match Backing::of(gpu.caps()) {
            Backing::Storage => TablePlace::Buffer(gpu.create_buffer(&BufferDesc {
                label,
                size: (words * 4).max(16),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            })),
            #[cfg(cpu_driven)]
            Backing::Texture => {
                let rows = rows(words, data_width(&gpu.caps().limits));
                TablePlace::Texture(data_texture(gpu, label, rows))
            }
        };
        Self { place }
    }

    /// Queues core `words` into the table from word `at`.
    pub(crate) fn write(self, gpu: &mut dyn Gpu, at: u32, words: &[u32]) {
        match self.place {
            TablePlace::Buffer(buffer) => {
                gpu.write_buffer(buffer, u64::from(at) * 4, bytemuck::cast_slice(words));
            }
            #[cfg(cpu_driven)]
            TablePlace::Texture(texture) => write_texels(gpu, texture, at, words),
        }
    }

    /// Its binding in a layout, visible to `stages`.
    pub(crate) fn layout_entry(
        backing: Backing,
        binding: u32,
        stages: ShaderStages,
    ) -> BindGroupLayoutEntry {
        match backing {
            Backing::Storage => BindGroupLayoutEntry {
                binding,
                visibility: stages,
                ty: BindingType::Storage { read_only: true },
            },
            #[cfg(cpu_driven)]
            Backing::Texture => BindGroupLayoutEntry::data_texture(binding, stages),
        }
    }

    /// Its binding in a group.
    pub(crate) fn entry(self, binding: u32) -> BindGroupEntry {
        match self.place {
            TablePlace::Buffer(buffer) => BindGroupEntry::buffer(binding, buffer),
            #[cfg(cpu_driven)]
            TablePlace::Texture(texture) => {
                BindGroupEntry::texture(binding, texture, TextureViewDimension::D2)
            }
        }
    }

    pub(crate) fn destroy(self, gpu: &mut dyn Gpu) {
        match self.place {
            TablePlace::Buffer(buffer) => gpu.destroy_buffer(buffer),
            #[cfg(cpu_driven)]
            TablePlace::Texture(texture) => gpu.destroy_texture(texture),
        }
    }
}

/// One table, to write without the rest: what the glyph atlas holds of
/// the property table its rows open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TableWriter {
    place: Place,
}

impl TableWriter {
    /// Queues `words` into the table from word `at`.
    pub(crate) fn write(self, gpu: &mut dyn Gpu, at: u32, words: Words<'_>) {
        self.place.write(
            gpu,
            at,
            words,
            #[cfg(cpu_driven)]
            None,
        );
    }

    /// Queues core `words` into the table from word `at`.
    pub(crate) fn write_words(self, gpu: &mut dyn Gpu, at: u32, words: &[u32]) {
        self.write(gpu, at, Words::Core(words));
    }
}

/// The scene's tables.
#[derive(Debug)]
pub(crate) struct Tables {
    backing: Backing,
    places: [Place; TABLES],
    /// Each block's vertex and index pools, in [`BlockPool`]'s order, by
    /// block slot; `None` where a block was freed.
    blocks: Vec<Option<[Place; 2]>>,
    #[cfg(cpu_driven)]
    mirrors: Mirrors,
    /// Writes into the data textures since the last flush.
    #[cfg(cpu_driven)]
    staged: Staged,
}

/// Copies of the tables the CPU driver reads, each as long as what was
/// written: words past the end are zero.
#[cfg(cpu_driven)]
#[derive(Debug, Default)]
struct Mirrors {
    tables: [Option<Vec<u32>>; TABLES],
    /// Each block's index pool, by block slot, when they're copied.
    indices: Option<Vec<Vec<u32>>>,
    /// Changes with every write to one.
    revision: u64,
}

#[cfg(cpu_driven)]
impl Mirrors {
    fn write(&mut self, gpu: &mut dyn Gpu, table: Table, at: u32, words: Words<'_>) {
        let Some(mirror) = &mut self.tables[table.index()] else {
            return;
        };
        self.revision += 1;
        Self::put(gpu, mirror, at, words);
    }

    /// Copies a write into block `block`'s pool, if its copy is kept.
    fn write_block(
        &mut self,
        gpu: &mut dyn Gpu,
        (block, pool): (usize, BlockPool),
        at: u32,
        words: Words<'_>,
    ) {
        let Some(blocks) = self.indices.as_mut().filter(|_| pool == BlockPool::Indices) else {
            return;
        };
        if blocks.len() <= block {
            blocks.resize_with(block + 1, Vec::new);
        }
        self.revision += 1;
        Self::put(gpu, &mut blocks[block], at, words);
    }

    fn put(gpu: &mut dyn Gpu, mirror: &mut Vec<u32>, at: u32, words: Words<'_>) {
        let at = at as usize;
        let end = at + words.len() as usize;
        if mirror.len() < end {
            mirror.resize(end, 0);
        }
        match words {
            Words::Core(words) => mirror[at..end].copy_from_slice(words),
            Words::Host { bytes, offset, len } => {
                let bytes = gpu.read_host_bytes(bytes, offset..offset + u64::from(len) * 4);
                bytemuck::cast_slice_mut(&mut mirror[at..end]).copy_from_slice(&bytes);
            }
        }
    }

    fn truncate(&mut self, table: Table, keep: u64) {
        if let Some(mirror) = &mut self.tables[table.index()] {
            mirror.truncate(keep as usize);
            self.revision += 1;
        }
    }
}

impl Tables {
    /// Tables of `words` words each, in [`Table`]'s order, on `backing`,
    /// with copies of `mirrored` for the CPU to read.
    #[cfg_attr(not(cpu_driven), allow(unused_variables))]
    pub(super) fn new(
        gpu: &mut dyn Gpu,
        backing: Backing,
        words: [u64; TABLES],
        (mirrored, block_indices): (&[Table], bool),
    ) -> Self {
        let places = Table::ALL.map(|table| match backing {
            Backing::Storage => Place::Buffer(gpu.create_buffer(&BufferDesc {
                label: table.label(),
                size: (words[table.index()] * 4).max(16),
                usage: table.usage(),
            })),
            #[cfg(cpu_driven)]
            Backing::Texture if table == Table::Candidates => Place::Cpu,
            #[cfg(cpu_driven)]
            Backing::Texture => {
                let rows = rows(words[table.index()], data_width(&gpu.caps().limits));
                Place::Texture(data_texture(gpu, table.label(), rows))
            }
        });
        #[cfg(cpu_driven)]
        let mut mirrors = Mirrors::default();
        #[cfg(cpu_driven)]
        for table in mirrored {
            mirrors.tables[table.index()] = Some(Vec::new());
        }
        #[cfg(cpu_driven)]
        if block_indices {
            mirrors.indices = Some(Vec::new());
        }
        Self {
            backing,
            places,
            blocks: Vec::new(),
            #[cfg(cpu_driven)]
            mirrors,
            #[cfg(cpu_driven)]
            staged: Staged::default(),
        }
    }

    pub(crate) fn backing(&self) -> Backing {
        self.backing
    }

    /// A writer of `table`, which no copy is kept of.
    pub(crate) fn writer(&self, table: Table) -> TableWriter {
        #[cfg(cpu_driven)]
        debug_assert!(self.mirrors.tables[table.index()].is_none());
        TableWriter {
            place: self.places[table.index()],
        }
    }

    /// Queues `words` into `table` from word `at`.
    pub(crate) fn write(&mut self, gpu: &mut dyn Gpu, table: Table, at: u32, words: Words<'_>) {
        #[cfg(cpu_driven)]
        self.mirrors.write(gpu, table, at, words);
        self.places[table.index()].write(
            gpu,
            at,
            words,
            #[cfg(cpu_driven)]
            Some(&mut self.staged),
        );
    }

    /// Makes block slot `block`'s pools, `words` long in [`BlockPool`]'s
    /// order, on the tables' backing.
    pub(super) fn add_block(&mut self, gpu: &mut dyn Gpu, block: usize, words: [u64; 2]) {
        let places = [0, 1].map(|i| {
            let pool = BlockPool::ALL[i];
            match self.backing {
                Backing::Storage => Place::Buffer(gpu.create_buffer(&BufferDesc {
                    label: pool.label(),
                    size: (words[i] * 4).max(16),
                    usage: BlockPool::USAGE,
                })),
                #[cfg(cpu_driven)]
                Backing::Texture => {
                    let rows = rows(words[i], data_width(&gpu.caps().limits));
                    Place::Texture(data_texture(gpu, pool.label(), rows))
                }
            }
        });
        if self.blocks.len() <= block {
            self.blocks.resize(block + 1, None);
        }
        assert!(self.blocks[block].is_none(), "block slot {block} is taken");
        self.blocks[block] = Some(places);
    }

    /// Frees block slot `block`'s pools.
    pub(super) fn remove_block(&mut self, gpu: &mut dyn Gpu, block: usize) {
        #[cfg(cpu_driven)]
        self.flush(gpu);
        let places = self.blocks[block].take().expect("a block's pools");
        for place in places {
            place.destroy(gpu);
        }
        #[cfg(cpu_driven)]
        if let Some(mirror) = self.mirrors.indices.as_mut().and_then(|m| m.get_mut(block)) {
            *mirror = Vec::new();
            self.mirrors.revision += 1;
        }
    }

    /// Moves pool `pool` of block `block` into one `words` long, its first
    /// `keep` words copied over on the GPU at the same offsets (the copy
    /// the CPU keeps of an index pool stays as it is).
    pub(super) fn resize_block(
        &mut self,
        gpu: &mut dyn Gpu,
        (block, pool): (usize, BlockPool),
        words: u64,
        keep: u64,
    ) {
        #[cfg(cpu_driven)]
        self.flush(gpu);
        let place = &mut self.blocks[block].as_mut().expect("a block's pools")[pool as usize];
        match *place {
            Place::Buffer(mut buffer) => {
                let desc = BufferDesc {
                    label: pool.label(),
                    size: (words * 4).max(16),
                    usage: BlockPool::USAGE,
                };
                move_buffer(gpu, &mut buffer, &desc, keep * 4);
                *place = Place::Buffer(buffer);
            }
            #[cfg(cpu_driven)]
            Place::Texture(texture) => {
                let width = data_width(&gpu.caps().limits);
                let moved = data_texture(gpu, pool.label(), rows(words, width));
                let kept = match keep {
                    0 => 0,
                    _ => rows(keep, width).min(rows(words, width)),
                };
                if kept > 0 {
                    gpu.submit(&[Pass::CopyTexture {
                        source: texture,
                        destination: moved,
                        copies: vec![TextureCopy {
                            from: [0; 3],
                            to: [0; 3],
                            size: [width, kept, 1],
                        }],
                    }]);
                }
                gpu.destroy_texture(texture);
                *place = Place::Texture(moved);
            }
            #[cfg(cpu_driven)]
            Place::Cpu => {}
        }
    }

    /// Adds to `passes`, for the caller to submit, the copies of runs of
    /// pool `pool`'s words from block `source` into block `destination`,
    /// each `(from, to, len)` in words; the copy the CPU keeps of an index
    /// pool takes them at once. How meshes move out of a block a view left
    /// mostly empty.
    #[cfg_attr(not(cpu_driven), allow(unused_variables))]
    pub(super) fn copy_between_blocks(
        &mut self,
        gpu: &mut dyn Gpu,
        pool: BlockPool,
        (source, destination): (usize, usize),
        runs: &[(u32, u32, u32)],
        passes: &mut Vec<Pass>,
    ) {
        #[cfg(cpu_driven)]
        self.flush(gpu);
        let place = |block: usize| self.blocks[block].expect("a block's pools")[pool as usize];
        match (place(source), place(destination)) {
            (Place::Buffer(from), Place::Buffer(to)) => {
                passes.extend(runs.iter().map(|&(at, to_at, len)| Pass::CopyBuffer {
                    source: from,
                    source_offset: u64::from(at) * 4,
                    destination: to,
                    destination_offset: u64::from(to_at) * 4,
                    size: u64::from(len) * 4,
                }));
            }
            #[cfg(cpu_driven)]
            (Place::Texture(from), Place::Texture(to)) => {
                let width = data_width(&gpu.caps().limits);
                passes.push(Pass::CopyTexture {
                    source: from,
                    destination: to,
                    copies: texel_copies(runs, width),
                });
            }
            #[cfg(cpu_driven)]
            places => panic!("block pools on different backings: {places:?}"),
        }
        #[cfg(cpu_driven)]
        if pool == BlockPool::Indices
            && let Some(mirrors) = self.mirrors.indices.as_mut()
        {
            if mirrors.len() <= source.max(destination) {
                mirrors.resize_with(source.max(destination) + 1, Vec::new);
            }
            let from = std::mem::take(&mut mirrors[source]);
            let to = &mut mirrors[destination];
            for &(at, to_at, len) in runs {
                let (at, to_at, len) = (at as usize, to_at as usize, len as usize);
                if to.len() < to_at + len {
                    to.resize(to_at + len, 0);
                }
                // Words past the end of a copy are zero.
                let kept = from.len().clamp(at, at + len);
                to[to_at..to_at + kept - at].copy_from_slice(&from[at..kept]);
                to[to_at + kept - at..to_at + len].fill(0);
            }
            mirrors[source] = from;
            self.mirrors.revision += 1;
        }
    }

    /// Queues `words` into pool `pool` of block `block` from word `at`.
    pub(crate) fn write_block(
        &mut self,
        gpu: &mut dyn Gpu,
        (block, pool): (usize, BlockPool),
        at: u32,
        words: Words<'_>,
    ) {
        #[cfg(cpu_driven)]
        self.mirrors.write_block(gpu, (block, pool), at, words);
        let place = self.blocks[block].expect("a block's pools")[pool as usize];
        place.write(
            gpu,
            at,
            words,
            #[cfg(cpu_driven)]
            Some(&mut self.staged),
        );
    }

    /// The storage buffer of block `block`'s pool `pool`.
    pub(crate) fn block_buffer(&self, block: usize, pool: BlockPool) -> BufferId {
        match self.blocks[block].expect("a block's pools")[pool as usize] {
            Place::Buffer(buffer) => buffer,
            #[cfg(cpu_driven)]
            place => panic!("{} has no buffer: {place:?}", pool.label()),
        }
    }

    /// The copy of block `block`'s index pool, if they're kept.
    #[cfg(cpu_driven)]
    pub(crate) fn block_indices(&self, block: usize) -> Option<&[u32]> {
        let blocks = self.mirrors.indices.as_ref()?;
        Some(blocks.get(block).map_or(&[][..], Vec::as_slice))
    }

    /// Issues the writes into data textures held since the last flush:
    /// before a submission that reads the tables, or moves one.
    #[cfg(cpu_driven)]
    pub(crate) fn flush(&mut self, gpu: &mut dyn Gpu) {
        self.staged.flush(gpu);
    }

    /// Queues core `words` into `table` from word `at`.
    pub(crate) fn write_words(&mut self, gpu: &mut dyn Gpu, table: Table, at: u32, words: &[u32]) {
        self.write(gpu, table, at, Words::Core(words));
    }

    /// The copy of `table` the CPU reads, if it keeps one.
    #[cfg(cpu_driven)]
    pub(crate) fn mirror(&self, table: Table) -> Option<&[u32]> {
        self.mirrors.tables[table.index()].as_deref()
    }

    /// The copy of `table`, for the CPU driver to write what only it
    /// reads (candidates' drawn words), leaving the GPU's as it is.
    #[cfg(cpu_driven)]
    pub(crate) fn mirror_mut(&mut self, table: Table) -> Option<&mut Vec<u32>> {
        self.mirrors.tables[table.index()].as_mut()
    }

    /// Forgets the copy of `table` past its first `keep` words, which say
    /// nothing any more.
    #[cfg(cpu_driven)]
    pub(crate) fn truncate_mirror(&mut self, table: Table, keep: u64) {
        self.mirrors.truncate(table, keep);
    }

    /// Changes with every write to a table the CPU keeps a copy of.
    #[cfg(cpu_driven)]
    pub(crate) fn revision(&self) -> u64 {
        self.mirrors.revision
    }

    /// Bytes of the heap the copies of tables take.
    pub(crate) fn mirror_bytes(&self) -> u64 {
        #[cfg(cpu_driven)]
        let bytes = (self.mirrors.tables.iter().flatten())
            .chain(self.mirrors.indices.iter().flatten())
            .map(|m| m.capacity() as u64 * 4)
            .sum();
        #[cfg(not(cpu_driven))]
        let bytes = 0;
        bytes
    }

    /// The storage buffer of `table`, on the storage backing.
    pub(crate) fn buffer(&self, table: Table) -> BufferId {
        match self.places[table.index()] {
            Place::Buffer(buffer) => buffer,
            #[cfg(cpu_driven)]
            place => panic!("{} has no buffer: {place:?}", table.label()),
        }
    }

    /// Moves `table` into a new one `words` long, its first `keep` words
    /// copied over on the GPU at the same offsets. Writers taken before
    /// write the old one.
    pub(super) fn resize(&mut self, gpu: &mut dyn Gpu, table: Table, words: u64, keep: u64) {
        #[cfg(cpu_driven)]
        self.flush(gpu);
        #[cfg(cpu_driven)]
        self.mirrors.truncate(table, keep);
        let place = &mut self.places[table.index()];
        match *place {
            Place::Buffer(mut buffer) => {
                let desc = BufferDesc {
                    label: table.label(),
                    size: (words * 4).max(16),
                    usage: table.usage(),
                };
                move_buffer(gpu, &mut buffer, &desc, keep * 4);
                *place = Place::Buffer(buffer);
            }
            #[cfg(cpu_driven)]
            Place::Texture(texture) => {
                let width = data_width(&gpu.caps().limits);
                let moved = data_texture(gpu, table.label(), rows(words, width));
                let kept = if keep == 0 {
                    0
                } else {
                    rows(keep, width).min(rows(words, width))
                };
                if kept > 0 {
                    gpu.submit(&[Pass::CopyTexture {
                        source: texture,
                        destination: moved,
                        copies: vec![TextureCopy {
                            from: [0; 3],
                            to: [0; 3],
                            size: [width, kept, 1],
                        }],
                    }]);
                }
                gpu.destroy_texture(texture);
                *place = Place::Texture(moved);
            }
            #[cfg(cpu_driven)]
            Place::Cpu => {}
        }
    }

    /// The scene group's entries for the tables (bindings 1 to 7; the
    /// image atlas's rects are 8), visible to the stages that read them.
    pub(crate) fn layout_entries(backing: Backing) -> Vec<BindGroupLayoutEntry> {
        let (vs, fs) = (ShaderStages::VERTEX, ShaderStages::FRAGMENT);
        let vf = vs | fs;
        let table = |binding, stages| DataTable::layout_entry(backing, binding, stages);
        vec![
            table(1, vf),
            table(2, vs),
            table(3, vs),
            table(4, vs),
            table(5, vs),
            // Extrusions' fragments read colours over the elevation.
            table(6, vf),
            table(7, vs),
        ]
    }

    /// The tables' entries in block `block`'s scene group (bindings 1 to
    /// 7): its pools, and the tables every block shares.
    pub(crate) fn entries(&self, block: usize) -> Vec<BindGroupEntry> {
        let entry = |binding, table: Table| self.entry(table, binding);
        let pools = self.blocks[block].expect("a block's pools");
        let [vertices, indices] =
            BlockPool::ALL.map(|pool| pools[pool as usize].entry(pool.binding()));
        vec![
            entry(1, Table::Tiles),
            entry(2, Table::Chunks),
            entry(3, Table::Visible),
            vertices,
            indices,
            entry(6, Table::Props),
            entry(7, Table::NodeCull),
        ]
    }

    /// `table`'s entry at `binding`: its buffer, or the texture it's in.
    pub(crate) fn entry(&self, table: Table, binding: u32) -> BindGroupEntry {
        self.places[table.index()].entry(binding)
    }

    pub(super) fn destroy(self, gpu: &mut dyn Gpu) {
        for place in self.blocks.into_iter().flatten().flatten() {
            place.destroy(gpu);
        }
        for place in self.places {
            place.destroy(gpu);
        }
    }
}
