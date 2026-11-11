//! The scene's tables as shaders read them: placements, chunks,
//! candidates, node rows, visible lists and the vertex, index and property
//! arenas.
//!
//! Every write to them, and every move into a larger one, goes through
//! [`Tables`], so where they live is one decision, the device's: storage
//! buffers where the vertex stage reads them, which the GPU driver's cull
//! and the draw stages bind (`fetch.wgsl`), or data textures where it
//! can't (WebGL 2, #140; `fetch_texture.wgsl`): the row tables packed into
//! one, at bases a small uniform gives, and each arena in its own. The
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
    Vertices,
    Indices,
    Props,
}

const TABLES: usize = 8;

/// The row tables, which the texture backing packs into one texture, in
/// its order.
#[cfg(cpu_driven)]
const ROWS: [Table; 4] = [Table::Tiles, Table::Chunks, Table::Visible, Table::NodeCull];

impl Table {
    const ALL: [Self; TABLES] = [
        Self::Tiles,
        Self::Chunks,
        Self::Candidates,
        Self::NodeCull,
        Self::Visible,
        Self::Vertices,
        Self::Indices,
        Self::Props,
    ];

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
    /// Words of a data texture from `base`.
    #[cfg(cpu_driven)]
    Texture {
        texture: TextureId,
        base: u32,
    },
    /// Only the CPU reads it: the texture backing's candidates.
    #[cfg(cpu_driven)]
    Cpu,
}

impl Place {
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
            Self::Texture { texture, base } => {
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
                    Some(staged) => staged.push(texture, base + at, words),
                    None => write_texels(gpu, texture, base + at, words),
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
        view_formats: &[],
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

#[cfg(cpu_driven)]
/// The texture backing's row tables: one texture, each table's rows from
/// its base, and a uniform of the bases in words (`fetch_texture.wgsl`'s
/// `RowBases`).
#[derive(Debug)]
#[cfg(cpu_driven)]
struct RowTexture {
    texture: TextureId,
    /// Its rows' words ([`data_width`]).
    width: u32,
    /// Per table of [`ROWS`], its first row and its rows.
    regions: [(u32, u32); 4],
    bases: BufferId,
}

#[cfg(cpu_driven)]
impl RowTexture {
    fn new(gpu: &mut dyn Gpu, words: [u64; 4]) -> Self {
        let bases = gpu.create_buffer(&BufferDesc {
            label: "row bases",
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let width = data_width(&gpu.caps().limits);
        let regions = layout(words, width);
        let texture = data_texture(gpu, "rows", total(&regions));
        let rows = Self {
            texture,
            width,
            regions,
            bases,
        };
        rows.write_bases(gpu);
        rows
    }

    fn write_bases(&self, gpu: &mut dyn Gpu) {
        let bases = self.regions.map(|(first, _)| first * self.width);
        gpu.write_buffer(self.bases, 0, bytemuck::cast_slice(&bases));
    }

    fn place(&self, region: usize) -> Place {
        Place::Texture {
            texture: self.texture,
            base: self.regions[region].0 * self.width,
        }
    }

    /// Moves table `region` into `words` words, its first `keep` copied
    /// over with every other table's rows.
    fn resize(&mut self, gpu: &mut dyn Gpu, region: usize, words: u64, keep: u64) {
        let mut sizes = self
            .regions
            .map(|(_, rows)| u64::from(rows) * u64::from(self.width));
        sizes[region] = words;
        let regions = layout(sizes, self.width);
        let texture = data_texture(gpu, "rows", total(&regions));
        let copies: Vec<TextureCopy> = (0..ROWS.len())
            .filter_map(|r| {
                let kept = if r == region {
                    if keep == 0 { 0 } else { rows(keep, self.width) }
                } else {
                    self.regions[r].1
                }
                .min(regions[r].1);
                (kept > 0).then(|| TextureCopy {
                    from: [0, self.regions[r].0, 0],
                    to: [0, regions[r].0, 0],
                    size: [self.width, kept, 1],
                })
            })
            .collect();
        gpu.submit(&[Pass::CopyTexture {
            source: self.texture,
            destination: texture,
            copies,
        }]);
        gpu.destroy_texture(std::mem::replace(&mut self.texture, texture));
        self.regions = regions;
        self.write_bases(gpu);
    }

    fn destroy(self, gpu: &mut dyn Gpu) {
        gpu.destroy_texture(self.texture);
        gpu.destroy_buffer(self.bases);
    }
}

#[cfg(cpu_driven)]
/// The row tables' regions, one after another, for tables of `words`
/// in rows `width` words wide.
fn layout(words: [u64; 4], width: u32) -> [(u32, u32); 4] {
    let mut first = 0;
    words.map(|w| {
        let region = (first, rows(w, width));
        first += region.1;
        region
    })
}

#[cfg(cpu_driven)]
fn total(regions: &[(u32, u32); 4]) -> u32 {
    regions.iter().map(|r| r.1).sum()
}

/// The scene's tables.
#[derive(Debug)]
pub(crate) struct Tables {
    backing: Backing,
    places: [Place; TABLES],
    #[cfg(cpu_driven)]
    rows: Option<RowTexture>,
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
        mirrored: &[Table],
    ) -> Self {
        #[cfg(cpu_driven)]
        let mut rows = None;
        let places = Table::ALL.map(|table| match backing {
            Backing::Storage => Place::Buffer(gpu.create_buffer(&BufferDesc {
                label: table.label(),
                size: (words[table.index()] * 4).max(16),
                usage: table.usage(),
            })),
            #[cfg(cpu_driven)]
            Backing::Texture => match ROWS.iter().position(|&t| t == table) {
                Some(region) => rows
                    .get_or_insert_with(|| RowTexture::new(gpu, ROWS.map(|t| words[t.index()])))
                    .place(region),
                None if table == Table::Candidates => Place::Cpu,
                None => {
                    let rows = self::rows(words[table.index()], data_width(&gpu.caps().limits));
                    Place::Texture {
                        texture: data_texture(gpu, table.label(), rows),
                        base: 0,
                    }
                }
            },
        });
        #[cfg(cpu_driven)]
        let mut mirrors = Mirrors::default();
        #[cfg(cpu_driven)]
        for table in mirrored {
            mirrors.tables[table.index()] = Some(Vec::new());
        }
        Self {
            backing,
            places,
            #[cfg(cpu_driven)]
            rows,
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

    /// Changes with every write to a table the CPU keeps a copy of.
    #[cfg(cpu_driven)]
    pub(crate) fn revision(&self) -> u64 {
        self.mirrors.revision
    }

    /// Bytes of the heap the copies of tables take.
    pub(crate) fn mirror_bytes(&self) -> u64 {
        #[cfg(cpu_driven)]
        let bytes = (self.mirrors.tables.iter().flatten())
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
            Place::Texture { .. } if let Some(region) = ROWS.iter().position(|&t| t == table) => {
                let rows = self.rows.as_mut().expect("the row texture");
                rows.resize(gpu, region, words, keep);
                for (r, &t) in ROWS.iter().enumerate() {
                    self.places[t.index()] = rows.place(r);
                }
            }
            #[cfg(cpu_driven)]
            Place::Texture { texture, .. } => {
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
                *place = Place::Texture {
                    texture: moved,
                    base: 0,
                };
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
        match backing {
            Backing::Storage => vec![
                table(1, vf),
                table(2, vs),
                table(3, vs),
                table(4, vs),
                table(5, vs),
                // Extrusions' fragments read colours over the elevation.
                table(6, vf),
                table(7, vs),
            ],
            // The row tables in one texture, at the bases of a uniform.
            #[cfg(cpu_driven)]
            Backing::Texture => vec![
                table(1, vf),
                BindGroupLayoutEntry::uniform(2, vf),
                table(4, vs),
                table(5, vs),
                table(6, vf),
            ],
        }
    }

    /// The tables' entries in the scene group (bindings 1 to 7).
    pub(crate) fn entries(&self) -> Vec<BindGroupEntry> {
        let entry = |binding, table: Table| self.entry(table, binding);
        match self.backing {
            Backing::Storage => vec![
                entry(1, Table::Tiles),
                entry(2, Table::Chunks),
                entry(3, Table::Visible),
                entry(4, Table::Vertices),
                entry(5, Table::Indices),
                entry(6, Table::Props),
                entry(7, Table::NodeCull),
            ],
            #[cfg(cpu_driven)]
            Backing::Texture => vec![
                entry(1, Table::Tiles),
                BindGroupEntry::buffer(2, self.rows.as_ref().expect("the row texture").bases),
                entry(4, Table::Vertices),
                entry(5, Table::Indices),
                entry(6, Table::Props),
            ],
        }
    }

    /// `table`'s entry at `binding`: its buffer, or the texture it's in.
    pub(crate) fn entry(&self, table: Table, binding: u32) -> BindGroupEntry {
        match self.places[table.index()] {
            Place::Buffer(buffer) => BindGroupEntry::buffer(binding, buffer),
            #[cfg(cpu_driven)]
            Place::Texture { texture, .. } => {
                BindGroupEntry::texture(binding, texture, TextureViewDimension::D2)
            }
            #[cfg(cpu_driven)]
            Place::Cpu => panic!("{} isn't on the GPU", table.label()),
        }
    }

    #[cfg_attr(not(cpu_driven), allow(unused_variables))]
    pub(super) fn destroy(self, gpu: &mut dyn Gpu) {
        for (i, place) in self.places.into_iter().enumerate() {
            match place {
                Place::Buffer(buffer) => gpu.destroy_buffer(buffer),
                #[cfg(cpu_driven)]
                Place::Texture { texture, .. } if !ROWS.iter().any(|t| t.index() == i) => {
                    gpu.destroy_texture(texture)
                }
                #[cfg(cpu_driven)]
                Place::Texture { .. } | Place::Cpu => {}
            }
        }
        #[cfg(cpu_driven)]
        if let Some(rows) = self.rows {
            rows.destroy(gpu);
        }
    }
}
