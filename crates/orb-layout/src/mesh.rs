/// Indices in a full chunk: the GPU draws every chunk as this many vertices
/// and discards the padding.
pub const CHUNK_INDICES: u32 = 768;

/// The most triangles in one chunk.
pub const MAX_CHUNK_TRIANGLES: u32 = CHUNK_INDICES / 3;

/// Maximum point instances per chunk.
pub const MAX_CHUNK_INSTANCES: u32 = 64;

/// A rectangle in tile data units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TileBox {
    pub min: [i16; 2],
    pub max: [i16; 2],
}

impl TileBox {
    const EMPTY: Self = Self {
        min: [i16::MAX; 2],
        max: [i16::MIN; 2],
    };

    fn include(&mut self, p: [i16; 2]) {
        self.min = [self.min[0].min(p[0]), self.min[1].min(p[1])];
        self.max = [self.max[0].max(p[0]), self.max[1].max(p[1])];
    }

    fn union(mut self, other: Self) -> Self {
        self.include(other.min);
        self.include(other.max);
        self
    }

    fn meets(self, other: Self) -> bool {
        (0..2).all(|i| self.min[i] <= other.max[i] && other.min[i] <= self.max[i])
    }
}

/// A run of triangles, segments or point instances culled as one unit.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    /// First index of the chunk in [`Mesh::indices`], counted in indices.
    pub first_index: u32,
    /// Indices in the chunk: three per triangle, two per segment, or one
    /// per point instance.
    pub index_count: u32,
    /// The vertex its indices count from: index value `i` is vertex
    /// `first_vertex + i`.
    pub first_vertex: u32,
    /// Bounds of the chunk's vertex positions. Families that extrude on the
    /// GPU (lines) widen it when culling.
    pub bounds: TileBox,
    /// Optional layer sort key; candidates merge these runs across tiles.
    pub sort_key: Option<f32>,
    /// Which of its tile's placements draw it.
    pub owner: Owner,
}

/// Which of its tile's placements draw a chunk: an extrusion's whole
/// buildings draw entire from the placement holding their centroid
/// (`orb_tile::Layer::is_whole`), so a chunk holds buildings of one owner
/// alone, and the cull's candidates keep a chunk only where its owner
/// may draw it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Owner {
    /// Every placement, each its own part: what isn't a whole building.
    #[default]
    Every,
    /// Whole buildings whose centroid the tile's square holds.
    Square,
    /// Whole buildings whose centroid lies in the tile beside it at bit
    /// `3 (dy + 1) + dx + 1` (rows down), never 4, its own: drawn there
    /// when that tile is placed alone over its square, else clipped here.
    Neighbour(u8),
}

impl Owner {
    /// The owner of a whole building with `centroid` in a tile of
    /// `extent` units: its square, or one of the eight around it. Beyond
    /// those, every placement draws its own part.
    pub fn of_whole(centroid: [i16; 2], extent: u32) -> Self {
        let cell = centroid.map(|c| i32::from(c).div_euclid(extent as i32) + 1);
        match cell {
            [1, 1] => Self::Square,
            [x @ 0..=2, y @ 0..=2] => Self::Neighbour((3 * y + x) as u8),
            _ => Self::Every,
        }
    }
}

/// GPU streams for one style layer of one tile.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Vertices as u32 words, a family-specific number per vertex.
    pub vertices: Vec<u32>,
    /// u16 indices two to a word ([`pack_indices`]), each relative to its
    /// chunk's first vertex.
    pub indices: Vec<u32>,
    pub chunks: Vec<Chunk>,
}

impl Mesh {
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Every chunk's indices in order, as vertex numbers.
    pub fn vertex_indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.chunks.iter().flat_map(|c| {
            (c.first_index..c.first_index + c.index_count)
                .map(|i| c.first_vertex + u32::from(index_at(&self.indices, i)))
        })
    }
}

/// Packs u16 indices two to a word, the first in the low half: a
/// little-endian `[u16]`, padded to whole words.
pub fn pack_indices(indices: impl IntoIterator<Item = u16>) -> Vec<u32> {
    let mut halves: Vec<u16> = indices.into_iter().collect();
    if halves.len() % 2 == 1 {
        halves.push(0);
    }
    let pairs = halves.as_chunks::<2>().0;
    pairs
        .iter()
        .map(|&[lo, hi]| u32::from(lo) | u32::from(hi) << 16)
        .collect()
}

/// Index `i` of indices packed by [`pack_indices`].
pub(crate) fn index_at(words: &[u32], i: u32) -> u16 {
    (words[i as usize / 2] >> (16 * (i % 2))) as u16
}

/// Accumulates vertices and triangles and cuts them into chunks.
#[derive(Debug)]
pub struct MeshBuilder {
    stride: usize,
    mesh: Mesh,
    /// Indices as vertex numbers, until [`finish`](Self::finish) packs
    /// them.
    indices: Vec<u32>,
    /// Position of every vertex, for chunk bounds.
    positions: Vec<[i16; 2]>,
    open: TileBox,
    sort_key: Option<f32>,
    owner: Owner,
    /// Where the open chunk's primitives lie against `focus`: meeting it,
    /// or all away from it.
    focus: Option<TileBox>,
    open_meets: bool,
}

impl MeshBuilder {
    /// A builder for vertices of `stride` u32 words each.
    pub fn new(stride: usize) -> Self {
        Self {
            stride,
            mesh: Mesh::default(),
            indices: Vec::new(),
            positions: Vec::new(),
            open: TileBox::EMPTY,
            sort_key: None,
            owner: Owner::Every,
            focus: None,
            open_meets: false,
        }
    }

    /// Keeps primitives meeting `focus` and those away from it in separate
    /// chunks, and chunks away from it from reaching it, so the cull can
    /// drop what lies away from the tile while keeping draw order. Lines
    /// lay out a tile past the square for wide strokes; drawing it all and
    /// discarding it per fragment cost a third of a pitched overzoomed
    /// view's main pass on the moto g (#180).
    pub(crate) fn set_focus(&mut self, focus: TileBox) {
        self.focus = Some(focus);
    }

    /// Starts a sorted run, closing the previous chunk when the key changes.
    pub(crate) fn set_sort_key(&mut self, key: Option<f32>) {
        if self.sort_key != key {
            self.close_chunk();
            self.sort_key = key;
        }
    }

    /// Gives what follows `owner`, closing the previous chunk when it
    /// changes.
    pub(crate) fn set_owner(&mut self, owner: Owner) {
        if self.owner != owner {
            self.close_chunk();
            self.owner = owner;
        }
    }

    pub fn vertex_count(&self) -> u32 {
        self.positions.len() as u32
    }

    /// Adds a vertex at `position` with its words, returning its index.
    pub fn push_vertex(&mut self, position: [i16; 2], words: &[u32]) -> u32 {
        debug_assert_eq!(words.len(), self.stride);
        self.mesh.vertices.extend_from_slice(words);
        self.positions.push(position);
        self.positions.len() as u32 - 1
    }

    pub fn push_triangle(&mut self, a: u32, b: u32, c: u32) {
        self.push_primitive(&[a, b, c], CHUNK_INDICES);
    }

    /// Adds a line segment, for meshes drawn as line lists. A mesh holds
    /// triangles or segments, not both.
    pub(crate) fn push_segment(&mut self, a: u32, b: u32) {
        self.push_primitive(&[a, b], CHUNK_INDICES);
    }

    /// Adds one point instance. A mesh holds only one primitive kind.
    pub(crate) fn push_point(&mut self, vertex: u32) {
        self.push_primitive(&[vertex], MAX_CHUNK_INSTANCES);
    }

    fn push_primitive(&mut self, vertices: &[u32], limit: u32) {
        let chunk_start = self
            .mesh
            .chunks
            .last()
            .map_or(0, |c| c.first_index + c.index_count);
        // Each primitive kind divides its limit, so none straddle chunks.
        if self.indices.len() as u32 - chunk_start == limit {
            self.close_chunk();
        }
        if let Some(focus) = self.focus {
            let mut bounds = TileBox::EMPTY;
            vertices
                .iter()
                .for_each(|&i| bounds.include(self.positions[i as usize]));
            let meets = bounds.meets(focus);
            let open = self.open != TileBox::EMPTY;
            if open
                && (meets != self.open_meets || (!meets && self.open.union(bounds).meets(focus)))
            {
                self.close_chunk();
            }
            self.open_meets = meets;
        }
        for &i in vertices {
            self.open.include(self.positions[i as usize]);
            self.indices.push(i);
        }
    }

    /// The mesh, with each chunk's indices relative to its lowest vertex.
    /// A chunk whose vertices lie more than `u16::MAX` apart (earcut can
    /// bridge across a huge ring) draws copies of them instead, appended
    /// to the vertices.
    pub fn finish(mut self) -> Mesh {
        self.close_chunk();
        let stride = self.stride;
        let mesh = &mut self.mesh;
        let mut relative = Vec::with_capacity(self.indices.len());
        let mut copied = Vec::new();
        for chunk in &mut mesh.chunks {
            let start = chunk.first_index as usize;
            let run = &self.indices[start..start + chunk.index_count as usize];
            let (lo, hi) = run
                .iter()
                .fold((u32::MAX, 0), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            if u16::try_from(hi - lo).is_ok() {
                chunk.first_vertex = lo;
                relative.extend(run.iter().map(|&v| (v - lo) as u16));
                continue;
            }
            copied.clear();
            copied.extend_from_slice(run);
            copied.sort_unstable();
            copied.dedup();
            chunk.first_vertex = (mesh.vertices.len() / stride) as u32;
            for &v in &copied {
                let at = v as usize * stride;
                mesh.vertices.extend_from_within(at..at + stride);
            }
            relative.extend(
                run.iter()
                    .map(|v| copied.binary_search(v).expect("copied above") as u16),
            );
        }
        mesh.indices = pack_indices(relative);
        self.mesh
    }

    fn close_chunk(&mut self) {
        let first_index = self
            .mesh
            .chunks
            .last()
            .map_or(0, |c| c.first_index + c.index_count);
        let index_count = self.indices.len() as u32 - first_index;
        if index_count > 0 {
            self.mesh.chunks.push(Chunk {
                first_index,
                index_count,
                first_vertex: 0,
                bounds: self.open,
                sort_key: self.sort_key,
                owner: self.owner,
            });
        }
        self.open = TileBox::EMPTY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_chunks_at_the_triangle_limit() {
        let mut b = MeshBuilder::new(1);
        let n = MAX_CHUNK_TRIANGLES * 2 + 10;
        for i in 0..n {
            let x = i as i16;
            let v = [[x, 0], [x, 1], [x + 1, 0]].map(|p| b.push_vertex(p, &[0]));
            b.push_triangle(v[0], v[1], v[2]);
        }
        let mesh = b.finish();
        let counts: Vec<_> = mesh.chunks.iter().map(|c| c.index_count / 3).collect();
        assert_eq!(counts, [MAX_CHUNK_TRIANGLES, MAX_CHUNK_TRIANGLES, 10]);
        assert_eq!(mesh.chunks[1].first_index, MAX_CHUNK_TRIANGLES * 3);
        assert_eq!(mesh.chunks[2].first_vertex, MAX_CHUNK_TRIANGLES * 2 * 3);
        assert!(mesh.vertex_indices().eq(0..n * 3));
        let last = mesh.chunks[2].bounds;
        let first_x = (MAX_CHUNK_TRIANGLES * 2) as i16;
        assert_eq!(last.min, [first_x, 0]);
        assert_eq!(last.max, [n as i16, 1]);
    }

    #[test]
    fn keeps_what_lies_away_from_the_focus_in_chunks_of_its_own() {
        let mut b = MeshBuilder::new(1);
        b.set_focus(TileBox {
            min: [0, 0],
            max: [10, 10],
        });
        let mut triangle = |x: i16, y: i16| {
            let v = [[x, y], [x + 1, y], [x, y + 1]].map(|p| b.push_vertex(p, &[0]));
            b.push_triangle(v[0], v[1], v[2]);
        };
        // In, in, west, west, east (whose chunk with the west would reach
        // over the focus), in.
        for (x, y) in [(1, 1), (5, 5), (-20, 5), (-30, 2), (40, 5), (2, 2)] {
            triangle(x, y);
        }
        let mesh = b.finish();
        let counts: Vec<_> = mesh.chunks.iter().map(|c| c.index_count / 3).collect();
        assert_eq!(counts, [2, 2, 1, 1]);
        assert_eq!(mesh.chunks[1].bounds.min, [-30, 2]);
        assert_eq!(mesh.chunks[1].bounds.max, [-19, 6]);
        // Draw order holds.
        assert!(mesh.vertex_indices().eq(0..18));
    }

    #[test]
    fn copies_the_vertices_of_a_chunk_spanning_more_than_u16() {
        let mut b = MeshBuilder::new(2);
        let far = 70_000;
        for i in 0..far {
            b.push_vertex([0, 0], &[i, 7]);
        }
        b.push_triangle(0, 1, far - 1);
        b.push_triangle(1, far - 1, 2);
        let mesh = b.finish();
        assert_eq!(mesh.chunks[0].first_vertex, far);
        assert_eq!(mesh.vertices.len(), (far as usize + 4) * 2);
        let word = |v: u32| mesh.vertices[v as usize * 2];
        let drawn: Vec<_> = mesh.vertex_indices().map(word).collect();
        assert_eq!(drawn, [0, 1, far - 1, 1, far - 1, 2]);
        assert!(mesh.vertex_indices().all(|v| v >= far));
    }
}
