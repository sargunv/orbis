/// Indices in a full chunk: the GPU draws every chunk as this many vertices
/// and discards the padding.
pub const CHUNK_INDICES: u32 = 768;

/// The most triangles in one chunk.
pub const MAX_CHUNK_TRIANGLES: u32 = CHUNK_INDICES / 3;

/// Maximum point instances per chunk.
pub const MAX_CHUNK_INSTANCES: u32 = 256;

/// A rectangle in tile data units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

/// A run of triangles, segments or point instances culled as one unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chunk {
    /// First index of the chunk in [`Mesh::indices`].
    pub first_index: u32,
    /// Indices in the chunk: three per triangle, two per segment, or one
    /// per point instance.
    pub index_count: u32,
    /// Bounds of the chunk's vertex positions. Families that extrude on the
    /// GPU (lines) widen it when culling.
    pub bounds: TileBox,
    /// Optional layer sort key; candidates merge these runs across tiles.
    pub sort_key: Option<f32>,
}

/// GPU streams for one style layer of one tile.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Vertices as u32 words, a family-specific number per vertex.
    pub vertices: Vec<u32>,
    /// Triangle indices into the mesh's vertices, counted in vertices.
    pub indices: Vec<u32>,
    pub chunks: Vec<Chunk>,
}

impl Mesh {
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Accumulates vertices and triangles and cuts them into chunks.
#[derive(Debug)]
pub struct MeshBuilder {
    stride: usize,
    mesh: Mesh,
    /// Position of every vertex, for chunk bounds.
    positions: Vec<[i16; 2]>,
    open: TileBox,
    sort_key: Option<f32>,
}

impl MeshBuilder {
    /// A builder for vertices of `stride` u32 words each.
    pub fn new(stride: usize) -> Self {
        Self {
            stride,
            mesh: Mesh::default(),
            positions: Vec::new(),
            open: TileBox::EMPTY,
            sort_key: None,
        }
    }

    /// Starts a sorted run, closing the previous chunk when the key changes.
    pub fn set_sort_key(&mut self, key: Option<f32>) {
        if self.sort_key != key {
            self.close_chunk();
            self.sort_key = key;
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
    pub fn push_segment(&mut self, a: u32, b: u32) {
        self.push_primitive(&[a, b], CHUNK_INDICES);
    }

    /// Adds one point instance. A mesh holds only one primitive kind.
    pub fn push_point(&mut self, vertex: u32) {
        self.push_primitive(&[vertex], MAX_CHUNK_INSTANCES);
    }

    fn push_primitive(&mut self, vertices: &[u32], limit: u32) {
        let chunk_start = self
            .mesh
            .chunks
            .last()
            .map_or(0, |c| c.first_index + c.index_count);
        // Each primitive kind divides its limit, so none straddle chunks.
        if self.mesh.indices.len() as u32 - chunk_start == limit {
            self.close_chunk();
        }
        for &i in vertices {
            self.open.include(self.positions[i as usize]);
            self.mesh.indices.push(i);
        }
    }

    pub fn finish(mut self) -> Mesh {
        self.close_chunk();
        self.mesh
    }

    fn close_chunk(&mut self) {
        let first_index = self
            .mesh
            .chunks
            .last()
            .map_or(0, |c| c.first_index + c.index_count);
        let index_count = self.mesh.indices.len() as u32 - first_index;
        if index_count > 0 {
            self.mesh.chunks.push(Chunk {
                first_index,
                index_count,
                bounds: self.open,
                sort_key: self.sort_key,
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
        let last = mesh.chunks[2].bounds;
        let first_x = (MAX_CHUNK_TRIANGLES * 2) as i16;
        assert_eq!(last.min, [first_x, 0]);
        assert_eq!(last.max, [n as i16, 1]);
    }
}
