//! `TileFrame`: the single in-memory representation of a vector tile, and
//! the decoders that produce it.
//!
//! A [`TileFrame`] is a list of [`Layer`]s. Each layer stores its features as
//! columns: ids, geometry types, geometry as nested offsets into one vertex
//! array, and one typed [`Column`] per property key. Layers are built with a
//! [`LayerBuilder`], which keeps the offsets and columns consistent.

#![forbid(unsafe_code)]

mod builder;
mod clip;
mod column;
mod decode;
mod dem;
pub mod geojson;
mod geometry;
pub mod mlt;
pub mod mvt;
mod overzoom;
mod subdivide;

pub use builder::LayerBuilder;
pub use clip::{Coord, clip_ring, clip_square, clip_step};
pub use column::{Bitmap, Column, ColumnData, Value, ValueRef};
pub use decode::DecodeError;
pub use dem::DemGrid;
pub use geometry::{FeatureGeometry, GeometryType, Part};
pub use overzoom::Ancestor;
pub use subdivide::grid_crossings;

/// JavaScript's `Math.round`: the nearest integer, ties toward +∞ (Rust's
/// `round` takes them away from 0), for the ports of GL JS that compute
/// what it computes.
pub fn js_round(x: f64) -> f64 {
    let floor = x.floor();
    if x - floor >= 0.5 { floor + 1.0 } else { floor }
}

#[cfg(test)]
mod tests;

/// Whether an edge of polygon `part` lies on the line its source clipped
/// along, `buffer` past the square of `extent` (none when `buffer` is 0).
pub(crate) fn part_cut(part: &Part<'_>, buffer: i32, extent: i32) -> bool {
    let (lo, hi) = (-buffer, extent + buffer);
    let cut = |a: [i32; 2], b: [i32; 2]| {
        buffer > 0 && (0..2).any(|axis| a[axis] == b[axis] && (a[axis] == lo || a[axis] == hi))
    };
    part.rings().any(|ring| {
        let n = ring.len();
        (0..n).any(|i| cut(ring[i], ring[(i + 1) % n]))
    })
}

/// The tile, relative to this one, holding a polygon's centroid as layout
/// takes it (`orb_layout::extrusion::centroid`: the mean of its rings'
/// vertices, a closed ring's repeated last one left out), when its rings
/// lie within `reach` of that tile's square and of this one's, and so are
/// whole in both. Tiles of `extent` units.
pub(crate) fn owner_keeps<'r>(
    rings: impl Iterator<Item = &'r [[i32; 2]]>,
    extent: i32,
    reach: i32,
) -> Option<[i64; 2]> {
    let (mut sum, mut n) = ([0i64; 2], 0i64);
    let (mut lo, mut hi) = ([i32::MAX; 2], [i32::MIN; 2]);
    for ring in rings {
        let closed = ring.len() > 1 && ring[0] == ring[ring.len() - 1];
        for p in &ring[..ring.len() - usize::from(closed)] {
            sum = [sum[0] + i64::from(p[0]), sum[1] + i64::from(p[1])];
            lo = [lo[0].min(p[0]), lo[1].min(p[1])];
            hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    let (extent, reach) = (i64::from(extent), i64::from(reach));
    let owner = sum.map(|s| s.div_euclid(n * extent));
    let within = |tile: [i64; 2]| {
        (0..2).all(|a| {
            i64::from(lo[a]) >= tile[a] * extent - reach
                && i64::from(hi[a]) <= (tile[a] + 1) * extent + reach
        })
    };
    (within([0, 0]) && within(owner)).then_some(owner)
}

/// The farthest any of `vertices` lies past the square of `extent`, 0 when
/// none does: how far past it their source clipped, when it doesn't say,
/// as a source clips the same distance on every side and clipped vertices
/// land on that line.
fn reach(vertices: &[[i32; 2]], extent: u32) -> i32 {
    let extent = extent as i32;
    let (lo, hi) =
        (vertices.iter().flatten()).fold((0, extent), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    lo.saturating_neg().max(hi - extent)
}

/// A decoded vector tile.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileFrame {
    pub layers: Vec<Layer>,
}

impl TileFrame {
    pub fn layer(&self, name: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Heap bytes the frame holds, for caches' budgets.
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(Layer::bytes).sum()
    }
}

/// One source layer of a tile, stored by column.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    name: String,
    /// Data units per tile side; vertices may lie outside `0..extent`.
    extent: u32,
    ids: Column,
    types: Vec<GeometryType>,
    /// Feature `f` owns parts `feature_parts[f]..feature_parts[f + 1]`.
    feature_parts: Vec<u32>,
    /// Part `p` owns rings `part_rings[p]..part_rings[p + 1]`.
    part_rings: Vec<u32>,
    /// Ring `r` owns vertices `ring_vertices[r]..ring_vertices[r + 1]`.
    ring_vertices: Vec<u32>,
    vertices: Vec<[i32; 2]>,
    properties: Vec<(String, Column)>,
    /// How far past the tile square its producer clipped geometry, in data
    /// units, when known (GeoJSON tiling, overzoom slices; an overzoom
    /// slice keeps its lines further).
    clip: Option<i32>,
    /// [`Layer::clip_buffer`], taken when the layer is built.
    buffer: i32,
    /// How far past the square a polygon may reach and still be drawn
    /// whole ([`Layer::is_whole`]), when it isn't the clip buffer: an
    /// overzoom slice keeps polygons unclipped to one slice past it.
    whole_reach: Option<i32>,
    /// What an overzoom slice lays its labels out from, when that is an
    /// ancestor's slice ([`Layer::labelled`]).
    labels: Option<Box<overzoom::Labels>>,
    /// An overzoom slice's features' indices in the data tile it slices
    /// ([`Layer::data_feature`]).
    data_features: Option<Vec<u32>>,
}

impl Layer {
    /// How far past the tile square geometry was clipped: as its producer
    /// says, else the farthest any vertex lies past the square (a source
    /// clips the same distance on every side, and clipped vertices land on
    /// that line). 0 when nothing reaches past it. Taken once, when the
    /// layer is built.
    pub fn clip_buffer(&self) -> i32 {
        self.buffer
    }

    /// How far past the square a polygon may reach and still be drawn
    /// whole: one slice for an overzoom slice, else the clip buffer
    /// `buffer` ([`clip_buffer`](Self::clip_buffer)).
    pub fn whole_reach(&self, buffer: i32) -> i32 {
        self.whole_reach.unwrap_or(buffer)
    }

    /// Whether `feature` is one polygon drawn whole, entire past the
    /// square, by the one tile at this zoom that holds its centroid, with
    /// `buffer` the layer's [`clip_buffer`](Self::clip_buffer), which
    /// callers take once. It is whole when no edge of it lies on the line
    /// the source clipped along and the tile owning its centroid keeps it
    /// whole too: it reaches no further past that tile than
    /// [`whole_reach`](Self::whole_reach). Every tile of the zoom holding
    /// the polygon decides alike, so a tile that leaves it to its owner
    /// knows the owner draws it. A feature of several polygons never is:
    /// a tile holds only the parts its buffer meets, so tiles would place
    /// its centroid apart; layout splits it into its polygons first.
    pub fn is_whole(&self, feature: usize, buffer: i32) -> bool {
        let geometry = self.geometry(feature);
        if self.types[feature] != GeometryType::Polygon || geometry.parts().len() != 1 {
            return false;
        }
        let part = geometry.parts().next().expect("one part");
        let extent = self.extent as i32;
        !part_cut(&part, buffer, extent)
            && owner_keeps(part.rings(), extent, self.whole_reach(buffer)).is_some()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn extent(&self) -> u32 {
        self.extent
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Feature ids; absent where a feature has none.
    pub fn ids(&self) -> &Column {
        &self.ids
    }

    pub fn geometry_type(&self, feature: usize) -> GeometryType {
        self.types[feature]
    }

    pub fn geometry_types(&self) -> &[GeometryType] {
        &self.types
    }

    pub fn geometry(&self, feature: usize) -> FeatureGeometry<'_> {
        let parts = self.feature_parts[feature] as usize..self.feature_parts[feature + 1] as usize;
        FeatureGeometry::new(self, parts)
    }

    /// The column for a property key.
    pub fn property(&self, key: &str) -> Option<&Column> {
        self.properties
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, c)| c)
    }

    fn bytes(&self) -> usize {
        let offsets = self.feature_parts.len() + self.part_rings.len() + self.ring_vertices.len();
        self.name.len()
            + self.ids.bytes()
            + self.types.len()
            + offsets * 4
            + self.vertices.len() * 8
            + self
                .properties
                .iter()
                .map(|(k, c)| size_of::<(String, Column)>() + k.len() + c.bytes())
                .sum::<usize>()
            + self.labels.as_ref().map_or(0, |l| l.bytes())
            + self.data_features.as_ref().map_or(0, |d| d.len() * 4)
    }

    /// Feature `feature`'s index in the data tile this layer came from: an
    /// overzoom slice keeps only the features it meets, so its indices
    /// differ from its data tile's.
    pub fn data_feature(&self, feature: usize) -> u32 {
        self.data_features
            .as_ref()
            .map_or(feature as u32, |d| d[feature])
    }

    /// Every property column with its key.
    pub fn properties(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.properties.iter().map(|(k, c)| (k.as_str(), c))
    }
}
