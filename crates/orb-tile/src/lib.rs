//! `TileFrame`: the single in-memory representation of a vector tile (plan
//! §10.1), and the decoders that produce it.
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
pub use clip::clip_ring;
pub use column::{Bitmap, Column, ColumnData, Value, ValueRef};
pub use decode::DecodeError;
pub use dem::DemGrid;
pub use geometry::{FeatureGeometry, GeometryType, Part};

#[cfg(test)]
mod tests;

/// A decoded vector tile.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileFrame {
    pub layers: Vec<Layer>,
}

impl TileFrame {
    pub fn layer(&self, name: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.name == name)
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
    /// units, when known (GeoJSON tiling, overzoom slices).
    clip: Option<i32>,
}

impl Layer {
    /// How far past the tile square geometry was clipped: as its producer
    /// says, else the farthest any vertex lies past the square (a source
    /// clips the same distance on every side, and clipped vertices land on
    /// that line). 0 when nothing reaches past it.
    pub fn clip_buffer(&self) -> i32 {
        self.clip.unwrap_or_else(|| {
            let extent = self.extent as i32;
            self.vertices
                .iter()
                .flatten()
                .map(|&v| (-v).max(v - extent))
                .max()
                .unwrap_or(0)
                .max(0)
        })
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

    /// Every property column with its key.
    pub fn properties(&self) -> impl Iterator<Item = (&str, &Column)> {
        self.properties.iter().map(|(k, c)| (k.as_str(), c))
    }
}
