//! `TileFrame`: the single in-memory representation of a vector tile (plan
//! §10.1), and the decoders that produce it.
//!
//! A [`TileFrame`] is a list of [`Layer`]s. Each layer stores its features as
//! columns: ids, geometry types, geometry as nested offsets into one vertex
//! array, and one typed [`Column`] per property key. Layers are built with a
//! [`LayerBuilder`], which keeps the offsets and columns consistent.

#![forbid(unsafe_code)]

mod builder;
mod column;
mod decode;
mod dem;
pub mod geojson;
mod geometry;
pub mod mlt;
pub mod mvt;
mod overzoom;

pub use builder::LayerBuilder;
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
}

impl Layer {
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

    /// Every vertex of the layer, in feature, part and ring order.
    pub fn vertices(&self) -> &[[i32; 2]] {
        &self.vertices
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
