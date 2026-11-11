//! Layout families: turn a tile layer's selected features into GPU streams.
//!
//! Layout is purely geometric. The caller decides which features a style
//! layer draws (its filter) and evaluates their paint values; layout gets the
//! selected feature indices and writes a [`Mesh`]: vertices, triangle indices
//! and chunks of at most [`MAX_CHUNK_TRIANGLES`] triangles or
//! [`MAX_CHUNK_INSTANCES`] point instances with their bounds.
//! Every vertex carries its feature slot, the feature's position in the
//! selection, which indexes the caller's per-feature property rows.
//!
//! Fills are clipped to the tile square. Extruded lines are clipped to a
//! padded square and rely on a fragment test at the tile edge, so no per-tile
//! stencil is needed. Points are owned by the half-open tile square; placement
//! clips test their anchors and keep their entire extruded footprints.

#![forbid(unsafe_code)]

mod clip;
pub mod extrusion;
mod family;
pub mod fill;
pub mod line;
mod mesh;
pub mod point;
mod round_corners;

pub use family::{Family, ImageKey, ImageMetadata, pack_color};
pub use mesh::{
    CHUNK_INDICES, Chunk, MAX_CHUNK_INSTANCES, MAX_CHUNK_TRIANGLES, Mesh, MeshBuilder, Owner,
    TileBox, pack_indices,
};
pub use round_corners::rounded_layer;
