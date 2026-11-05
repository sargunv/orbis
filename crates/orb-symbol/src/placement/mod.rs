//! The placement lane: GL JS's collision detection,
//! cross-tile symbol index and fades, run against an immutable camera
//! snapshot. It reads instances, placed symbols, lines, glyph offsets,
//! anchor offsets and sort-key ranges, never quads, and writes the state
//! each placed symbol draws with.

mod along_line;
mod collision;
mod cross_tile;
mod grid;
mod place;
mod tile_scope;
mod view;

pub use grid::Overlap;
pub use place::{
    BucketStates, Committed, Frame, GroundRef, LayerFrame, LayerParams, PlacedTile,
    PlacementTables, Placer, Scope, ScopeTile, ZoomSize, label_circles, state,
};
pub use view::{Camera, Ground, Mat4, TileGlobe, TileId, TileView};
